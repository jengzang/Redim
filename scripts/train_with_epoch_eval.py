#!/usr/bin/env python3
"""Project-owned DDP entrypoint composed from pinned WeSpeaker public APIs.

WeSpeaker supplies ``Dataset``, ReDimNet2, projection/loss construction,
``run_epoch`` and checkpoint utilities.  This entrypoint keeps the same
training contract while adding a rank-0-only, full-coverage fixed-trial
validation step after selected epochs.  It is not a vendored copy of the
WeSpeaker source tree.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import sys
from pprint import pformat


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
upstream_dir = os.environ.get("WESPEAKER_DIR")
if not upstream_dir:
    raise RuntimeError("WESPEAKER_DIR must point to the pinned WeSpeaker checkout")
sys.path.insert(0, str(Path(upstream_dir).expanduser().resolve()))

import fire
import tableprint as tp
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader

import wespeaker.utils.schedulers as schedulers
from wespeaker.dataset.dataset import Dataset
from wespeaker.frontend import frontend_class_dict
from wespeaker.models.projections import get_projection
from wespeaker.models.speaker_model import get_speaker_model
from wespeaker.utils.checkpoint import load_checkpoint, save_checkpoint
from wespeaker.utils.executor import run_epoch
from wespeaker.utils.file_utils import read_table
from wespeaker.utils.utils import get_logger, parse_config_or_kwargs, set_seed, spk2id

from redim_repro.epoch_eval import evaluate_in_memory_model, write_epoch_result


def _as_gpu_ids(value) -> list[int]:
    if isinstance(value, str):
        text = value.strip().strip("[]")
        values = [item.strip() for item in text.split(",") if item.strip()]
    else:
        values = list(value)
    if not values or any(not str(item).isdigit() for item in values):
        raise ValueError(f"invalid GPU configuration: {value!r}")
    return [int(item) for item in values]


def _replace_final_model(model_dir: str, epoch: int) -> None:
    final_path = Path(model_dir) / "final_model.pt"
    if final_path.exists() or final_path.is_symlink():
        final_path.unlink()
    final_path.symlink_to(f"model_{epoch}.pt")


def train(
    config="conf/config.yaml",
    eval_data=None,
    eval_trials=None,
    eval_interval=1,
    eval_num_workers=1,
    **kwargs,
):
    """Run the pinned WeSpeaker loop and validate rank 0 at epoch boundaries."""

    if not eval_data or not eval_trials:
        raise ValueError("eval_data and eval_trials are required for epoch evaluation")
    eval_interval = int(eval_interval)
    eval_num_workers = int(eval_num_workers)
    if eval_interval < 1:
        raise ValueError("eval_interval must be positive")
    if eval_num_workers < 0:
        raise ValueError("eval_num_workers must be non-negative")
    eval_data = Path(eval_data).expanduser().resolve()
    eval_trials = Path(eval_trials).expanduser().resolve()
    if not eval_data.is_file():
        raise FileNotFoundError(f"missing evaluation raw list: {eval_data}")
    if not eval_trials.is_file():
        raise FileNotFoundError(f"missing evaluation trials: {eval_trials}")

    configs = parse_config_or_kwargs(config, **kwargs)
    configs["gpus"] = _as_gpu_ids(configs["gpus"])
    checkpoint = configs.get("checkpoint")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ["WORLD_SIZE"])
    if local_rank >= len(configs["gpus"]):
        raise ValueError(
            f"LOCAL_RANK={local_rank} exceeds configured GPUs {configs['gpus']}"
        )
    gpu = configs["gpus"][local_rank]
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend="nccl")

    model_dir = Path(configs["exp_dir"]) / "models"
    if rank == 0:
        try:
            model_dir.mkdir(parents=True)
        except FileExistsError:
            if checkpoint is None:
                raise RuntimeError(
                    f"experiment exists without a checkpoint: {model_dir.parent}"
                )
    dist.barrier(device_ids=[gpu])

    logger = get_logger(configs["exp_dir"], "train.log")
    if rank == 0:
        logger.info("exp_dir is: %s", configs["exp_dir"])
        logger.info("<== Passed Arguments ==>")
        for line in pformat(configs).split("\n"):
            logger.info(line)
        logger.info(
            "fixed validation: data=%s trials=%s interval=%d",
            eval_data,
            eval_trials,
            eval_interval,
        )
    if world_size > 1:
        logger.info("training on multiple GPUs, this GPU %d", gpu)
    set_seed(configs["seed"] + rank)

    train_utt_spk_list = read_table(configs["train_label"])
    spk2id_dict = spk2id(train_utt_spk_list)
    if rank == 0:
        logger.info(
            "train data num: %d, spk num: %d",
            len(train_utt_spk_list),
            len(spk2id_dict),
        )
    train_dataset = Dataset(
        configs["data_type"],
        configs["train_data"],
        configs["dataset_args"],
        spk2id_dict,
        reverb_lmdb_file=configs.get("reverb_data"),
        noise_lmdb_file=configs.get("noise_data"),
    )
    train_dataloader = DataLoader(train_dataset, **configs["dataloader_args"])
    batch_size = configs["dataloader_args"]["batch_size"]
    sample_num_per_epoch = configs["dataset_args"].get("sample_num_per_epoch", 0)
    if sample_num_per_epoch <= 0:
        sample_num_per_epoch = len(train_utt_spk_list)
        if (
            configs["data_type"] != "feat"
            and configs["dataset_args"].get("speed_perturb", False)
            and configs["dataset_args"].get("speed_perturb_mode") == "expanded"
        ):
            sample_num_per_epoch *= 3
    epoch_iter = sample_num_per_epoch // world_size // batch_size
    if epoch_iter < 1:
        raise RuntimeError(
            "global batch size exceeds available training records; reduce batch size"
        )
    if rank == 0:
        logger.info("epoch iteration number: %d", epoch_iter)

    frontend_type = configs["dataset_args"].get("frontend", "fbank")
    if frontend_type != "fbank":
        frontend_args = frontend_type + "_args"
        frontend = frontend_class_dict[frontend_type](
            **configs["dataset_args"][frontend_args],
            sample_rate=configs["dataset_args"]["resample_rate"],
        )
        configs["model_args"]["feat_dim"] = frontend.output_size()
        model = get_speaker_model(configs["model"])(**configs["model_args"])
        model.add_module("frontend", frontend)
    else:
        model = get_speaker_model(configs["model"])(**configs["model_args"])
    if rank == 0:
        logger.info(
            "speaker_model size: %d", sum(parameter.numel() for parameter in model.parameters())
        )
    if configs["model_init"] is not None:
        load_checkpoint(model, configs["model_init"])
    elif checkpoint is None:
        logger.info("Train model from scratch ...")

    configs["projection_args"]["embed_dim"] = configs["model_args"]["embed_dim"]
    configs["projection_args"]["num_class"] = len(spk2id_dict)
    configs["projection_args"]["do_lm"] = configs.get("do_lm", False)
    if configs["data_type"] != "feat" and configs["dataset_args"]["speed_perturb"]:
        configs["projection_args"]["num_class"] *= 3
        if configs.get("do_lm", False):
            logger.info("No speed perturb while doing large margin fine-tuning")
            configs["dataset_args"]["speed_perturb"] = False
    model.add_module("projection", get_projection(configs["projection_args"]))

    if checkpoint is not None:
        load_checkpoint(model, checkpoint)
        matches = re.findall(r"(?<=model_)\d*(?=.pt)", str(checkpoint))
        if not matches:
            raise ValueError(f"checkpoint must be named model_<epoch>.pt: {checkpoint}")
        start_epoch = int(matches[0]) + 1
        logger.info("Load checkpoint: %s", checkpoint)
    else:
        start_epoch = 1
    logger.info("start_epoch: %d", start_epoch)

    # Training uses DDP, but validation below deliberately receives the bare
    # model so its frontend and embeddings can be evaluated on rank 0.
    model.cuda()
    ddp_model = torch.nn.parallel.DistributedDataParallel(model)
    device = torch.device("cuda")
    criterion = getattr(torch.nn, configs["loss"])(**configs["loss_args"])
    if "initial_lr" in configs["scheduler_args"]:
        configs["optimizer_args"]["lr"] = configs["scheduler_args"]["initial_lr"]
    optimizer = getattr(torch.optim, configs["optimizer"])(
        ddp_model.parameters(), **configs["optimizer_args"]
    )
    configs["scheduler_args"]["num_epochs"] = configs["num_epochs"]
    configs["scheduler_args"]["epoch_iter"] = epoch_iter
    configs["scheduler_args"]["scale_ratio"] = (
        world_size * configs["dataloader_args"]["batch_size"] / 64
    )
    scheduler = getattr(schedulers, configs["scheduler"])(
        optimizer, **configs["scheduler_args"]
    )
    configs["margin_update"]["epoch_iter"] = epoch_iter
    margin_scheduler = getattr(schedulers, configs["margin_scheduler"])(
        model=model, **configs["margin_update"]
    )
    if rank == 0:
        configs["epoch_eval"] = {
            "data": str(eval_data),
            "trials": str(eval_trials),
            "interval": eval_interval,
            "num_workers": eval_num_workers,
        }
        with (Path(configs["exp_dir"]) / "config.yaml").open("w") as stream:
            yaml.safe_dump(configs, stream, sort_keys=False)
        header = ["Epoch", "Batch", "Lr", "Margin", "Loss", "Acc"]
        for line in tp.header(header, width=10, style="grid").split("\n"):
            logger.info(line)
    dist.barrier(device_ids=[gpu])

    scaler = torch.cuda.amp.GradScaler(enabled=configs["enable_amp"])
    for epoch in range(start_epoch, configs["num_epochs"] + 1):
        train_dataset.set_epoch(epoch)
        run_epoch(
            train_dataloader,
            epoch_iter,
            ddp_model,
            criterion,
            optimizer,
            scheduler,
            margin_scheduler,
            epoch,
            logger,
            scaler,
            device=device,
            configs=configs,
        )
        evaluate_this_epoch = (
            epoch % eval_interval == 0 or epoch == configs["num_epochs"]
        )
        save_this_epoch = (
            evaluate_this_epoch
            or epoch % configs["save_epoch_interval"] == 0
            or epoch > configs["num_epochs"] - configs["num_avg"]
        )
        checkpoint_path = model_dir / f"model_{epoch}.pt"
        if rank == 0 and save_this_epoch:
            save_checkpoint(model, str(checkpoint_path))
        # Every rank must observe the completed checkpoint decision before the
        # rank-0-only forward pass; otherwise the next DDP collective can hang.
        dist.barrier(device_ids=[gpu])
        if rank == 0 and evaluate_this_epoch:
            result = evaluate_in_memory_model(
                model,
                configs,
                eval_data,
                eval_trials,
                device,
                num_workers=eval_num_workers,
            )
            best = write_epoch_result(
                Path(configs["exp_dir"]), epoch, result, checkpoint_path
            )
            logger.info(
                "epoch %d fixed-validation EER=%.4f%% minDCF=%.6f "
                "target=%d nontarget=%d elapsed=%.1fs best_epoch=%d",
                epoch,
                result.eer * 100,
                result.min_dcf,
                result.target_trials,
                result.nontarget_trials,
                result.elapsed_seconds,
                best["epoch"],
            )
        # Non-zero ranks wait while rank 0 embeds the complete validation set.
        dist.barrier(device_ids=[gpu])

    if rank == 0:
        _replace_final_model(str(model_dir), configs["num_epochs"])
        logger.info(tp.bottom(6, width=10, style="grid"))
    dist.barrier(device_ids=[gpu])
    dist.destroy_process_group()


if __name__ == "__main__":
    fire.Fire(train)
