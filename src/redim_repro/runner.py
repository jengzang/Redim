"""Project-owned command and configuration glue around a pinned WeSpeaker.

No model code is vendored here.  These helpers only turn this repository's
experiment choices into commands for WeSpeaker's public scripts, keeping those
commands testable without GPUs or an installed WeSpeaker checkout.
"""

from __future__ import annotations

import shutil
from pathlib import Path
import sys
from typing import Sequence, Union


# The bootstrap script uses this exact public upstream revision; pinning it
# prevents a future WeSpeaker change from silently changing this reproduction.
WESPEAKER_REPO = "https://github.com/wenet-e2e/wespeaker.git"
WESPEAKER_COMMIT = "9fecd6cb4f47475d01761d87c826298dff4ef18c"
GpuValue = Union[str, int, Sequence[Union[str, int]]]
DEFAULT_NUM_AVG = {"ptn": 10, "lm": 1}


def normalize_gpus(value: GpuValue) -> str:
    """Return the list syntax expected by WeSpeaker's ``run.sh``."""

    if isinstance(value, (list, tuple)):
        items = [str(item).strip() for item in value]
    else:
        text = str(value).strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        items = [item.strip() for item in text.split(",") if item.strip()]
    if not items:
        raise ValueError("at least one GPU id is required")
    if any(not item.isdigit() for item in items):
        raise ValueError(f"invalid GPU list: {value!r}")
    return "[{}]".format(",".join(items))


def default_num_avg(phase: str) -> int:
    """Return the checkpoint averaging policy for a training phase."""

    try:
        return DEFAULT_NUM_AVG[phase]
    except KeyError as exc:
        raise ValueError(f"unsupported training phase: {phase!r}") from exc


def stage_lm_checkpoint(source: Path, lm_exp_dir: Path) -> Path:
    """Copy a PTN checkpoint to the filename expected by upstream train.py.

    WeSpeaker derives the resume epoch from ``model_<N>.pt``.  The averaged
    PTN checkpoint is named ``avg_model.pt``, so it must be staged as
    ``model_0.pt`` before launching LM training.
    """

    source = Path(source).expanduser().absolute()
    if not source.is_file():
        raise FileNotFoundError(f"LM source checkpoint not found: {source}")

    destination = (
        Path(lm_exp_dir).expanduser().absolute() / "models" / "model_0.pt"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        try:
            if source.samefile(destination):
                return destination
        except FileNotFoundError:
            pass
    shutil.copy2(source, destination)
    return destination


def materialize_config(
    config_path: Path,
    destination: Path,
    *,
    batch_size: int | None = None,
    num_workers: int | None = None,
    prefetch_factor: int | None = None,
    pin_memory: bool | None = None,
    amp: bool | None = None,
) -> Path:
    """Write a server-specific YAML override without editing a recipe file."""

    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_workers is not None and num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if prefetch_factor is not None and prefetch_factor < 1:
        raise ValueError("prefetch_factor must be positive")

    overrides = (
        batch_size,
        num_workers,
        prefetch_factor,
        pin_memory,
        amp,
    )
    if all(value is None for value in overrides):
        return Path(config_path).expanduser().resolve()

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised in deployment
        raise RuntimeError(
            "PyYAML is required for --batch-size/--num-workers/--prefetch-factor/"
            "--amp overrides; install requirements.txt"
        ) from exc

    config_path = Path(config_path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or not isinstance(
        config.get("dataloader_args"), dict
    ):
        raise ValueError(f"invalid WeSpeaker config structure: {config_path}")

    dataloader_args = config["dataloader_args"]
    if batch_size is not None:
        dataloader_args["batch_size"] = batch_size
    if num_workers is not None:
        dataloader_args["num_workers"] = num_workers
    if prefetch_factor is not None:
        dataloader_args["prefetch_factor"] = prefetch_factor
    if pin_memory is not None:
        dataloader_args["pin_memory"] = pin_memory
    if amp is not None:
        config["enable_amp"] = amp

    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    return destination


def build_upstream_run_command(
    *,
    upstream_dir: Path,
    data_dir: Path,
    config_path: Path,
    gpus: GpuValue,
    stage: int,
    stop_stage: int,
    exp_dir: Path,
    num_avg: int = 10,
    checkpoint: Path | None = None,
    lm_config_path: Path | None = None,
    data_type: str = "shard",
    num_nodes: int = 1,
    host_node_addr: str = "localhost:29400",
    job_id: str | int = "redimnet2",
) -> list[str]:
    """Build an invocation of the pinned upstream VoxCeleb recipe."""

    if num_nodes < 1:
        raise ValueError("num_nodes must be positive")
    if not host_node_addr:
        raise ValueError("host_node_addr must not be empty")
    if not data_type:
        raise ValueError("data_type must not be empty")
    if num_avg < 1:
        raise ValueError("num_avg must be positive")

    run_script = upstream_dir / "examples" / "voxceleb" / "v2" / "run.sh"
    command = [
        "bash",
        str(run_script),
        "--stage",
        str(stage),
        "--stop_stage",
        str(stop_stage),
        "--data",
        str(data_dir),
        "--config",
        str(config_path),
        "--exp_dir",
        str(exp_dir),
        "--gpus",
        normalize_gpus(gpus),
        "--data_type",
        data_type,
        "--num_avg",
        str(num_avg),
        "--HOST_NODE_ADDR",
        host_node_addr,
        "--num_nodes",
        str(num_nodes),
        "--job_id",
        str(job_id),
    ]
    if lm_config_path is not None:
        command.extend(["--lm_config", str(lm_config_path)])
    if checkpoint is not None:
        command.extend(["--checkpoint", str(checkpoint)])
    return command


def build_epoch_train_command(
    *,
    project_root: Path,
    upstream_dir: Path,
    data_dir: Path,
    config_path: Path,
    exp_dir: Path,
    gpus: GpuValue,
    eval_data: Path,
    eval_trials: Path,
    eval_interval: int,
    eval_num_workers: int = 1,
    checkpoint: Path | None = None,
    data_type: str = "raw",
    reverb_data: Path | None = None,
    noise_data: Path | None = None,
    num_nodes: int = 1,
    host_node_addr: str = "localhost:29400",
    job_id: str | int = "redimnet2",
) -> list[str]:
    """Build this project's ``torchrun`` command with rank-0 epoch validation.

    This differs from the stock upstream ``run.sh`` only where the server
    index layout requires it: it invokes our small entrypoint so validation can
    happen between epochs on the fixed VoxCeleb2-test trials.
    """

    if num_nodes < 1:
        raise ValueError("num_nodes must be positive")
    if eval_interval < 1:
        raise ValueError("eval_interval must be positive")
    if eval_num_workers < 0:
        raise ValueError("eval_num_workers must be non-negative")
    if not host_node_addr:
        raise ValueError("host_node_addr must not be empty")
    if data_type not in {"raw", "shard"}:
        raise ValueError(f"unsupported epoch-training data type: {data_type!r}")

    normalized_gpus = normalize_gpus(gpus)
    gpu_count = len(normalized_gpus.strip("[]").split(","))
    command = [
        "torchrun",
        "--nnodes",
        str(num_nodes),
        "--nproc_per_node",
        str(gpu_count),
        "--rdzv_id",
        str(job_id),
        "--rdzv_backend",
        "c10d",
        "--rdzv_endpoint",
        host_node_addr,
        str(Path(project_root) / "scripts" / "train_with_epoch_eval.py"),
        "--config",
        str(config_path),
        "--exp_dir",
        str(exp_dir),
        "--gpus",
        normalized_gpus,
        "--data_type",
        data_type,
        "--train_data",
        str(Path(data_dir) / "vox2_dev" / f"{data_type}.list"),
        "--train_label",
        str(Path(data_dir) / "vox2_dev" / "utt2spk"),
        "--eval_data",
        str(eval_data),
        "--eval_trials",
        str(eval_trials),
        "--eval_interval",
        str(eval_interval),
        "--eval_num_workers",
        str(eval_num_workers),
    ]
    if reverb_data is not None:
        command.extend(["--reverb_data", str(reverb_data)])
    if noise_data is not None:
        command.extend(["--noise_data", str(noise_data)])
    if checkpoint is not None:
        command.extend(["--checkpoint", str(checkpoint)])
    return command


def build_average_command(
    *, upstream_dir: Path, exp_dir: Path, num_avg: int
) -> list[str]:
    """Build the upstream checkpoint averaging command."""

    if num_avg < 1:
        raise ValueError("num_avg must be positive")

    return [
        sys.executable,
        "wespeaker/bin/average_model.py",
        "--dst_model",
        str(exp_dir / "models" / "avg_model.pt"),
        "--src_path",
        str(exp_dir / "models"),
        "--num",
        str(num_avg),
    ]


def build_raw_score_command(
    *,
    upstream_dir: Path,
    data_dir: Path,
    exp_dir: Path,
    trial_name: str,
) -> list[str]:
    """Build a raw-cosine score command without mean centering."""

    return [
        sys.executable,
        "wespeaker/bin/score.py",
        "--exp_dir",
        str(exp_dir),
        "--eval_scp_path",
        str(exp_dir / "embeddings" / "vox1" / "xvector.scp"),
        "--cal_mean",
        "False",
        "--cal_mean_dir",
        str(exp_dir / "embeddings" / "vox2_dev"),
        str(data_dir / "vox1" / "trials" / trial_name),
    ]
