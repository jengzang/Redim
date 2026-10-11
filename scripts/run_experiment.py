#!/usr/bin/env python3
"""Project CLI that prepares server indexes and invokes pinned WeSpeaker.

This file owns orchestration and safety checks only.  ReDimNet2 construction,
data loading, optimisation, embedding extraction and standard score tools come
from the external WeSpeaker checkout resolved through ``WESPEAKER_DIR``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from redim_repro.data_validation import (
    ManifestError,
    validate_server_vox2_layout,
    validate_voxceleb_layout,
)
from redim_repro.runner import (
    build_average_command,
    build_epoch_train_command,
    build_raw_score_command,
    build_upstream_run_command,
    default_num_avg,
    materialize_config,
    stage_lm_checkpoint,
)
from redim_repro.vox2_index import prepare_server_vox2


MODEL_NAMES = ("b3", "b6")
PHASES = ("ptn", "lm")
TRIAL_NAMES = (
    "vox1_O_cleaned.kaldi",
    "vox1_E_cleaned.kaldi",
    "vox1_H_cleaned.kaldi",
)
DATA_TYPES = ("shard", "raw")


def _upstream_dir() -> Path:
    return Path(
        os.environ.get("WESPEAKER_DIR", str(ROOT / ".deps" / "wespeaker"))
    ).expanduser().resolve()


def _recipe_dir(upstream_dir: Path) -> Path:
    return upstream_dir / "examples" / "voxceleb" / "v2"


def _config(model: str, phase: str) -> Path:
    return ROOT / "configs" / f"redimnet2_{model}_{phase}.yaml"


def _default_exp_dir(model: str, phase: str) -> Path:
    return ROOT / "exp" / f"redimnet2_{model}" / phase


def _resolve_config(args: argparse.Namespace, model: str, phase: str) -> Path:
    config = args.config or _config(model, phase)
    return config.expanduser().resolve()


def _require_converted_data(
    data: Path, data_type: str, *, require_augmentation: bool = True
) -> None:
    """Fail early when --skip-conversion is used with incomplete artifacts."""

    required = [
        data / "vox2_dev" / f"{data_type}.list",
        data / "vox2_dev" / "utt2spk",
    ]
    if require_augmentation:
        required.extend([data / "musan" / "lmdb", data / "rirs" / "lmdb"])
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "converted training data is incomplete; missing: " + ", ".join(missing)
        )


def _require_evaluation_data(data: Path, data_type: str) -> None:
    required = [
        data / "vox2_dev" / f"{data_type}.list",
        data / "vox1" / f"{data_type}.list",
        data / "vox1" / "wav.scp",
        data / "vox1" / "trials",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "converted evaluation data is incomplete; missing: " + ", ".join(missing)
        )


def _require_server_vox2_training_data(
    data: Path,
    data_type: str,
    *,
    reverb_data: Path | None,
    noise_data: Path | None,
) -> None:
    required = [
        data / "vox2_dev" / f"{data_type}.list",
        data / "vox2_dev" / "utt2spk",
        data / "vox2_test" / "raw.list",
        data / "vox2_test" / "trials" / "vox2_test_fixed.kaldi",
    ]
    if reverb_data is not None:
        required.append(reverb_data)
    if noise_data is not None:
        required.append(noise_data)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "server VoxCeleb2 data is incomplete; missing: " + ", ".join(missing)
        )


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    print("$ " + " ".join(command), flush=True)
    command_env = os.environ.copy()
    if env:
        command_env.update(env)
    subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        env=command_env,
        check=True,
    )


def ensure_upstream() -> Path:
    """Fetch/verify the pinned third-party WeSpeaker checkout when requested."""

    bootstrap = ROOT / "scripts" / "bootstrap_wespeaker.sh"
    _run(["bash", str(bootstrap)])
    upstream = _upstream_dir()
    if not (upstream / "examples" / "voxceleb" / "v2" / "run.sh").is_file():
        raise RuntimeError(f"invalid WeSpeaker checkout: {upstream}")
    return upstream


def resolve_upstream(*, skip_bootstrap: bool) -> Path:
    """Resolve WeSpeaker, optionally without any network or Git operation."""

    if not skip_bootstrap:
        return ensure_upstream()
    upstream = _upstream_dir()
    if not (upstream / "examples" / "voxceleb" / "v2" / "run.sh").is_file():
        raise RuntimeError(
            f"invalid or missing WeSpeaker checkout: {upstream}; "
            "run bootstrap once or omit --skip-bootstrap"
        )
    return upstream


def add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data",
        help="VoxCeleb data root (default: ./data)",
    )


def add_data_type_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-type",
        choices=DATA_TYPES,
        default="shard",
        help="prepared list format consumed by WeSpeaker (default: shard)",
    )


def add_upstream_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--skip-bootstrap",
        action="store_true",
        help="use WESPEAKER_DIR as-is; do not run Git bootstrap/fetch",
    )


def add_distributed_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=1,
        help="torchrun node count (default: 1)",
    )
    parser.add_argument(
        "--host-node-addr",
        default="localhost:29400",
        help="rendezvous host:port for multi-node training",
    )
    parser.add_argument(
        "--job-id",
        default="redimnet2",
        help="unique torchrun rendezvous id for this job",
    )


def add_runtime_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        help="optional YAML recipe override instead of the bundled config",
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--prefetch-factor", type=int)
    pin_memory = parser.add_mutually_exclusive_group()
    pin_memory.add_argument(
        "--pin-memory", dest="pin_memory", action="store_true"
    )
    pin_memory.add_argument(
        "--no-pin-memory", dest="pin_memory", action="store_false"
    )
    amp = parser.add_mutually_exclusive_group()
    amp.add_argument("--amp", dest="amp", action="store_true")
    amp.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(pin_memory=None, amp=None)


def add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        help="optional YAML recipe used for model extraction",
    )


def command_validate(args: argparse.Namespace) -> int:
    try:
        report = validate_voxceleb_layout(args.data, check_files=args.check_files)
    except ManifestError as exc:
        print(f"data validation failed: {exc}", file=sys.stderr)
        return 1
    print(report)
    return 0


def command_validate_server_vox2(args: argparse.Namespace) -> int:
    try:
        report = validate_server_vox2_layout(args.data, check_files=args.check_files)
    except ManifestError as exc:
        print(f"server VoxCeleb2 data validation failed: {exc}", file=sys.stderr)
        return 1
    print(report)
    return 0


def command_prepare(args: argparse.Namespace) -> int:
    upstream = resolve_upstream(skip_bootstrap=args.skip_bootstrap)
    recipe = _recipe_dir(upstream)
    script = recipe / "local" / "prepare_data.sh"
    _run(
        [
            "bash",
            str(script),
            "--stage",
            str(args.stage),
            "--stop_stage",
            str(args.stop_stage),
            "--data",
            str(args.data.expanduser().resolve()),
        ],
        cwd=recipe,
    )
    return 0


def command_prepare_server_vox2(args: argparse.Namespace) -> int:
    """Convert index files first, then delegate optional shard writing upstream."""

    corpus = args.corpus.expanduser().resolve()
    data = args.data.expanduser().resolve()
    report = prepare_server_vox2(
        corpus_root=corpus,
        output_root=data,
        pairs_per_speaker=args.pairs_per_speaker,
        seed=args.seed,
        check_files=args.check_files,
    )
    if args.data_type == "shard":
        if args.shard_workers < 1 or args.utts_per_shard < 1:
            raise ValueError("shard-workers and utts-per-shard must be positive")
        upstream = resolve_upstream(skip_bootstrap=args.skip_bootstrap)
        _run(
            [
                sys.executable,
                str(upstream / "tools" / "make_shard_list.py"),
                "--num_utts_per_shard",
                str(args.utts_per_shard),
                "--num_threads",
                str(args.shard_workers),
                "--prefix",
                "shards",
                "--seed",
                str(args.seed),
                "--shuffle",
                str(data / "vox2_dev" / "wav.scp"),
                str(data / "vox2_dev" / "utt2spk"),
                str(data / "vox2_dev" / "shards"),
                str(data / "vox2_dev" / "shard.list"),
            ],
            cwd=upstream,
        )
    print(report)
    return 0


def _server_augmentation_paths(
    args: argparse.Namespace, data: Path
) -> tuple[Path | None, Path | None]:
    if args.phase == "lm":
        return None, None
    rirs = (args.rirs_lmdb or data / "rirs" / "lmdb").expanduser().resolve()
    musan = (args.musan_lmdb or data / "musan" / "lmdb").expanduser().resolve()
    return rirs, musan


def command_train(args: argparse.Namespace) -> int:
    upstream = resolve_upstream(skip_bootstrap=args.skip_bootstrap)
    data = args.data.expanduser().resolve()
    exp_dir = (args.exp_dir or _default_exp_dir(args.model, args.phase)).expanduser().resolve()
    config = _resolve_config(args, args.model, args.phase)
    config = materialize_config(
        config,
        exp_dir / "runtime_config.yaml",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=args.pin_memory,
        amp=args.amp,
    )
    num_avg = args.num_avg if args.num_avg is not None else default_num_avg(args.phase)
    if args.server_vox2:
        # The project-specific path uses raw or shard manifests generated from
        # server indexes and our entrypoint for epoch-level Vox2-test validation.
        reverb_data, noise_data = _server_augmentation_paths(args, data)
        _require_server_vox2_training_data(
            data,
            args.data_type,
            reverb_data=reverb_data,
            noise_data=noise_data,
        )
        checkpoint = None
        if args.phase == "lm":
            source_checkpoint = (
                args.checkpoint.expanduser().resolve()
                if args.checkpoint
                else _default_exp_dir(args.model, "ptn") / "models" / "avg_model.pt"
            )
            if not source_checkpoint.is_file():
                raise FileNotFoundError(
                    f"LM checkpoint not found: {source_checkpoint}; "
                    "run PTN first or pass --checkpoint"
                )
            checkpoint = stage_lm_checkpoint(source_checkpoint, exp_dir)
        elif args.checkpoint:
            checkpoint = args.checkpoint.expanduser().resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(f"PTN checkpoint not found: {checkpoint}")

        command = build_epoch_train_command(
            project_root=ROOT,
            upstream_dir=upstream,
            data_dir=data,
            config_path=config,
            exp_dir=exp_dir,
            gpus=args.gpus,
            checkpoint=checkpoint,
            data_type=args.data_type,
            reverb_data=reverb_data,
            noise_data=noise_data,
            num_nodes=args.num_nodes,
            host_node_addr=args.host_node_addr,
            job_id=args.job_id,
            eval_data=data / "vox2_test" / "raw.list",
            eval_trials=(
                args.eval_trials.expanduser().resolve()
                if args.eval_trials
                else data / "vox2_test" / "trials" / "vox2_test_fixed.kaldi"
            ),
            eval_interval=args.eval_interval,
            eval_num_workers=args.eval_num_workers,
        )
        _run(command, cwd=ROOT, env={"WESPEAKER_DIR": str(upstream)})
        _run(
            build_average_command(
                upstream_dir=upstream,
                exp_dir=exp_dir,
                num_avg=num_avg,
            ),
            cwd=upstream,
        )
        return 0

    # The legacy VoxCeleb1 path below stays delegated to WeSpeaker's official
    # recipe and is only retained for paper-protocol final evaluation.
    checkpoint = None
    if args.phase == "lm":
        _require_converted_data(
            data, args.data_type, require_augmentation=False
        )
        source_checkpoint = (
            args.checkpoint.expanduser().resolve()
            if args.checkpoint
            else _default_exp_dir(args.model, "ptn") / "models" / "avg_model.pt"
        )
        if not source_checkpoint.is_file():
            raise FileNotFoundError(
                f"LM checkpoint not found: {source_checkpoint}; "
                "run PTN first or pass --checkpoint"
            )
        checkpoint = stage_lm_checkpoint(source_checkpoint, exp_dir)
        stage, stop_stage = 3, 3
    else:
        if args.checkpoint:
            checkpoint = args.checkpoint.expanduser().resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(f"PTN checkpoint not found: {checkpoint}")
        if args.skip_conversion:
            _require_converted_data(data, args.data_type)
            stage = 3
        else:
            stage = 2
        stop_stage = 3

    command = build_upstream_run_command(
        upstream_dir=upstream,
        data_dir=data,
        config_path=config,
        gpus=args.gpus,
        stage=stage,
        stop_stage=stop_stage,
        exp_dir=exp_dir,
        num_avg=num_avg,
        checkpoint=checkpoint,
        data_type=args.data_type,
        num_nodes=args.num_nodes,
        host_node_addr=args.host_node_addr,
        job_id=args.job_id,
    )
    _run(command, cwd=_recipe_dir(upstream))
    _run(
        build_average_command(
            upstream_dir=upstream,
            exp_dir=exp_dir,
            num_avg=num_avg,
        ),
        cwd=upstream,
    )
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    upstream = resolve_upstream(skip_bootstrap=args.skip_bootstrap)
    data = args.data.expanduser().resolve()
    exp_dir = (args.exp_dir or _default_exp_dir(args.model, args.phase)).expanduser().resolve()
    _require_evaluation_data(data, args.data_type)
    config = _resolve_config(args, args.model, args.phase)
    extract_command = build_upstream_run_command(
        upstream_dir=upstream,
        data_dir=data,
        config_path=config,
        gpus=args.gpus,
        stage=4,
        stop_stage=6 if args.asnorm else 4,
        exp_dir=exp_dir,
        num_avg=(
            args.num_avg
            if args.num_avg is not None
            else default_num_avg(args.phase)
        ),
        data_type=args.data_type,
        num_nodes=args.num_nodes,
        host_node_addr=args.host_node_addr,
        job_id=args.job_id,
    )
    _run(extract_command, cwd=_recipe_dir(upstream))
    if args.asnorm:
        return 0

    for trial_name in TRIAL_NAMES:
        _run(
            build_raw_score_command(
                upstream_dir=upstream,
                data_dir=data,
                exp_dir=exp_dir,
                trial_name=trial_name,
            ),
            cwd=upstream,
        )
        _run(
            [
                sys.executable,
                "wespeaker/bin/compute_metrics.py",
                "--p_target",
                "0.01",
                "--c_fa",
                "1",
                "--c_miss",
                "1",
                str(exp_dir / "scores" / f"{trial_name}.score"),
            ],
            cwd=upstream,
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("bootstrap", help="clone and pin WeSpeaker")

    validate = subparsers.add_parser("validate", help="validate prepared manifests")
    add_data_args(validate)
    validate.add_argument("--check-files", action="store_true")

    validate_server = subparsers.add_parser(
        "validate-server-vox2", help="validate prepared server VoxCeleb2 manifests"
    )
    add_data_args(validate_server)
    validate_server.add_argument("--check-files", action="store_true")

    prepare = subparsers.add_parser("prepare", help="run upstream data preparation")
    add_data_args(prepare)
    add_upstream_args(prepare)
    prepare.add_argument("--stage", type=int, default=1)
    prepare.add_argument("--stop-stage", type=int, default=4)

    prepare_server = subparsers.add_parser(
        "prepare-server-vox2",
        help="convert server dev.spk2utt/test.spk2utt into WeSpeaker manifests",
    )
    prepare_server.add_argument("--corpus", type=Path, required=True)
    add_data_args(prepare_server)
    prepare_server.add_argument("--data-type", choices=DATA_TYPES, default="raw")
    prepare_server.add_argument("--pairs-per-speaker", type=int, default=300)
    prepare_server.add_argument("--seed", type=int, default=42)
    prepare_server.add_argument("--check-files", action="store_true")
    prepare_server.add_argument("--shard-workers", type=int, default=16)
    prepare_server.add_argument("--utts-per-shard", type=int, default=1000)
    add_upstream_args(prepare_server)

    train = subparsers.add_parser("train", help="run PTN or LM training")
    add_data_args(train)
    add_data_type_args(train)
    add_upstream_args(train)
    add_distributed_args(train)
    add_runtime_config_args(train)
    train.add_argument("--model", choices=MODEL_NAMES, default="b3")
    train.add_argument("--phase", choices=PHASES, default="ptn")
    train.add_argument("--gpus", default="0")
    train.add_argument("--exp-dir", type=Path)
    train.add_argument("--checkpoint", type=Path)
    train.add_argument("--num-avg", type=int)
    train.add_argument(
        "--server-vox2",
        action="store_true",
        help="train prepared Vox2 server data and validate with fixed test trials",
    )
    train.add_argument("--eval-interval", type=int, default=1)
    train.add_argument("--eval-num-workers", type=int, default=1)
    train.add_argument("--eval-trials", type=Path)
    train.add_argument("--musan-lmdb", type=Path)
    train.add_argument("--rirs-lmdb", type=Path)
    train.add_argument(
        "--skip-conversion",
        action="store_true",
        help="skip local shard/raw/LMDB conversion; require prepared artifacts",
    )

    evaluate = subparsers.add_parser("evaluate", help="extract embeddings and score Vox1")
    add_data_args(evaluate)
    add_data_type_args(evaluate)
    add_upstream_args(evaluate)
    add_distributed_args(evaluate)
    add_config_arg(evaluate)
    evaluate.add_argument("--model", choices=MODEL_NAMES, default="b3")
    evaluate.add_argument("--phase", choices=PHASES, default="lm")
    evaluate.add_argument("--gpus", default="0")
    evaluate.add_argument("--exp-dir", type=Path)
    evaluate.add_argument("--num-avg", type=int)
    evaluate.add_argument(
        "--asnorm",
        action="store_true",
        help="run upstream mean-centered cosine + top-300 AS-Norm instead of raw cosine",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "bootstrap":
        ensure_upstream()
        return 0
    if args.command == "validate":
        return command_validate(args)
    if args.command == "validate-server-vox2":
        return command_validate_server_vox2(args)
    if args.command == "prepare":
        return command_prepare(args)
    if args.command == "prepare-server-vox2":
        return command_prepare_server_vox2(args)
    if args.command == "train":
        return command_train(args)
    if args.command == "evaluate":
        return command_evaluate(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
