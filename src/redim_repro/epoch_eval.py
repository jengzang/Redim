"""Project-owned fixed-trial validation between WeSpeaker training epochs.

The model and dataset machinery remain in the pinned WeSpeaker dependency.
This module adds the experiment policy missing from its stock recipe: score a
stable VoxCeleb2-test trial set at epoch boundaries and retain the checkpoint
with the best validation EER.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import json
import math
import os
from pathlib import Path
import time
from typing import Mapping, Sequence

from .vox2_index import Trial


class EpochEvaluationError(ValueError):
    """Raised when fixed-trial scoring cannot produce a valid metric."""


@dataclass(frozen=True)
class EvaluationResult:
    """Metrics and accounting for one fixed-trial validation pass."""

    eer: float
    min_dcf: float
    target_trials: int
    nontarget_trials: int
    elapsed_seconds: float = 0.0


def read_trials(path: Path | str) -> list[Trial]:
    """Read a strict ``enroll verify target|nontarget`` Kaldi trial file."""

    path = Path(path)
    if not path.is_file():
        raise EpochEvaluationError(f"missing evaluation trial file: {path}")
    trials: list[Trial] = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 3 or fields[2] not in {"target", "nontarget"}:
            raise EpochEvaluationError(
                f"malformed fixed trial at {path}:{line_number}; "
                "expected enroll verify target|nontarget"
            )
        enroll, verify, label = fields
        if enroll == verify:
            raise EpochEvaluationError(
                f"fixed trial uses one utterance twice: {path}:{line_number}"
            )
        trials.append(Trial(enroll=enroll, verify=verify, label=label))
    if not trials:
        raise EpochEvaluationError(f"empty evaluation trial file: {path}")
    return trials


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise EpochEvaluationError("embedding dimensions must be equal and non-empty")
    numerator = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise EpochEvaluationError("cannot score a zero-norm embedding")
    return numerator / (left_norm * right_norm)


def _fallback_metrics(scores: Sequence[float], labels: Sequence[bool]) -> tuple[float, float]:
    targets = [score for score, label in zip(scores, labels) if label]
    nontargets = [score for score, label in zip(scores, labels) if not label]
    if not targets or not nontargets:
        raise EpochEvaluationError("EER requires target and nontarget trials")
    thresholds = sorted(set(scores), reverse=True)
    points: list[tuple[float, float]] = []
    for threshold in thresholds:
        fnr = sum(score < threshold for score in targets) / len(targets)
        fpr = sum(score >= threshold for score in nontargets) / len(nontargets)
        points.append((fnr, fpr))
    fnr, fpr = min(points, key=lambda point: (abs(point[0] - point[1]), sum(point)))
    eer = (fnr + fpr) / 2.0
    p_target = 0.01
    min_dcf = min(
        p_target * current_fnr + (1.0 - p_target) * current_fpr
        for current_fnr, current_fpr in points
    ) / p_target
    return eer, min_dcf


def _metrics(scores: Sequence[float], labels: Sequence[bool]) -> tuple[float, float]:
    """Use WeSpeaker's metric implementation when installed; keep a test fallback."""

    try:
        import numpy as np
        from wespeaker.utils.score_metrics import (
            compute_c_norm,
            compute_eer,
            compute_pmiss_pfa_rbst,
        )
    except ImportError:
        return _fallback_metrics(scores, labels)
    score_array = np.asarray(scores, dtype=float)
    label_array = np.asarray(labels, dtype=bool)
    fnr, fpr = compute_pmiss_pfa_rbst(score_array, label_array)
    eer, _ = compute_eer(fnr, fpr, score_array)
    min_dcf = compute_c_norm(fnr, fpr, p_target=0.01, c_miss=1, c_fa=1)
    return float(eer), float(min_dcf)


def score_trials(
    embeddings: Mapping[str, Sequence[float]], trials: Sequence[Trial]
) -> EvaluationResult:
    """Score fixed trials with raw cosine and return EER/minDCF."""

    scores: list[float] = []
    labels: list[bool] = []
    for trial in trials:
        try:
            enroll = embeddings[trial.enroll]
            verify = embeddings[trial.verify]
        except KeyError as exc:
            raise EpochEvaluationError(
                f"missing embedding for fixed trial utterance: {exc.args[0]}"
            ) from exc
        scores.append(_cosine(enroll, verify))
        labels.append(trial.label == "target")
    eer, min_dcf = _metrics(scores, labels)
    return EvaluationResult(
        eer=eer,
        min_dcf=min_dcf,
        target_trials=sum(labels),
        nontarget_trials=len(labels) - sum(labels),
    )


def disable_distributed_partition(dataset) -> None:
    """Make a rank-0-only evaluation iterator cover every indexed utterance.

    WeSpeaker's ``DataList`` normally partitions samples by DDP rank.  Only
    rank 0 evaluates here, so leaving that partition enabled would silently
    embed only one worker's fraction of test utterances and invalidate trials.
    """

    node = dataset
    while node is not None:
        sampler = getattr(node, "sampler", None)
        if sampler is not None and hasattr(sampler, "partition"):
            sampler.partition = False
            return
        node = getattr(node, "source", None)
    raise EpochEvaluationError(
        "could not locate WeSpeaker DataList sampler to disable DDP partition"
    )


def evaluate_in_memory_model(
    model,
    configs: Mapping,
    data_list: Path | str,
    trials_path: Path | str,
    device,
    *,
    num_workers: int = 1,
) -> EvaluationResult:
    """Extract full-utterance embeddings with the live WeSpeaker model once.

    Augmentation and random cropping are disabled for validation.  The caller
    runs this only on rank 0, while all other ranks wait at barriers in the
    training entrypoint before collective training resumes.
    """

    try:
        import torch
        from torch.utils.data import DataLoader
        from wespeaker.dataset.dataset import Dataset
        from wespeaker.dataset.dataset_utils import apply_cmvn
    except ImportError as exc:  # pragma: no cover - requires server runtime
        raise RuntimeError(
            "in-memory epoch evaluation requires the pinned WeSpeaker runtime"
        ) from exc

    data_list = Path(data_list)
    if not data_list.is_file():
        raise EpochEvaluationError(f"missing evaluation raw list: {data_list}")
    test_conf = copy.deepcopy(configs["dataset_args"])
    test_conf.update(
        {
            "speed_perturb": False,
            "spec_aug": False,
            "shuffle": False,
            "aug_prob": 0.0,
            "filter": False,
        }
    )
    dataset = Dataset(
        "raw",
        str(data_list),
        test_conf,
        spk2id_dict={},
        whole_utt=True,
        repeat_dataset=False,
    )
    disable_distributed_partition(dataset)
    dataloader_args = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        dataloader_args["prefetch_factor"] = 4
    dataloader = DataLoader(dataset, **dataloader_args)

    was_training = model.training
    model.eval()
    embeddings: dict[str, list[float]] = {}
    started_at = time.monotonic()
    with torch.inference_mode():
        for batch in dataloader:
            wavs = batch["wav"].squeeze(1).float().to(device)
            wavs_len = torch.full(
                (wavs.shape[0],), wavs.shape[1], dtype=torch.long, device=device
            )
            features, _ = model.frontend(wavs, wavs_len)
            if test_conf.get("cmvn", True):
                features = apply_cmvn(features, **test_conf.get("cmvn_args", {}))
            outputs = model(features)
            embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
            for utterance, embedding in zip(batch["key"], embeds.detach().cpu().tolist()):
                embeddings[utterance] = embedding
    if was_training:
        model.train()
    result = score_trials(embeddings, read_trials(trials_path))
    return EvaluationResult(
        eer=result.eer,
        min_dcf=result.min_dcf,
        target_trials=result.target_trials,
        nontarget_trials=result.nontarget_trials,
        elapsed_seconds=time.monotonic() - started_at,
    )


def is_better(candidate: Mapping[str, float], current: Mapping[str, float]) -> bool:
    """Choose lower EER, then lower minDCF deterministically."""

    return (candidate["eer"], candidate["min_dcf"]) < (
        current["eer"],
        current["min_dcf"],
    )


def _replace_best_symlink(checkpoint_path: Path, destination: Path) -> None:
    """Atomically replace the stable best-model link after a successful epoch."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    relative_target = os.path.relpath(checkpoint_path, destination.parent.resolve())
    os.symlink(relative_target, temporary)
    os.replace(temporary, destination)


def write_epoch_result(
    exp_dir: Path | str,
    epoch: int,
    result: EvaluationResult,
    checkpoint_path: Path | str,
) -> dict:
    """Persist one result and update ``best_model.pt`` using EER then minDCF.

    The link is outside ``models/`` so WeSpeaker's upstream checkpoint-averager
    cannot mistake it for another epoch checkpoint.
    """

    if epoch < 1:
        raise ValueError("epoch must be positive")
    exp_dir = Path(exp_dir)
    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing checkpoint for epoch result: {checkpoint_path}")
    epoch_dir = exp_dir / "epoch_eval"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    payload = {"epoch": epoch, "checkpoint": str(checkpoint_path), **asdict(result)}
    metrics_path = epoch_dir / "metrics.jsonl"
    with metrics_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
    (epoch_dir / f"epoch_{epoch:03d}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    best_path = epoch_dir / "best.json"
    current = json.loads(best_path.read_text()) if best_path.is_file() else None
    if current is None or is_better(payload, current):
        best_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _replace_best_symlink(checkpoint_path, exp_dir / "best_model.pt")
        return payload
    return current
