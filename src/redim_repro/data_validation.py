"""Validate project-generated manifests before a costly WeSpeaker launch.

These checks are intentionally dependency-light: they catch inconsistent
indexes, trials and JSON-lines lists before a GPU job imports WeSpeaker or
starts distributed workers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple


class ManifestError(ValueError):
    """Raised when a manifest or trial file is internally inconsistent."""


def _read_map(path: Path, name: str) -> Dict[str, str]:
    if not path.is_file():
        raise ManifestError(f"missing {name}: {path}")

    values: Dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text().splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise ManifestError(
                f"malformed {name} at {path}:{line_number}; expected key and value"
            )
        key, value = fields
        if key in values:
            raise ManifestError(f"duplicate key {key!r} in {name}: {path}:{line_number}")
        values[key] = value
    if not values:
        raise ManifestError(f"empty {name}: {path}")
    return values


def _validate_dataset_dir(path: Path, *, check_files: bool) -> Dict[str, int]:
    wavs = _read_map(path / "wav.scp", f"{path.name}/wav.scp")
    speakers = _read_map(path / "utt2spk", f"{path.name}/utt2spk")
    wav_keys = set(wavs)
    speaker_keys = set(speakers)
    if wav_keys != speaker_keys:
        missing_speaker = sorted(wav_keys - speaker_keys)[:3]
        missing_wav = sorted(speaker_keys - wav_keys)[:3]
        raise ManifestError(
            f"{path.name} manifest keys differ: missing utt2spk={missing_speaker}, "
            f"missing wav.scp={missing_wav}"
        )

    if check_files:
        for utt, wav_spec in wavs.items():
            if "|" in wav_spec:
                continue
            wav_path = Path(wav_spec)
            if not wav_path.is_file():
                raise ManifestError(f"audio path for {utt} does not exist: {wav_path}")

    return {
        "utterances": len(wavs),
        "speakers": len(set(speakers.values())),
    }


def _trial_fields(fields: Iterable[str]) -> Tuple[str, str, str]:
    fields = tuple(fields)
    if len(fields) != 3:
        raise ManifestError(
            "malformed trial row; expected enroll verify label or label enroll verify"
        )
    labels = {"target", "nontarget", "1", "0"}
    if fields[0] in labels:
        label, enroll, verify = fields
    else:
        enroll, verify, label = fields
    if label not in labels:
        raise ManifestError(f"unsupported trial label: {label!r}")
    return enroll, verify, label


def _validate_trials(path: Path, valid_utts: set[str]) -> int:
    if not path.is_file():
        raise ManifestError(f"missing trial file: {path}")

    count = 0
    for line_number, raw_line in enumerate(path.read_text().splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            enroll, verify, _ = _trial_fields(line.split())
        except ManifestError as exc:
            raise ManifestError(f"{path}:{line_number}: {exc}") from exc
        for utt in (enroll, verify):
            if utt not in valid_utts:
                raise ManifestError(
                    f"trial reference {utt!r} is missing from wav.scp: {path}:{line_number}"
                )
        count += 1
    if count == 0:
        raise ManifestError(f"empty trial file: {path}")
    return count


def _validate_raw_list(
    path: Path,
    wavs: Dict[str, str],
    speakers: Dict[str, str],
) -> int:
    """Verify that a WeSpeaker raw-list row mirrors ``wav.scp``/``utt2spk``."""

    if not path.is_file():
        raise ManifestError(f"missing raw list: {path}")
    seen: set[str] = set()
    for line_number, raw_line in enumerate(path.read_text().splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ManifestError(f"malformed JSON in raw list {path}:{line_number}") from exc
        if not isinstance(item, dict):
            raise ManifestError(f"raw list row is not an object: {path}:{line_number}")
        key = item.get("key")
        if not isinstance(key, str) or key not in wavs:
            raise ManifestError(
                f"raw list key is missing from wav.scp: {key!r} at {path}:{line_number}"
            )
        if key in seen:
            raise ManifestError(f"duplicate raw list key {key!r}: {path}:{line_number}")
        if item.get("spk") != speakers[key]:
            raise ManifestError(
                f"raw list speaker differs from utt2spk for {key!r}: {path}:{line_number}"
            )
        if item.get("wav") != wavs[key]:
            raise ManifestError(
                f"raw list wav differs from wav.scp for {key!r}: {path}:{line_number}"
            )
        seen.add(key)
    if seen != set(wavs):
        missing = sorted(set(wavs) - seen)[:3]
        raise ManifestError(f"raw list keys differ from wav.scp; missing={missing}")
    return len(seen)


def validate_voxceleb_layout(root: Path | str, *, check_files: bool = False) -> dict:
    """Validate a WeSpeaker VoxCeleb data directory and return JSON-safe counts.

    The expected layout is ``vox2_dev/{wav.scp,utt2spk}`` and
    ``vox1/{wav.scp,utt2spk,trials/*.kaldi}``. Audio paths are only checked when
    ``check_files`` is true because checking a complete VoxCeleb2 manifest can
    take noticeable time on network filesystems.
    """

    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ManifestError(f"data root does not exist: {root}")

    vox2_dir = root / "vox2_dev"
    vox1_dir = root / "vox1"
    vox2_report = _validate_dataset_dir(vox2_dir, check_files=check_files)
    vox1_report = _validate_dataset_dir(vox1_dir, check_files=check_files)

    trial_dir = vox1_dir / "trials"
    trial_files = sorted(trial_dir.glob("*.kaldi"))
    if not trial_files:
        raise ManifestError(f"no .kaldi trial files found: {trial_dir}")
    valid_utts = set(_read_map(vox1_dir / "wav.scp", "vox1/wav.scp"))
    trial_report = {
        path.name: _validate_trials(path, valid_utts) for path in trial_files
    }

    vox1_report["trials"] = trial_report
    return {
        "root": str(root),
        "vox2_dev": vox2_report,
        "vox1": vox1_report,
    }


def validate_server_vox2_layout(
    root: Path | str, *, check_files: bool = False
) -> dict:
    """Validate the project-specific server layout and fixed validation trials.

    Unlike :func:`validate_voxceleb_layout`, this path has no VoxCeleb1 data:
    it validates the prepared VoxCeleb2 test split used for model selection.
    """

    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ManifestError(f"data root does not exist: {root}")

    reports: dict[str, dict] = {}
    for dataset_name in ("vox2_dev", "vox2_test"):
        dataset_dir = root / dataset_name
        report = _validate_dataset_dir(dataset_dir, check_files=check_files)
        wavs = _read_map(dataset_dir / "wav.scp", f"{dataset_name}/wav.scp")
        speakers = _read_map(dataset_dir / "utt2spk", f"{dataset_name}/utt2spk")
        report["raw_list"] = _validate_raw_list(
            dataset_dir / "raw.list", wavs, speakers
        )
        reports[dataset_name] = report

    trial_path = root / "vox2_test" / "trials" / "vox2_test_fixed.kaldi"
    test_wavs = _read_map(root / "vox2_test" / "wav.scp", "vox2_test/wav.scp")
    reports["vox2_test"]["trials"] = {
        trial_path.name: _validate_trials(trial_path, set(test_wavs))
    }
    return {"root": str(root), **reports}
