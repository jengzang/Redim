"""Turn the server's authoritative VoxCeleb2 indexes into WeSpeaker inputs.

This is project-owned glue code; it deliberately does not import WeSpeaker.
The server's ``*.spk2utt`` files have one ``speaker relative/wav/path`` pair per
line (rather than Kaldi's usual one-speaker-per-line form).  Reading those
indexes, instead of recursively walking the corpus, makes the exact training
and validation population reproducible.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import random
from typing import Iterable, Sequence


class ServerIndexError(ValueError):
    """Raised when a server VoxCeleb2 index cannot be used safely."""


@dataclass(frozen=True)
class IndexRecord:
    """One authoritative speaker/audio mapping from a server index row."""

    speaker: str
    utt_id: str
    audio_path: Path


@dataclass(frozen=True)
class Trial:
    """A Kaldi-style verification trial."""

    enroll: str
    verify: str
    label: str


def _canonical_utt_id(relative_path: PurePosixPath, split: str) -> str:
    expected_prefix = PurePosixPath(split) / "audio"
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ServerIndexError(f"audio path must be a safe relative path: {relative_path}")
    if relative_path.parts[:2] != expected_prefix.parts:
        raise ServerIndexError(
            f"expected {expected_prefix.as_posix()}/ path, got {relative_path}"
        )
    if relative_path.suffix.lower() != ".wav":
        raise ServerIndexError(f"expected .wav audio path, got {relative_path}")
    utterance_path = relative_path.relative_to(expected_prefix).with_suffix("")
    if len(utterance_path.parts) < 3:
        raise ServerIndexError(
            f"expected speaker/video/utterance under {expected_prefix}, got {relative_path}"
        )
    return utterance_path.as_posix()


def read_server_index(
    index_path: Path | str,
    corpus_root: Path | str,
    split: str,
    *,
    check_files: bool = False,
) -> list[IndexRecord]:
    """Read one source index and convert paths to stable utterance identifiers.

    For example, ``test/audio/id04627/video/00022.wav`` becomes
    ``id04627/video/00022``.  That identifier is shared by ``wav.scp``,
    ``raw.list`` and trials, so all later stages address the same utterance.
    """

    if split not in {"dev", "test"}:
        raise ValueError(f"unsupported VoxCeleb2 split: {split!r}")
    index_path = Path(index_path).expanduser().resolve()
    corpus_root = Path(corpus_root).expanduser().resolve()
    if not corpus_root.is_dir():
        raise ServerIndexError(f"corpus root does not exist: {corpus_root}")
    if not index_path.is_file():
        raise ServerIndexError(f"missing {split} index: {index_path}")

    records: list[IndexRecord] = []
    seen_utts: dict[str, str] = {}
    for line_number, raw_line in enumerate(index_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ServerIndexError(
                f"malformed {split} index at {index_path}:{line_number}; "
                "expected speaker and relative wav path"
            )
        speaker, raw_relative_path = fields
        if not speaker:
            raise ServerIndexError(
                f"empty speaker id in {split} index at {index_path}:{line_number}"
            )
        relative_path = PurePosixPath(raw_relative_path)
        utt_id = _canonical_utt_id(relative_path, split)
        audio_path = (corpus_root / Path(relative_path)).resolve()
        if corpus_root not in audio_path.parents:
            raise ServerIndexError(
                f"audio path escapes corpus root at {index_path}:{line_number}: "
                f"{raw_relative_path}"
            )
        if check_files and not audio_path.is_file():
            raise ServerIndexError(
                f"indexed audio path does not exist at {index_path}:{line_number}: "
                f"{audio_path}"
            )
        if utt_id in seen_utts:
            raise ServerIndexError(
                f"duplicate utterance id {utt_id!r} in {split} index; "
                f"speakers {seen_utts[utt_id]!r} and {speaker!r}"
            )
        seen_utts[utt_id] = speaker
        records.append(IndexRecord(speaker=speaker, utt_id=utt_id, audio_path=audio_path))

    if not records:
        raise ServerIndexError(f"empty {split} index: {index_path}")
    return sorted(records, key=lambda record: (record.utt_id, record.speaker))


def _write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def _write_manifests(records: Sequence[IndexRecord], dataset_dir: Path) -> None:
    """Write both Kaldi manifests and WeSpeaker's JSON-lines raw input list."""

    by_speaker: dict[str, list[str]] = defaultdict(list)
    for record in records:
        by_speaker[record.speaker].append(record.utt_id)
    _write_lines(
        dataset_dir / "wav.scp",
        (f"{record.utt_id} {record.audio_path}" for record in records),
    )
    _write_lines(
        dataset_dir / "utt2spk",
        (f"{record.utt_id} {record.speaker}" for record in records),
    )
    _write_lines(
        dataset_dir / "spk2utt",
        (
            f"{speaker} {' '.join(sorted(utterances))}"
            for speaker, utterances in sorted(by_speaker.items())
        ),
    )
    _write_lines(
        dataset_dir / "raw.list",
        (
            json.dumps(
                {
                    "key": record.utt_id,
                    "spk": record.speaker,
                    "wav": str(record.audio_path),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            for record in records
        ),
    )


def _target_pair_at(utterances: Sequence[str], index: int) -> tuple[str, str]:
    """Map one index in ``range(n choose 2)`` to its utterance pair."""

    for left_index in range(len(utterances) - 1):
        row_size = len(utterances) - left_index - 1
        if index < row_size:
            return utterances[left_index], utterances[left_index + index + 1]
        index -= row_size
    raise IndexError("target pair index is outside the combination space")


def _sample_target_pairs(
    utterances: Sequence[str], count: int, rng: random.Random
) -> list[tuple[str, str]]:
    """Sample target pairs by combination index without building ``n choose 2`` tuples."""

    candidate_count = len(utterances) * (len(utterances) - 1) // 2
    if candidate_count < count:
        raise ServerIndexError(
            f"cannot create {count} target trials from {candidate_count} unique pairs"
        )
    return [
        _target_pair_at(utterances, index)
        for index in rng.sample(range(candidate_count), count)
    ]


def _sample_nontarget_pairs(
    enroll_utterances: Sequence[str],
    other_utterances: Sequence[str],
    *,
    count: int,
    rng: random.Random,
    used_pairs: set[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Sample cross-speaker pairs by index instead of materializing a product.

    A complete VoxCeleb2-test Cartesian product is much too large to hold in
    memory.  Each candidate rank maps directly to one ``enroll × other`` pair,
    so this function retains only the requested trials and their small set of
    sampled ranks.  ``used_pairs`` prevents a later speaker from adding the
    same unordered cross-speaker pair in reverse order.
    """

    other_count = len(other_utterances)
    candidate_count = len(enroll_utterances) * other_count
    if candidate_count < count:
        raise ServerIndexError(
            f"cannot create {count} nontarget trials from {candidate_count} candidates"
        )

    sampled_ranks: set[int] = set()
    selected: list[tuple[str, str]] = []
    while len(selected) < count and len(sampled_ranks) < candidate_count:
        rank = rng.randrange(candidate_count)
        if rank in sampled_ranks:
            continue
        sampled_ranks.add(rank)
        enroll = enroll_utterances[rank // other_count]
        verify = other_utterances[rank % other_count]
        pair = tuple(sorted((enroll, verify)))
        if pair in used_pairs:
            continue
        used_pairs.add(pair)
        selected.append(pair)

    if len(selected) != count:
        raise ServerIndexError(
            f"cannot create {count} unique nontarget trials from {candidate_count} candidates"
        )
    return selected


def build_fixed_trials(
    records: Sequence[IndexRecord], *, pairs_per_speaker: int, seed: int
) -> list[Trial]:
    """Build balanced, deterministic target/nontarget trials from test records.

    Each test speaker contributes ``pairs_per_speaker`` same-speaker pairs and
    the same number of cross-speaker pairs.  Keeping a seed and sorting the
    result means an EER from one epoch is comparable with every later epoch.
    """

    if pairs_per_speaker < 1:
        raise ValueError("pairs_per_speaker must be positive")
    by_speaker: dict[str, list[str]] = defaultdict(list)
    for record in records:
        by_speaker[record.speaker].append(record.utt_id)
    speakers = sorted(by_speaker)
    if len(speakers) < 2:
        raise ServerIndexError("test index requires at least two speakers")
    for utterances in by_speaker.values():
        utterances.sort()

    rng = random.Random(seed)
    used_nontargets: set[tuple[str, str]] = set()
    trials: list[Trial] = []
    for speaker in speakers:
        utterances = by_speaker[speaker]
        selected_targets = _sample_target_pairs(utterances, pairs_per_speaker, rng)
        trials.extend(Trial(enroll, verify, "target") for enroll, verify in selected_targets)

        other_utterances = [
            utterance
            for other_speaker in speakers
            if other_speaker != speaker
            for utterance in by_speaker[other_speaker]
        ]
        selected_nontargets = _sample_nontarget_pairs(
            utterances,
            other_utterances,
            count=pairs_per_speaker,
            rng=rng,
            used_pairs=used_nontargets,
        )
        trials.extend(
            Trial(enroll, verify, "nontarget")
            for enroll, verify in selected_nontargets
        )
    return sorted(trials, key=lambda trial: (trial.label, trial.enroll, trial.verify))


def _index_checksum(index_path: Path) -> str:
    return sha256(index_path.read_bytes()).hexdigest()


def prepare_server_vox2(
    corpus_root: Path | str,
    output_root: Path | str,
    *,
    pairs_per_speaker: int = 300,
    seed: int = 42,
    check_files: bool = False,
) -> dict:
    """Materialize train/validation inputs without scanning the audio tree.

    ``dev.spk2utt`` is the training source.  ``test.spk2utt`` becomes a fixed
    validation population and trial list, not an official VoxCeleb1 benchmark.
    The JSON sidecar records index checksums and sampling settings so a server
    run remains traceable even if the corpus changes later.
    """

    corpus_root = Path(corpus_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    dev_index = corpus_root / "dev.spk2utt"
    test_index = corpus_root / "test.spk2utt"
    dev_records = read_server_index(
        dev_index, corpus_root, "dev", check_files=check_files
    )
    test_records = read_server_index(
        test_index, corpus_root, "test", check_files=check_files
    )

    dev_dir = output_root / "vox2_dev"
    test_dir = output_root / "vox2_test"
    _write_manifests(dev_records, dev_dir)
    _write_manifests(test_records, test_dir)
    trials = build_fixed_trials(
        test_records, pairs_per_speaker=pairs_per_speaker, seed=seed
    )
    trial_dir = test_dir / "trials"
    _write_lines(
        trial_dir / "vox2_test_fixed.kaldi",
        (f"{trial.enroll} {trial.verify} {trial.label}" for trial in trials),
    )
    target_trials = sum(trial.label == "target" for trial in trials)
    nontarget_trials = len(trials) - target_trials
    metadata = {
        "dev_index_sha256": _index_checksum(dev_index),
        "nontarget_trials": nontarget_trials,
        "pairs_per_speaker": pairs_per_speaker,
        "seed": seed,
        "target_trials": target_trials,
        "test_index_sha256": _index_checksum(test_index),
    }
    (trial_dir / "vox2_test_fixed.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "corpus_root": str(corpus_root),
        "output_root": str(output_root),
        "vox2_dev": {
            "utterances": len(dev_records),
            "speakers": len({record.speaker for record in dev_records}),
        },
        "vox2_test": {
            "utterances": len(test_records),
            "speakers": len({record.speaker for record in test_records}),
            "target_trials": target_trials,
            "nontarget_trials": nontarget_trials,
        },
    }
