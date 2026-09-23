# VoxCeleb2 Indexed Per-Epoch Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train ReDimNet2 from the server's VoxCeleb2 dev/test index files and record a fixed VoxCeleb2-test EER/minDCF after every epoch.

**Architecture:** A dependency-light index module converts the two-column source indexes into Kaldi manifests and deterministic balanced trials without scanning the corpus.  A local DDP entrypoint mirrors the pinned WeSpeaker train loop and evaluates the rank-0 in-memory backbone at configured epoch boundaries; a command builder invokes it instead of WeSpeaker's Vox1-hardcoded recipe.

**Tech Stack:** Python 3.9+, PyYAML, PyTorch/torchaudio, pinned WeSpeaker `9fecd6cb4f47475d01761d87c826298dff4ef18c`, pytest.

---

### Task 1: Parse and materialize server VoxCeleb2 indexes

**Files:**

- Create: `src/redim_repro/vox2_index.py`
- Create: `tests/test_vox2_index.py`
- Modify: `src/redim_repro/__init__.py`

- [ ] **Step 1: Write failing conversion and validation tests**

```python
def test_prepare_indexes_writes_canonical_manifests_and_raw_lists(tmp_path):
    corpus = tmp_path / "corpus"
    _touch(corpus / "dev/audio/id00001/v/00001.wav")
    _touch(corpus / "test/audio/id00002/v/00002.wav")
    _write(corpus / "dev.spk2utt", "vox2_id00001 dev/audio/id00001/v/00001.wav\n")
    _write(corpus / "test.spk2utt", "vox2_id00002 test/audio/id00002/v/00002.wav\n")
    report = prepare_server_vox2(corpus, tmp_path / "prepared", pairs_per_speaker=1, seed=7)
    assert (tmp_path / "prepared/vox2_dev/utt2spk").read_text() == "id00001/v/00001 vox2_id00001\n"
    assert '"key": "id00002/v/00002"' in (tmp_path / "prepared/vox2_test/raw.list").read_text()
    assert report["vox2_test"]["utterances"] == 1

def test_index_rejects_wrong_split_and_duplicate_utterance(tmp_path):
    with pytest.raises(ServerIndexError, match="expected dev/audio"):
        read_server_index(tmp_path / "dev.spk2utt", tmp_path, "dev")
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `python -m pytest tests/test_vox2_index.py -q`

Expected: FAIL because `redim_repro.vox2_index` does not exist.

- [ ] **Step 3: Implement the smallest complete index module**

```python
def canonical_utt_id(relative_path: PurePosixPath, split: str) -> str:
    prefix = PurePosixPath(split) / "audio"
    if relative_path.parts[:2] != prefix.parts or relative_path.suffix != ".wav":
        raise ServerIndexError(...)
    return relative_path.relative_to(prefix).with_suffix("").as_posix()

def write_raw_list(records: Sequence[IndexRecord], destination: Path) -> None:
    _write_lines(destination, [json.dumps({"key": r.utt_id, "spk": r.speaker, "wav": str(r.audio_path)}) for r in records])
```

Implement strict two-column parsing, absolute-path containment checks, duplicate checks, sorted manifest writing, standard `spk2utt`, and a JSON report.  Export the public functions from `__init__.py`.

- [ ] **Step 4: Run the focused test to verify it passes**

Run: `python -m pytest tests/test_vox2_index.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/redim_repro/vox2_index.py src/redim_repro/__init__.py tests/test_vox2_index.py
git commit -m "feat: prepare VoxCeleb2 server indexes"
```

### Task 2: Generate and validate deterministic fixed VoxCeleb2-test trials

**Files:**

- Modify: `src/redim_repro/vox2_index.py`
- Modify: `tests/test_vox2_index.py`
- Modify: `src/redim_repro/data_validation.py`
- Modify: `tests/test_data_validation.py`

- [ ] **Step 1: Write failing trial tests**

```python
def test_fixed_trials_are_reproducible_balanced_and_valid(tmp_path):
    records = _test_records_three_speakers(tmp_path)
    first = build_fixed_trials(records, pairs_per_speaker=2, seed=123)
    second = build_fixed_trials(records, pairs_per_speaker=2, seed=123)
    assert first == second
    assert {row.label for row in first} == {"target", "nontarget"}
    assert sum(row.label == "target" for row in first) == sum(row.label == "nontarget" for row in first)
    assert all(row.enroll != row.verify for row in first)

def test_vox2_server_layout_requires_test_trial_members_in_manifest(tmp_path):
    with pytest.raises(ManifestError, match="missing from wav.scp"):
        validate_server_vox2_layout(tmp_path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_vox2_index.py tests/test_data_validation.py -q`

Expected: FAIL because trial construction and server-layout validation do not exist.

- [ ] **Step 3: Implement deterministic sampling and validation**

```python
def build_fixed_trials(records, *, pairs_per_speaker, seed):
    rng = random.Random(seed)
    # Choose distinct within-speaker pairs and cross-speaker pairs; store sorted
    # canonical pair keys to eliminate duplicates.
    return sorted(trials, key=lambda row: (row.label, row.enroll, row.verify))
```

Write `vox2_test/trials/vox2_test_fixed.kaldi` in `enroll verify label` form and a sidecar JSON containing seed, requested/realized counts, and SHA-256 checksums of the source indexes.  Add `validate_server_vox2_layout`, which checks `vox2_dev`, `vox2_test`, `raw.list`, and fixed-trial references without requiring Vox1.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_vox2_index.py tests/test_data_validation.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/redim_repro/vox2_index.py src/redim_repro/data_validation.py tests/test_vox2_index.py tests/test_data_validation.py
git commit -m "feat: add fixed VoxCeleb2 validation trials"
```

### Task 3: Add in-memory validation scoring utilities

**Files:**

- Create: `src/redim_repro/epoch_eval.py`
- Create: `tests/test_epoch_eval.py`

- [ ] **Step 1: Write failing pure-scoring tests**

```python
def test_score_embeddings_returns_perfect_eer_for_separated_pairs():
    embeddings = {"a": np.array([1.0, 0.0]), "b": np.array([0.9, 0.1]), "c": np.array([0.0, 1.0])}
    result = score_trials(embeddings, [Trial("a", "b", "target"), Trial("a", "c", "nontarget")])
    assert result.eer == pytest.approx(0.0)
    assert result.target_trials == 1
    assert result.nontarget_trials == 1

def test_select_best_prefers_lower_eer_then_lower_mindcf():
    assert is_better({"eer": 0.01, "min_dcf": 0.3}, {"eer": 0.01, "min_dcf": 0.4})
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `python -m pytest tests/test_epoch_eval.py -q`

Expected: FAIL because `epoch_eval` does not exist.

- [ ] **Step 3: Implement CPU-testable scoring and lazy runtime extraction**

```python
def evaluate_model(model, configs, raw_list, trials, device):
    # Import WeSpeaker Dataset only here, turn off speed/augmentation, use
    # whole_utt=True and batch_size=1, and call model.frontend then model.
    ...

def write_epoch_result(exp_dir, epoch, result):
    # Append JSONL and atomically replace epoch_eval/best.json when improved.
    ...
```

Use raw cosine only.  Derive EER/minDCF via WeSpeaker's score metric helpers at runtime, but keep direct NumPy/Python calculation interfaces available to tests.  Save an epoch JSON record and update `best_model.pt` at the experiment root as an atomic symlink to the saved epoch checkpoint; keeping it outside `models/` avoids the upstream averaging glob.

- [ ] **Step 4: Run the focused test to verify it passes**

Run: `python -m pytest tests/test_epoch_eval.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/redim_repro/epoch_eval.py tests/test_epoch_eval.py
git commit -m "feat: score in-memory epoch validation"
```

### Task 4: Add a local DDP trainer with rank-0 epoch evaluation

**Files:**

- Create: `scripts/train_with_epoch_eval.py`
- Modify: `src/redim_repro/runner.py`
- Modify: `tests/test_runner.py`

- [ ] **Step 1: Write failing command-construction tests**

```python
def test_epoch_train_command_uses_torchrun_and_server_validation_paths():
    command = build_epoch_train_command(..., gpus="0,1", eval_data=Path("/data/vox2_test/raw.list"), eval_trials=Path("/data/vox2_test/trials/fixed.kaldi"), eval_interval=1)
    assert command[:6] == ["torchrun", "--nnodes", "2", "--nproc_per_node", "2", "--rdzv_id"]
    assert "scripts/train_with_epoch_eval.py" in " ".join(command)
    assert command[command.index("--eval_interval") + 1] == "1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_runner.py -q`

Expected: FAIL because `build_epoch_train_command` does not exist.

- [ ] **Step 3: Implement the builder and trainer**

```python
for epoch in range(start_epoch, configs["num_epochs"] + 1):
    train_dataset.set_epoch(epoch)
    run_epoch(...)
    must_evaluate = epoch % eval_interval == 0 or epoch == configs["num_epochs"]
    if rank == 0 and (must_evaluate or checkpoint_is_due):
        save_checkpoint(model, model_path)
    dist.barrier(device_ids=[gpu])
    if rank == 0 and must_evaluate:
        evaluate_and_record(model, configs, eval_data, eval_trials, exp_dir, epoch, device)
    dist.barrier(device_ids=[gpu])
```

Mirror the pinned WeSpeaker setup, model construction, scheduler, margin update, and checkpoint loading.  Do not alter the upstream checkout.  Insert its directory into `sys.path` from `WESPEAKER_DIR`, require a CUDA/NCCL torchrun environment, enforce a positive evaluation interval, and pass all source/validation paths from the command builder.

- [ ] **Step 4: Run tests and static compilation**

Run: `python -m pytest tests/test_runner.py tests/test_epoch_eval.py -q && python -m py_compile scripts/train_with_epoch_eval.py`

Expected: PASS and no compile output.

- [ ] **Step 5: Commit**

```bash
git add scripts/train_with_epoch_eval.py src/redim_repro/runner.py tests/test_runner.py
git commit -m "feat: evaluate VoxCeleb2 test during DDP training"
```

### Task 5: Wire the server CLI, presets, and operational documentation

**Files:**

- Modify: `scripts/run_experiment.py`
- Modify: `configs/redimnet2_b3_ptn.yaml`
- Modify: `configs/redimnet2_b6_ptn.yaml`
- Modify: `README.md`
- Modify: `tests/test_runner.py`

- [ ] **Step 1: Write failing CLI/parser tests**

```python
def test_server_prepare_and_train_parser_accept_index_and_epoch_eval_options():
    args = build_parser().parse_args(["prepare-server-vox2", "--corpus", "/vox", "--data", "/prepared"])
    assert args.command == "prepare-server-vox2"
    args = build_parser().parse_args(["train", "--server-vox2", "--eval-interval", "5"])
    assert args.server_vox2 and args.eval_interval == 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_runner.py -q`

Expected: FAIL because the new subcommand and flags are absent.

- [ ] **Step 3: Implement CLI and update recipes/docs**

```python
prepare_server = subparsers.add_parser("prepare-server-vox2")
prepare_server.add_argument("--corpus", type=Path, required=True)
prepare_server.add_argument("--pairs-per-speaker", type=int, default=300)
prepare_server.add_argument("--seed", type=int, default=42)

train.add_argument("--server-vox2", action="store_true")
train.add_argument("--eval-interval", type=int, default=1)
```

For `--server-vox2`, require `vox2_dev/<data_type>.list`, `vox2_test/raw.list`, and `vox2_test/trials/vox2_test_fixed.kaldi`; invoke the local trainer and only require MUSAN/RIRS LMDB during PTN.  Set PTN `save_epoch_interval: 1`.  Document exact server commands, the distinction between validation and final test, raw versus shard preparation, the cadence fallback, and multi-node launch requirements.

- [ ] **Step 4: Run the complete test suite**

Run: `python -m pytest -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_experiment.py configs/redimnet2_b3_ptn.yaml configs/redimnet2_b6_ptn.yaml README.md tests/test_runner.py
git commit -m "feat: expose server VoxCeleb2 epoch evaluation workflow"
```

### Task 6: Final verification

**Files:**

- Modify: `README.md` only if verification exposes an instruction defect.

- [ ] **Step 1: Verify unit tests, syntax, and command help**

Run: `python -m pytest -q && python -m py_compile scripts/run_experiment.py scripts/train_with_epoch_eval.py && python scripts/run_experiment.py --help`

Expected: all tests pass, compilation has no output, and help lists `prepare-server-vox2`.

- [ ] **Step 2: Verify the server path is constructed without CUDA**

Run: `python scripts/run_experiment.py prepare-server-vox2 --help && python scripts/run_experiment.py train --help`

Expected: both commands exit 0 and list the index/epoch-evaluation flags.

- [ ] **Step 3: Record manual-server limitations**

Document that the local machine cannot validate mounted audio paths, CUDA/NCCL, or a full 120-epoch run; the server preflight command performs those checks before training.
