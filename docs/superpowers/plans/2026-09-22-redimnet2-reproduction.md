# ReDimNet2 VoxCeleb2 Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pinned, runnable VoxCeleb2 + WeSpeaker ReDimNet2 reproduction wrapper with B3/B6 PTN and LM recipes, data validation, and evaluation commands.

**Architecture:** Keep the workspace as a thin experiment project. A bootstrap script obtains one pinned WeSpeaker revision, YAML files provide the published ReDimNet2-compatible settings, a Python runner delegates to the upstream VoxCeleb recipe, and a standard-library validator catches malformed manifests before training.

**Tech Stack:** Bash, Python 3.9+, Python standard library, PyTorch/torchaudio and the pinned WeSpeaker dependency, pytest for local tests.

---

### Task 1: Add failing tests for data validation and command construction

**Files:**
- Create: `tests/test_data_validation.py`
- Create: `tests/test_runner.py`

- [x] **Step 1: Write tests for manifest consistency and trial references**

The tests create a temporary Kaldi-style data directory, require valid manifests to return a report, and require duplicate utterance IDs or missing trial references to raise `ManifestError`.

- [x] **Step 2: Write tests for normalized GPU arguments and upstream run arguments**

The tests require `"0,1"` to become `"[0,1]"`, and require the generated command to contain the pinned recipe, data path, config, stage range, GPU list, experiment directory, and optional checkpoint.

- [x] **Step 3: Run the focused tests and verify they fail because the package is not implemented**

Run: `python3 -m pytest -q tests`

Expected: collection failure reporting that `redim_repro` cannot be imported.

### Task 2: Implement pure-Python validation and runner helpers

**Files:**
- Create: `src/redim_repro/__init__.py`
- Create: `src/redim_repro/data_validation.py`
- Create: `src/redim_repro/runner.py`

- [x] **Step 1: Implement manifest parsing and validation**

Parse `wav.scp`, `utt2spk`, and cleaned trial files; reject duplicate keys, malformed rows, mismatched `wav.scp`/`utt2spk` keys, and trial references absent from `vox1/wav.scp`. Return serializable counts for utterances, speakers, and trials.

- [x] **Step 2: Implement runner argument normalization**

Normalize GPU strings and construct the upstream `examples/voxceleb/v2/run.sh` command without executing it. Keep the repository URL and commit in constants so bootstrap and documentation use one source of truth.

- [x] **Step 3: Run the focused tests and verify they pass**

Run: `python3 -m pytest -q tests`

Expected: all tests pass.

### Task 3: Add pinned bootstrap, configs, and command-line entrypoint

**Files:**
- Create: `scripts/bootstrap_wespeaker.sh`
- Create: `scripts/check_data.py`
- Create: `scripts/run_experiment.py`
- Create: `configs/redimnet2_b3_ptn.yaml`
- Create: `configs/redimnet2_b3_lm.yaml`
- Create: `configs/redimnet2_b6_ptn.yaml`
- Create: `configs/redimnet2_b6_lm.yaml`

- [x] **Step 1: Add bootstrap script**

Clone or update the pinned WeSpeaker revision under `.deps/wespeaker`, allow `WESPEAKER_DIR` to override the location, and fail clearly if Git is unavailable.

- [x] **Step 2: Add executable B3 and B6 PTN/LM recipes**

Use TFMel 72, ASTP, 192-dimensional embeddings, SphereFace2-C, the two-stage schedule, and the stage-specific augmentation/segment settings described in the design.

- [x] **Step 3: Add runner subcommands**

Implement `bootstrap`, `validate`, `prepare`, `train`, and `evaluate`. `train --phase ptn` runs upstream data conversion plus training; `train --phase lm` starts from the PTN checkpoint; `evaluate` runs embedding extraction and cosine scoring.

- [x] **Step 4: Run `--help` and syntax checks**

Run: `bash -n scripts/bootstrap_wespeaker.sh && python3 -m py_compile scripts/check_data.py scripts/run_experiment.py src/redim_repro/*.py`

Expected: exit code 0.

### Task 4: Add documentation and project hygiene

**Files:**
- Create: `README.md`
- Create: `.gitignore`
- Create: `requirements.txt`

- [x] **Step 1: Document environment, authorized data sources, commands, expected counts, and B3/B6 targets**

- [x] **Step 2: Ignore generated data, experiments, dependency clones, caches, and PDF render intermediates**

- [x] **Step 3: Run the complete local verification suite**

Run: `python3 -m pytest -q tests && bash -n scripts/bootstrap_wespeaker.sh && python3 -m py_compile scripts/check_data.py scripts/run_experiment.py src/redim_repro/*.py`

Expected: all tests pass and both syntax checks exit successfully.
