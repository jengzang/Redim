# VoxCeleb2 Server Index and Per-Epoch Evaluation Design

## Goal

Run the ReDimNet2 WeSpeaker recipe directly from the server-provided
`dev.spk2utt` and `test.spk2utt` files.  Train only on `dev`, derive a fixed,
reproducible verification set from `test`, and report EER/minDCF after every
training epoch.

## Server input contract

Each source index is a whitespace-separated two-column file:

```text
vox2_id04627 test/audio/id04627/9iu-egicBNg/00022.wav
```

The first column is the speaker label.  The second column is a path relative
to the VoxCeleb2 corpus root.  Despite its filename, each row represents one
speaker-to-utterance association; it is the authoritative source for both
audio selection and labels.  The converter must not scan the corpus tree.

## Prepared layout

`prepare-server-vox2` writes the following under a user-selected prepared-data
root without changing the original corpus:

```text
vox2_dev/{wav.scp,utt2spk,spk2utt,shard.list}
vox2_test/{wav.scp,utt2spk,spk2utt,raw.list,trials/vox2_test_fixed.kaldi}
```

An utterance id is the relative path with the split prefix (`dev/audio/` or
`test/audio/`) and `.wav` suffix removed, for example
`id04627/9iu-egicBNg/00022`.  `wav.scp` contains its absolute corpus path and
`utt2spk` preserves the source speaker id.  The converter rejects malformed
rows, duplicate utterance ids, an utterance assigned to two speakers, source
paths outside the corpus root, and missing files when `--check-files` is set.

## Fixed verification protocol

The `test.spk2utt` index is grouped by speaker and deterministically sampled
with a recorded seed.  The generated trial file contains a balanced fixed
number of target and nontarget pairs.  A target pair has two different
utterances of one speaker; a nontarget pair has one utterance from each of two
different speakers.  Both members must be present in `vox2_test/wav.scp`.

The default is 300 target pairs and 300 nontarget pairs per test speaker,
deduplicated globally.  This avoids the infeasible all-vs-all test matrix while
providing a stable, sufficiently broad development metric.  The exact seed,
requested pairs-per-speaker, realised pair counts, and source-index checksums
are written beside the trial file as JSON.  Re-running preparation with the
same input and options produces byte-identical artifacts.

`test` is used as the project validation set for model selection.  It should
not be quoted as an untouched final benchmark after choosing epochs from its
EER; an independent official evaluation set can be added later for that claim.

## Training and epoch evaluation

The current upstream `examples/voxceleb/v2/run.sh` assumes a `vox1` evaluation
directory and has no validation hook.  This project will not modify the pinned
checkout in place.  It will provide a local training entrypoint that imports
the pinned WeSpeaker components, preserves its DDP setup and optimizer/model
behaviour, and adds an explicit validation callback.

PTN configurations save every epoch.  After each epoch all DDP ranks reach a
barrier; rank 0 evaluates the in-memory model once on `vox2_test/raw.list` and
the fixed test trial file, writes a structured epoch record under the
experiment directory, then releases the remaining ranks through a second
barrier.  This avoids launching a concurrent GPU process while DDP retains
model memory.  The callback records EER, minDCF, number of embeddings, target
and nontarget trial counts, and evaluation wall time.  The default cadence is
one epoch; `--eval-interval 5` is the documented performance fallback for PTN,
while LM remains one epoch by default.

The best checkpoint is selected by lowest validation EER, with minDCF as a
deterministic tie breaker.  It is linked as `exp/.../best_model.pt` (outside
`models/`, so the upstream averaging glob only sees `model_<N>.pt`). The
existing checkpoint average and LM staging continue to work, but final
reporting distinguishes the selected validation checkpoint from a final test
result.

## CLI and compatibility

`scripts/run_experiment.py` gains a server-index preparation command, a
VoxCeleb2-test evaluation command, and training flags for the source index,
test trial path, seed, pair count, and evaluation interval.  Existing
VoxCeleb1 commands remain compatible only when explicitly requested; the
server preset uses the new VoxCeleb2 paths.

## Tests

Unit tests cover source-index parsing, canonical utterance ids, generated
Kaldi manifests, rejection cases, deterministic/balanced trial generation, and
the training/evaluation command construction.  Tests use small temporary WAV
placeholders and do not require CUDA, a mounted corpus, or a WeSpeaker clone.
