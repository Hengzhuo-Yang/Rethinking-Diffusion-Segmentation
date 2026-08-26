# Modification record

This file distinguishes pristine upstream material, locally validated audit
work, and release-only cleanup. The source project was treated as read-only;
all release edits were made in this directory.

## Existing local scientific modifications preserved

- `code/ldm/audit.py`: normalization and implementation for
  `train_random_yt` and `train_shuffle_yt`, including the batch derangement and
  audit trace.
- `code/ldm/models/diffusion/ddpm.py`: integration of the two input-side audits
  and the existing `core_no_diff` structural audit; epsilon target and original
  loss preservation for random/shuffle; direct core path and metadata.
- `code/main.py`: audit metadata, parameter counting, validation CSV records,
  resolved config saving, and safe checkpoint behavior.
- `code/ldm/data/{synapse,acdc,isic}.py`: locally validated BTCV, ACDC, and
  official ISIC2018 PNG adapters and class handling.
- `code/scripts/evaluate_*`, `promote_best_checkpoint.py`, and target-dataset
  branches of `slice2seg.py`: locally validated inference/evaluation support.
- the three target configs: locally validated batch/LR/objective/model settings.

The three audit prompts supplied with the cleanup request were used as
scientific constraints only. Their audit implementations were **not recreated
or redesigned** during release assembly.

## Release-only semantic cleanup

| File(s) | Release change | Reason |
|---|---|---|
| `code/main.py` | `val_avg_*` checkpoint namespace; separate `validation_metrics` dataset view; test-loader type check fixed; strict RTX 5090 gate; optional Lightning test loads `best` | Enforce train→validation selection→best checkpoint→test |
| `code/ldm/models/diffusion/ddpm.py` | internal `log_dice` requires `validation_metrics`; explicit `val`/`test` metric prefix | Compute selection metrics from full evaluation-form labels without changing validation loss targets |
| target configs | portable env-based asset/data paths; fixed manifests; real test partitions; `val_avg_dice` monitor | Remove machine paths and historical split misuse |
| `code/ldm/data/manifest.py` and loaders | strict ID-only manifest filter with missing-ID failure | Make fixed splits executable and auditable |
| `code/scripts/slice2seg.py` | target datasets only; explicit data/manifest/config/checkpoint; strict CUDA; test metric prefix | Remove hard-coded legacy branches and implicit paths |
| preprocessing scripts | explicit paths, fixed manifests, portable generated metadata | Recreate only the validated three data layouts |
| `code/ldm/runtime.py`, `gpu_preflight.py` | exact RTX 5090 and CUDA backward checks | No silent CPU/wrong-GPU fallback |
| `code/ldm/modules/encoders/modules.py` | unused CLIP image embedder default changed from conditional CPU fallback to CUDA | Match the release-wide CUDA-only contract |
| `code/ldm/models/autoencoder.py` | removed a local hard-coded demonstration block | Eliminate a non-entry-point absolute path |
| `code/ldm/data/__init__.py` | exports only BTCV/ACDC/ISIC2018 adapters | Avoid importing omitted legacy datasets |

No release cleanup changed the random-Y_t or shuffled-Y_t prediction target,
loss, model output semantics, auxiliary-module wiring, optimizer, scheduler,
augmentation, or inference sampler. Core-no-diff's changed objective and 8→4
input convolution are part of that already implemented structural audit.

## New release material

- root compliance/provenance/license/environment documents
- fixed manifests and split validator
- release config/data-flow/static tests
- strict GPU preflight and 12-condition train/validation/checkpoint/test GPU smoke test
- preprocessing scripts for the exact public split contract

`docs/SOURCE_PROVENANCE.csv` gives a SHA-256 and provenance classification for
every source, config, script, and manifest under `code/`.
