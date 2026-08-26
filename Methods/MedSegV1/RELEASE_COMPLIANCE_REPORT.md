# Release compliance report

## Decision

**Category A — redistribution of a modified version is allowed subject to the
MIT attribution and notice conditions.** The selected upstream baseline contains
the MIT license, and the identified directly incorporated OpenAI and DPM-Solver
components are also MIT-licensed. The root license, upstream README, license
copies and third-party notices are retained.

This is an evidence-based release assessment, not a guarantee of zero legal
risk. Residual provenance uncertainty is limited to the upstream repository's
aggregate acknowledgement of several projects without a file-by-file donor map;
see `THIRD_PARTY_NOTICES.md`.

## Included

- The current local MedSegDiff V1 working-tree source required for BTCV, ACDC and
  ISIC2018.
- Full plus the existing Random-Yt, Shuffle-Yt and Core-No-Diff conditions.
- Preprocessing and evaluation source, fixed ID-only manifests, environment
  specifications, tests and documentation.

## Excluded

- All raw and preprocessed medical data and masks.
- All checkpoints, pretrained weights and directly loadable model assets.
- Experiment runs, logs, prediction images, metrics, TensorBoard/W&B output and
  temporary smoke artifacts.
- The paper PDF and local copies of paper figures.
- The nested `.git` directory, caches, bytecode, virtual environments and local
  package caches.
- RunPod/Vast queue material, cloud paths and historical reduced-test runners.
- BTCV's former 10% test subset, ACDC/ISIC validation-as-final-test commands and
  dedicated sampling-step comparison material.

## Verification summary

| Check | Status | Evidence |
|---|---:|---|
| Upstream revision and license identified | PASS | `UPSTREAM.md`, `LICENSE`, `LICENSES/` |
| Official README retained unchanged | PASS | `code/README.md` and release hash check |
| Fixed split IDs/counts/leakage | PASS | `manifests/`, `tests/validate_splits.py` |
| CUDA tensor, matrix multiply and backward on RTX 5090 | PASS | `tests/gpu_preflight.py`, `docs/GPU_SMOKE_TEST.md` |
| Formal MedSegDiff V1 model forward on RTX 5090 | PASS | `tests/gpu_preflight.py`, `docs/GPU_SMOKE_TEST.md` |
| Twelve train/validation/checkpoint/test smoke paths | PASS | 12/12 PASS recorded in `docs/GPU_SMOKE_TEST.md` |
| Data/weight/log/cache exclusion | PASS | Final no-ignore release-tree scan found none |
| Absolute-path and secret scan | PASS | Final release-tree scans returned zero matches |

All release gates recorded in this report are satisfied, so this tree is ready
for upload or publication; no upload or publication was performed as part of
this release preparation. Any future failed hygiene or smoke check must be
recorded and prevents a claim that the corresponding release check passed.
