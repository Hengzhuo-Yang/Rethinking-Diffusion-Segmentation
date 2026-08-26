# RTX 5090 GPU Smoke-Test Record

## Result

The mandatory 3-dataset × 4-condition acceptance run completed on 2026-07-19:

- Overall: **PASS**
- Combinations: 12 PASS, 0 FAIL, 0 BLOCKED
- Total wall time: 1,130.191 seconds
- Device: NVIDIA GeForce RTX 5090, logical `cuda:0`
- Capability / compiled architecture: 12.0 / `sm_120`
- Runtime: Python 3.10.20, PyTorch 2.7.0+cu128, CUDA runtime 12.8,
  cuDNN 9.7.1, driver 591.86
- Formal dtype: FP32; AMP disabled; matmul TF32 false; cuDNN TF32 true

These are execution checks, not trained-model results or benchmark scores. The
one-sample metrics came from temporary, minimally updated models and are not
reported as scientific performance.

## Command

From the repository root in the activated `ensemdiff` environment:

    python tests/gpu_smoke_12.py --data-root btcv=prepared/btcv --data-root acdc=prepared/acdc --data-root isic2018=prepared/isic2018

The program also runs `tests/gpu_preflight.py`. Preflight passed CUDA
arithmetic/backward and a formal 224×224 UNet forward; its peak allocated CUDA
memory was 712,776,704 bytes.

## Coverage per combination

Every row performed and asserted all of the following:

1. exact, disjoint release manifests and expected local partition totals;
2. a manifest-locked training batch;
3. formal UNet parameters, inputs, outputs, and loss on `cuda:0` in FP32;
4. forward, finite formal loss, backward, and an AdamW parameter update;
5. the requested Full/audit training branch while the model was in training mode;
6. a manifest-locked validation sample and finite Dice/IoU metric;
7. a temporary validation-selected `best_model.pt` plus metadata matching
   `TrainLoop._save_best_checkpoint`;
8. strict state-dict reload onto `cuda:0`;
9. a manifest-locked test sample using only the reloaded checkpoint; and
10. unchanged AMP and TF32 state, nonzero CUDA memory, and no CPU fallback.

Diffusion rows used the formal 100-step DDPM validation policy with ensemble 1
and the formal 1,000-step DDPM final-test policy with ensemble 5. Forward hooks
observed 100 validation model calls and 5,000 final-test model calls, all with
CUDA inputs and outputs. Core-No-Diff used one direct validation and one direct
test forward, with no reverse process.

| Dataset | Condition | Smoke batch | Branch executed | Train F/B/step | Val calls | Best save/reload | Test calls | Peak CUDA allocated (bytes) | Time (s) | Status |
| --- | --- | ---: | --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| BTCV | Full | 1 | reference q-sampled Yt | yes/yes/yes | 100 | yes/strict | 5,000 | 2,320,711,168 | 121.832 | PASS |
| BTCV | Random-Yt | 1 | independent Gaussian Yt input | yes/yes/yes | 100 | yes/strict | 5,000 | 2,320,383,488 | 121.829 | PASS |
| BTCV | Shuffle-Yt | 2 | batch-deranged mask Yt input | yes/yes/yes | 100 | yes/strict | 5,000 | 3,866,618,368 | 127.006 | PASS |
| BTCV | Core-No-Diff | 1 | direct image-only segmentation | yes/yes/yes | 1 | yes/strict | 1 | 2,211,350,528 | 2.294 | PASS |
| ACDC | Full | 1 | reference q-sampled Yt | yes/yes/yes | 100 | yes/strict | 5,000 | 2,325,615,616 | 124.778 | PASS |
| ACDC | Random-Yt | 1 | independent Gaussian Yt input | yes/yes/yes | 100 | yes/strict | 5,000 | 2,325,615,616 | 121.970 | PASS |
| ACDC | Shuffle-Yt | 2 | batch-deranged mask Yt input | yes/yes/yes | 100 | yes/strict | 5,000 | 3,873,506,304 | 125.770 | PASS |
| ACDC | Core-No-Diff | 1 | direct image-only segmentation | yes/yes/yes | 1 | yes/strict | 1 | 2,214,191,104 | 1.954 | PASS |
| ISIC2018 | Full | 1 | reference q-sampled Yt | yes/yes/yes | 100 | yes/strict | 5,000 | 2,321,232,384 | 128.328 | PASS |
| ISIC2018 | Random-Yt | 1 | independent Gaussian Yt input | yes/yes/yes | 100 | yes/strict | 5,000 | 2,321,232,384 | 124.746 | PASS |
| ISIC2018 | Shuffle-Yt | 2 | batch-deranged mask Yt input | yes/yes/yes | 100 | yes/strict | 5,000 | 3,869,437,440 | 123.080 | PASS |
| ISIC2018 | Core-No-Diff | 1 | direct image-only segmentation | yes/yes/yes | 1 | yes/strict | 1 | 2,213,379,072 | 2.084 | PASS |

“F/B/step” means forward, backward, and optimizer step. Shuffle-Yt alone uses
batch 2 because its within-batch derangement requires more than one sample.
The formal experiment batch remains 8 for every row.

## Partition evidence

All four conditions used the same fixed manifests. Loader-observed totals
matched exactly:

| Dataset | Training | Validation selection | Final test |
| --- | ---: | ---: | ---: |
| BTCV | 2,211 slices | 295 slices | 1,273 slices |
| ACDC | 1,304 slices | 182 slices | 416 slices |
| ISIC2018 | 2,594 images | 100 images | 1,000 images |

BTCV validation used only `case0008` and `case0001`; its ten final-test cases
were disjoint. A historical physical held-out pool was exposed through a
temporary read-only view during local acceptance, while the validation and test
loaders were independently filtered by the fixed manifests. The public BTCV
preprocessor writes clean `validation/` and `testing/` directories directly.

## Cleanup and remaining scope

Per-combination checkpoints, metadata, CSV files, and logger directories were
created only inside managed temporary directories and removed automatically.
The temporary acceptance report and read-only BTCV view used to prepare this
record were removed after verification; neither data nor generated weights are
part of the release.

No unresolved CUDA, data, or implementation failure remained in this smoke
scope. The run validates executable paths, not full 40,000/60,000-step training,
convergence, paper-level metrics, other GPUs, CPU-only execution, or another
operating system.
