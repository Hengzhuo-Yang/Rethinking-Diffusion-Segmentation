# RTX 5090 GPU verification

## Result

Verification completed on **2026-07-19** with an NVIDIA GeForce RTX 5090 at
`cuda:0`. The fail-fast preflight passed, and all twelve dataset/condition
paths passed the GPU smoke matrix.

After the paper-aligned V1 condition-only/sample-only routing change on
**2026-07-20**, targeted checks passed in both source trees: Python compilation,
encoder-gradient/localization-no-gradient regression, formal 256 x 256 GPU
forward with zero localization-decoder calls, and a one-step normal V1 sample
with `cal=None` and `calout=None`. The complete twelve-path data-dependent
matrix below records the 2026-07-19 run and has not yet been rerun after this
routing-only change.

| Verification | Result |
|---|---:|
| CUDA tensor forward and backward | PASS |
| Formal 256 x 256 MedSegDiff V1 forward | PASS |
| BTCV: Full, Random-Yt, Shuffle-Yt, Core-No-Diff | 4/4 PASS |
| ACDC: Full, Random-Yt, Shuffle-Yt, Core-No-Diff | 4/4 PASS |
| ISIC2018: Full, Random-Yt, Shuffle-Yt, Core-No-Diff | 4/4 PASS |
| Complete matrix | **12/12 PASS** |

No scientific performance values are reported here. The smoke inputs and
truncated sampling schedule are designed to verify executable routing, not to
estimate segmentation quality.

## Preflight evidence

The command

```powershell
python tests\gpu_preflight.py
```

verified all of the following in the pinned environment:

- CUDA was available and `cuda:0` was exactly `NVIDIA GeForce RTX 5090`.
- A CUDA matrix multiplication and its backward pass completed with the
  gradient on the GPU.
- A formal-size binary MedSegDiff V1 model was created with 130,527,814
  parameters.
- The model parameters, 256 x 256 input, timestep tensor, and model outputs
  remained on `cuda:0`.
- Peak allocated CUDA memory during the preflight was 861,782,016 bytes.
- No silent CPU fallback occurred.

The verified software and device details are recorded in
`docs/GPU_ENVIRONMENT.md`.

## Twelve-path matrix

Each row below passed the same required stages: production training loss,
forward pass, finite loss, backward pass, finite active gradients, one AdamW
optimizer step, fixed-validation-manifest inference, a finite validation metric,
temporary checkpoint save, checkpoint reload into a new model, and
fixed-test-manifest inference. The temporary checkpoint was removed after the
test path.

| Dataset | Condition | Train / backward / optimizer | Validation | Temporary checkpoint reload | Test | Result |
|---|---|---:|---:|---:|---:|---:|
| BTCV | Full | PASS | PASS | PASS | PASS | PASS |
| BTCV | Random-Yt | PASS | PASS | PASS | PASS | PASS |
| BTCV | Shuffle-Yt | PASS | PASS | PASS | PASS | PASS |
| BTCV | Core-No-Diff | PASS | PASS | PASS | PASS | PASS |
| ACDC | Full | PASS | PASS | PASS | PASS | PASS |
| ACDC | Random-Yt | PASS | PASS | PASS | PASS | PASS |
| ACDC | Shuffle-Yt | PASS | PASS | PASS | PASS | PASS |
| ACDC | Core-No-Diff | PASS | PASS | PASS | PASS | PASS |
| ISIC2018 | Full | PASS | PASS | PASS | PASS | PASS |
| ISIC2018 | Random-Yt | PASS | PASS | PASS | PASS | PASS |
| ISIC2018 | Shuffle-Yt | PASS | PASS | PASS | PASS | PASS |
| ISIC2018 | Core-No-Diff | PASS | PASS | PASS | PASS | PASS |

Across the twelve rows, peak allocated CUDA memory ranged from 1,342,956,544
to 5,200,393,216 bytes. Every checked model parameter, input, and loss tensor
was on `cuda:0`.

## Smoke scope versus formal experiments

The matrix deliberately uses one optimizer update, one validation sample, and
one test sample. Batch size is one except for Shuffle-Yt, which uses two samples
from distinct units so its no-self-match permutation is exercised.

For Full, Random-Yt, and Shuffle-Yt, smoke validation and test use **one reverse
step**. Their formal configuration remains **1,000 reverse steps with ensemble
size 5 for final testing**; validation uses 20-step DPM-Solver++ with ensemble
size 1 for all three datasets. Core-No-Diff uses **zero reverse
steps** in both smoke and formal execution because it performs a single direct
image-only forward.

The smoke runner reads the formal configuration but does not modify it. A smoke
PASS therefore establishes that all required CUDA routes execute; it is not a
substitute for completing the twelve formal training runs.

## Re-run command

After data preparation and split validation, the matrix can be repeated with:

```powershell
python tests\gpu_smoke_matrix.py `
  --btcv-root <processed-btcv-root> `
  --acdc-root <processed-acdc-root> `
  --isic2018-root <processed-isic2018-root> `
  --smoke-sampling-steps 1 `
  --report <temporary-smoke-report.json>
```

The JSON report is a generated verification artifact and is intentionally not
part of the source release. A rerun is valid only if all twelve rows report
`PASS`; any `FAIL` must be retained and investigated rather than summarized as
a successful matrix.
