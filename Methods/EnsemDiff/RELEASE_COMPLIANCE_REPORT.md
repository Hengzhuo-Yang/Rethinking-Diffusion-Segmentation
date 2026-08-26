# Release Compliance Report

## Identification and decision

- Release project: EnsemDiff RTX 5090 audit release
- Primary upstream: JuliaWolleb/Diffusion-based-Segmentation
- Associated paper: *Diffusion Models for Implicit Image Segmentation Ensembles*
- Upstream baseline: `676f214035e90edd0357f51feab45841e4aefcfb`
- Preserved official README: `code/README.md` (fingerprint-guarded and unchanged)
- License decision: **A — modified-source publication is permitted subject to
  the conditions below**

No reviewed governing license imposes a non-commercial restriction. Category C
is not assigned because the direct redistribution chain is MIT/Apache-2.0 as
listed below. This is an evidence-based source review, not a guarantee of zero
legal risk.

## License coverage

| Material | Terms | Preserved evidence |
| --- | --- | --- |
| JuliaWolleb primary baseline | MIT, Copyright (c) 2023 JuliaWolleb | `LICENSE`; `LICENSES/JuliaWolleb-Diffusion-based-Segmentation-MIT.txt` |
| OpenAI improved/guided diffusion and CLIP-derived components | MIT, Copyright (c) 2021 OpenAI | `LICENSES/OpenAI-improved-guided-CLIP-MIT.txt` |
| OpenAI Baselines logger lineage | MIT, Copyright (c) 2017 OpenAI | `LICENSES/OpenAI-Baselines-MIT.txt` |
| TransUNet-derived BTCV list artifacts | Apache License 2.0 | `LICENSES/TransUNet-Apache-2.0.txt`; prominent modification notice |
| Release-local additions | Rights confirmation required | `UPSTREAM.md`, `MODIFICATIONS.md`, `THIRD_PARTY_NOTICES.md`, `docs/LICENSING_COMPLIANCE.md` |

The selected JuliaWolleb commit is MIT-licensed; an older Apache-2.0 file in
that repository's history is recorded but is not presented as the current
governing license. No separate upstream NOTICE file was found in the reviewed
snapshots. The no-license Jonathan Ho DDPM ancestor remains disclosed; the
distributed PyTorch implementation is received through OpenAI's MIT-licensed
ports.

Before publication, the project owner must retain confirmation that the
release-local loaders, validation support, preprocessing/evaluation scripts,
condition changes, manifests, configuration, documentation, and tests are
original or otherwise authorized for redistribution. This is the remaining
publication condition; unsupported copied material must be removed or licensed.

## RTX 5090 environment and execution acceptance

The public runtime is based on the existing, locally successful Conda
environment `ensemdiff`, selected from launchers and successful run evidence
before read-only execution checks. No environment was created or modified
during this audit.

| Item | Verified value |
| --- | --- |
| OS / Conda | Windows 11 Pro build 26200 / Conda 26.1.1 |
| GPU | NVIDIA GeForce RTX 5090, logical `cuda:0`, capability 12.0 |
| Driver | 591.86; driver-reported CUDA compatibility 13.1 |
| Python / PyTorch | 3.10.20 / 2.7.0+cu128 |
| PyTorch CUDA / cuDNN | 12.8 / 9.7.1 |
| Compiled architecture | `sm_120` present |
| Numeric policy | FP32; AMP disabled; matmul TF32 false; cuDNN TF32 true |
| cuDNN / deterministic | benchmark false; cuDNN deterministic false; deterministic algorithms false |
| Custom CUDA extension | None found; no project-local CUDA/C++ build is required |

Formal device setup raises if CUDA is unavailable or logical `cuda:0` is not
the exact RTX 5090. Model computation never falls back to CPU. CPU remains
valid for preprocessing, dataloader work, serialization, and metric aggregation.

Real acceptance completed on 2026-07-19:

- GPU preflight: PASS;
- 12 dataset-condition combinations: 12 PASS, 0 FAIL, 0 BLOCKED;
- every combination: correct manifests, CUDA training forward/loss/backward,
  AdamW step, CUDA validation/metric, temporary validation-best save, strict
  CUDA reload, and CUDA final-test inference;
- diffusion validation/final test: actual 100 / 1,000 DDPM steps and final
  ensemble 5; Core-No-Diff: one direct forward;
- temporary generated checkpoints, metrics, logs, data views, and report:
  cleaned and excluded from the release.

See `docs/GPU_ENVIRONMENT.md` and `docs/GPU_SMOKE_TEST.md` for exact evidence.
The smoke result validates execution, not full convergence or paper metrics.

## Conditions and formal configuration

All three datasets support Full, Random-Yt, Shuffle-Yt, and Core-No-Diff. Full
is the reproduction baseline, not a fourth audit. Random-Yt replaces only the
training model's Yt input with independent Gaussian noise. Shuffle-Yt replaces
that input with a batch-deranged-mask Yt while preserving the current timestep,
target, noise, and diffusion loss. Core-No-Diff is an image-only direct
segmentation counterfactual and intentionally removes timestep/Yt/reverse
diffusion from its main path.

The non-batch flags explicitly published by the upstream README match the Full
baseline. Local formal additions not specified there are dataset channels and
targets, 224×224 prepared arrays with model `image_size=256`, seed 10,
40,000/60,000 step budgets, validation intervals, and validation/final sampling
policies. The formal training batch remains the locally retained value 8;
acceptance used batch 1 except Shuffle-Yt batch 2. No audit target, loss,
timestep, sampling policy, AMP, dtype, or TF32 behavior was altered for smoke.

## Split and checkpoint compliance

The only allowed flow is training → validation selection → explicit best-model
reload → one final test. All conditions and seeds share the same manifests.

| Dataset | Training | Validation-only selection | Final test |
| --- | --- | --- | --- |
| BTCV | 18 cases / 2,211 slices | `case0008`, `case0001` / 295 slices | 10 cases / 1,273 slices |
| ACDC | 70 subjects / 1,304 slices | 10 subjects / 182 slices | 20 subjects / 416 slices |
| ISIC2018 | 2,594 images | 100 images | 1,000 images |

The release removes the BTCV ten-percent/quick final-test entry path, does not
use BTCV test data for checkpoint choice, and does not relabel ACDC or ISIC2018
validation as final test. `best_checkpoint.json` must identify validation,
the fixed validation manifest, seeds, condition, and `best_model.pt` before the
non-overwriting final-test gate runs. The dedicated sampling-step comparison
experiment is excluded while the fixed sampling infrastructure required by the
four retained conditions remains.

## Publication contents and exclusions

The release publishes source, fixed ID-only manifests, preprocessing,
environment contracts, provenance/license records, documentation, and tests.
It excludes all medical images/masks, NumPy/NIfTI/HDF5 payloads, archives,
weights/checkpoints, private metrics/logs, paper PDFs, cloud bundles, environment
directories, caches, secrets, and local paths. The selected upstream commit
does not advertise an applicable official checkpoint for this 12-run matrix;
no link or weight artifact is invented.

## Readiness and residual risk

Technically, the source release is ready for the documented Windows/Conda/RTX
5090 target: manifests, static checks, preflight, and all 12 CUDA smoke paths
passed. Public GitHub publication is suitable under category A only after the
release-local rights confirmation above is recorded.

Residual risks are the disclosed DDPM ancestor licensing uncertainty, the
unfixed inherited CLIP source-commit reference, external dependency and dataset
terms, and the lack of a full historical environment freeze for earlier private
runs. Other GPUs, CPU-only execution, other operating systems, full training
convergence, and bitwise determinism are outside the verified scope.
