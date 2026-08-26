# Code and release structure

This document describes the public release tree, the role of each meaningful
file, and the assets deliberately kept out of the release. The runnable source
of truth is the code under `code/`; `code/README.md` is the unmodified historical
upstream README and is not the authority for the release environment, fixed
splits, audit semantics, or commands.

## Top-level files and directories

| Path | Purpose |
|---|---|
| `README.md` | Public entry point for environment creation, preprocessing, split validation, training, checkpoint selection, and final testing. |
| `LICENSE` | Root MIT license for the redistributed modified project. |
| `.gitignore` | Prevents datasets, weights, runs, predictions, caches, local environments, and generated binary artifacts from entering version control. |
| `environment.yml` | Reproducible Python 3.10 Conda environment; installs the non-PyTorch runtime requirements. |
| `requirements.txt` | Pinned non-PyTorch Python dependencies used by the release. |
| `requirements-torch-cu128.txt` | PyTorch 2.7.0, torchvision 0.22.0, and torchaudio 2.7.0 from the CUDA 12.8 wheel index. |
| `UPSTREAM.md` | Identifies the upstream repository, baseline commit, local-working-tree provenance, and limits of the historical source correlation. |
| `MODIFICATIONS.md` | Separates dataset/evaluation work, audit branches, environment compatibility, and release-only corrections from upstream behavior. |
| `THIRD_PARTY_NOTICES.md` | Records directly identifiable incorporated projects and retained upstream acknowledgements. |
| `RELEASE_COMPLIANCE_REPORT.md` | Release inclusion/exclusion and verification decision record. It must agree with the final hygiene and GPU-smoke results before publication. |
| `LICENSES/` | Copies of third-party license texts retained for attribution. |
| `code/` | Runnable MedSegDiff source, formal experiment configuration, preprocessing, training, sampling, and evaluation entry points. |
| `manifests/` | ID-only train/validation/test definitions. No medical images or masks are stored here. |
| `tests/` | Split, semantic-audit, loader/metric, GPU preflight, and GPU matrix checks. |
| `docs/` | Release-specific method, split, environment, verification, and structure documentation. |

## License files

| Path | Purpose |
|---|---|
| `LICENSES/MedSegDiff-MIT.txt` | Retained upstream MedSegDiff MIT text. |
| `LICENSES/OpenAI-MIT.txt` | MIT text for directly identifiable OpenAI-derived guided/improved-diffusion and logging components. |
| `LICENSES/DPM-Solver-MIT.txt` | MIT text for the incorporated DPM-Solver implementation. |

## `code/`: source and configuration

| Path | Purpose and status |
|---|---|
| `code/README.md` | Unmodified upstream project README, retained for attribution and historical context. Its old commands, dependency filename, and dataset layout are not the public release procedure. |
| `code/requirement-upstream.txt` | Historical broad upstream dependency list, renamed to distinguish it from the pinned root environment. Do not use it in place of the root requirement files. |
| `code/configs/experiments.json` | Canonical formal configuration for the RTX 5090 target, three datasets, fixed split paths, and the four conditions: Full, Random-Yt, Shuffle-Yt, and Core-No-Diff. |

### `code/scripts/`: public entry points and utilities

| File | Purpose |
|---|---|
| `build_release_manifests.py` | Converts local preprocessing CSV manifests into public ID-only split files and verifies the recorded unit/sample counts. |
| `preprocess_btcv_synapse_to_medsegv1.py` | Converts BTCV/Synapse NIfTI volumes into normalized binary 2D PNG image/mask pairs using the fixed case-level split. |
| `prepare_acdc.py` | Converts labelled ACDC ED/ES frames into RGB images, three foreground mask channels, and label maps. For each split it prefers the packaged release `*_subjects.txt` manifest and remains compatible with a custom legacy `*_patients.txt` manifest. |
| `preprocess_isic2018_to_medsegv1.py` | Converts official ISIC2018 Task 1 ZIP archives into paired training/validation/testing PNG directories. |
| `segmentation_train.py` | Main training entry point. It loads a fixed manifest, normalizes the audit mode, creates MedSegDiff V1, writes audit metadata, and invokes the training/validation loop. |
| `validation_runner.py` | Runs complete fixed-validation-set inference, computes the checkpoint-selection metric, and records validation/checkpoint status. |
| `segmentation_sample.py` | Final inference entry point for a validation-selected checkpoint; implements reverse-diffusion or direct Core-No-Diff inference, prediction export, and supported metrics. |
| `run_experiment.py` | Configuration-driven orchestration for one dataset/condition/seed: training, validation selection, and one final test. It never substitutes a test metric for checkpoint selection. |
| `evaluate_btcv_samples.py` | Standalone binary BTCV Dice/IoU evaluation for exported prediction PNGs. |
| `evaluate_isic2018_samples.py` | Standalone binary ISIC2018 Dice/IoU evaluation for exported prediction PNGs. |
| `evaluate_acdc_multiclass.py` | Standalone ACDC foreground-class Dice/IoU aggregation for exported NPZ predictions. |
| `segmentation_env.py` | Retained legacy upstream binary evaluation utility. It is not the formal BTCV/ISIC2018 evaluator used by the release runner. |
| `segmentation_env_PerClass.py` | Retained legacy upstream per-class evaluation utility. It is not the formal ACDC evaluator used by the release runner. |

All three preprocessors retain historical relative defaults for developer
convenience. Public commands must pass their raw-data, split, and output paths
explicitly; no local data path is part of the release contract.

### `code/guided_diffusion/`: model and runtime package

| File | Purpose |
|---|---|
| `__init__.py` | Package marker and upstream project description. |
| `script_util.py` | Model/diffusion defaults, V1/new architecture selection, output-channel sizing, epsilon-vs-x0 target selection, learned variance, and argument helpers. |
| `unet.py` | MedSegDiff V1 and newer U-Net variants, timestep-conditioned diffusion path, highway segmentation network, and the direct image-only `core_no_diff_logits` path. |
| `gaussian_diffusion.py` | Forward noising, reverse DDPM/DDIM sampling, calibration fusion, learned-variance terms, and the authoritative implementations of all four training semantics. |
| `train_util.py` | Optimizer/training loop, timestep sampling, checkpoint I/O, complete validation calls, and best-checkpoint retention. |
| `resample.py` | Uniform and loss-aware training-timestep samplers. The formal configuration uses uniform timesteps. |
| `respace.py` | Timestep respacing wrappers around the base Gaussian diffusion process. |
| `dpm_solver.py` | Incorporated DPM-Solver implementation used by supported accelerated validation paths; formal final testing uses the configured DDPM path. |
| `nn.py` | Shared neural-network layers, normalization, timestep embeddings, EMA, and gradient-checkpoint helpers. |
| `losses.py` | KL and discretized Gaussian likelihood primitives used by diffusion/VB calculations. |
| `fp16_util.py` | Mixed-precision/master-parameter utilities retained from upstream. Formal release experiments disable AMP/fp16. |
| `dist_util.py` | Windows distributed setup, model-state loading, device selection, and fail-fast RTX 5090/CUDA checks. |
| `logger.py` | Human, JSON, CSV, and TensorBoard-compatible training logging. Generated logs are excluded from the release. |
| `utils.py` | STAPLE fusion, Dice, normalization, initialization, and tensor/image helpers. |
| `btcvloader.py` | Paired binary BTCV PNG loader with strict ID-only manifest filtering at case level. |
| `acdcloader.py` | Four-class ACDC loader, three-channel foreground targets, strict subject-manifest filtering, and multiclass metric helpers. |
| `isicloader.py` | Official-layout binary ISIC2018 loader with strict image-ID manifest filtering. |
| `bratsloader.py` | Retained BraTS compatibility loaders, including cached 2D slices. BraTS is not part of the formal three-dataset audit matrix. |
| `custom_dataset_loader.py` | Retained generic 2D/3D compatibility loaders. They are not part of the fixed public audit matrix. |

## `manifests/`: fixed split identities

| Path | Contents |
|---|---|
| `manifests/split_spec.json` | Machine-readable role policy plus every split ID, expected unit count, and expected sample count. |
| `manifests/btcv/train_cases.txt` | The 18 BTCV training volume IDs. |
| `manifests/btcv/val_cases.txt` | The 2 BTCV validation volume IDs. |
| `manifests/btcv/test_cases.txt` | The 10 BTCV final-test volume IDs. |
| `manifests/acdc/train_subjects.txt` | The 70 ACDC training patient IDs. |
| `manifests/acdc/val_subjects.txt` | The 10 ACDC validation patient IDs. |
| `manifests/acdc/test_subjects.txt` | The 20 ACDC final-test patient IDs. |
| `manifests/isic2018/train_images.txt` | The 2,594 official ISIC2018 training image IDs. |
| `manifests/isic2018/val_images.txt` | The 100 official ISIC2018 validation image IDs. |
| `manifests/isic2018/test_images.txt` | The 1,000 official ISIC2018 final-test image IDs. |

These are ID-only files, not copies of dataset metadata with local paths. Loader
coverage checks require every requested ID to exist in the selected physical
directory and reject missing, duplicate, path-like, or ambiguous IDs.

## `tests/`: verification files

| File | Purpose |
|---|---|
| `validate_splits.py` | Checks manifest counts, overlap, filename-to-ID mapping, image/mask pairing, and optional prepared-data coverage. |
| `test_train_random_yt_audit.py` | CPU semantic unit tests for Full, Random-Yt, Shuffle-Yt, batch-size enforcement, and Core-No-Diff. The historical filename is narrower than its present coverage. |
| `test_isic2018_binary_loader_and_metrics.py` | Tests ISIC binary-mask loading, class/channel validation, foreground counts, and empty-ground-truth metric policy. |
| `gpu_preflight.py` | Fails unless CUDA, the required RTX 5090, tensor math, backward, and a formal MedSegDiff V1 forward all work. |
| `gpu_smoke_matrix.py` | Exercises the 3 datasets x 4 conditions across train, validation, checkpoint reload, and test paths on the target GPU. |

## `docs/`: release documentation

| File | Purpose |
|---|---|
| `DATA_SPLITS.md` | Human-readable fixed split identities, counts, role policy, and split-validation commands. |
| `SPLIT_USAGE_MAP.md` | Maps every dataset partition to its manifest, loader, entry point, device, and permitted role. |
| `EXPERIMENT_MATRIX.md` | Canonical dataset-by-condition protocol, including formal and smoke-only settings. |
| `REPRODUCTION.md` | End-to-end environment, data, validation-selection, final-test, and verification procedure. |
| `GPU_ENVIRONMENT.md` | Verified Windows/Conda/PyTorch/CUDA/RTX 5090 environment and rebuild procedure. |
| `GPU_SMOKE_TEST.md` | Final evidence and status for GPU preflight and the twelve-condition smoke matrix. A release is incomplete if this referenced verification record is absent. |
| `AUDIT_IMPLEMENTATION_MAP.md` | Source-and-line mapping for Full and the three audit conditions. |
| `CODE_STRUCTURE.md` | This inventory and exclusion boundary. |

## Deliberately excluded material

| Category | Examples | Reason |
|---|---|---|
| Raw and prepared medical data | NIfTI/HDF5 files, PNG images, masks, label maps, ZIP archives | Dataset redistribution is outside this source release; manifests provide identities only. |
| Model assets | `.pt`, `.pth`, `.ckpt`, pretrained parts, optimizer/EMA checkpoints | Large directly loadable research artifacts are not source code and may have separate provenance/redistribution constraints. |
| Experiment products | run directories, logs, CSV/JSON metrics, prediction PNG/NPZ files, TensorBoard and W&B data | These are generated results, can contain local paths, and are not required to inspect the implementation. |
| Paper assets | paper PDFs, copied figures, reading notes | Copyrighted/publication material is referenced rather than redistributed. |
| Repository and environment state | nested `.git/`, virtual environments, package caches, IDE state | Local machine state is neither portable nor part of the release. |
| Generated caches | `__pycache__/`, `.pyc`, `.pytest_cache/`, coverage files | Verification may recreate these locally; they must be removed or excluded from the uploaded tree. |
| Local/cloud orchestration | absolute-path PowerShell commands, RunPod/Vast queues, uploaded tarballs, host-specific launchers | They encode machine-specific paths or historical infrastructure and are replaced by parameterized public entry points. |
| Superseded evaluation material | BTCV slice-level 10% debug subset, reduced-validation runners, validation-as-test commands, dedicated sampling-step sweeps | They conflict with the fixed partition and checkpoint-selection policy. Normal reverse-sampling code remains because three formal conditions require it. |

The final publication scan should enumerate ignored files as well as tracked
files. In particular, a clean upload contains no `__pycache__` directory or
`.pyc` file even if local compilation or tests have recreated them.
