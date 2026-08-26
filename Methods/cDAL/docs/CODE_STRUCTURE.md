# Code structure and provenance

Provenance categories follow the release audit vocabulary. “RTX” marks files containing or directly enforcing RTX 5090/CUDA compatibility behavior. All included locally added files that couple to cDAL are conservatively distributed under the root NVIDIA non-commercial terms.

## Release, training, evaluation, and configuration

| Relative path | Function | Provenance / upstream relation | RTX | License | Dataset / condition | Train-val-test role | Common | Included |
|---|---|---|---|---|---|---|---|---|
| `code/README.md` | Preserved official documentation | Upstream unchanged; byte-identical to selected worktree | no | NVIDIA NC / upstream notices | upstream reference | archival only | yes | yes |
| `code/LICENSE.txt` | Preserved upstream root license | Upstream unchanged | no | NVIDIA NC | all | distribution | yes | yes |
| `code/train_cDal_monu_and_lung.py` | Shared training, audit branches, validation, best checkpoint | Upstream modified locally for audit/datasets; release-only split/device/path hardening | yes | NVIDIA NC | all 12 | train + validation selection | yes | yes |
| `code/audit_modes.py` | Random-Yt and Shuffle-Yt construction and mode validation | User-added, tightly coupled | no | NVIDIA NC | three audits | training branch | yes | yes |
| `code/metrics.py` | Binary/multiclass Dice/IoU and sampling evaluation | Upstream modified for target datasets, batching, audit paths, and release privacy | yes | NVIDIA NC | all | validation + test | yes | yes |
| `code/evaluate_btcv_cdal.py` | BTCV final test | User-added; release-hardened | yes | NVIDIA NC | BTCV/all | test only | no | yes |
| `code/evaluate_acdc_cdal.py` | ACDC final test | User-added; release-hardened | yes | NVIDIA NC | ACDC/all | test only | no | yes |
| `code/evaluate_isic2018_cdal.py` | ISIC2018 final test | User-added; release-hardened | yes | NVIDIA NC | ISIC2018/all | test only | no | yes |
| `code/release_validation.py` | Manifest hashing and validation-selected checkpoint enforcement | Generated release support | no | NVIDIA NC | all | test gate | yes | yes |
| `code/scripts/run_condition.py` | Formal train -> validation best -> test orchestration | Generated release support | yes | NVIDIA NC | all 12 | all three stages | yes | yes |
| `code/parameters_btcv.json` | BTCV formal configuration | User-added; split/path release correction | yes | NVIDIA NC | BTCV/all | train + validation defaults | no | yes |
| `code/parameters_acdc.json` | ACDC formal configuration | User-added; split/path release correction | yes | NVIDIA NC | ACDC/all | train + validation defaults | no | yes |
| `code/parameters_isic2018.json` | ISIC2018 formal configuration | User-added; split/path release correction | yes | NVIDIA NC | ISIC2018/all | train + validation defaults | no | yes |
| `code/utils.py` | Seeds, model unwrap, distributed helpers, required device | Upstream modified locally and release-hardened | yes | NVIDIA NC | all | all stages | yes | yes |
| `code/EMA.py` | Optional optimizer EMA wrapper; disabled by formal configs | Upstream unchanged; DDGAN lineage file names NVlabs/LSGM as adaptation source | no | NVIDIA NC + NVIDIA LSGM terms | all | training infrastructure | yes | yes |
| `code/logger.py` | Tabular/text logging | Upstream unchanged; portions attributed to OpenAI Baselines | no | NVIDIA NC + MIT notice | all | runtime logging | yes | yes |

## Dataset loaders and preprocessing

| Relative path | Function | Provenance / upstream relation | RTX | License | Dataset / condition | Role | Common | Included |
|---|---|---|---|---|---|---|---|---|
| `code/preprocess_dataset/dataset.py` | Loader factory | Upstream modified locally; manifest support added for release | no | NVIDIA NC | all | all partitions | yes | yes |
| `code/preprocess_dataset/manifest.py` | Strict ID-manifest parser | Generated release support | no | NVIDIA NC | all | all partitions | yes | yes |
| `code/preprocess_dataset/BTCV.py` | Binary BTCV loader and held-out-pool role filtering | User-added; release split fix | no | NVIDIA NC | BTCV/all | train/val/test | no | yes |
| `code/preprocess_dataset/ACDC.py` | Three-channel ACDC loader | User-added; manifest and semantic-role hardening | no | NVIDIA NC | ACDC/all | train/val/test | no | yes |
| `code/preprocess_dataset/ISIC2018.py` | Binary ISIC2018 loader | User-added; manifest and semantic-role hardening | no | NVIDIA NC | ISIC2018/all | train/val/test | no | yes |
| `code/preprocess_dataset/transforms.py` | Paired image/mask resize and augmentation | Upstream modified locally for multi-channel masks | no | NVIDIA NC | all | train augmentation; val/test deterministic transform | yes | yes |
| `code/scripts/preprocess_btcv_synapse_to_cdal.py` | NIfTI -> fixed binary 2D cache | User-added local preprocessing, corrected to fixed manifests and no 10% subset | no | NVIDIA NC | BTCV | preprocessing | no | yes |
| `code/scripts/preprocess_acdc_to_cdal.py` | NIfTI -> three-channel 2D cache | User-added; path and fixed-manifest cleanup | no | NVIDIA NC | ACDC | preprocessing | no | yes |
| `code/scripts/preprocess_isic2018_to_cdal.py` | Official zip -> binary 2D cache | User-added; path cleanup | no | NVIDIA NC | ISIC2018 | preprocessing | no | yes |

## Model and diffusion infrastructure

| Relative path | Function | Provenance / upstream relation | RTX | License | Dataset / condition | Role/common | Included |
|---|---|---|---|---|---|---|---|
| `code/score_sde/distribution.py` | Distribution helper | Upstream unchanged except removal of an unused global CPU-fallback device | yes | NVIDIA NC / Apache-derived | all | common model | yes |
| `code/score_sde/models/ncsnpp_generator_adagn.py` | cDAL generator and Core-No-Diff path | Upstream modified for audit | no | NVIDIA NC + Apache notice | all; Core-No-Diff | common model | yes |
| `code/score_sde/models/discriminator.py` | Time-dependent discriminator and feature attention | Upstream unchanged | no | NVIDIA NC + Apache-derived | Full/Random/Shuffle | common model | yes |
| `code/score_sde/models/dense_layer.py` | Dense/equalized layers | Third-party derived, upstream unchanged | no | MIT (CW Huang) | all | common model | yes |
| `code/score_sde/models/layers.py` | Score-SDE layers | Upstream unchanged | no | Apache-2.0 / NVIDIA NC combination | all | common model | yes |
| `code/score_sde/models/layerspp.py` | NCSN++ residual/attention blocks | Upstream unchanged | no | Apache-2.0 / NVIDIA NC combination | all | common model | yes |
| `code/score_sde/models/nn.py` | Normalization and convolution helpers | Third-party derived, upstream unchanged | no | MIT MedSegDiff + NVIDIA NC combination | all | common model | yes |
| `code/score_sde/models/unet.py` | Retained upstream U-Net utilities | Upstream unchanged | no | upstream notices / NVIDIA NC | shared support | common model | yes |
| `code/score_sde/models/up_or_down_sampling.py` | FIR up/downsampling wrappers | Upstream unchanged | no | NVIDIA NC + MIT op notice | all | common model | yes |
| `code/score_sde/models/utils.py` | Model registry and timestep embedding | Upstream unchanged | no | Apache-derived / NVIDIA NC | all | common model | yes |
| `code/score_sde/models/__init__.py`, `code/score_sde/__init__.py` | Packages | Upstream unchanged | no | applicable directory licenses | all | common | yes |
| `code/score_sde/op/fused_act.py` | Fused activation with CUDA-resident PyTorch fallback | Upstream modified locally for Windows/modern CUDA | yes | NVIDIA NC + MIT StyleGAN2 | all | common model | yes |
| `code/score_sde/op/upfirdn2d.py` | Upfirdn with CUDA-resident PyTorch fallback | Upstream modified locally for Windows/modern CUDA | yes | NVIDIA NC + MIT StyleGAN2 | all | common model | yes |
| `code/score_sde/op/fused_bias_act.cpp`, `fused_bias_act_kernel.cu` | Optional fused extension sources | Upstream unchanged | yes | NVIDIA NC + MIT notice | all | optional common op | yes |
| `code/score_sde/op/upfirdn2d.cpp`, `upfirdn2d_kernel.cu` | Optional upfirdn extension sources | Upstream unchanged | yes | NVIDIA NC + MIT notice | all | optional common op | yes |
| `code/score_sde/op/__init__.py` | Operation exports | Upstream unchanged | no | directory licenses | all | common | yes |
| `code/score_sde/models/LICENSE_MIT`, `code/score_sde/op/LICENSE_MIT` | Original third-party notices | Upstream unchanged | no | MIT | applicable files | distribution | yes |

## Manifests, validation, environment, and documentation

| Relative path(s) | Function | Provenance | License | Dataset/role | Included |
|---|---|---|---|---|---|
| `manifests/btcv/train_cases.txt`, `validation_cases.txt`, `test_cases.txt` | Fixed 18/2/10 case IDs | Generated from recovered validated IDs and required fixed held-out division | NVIDIA NC release terms; IDs only | BTCV all roles | yes |
| `manifests/acdc/training_subjects.txt`, `validation_subjects.txt`, `test_subjects.txt` | Fixed 70/10/20 subject IDs | Recovered from completed local manifests | NVIDIA NC release terms; IDs only | ACDC all roles | yes |
| `manifests/isic2018/training_images.txt`, `validation_images.txt`, `test_images.txt` | Official 2594/100/1000 image IDs | Copied from validated local official-split lists | NVIDIA NC release terms; IDs only | ISIC2018 all roles | yes |
| `tests/gpu_preflight.py` | Exact RTX 5090 and project-model CUDA test | Generated release test | NVIDIA NC | all/device | yes |
| `tests/gpu_smoke_matrix.py` | 12-combination end-to-end CUDA test | Generated release test | NVIDIA NC | all roles/conditions | yes |
| `tests/validate_splits.py` | ID/count/pair/leakage validator | Generated release test | NVIDIA NC | all splits | yes |
| `environment.yml`, `requirements.txt` | Minimal reconstruction of selected successful environment | Generated from imports and installed versions | configuration | all | yes |
| `README.md`, `UPSTREAM.md`, `MODIFICATIONS.md`, `RELEASE_COMPLIANCE_REPORT.md`, `THIRD_PARTY_NOTICES.md`, `docs/*.md` | Public reproduction, provenance, split, parameter, and GPU evidence | Generated release documentation | documentation under root terms | all | yes |
| `LICENSE`, `LICENSES/*` | Root and third-party license texts | Upstream-preserved or fetched from official source repositories | named text | distribution | yes |

## Excluded source-project material

| Category | Examples | Reason |
|---|---|---|
| Weight/data/asset | `saved_models/`, `saved_info/`, `.pth`, raw/preprocessed images and masks | Redistribution and privacy risk; not source |
| Experiment output | logs, CSV/JSON results, predictions, TensorBoard, temporary smoke outputs | Generated results; may contain paths or medical-derived images |
| Cloud package | Vast/RunPod bundles, archives, cloud launchers | Duplicate code, absolute cloud paths, data/results/weights |
| Obsolete split mechanism | BTCV 10% test runner and test-subset data | Used test slices for checkpoint selection; prohibited |
| Legacy quick final test | ACDC/ISIC wrappers using validation as final test | Violates task-role separation |
| Sampling-step comparison | dedicated step-count sweeps, launchers, results, notes | Outside the three target audits |
| Original-dataset execution | MoNuSeg, CXR, Hippocampus loaders/trainers/samplers | Not required for the released three-dataset matrix; available upstream |
| Publication material | paper PDF, figures, supplements | No redistribution basis established and not needed to run code |
| Local state | `.git`, IDE files, caches, environment folders, absolute-path notes | Not release source |

No required released file has unknown provenance. This classification does not convert visibility or local authorship into broader licensing rights; the root non-commercial restriction remains.
