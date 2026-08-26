# cDAL controlled-audit release

This repository is a reproducible, non-commercial research release of **Conditional Diffusion Model with Spatial Attention and Latent Embedding for Medical Image Segmentation (cDAL)**, by Behzad Hejrati, Soumyanil Banerjee, Carri Glide-Hurst, and Ming Dong (MICCAI 2024, DOI: `10.1007/978-3-031-72114-4_20`). The official implementation is [Hejrati/cDAL](https://github.com/Hejrati/cDAL). This release is based on upstream commit `ade823bd0a92571dba72f90fb7e9043c3e66b6a9` plus the locally validated dataset and audit extensions described in [UPSTREAM.md](UPSTREAM.md) and [MODIFICATIONS.md](MODIFICATIONS.md).

The release studies whether cDAL's performance depends on the intended diffusion-state mechanism. It is a controlled, falsification-oriented mechanism audit, not a new segmentation method. **Full** is the unablated cDAL baseline; the three audits are **Random-Yt**, **Shuffle-Yt**, and **Core-No-Diff**.

## Scope and status

- Datasets: BTCV/Synapse, ACDC, and ISIC2018 Task 1.
- Conditions: Full plus the three audits, for 12 combinations.
- Data flow: `train -> validation -> validation-selected best checkpoint -> test`.
- Validated hardware: one NVIDIA GeForce RTX 5090 at `cuda:0`.
- Validated source environment: Conda environment `medsegv1`, Python 3.10.20, PyTorch 2.7.0+cu128, CUDA runtime 12.8, cuDNN 9.7.1, driver 591.86.
- GPU validation: preflight PASS and all 12 end-to-end smoke combinations PASS. See [docs/GPU_SMOKE_TEST.md](docs/GPU_SMOKE_TEST.md).
- CPU-only execution, other GPUs, and the paper's legacy environment are outside this release's validation scope. Formal entries fail instead of falling back to CPU.
- No dataset, medical image, checkpoint, pretrained weight, experiment log, paper PDF, or paper figure is distributed.

## Audit conditions

| Public name | Configuration value | Controlled intervention |
|---|---|---|
| Full | `none` | Unmodified local cDAL baseline. |
| Random-Yt | `train_random_yt` | Replaces only the training generator's noisy-mask input with `torch.randn_like`; clean target, image condition, timestep, MSE objective, validation, and inference remain Full. |
| Shuffle-Yt | `train_shuffle_yt` | Builds only the training generator's noisy-mask input from a deranged batch peer's clean mask using the current sample's timestep and diffusion noise; target and inference remain Full. |
| Core-No-Diff | `core_no_diff` | Uses the condition image and non-diffusion latent `z`; removes `Y_t`, timestep conditioning, forward diffusion, discriminator attention, and reverse diffusion from the main path while retaining clean-mask MSE. |

Implementation-level details and final source line numbers are in [docs/AUDIT_IMPLEMENTATION_MAP.md](docs/AUDIT_IMPLEMENTATION_MAP.md).

## License restriction

This is a **B-class release**: redistribution and modification are permitted only under the applicable NVIDIA Source Code License-NC terms for non-commercial research or evaluation. It is not an OSI-approved open-source license and does not authorize commercial use. The LSGM-derived EMA portion also carries the applicable NVIDIA-processor restriction documented in its included license text; the validated RTX 5090 target satisfies that hardware condition. The root [LICENSE](LICENSE), all notices, and applicable third-party license texts must remain with redistributed copies. See [RELEASE_COMPLIANCE_REPORT.md](RELEASE_COMPLIANCE_REPORT.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). This is an evidence-based compliance assessment, not a guarantee of zero legal risk.

## Environment

Create the cleaned environment:

```powershell
conda env create -f environment.yml
conda activate cdal-rtx5090
```

The CUDA PyTorch installation used for validation is also expressible directly through the official PyTorch wheel channel:

```powershell
python -m pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

Run the mandatory preflight before preprocessing or training:

```powershell
python tests/gpu_preflight.py
```

The command must report `NVIDIA GeForce RTX 5090`; it exits with an error for unavailable CUDA, a different GPU, or model execution outside CUDA. Full environment evidence is in [docs/GPU_ENVIRONMENT.md](docs/GPU_ENVIRONMENT.md).

## Data preparation and fixed splits

Obtain each dataset from its official provider and accept its terms. Place raw files below a local `raw/` directory or pass another configurable path. Preprocess from the repository root:

```powershell
python code/scripts/preprocess_btcv_synapse_to_cdal.py --raw_root raw/BTCV --output_root data_preprocessed/btcv_synapse_cdal_binary_png
python code/scripts/preprocess_acdc_to_cdal.py --raw_root raw/ACDC/database --split_root manifests/acdc --output_root data_preprocessed/acdc_mt_unet_cascade_cdal_png
python code/scripts/preprocess_isic2018_to_cdal.py --raw_root raw/ISIC2018 --output_root data_preprocessed/isic2018_task1_cdal_png
```

Validate all fixed IDs, counts, mask pairs, and leakage constraints:

```powershell
python tests/validate_splits.py --btcv-root data_preprocessed/btcv_synapse_cdal_binary_png --acdc-root data_preprocessed/acdc_mt_unet_cascade_cdal_png --isic-root data_preprocessed/isic2018_task1_cdal_png
```

| Dataset | Training | Validation, checkpoint selection only | Test, final inference only |
|---|---:|---:|---:|
| BTCV/Synapse | 18 cases, 2,211 slices | `case0008`, `case0001`; 295 slices | 10 cases; 1,273 slices |
| ACDC | 70 subjects, 1,304 slices | 10 subjects, 182 slices | 20 subjects, 416 slices |
| ISIC2018 | 2,594 images | 100 images | 1,000 images |

BTCV validation and test are disjoint case-filtered views of the standard 12-case held-out pool. The historical slice-level `test_10pct_seed23` mechanism is not included or referenced. Details are in [docs/DATA_SPLITS.md](docs/DATA_SPLITS.md).

## Formal 12-combination commands

Run commands from the repository root in the RTX 5090 environment. Each command performs formal training with batch size 4 and 10,000 optimizer steps, evaluates the fixed validation manifest at the configured interval, saves `best_checkpoint.pt` by mean validation Dice, reloads that checkpoint, and performs final inference on the fixed test manifest.

```powershell
python code/scripts/run_condition.py --dataset btcv --condition full --data-root data_preprocessed/btcv_synapse_cdal_binary_png --device 0
python code/scripts/run_condition.py --dataset btcv --condition random-yt --data-root data_preprocessed/btcv_synapse_cdal_binary_png --device 0
python code/scripts/run_condition.py --dataset btcv --condition shuffle-yt --data-root data_preprocessed/btcv_synapse_cdal_binary_png --device 0
python code/scripts/run_condition.py --dataset btcv --condition core-no-diff --data-root data_preprocessed/btcv_synapse_cdal_binary_png --device 0

python code/scripts/run_condition.py --dataset acdc --condition full --data-root data_preprocessed/acdc_mt_unet_cascade_cdal_png --device 0
python code/scripts/run_condition.py --dataset acdc --condition random-yt --data-root data_preprocessed/acdc_mt_unet_cascade_cdal_png --device 0
python code/scripts/run_condition.py --dataset acdc --condition shuffle-yt --data-root data_preprocessed/acdc_mt_unet_cascade_cdal_png --device 0
python code/scripts/run_condition.py --dataset acdc --condition core-no-diff --data-root data_preprocessed/acdc_mt_unet_cascade_cdal_png --device 0

python code/scripts/run_condition.py --dataset isic2018 --condition full --data-root data_preprocessed/isic2018_task1_cdal_png --device 0
python code/scripts/run_condition.py --dataset isic2018 --condition random-yt --data-root data_preprocessed/isic2018_task1_cdal_png --device 0
python code/scripts/run_condition.py --dataset isic2018 --condition shuffle-yt --data-root data_preprocessed/isic2018_task1_cdal_png --device 0
python code/scripts/run_condition.py --dataset isic2018 --condition core-no-diff --data-root data_preprocessed/isic2018_task1_cdal_png --device 0
```

Outputs are written below `outputs/<dataset>/<dataset>_<condition>/`. Validation selection metadata is stored as `best_checkpoint_meta.json`. Final inference refuses checkpoints without metadata proving validation-based selection. Runtime output directories and all weight extensions are ignored by Git.

## GPU smoke matrix

The smoke matrix uses the same models, audit branches, transforms, targets, loss, timestep count, sampling path, checkpoint code, and fixed manifests. It shortens training to one optimizer step, evaluates one validation item, and evaluates one test item. Full, Random-Yt, and Core-No-Diff use smoke training batch size 1; Shuffle-Yt uses 2 so its no-self-match derangement remains real.

```powershell
python tests/gpu_smoke_matrix.py --btcv-root data_preprocessed/btcv_synapse_cdal_binary_png --acdc-root data_preprocessed/acdc_mt_unet_cascade_cdal_png --isic-root data_preprocessed/isic2018_task1_cdal_png
```

Smoke metrics are execution checks, not paper performance. Temporary checkpoints, logs, predictions, and work directories are deleted by the test runner.

## Official pretrained checkpoints

The upstream authors publish checkpoints for the paper's original datasets at [Hugging Face](https://huggingface.co/Hejrati/cDAL/tree/main). They are not redistributed here and were not used for the three target-dataset smoke tests. The BTCV, ACDC, and ISIC2018 commands train their own condition-specific models.

## Documentation

- [docs/REPRODUCTION.md](docs/REPRODUCTION.md): end-to-end reproduction and checkpoint flow.
- [docs/EXPERIMENT_MATRIX.md](docs/EXPERIMENT_MATRIX.md): formal parameters for all 12 combinations.
- [docs/SPLIT_USAGE_MAP.md](docs/SPLIT_USAGE_MAP.md): loader and role mapping.
- [docs/CODE_STRUCTURE.md](docs/CODE_STRUCTURE.md): file-level function, provenance, license, and release status.
- [code/README.md](code/README.md): byte-for-byte preserved upstream README; its legacy hardware and dataset instructions are archival, not the primary environment for this release.

## Citation

```bibtex
@inproceedings{hejrati2024cdal,
  title={Conditional Diffusion Model with Spatial Attention and Latent Embedding for Medical Image Segmentation},
  author={Hejrati, Behzad and Banerjee, Soumyanil and Glide-Hurst, Carri and Dong, Ming},
  booktitle={Medical Image Computing and Computer Assisted Intervention -- MICCAI 2024},
  pages={202--212},
  year={2024},
  doi={10.1007/978-3-031-72114-4_20}
}
```
