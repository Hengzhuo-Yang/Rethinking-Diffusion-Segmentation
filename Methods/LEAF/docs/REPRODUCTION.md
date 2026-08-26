# Reproduction

This procedure recreates the fixed 12-run release matrix without placing datasets, pretrained parameters, or generated results in version control. Run commands from the release root unless stated otherwise.

## 1. Hardware and environment

Formal execution requires exactly one NVIDIA GeForce RTX 5090. The runtime rejects CPU execution and any other GPU model.

Create the pinned Conda environment:

```bash
conda env create -f environment.yml
conda activate leaf-rtx5090
```

Alternatively, create a Python 3.11.11 environment and install `requirements.txt`. The PyTorch packages are pinned to the CUDA 12.8 wheel index.

The formal configurations use BF16. Training and training-time validation enable TF32 matrix multiplication and cuDNN TF32. Seed 1337 controls the explicitly seeded generators, but bitwise identity across driver or library revisions is not guaranteed.

## 2. Prepare local pretrained components

Pretrained parameters are not distributed in this repository. Prepare a read-only local root with the directory structure expected by `from_pretrained`:

```text
assets/
|-- vae/
|   |-- config.json
|   `-- diffusion_pytorch_model.safetensors or diffusion_pytorch_model.bin
`-- unet/
    |-- config.json
    `-- diffusion_pytorch_model.safetensors or diffusion_pytorch_model.bin
```

The aligned-feature loss also loads DINOv2 through Torch Hub. Make the required DINOv2 repository and parameters available in the local Torch Hub cache before an offline run. For the GPU smoke harness, a local DINOv2 checkout can instead be passed with `--dinov2-repo`.

`code/extract_weights.py` is retained as the upstream conversion helper for compatible source checkpoints. It expects source checkpoint files under `code/assets/` and must be run from `code/`; do not add those files or converted parameters to version control.

## 3. Preprocess the datasets

Obtain each dataset under its own terms. The release includes no patient data or images. The following commands write only to the new `data_preprocessed/` tree and use the fixed manifests in `code/manifests/` by default.

```bash
python code/scripts/preprocess_btcv_synapse_to_leaf.py --raw_root raw/BTCV --output_root data_preprocessed/btcv_synapse_leaf_binary_png
python code/scripts/preprocess_acdc_to_leaf.py --raw_root raw/ACDC --output_root data_preprocessed/acdc_mt_unet_cascade_leaf_png
python code/scripts/preprocess_isic2018_to_leaf.py --raw_root raw/ISIC2018 --output_root data_preprocessed/isic2018_task1_leaf_png
```

The output roots must be empty. `--overwrite` explicitly replaces a non-empty dataset-specific output root; use it only when regeneration is intended.

Expected generated sample counts are:

| Dataset | Train | Validation | Test |
|---|---:|---:|---:|
| BTCV | 2,211 slices | 295 slices | 1,273 slices |
| ACDC | 1,304 slices | 182 slices | 416 slices |
| ISIC2018 | 2,594 images | 100 images | 1,000 images |

See [DATA_SPLITS.md](DATA_SPLITS.md) for subject/image allocations and [SPLIT_USAGE_MAP.md](SPLIT_USAGE_MAP.md) for physical directory routing.

## 4. Validate split integrity

First validate the committed ID manifests alone:

```bash
python tests/validate_splits.py
```

Then validate the generated image/mask caches against those manifests:

```bash
python tests/validate_splits.py --data-root btcv=data_preprocessed/btcv_synapse_leaf_binary_png --data-root acdc=data_preprocessed/acdc_mt_unet_cascade_leaf_png --data-root isic2018=data_preprocessed/isic2018_task1_leaf_png
```

Both commands must end with `split validation passed`. The validator checks exact ordered-manifest fingerprints, counts, duplicate IDs, cross-partition leakage, paired images/masks, sample ownership, and expected cache sizes. ACDC label maps are checked as well.

## 5. Run the strict GPU preflight

```bash
python tests/gpu_preflight.py --device cuda:0 --json-out runs/gpu_preflight.json
```

A passing report has top-level status `PASS` and records the exact GPU identity, CUDA availability, `nvidia-smi` agreement, CUDA tensor forward/backward evidence, a tiny CUDA model optimizer update, and peak allocated memory. A failure terminates formal execution; there is no CPU fallback.

## 6. Run one-step GPU coverage

Before formal training, the complete smoke harness can exercise all twelve dataset/condition combinations with real LEAF components:

```bash
python tests/gpu_smoke_12.py --dataset all --condition all --data-root btcv=data_preprocessed/btcv_synapse_leaf_binary_png --data-root acdc=data_preprocessed/acdc_mt_unet_cascade_leaf_png --data-root isic2018=data_preprocessed/isic2018_task1_leaf_png --pretrained-root assets --work-dir runs/gpu-smoke
```

The harness performs one training batch with forward, loss, backward, and optimizer update; one validation batch; temporary best-checkpoint save and reload; and one held-out test batch. Temporary per-combination artifacts are removed. See [GPU_SMOKE_TEST.md](GPU_SMOKE_TEST.md) for evidence fields and failure interpretation.

## 7. Train, select on validation, and test once

Use the release entry point. For example:

```bash
python code/scripts/release_pipeline.py run --dataset btcv --condition full --data-root data_preprocessed/btcv_synapse_leaf_binary_png --assets-root assets --runs-root runs
```

The pipeline enforces this sequence:

1. Validate the selected physical cache against the fixed manifests and counts.
2. Run the strict GPU preflight.
3. Train only on the fixed training partition.
4. Starting at step 30,000, evaluate the fixed validation partition every 200 steps.
5. Select the checkpoint with the highest mean validation Dice.
6. Write `best_checkpoint.json` inside the condition's run directory.
7. Reload only the EMA weights from the checkpoint named by that metadata.
8. Evaluate the fixed final-test partition once and write results under `final_test/`.

The final-test evaluator does not choose a checkpoint. The pipeline verifies that selection metadata names the requested dataset and condition, records the exact configured validation split, declares `selection_partition` as `validation`, and declares `selection_metric` as `mean_validation_dice`. It refuses a missing or outside-run checkpoint path, missing requested EMA weights, and existing final-test output.

Run all twelve commands in [EXPERIMENT_MATRIX.md](EXPERIMENT_MATRIX.md). For staged execution, use `train` first and `test` later with otherwise identical arguments:

```bash
python code/scripts/release_pipeline.py train --dataset btcv --condition full --data-root data_preprocessed/btcv_synapse_leaf_binary_png --assets-root assets --runs-root runs
python code/scripts/release_pipeline.py test --dataset btcv --condition full --data-root data_preprocessed/btcv_synapse_leaf_binary_png --assets-root assets --runs-root runs
```

## 8. Expected run records

Each configuration supplies a unique `job_name`, so its artifacts occupy a separate directory under `runs/`. The important records are:

```text
runs/<job_name>/
|-- best_checkpoint.json
|-- checkpoint/step-*/
|-- validation_metrics.csv
`-- final_test/
    |-- final_test_request.json
    |-- metrics_summary.csv
    |-- metrics_summary.json
    |-- per_slice_metrics.csv or per_image_metrics.csv
    `-- samples/
```

Exact training-side checkpoint and validation filenames can vary with save step, but `best_checkpoint.json` is the authoritative bridge between validation selection and final testing. Preserve the YAML configuration, environment lock, split fingerprints, preflight report, selection metadata, and final metrics together when reporting a result.
