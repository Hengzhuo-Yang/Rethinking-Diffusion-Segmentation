# Reproduction guide

This guide uses the guarded release pipeline. Run commands from `code/` unless
stated otherwise. Replace angle-bracket placeholders with local paths; do not
commit those local paths, data, weights, checkpoints, or outputs.

## 1. Create a separate environment

The existing proven `sdseg` environment is read-only evidence and must not be
updated in place. From the release root, new users may create a separate
environment:

```bash
conda env create -f environment.yml
conda activate sdseg-release
cd code
```

The key proven versions are Python `3.10.20`, PyTorch `2.11.0+cu128`,
torchvision `0.26.0+cu128`, and PyTorch Lightning `1.9.5`. Review
`docs/GPU_ENVIRONMENT.md` before treating a newly solved environment as
equivalent to the proven environment.

## 2. Supply local-only assets

This source release does not include data or checkpoints. Obtain each dataset
and any pretrained assets from authorized sources under their respective terms.

Expected pretrained layout:

```text
<PRETRAINED_ROOT>/
|-- ldm/lsun_churches256/model.ckpt
`-- first_stage_models/kl-f8/model.ckpt
```

Expected prepared-data layout:

```text
<DATA_ROOT>/
|-- btcv/
|   |-- train/{images,masks}/
|   |-- validation/{images,masks}/
|   `-- test/{images,masks}/
|-- acdc/
|   |-- train/{images,masks}/
|   |-- validation/{images,masks}/
|   `-- test/{images,masks}/
`-- isic2018/
    |-- training/{images,masks}/
    |-- validation/{images,masks}/
    `-- testing/{images,masks}/
```

## 3. Preprocess the official datasets

Use full official inputs and an empty destination. The `max-*` options exposed
by the scripts are smoke-only; leave them at their default zero for formal data.

```bash
python scripts/preprocess_btcv_synapse_to_sdseg.py \
  --raw-root <BTCV_RAW> \
  --output-root <DATA_ROOT>/btcv

python scripts/preprocess_acdc_to_sdseg.py \
  --raw-root <ACDC_RAW> \
  --output-root <DATA_ROOT>/acdc

python scripts/preprocess_isic2018_to_sdseg.py \
  --raw-root <ISIC2018_ARCHIVES> \
  --output-root <DATA_ROOT>/isic2018
```

The ACDC preprocessor defaults to the bundled patient manifests. BTCV assigns
whole cases to the fixed 18/2/10 split. ISIC2018 preserves the official archive
partitions. Preprocessors refuse a non-empty destination unless `--overwrite`
is explicit.

Validate the prepared data before training:

```bash
python tests/validate_splits.py --data-root <DATA_ROOT>
```

Run the source-only release checks as a separate gate:

```bash
python tests/test_release_static.py -v
```

Required totals are BTCV `2211/295/1273`, ACDC `1304/182/416`, and ISIC2018
`2594/100/1000` for train/validation/test. A mismatch, duplicate, missing
image/mask pair, or cross-partition identifier blocks a formal run.

## 4. Inspect a registered experiment

The pipeline accepts:

- actions: `plan`, `validate`, `train`, `test`, `run`;
- datasets: `btcv`, `acdc`, `isic2018`;
- conditions: `full`, `random-yt`, `shuffle-yt`, `core-no-diff`.

Show the complete interface:

```bash
python scripts/release_pipeline.py --help
```

Plan one row without launching training:

```bash
python scripts/release_pipeline.py plan \
  --dataset btcv --condition full \
  --data-root <DATA_ROOT> \
  --pretrained-root <PRETRAINED_ROOT> \
  --runs-root <RUNS_ROOT>
```

`plan` validates the formal configuration and data splits, then prints the GPU
preflight and training commands. It does not execute them.

The formal config gate requires seed `23`, batch size `4`, workers `8`, maximum
steps `100000`, base learning rate `1e-5`, the registered audit mode, and
separate metric-validation and test datasets. Sampling-step audit configuration
is rejected.

## 5. Validate CUDA and model placement

Run the registered configuration and GPU preflight without training:

```bash
python scripts/release_pipeline.py validate \
  --dataset btcv --condition full \
  --data-root <DATA_ROOT> \
  --pretrained-root <PRETRAINED_ROOT>
```

The pipeline calls `tests/gpu_preflight.py` with the selected formal config and
pretrained root. The formal device name must exactly match
`NVIDIA GeForce RTX 5090`; CPU fallback is disabled. A failed identity,
capability, compiled-architecture, configuration, pretrained-input,
model-placement, or finite CUDA forward-probe check blocks the run. Backward
execution and the optimizer update are separate checks performed by
`tests/gpu_smoke_12.py`, not by the preflight.

To store a standalone preflight report explicitly:

```bash
python tests/gpu_preflight.py \
  --config configs/experiments/btcv/full.yaml \
  --pretrained-root <PRETRAINED_ROOT> \
  --cuda-device 0 \
  --expected-gpu-name "NVIDIA GeForce RTX 5090" \
  --expected-capability 12.0 \
  --report <LOCAL_REPORT_DIR>/btcv-full-preflight.json
```

## 6. Train, select on validation, and test once

The preferred complete command is:

```bash
python scripts/release_pipeline.py run \
  --dataset btcv --condition full \
  --data-root <DATA_ROOT> \
  --pretrained-root <PRETRAINED_ROOT> \
  --runs-root <RUNS_ROOT>
```

`run` performs the guarded sequence:

1. validate the registered config and fixed manifests;
2. run GPU preflight;
3. train with the fixed formal settings and without test access;
4. locate the single new run directory;
5. require `best_checkpoint.json` produced by maximum validation Dice;
6. load the exact checkpoint named by that metadata;
7. run direct one-step final inference on the test split exactly once;
8. validate the test summary and write `final_test_receipt.json`.

By default, the final-test directory is `<RUN_DIR>/final_test`. The pipeline
refuses to run if that path already exists, including after a failed attempt;
this is the one-time boundary.

Do not use `--save-results` unless local prediction files are specifically
needed and can be stored securely. Predictions must not be committed.

## 7. Separate training and testing when necessary

Train without touching the test partition:

```bash
python scripts/release_pipeline.py train \
  --dataset acdc --condition random-yt \
  --data-root <DATA_ROOT> \
  --pretrained-root <PRETRAINED_ROOT> \
  --runs-root <RUNS_ROOT>
```

After training completes and best-checkpoint metadata is available, run the
one-time final test using the selected run directory:

```bash
python scripts/release_pipeline.py test \
  --dataset acdc --condition random-yt \
  --data-root <DATA_ROOT> \
  --pretrained-root <PRETRAINED_ROOT> \
  --run-dir <RUN_DIR>
```

Alternatively, provide the proof file and a new output path explicitly:

```bash
python scripts/release_pipeline.py test \
  --dataset isic2018 --condition core-no-diff \
  --data-root <DATA_ROOT> \
  --pretrained-root <PRETRAINED_ROOT> \
  --selection-metadata <RUN_DIR>/best_checkpoint.json \
  --outdir <NEW_FINAL_TEST_OUTPUT>
```

The metadata must declare:

```text
selection_partition = validation
metric_dataset_key  = metric_validation
monitor             = val_avg_dice
mode                = max
```

It must include a validation score and an existing `best_model_path`. The
evaluator also checks that the requested checkpoint is exactly that path.

## 8. Cover the entire matrix

Repeat the registered `run` command for all combinations:

```text
datasets   = btcv, acdc, isic2018
conditions = full, random-yt, shuffle-yt, core-no-diff
```

Use a fresh run directory and final-test destination per row. The exact config
mapping is in `docs/EXPERIMENT_MATRIX.md`.

Run the complete smoke matrix with:

```bash
python tests/gpu_smoke_12.py \
  --data-root <DATA_ROOT> \
  --pretrained-root <PRETRAINED_ROOT> \
  --report <LOCAL_REPORT_DIR>/gpu-smoke-12.json \
  --max-eval-items 1 \
  --expected-gpu-name "NVIDIA GeForce RTX 5090"
```

Before full training, follow the remaining evidence requirements in
`docs/GPU_SMOKE_TEST.md`. The JSON report must contain all 12 explicit passing
rows before claiming that the release smoke matrix passed. A filtered run,
single-config preflight, or dry run is not equivalent to the complete matrix.

## 9. Preserve evidence without publishing restricted artifacts

For each formal row, retain locally:

- resolved config and its hash;
- split-manifest hashes and validated counts;
- GPU/environment record;
- `audit_metadata.json`;
- `best_checkpoint.json`;
- validation and final-test metric summaries;
- `final_test_receipt.json`;
- complete failure information for any blocked or failed run.

Do not publish datasets, clinical images, masks, pretrained weights, trained
checkpoints, predictions, private paths, or local run directories. Checkpoint
files are deliberately outside this source-only release.
