# Reproduction guide

This guide reproduces the fixed 3-dataset x 4-condition experiment matrix. Run
all commands from the release root in PowerShell. The root `README.md` is the
public entry point; `code/README.md` is retained unchanged as historical
upstream documentation.

## 1. Create the supported environment

The supported target is Windows, Conda, and an NVIDIA GeForce RTX 5090 at
`cuda:0`. CPU fallback and other GPU models are rejected by the public runner.

```powershell
conda env create -f environment.yml
conda activate medsegdiff-audit-rtx5090
python -m pip install -r requirements-torch-cu128.txt
python tests\gpu_preflight.py
```

The last command must report `PASS` before data preparation, training, or
sampling. Exact verified versions and runtime flags are in
`docs/GPU_ENVIRONMENT.md`.

## 2. Obtain and prepare the datasets

This repository supplies neither medical data nor permission to redistribute
it. Obtain BTCV/Synapse, ACDC, and ISIC2018 Task 1 from their authorized sources
and comply with their terms. Use new processed output directories; do not place
data inside a source upload.

### BTCV / Synapse

The preprocessor requires all 30 labelled source volumes and writes the fixed
18/2/10 case-level train/validation/test split.

```powershell
python code\scripts\preprocess_btcv_synapse_to_medsegv1.py `
  --raw_root <btcv-training-root> `
  --output_root <processed-btcv-root>
```

It fails if the source cases or resulting slice counts differ from the fixed
release contract.

### ACDC

Use the packaged subject manifests directly as the split root. The preprocessor
accepts `train_subjects.txt`, `val_subjects.txt`, and `test_subjects.txt` and
writes `training`, `validation`, and `testing` image/mask directories. A custom
split root may instead use the legacy `train_patients.txt`, `val_patients.txt`,
and `test_patients.txt` filenames.

```powershell
python code\scripts\prepare_acdc.py `
  --raw_root <acdc-database-root> `
  --split_root manifests\acdc `
  --out_dir <processed-acdc-root>
```

All labelled ED/ES frames and slices from one patient remain in the same
partition.

### ISIC2018 Task 1

Place the official training, validation, and testing image/mask ZIP archives
under one raw root. The preprocessor preserves those official roles.

```powershell
python code\scripts\preprocess_isic2018_to_medsegv1.py `
  --raw_root <isic2018-archive-root> `
  --output_root <processed-isic2018-root>
```

The generated `manifest.csv` and `summary.json` files are local preparation
records. The ID-only files under `manifests/` remain the authoritative public
split identities used by loaders and the experiment runner.

## 3. Validate identities, pairings, and leakage

First validate the packaged manifests alone:

```powershell
python tests\validate_splits.py
```

Then validate all three prepared trees:

```powershell
python tests\validate_splits.py `
  --btcv-root <processed-btcv-root> `
  --acdc-root <processed-acdc-root> `
  --isic2018-root <processed-isic2018-root>
```

Do not start training unless both commands pass. The prepared-data check covers
expected unit and sample counts, disjoint roles, ID mapping, image/mask pairing,
and complete manifest coverage. The expected counts and identities are in
`docs/DATA_SPLITS.md`.

## 4. Run the twelve formal experiments

Each dataset/condition pair is an independent run. `run_experiment.py` resolves
the fixed architecture, optimizer, loss, seeds, sampling policy, manifests, and
hardware requirements from `code/configs/experiments.json`.

```powershell
$dataRoots = @{
  btcv = '<processed-btcv-root>'
  acdc = '<processed-acdc-root>'
  isic2018 = '<processed-isic2018-root>'
}
$conditions = @('full', 'train_random_yt', 'train_shuffle_yt', 'core_no_diff')

foreach ($dataset in @('btcv', 'acdc', 'isic2018')) {
  foreach ($condition in $conditions) {
    python code\scripts\run_experiment.py `
      --dataset $dataset `
      --condition $condition `
      --stage all `
      --data-root $dataRoots[$dataset] `
      --output-root <output-root>
  }
}
```

This expands to exactly twelve runs. The resolved directory for each run is:

```text
<output-root>/<dataset>/<condition>/seed_23/
```

Keep the output root outside the source release. It contains checkpoints, logs,
validation records, predictions, metrics, and local resolved paths that must not
be uploaded with the code.

## 5. Checkpoint selection and final testing

With `--stage all`, the sequence is:

```text
training update
  -> scheduled complete validation-manifest evaluation
  -> maximize validation dice_mean independently for this run
  -> retain best checkpoint and best_checkpoint_meta.json
  -> reload that checkpoint
  -> one fixed test-manifest evaluation
```

The validation set is used only for checkpoint selection. The test set is not
read to choose a checkpoint, tune a threshold, stop training, or select a
method. Final test metadata records that its checkpoint source was validation.

The two phases may be resumed separately:

```powershell
python code\scripts\run_experiment.py `
  --dataset <btcv|acdc|isic2018> `
  --condition <full|train_random_yt|train_shuffle_yt|core_no_diff> `
  --stage train `
  --data-root <processed-dataset-root> `
  --output-root <output-root>

python code\scripts\run_experiment.py `
  --dataset <btcv|acdc|isic2018> `
  --condition <full|train_random_yt|train_shuffle_yt|core_no_diff> `
  --stage test `
  --data-root <processed-dataset-root> `
  --output-root <output-root>
```

`--stage test` requires the existing validation-generated
`best_checkpoint_meta.json` and its referenced checkpoint. It intentionally
cannot select a checkpoint from test metrics.

## 6. Verification before a formal campaign

Run the focused semantic and loader checks, then repeat the GPU checks on the
target host:

```powershell
python tests\test_train_random_yt_audit.py
python tests\test_isic2018_binary_loader_and_metrics.py
python tests\gpu_preflight.py
python tests\gpu_smoke_matrix.py `
  --btcv-root <processed-btcv-root> `
  --acdc-root <processed-acdc-root> `
  --isic2018-root <processed-isic2018-root> `
  --smoke-sampling-steps 1 `
  --report <temporary-smoke-report.json>
```

The release-preparation run passed all twelve matrix rows on 2026-07-19; see
`docs/GPU_SMOKE_TEST.md`. The one-step smoke schedule tests routing only. It
does not change or replace formal 1,000-step final diffusion inference. ACDC
Core-No-Diff and the two binary Core-No-Diff runs use no reverse steps.

## 7. Reporting and reproducibility boundary

For each formal result, retain and report the dataset, condition, seed, best
validation step, validation score, final test metric policy, and
`resolved_config.json`. Report all twelve runs or state omissions explicitly.

The release fixes the source, manifests, environment versions, architecture,
optimizer, seeds, and evaluation roles. It does not claim bitwise replay across
driver or hardware changes: the verified cuDNN configuration is not
deterministic, and the public target is limited to the recorded RTX 5090 stack.
The smoke result is execution evidence, not a scientific accuracy result. No
weights or completed formal-run metrics are distributed.
