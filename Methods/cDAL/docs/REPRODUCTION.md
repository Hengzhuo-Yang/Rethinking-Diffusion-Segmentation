# Reproduction guide

## 1. Environment

Create and activate the environment, then require an RTX 5090 preflight pass:

```powershell
conda env create -f environment.yml
conda activate cdal-rtx5090
python tests/gpu_preflight.py
```

Do not continue on CPU or another GPU. See `GPU_ENVIRONMENT.md` for exact versions and runtime flags.

## 2. Obtain data

Download BTCV/Synapse, ACDC, and ISIC2018 Task 1 only from their official providers and follow their terms. Data and masks are not supplied. Keep raw and processed data below ignored directories or another local path passed on the command line.

## 3. Preprocess

```powershell
python code/scripts/preprocess_btcv_synapse_to_cdal.py --raw_root raw/BTCV --output_root data_preprocessed/btcv_synapse_cdal_binary_png
python code/scripts/preprocess_acdc_to_cdal.py --raw_root raw/ACDC/database --split_root manifests/acdc --output_root data_preprocessed/acdc_mt_unet_cascade_cdal_png
python code/scripts/preprocess_isic2018_to_cdal.py --raw_root raw/ISIC2018 --output_root data_preprocessed/isic2018_task1_cdal_png
```

BTCV preprocessing maps the standard eight Synapse organs and then collapses them to binary foreground/background, matching the completed cDAL experiments. ACDC preserves RV, myocardium, and LV as three foreground channels. ISIC2018 preserves its official three partitions and binary lesion masks.

## 4. Validate splits before training

```powershell
python tests/validate_splits.py --btcv-root data_preprocessed/btcv_synapse_cdal_binary_png --acdc-root data_preprocessed/acdc_mt_unet_cascade_cdal_png --isic-root data_preprocessed/isic2018_task1_cdal_png
```

The command must report PASS and the exact counts in `DATA_SPLITS.md`.

## 5. Run a formal condition

The wrapper enforces a three-stage sequence:

```text
fixed training manifest
    -> gradient updates on RTX 5090
fixed validation manifest
    -> mean Dice checkpoint selection on RTX 5090
best_checkpoint.pt + best_checkpoint_meta.json
    -> fresh model and checkpoint reload
fixed test manifest
    -> final inference on RTX 5090
```

Example:

```powershell
python code/scripts/run_condition.py --dataset btcv --condition full --data-root data_preprocessed/btcv_synapse_cdal_binary_png --device 0
```

Accepted conditions are `full`, `random-yt`, `shuffle-yt`, and `core-no-diff`. The root README lists all 12 explicit commands. The wrapper refuses to overwrite a non-empty condition directory.

Formal output layout:

```text
outputs/<dataset>/<dataset>_<condition>/
  best_checkpoint.pt
  best_checkpoint_meta.json
  validation_history.csv
  validation_latest_summary.json
  final_test/
    summary.csv
    summary.json
    per_item.csv
```

All of `outputs/` is ignored and must remain out of source-control publication.

## 6. Checkpoint selection guarantees

- Training code receives only train and validation manifests.
- `best_checkpoint.pt` is promoted only when mean validation Dice improves.
- Metadata stores the dataset, condition, semantic validation split, validation-manifest hash, training step, Dice, and IoU.
- Final-test CLIs accept only test aliases and require both the checkpoint and metadata.
- Metadata dataset, condition, validation role, manifest evidence, and checkpoint filename are checked before model loading.

To run only final inference manually, use the corresponding evaluator and supply both files. For example:

```powershell
python code/evaluate_btcv_cdal.py --parameters code/parameters_btcv.json --model_path outputs/btcv/btcv_full/best_checkpoint.pt --checkpoint_metadata outputs/btcv/btcv_full/best_checkpoint_meta.json --data_dir data_preprocessed/btcv_synapse_cdal_binary_png --manifest manifests/btcv/test_cases.txt --split test --output_folder outputs/btcv/btcv_full/final_test/artifacts --summary_csv outputs/btcv/btcv_full/final_test/summary.csv --summary_json outputs/btcv/btcv_full/final_test/summary.json --per_slice_csv outputs/btcv/btcv_full/final_test/per_item.csv --audit_mode none --major_vote_number 5 --local_rank 0
```

## 7. GPU smoke acceptance

Run the 12-combination matrix after data validation and after any code change:

```powershell
python tests/gpu_smoke_matrix.py --btcv-root data_preprocessed/btcv_synapse_cdal_binary_png --acdc-root data_preprocessed/acdc_mt_unet_cascade_cdal_png --isic-root data_preprocessed/isic2018_task1_cdal_png
```

It is successful only when all 12 rows pass. The runner cleans temporary checkpoints and outputs by default. Do not report smoke metrics as final performance.

## 8. Weights and upstream assets

These target experiments train from scratch. Upstream paper-dataset checkpoints are available from the authors at `https://huggingface.co/Hejrati/cDAL/tree/main`; do not copy them into this repository. No paper PDF or figure is required to execute the release.
