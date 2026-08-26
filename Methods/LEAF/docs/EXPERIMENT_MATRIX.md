# Experiment Matrix

The formal release matrix is the Cartesian product of three datasets and four conditions: 3 x 4 = 12 runs. It intentionally excludes sampling-step sweeps.

Run every command below from the release root. The examples use this local, untracked layout:

```text
data_preprocessed/
|-- btcv_synapse_leaf_binary_png/BTCV/{train,val,test}
|-- acdc_mt_unet_cascade_leaf_png/ACDC/{training,validation,testing}
`-- isic2018_task1_leaf_png/ISIC18/{training,validation,testing}

assets/{vae,unet}
runs/
```

## Shared formal settings

All twelve YAML configurations use seed 1337, 256 x 256 inputs, batch size 4, validation batch size 32, BF16 mixed precision, one-process training, AdamW with learning rate `4e-5` and weight decay `0.01`, 10,000 warmup steps, 100,000 total steps, EMA, and aligned-feature weight `0.75`. Validation begins at step 30,000 and repeats every 200 steps. The selected metric is mean validation Dice.

## Conditions

| Public condition | Configuration `audit_mode` | Training input distinction |
|---|---|---|
| `full` | `none` | Standard diffusion-corrupted mask latent `Y_t` |
| `random-yt` | `train_random_yt` | Independent Gaussian tensor replaces `Y_t` during training |
| `shuffle-yt` | `train_shuffle_yt` | Deranged mask source is corrupted at the current timestep during training |
| `core-no-diff` | `core_no_diff` | Image-only segmentation path; no diffusion corruption or timestep conditioning |

The target and loss controls are described in [AUDIT_IMPLEMENTATION_MAP.md](AUDIT_IMPLEMENTATION_MAP.md).

## Twelve fixed combinations

| Dataset | Condition | Config file | Validation partition | Final-test partition |
|---|---|---|---|---|
| BTCV | Full | `config-btcv.yaml` | `val` | `test` |
| BTCV | Random-Yt | `config-btcv-train-random-yt.yaml` | `val` | `test` |
| BTCV | Shuffle-Yt | `config-btcv-train-shuffle-yt.yaml` | `val` | `test` |
| BTCV | Core-No-Diff | `config-btcv-core-no-diff.yaml` | `val` | `test` |
| ACDC | Full | `config-acdc.yaml` | `validation` | `testing` |
| ACDC | Random-Yt | `config-acdc-train-random-yt.yaml` | `validation` | `testing` |
| ACDC | Shuffle-Yt | `config-acdc-train-shuffle-yt.yaml` | `validation` | `testing` |
| ACDC | Core-No-Diff | `config-acdc-core-no-diff.yaml` | `validation` | `testing` |
| ISIC2018 | Full | `config-isic2018.yaml` | `validation` | `testing` |
| ISIC2018 | Random-Yt | `config-isic2018-train-random-yt.yaml` | `validation` | `testing` |
| ISIC2018 | Shuffle-Yt | `config-isic2018-train-shuffle-yt.yaml` | `validation` | `testing` |
| ISIC2018 | Core-No-Diff | `config-isic2018-core-no-diff.yaml` | `validation` | `testing` |

## BTCV commands

```bash
python code/scripts/release_pipeline.py run --dataset btcv --condition full --data-root data_preprocessed/btcv_synapse_leaf_binary_png --assets-root assets --runs-root runs
python code/scripts/release_pipeline.py run --dataset btcv --condition random-yt --data-root data_preprocessed/btcv_synapse_leaf_binary_png --assets-root assets --runs-root runs
python code/scripts/release_pipeline.py run --dataset btcv --condition shuffle-yt --data-root data_preprocessed/btcv_synapse_leaf_binary_png --assets-root assets --runs-root runs
python code/scripts/release_pipeline.py run --dataset btcv --condition core-no-diff --data-root data_preprocessed/btcv_synapse_leaf_binary_png --assets-root assets --runs-root runs
```

## ACDC commands

```bash
python code/scripts/release_pipeline.py run --dataset acdc --condition full --data-root data_preprocessed/acdc_mt_unet_cascade_leaf_png --assets-root assets --runs-root runs
python code/scripts/release_pipeline.py run --dataset acdc --condition random-yt --data-root data_preprocessed/acdc_mt_unet_cascade_leaf_png --assets-root assets --runs-root runs
python code/scripts/release_pipeline.py run --dataset acdc --condition shuffle-yt --data-root data_preprocessed/acdc_mt_unet_cascade_leaf_png --assets-root assets --runs-root runs
python code/scripts/release_pipeline.py run --dataset acdc --condition core-no-diff --data-root data_preprocessed/acdc_mt_unet_cascade_leaf_png --assets-root assets --runs-root runs
```

## ISIC2018 commands

```bash
python code/scripts/release_pipeline.py run --dataset isic2018 --condition full --data-root data_preprocessed/isic2018_task1_leaf_png --assets-root assets --runs-root runs
python code/scripts/release_pipeline.py run --dataset isic2018 --condition random-yt --data-root data_preprocessed/isic2018_task1_leaf_png --assets-root assets --runs-root runs
python code/scripts/release_pipeline.py run --dataset isic2018 --condition shuffle-yt --data-root data_preprocessed/isic2018_task1_leaf_png --assets-root assets --runs-root runs
python code/scripts/release_pipeline.py run --dataset isic2018 --condition core-no-diff --data-root data_preprocessed/isic2018_task1_leaf_png --assets-root assets --runs-root runs
```

Each `run` command first validates its physical cache against the fixed manifests, then performs a strict RTX 5090 preflight, formal training with validation-only checkpoint selection, and exactly one final-test evaluation using the recorded EMA checkpoint. To inspect command construction without reading data or starting a run, append `--dry-run`.

To separate the stages, replace `run` with `train`, then invoke the otherwise identical command with `test`. The `test` action fails unless the run directory contains valid validation-selection metadata, and it refuses to overwrite an existing non-empty `final_test/` directory.
