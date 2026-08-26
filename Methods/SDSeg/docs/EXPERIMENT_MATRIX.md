# Formal experiment matrix

The release matrix is the Cartesian product of three datasets and four existing
conditions. All rows use the same fixed formal protocol.

## Shared parameters

| Parameter | Formal value |
|---|---|
| Seed | `23` |
| Batch size | `4` |
| Data-loader workers | `8` |
| Maximum training steps | `100000` |
| Base learning rate | `1e-5` |
| Learning-rate scaling | disabled |
| Optimizer | existing AdamW path |
| Diffusion training timesteps | `1000` where applicable |
| Checkpoint monitor | `val_avg_dice` |
| Checkpoint mode | `max` |
| Selection data | validation only (`metric_validation`) |
| Formal segmentation evaluation | direct, fixed one-step |
| DDIM eta argument | `0.0` |
| Final test repetitions | `1` |
| Device | CUDA only; exact `NVIDIA GeForce RTX 5090` |

The one-step evaluation setting is fixed; it is not a sampling-step comparison
or sweep. `Core-No-Diff` performs its existing direct image-only main-core
forward, for which reverse sampling is not applicable.

## The 12 configurations

| ID | Dataset | Condition | Configuration | `audit_mode` | Final-test CLI dataset |
|---:|---|---|---|---|---|
| 1 | BTCV | Full | `configs/experiments/btcv/full.yaml` | `full_diffusion` | `btcv-b` |
| 2 | BTCV | Random-Yt | `configs/experiments/btcv/random-yt.yaml` | `train_random_yt` | `btcv-b` |
| 3 | BTCV | Shuffle-Yt | `configs/experiments/btcv/shuffle-yt.yaml` | `train_shuffle_yt` | `btcv-b` |
| 4 | BTCV | Core-No-Diff | `configs/experiments/btcv/core-no-diff.yaml` | `core_no_diff` | `btcv-b` |
| 5 | ACDC | Full | `configs/experiments/acdc/full.yaml` | `full_diffusion` | `acdc` |
| 6 | ACDC | Random-Yt | `configs/experiments/acdc/random-yt.yaml` | `train_random_yt` | `acdc` |
| 7 | ACDC | Shuffle-Yt | `configs/experiments/acdc/shuffle-yt.yaml` | `train_shuffle_yt` | `acdc` |
| 8 | ACDC | Core-No-Diff | `configs/experiments/acdc/core-no-diff.yaml` | `core_no_diff` | `acdc` |
| 9 | ISIC2018 | Full | `configs/experiments/isic2018/full.yaml` | `full_diffusion` | `isic2018` |
| 10 | ISIC2018 | Random-Yt | `configs/experiments/isic2018/random-yt.yaml` | `train_random_yt` | `isic2018` |
| 11 | ISIC2018 | Shuffle-Yt | `configs/experiments/isic2018/shuffle-yt.yaml` | `train_shuffle_yt` | `isic2018` |
| 12 | ISIC2018 | Core-No-Diff | `configs/experiments/isic2018/core-no-diff.yaml` | `core_no_diff` | `isic2018` |

Paths in the table are relative to `code/`.

## Comparability rules

For comparisons across conditions, keep constant:

- dataset split manifests and prepared files;
- initialization assets and their checksums;
- seed and formal training schedule;
- batch size, workers, augmentation, and preprocessing;
- validation metric and checkpoint rule;
- final-test checkpoint gate and evaluation protocol;
- software and GPU environment.

Only the condition-specific existing implementation may differ. Random-Yt and
Shuffle-Yt alter training input only; their validation/inference route remains
the shared route. Core-No-Diff is an explicitly objective-changing ablation.

## Valid result unit

A complete formal result row consists of one configuration, its validation-
selected best checkpoint proof, and its one-time final test summary. A partial
training run, a smoke-limited split, a last checkpoint, or a test-selected
checkpoint is not a formal matrix result.
