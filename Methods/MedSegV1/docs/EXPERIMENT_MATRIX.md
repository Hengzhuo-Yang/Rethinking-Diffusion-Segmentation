# Formal experiment matrix

`code/configs/experiments.json` is the machine-readable source for this matrix.
Each row is an independent training run with its own validation-selected best
checkpoint and one final test evaluation.

## Twelve runs

| Dataset | Condition | Internal mode | Target / output | Train batch | Validation reverse steps | Final reverse steps / ensemble |
|---|---|---|---|---:|---:|---:|
| BTCV | Full | `none` | Binary foreground / 1 mask channel | 8 | 20 | 1,000 / 5 |
| BTCV | Random-Yt | `train_random_yt` | Binary foreground / 1 mask channel | 8 | 20 | 1,000 / 5 |
| BTCV | Shuffle-Yt | `train_shuffle_yt` | Binary foreground / 1 mask channel | 8 | 20 | 1,000 / 5 |
| BTCV | Core-No-Diff | `core_no_diff` | Binary foreground / 1 logit | 8 | 0 | 0 / 1 |
| ACDC | Full | `none` | RV, myocardium, LV / 3 foreground channels | 8 | 20 | 1,000 / 5 |
| ACDC | Random-Yt | `train_random_yt` | RV, myocardium, LV / 3 foreground channels | 8 | 20 | 1,000 / 5 |
| ACDC | Shuffle-Yt | `train_shuffle_yt` | RV, myocardium, LV / 3 foreground channels | 8 | 20 | 1,000 / 5 |
| ACDC | Core-No-Diff | `core_no_diff` | Background + 3 foreground classes / 4 logits | 8 | 0 | 0 / 1 |
| ISIC2018 | Full | `none` | Binary lesion / 1 mask channel | 8 | 20 | 1,000 / 5 |
| ISIC2018 | Random-Yt | `train_random_yt` | Binary lesion / 1 mask channel | 8 | 20 | 1,000 / 5 |
| ISIC2018 | Shuffle-Yt | `train_shuffle_yt` | Binary lesion / 1 mask channel | 8 | 20 | 1,000 / 5 |
| ISIC2018 | Core-No-Diff | `core_no_diff` | Binary lesion / 1 logit | 8 | 0 | 0 / 1 |

The reverse-step counts describe validation and final inference. Training uses a
1,000-state linear diffusion process for Full, Random-Yt and Shuffle-Yt.
Core-No-Diff does not construct `Y_t`, sample a timestep or run a reverse chain.

## Shared training configuration

| Field | Fixed value |
|---|---|
| Architecture | MedSegDiff V1, 256×256, base channels 128, 2 residual blocks, 1 attention head, attention resolution 16 |
| Diffusion | 1,000 steps, linear noise schedule, uniform timestep sampling over 0…999, learned variance |
| Optimizer | AdamW |
| Learning rate / weight decay | `5e-5` / `0.0` |
| Updates / save interval / log interval | 100,000 / 5,000 / 100 |
| Training seed | 23 |
| Sampling-module seed | 10 |
| Device / dtype | `cuda:0` on NVIDIA GeForce RTX 5090 / float32 |
| AMP | Disabled |
| TF32 observed in verified runtime | Matmul disabled; cuDNN enabled |
| cuDNN | Benchmark disabled; deterministic disabled |
| Workers | 4, pinned memory, persistent workers, prefetch factor 2 |
| Augmentation | None beyond deterministic resize |

Full, Random-Yt and Shuffle-Yt optimize epsilon MSE plus the learned-variance
VB term. The locally disabled calibration MSE does not enter any condition.
Random-Yt and Shuffle-Yt change only the training-time mask input to the model;
their timestep, epsilon target, loss and inference remain identical to Full.
Core-No-Diff uses binary BCE-with-logits plus soft Dice, or ACDC multiclass
cross-entropy plus foreground soft Dice.

## Dataset policies

| Dataset | Normalization | Class mapping | Metric / post-processing |
|---|---|---|---|
| BTCV | CT window `[-175, 250]`, uint8 RGB, then `[0,1]` | background; union of remapped labels 1…8 as foreground | Slice foreground Dice/IoU; diffusion `sample`, threshold 0.5; empty-ground-truth slices excluded from aggregate mean |
| ACDC | Per labelled ED/ES frame, nonzero percentile clip `[1,99]`, uint8 RGB, then `[0,1]` | 0 background, 1 RV, 2 myocardium, 3 LV | Foreground class Dice/IoU; threshold foreground channels then argmax; empty-ground-truth observations excluded |
| ISIC2018 | RGB uint8 scaled to `[0,1]` | background; lesion | Image foreground Dice/IoU; diffusion `sample`, threshold 0.5; empty-ground-truth images excluded from aggregate mean |

## Selection and final evaluation

At every scheduled checkpoint from step 5,000 onward, `ValidationRunner` scores
the complete fixed validation manifest. `dice_mean` is maximized independently
for each run. With `save_best_only=True`, a scored non-best checkpoint and its
optimizer/EMA sidecars are deleted; `best_checkpoint_meta.json` records the
retained step and score. The final test stage reads that metadata, reloads the
retained model, and evaluates the fixed test manifest once.

The test partition is not consulted during training or model selection. Formal
results should report the dataset, condition, seed, best validation step,
validation score, final test metric policy, and the immutable resolved config.

## Smoke configuration is not a formal experiment

`tests/gpu_smoke_matrix.py` uses batch size 1 (2 for Shuffle-Yt), one optimizer
step, one validation sample, one test sample and one reverse step. Its purpose is
only to prove CUDA routing and executable coverage. It reads the formal config
without changing it, stores no model artifact, and must not be reported as a
scientific result.
