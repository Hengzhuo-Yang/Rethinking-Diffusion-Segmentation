# Experiment matrix

These are the local formal settings that produced the completed runs and are now the public defaults. They are not replaced by smoke settings.

## Dataset-specific invariants

| Dataset | Target and classes | Image/crop | Normalization | Training augmentation | Evaluation/post-processing |
|---|---|---|---|---|---|
| BTCV | One clean binary foreground mask channel; background 0, any mapped Synapse organ foreground 1 | Resize to 256×256 | RGB source averaged to one condition channel; divide by 255 and map to `[-1,1]`; mask to `[-1,1]` | Random affine ±22 degrees, scale 0.75–1.25 | Five reverse samples averaged, clamp `[0,1]`, round; foreground Dice/IoU, empty-GT slices skipped |
| ACDC | Three clean foreground channels: RV, myocardium, LV; background is all-zero | Resize to 256×256 | RGB image averaged to one condition channel in `[-1,1]`; three masks mapped to `[-1,1]` | Random affine ±22 degrees, scale 0.75–1.25 | Five samples averaged; labels recovered by background/foreground argmax; mean Dice/IoU over classes 1–3, empty slice/class observations skipped |
| ISIC2018 | One clean binary lesion channel | Resize to 256×256 | RGB source averaged to one condition channel in `[-1,1]`; mask to `[-1,1]` | Random affine ±22 degrees, scale 0.75–1.25 | Five reverse samples averaged, clamp `[0,1]`, round; lesion Dice/IoU, empty-GT images skipped |

Validation and test use resize plus tensor conversion only; random augmentation is training-only.

## Twelve formal conditions

| Dataset | Condition | Target | Formal batch | Smoke train batch | Active duration | Optimizer and LR | Loss | Seed | Timesteps / sampling | Checkpoint metric | Device / dtype / numeric settings |
|---|---|---|---:|---:|---|---|---|---:|---|---|---|
| BTCV | Full | clean binary mask | 4 | 1 | 10,000 steps | Adam G `2e-4`, Adam D `1e-5`; betas 0.5/0.9 | G clean-mask MSE; adversarial D with R1 | 47 | train `t` uniform over 4; test 4 reverse steps, ensemble 5 | mean validation Dice | RTX 5090 `cuda:0`, float32, AMP off, matmul TF32 off, cuDNN TF32 on |
| BTCV | Random-Yt | same | 4 | 1 | 10,000 steps | same | same target/loss | 47 | same `t`; random training `Y_t`; Full inference | same | same |
| BTCV | Shuffle-Yt | same | 4 | 2 | 10,000 steps | same | same target/loss | 47 | same `t` and noise, peer-mask `Y_t`; Full inference | same | same |
| BTCV | Core-No-Diff | same | 4 | 1 | 10,000 steps | Adam G `2e-4`; D frozen | clean-mask MSE | 47 | no `t`, forward diffusion, or reverse steps in main core | same | same |
| ACDC | Full | clean 3-channel foreground mask | 4 | 1 | 10,000 steps | Adam G `2e-4`, Adam D `1e-5`; betas 0.5/0.9 | G clean-mask MSE; adversarial D with R1 | 47 | train `t` uniform over 4; test 4 reverse steps, ensemble 5 | mean foreground validation Dice | same |
| ACDC | Random-Yt | same | 4 | 1 | 10,000 steps | same | same target/loss | 47 | same `t`; random training `Y_t`; Full inference | same | same |
| ACDC | Shuffle-Yt | same | 4 | 2 | 10,000 steps | same | same target/loss | 47 | same `t` and noise, peer-mask `Y_t`; Full inference | same | same |
| ACDC | Core-No-Diff | same | 4 | 1 | 10,000 steps | Adam G `2e-4`; D frozen | clean-mask MSE | 47 | no `t`, forward diffusion, or reverse steps in main core | same | same |
| ISIC2018 | Full | clean binary lesion mask | 4 | 1 | 10,000 steps | Adam G `2e-4`, Adam D `1e-5`; betas 0.5/0.9 | G clean-mask MSE; adversarial D with R1 | 47 | train `t` uniform over 4; test 4 reverse steps, ensemble 5 | mean validation Dice | same |
| ISIC2018 | Random-Yt | same | 4 | 1 | 10,000 steps | same | same target/loss | 47 | same `t`; random training `Y_t`; Full inference | same | same |
| ISIC2018 | Shuffle-Yt | same | 4 | 2 | 10,000 steps | same | same target/loss | 47 | same `t` and noise, peer-mask `Y_t`; Full inference | same | same |
| ISIC2018 | Core-No-Diff | same | 4 | 1 | 10,000 steps | Adam G `2e-4`; D frozen | clean-mask MSE | 47 | no `t`, forward diffusion, or reverse steps in main core | same | same |

The configuration field `num_epoch=12000` is legacy and ignored by the active step-based loop; it must not be interpreted as 12,000 completed epochs. The cosine learning-rate schedule spans 10,000 steps with a minimum of `1e-5`; validation runs every 1,000 steps. EMA is disabled. `cudnn.benchmark=false`, `cudnn.deterministic=false`, and deterministic algorithms are disabled, matching the validated local runtime.

Formal commands for all 12 rows are in the root README. The smoke matrix shortens only duration and batch size. It does not change image size, model, target, loss, timestep count, sampling loop, seed configuration, transform definitions, dtype, AMP, TF32, or checkpoint policy.
