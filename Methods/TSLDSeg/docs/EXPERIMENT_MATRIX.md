# Experiment matrix

This document is the complete configuration record for the 3 datasets x 4
conditions. Full is the baseline; Random-Yt, Shuffle-Yt, and Core-No-Diff are the
three audits. Every cell is trained independently and selects its own checkpoint
from validation only.

## Settings shared by all 12 cells

| Field | Formal setting | Smoke-only setting/qualification |
|---|---|---|
| Spatial input | 256 x 256 pixels; no random crop | unchanged |
| Formal batch size | 12 | training batch 2; validation/test batch 1 |
| Epoch/length | No fixed epoch count; step-based `max_steps=50000` | exactly one optimizer step per cell |
| Optimizer | AdamW; default weight decay 0.01, betas (0.9, 0.999), epsilon 1e-8 | formal AdamW step executed |
| Learning-rate schedule | LambdaLR each step; 5,000-step multiplier warm-up from 1e-6 to 1, then effective constant multiplier 1 | one scheduler step executed |
| Seed | 23 | 23 |
| Diffusion schedule | 1,000 steps; linear beta schedule 0.0015 to 0.0155 | unchanged |
| Validation cadence | `val_avg_dice` every 2,000 training steps | one evaluation call after the training step |
| Checkpoint rule | maximize `val_avg_dice`, `save_top_k=1`, no `last` checkpoint | one temporary best save, strict reload, then deletion |
| Evaluation metrics | foreground Dice and IoU; background excluded | execution-only Dice/IoU, not performance evidence |
| Device | NVIDIA GeForce RTX 5090, `cuda:0`; wrong/missing GPU fails | exact same device gate |
| Training dtype/AMP | Lightning default FP32; AMP is not enabled for training | FP32 training |
| Validation/test dtype | model parameters FP32; `log_dice` runs under CUDA autocast | same |
| TF32 | `cuda.matmul.allow_tf32=false`; `cudnn.allow_tf32=true` | same validated process defaults |
| cuDNN | `benchmark=true`; `deterministic=false` | benchmark enabled |
| EMA | enabled and used for validation/test | EMA update and EMA inference executed |
| Gradient accumulation | 1 | 1 |
| Pretrained inputs | LSUN-churches latent-diffusion checkpoint and KL-f8 VAE checkpoint from explicit environment variables | same local trusted checkpoints |

The configured base learning rate is 1e-5 and `--scale_lr False` is mandatory in
the commands below. ACDC has a real dataset-specific optimizer exception: its
class-label embedding parameter group uses 100 times the base rate (1e-3), while
the UNet and trainable condition encoder use 1e-5. BTCV and ISIC2018 have no
class-label embedding group and use 1e-5 throughout. This difference comes from
`LatentDiffusion.configure_optimizers` and is not normalized across datasets.

## Dataset profiles

| Dataset | Config | Target/task and class mapping | Size/crop and preprocessing | Augmentation | Normalization | Evaluation post-processing |
|---|---|---|---|---|---|---|
| BTCV/Synapse | `configs/latent-diffusion/btcv-cls2-ldm-kl-8.yaml` | Binary: 0 background, 1 foreground. Raw labels are mapped to the Synapse eight-organ subset, then collapsed to foreground/background. | NIfTI axial slices resized bilinearly to 256 x 256; masks nearest-neighbor; no crop. Default CT window [-175, 250]. Loader requires the prepared size. | Paired horizontal flip p=0.5 and vertical flip p=0.5, training only. | CT window to uint8 PNG, then RGB image to [-1,1]. Training/validation mask to [-1,1]; metric/test mask in {0,1}. | Decode latent; binary probability threshold 0.5. Report non-empty-foreground slice Dice/IoU; background excluded. |
| ACDC | `configs/latent-diffusion/acdc-cls4-ldm-kl-8.yaml` | Four classes: 0 background, 1 right ventricle, 2 myocardium, 3 left ventricle. Training samples a present class and uses a binary class-conditioned mask; evaluation reconstructs all four labels. | Labelled NIfTI frames resized bilinearly to 256 x 256; label map nearest-neighbor; no crop. Intensities clipped to foreground percentiles [1,99]. | Paired horizontal flip p=0.5 and vertical flip p=0.5, training only. | Preprocessed image to uint8 then RGB [-1,1]. Selected binary training/validation mask to [-1,1]; metric/test label map in {0,1,2,3}. | One prediction per class, class scores combined with softmax and argmax. Report class 1-3 Dice/IoU and foreground mean; background excluded. |
| ISIC2018 | `configs/latent-diffusion/isic-ldm-kl-8.yaml` | Task 1 binary lesion segmentation: 0 background, 1 lesion. | Official RGB/mask PNGs resized to 256 x 256 (bilinear image, nearest mask); no crop. | Paired horizontal flip p=0.5 and vertical flip p=0.5, training only. | RGB uint8 to [-1,1]. Mask threshold >127, training/validation to [-1,1], metric/test mask in {0,1}. | Decode latent; binary probability threshold 0.5. Report non-empty-lesion image Dice/IoU; background excluded. |

## Condition profiles

| Condition | Model-output target | Formal loss | Training intervention | Timestep | Validation/final sampling |
|---|---|---|---|---|---|
| Full | epsilon/noise | L1(output, epsilon) + L1(reconstructed clean mask latent, clean mask latent); ELBO weight 0 | none; model receives reference noisy mask latent | per-item uniform integer in [0,999] | DDIM, 10 steps |
| Random-Yt | same epsilon and same loss as Full | unchanged | replace only training-time Y_t model input with independent `randn_like`; image, t, target, and reference latent reconstruction remain unchanged | sampled and passed exactly as Full | unchanged DDIM, 10 steps |
| Shuffle-Yt | same epsilon and same loss as Full | unchanged | derange clean mask latents in-batch, then q-sample with current item's t and epsilon; no self-match; formal/smoke batch is greater than one | current item's t reused | unchanged DDIM, 10 steps |
| Core-No-Diff | clean mask latent Y_0 | direct L1(output, clean mask latent) | image-conditioning latent only; main UNet input 8 to 4 channels; TAM/HSEM/MCF retained | absent from main core | direct; no reverse-diffusion step count |

## Twelve formal training bindings and commands

Run from `code/` in the validated environment after setting the asset/data
environment variables described in `REPRODUCTION.md`. These are GPU commands:
each config selects GPU 0, and `main.py` refuses to run unless `cuda:0` is the
RTX 5090. `--no-test True` deliberately prevents test use during training and
checkpoint selection.

| Dataset | Condition | Target profile | Size/profile | Batch formal/smoke | Steps | LR | Sampling | Checkpoint/evaluation | Actual training command |
|---|---|---|---|---|---:|---|---|---|---|
| BTCV | Full | Full | BTCV, 256, no crop | 12 / 2 | 50000 | 1e-5 | DDIM-10 | max val Dice; Dice/IoU | `python main.py --base configs\latent-diffusion\btcv-cls2-ldm-kl-8.yaml --name btcv_full_s23 --logdir ..\runs --seed 23 --scale_lr False --no-test True -t model.params.audit_mode=none` |
| BTCV | Random-Yt | Random-Yt | BTCV, 256, no crop | 12 / 2 | 50000 | 1e-5 | DDIM-10 | max val Dice; Dice/IoU | `python main.py --base configs\latent-diffusion\btcv-cls2-ldm-kl-8.yaml --name btcv_random_yt_s23 --logdir ..\runs --seed 23 --scale_lr False --no-test True -t model.params.audit_mode=train_random_yt` |
| BTCV | Shuffle-Yt | Shuffle-Yt | BTCV, 256, no crop | 12 / 2 | 50000 | 1e-5 | DDIM-10 | max val Dice; Dice/IoU | `python main.py --base configs\latent-diffusion\btcv-cls2-ldm-kl-8.yaml --name btcv_shuffle_yt_s23 --logdir ..\runs --seed 23 --scale_lr False --no-test True -t model.params.audit_mode=train_shuffle_yt` |
| BTCV | Core-No-Diff | Core-No-Diff | BTCV, 256, no crop | 12 / 2 | 50000 | 1e-5 | direct | max val Dice; Dice/IoU | `python main.py --base configs\latent-diffusion\btcv-cls2-ldm-kl-8.yaml --name btcv_core_no_diff_s23 --logdir ..\runs --seed 23 --scale_lr False --no-test True -t model.params.audit_mode=core_no_diff` |
| ACDC | Full | Full | ACDC, 256, no crop | 12 / 2 | 50000 | 1e-5; label embedding 1e-3 | DDIM-10 | max val Dice; Dice/IoU | `python main.py --base configs\latent-diffusion\acdc-cls4-ldm-kl-8.yaml --name acdc_full_s23 --logdir ..\runs --seed 23 --scale_lr False --no-test True -t model.params.audit_mode=none` |
| ACDC | Random-Yt | Random-Yt | ACDC, 256, no crop | 12 / 2 | 50000 | 1e-5; label embedding 1e-3 | DDIM-10 | max val Dice; Dice/IoU | `python main.py --base configs\latent-diffusion\acdc-cls4-ldm-kl-8.yaml --name acdc_random_yt_s23 --logdir ..\runs --seed 23 --scale_lr False --no-test True -t model.params.audit_mode=train_random_yt` |
| ACDC | Shuffle-Yt | Shuffle-Yt | ACDC, 256, no crop | 12 / 2 | 50000 | 1e-5; label embedding 1e-3 | DDIM-10 | max val Dice; Dice/IoU | `python main.py --base configs\latent-diffusion\acdc-cls4-ldm-kl-8.yaml --name acdc_shuffle_yt_s23 --logdir ..\runs --seed 23 --scale_lr False --no-test True -t model.params.audit_mode=train_shuffle_yt` |
| ACDC | Core-No-Diff | Core-No-Diff | ACDC, 256, no crop | 12 / 2 | 50000 | 1e-5; label embedding 1e-3 | direct | max val Dice; Dice/IoU | `python main.py --base configs\latent-diffusion\acdc-cls4-ldm-kl-8.yaml --name acdc_core_no_diff_s23 --logdir ..\runs --seed 23 --scale_lr False --no-test True -t model.params.audit_mode=core_no_diff` |
| ISIC2018 | Full | Full | ISIC, 256, no crop | 12 / 2 | 50000 | 1e-5 | DDIM-10 | max val Dice; Dice/IoU | `python main.py --base configs\latent-diffusion\isic-ldm-kl-8.yaml --name isic2018_full_s23 --logdir ..\runs --seed 23 --scale_lr False --no-test True -t model.params.audit_mode=none` |
| ISIC2018 | Random-Yt | Random-Yt | ISIC, 256, no crop | 12 / 2 | 50000 | 1e-5 | DDIM-10 | max val Dice; Dice/IoU | `python main.py --base configs\latent-diffusion\isic-ldm-kl-8.yaml --name isic2018_random_yt_s23 --logdir ..\runs --seed 23 --scale_lr False --no-test True -t model.params.audit_mode=train_random_yt` |
| ISIC2018 | Shuffle-Yt | Shuffle-Yt | ISIC, 256, no crop | 12 / 2 | 50000 | 1e-5 | DDIM-10 | max val Dice; Dice/IoU | `python main.py --base configs\latent-diffusion\isic-ldm-kl-8.yaml --name isic2018_shuffle_yt_s23 --logdir ..\runs --seed 23 --scale_lr False --no-test True -t model.params.audit_mode=train_shuffle_yt` |
| ISIC2018 | Core-No-Diff | Core-No-Diff | ISIC, 256, no crop | 12 / 2 | 50000 | 1e-5 | direct | max val Dice; Dice/IoU | `python main.py --base configs\latent-diffusion\isic-ldm-kl-8.yaml --name isic2018_core_no_diff_s23 --logdir ..\runs --seed 23 --scale_lr False --no-test True -t model.params.audit_mode=core_no_diff` |

The target/loss, class mapping, augmentation, normalization, post-processing,
device, dtype, AMP/TF32, and seed for each row are given by its named condition
and dataset profile plus the shared table; there are no undocumented per-cell
overrides.

## Validation selection and final test commands

The selected checkpoint is the single `best-*.ckpt` under that cell's timestamped
run `checkpoints/` directory. It is selected only from the configured val manifest.
After training, use `scripts/promote_best_checkpoint.py` if a stable filename is
needed, then run `scripts/slice2seg.py` against the fixed test manifest.

Diffusion cells use this dataset binding with `--sampler ddim --ddim_steps 10`:

```powershell
# BTCV Full/Random-Yt/Shuffle-Yt: substitute that cell's validation-selected checkpoint/output name.
python scripts\slice2seg.py --dataset btcv --data_dir $env:TSLDSEG_BTCV_PREPROCESSED_ROOT\BTCV\test --manifest manifests\btcv\test.txt --config configs\latent-diffusion\btcv-cls2-ldm-kl-8.yaml --ckpt X:\runs\<cell>\checkpoints\best-<step>-<dice>.ckpt --outdir X:\results\<cell> --sampler ddim --ddim_steps 10 --num_classes 2 --seed 23 --save_results

# ACDC Full/Random-Yt/Shuffle-Yt.
python scripts\slice2seg.py --dataset acdc --data_dir $env:TSLDSEG_ACDC_PREPROCESSED_ROOT\ACDC\testing --manifest manifests\acdc\test.txt --config configs\latent-diffusion\acdc-cls4-ldm-kl-8.yaml --ckpt X:\runs\<cell>\checkpoints\best-<step>-<dice>.ckpt --outdir X:\results\<cell> --sampler ddim --ddim_steps 10 --num_classes 4 --seed 23 --save_results

# ISIC2018 Full/Random-Yt/Shuffle-Yt.
python scripts\slice2seg.py --dataset isic2018 --data_dir $env:TSLDSEG_ISIC2018_PREPROCESSED_ROOT\ISIC18\testing --manifest manifests\isic2018\test.txt --config configs\latent-diffusion\isic-ldm-kl-8.yaml --ckpt X:\runs\<cell>\checkpoints\best-<step>-<dice>.ckpt --outdir X:\results\<cell> --sampler ddim --ddim_steps 10 --num_classes 2 --seed 23 --save_results
```

Core-No-Diff uses the same dataset-specific command but replaces the sampler
arguments with `--sampler direct --audit_mode core_no_diff`; it does not accept a
reverse-diffusion interpretation. Random-Yt and Shuffle-Yt do not pass an inference
override because their interventions are training-only and inference is identical
to Full.
