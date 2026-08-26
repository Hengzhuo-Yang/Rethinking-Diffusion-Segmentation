# Experiment Matrix

This release exposes exactly 12 primary experiments: three datasets crossed
with four conditions. The conditions share data splits, optimizer settings,
validation policy, and final-test policy.

## Conditions

| Public name | Internal audit_mode | Main-path behavior |
| --- | --- | --- |
| Full | none | Standard mask diffusion input and diffusion objective |
| Random-Yt | train_random_yt | During training only, replace the model's Yt input with independent standard Gaussian noise; retain the original timestep, epsilon target, loss, validation, and sampling |
| Shuffle-Yt | train_shuffle_yt | During training only, construct Yt from a batch-deranged mask using the current sample's timestep and noise; retain the original target and loss |
| Core-No-Diff | core_no_diff | Image-only discriminative core with direct segmentation logits and Dice plus BCE/CE loss; no Yt, timestep embedding, q_sample, or reverse sampler in the main path |

Core-No-Diff is a discriminative-capacity counterfactual, not an
objective-preserving ablation. Random-Yt and Shuffle-Yt preserve the original
diffusion training target and loss.

## Dataset semantics

| Dataset | Target and class mapping | Input / crop | Normalization | Augmentation | Final post-processing and reported metrics |
| --- | --- | --- | --- | --- | --- |
| BTCV | One binary channel after the local `transfer_to_9` remapping and foreground merge; model classes background/foreground | 1-channel CT; preprocessed to 224×224; model configuration `image_size=256` | Default `sdseg_2d`: clip to [-175, 250], then `2 * ((value + 125) / 400) - 1`; no loader-time statistic fitting | None in the formal loader | Mean five diffusion scores then threshold at 0.5; Core uses one direct prediction; per-slice Dice/IoU with explicitly reported non-empty-GT policy |
| ACDC | Four one-hot channels: background, right ventricle, myocardium, left ventricle | 1-channel MRI; preprocessed to 224×224; model configuration `image_size=256` | Per labelled ED/ES frame, nonzero 1st–99th percentile clip, scale to [-1, 1] | None in the formal loader | Mean scores, argmax, regroup slices by subject/frame; per-volume foreground-class Dice/IoU and their mean |
| ISIC2018 | One binary lesion channel; model classes background/lesion | 3-channel RGB; preprocessed to 224×224; model configuration `image_size=256` | RGB uint8 scaled to [-1, 1] | None in the formal loader | Mean five diffusion scores then threshold at 0.5; Core uses one direct prediction; per-image lesion Dice/IoU |

No cross-sample normalization statistic is learned from validation or test.
Resizing and normalization occur in fixed preprocessing; loaders only enforce
shape, center-crop inputs larger than 224, convert targets, and apply manifests.

## Formal settings shared without forcing dataset outputs to match

| Field | Formal value |
| --- | --- |
| Model | Width 128, 2 residual blocks, channel multipliers 1,1,2,2,4,4, 1 attention head at resolution 16, learned variance for diffusion conditions |
| Optimizer | AdamW, learning rate 1e-4, weight decay 0 |
| Training batch / microbatch | 8 / -1 (no split); retained from the local formal configuration |
| Epoch | Not applicable: this implementation is iteration/step based |
| Schedule | Uniform timestep sampling over the 1,000-step linear diffusion process |
| Save / validation interval | Every 5,000 optimizer steps |
| EMA | 0.9999 restart artifact; `best_model.pt` is the validation-selected live model state |
| Checkpoint metric | Mean validation foreground Dice over non-empty ground-truth class observations |
| Seed | Training 10; validation 10; final inference 10 |
| Device | NVIDIA GeForce RTX 5090 exposed as logical `cuda:0`; fail closed otherwise |
| Numeric mode | `torch.float32`; `use_fp16=False`; no autocast or GradScaler |
| Runtime flags | matmul TF32 false; cuDNN TF32 true; cuDNN benchmark false; cuDNN deterministic false; deterministic algorithms false |

Explicitly seeding Full is a release-protocol correction and is not a claim
that an earlier unseeded private trajectory can be reproduced bit for bit.

## Twelve entries

“Epoch” is N/A in every row because termination is by optimizer step. Smoke
batches are acceptance-only overrides and never replace formal batch 8;
Shuffle-Yt requires two samples so its derangement cannot become an identity.

| Dataset | Condition | Target / loss | Formal steps | Formal train / val / test batch | Smoke batch | Validation path | Final inference / ensemble |
| --- | --- | --- | ---: | --- | ---: | --- | --- |
| BTCV | Full | epsilon / diffusion MSE + learned-variance VB | 40,000 | 8 / 8 / 8 | 1 | 100-step DDPM, ensemble 1 | 1,000-step DDPM / 5 |
| BTCV | Random-Yt | epsilon / unchanged diffusion MSE + VB | 60,000 | 8 / 8 / 8 | 1 | 100-step DDPM, ensemble 1 | 1,000-step DDPM / 5 |
| BTCV | Shuffle-Yt | epsilon / unchanged diffusion MSE + VB | 60,000 | 8 / 64 / 8 | 2 | 100-step DDPM, ensemble 1 | 1,000-step DDPM / 5 |
| BTCV | Core-No-Diff | binary mask / foreground Dice + 2-class CE | 60,000 | 8 / 64 / 8 | 1 | direct forward, 0 diffusion steps | direct forward / 1 |
| ACDC | Full | epsilon / diffusion MSE + learned-variance VB | 60,000 | 8 / 64 / 32 | 1 | 100-step DDPM, ensemble 1 | 1,000-step DDPM / 5 |
| ACDC | Random-Yt | epsilon / unchanged diffusion MSE + VB | 60,000 | 8 / 32 / 32 | 1 | 100-step DDPM, ensemble 1 | 1,000-step DDPM / 5 |
| ACDC | Shuffle-Yt | epsilon / unchanged diffusion MSE + VB | 60,000 | 8 / 32 / 32 | 2 | 100-step DDPM, ensemble 1 | 1,000-step DDPM / 5 |
| ACDC | Core-No-Diff | four-class mask / Dice + CE | 60,000 | 8 / 32 / 32 | 1 | direct forward, 0 diffusion steps | direct forward / 1 |
| ISIC2018 | Full | epsilon / diffusion MSE + learned-variance VB | 60,000 | 8 / 32 / 16 | 1 | 100-step DDPM, ensemble 1 | 1,000-step DDPM / 5 |
| ISIC2018 | Random-Yt | epsilon / unchanged diffusion MSE + VB | 60,000 | 8 / 32 / 16 | 1 | 100-step DDPM, ensemble 1 | 1,000-step DDPM / 5 |
| ISIC2018 | Shuffle-Yt | epsilon / unchanged diffusion MSE + VB | 60,000 | 8 / 32 / 16 | 2 | 100-step DDPM, ensemble 1 | 1,000-step DDPM / 5 |
| ISIC2018 | Core-No-Diff | binary mask / foreground Dice + 2-class CE | 60,000 | 8 / 32 / 16 | 1 | direct forward, 0 diffusion steps | direct forward / 1 |

Checkpoints and validation run every 5,000 steps. Diffusion validation uses a
100-step DDPM schedule and one sample; final testing uses the full 1,000-step
DDPM schedule and five samples. Core-No-Diff validation and final testing use
one direct forward prediction, so their main-path sampling-step count is zero.

## Actual GPU commands

These are the formal commands represented by the table. Each action invokes
the exact RTX 5090 preflight before doing work.

    python code/scripts/release_pipeline.py run --dataset btcv --condition full --data-root prepared/btcv --runs-root runs --cuda-device 0
    python code/scripts/release_pipeline.py run --dataset btcv --condition random-yt --data-root prepared/btcv --runs-root runs --cuda-device 0
    python code/scripts/release_pipeline.py run --dataset btcv --condition shuffle-yt --data-root prepared/btcv --runs-root runs --cuda-device 0
    python code/scripts/release_pipeline.py run --dataset btcv --condition core-no-diff --data-root prepared/btcv --runs-root runs --cuda-device 0
    python code/scripts/release_pipeline.py run --dataset acdc --condition full --data-root prepared/acdc --runs-root runs --cuda-device 0
    python code/scripts/release_pipeline.py run --dataset acdc --condition random-yt --data-root prepared/acdc --runs-root runs --cuda-device 0
    python code/scripts/release_pipeline.py run --dataset acdc --condition shuffle-yt --data-root prepared/acdc --runs-root runs --cuda-device 0
    python code/scripts/release_pipeline.py run --dataset acdc --condition core-no-diff --data-root prepared/acdc --runs-root runs --cuda-device 0
    python code/scripts/release_pipeline.py run --dataset isic2018 --condition full --data-root prepared/isic2018 --runs-root runs --cuda-device 0
    python code/scripts/release_pipeline.py run --dataset isic2018 --condition random-yt --data-root prepared/isic2018 --runs-root runs --cuda-device 0
    python code/scripts/release_pipeline.py run --dataset isic2018 --condition shuffle-yt --data-root prepared/isic2018 --runs-root runs --cuda-device 0
    python code/scripts/release_pipeline.py run --dataset isic2018 --condition core-no-diff --data-root prepared/isic2018 --runs-root runs --cuda-device 0

Use `train` and `test` separately when an explicit review boundary is desired
before the one-time final test. `--dry-run` prints the preflight, train, sample,
and evaluation command graph without probing data/GPU or writing output.

## Evaluation

- BTCV reports binary slice metrics, including an explicitly labeled
  non-empty-ground-truth policy.
- ACDC groups slices by subject/frame and reports foreground-class volume
  metrics.
- ISIC2018 reports per-image and aggregate binary lesion metrics.

This source release includes no private-run metrics and makes no corrected
final-test performance claim. New results must come from the manifest-locked
pipeline and retain their best-checkpoint and final-test request metadata.

The separate acceptance run used one optimizer step and one validation/test
sample per row while retaining the formal architecture, 224×224 tensors, loss,
timestep policies, validation DDPM 100, final DDPM 1,000, ensemble policy,
seed, FP32, AMP, and TF32 state. All 12 rows passed on the RTX 5090; exact CUDA
evidence and peak memory are in `docs/GPU_SMOKE_TEST.md`.
