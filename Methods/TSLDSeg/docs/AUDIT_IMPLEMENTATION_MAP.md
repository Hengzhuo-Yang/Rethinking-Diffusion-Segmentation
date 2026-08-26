# Audit implementation map

This map describes the already validated local implementation. Release assembly did
not recreate or redesign any audit. Full diffusion is the unmodified baseline; only
Random-Yt, Shuffle-Yt, and Core-No-Diff are audits. Line numbers below refer to the
final public files in this directory.

## Final implementation landmarks

| Responsibility | Final file and line | Function/class |
|---|---|---|
| Training CLI and config merge | `code/main.py:605`, `code/main.py:706-715` | module entry point |
| Exact RTX 5090 gate | `code/main.py:665` | `require_rtx5090` call |
| Training launch | `code/main.py:929-932` | `Trainer.fit` |
| Data construction, including the separate validation-metric view | `code/main.py:180-230` | `DataModuleFromConfig` |
| Resolved config and audit metadata | `code/main.py:261-335` | `SetupCallback` |
| Periodic validation metric | `code/main.py:377`, `code/main.py:478` | `ImageLogger`, `LatentDiffusion.log_dice` call |
| Mode normalization | `code/ldm/audit.py:9` | `normalize_audit_mode` |
| No-self-match permutation | `code/ldm/audit.py:18` | `_batch_derangement` |
| Random/Shuffle input construction and invariant trace | `code/ldm/audit.py:32` | `audited_y_t_input` |
| Forward diffusion | `code/ldm/models/diffusion/ddpm.py:330` | `DDPM.q_sample` |
| Audit helper integration | `code/ldm/models/diffusion/ddpm.py:335` | `DDPM.audit_training_y_t_input` |
| Main latent training path and timestep draw | `code/ldm/models/diffusion/ddpm.py:1020-1034` | `LatentDiffusion.forward` |
| Full/Random/Shuffle target and loss | `code/ldm/models/diffusion/ddpm.py:1294-1341` | `LatentDiffusion.p_losses` |
| Core input-convolution configuration | `code/ldm/models/diffusion/ddpm.py:539-556` | `LatentDiffusion._configure_core_no_diff_unet` |
| Core image-only/no-timestep forward | `code/ldm/models/diffusion/ddpm.py:1036-1119` | `_is_core_no_diff` through `_forward_core_no_diff` |
| Core clean-latent target and direct L1 loss | `code/ldm/models/diffusion/ddpm.py:1253-1269` | `_core_no_diff_direct_loss` |
| Validation/test sampler and metric path | `code/ldm/models/diffusion/ddpm.py:1549-2011` | `LatentDiffusion.log_dice` |
| Default evaluation step count | `code/ldm/models/diffusion/ddpm.py:1551-1552` | `TSLDSEG_EVAL_DDIM_STEPS`, default 10 |
| DDIM versus direct selection | `code/ldm/models/diffusion/ddpm.py:1963-1984` | nested `get_dice`, sampler selection |
| Explicit final-test CLI | `code/scripts/slice2seg.py:111-212` | `main` |
| Final-test RTX 5090 gate/model placement | `code/scripts/slice2seg.py:155`, `code/scripts/slice2seg.py:191-192` | `require_rtx5090`, `model.to(cuda:0)` |

## Condition semantics

For all diffusion-interface conditions, the model is in epsilon-prediction mode.
`LatentDiffusion.forward` samples one `t_i ~ Uniform{0,...,999}` per item at line
1024. `p_losses` samples `epsilon_i`, builds `Y_t_ref` at lines 1295-1296, chooses
the model input at line 1297, and calls the unchanged model at line 1298. The output
target is still `epsilon_i` at line 1306. The local validated loss is the fixed sum
of L1 epsilon prediction and L1 latent segmentation reconstruction: lines 1312,
1316, 1329, and 1337-1338. The ELBO term has configured weight zero.

| Condition | Target and loss | Input intervention | Timestep and denoising | Exact difference from Full | CUDA state |
|---|---|---|---|---|---|
| Full (`audit_mode=none`) | Output target `epsilon_i`; L1 epsilon + L1 reconstructed clean mask latent | None: UNet receives `Y_t_ref=q_sample(Y_0_i,t_i,epsilon_i)` | Per-item training `t_i`; DDIM-10 validation/test | Baseline; no audit branch active | Model, image latent, `Y_t_ref`, `t`, epsilon, output, and loss on `cuda:0` |
| Random-Yt (`train_random_yt`) | Identical to Full; target/loss/output semantics unchanged | Training-only UNet input becomes `torch.randn_like(Y_t_ref)` at `code/ldm/audit.py:48`; image condition is unchanged | Original `t_i` is still passed to the UNet; original epsilon target remains; DDIM-10 validation/test is unchanged | Only the training-time Y_t input tensor changes; validation/inference audit replacement is inactive | Replacement inherits shape/dtype/device from `Y_t_ref`; all model tensors and loss on `cuda:0` |
| Shuffle-Yt (`train_shuffle_yt`) | Identical to Full; target/loss/output semantics unchanged | Training-only: `_batch_derangement` selects `j != i`, then line 71 constructs `q_sample(Y_0_j,t_i,epsilon_i)` | Current sample's `t_i` and `epsilon_i` are reused; DDIM-10 validation/test is unchanged | Only the source mask used to construct training Y_t changes; batch size one fails clearly; no dataset fallback | Permutation and reconstructed Y_t are created on the current tensor device; smoke trace proved same shape/dtype/device on `cuda:0` |
| Core-No-Diff (`core_no_diff`) | Target becomes clean mask latent `Y_0`; direct L1 regression at lines 1253-1269 | Main UNet receives only the image-conditioning latent; input convolution changes 8 to 4 channels | No main-core timestep, q-sample, or reverse process; validation/test use direct inference | Structural audit: target, loss, output semantics, input channels, and sampling differ intentionally; TAM/HSEM/MCF remain | Image conditioning, 4-channel UNet, clean target, output, and loss on `cuda:0` |

The input-side audit trace records `target_changed=false`, `loss_changed=false`,
`model_output_changed=false`, `image_condition_changed=false`, and
`timestep_changed=false` at `code/ldm/audit.py:106-110`. Core-No-Diff explicitly
records the opposite structural changes at
`code/ldm/models/diffusion/ddpm.py:1091-1114`.

## Twelve executable bindings

Each row is an independent run. The mode is supplied as the OmegaConf override
`model.params.audit_mode=<mode>`; the configs retain `none` as their default at
line 14, so default Full behavior is unchanged.

| Dataset | Condition | Training entry point | Final config (key lines) | Mode override | Implementation functions (final lines) | Validation/final-test path | CUDA device |
|---|---|---|---|---|---|---|---|
| BTCV | Full | `code/main.py:605`, `:931` | `code/configs/latent-diffusion/btcv-cls2-ldm-kl-8.yaml:1-146`; data `:87-121` | `none` | `forward:1020`; `p_losses:1294`; `q_sample:330` | `log_dice:1549`, DDIM-10; `slice2seg.main:111` | `cuda:0` |
| BTCV | Random-Yt | same | same | `train_random_yt` | above + `audited_y_t_input:32` (`torch.randn_like`:48) | same DDIM-10 path | `cuda:0` |
| BTCV | Shuffle-Yt | same | same | `train_shuffle_yt` | above + `_batch_derangement:18`; shuffled q-sample `:71` | same DDIM-10 path | `cuda:0` |
| BTCV | Core-No-Diff | same | same | `core_no_diff` | `_configure_core_no_diff_unet:539`; `_forward_core_no_diff:1116`; direct loss `:1253` | `log_dice:1549`, direct; `slice2seg --audit_mode core_no_diff` | `cuda:0` |
| ACDC | Full | `code/main.py:605`, `:931` | `code/configs/latent-diffusion/acdc-cls4-ldm-kl-8.yaml:1-149`; data `:88-122` | `none` | `forward:1020`; `p_losses:1294`; `q_sample:330` | `log_dice:1549`, DDIM-10; `slice2seg.main:111` | `cuda:0` |
| ACDC | Random-Yt | same | same | `train_random_yt` | above + `audited_y_t_input:32` (`torch.randn_like`:48) | same DDIM-10 path | `cuda:0` |
| ACDC | Shuffle-Yt | same | same | `train_shuffle_yt` | above + `_batch_derangement:18`; shuffled q-sample `:71` | same DDIM-10 path | `cuda:0` |
| ACDC | Core-No-Diff | same | same | `core_no_diff` | `_configure_core_no_diff_unet:539`; `_forward_core_no_diff:1116`; direct loss `:1253` | `log_dice:1549`, direct; `slice2seg --audit_mode core_no_diff` | `cuda:0` |
| ISIC2018 | Full | `code/main.py:605`, `:931` | `code/configs/latent-diffusion/isic-ldm-kl-8.yaml:1-152`; data `:87-125` | `none` | `forward:1020`; `p_losses:1294`; `q_sample:330` | `log_dice:1549`, DDIM-10; `slice2seg.main:111` | `cuda:0` |
| ISIC2018 | Random-Yt | same | same | `train_random_yt` | above + `audited_y_t_input:32` (`torch.randn_like`:48) | same DDIM-10 path | `cuda:0` |
| ISIC2018 | Shuffle-Yt | same | same | `train_shuffle_yt` | above + `_batch_derangement:18`; shuffled q-sample `:71` | same DDIM-10 path | `cuda:0` |
| ISIC2018 | Core-No-Diff | same | same | `core_no_diff` | `_configure_core_no_diff_unet:539`; `_forward_core_no_diff:1116`; direct loss `:1253` | `log_dice:1549`, direct; `slice2seg --audit_mode core_no_diff` | `cuda:0` |

## Auxiliary-module preservation

The condition encoder instantiates TAM, HSEM, and MCF at
`code/ldm/modules/encoders/modules.py:21-33` and calls all three at lines 39-43.
No audit disables these branches, EMA, or evaluation post-processing. Forward hooks
in `tests/gpu_smoke_12.py` confirmed TAM, HSEM, MCF, and UNet execution in every one
of the 12 bindings. Full/Random/Shuffle have identical parameter counts per dataset;
Core-No-Diff has exactly 6,912 fewer parameters solely because its validated
structural audit changes the first UNet convolution from 8 to 4 channels.

## Interpretation boundary

Robustness to an input-side audit indicates weak dependence on that particular
case-conditioned training input under the tested objective. A large Shuffle-Yt drop
for an epsilon-prediction method can instead be positive-control evidence for
dependence on the original `(Y_t,t,epsilon)` coupling. Neither outcome establishes
that auxiliary modules are useless or that reverse diffusion is generally
unnecessary.
