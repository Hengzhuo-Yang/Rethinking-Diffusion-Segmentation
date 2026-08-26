# Audit implementation map

Line numbers below refer to the final public files in this release. The three audit definitions were recovered from the already validated local code. External design notes were used only to check interpretation; they did not replace the implementation.

## Shared implementation points

| Mechanism | Final implementation |
|---|---|
| Audit value validation | `code/audit_modes.py:7`, `normalize_audit_mode` |
| Full forward-diffusion pair | `code/train_cDal_monu_and_lung.py:220`, `q_sample_pairs` |
| Full reverse sampling | `code/train_cDal_monu_and_lung.py:289`, `sample_from_model` |
| Generator Full forward | `code/score_sde/models/ncsnpp_generator_adagn.py:369`, `NCSNpp.forward` |
| Clean-mask MSE | `code/train_cDal_monu_and_lung.py:656`; Core-No-Diff equivalent at line 531 |
| Random-Yt replacement | `code/audit_modes.py:74`, `maybe_replace_training_yt`; replacement at line 76; called at training lines 604 and 653 |
| Shuffle derangement | `code/audit_modes.py:21`, `derange_batch_indices`; `shuffle_batch_tensor` at line 32 |
| Shuffle diffusion reconstruction | `code/audit_modes.py:54`, `construct_training_yt_from_y0`; called at training lines 588 and 636 |
| Core-No-Diff training branch | `code/train_cDal_monu_and_lung.py:525`, `train` |
| Core-No-Diff generator path | `code/score_sde/models/ncsnpp_generator_adagn.py:568`, `NCSNpp.forward_core_no_diff` |
| Core-No-Diff validation/test path | `code/train_cDal_monu_and_lung.py:291`, `sample_from_model` |
| Validation metric | `code/metrics.py:222`, `sampling_major_vote_func` |
| Best-checkpoint save | `code/train_cDal_monu_and_lung.py:345`, `save_best_checkpoint`; called after validation at line 734 |
| Final-test selection check | `code/release_validation.py:14`, `validate_best_checkpoint` |
| Required CUDA device | `code/utils.py:41`, `dev` |

## Dataset targets

| Dataset | Target | Loader and class mapping | Final-test entry |
|---|---|---|---|
| BTCV/Synapse | Clean one-channel binary foreground mask in `[-1,1]`; the eight mapped organs are collapsed to foreground | `code/preprocess_dataset/BTCV.py:30`, `BTCVDataset` | `code/evaluate_btcv_cdal.py:71`, `main` |
| ACDC | Clean three-channel foreground mask in `[-1,1]`: RV, myocardium, LV; background is all channels absent | `code/preprocess_dataset/ACDC.py:36`, `ACDCDataset` | `code/evaluate_acdc_cdal.py:70`, `main` |
| ISIC2018 | Clean one-channel binary lesion mask in `[-1,1]` | `code/preprocess_dataset/ISIC2018.py:34`, `ISIC2018Dataset` | `code/evaluate_isic2018_cdal.py:70`, `main` |

## Twelve dataset-condition mappings

| Dataset | Condition | Configuration | Entry point | Input intervention | Timestep and denoising state | Exact difference from Full | CUDA placement |
|---|---|---|---|---|---|---|---|
| BTCV | Full | `parameters_btcv.json`, `audit_mode=none` | `train_cDal_monu_and_lung.py:391` | None; generator receives cDAL attention-weighted noisy mask, image, `t`, and `z` | Uniform training `t` over four steps; four-step reverse sampling, ensemble 5 | Baseline | Mask target, image, model, output, MSE, backward, validation, and test are asserted on `cuda:0` |
| BTCV | Random-Yt | same file, `train_random_yt` | same | Training generator `Y_t` becomes independent `torch.randn_like(Y_t)` | Current `t` is unchanged; validation/test denoising is Full | Only training generator noisy-mask input | same |
| BTCV | Shuffle-Yt | same file, `train_shuffle_yt` | same | Training generator `Y_t` is rebuilt from a deranged peer mask with current sample's diffusion noise | Current `t` and noise retained; validation/test denoising is Full | Only source mask used to construct training `Y_t`; clean target stays current sample | same |
| BTCV | Core-No-Diff | same file, `core_no_diff` | same | Main generator receives image condition plus non-diffusion `z` | No `Y_t`, `t`, forward diffusion, attention, or reverse sampling in main core | Structural counterfactual; target and clean-mask MSE retained | same |
| ACDC | Full | `parameters_acdc.json`, `audit_mode=none` | `train_cDal_monu_and_lung.py:391` | None | Uniform `t` over four steps; four-step reverse sampling, ensemble 5 | Baseline | All three mask channels and core tensors on `cuda:0` |
| ACDC | Random-Yt | same file, `train_random_yt` | same | Same Random-Yt intervention applied to the three-channel noisy state | `t` unchanged; Full validation/test | Only training `Y_t` | same |
| ACDC | Shuffle-Yt | same file, `train_shuffle_yt` | same | Three-channel peer mask is deranged and diffused with current `t`/noise | `t`/noise unchanged; Full validation/test | Only training `Y_t` source | same |
| ACDC | Core-No-Diff | same file, `core_no_diff` | same | Image condition plus `z`; no diffusion state | No diffusion/reverse main path | Structural counterfactual; three-channel clean target and MSE retained | same |
| ISIC2018 | Full | `parameters_isic2018.json`, `audit_mode=none` | `train_cDal_monu_and_lung.py:391` | None | Uniform `t` over four steps; four-step reverse sampling, ensemble 5 | Baseline | Lesion target and core tensors on `cuda:0` |
| ISIC2018 | Random-Yt | same file, `train_random_yt` | same | Independent random training `Y_t` | `t` unchanged; Full validation/test | Only training `Y_t` | same |
| ISIC2018 | Shuffle-Yt | same file, `train_shuffle_yt` | same | Peer lesion mask is deranged and diffused with current `t`/noise | `t`/noise unchanged; Full validation/test | Only training `Y_t` source | same |
| ISIC2018 | Core-No-Diff | same file, `core_no_diff` | same | Image condition plus `z`; no diffusion state | No diffusion/reverse main path | Structural counterfactual; clean lesion target and MSE retained | same |

Random-Yt and Shuffle-Yt are training-only audits by definition. Their validation and test paths deliberately remain Full. Core-No-Diff changes both training and inference because retaining the reverse diffusion loop would reintroduce the mechanism it removes.

The released code contains no separate audit, sweep, configuration, or entry point for comparing different sampling-step counts. The fixed four-step sampling loop remains because it is required by Full and the two training-only audits.
