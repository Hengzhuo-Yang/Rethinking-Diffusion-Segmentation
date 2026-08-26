# Modifications from upstream

This file distinguishes method/audit work, RTX 5090 compatibility work, dataset extensions, and release-only safety corrections. It does not claim environment compatibility changes as methodological contributions.

## Controlled audit implementation

- Added explicit `none`, `train_random_yt`, `train_shuffle_yt`, and `core_no_diff` modes.
- Random-Yt changes only the training generator's noisy-mask input; clean target, MSE loss, image, timestep, validation, and inference are unchanged.
- Shuffle-Yt uses a batch derangement and reconstructs the training noisy-mask input from another sample's clean mask with the current timestep and original diffusion noise. It requires training batch size greater than one.
- Core-No-Diff adds a generator path driven by image-condition features and the paper's non-diffusion latent `z`. It removes noisy-mask state, timestep conditioning, discriminator-derived attention, forward diffusion, and reverse diffusion from the main path while retaining clean-mask MSE.
- Full remains the baseline and is not described as an audit.

## Dataset and evaluation extensions

- Added BTCV/Synapse binary 2D, ACDC three-foreground-channel multi-class 2D, and ISIC2018 Task 1 binary 2D loaders, preprocessors, metrics, configurations, and final-test entries.
- Added step-based 10,000-step training, validation intervals, per-step validation history, and best-checkpoint metadata.
- Added mean foreground Dice/IoU evaluation. Empty-ground-truth slice observations are skipped consistently across conditions; ACDC averages foreground classes 1–3 and excludes background.
- Added fixed manifests and manifest-enforced loaders. No condition or seed can alter a split.

## Historical split and role corrections in this release

- BTCV `test_10pct_seed23` fallback and its runner are excluded. The fixed validation cases are `case0008` and `case0001`; the other 10 held-out cases are test.
- ACDC validation and ISIC2018 validation are used only for checkpoint selection. Legacy cloud wrappers that treated them as fast final tests are excluded.
- Formal final-test entries accept only a test alias, require a test manifest, require the selected checkpoint's metadata, and verify that the checkpoint was chosen from validation.
- Checkpoint metadata records the validation manifest SHA-256 and a relative checkpoint filename.
- Evaluation visualization output is disabled by default so medical images are not written unless explicitly requested; metric computation is unchanged.

## RTX 5090 and modern PyTorch compatibility

- `score_sde/op/fused_act.py` and `score_sde/op/upfirdn2d.py` catch unavailable JIT extension builds and execute the original mathematical operations through PyTorch on the input CUDA device. On the validated Windows system both optional extensions used this CUDA-resident fallback.
- `preprocess_dataset/transforms.py` retains the local multi-channel-mask tensor handling required for ACDC. The unchanged augmentation path was validated with the installed torch/torchvision versions.
- Device index parsing and distributed helpers were updated locally.
- Release formal entries now require the exact GPU name `NVIDIA GeForce RTX 5090`; CUDA unavailability or another device is an error. The earlier silent CPU path is removed.
- An unused global `cuda-if-available-else-cpu` declaration in `score_sde/distribution.py` was removed so the public source contains no latent model-device fallback.
- No AMP, GradScaler, dtype, TF32, cuDNN benchmark, or deterministic setting was changed for the release. The actual runtime values are documented in `docs/GPU_ENVIRONMENT.md`.

## Release-only changes

- Replaced local or cloud absolute paths with repository-relative defaults and CLI arguments.
- Added `--output_dir`, fixed manifest arguments, a formal condition runner, split validator, GPU preflight, and 12-combination smoke runner.
- Preserved the official README unchanged under `code/README.md`; added a release-specific root README.
- Excluded data, weights, paper material, logs, caches, cloud archives, and obsolete experiment wrappers.
- Added license inventory, provenance tables, environment reconstruction files, and third-party notices.

## Parameter comparison with official configurations

The official repository has no BTCV, ACDC, or ISIC2018 configuration, so there is no same-dataset upstream parameter set to reproduce. The local target configurations are closest to upstream `parameters_lung.json` and retain its architecture (`cond_enc_layers=3`, `cond_enc_num_res_blocks=2`, `num_channels_dae=32`, `ch_mult=[1,1,2,2,4,4]`, one residual block, attention at 16, `attn_scale=2`, `lr_g=2e-4`, `lr_d=1e-5`, no EMA, and no lazy regularization). Non-batch differences are:

- `num_timesteps=4` instead of the current upstream CXR file's 2.
- `seed=47` is retained from the upstream CXR configuration, while the upstream MoNuSeg file uses 1024.
- Training is bounded by `max_train_steps=10000`, with validation every 1,000 steps and cosine decay over 10,000 steps; these step-based fields are local extensions. `num_epoch=12000` and `T_max=500` remain legacy metadata and are ignored by the active loop.
- ACDC requires three output mask channels and six discriminator input channels; BTCV and ISIC2018 use one and two respectively.
- The fixed dataset roots, manifests, class mappings, metric policy, checkpoint policy, and five-sample ensemble are local target-dataset settings.

Formal batch size 4 comes from the locally completed runs for all 12 target combinations and was not retuned. Smoke-only batch sizes do not replace it.
