# Audit Implementation Map

## Shared contract

`code/scripts/release_pipeline.py` maps each public condition to one YAML file
and verifies that its `audit_mode` matches the requested condition. All formal
configs retain the same seed, optimizer settings, training length, validation
cadence, BF16 policy, EMA policy, prediction type, and split roles.

| Condition | `audit_mode` | Main training implementation | Evaluation path |
| --- | --- | --- | --- |
| Full | `none` | Main diffusion branch in `code/train.py` | `LeafPipeline` |
| Random-Yt | `train_random_yt` | `random_yt_like` and the main training loop | Same diffusion evaluator as Full |
| Shuffle-Yt | `train_shuffle_yt` | `derangement_like_permutation` and shuffled-latent noising in the main loop | Same diffusion evaluator as Full |
| Core-No-Diff | `core_no_diff` | `train_core_no_diff` and `CoreNoDiffSegmentor` | Dedicated image-only evaluator branch |

`replace_unet_input` is the shared diffusion-model initialization step that
adapts the U-Net input convolution from four to eight channels. It is not a
Shuffle-Yt intervention.

Every audit condition writes `audit_metadata.json`. Every successful formal
training run must write `best_checkpoint.json` from validation Dice selection.

## Full

Full is the reference LEAF path, not an audit intervention.

- `audit_mode: none` appears in the three base YAML files.
- Training passes image latent and noisy mask latent to the diffusion U-Net
  with the configured timestep and original prediction target.
- `log_validation` performs CUDA validation and returns Dice.
- Final evaluation loads the validation-selected checkpoint and calls
  `LeafPipeline`.

## Random-Yt

Random-Yt changes only the tensor supplied as the training-time `Y_t` input.

- `random_yt_like` creates independent Gaussian values with the same shape,
  dtype, and device as the reference noisy-mask latent.
- Reference `Y_t` construction, timestep, target, image, loss, validation, and
  final evaluation remain unchanged.
- Evaluation uses the normal diffusion path because the intervention is
  training-only.
- `tests/test_train_random_yt_audit.py` checks these invariants.

## Shuffle-Yt

Shuffle-Yt replaces only training-time `Y_t` with a tensor derived from another
batch member's clean mask latent.

- `derangement_like_permutation` creates a no-self-match permutation.
- The permuted clean mask latent is re-noised using the current target sample's
  timestep and noise tensor; another member's already-created `Y_t` is not
  copied directly.
- Image alignment and the original objective target are preserved.
- Batch size must exceed one.
- Timestep, target, loss, validation, and final evaluation remain unchanged.
- `tests/test_train_shuffle_yt_audit.py` checks these invariants.

## Core-No-Diff

Core-No-Diff removes the diffusion-dependent main-path inputs.

- `code/leaf/core_no_diff.py` defines `CoreNoDiffSegmentor`.
- `from_pretrained_components` initializes it from the same prepared LEAF
  components.
- `forward` accepts only the RGB image.
- `_forward_unet_without_timestep` runs retained U-Net blocks without an
  external timestep input.
- `train_core_no_diff` and `log_core_no_diff_validation` implement formal
  optimization and validation.
- Each evaluator selects `load_core_no_diff_model` and bypasses the reverse
  process when this mode is active.
- The clean mask latent target, optional DINOv2 alignment, EMA policy, fixed
  splits, and validation selection remain part of the comparison.
- `tests/test_core_no_diff.py` covers this branch.

## Configuration map

| Dataset | Full | Random-Yt | Shuffle-Yt | Core-No-Diff |
| --- | --- | --- | --- | --- |
| BTCV | `config-btcv.yaml` | `config-btcv-train-random-yt.yaml` | `config-btcv-train-shuffle-yt.yaml` | `config-btcv-core-no-diff.yaml` |
| ACDC | `config-acdc.yaml` | `config-acdc-train-random-yt.yaml` | `config-acdc-train-shuffle-yt.yaml` | `config-acdc-core-no-diff.yaml` |
| ISIC2018 | `config-isic2018.yaml` | `config-isic2018-train-random-yt.yaml` | `config-isic2018-train-shuffle-yt.yaml` | `config-isic2018-core-no-diff.yaml` |

Paths are relative to `code/config/`.

## Selection and final-test guard

Both training branches save only when mean validation Dice improves.
`write_best_checkpoint_metadata` records checkpoint, step,
`selection_partition: validation`, `selection_metric: mean_validation_dice`,
dataset, condition, and validation split.

`release_pipeline.py` first validates the selected physical cache against the
fixed split contract. It then checks all selection fields, resolves the
checkpoint inside the requested run directory, requires the requested EMA
subfolder, and invokes the dataset-specific test split. Output is isolated
under `final_test/` and is not overwritten.

`code/src/util/runtime.py` normalizes the allowed modes and
`require_rtx5090` rejects non-CUDA devices, unavailable CUDA, and the wrong GPU
name. `tests/gpu_smoke_12.py` independently exercises condition execution,
validation selection, checkpoint save/load, and test inference.
