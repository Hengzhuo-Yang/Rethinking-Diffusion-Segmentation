# Audit implementation map

This document maps the four release conditions to the existing implementation.
It is descriptive: no condition was reimplemented while preparing this release.

## Mode selection

All experiment configurations instantiate
`ldm.models.diffusion.SDSeg.SDSeg` and set exactly one `model.params.audit_mode`:

| Public condition | `audit_mode` | Scope |
|---|---|---|
| Full | `full_diffusion` | Baseline training and inference |
| Random-Yt | `train_random_yt` | Training input ablation only |
| Shuffle-Yt | `train_shuffle_yt` | Training input ablation only |
| Core-No-Diff | `core_no_diff` | Main segmentation core and objective |

`SDSeg._SUPPORTED_AUDIT_MODES` rejects any other value. At fit start the model
writes `audit_metadata.json`, including the selected mode, seed, parameter count,
and mode-specific semantics.

## Full

`full_diffusion` is the preserved baseline. Training constructs
`Y_t_ref = q_sample(Y_0, t, epsilon)` and passes that reference noisy latent to
the diffusion core. The configured prediction target, latent segmentation term,
variational term, autoencoder, conditional encoder, U-Net, attention/skip paths,
EMA, decoding, and metric logging follow the existing baseline implementation.

No sampling-step sweep is part of the release. Formal segmentation evaluation
uses the fixed direct one-step protocol shared by the applicable conditions.

## Random-Yt

The implementation is in `SDSeg._maybe_replace_train_random_yt` and
`SDSeg.p_losses`.

- Activation requires both training mode and `audit_mode=train_random_yt`.
- The model input `Y_t_input` is replaced with an independent
  `torch.randn_like(Y_t_ref)` standard-Gaussian tensor.
- Image conditioning, timestep `t`, clean latent `Y_0`, original sampled noise,
  prediction target, and loss formulas are unchanged.
- The latent segmentation loss continues to reference `Y_t_ref`, not the random
  replacement supplied to the model input.
- Validation and inference do not activate the replacement.

Runtime guards assert shape, dtype, device, and batch/timestep consistency and
emit explicit audit metrics/metadata.

## Shuffle-Yt

The implementation is in `SDSeg._make_batch_derangement` and `SDSeg.p_losses`.

- Activation requires both training mode and `audit_mode=train_shuffle_yt`.
- A batch-level derangement maps every example `i` to another example `j`, with
  `j != i`; self-matches are asserted absent.
- `Y_t_input` is reconstructed as
  `q_sample(Y_0[j], t[i], epsilon[i])`.
- The original `Y_t_ref`, clean target, sampled noise target, timestep, image,
  and loss formulas remain associated with example `i` and are not shuffled.
- There is no dataset-level fallback; batch size must be greater than one.
- Validation and inference do not activate the shuffle.

The formal batch size of four satisfies the derangement requirement. A smoke
test must also use at least two samples for this condition.

## Core-No-Diff

The existing `core_no_diff` path configures the main U-Net input for image
features only and invokes a single direct forward pass.

- `Y_t`, timestep `t`, `q_sample`, and reverse-chain state are absent from the
  main segmentation core.
- Binary configurations use image-only input. Multiclass ACDC retains the class
  query solely to select the one-vs-rest target class.
- The output target is the clean segmentation latent `Y_0`.
- The main-core loss is direct clean-latent L1 regression.
- Diffusion noise prediction, the variational loss, and the reverse sampler are
  not used for the main core.
- The first-stage autoencoder, condition encoder, U-Net multiscale/attention
  structure, EMA, decoding/postprocessing, validation metrics, and image logging
  remain available as supporting components.

Runtime guards assert image-only channel use and output/target shape equality.
This condition is not described as objective-preserving: both the main input
and objective differ from Full.

## Shared validation and test boundary

The audit mode does not change split policy. Every condition follows:

```text
train -> metric_validation -> max val_avg_dice -> best checkpoint -> final test
```

`main.DataModuleFromConfig` exposes `validation`, `metric_validation`, and
`test` separately. Periodic Dice and checkpoint selection use
`metric_validation`. `scripts/slice2seg.py` requires `best_checkpoint.json` and
verifies that its path and validation-only selection fields match the requested
checkpoint before final testing.

## Evidence expected from a formal run

Retain, without publishing medical data or checkpoints:

- the resolved experiment configuration;
- `audit_metadata.json`;
- `best_checkpoint.json` proving validation selection;
- environment and GPU preflight output;
- metric summaries identifying validation or test partition;
- the corresponding split-manifest hashes/counts;
- smoke or formal run status, including failures.

Do not report a mode as exercised merely because its configuration exists; the
run evidence must identify the active branch.
