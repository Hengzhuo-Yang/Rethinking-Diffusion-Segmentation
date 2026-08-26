# Modification record

This release preserves the locally trained MedSegDiff V1 implementation and its
three audit branches. Changes are grouped below so environment work is not
mistaken for a method contribution.

## Local working-tree changes relative to upstream commit 28b343f

### Dataset and evaluation support

- Added BTCV/Synapse binary PNG loading and evaluation.
- Replaced the legacy ISIC loader with the official ISIC2018 Task 1
  training/validation/testing PNG layout.
- Added ACDC four-class loading, foreground-channel targets, preprocessing and
  evaluation.
- Added configurable mask-channel and class counts throughout model creation,
  diffusion, sampling and metric code.

### Audit implementation

- Added `train_random_yt`, which replaces only the training-time noisy-mask
  input with independent Gaussian noise. The image, timestep, diffusion target
  and loss remain unchanged.
- Added `train_shuffle_yt`, which constructs the training-time noisy-mask input
  from another batch member's clean mask while retaining the current sample's
  timestep and noise. Batch size greater than one is enforced.
- Added `core_no_diff`, which uses the existing image-only highway segmentation
  path, skips q-sampling and timestep conditioning, and performs direct
  segmentation. Binary tasks use BCE-with-logits plus soft Dice; ACDC uses
  cross-entropy plus foreground soft Dice. This is an objective-changing
  discriminative-capacity counterfactual.

### CUDA, Windows and dependency compatibility

- Kept the Windows single-process `gloo` setup while model tensors remain on
  CUDA.
- Updated deprecated NumPy integer usage.
- Kept local DPM-Solver batch handling, tensor-valued dynamic thresholding and
  final-sample correction.
- Kept local multi-channel device placement and model-output handling.
- The release entry points now fail when CUDA is unavailable and require an
  RTX 5090 at `cuda:0`; there is no model-compute CPU fallback.

### Important Full-baseline difference

The upstream commit included a calibration MSE term and added it to the training
objective with a factor of 10. The locally trained working tree disables that
term globally and trains Full, Random-Yt and Shuffle-Yt with epsilon MSE plus the
learned-variance VB term. The V1 highway condition encoder remains active, but
normal V1 forward now stops at its bottleneck embedding: the Generic_UNet
localization decoder is not executed and normal sampling is diffusion
sample-only. Decoder parameters remain registered solely so historical
checkpoints load strictly. The explicit Core-No-Diff audit remains the only path
that invokes the legacy localization head.

## Release-only corrections

- Replaced BTCV's historical 12-case test plus slice-level 10% debug subset with
  a fixed case-level 18/2/10 train/validation/test split. Validation is
  `case0008` plus `case0001`; the other ten held-out cases are final test.
- Replaced reduced quick-checkpoint scoring with complete fixed-validation-set
  scoring for checkpoint selection.
- Added ID-only manifests and strict leakage/count/pairing validation.
- Kept ACDC and ISIC2018 validation exclusively for checkpoint selection and
  their official test partitions exclusively for final inference.
- Added parameterized data/output/checkpoint entry points and removed local and
  cloud absolute paths from the release tree.
- Excluded the old BTCV 10% test runners, ISIC test-subset runner, reduced
  validation commands, cloud queue files and any dedicated sampling-step sweep.
  Normal fixed-step reverse-sampling infrastructure remains because Full,
  Random-Yt and Shuffle-Yt require it.
- Added RTX 5090 preflight, twelve-condition smoke coverage, split validation,
  provenance documentation and release hygiene checks.

## Parameter comparison with upstream documentation

The formal local configuration retains the upstream-recommended batch size 8,
learning rate `5e-5`, 256-pixel input, 128 base channels, two residual blocks,
one attention head, 1000 diffusion steps, linear schedule and ensemble size 5.
The public commands explicitly select V1 because the upstream code default is a
newer architecture. Final diffusion inference uses the locally recorded
1000-step DDPM path; validation uses 20-step DPM-Solver++ for all three datasets
with ensemble size 1. Training seed is 23; the retained sampling module
initializes its inference RNG with seed 10. AMP is disabled and computation is
float32.
