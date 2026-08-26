# Modifications

This document records material changes relative to JuliaWolleb/Diffusion-based-Segmentation commit 676f214035e90edd0357f51feab45841e4aefcfb. Pure CRLF/LF differences were ignored when deciding whether a tracked file was substantively modified.

## Modified upstream files

The following files contain substantive changes relative to the baseline:

- code/guided_diffusion/bratsloader.py
- code/guided_diffusion/dist_util.py
- code/guided_diffusion/gaussian_diffusion.py
- code/guided_diffusion/script_util.py
- code/guided_diffusion/train_util.py
- code/guided_diffusion/unet.py
- code/scripts/segmentation_sample.py
- code/scripts/segmentation_train.py

The audited working tree already contained substantial research modifications,
and the release copy adds manifest locking, validation-only selection metadata,
portable orchestration, and publication checks. The changes principally add or
support:

- ACDC, BTCV/Synapse, and ISIC2018 dataset selection;
- multi-class and binary segmentation targets;
- controlled Random-Yt, Shuffle-Yt, and Core-No-Diff conditions;
- validation-only checkpoint selection;
- explicit final-test checkpoint loading;
- fail-closed Windows single-device setup for an exact RTX 5090, with no
  formal CPU fallback;
- configurable paths and non-interactive visualization behavior.

These files remain derivatives of the JuliaWolleb and, where applicable, OpenAI upstream code. Their modification does not replace the upstream MIT notices or the embedded Baselines, CLIP, and DDPM source references.

## Upstream files retained without substantive source changes

The following baseline files were copied without substantive source changes:

- code/README.md
- code/guided_diffusion/__init__.py
- code/guided_diffusion/fp16_util.py
- code/guided_diffusion/logger.py
- code/guided_diffusion/losses.py
- code/guided_diffusion/nn.py
- code/guided_diffusion/resample.py
- code/guided_diffusion/respace.py

The baseline requirements file was not copied verbatim as the release installation contract because the verified runtime uses a newer, narrower dependency set. The original dependency lineage remains documented by the preserved upstream repository and commit.

## Release-local source and configuration

The following files or file families were added after the JuliaWolleb baseline:

- code/guided_diffusion/acdcloader.py
- code/guided_diffusion/btcvloader.py
- code/guided_diffusion/isicloader.py
- code/guided_diffusion/validation_util.py
- code/guided_diffusion/visdom_util.py
- code/scripts/preprocess_acdc_to_npy.py
- code/scripts/preprocess_btcv_synapse_to_npy.py
- code/scripts/preprocess_isic2018_to_npy.py
- code/scripts/evaluate_acdc_samples.py
- code/scripts/evaluate_btcv_samples.py
- code/scripts/evaluate_isic2018_samples.py
- code/scripts/validate_checkpoint.py
- code/scripts/fixed_split_manifests.py
- code/scripts/release_pipeline.py
- environment.yml
- requirements.txt
- code/manifests/acdc/*
- code/manifests/isic2018/*
- release documentation and validation tests under tests/

The root `environment.yml` and `requirements.txt` are the only installation
contracts. No duplicate environment file is published below `code/`.

## RTX 5090 compatibility and runtime controls

Operational compatibility changes are separated from algorithmic conditions:

- `code/guided_diffusion/dist_util.py` validates CUDA and the exact device name,
  selects logical `cuda:0`, uses Windows-compatible single-process distributed
  initialization, and raises instead of falling back to CPU.
- The formal train, standalone validation, and sample entry points initialize
  the validated CUDA device before model computation and use CUDA timing and
  memory reporting without a CPU alternative.
- `tests/gpu_preflight.py` checks compute capability 12.0, compiled `sm_120`
  support, CUDA arithmetic/backward, formal-model placement/forward, and CUDA
  allocation before a real pipeline action.
- The environment contract replaces the upstream PyTorch 1.9-era requirements
  with the locally successful Python 3.10 / PyTorch 2.7.0 CUDA 12.8 package set.
- No custom CUDA/C++ extension, local wheel, architecture build flag, or patched
  kernel is required by this project.

These changes are runtime compatibility and acceptance controls, not method
contributions. The formal configuration remains FP32 with AMP disabled;
matmul TF32 is false, cuDNN TF32 is true, cuDNN benchmark is false, and neither
cuDNN deterministic mode nor deterministic algorithms are enabled.

## Formal-parameter comparison with upstream

Apart from the locally retained batch size, the flags explicitly published in
the preserved upstream README match the release baseline: model image setting
256, width 128, two residual blocks, one head, learned variance, attention 16,
no scale-shift normalization, 1,000 linear diffusion steps, no rescaling, and
learning rate 1e-4. The release additionally makes local settings explicit that
the upstream README does not specify: dataset-specific channels/classes and
targets, 224×224 prepared inputs, seed 10, 40,000/60,000 step budgets,
validation every 5,000 steps, 100-step validation DDPM, and 1,000-step final
DDPM. Fixed train/validation/test manifests and the three controlled conditions
are release protocol or mechanism-analysis changes, not upstream parameter
claims.

These are classified as release-local additions, not as proven original works solely because they were absent from Git. Their inclusion is conditional on confirmation by the contributor or project owner that the necessary authorship or redistribution rights are held.

## TransUNet list transformations

The three BTCV manifests are modified derivatives of the Apache-2.0-licensed TransUNet Synapse lists:

- code/manifests/btcv/train_cases.txt reduces the official slice-level training list to its ordered unique case identifiers.
- code/manifests/btcv/val_cases.txt selects case0008 and case0001 from the original 12-case held-out list for validation and checkpoint selection.
- code/manifests/btcv/test_cases.txt retains the other ten held-out cases for final testing.

This section is the prominent release-level modification notice for those plain-text list artifacts. Their parsers expect one identifier per line, so license prose is kept outside the machine-readable manifests. The unmodified Apache License 2.0 text is distributed at LICENSES/TransUNet-Apache-2.0.txt.

## Deliberate exclusions

The release excludes:

- all tracked or untracked medical image and mask data;
- the 38 BraTS NIfTI mini-example files present in the upstream Git tree but absent from the audited working tree;
- all model weights and checkpoints;
- results, logs, metrics exports, and cached bytecode;
- the dedicated different-sampling-steps experiment and its registration or orchestration scripts;
- the legacy BTCV ten-percent quick-test subset utility and related outputs;
- local machine paths, cloud deployment bundles, and private experiment packages.

These exclusions avoid redistributing assets with separate rights, publishing runtime artifacts, or presenting obsolete diagnostic partitions as final tests.

## License-history clarification

Older commits of the JuliaWolleb repository carried Apache License 2.0. The selected baseline commit 676f214 carries MIT, Copyright (c) 2023 JuliaWolleb. This release records the older Apache license as history only and does not describe it as the current governing license. The separately preserved TransUNet Apache License 2.0 applies only to TransUNet-derived material.
