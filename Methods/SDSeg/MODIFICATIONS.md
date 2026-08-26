# Modifications from Upstream SDSeg

This document is the prominent release-wide modification notice for a modified
distribution of Stable Diffusion Segmentation (SDSeg). It must remain with all
copies of this release.

- Upstream repository: <https://github.com/lin-tianyu/Stable-Diffusion-Seg>
- Upstream baseline: `0b0aa388a5e2def75abfbef90d7bcfc5c16f2704`
- Release preparation date: 2026-07-19
- Distribution form: source code, configuration, documentation, licenses, and
  small split-identifier manifests only

## Summary of changes

The release was narrowed to the requested BTCV, ACDC, and ISIC2018 segmentation
workflows and four existing experiment conditions:

- `full_diffusion`
- `train_random_yt`
- `train_shuffle_yt`
- `core_no_diff`

The three audit descriptions supplied during release preparation were used only
to understand and preserve the already-implemented branches. They were **not**
used to reimplement those algorithms.

Release-side changes comprise:

1. preserving the existing audit-mode implementation and exposing it through a
   3-dataset by 4-condition configuration matrix;
2. adding fixed, explicit case/subject/image split manifests;
3. separating training-loss validation, metric validation, and final test data
   so the test partition is not used for checkpoint selection;
4. requiring best-checkpoint metadata to identify validation-only selection
   before final evaluation;
5. replacing machine-specific data and pretrained-model paths with explicit
   arguments or environment-backed configuration;
6. adding dataset loaders and preprocessing entry points for ACDC, BTCV, and
   ISIC2018;
7. disabling formal CPU fallback and making the intended CUDA execution path
   explicit;
8. adding compatibility loading for current PyTorch checkpoint behavior;
9. limiting package discovery to the retained `ldm` namespace; and
10. removing unrelated and unpublished artifacts from the public package.

## Modified upstream paths

The following retained paths differ from the pinned upstream Git commit:

- `.gitignore`
- `README.md`
- `code/configs/SDSeg/synapse-cls2-ldm-kl-8.yaml`
- `code/ldm/data/synapse.py`
- `code/ldm/models/autoencoder.py`
- `code/ldm/models/diffusion/SDSeg.py`
- `code/ldm/modules/diffusionmodules/model.py`
- `code/ldm/util.py`
- `code/main.py`
- `code/scripts/slice2seg.py`
- `code/setup.py`

The `autoencoder.py`, diffusion-module `model.py`, and `util.py` changes route
checkpoint deserialization through a compatibility helper. The `setup.py`
change discovers the retained `ldm` namespace explicitly. The remaining paths
implement or wire the audit modes, portable roots, fixed partitions,
validation-only model selection, final evaluation, and CUDA-only release
policy described above.

## Added release paths

The following groups were added after the pinned upstream commit:

- `code/configs/SDSeg/acdc-cls4-ldm-kl-8.yaml`
- `code/configs/SDSeg/isic2018-ldm-kl-8.yaml`
- `code/configs/experiments/{acdc,btcv,isic2018}/*.yaml`
- `code/ldm/data/{acdc,btcv,isic2018}.py`
- `code/ldm/__init__.py`
- `code/manifests/acdc/*.txt`
- `code/manifests/btcv/*.txt`
- `code/manifests/isic2018/*.txt`
- `code/scripts/preprocess_acdc_to_sdseg.py`
- `code/scripts/preprocess_btcv_synapse_to_sdseg.py`
- `code/scripts/preprocess_isic2018_to_sdseg.py`
- `code/scripts/release_pipeline.py`
- `code/tests/*.py`
- `code/sitecustomize.py`
- `docs/*.md`
- `environment.yml` and `requirements.txt`
- the release compliance and provenance documents at repository root

The text manifests contain identifiers only; they contain no medical image or
annotation data.

## Intentionally excluded

This focused package excludes all model weights, checkpoints, datasets,
medical images, masks, predictions, results, logs, paper assets, local runtime
wrappers, and machine-specific paths.

It also excludes the unused upstream `ldm/modules/image_degradation/` and
ImageNet degradation routes. As a result, BlindSR-derived, BSRGAN, and
Real-ESRGAN degradation code is not part of this distribution and its more
restrictive or separate downstream terms are not introduced into this focused
package. Normal fixed-step segmentation sampling remains; unrelated
sampling-step comparison audits and their outputs are excluded.

## Downstream modification duty

Anyone who changes this release should add a dated entry identifying the files
and substantive changes, retain the upstream license and attributions, and add
prominent notices to modified files as required by the CreativeML Open RAIL-M.
