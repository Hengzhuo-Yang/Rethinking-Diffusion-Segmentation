# Upstream Sources

This release is an auditable derivative of several separately licensed sources. The repository-level license does not erase or replace the notices attached to those sources.

## Primary project

- Project: Diffusion Models for Implicit Image Segmentation Ensembles
- Authors listed by the project: Julia Wolleb, Robin Sandkühler, Florentin Bieder, Philippe Valmaggia, and Philippe C. Cattin
- Paper: https://arxiv.org/abs/2112.03145
- Official repository: https://github.com/JuliaWolleb/Diffusion-based-Segmentation
- Baseline commit: 676f214035e90edd0357f51feab45841e4aefcfb
- Baseline license at that commit: MIT, Copyright (c) 2023 JuliaWolleb
- Preserved upstream README: code/README.md

The root LICENSE and LICENSES/JuliaWolleb-Diffusion-based-Segmentation-MIT.txt are exact copies of the MIT license at the baseline commit, apart from possible line-ending representation.

### License history

The repository history contains a license transition:

- b40c8cc69491f89c03ded92f80edfc18fab5b18c introduced an Apache License 2.0 text.
- 2891a016bad1f2cd5fda3811050cf9e128e13673 deleted that license.
- 676f214035e90edd0357f51feab45841e4aefcfb added the current MIT license.

The historical Apache text governed relevant historical snapshots. It is not represented here as the governing license for commit 676f214 or for this release, and the history is not used to claim an unverified dual license.

## OpenAI diffusion framework lineage

The primary upstream README expressly states that its diffusion implementation is based on OpenAI improved-diffusion:

- Repository: https://github.com/openai/improved-diffusion
- Reference commit used for content and license verification: 1bc7bbbdc414d83d4abf2ad8cc1446dc36c4e4d5
- License: MIT, Copyright (c) 2021 OpenAI

Content comparison also shows substantial common lineage with OpenAI guided-diffusion:

- Repository: https://github.com/openai/guided-diffusion
- Reference commit: 22e0df8183507e13a7813f8d38d51b072ca1e67c
- License: MIT, Copyright (c) 2021 OpenAI

The common OpenAI MIT text is preserved at LICENSES/OpenAI-improved-guided-CLIP-MIT.txt. Framework-derived files include much of code/guided_diffusion and the original scaffolding of the segmentation train and sample entry points. Some of those files have subsequently been modified for medical segmentation, additional datasets, validation, and controlled experimental conditions.

## OpenAI Baselines logger

code/guided_diffusion/logger.py identifies its direct source in its existing file-level documentation:

- Repository: https://github.com/openai/baselines
- Source commit: ea25b9e8b234e6ee1bca43083f8f3cf974143998
- Source files: baselines/logger.py and baselines/common/mpi_util.py
- License: MIT, Copyright (c) 2017 OpenAI

The applicable license is preserved at LICENSES/OpenAI-Baselines-MIT.txt. The existing source links in logger.py must be retained.

## OpenAI CLIP component

code/guided_diffusion/unet.py states that its AttentionPool2d implementation was adapted from OpenAI CLIP:

- Repository: https://github.com/openai/CLIP
- Source path recorded by the file: clip/model.py on the main branch
- License: MIT, Copyright (c) 2021 OpenAI

The source file does not pin a CLIP commit, so this release does not invent one. The OpenAI MIT license text is identical to the copy in LICENSES/OpenAI-improved-guided-CLIP-MIT.txt, and the existing CLIP attribution in unet.py must be retained.

## Original DDPM implementation references

The following files retain explicit references to Jonathan Ho's original DDPM repository:

- code/guided_diffusion/gaussian_diffusion.py
- code/guided_diffusion/losses.py
- code/guided_diffusion/unet.py

Recorded source:

- Repository: https://github.com/hojonathanho/diffusion
- Referenced commit: 1e0dceb3b3495bbe19116a5e1b3596cd0706c543

No public LICENSE or NOTICE file was present at the referenced repository snapshot. The distributed code in this release comes through OpenAI's PyTorch ports and adaptations, which OpenAI published under its MIT license. This is the direct redistribution license relied upon here. The Ho repository references are nevertheless retained as provenance, and the absence of a separate public license at that ancestor is disclosed as a residual provenance risk rather than silently assigning it a license.

## TransUNet-derived BTCV split lists

The BTCV case universe and original 18-case training / 12-case held-out organization are derived from the official TransUNet Synapse lists:

- Repository: https://github.com/Beckschen/TransUNet
- Reference commit verified for this release: 02ef0010b36eb8328b5e689eadaf613602edf9b8
- Source paths: lists/lists_Synapse/train.txt and lists/lists_Synapse/test_vol.txt
- License: Apache License 2.0

The release transforms the source lists into case-level manifests and divides the original 12-case held-out list into a fixed two-case validation set and a ten-case final test set. The affected derived artifacts are:

- code/manifests/btcv/train_cases.txt
- code/manifests/btcv/val_cases.txt
- code/manifests/btcv/test_cases.txt

The TransUNet Apache License 2.0 text is preserved at LICENSES/TransUNet-Apache-2.0.txt. This license applies to the TransUNet-derived list artifacts; it is not the governing license for the JuliaWolleb/OpenAI diffusion code. The source snapshot did not contain a separate NOTICE file.

## Release-local additions

Files absent from the JuliaWolleb baseline are classified as release-local additions. This category describes their relationship to that baseline; it is not, by itself, proof of authorship. It includes the ACDC, BTCV, and ISIC2018 loaders, validation support, preprocessing and evaluation scripts, fixed manifests, environment files, and release documentation.

Publication under the release's stated terms is conditional on the contributor or project owner confirming that they authored these additions or otherwise hold sufficient rights to distribute and license them. Third-party material within an added file remains governed by its own license.

## Excluded materials

This release does not include medical images, masks, NIfTI examples, dataset archives, model weights, checkpoints, run logs, cached bytecode, paper PDFs, or cloud deployment bundles. No software license in this repository grants rights to obtain or redistribute those materials.
