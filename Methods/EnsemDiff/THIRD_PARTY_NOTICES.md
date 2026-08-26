# Third-Party Notices

This file identifies third-party material incorporated into or used to derive this source release. It supplements the license texts in LICENSES and does not replace them.

## JuliaWolleb Diffusion-based-Segmentation

- Project: Diffusion Models for Implicit Image Segmentation Ensembles
- Repository: https://github.com/JuliaWolleb/Diffusion-based-Segmentation
- Commit: 676f214035e90edd0357f51feab45841e4aefcfb
- License: MIT
- Copyright notice: Copyright (c) 2023 JuliaWolleb
- License copy: LICENSE and LICENSES/JuliaWolleb-Diffusion-based-Segmentation-MIT.txt

This is the primary upstream source for the code directory. Several files are modified as described in MODIFICATIONS.md.

## OpenAI improved-diffusion and guided-diffusion

- improved-diffusion: https://github.com/openai/improved-diffusion
- improved-diffusion reference commit: 1bc7bbbdc414d83d4abf2ad8cc1446dc36c4e4d5
- guided-diffusion: https://github.com/openai/guided-diffusion
- guided-diffusion reference commit: 22e0df8183507e13a7813f8d38d51b072ca1e67c
- License: MIT
- Copyright notice: Copyright (c) 2021 OpenAI
- License copy: LICENSES/OpenAI-improved-guided-CLIP-MIT.txt

The JuliaWolleb upstream expressly identifies improved-diffusion as its base. Content comparison also establishes guided-diffusion lineage for framework components. Relevant material includes diffusion schedules and processes, mixed-precision utilities, neural-network helpers, timestep resampling and respacing, UNet components, distributed utilities, training utilities, and training/sampling entry-point scaffolding. These components may be modified in this release.

## OpenAI Baselines

- Repository: https://github.com/openai/baselines
- Commit recorded in the source file: ea25b9e8b234e6ee1bca43083f8f3cf974143998
- License: MIT
- Copyright notice: Copyright (c) 2017 OpenAI (http://openai.com)
- License copy: LICENSES/OpenAI-Baselines-MIT.txt
- Affected file: code/guided_diffusion/logger.py

logger.py also identifies a copied utility from baselines/common/mpi_util.py at the same commit. Its existing source links are retained.

## OpenAI CLIP

- Repository: https://github.com/openai/CLIP
- Source path identified in code: clip/model.py on the main branch
- License: MIT
- Copyright notice: Copyright (c) 2021 OpenAI
- License copy: LICENSES/OpenAI-improved-guided-CLIP-MIT.txt
- Affected component: AttentionPool2d in code/guided_diffusion/unet.py

The file-level attribution is retained. No exact CLIP source commit was recorded by the upstream file, so none is asserted here.

## TransUNet

- Repository: https://github.com/Beckschen/TransUNet
- Reference commit: 02ef0010b36eb8328b5e689eadaf613602edf9b8
- License: Apache License 2.0
- License copy: LICENSES/TransUNet-Apache-2.0.txt
- Source lists: lists/lists_Synapse/train.txt and lists/lists_Synapse/test_vol.txt
- Derived artifacts: code/manifests/btcv/train_cases.txt, code/manifests/btcv/val_cases.txt, and code/manifests/btcv/test_cases.txt

The manifests are transformed and repartitioned derivatives, not verbatim copies. The changes are stated in MODIFICATIONS.md. No TransUNet model implementation is included merely because its split lists were used, and the TransUNet Apache license is not the governing license for the diffusion implementation. No separate NOTICE file was present in the referenced TransUNet snapshot.

## Jonathan Ho DDPM repository: residual provenance

- Repository: https://github.com/hojonathanho/diffusion
- Commit referenced by the source: 1e0dceb3b3495bbe19116a5e1b3596cd0706c543
- Referenced by: code/guided_diffusion/gaussian_diffusion.py, code/guided_diffusion/losses.py, and code/guided_diffusion/unet.py

The referenced repository snapshot did not contain a public LICENSE or NOTICE file. No license is assigned to that repository by this release. The affected Python implementation is received through OpenAI improved-diffusion/guided-diffusion and is redistributed in reliance on OpenAI's MIT grant for those PyTorch ports and adaptations. The original source references are retained to disclose the ancestor and the remaining provenance uncertainty.

## Runtime dependencies

The source imports external packages including PyTorch, torchvision, NumPy, SciPy, NiBabel, Pillow, blobfile, and Visdom. They are installed separately and are not vendored in this repository. Each remains subject to its own license. A dependency import is not classified here as copied source code.

## Dataset and model assets

BTCV/Synapse, ACDC, ISIC2018, and BraTS data are not distributed. Model checkpoints and pretrained weights are not distributed. The software licenses above do not grant dataset access, medical-image redistribution rights, model-weight rights, privacy clearance, or permission to use third-party trademarks.

## Release-local additions

Files categorized as release-local additions have no JuliaWolleb upstream Git history. Their distribution is conditional on confirmation that their contributor or project owner holds the necessary authorship or redistribution rights. The repository-level MIT text cannot grant rights that the releaser does not hold.
