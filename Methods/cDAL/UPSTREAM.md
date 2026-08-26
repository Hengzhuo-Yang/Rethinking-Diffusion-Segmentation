# Upstream identity and baseline

## Identified project

- Project: cDAL.
- Paper: *Conditional Diffusion Model with Spatial Attention and Latent Embedding for Medical Image Segmentation*.
- Authors: Behzad Hejrati, Soumyanil Banerjee, Carri Glide-Hurst, and Ming Dong.
- Venue: MICCAI 2024, LNCS 15009, pages 202–212.
- DOI: `10.1007/978-3-031-72114-4_20`.
- Official repository: `https://github.com/Hejrati/cDAL`.
- Baseline branch: `master`.
- Baseline commit: `ade823bd0a92571dba72f90fb7e9043c3e66b6a9`.

The identity was established from the Git remote, Git history, upstream README, repository license, and the local paper. The paper PDF was used only as method evidence and is not distributed.

## Selected local source

The release source is the working tree whose Git repository points to the official remote at the commit above. Three local compute packages were inspected as execution evidence. Their meaningful source files matched the selected working tree when pure CRLF/LF differences were ignored. All three packages omitted the obsolete BTCV 10% test runner; none supplied a distinct algorithm version. Therefore the packages, copied data, checkpoints, logs, and cloud-specific launchers are excluded.

The preserved upstream README is `code/README.md`; its SHA-256 in this release is `2453209818D5496817C709D3B4A30E16E1217EE9992C2A3AFEAE32B7B3A4E2BD` and it is byte-identical to the selected working-tree README.

## License history

- The initial June 2024 history did not contain a root license.
- MIT text was added on 2024-07-05 (`5028312` and `f9bab843` in the local history).
- Commit `7a4bafd3` on 2025-03-17 replaced that root license with the NVIDIA Source Code License-NC used by the Denoising Diffusion GAN-derived code.
- The selected baseline commit contains the NVIDIA non-commercial license. This release retains that license and does not claim that a later permissive license applies retroactively.

## Relation to the paper and official code

The paper and upstream release cover the original cDAL method and its original medical segmentation datasets. BTCV, ACDC, and ISIC2018 are locally validated extensions; the official repository has no authoritative configuration for these three datasets. Their code, configurations, fixed splits, and audit conditions are therefore recorded as local modifications, while the generator/discriminator backbone remains coupled to and governed by the upstream license.

The upstream authors' original pretrained checkpoints are linked from the official README and [Hugging Face repository](https://huggingface.co/Hejrati/cDAL/tree/main). No checkpoint is copied into this release.
