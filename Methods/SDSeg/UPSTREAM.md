# Upstream and License Provenance

## Primary upstream

- Project: **Stable Diffusion Segmentation (SDSeg)**
- Repository: <https://github.com/lin-tianyu/Stable-Diffusion-Seg>
- Release baseline: commit `0b0aa388a5e2def75abfbef90d7bcfc5c16f2704`
- Baseline commit date: 2025-06-19
- Paper: *Stable Diffusion Segmentation for Biomedical Images with Single-step Reverse Process* (MICCAI 2024)
- Paper record: <https://arxiv.org/abs/2406.18361>
- Primary license: **CreativeML Open RAIL-M**, dated 2022-08-22
- Authoritative upstream license: <https://github.com/lin-tianyu/Stable-Diffusion-Seg/blob/0b0aa388a5e2def75abfbef90d7bcfc5c16f2704/LICENSE>

The complete upstream license is retained at [`LICENSE`](LICENSE). Its text is
unchanged from the pinned upstream baseline. Copyright, attribution, patent,
and trademark notices from retained upstream files must not be removed.

## Why a modified code-only release is permitted

The CreativeML Open RAIL-M defines **Complementary Material** to include source
code, scripts, data-preparation code, documentation, tutorials, and examples
used to define, run, load, benchmark, or evaluate a model. Section II grants a
perpetual, worldwide, royalty-free license to reproduce, modify, sublicense,
and distribute that material, subject to the license conditions.

This distribution therefore classifies as a **Category A conditional public
modified-code release**. The conditions relevant to this package include:

1. retain the complete license and all pertinent attribution notices;
2. prominently identify modified files and the nature of the changes;
3. do not imply endorsement by the upstream authors or use their trademarks
   outside the rights granted by the license; and
4. preserve the applicable third-party notices and licenses.

The release-wide prominent change record is [`MODIFICATIONS.md`](MODIFICATIONS.md).
Downstream redistributors must retain it and should keep any file-local change
notices that accompany modified files.

## Code-only boundary

This repository contains Complementary Material only. It intentionally does
**not** contain:

- pretrained model weights;
- training checkpoints, optimizer states, or derivative-model weights;
- BTCV, ACDC, ISIC2018, or any other dataset content;
- input images, masks, predictions, result files, or preview assets;
- paper PDFs, posters, or other publication assets; or
- experiment logs, cached tensors, or packaged environments.

The license expressly states that datasets are not licensed under it. Users
must obtain every dataset from its official source and comply with its own
terms. Likewise, users who separately obtain a Model or Derivative of the Model
must comply with the CreativeML Open RAIL-M model-use and redistribution terms,
including its use restrictions. No permission to redistribute a dataset or a
checkpoint is implied by this code release.

## Upstream-integrated lineage

SDSeg is derived from and contains adapted portions of the CompVis Stable
Diffusion codebase. Retained files also contain upstream attribution markers
for OpenAI diffusion utilities, OpenAI CLIP, and Phil Wang's
`x-transformers`/diffusion implementations. These attributions are preserved;
see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

The reproducible environment used for this release resolved two external
source dependencies to the following revisions:

| Dependency | Repository | Resolved revision | License copy |
|---|---|---|---|
| taming-transformers | <https://github.com/CompVis/taming-transformers> | `3ba01b241669f5ade541ce990f7650a3b8f65318` | `LICENSES/MIT-CompVis-taming-transformers.txt` |
| OpenAI CLIP | <https://github.com/openai/CLIP> | `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6` | `LICENSES/MIT-OpenAI.txt` |

Those dependency repositories and their weights are not vendored here. A
dependency name in configuration or installation metadata does not change the
license of that dependency.

## Provenance of release additions

Files that did not exist at the pinned upstream commit are identified as local
release additions in [`MODIFICATIONS.md`](MODIFICATIONS.md). Git tracked versus
untracked status in a local working tree is not, by itself, evidence that a
file cannot be redistributed. The additions are treated here as release-side
modifications supplied for this project, and no contrary third-party source or
license marker was found in the included files.

The publisher should nevertheless retain its underlying authorship records.
If any local addition was copied from another source, that source and its
license must be recorded before publication.

## No endorsement and no warranty

This modified release is not endorsed by the upstream SDSeg, CompVis, OpenAI,
or other cited authors. It is provided subject to the disclaimer and limitation
of liability in [`LICENSE`](LICENSE). This provenance note is a release record,
not legal advice.
