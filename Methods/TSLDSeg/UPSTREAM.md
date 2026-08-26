# Upstream and provenance

## Primary work

- Project: TSLDSeg
- Official repository: <https://github.com/Saury997/TSLDSeg>
- Upstream commit used as the clean reference: `381827e1dc64a99132bbc4adf68e3c73d295c15a`
- Paper: *TSLDSeg: A Texture-aware and Semantic-enhanced Latent Diffusion Model for Medical Image Segmentation*
- DOI: <https://doi.org/10.1016/j.patcog.2025.112795>
- Upstream code license: MIT, Copyright 2025 Zongjian Yang

`code/README.md` and `code/LICENSE` are unchanged copies from that TSLDSeg
commit. The root `LICENSE` is the same MIT text.

## Declared lineage

The official TSLDSeg README states that the project is based on:

- SDSeg / Stable-Diffusion-Seg: <https://github.com/lin-tianyu/Stable-Diffusion-Seg>
- CompVis latent-diffusion: <https://github.com/CompVis/latent-diffusion>
- DeepHypergraph: <https://github.com/iMoonLab/DeepHypergraph>

The locally inspected SDSeg reference was commit
`0b0aa388a5e2def75abfbef90d7bcfc5c16f2704`. This is provenance evidence, not
a claim that TSLDSeg identified that exact commit as its historical base.

## Local validated source of truth

The release was assembled from the current validated working tree under
`TSLDSeg/official_code`, not by checking out and replacing it with pristine
upstream code. That tree contained the existing three audit implementations,
dataset adapters, evaluation scripts, RTX 5090 environment work, and local
metric fixes. The source tree itself was not edited during release assembly.

Every release-only semantic change is listed in `MODIFICATIONS.md`. Data,
weights, PDFs, logs, caches, and historical run wrappers were not copied.

