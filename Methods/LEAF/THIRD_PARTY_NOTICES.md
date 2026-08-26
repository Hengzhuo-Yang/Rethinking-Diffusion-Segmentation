# Third-party notices

The root MIT license covers LEAF code only to the extent granted by the LEAF copyright holders. It does not replace the licenses or attribution attached to third-party-derived files. The following notices and the corresponding full texts in `LICENSES/` must remain with redistributed copies.

| Distributed file | Upstream source or lineage | License and retained attribution |
|---|---|---|
| `code/leaf/autoencoder.py` | CompVis `latent-diffusion`, principally the autoencoder building blocks in `ldm/modules/diffusionmodules/model.py` | MIT; Copyright (c) 2022 Machine Vision and Learning Group, LMU Munich. See `LICENSES/CompVis-latent-diffusion-MIT.txt`. |
| `code/leaf/unet.py` | OpenAI `guided-diffusion`, principally `guided_diffusion/unet.py`, as adapted in LEAF | MIT; Copyright (c) 2021 OpenAI. See `LICENSES/OpenAI-guided-diffusion-MIT.txt`. |
| `code/src/util/seeding.py` | `prs-eth/Marigold` seeding utility | Apache License 2.0; the file header identifying Copyright 2023 Bingxin Ke, ETH Zurich is retained. See `LICENSES/Marigold-Apache-2.0.txt`. |

## Jonathan Ho source citation

The `AttentionBlock` documentation in `code/leaf/unet.py` retains the OpenAI lineage note that the attention implementation was originally ported from Jonathan Ho's `hojonathanho/diffusion` file `diffusion_tf/models/unet.py` at commit `1e0dceb3b3495bbe19116a5e1b3596cd0706c543`. The code distributed here follows the OpenAI `guided-diffusion` lineage and carries OpenAI's MIT license text. This historical link is preserved as provenance; it does not relicense or make a separate rights statement about the contents of the linked Jonathan Ho repository.

## External dependencies and artifacts

Python packages imported by the source are dependencies, not vendored copies, unless a file is expressly mapped above. Their package licenses apply independently.

The pretrained latent-diffusion U-Net and VAE referenced by the official README are **not included**. Neither are any trained LEAF weights, downloaded model assets, datasets, medical images, or generated predictions. A download link or filename is not a grant to redistribute an artifact; users must review the license and access terms at its source.

Project names and trademarks are used only to identify provenance. No endorsement is implied.
