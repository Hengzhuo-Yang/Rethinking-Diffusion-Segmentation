# Third-party notices

This code-only release preserves layered provenance; the root MIT license does
not erase licenses inherited from upstream components or dependencies.

| Component | Role | License / notice |
|---|---|---|
| TSLDSeg | Primary upstream project | Root `LICENSE` and `code/LICENSE` (MIT) |
| SDSeg / Stable-Diffusion-Seg | Declared TSLDSeg code/model lineage | `LICENSES/SDSeg-CreativeML-Open-RAIL-M.txt` |
| CompVis latent-diffusion | Diffusion/VAE implementation lineage | `LICENSES/CompVis-latent-diffusion-MIT.txt` |
| CompVis taming-transformers | Installed dependency | `LICENSES/CompVis-taming-transformers-MIT.txt` |
| OpenAI CLIP | Installed dependency | `LICENSES/OpenAI-CLIP-MIT.txt` |
| DeepHypergraph | Method/library acknowledged by upstream; no `dhg` package is vendored here | <https://github.com/iMoonLab/DeepHypergraph> |

The CreativeML Open RAIL-M text is retained because the SDSeg distribution
applies it to the accompanying model lineage and includes use-based restrictions
for model derivatives. This release distributes no weights, but downstream
users must review those terms before obtaining, modifying, or distributing
upstream pretrained or trained model artifacts.

Python dependencies are installed from their own distributions and remain
subject to their respective licenses. Dataset files are not distributed and
remain subject to the original dataset terms.

