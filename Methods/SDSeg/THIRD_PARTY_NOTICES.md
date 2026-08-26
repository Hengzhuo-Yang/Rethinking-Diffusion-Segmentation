# Third-Party Notices

This file records third-party material retained by, adapted into, or required
by this focused SDSeg source release. It supplements, and does not replace, the
complete license texts in [`LICENSE`](LICENSE) and [`LICENSES/`](LICENSES/).

## SDSeg and CompVis Stable Diffusion lineage

The primary upstream is Stable Diffusion Segmentation:
<https://github.com/lin-tianyu/Stable-Diffusion-Seg>.

SDSeg identifies its diffusion implementation as substantially derived from
the CompVis Stable Diffusion codebase. The distribution therefore retains the
CreativeML Open RAIL-M text and the copyright notice for Robin Rombach, Patrick
Esser, and contributors in the root [`LICENSE`](LICENSE). The upstream SDSeg
repository and paper attribution are recorded in [`UPSTREAM.md`](UPSTREAM.md).

## CompVis taming-transformers

- Repository: <https://github.com/CompVis/taming-transformers>
- Resolved revision: `3ba01b241669f5ade541ce990f7650a3b8f65318`
- Copyright: Patrick Esser, Robin Rombach, Björn Ommer
- License: MIT
- License copy: `LICENSES/MIT-CompVis-taming-transformers.txt`

The dependency source is not vendored in this package.

## OpenAI CLIP and diffusion utilities

- CLIP repository: <https://github.com/openai/CLIP>
- Resolved CLIP revision: `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`
- guided-diffusion repository: <https://github.com/openai/guided-diffusion>
- improved-diffusion repository: <https://github.com/openai/improved-diffusion>
- Copyright: OpenAI
- License: MIT
- License copy: `LICENSES/MIT-OpenAI.txt`

The release retains upstream comments identifying adapted CLIP and diffusion
utility code. The CLIP dependency repository is not vendored here.

## Phil Wang implementations

- x-transformers repository: <https://github.com/lucidrains/x-transformers>
- denoising-diffusion-pytorch repository:
  <https://github.com/lucidrains/denoising-diffusion-pytorch>
- Copyright: Phil Wang
- License: MIT
- License copy: `LICENSES/MIT-Phil-Wang.txt`

The retained upstream files preserve their source attribution comments.

## Other upstream attribution markers

Retained upstream-integrated comments also reference Jonathan Ho's diffusion
implementation and a Sean Naren minGPT callback implementation. This release
does not present those portions as newly authored locally. They arrived through
the pinned SDSeg/Stable Diffusion lineage and remain subject to the root license
and preserved attribution markers.

## External Python packages

The environment specification names additional Python packages. Those packages
are installed separately and are governed by their respective licenses. No
license in this repository overrides an external package's terms.

## Excluded components and assets

The following are deliberately absent from this distribution:

- `ldm/modules/image_degradation/` and the unused ImageNet degradation paths;
- BlindSR, BSRGAN, and Real-ESRGAN degradation implementations;
- DPM-Solver code;
- all pretrained or trained weights and checkpoints;
- all dataset content, medical images, masks, and annotations;
- all result files, predictions, figures, logs, and paper assets.

Because the degradation implementations are absent, their separate academic,
non-commercial, Apache-2.0, or BSD notice obligations do not attach to this
package's included files. If a downstream distributor restores any excluded
component, it must perform a new provenance and license review and include the
corresponding notices.

## Dataset and checkpoint notice

BTCV, ACDC, and ISIC2018 are referenced by name only. No rights to those
datasets are granted here. The small manifest files contain split identifiers,
not dataset content. Users must acquire data and any pretrained model weights
separately from authorized sources and follow the terms supplied with them.
