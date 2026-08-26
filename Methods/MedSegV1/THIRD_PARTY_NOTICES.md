# Third-party notices

The combined source remains under the upstream MedSegDiff MIT license. This file
records material origins identified from file headers and the upstream README;
it does not replace the applicable license texts.

## Directly identifiable components

- MedSegDiff, ImprintLab / Kids with Tokens — MIT. License text:
  `LICENSES/MedSegDiff-MIT.txt`.
- OpenAI `guided-diffusion`, `improved-diffusion`, CLIP-derived attention code,
  and OpenAI Baselines-derived logging code — MIT. License text:
  `LICENSES/OpenAI-MIT.txt`.
- Cheng Lu et al. `dpm-solver` — MIT. License text:
  `LICENSES/DPM-Solver-MIT.txt`.
- Ho et al. diffusion-model utilities are identified by source URLs retained in
  `gaussian_diffusion.py`, `losses.py`, and `unet.py`.

## Additional projects acknowledged by upstream

The unmodified upstream README also acknowledges MrPrism, DiagnosisFirst,
Diffusion-based-Segmentation, Nested U-Net, nnU-Net and vit-pytorch. The local
Git history does not map every expression in the aggregate MedSegDiff files to a
specific donor revision. Those acknowledgements are retained through
`code/README.md`; no claim of original authorship over those upstream-derived
portions is made here.

## Release-added files

The split manifests, release configuration, validation/smoke tooling and release
documentation are distributed with this aggregate under the root MIT terms.
Dataset contents, model weights and paper assets are not redistributed.

