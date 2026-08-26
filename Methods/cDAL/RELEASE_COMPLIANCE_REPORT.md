# Release compliance report

## Decision

**B — publication is permitted only for the specific non-commercial research/evaluation purposes and conditions in the NVIDIA Source Code License-NC.** The repository can be uploaded to a public Git host only with the included license, notices, attribution, modification record, and usage restriction intact. It must not be advertised as MIT, unrestricted open source, or commercially usable.

This conclusion is based on the selected upstream commit, its root license, preserved file headers, license history, and identified third-party sources. It is not legal advice and cannot eliminate all provenance or enforceability risk.

## Evidence

- Official remote: `https://github.com/Hejrati/cDAL`.
- Selected commit: `ade823bd0a92571dba72f90fb7e9043c3e66b6a9`.
- Selected commit contains the NVIDIA Source Code License-NC and describes the repository as derived from Denoising Diffusion GAN.
- The license grants use, reproduction, modification, and distribution for non-commercial research/evaluation subject to retention of the agreement and notices.
- Earlier repository history contained different or absent license text. This report applies the license present at the selected source baseline and does not infer retroactive permission from a later permissive file.
- `EMA.py` preserves its NVIDIA copyright/license header and expressly identifies adaptation from NVlabs/LSGM. The LSGM license text is therefore retained conservatively; its use condition is limited to non-commercial research/evaluation with NVIDIA processors. The validated RTX 5090 target meets that hardware condition.
- Apache-2.0, MIT, and BSD-3-Clause texts for identifiable embedded or historically derived components are retained under `LICENSES/`.

## Included release material

- Source required for the three datasets and four conditions.
- Fixed ID-only manifests, preprocessing source, configurations, GPU tests, and documentation.
- Root and third-party license texts, upstream identity, provenance, and modification records.
- Byte-identical upstream README under `code/README.md`.

## Excluded material

- All raw and preprocessed datasets, medical images, masks, HDF5/NumPy caches, and dataset summaries containing local paths.
- All trained or pretrained checkpoints, Git LFS weight pointers, third-party weights, and directly loadable model assets.
- Paper PDF, figures, supplementary files, and other publication assets.
- Training logs, validation/test results, TensorBoard/W&B artifacts, cached predictions, temporary smoke checkpoints, and test work directories.
- Conda installations, CUDA toolkit or driver copies, local wheel caches, virtual environments, and package caches.
- Cloud/Vast/RunPod archives and launchers, duplicate package copies, absolute-path run notes, and platform-specific data bundles.
- The BTCV 10% test runner and all legacy quick-final-test wrappers.
- Any dedicated experiment that sweeps or compares alternative sampling-step counts. The fixed four-step diffusion loop required by Full, Random-Yt, and Shuffle-Yt remains.
- Upstream MoNuSeg, CXR, and Hippocampus execution files that are not needed for this three-dataset release. Their official history remains available upstream.

## File provenance policy

- Upstream unchanged files retain their existing terms and headers.
- Upstream files modified for the audit or RTX 5090 remain governed by their applicable upstream terms.
- Locally added files that are tightly coupled to cDAL are distributed under the root non-commercial restriction; they are not separately relicensed as MIT.
- Generated manifests and documentation contain no dataset samples or weight material.
- Unknown provenance would be reported rather than guessed. No unknown-provenance file required for the released execution paths remains in the release.

## Publication assessment

The directory is technically suitable for a public GitHub repository **only as a non-commercial research source release**, with the additional NVIDIA-processor condition applied conservatively to the LSGM-derived EMA portion. A commercial, clinical, production, non-NVIDIA execution of that portion, or otherwise out-of-scope publication requires separate permission from the relevant rights holders. Dataset redistribution rights and medical-use approval are not supplied.

## Residual risks

- Historical license changes can create interpretation questions for material authored before the current root license; the selected baseline clearly carries the current non-commercial terms, but only the rights holders can give definitive clarification.
- `EMA.py` was redistributed by the DDGAN/cDAL lineage with a DDGAN license header while also naming LSGM as its adaptation source. This release includes both relevant NVIDIA texts and follows the more restrictive overlapping conditions; only NVIDIA can definitively resolve any intended relicensing history.
- Highly coupled local additions depend on upstream code and are conservatively treated under the same non-commercial restriction.
- The external datasets and upstream pretrained models have separate terms; users must review them before download or use.
- Only the documented Windows/RTX 5090 environment was executed. Other systems may require changes that have not been reviewed here.
