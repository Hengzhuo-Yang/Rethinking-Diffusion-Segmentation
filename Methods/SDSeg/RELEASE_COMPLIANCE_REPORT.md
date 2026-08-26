# SDSeg Public-Release Compliance Report

Status: **READY WITH CONDITIONS — Category A modified code release**  
Report date: 2026-07-19  
Public GitHub suitability: **Yes, for the code-only package described here**  
Residual license/provenance risk: **Moderate**

## Decision

The pinned upstream CreativeML Open RAIL-M permits reproduction,
modification, sublicensing, and distribution of SDSeg's **Complementary
Material**, which the license defines to include source code, scripts,
data-preparation code, and documentation. The current package is therefore not
blocked merely because it contains modified or locally added source files.

This release is classified as **Category A: public modified-code distribution
permitted subject to license, attribution, and modification-notice
conditions**. It may be prepared for a public GitHub repository provided that
the final package retains:

- the complete root [`LICENSE`](LICENSE);
- upstream identification in [`UPSTREAM.md`](UPSTREAM.md);
- the prominent change record in [`MODIFICATIONS.md`](MODIFICATIONS.md); and
- the applicable licenses and notices in [`LICENSES/`](LICENSES/) and
  [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

This is deliberately a **code-only** release. No checkpoint or dataset is
authorized or included by this decision.

## License basis

The authoritative upstream is:

- project: Stable Diffusion Segmentation (SDSeg);
- repository: <https://github.com/lin-tianyu/Stable-Diffusion-Seg>;
- pinned baseline: `0b0aa388a5e2def75abfbef90d7bcfc5c16f2704`;
- license: CreativeML Open RAIL-M, dated 2022-08-22; and
- paper: *Stable Diffusion Segmentation for Biomedical Images with Single-step
  Reverse Process*, MICCAI 2024.

The root license text in this package matches the pinned upstream license
line-for-line and is retained in full. Its relevant provisions are:

1. source code, scripts, documentation, examples, and data-preparation code are
   Complementary Material;
2. Section II grants copyright and patent permissions for the Complementary
   Material, subject to the stated terms;
3. modified files must be identified prominently;
4. pertinent copyright, patent, trademark, and attribution notices must be
   retained; and
5. the license does not grant rights to datasets.

Section III distinguishes Complementary Material from a **Model**, whose
definition includes learned weights, checkpoints, and optimizer states. This
package contains no Model or Derivative of the Model. Anyone who separately
obtains, trains, or redistributes weights must review and comply with the
license's model conditions and use restrictions. In particular, this research
code must not be represented as a clinical system or used with a model to
provide medical advice or medical-results interpretation in violation of the
license.

## Final package boundary

The intended public package contains only:

- Python source for the focused SDSeg runtime;
- BTCV, ACDC, and ISIC2018 data loaders and preprocessing scripts;
- fixed split-identifier manifests;
- the 3-dataset by 4-condition experiment configurations;
- environment and dependency metadata;
- release tests and technical documentation; and
- license, attribution, provenance, and modification records.

It explicitly excludes:

- pretrained weights and archives;
- training checkpoints, optimizer state, and derived model weights;
- every dataset, medical image, mask, annotation, and raw archive;
- predictions, metrics exports, figures, and result tables;
- run logs, resolved local configurations, and packaged environments;
- paper PDFs, posters, screenshots, and publication assets;
- machine-specific wrappers and private absolute paths;
- nested Git metadata and generated caches; and
- sampling-step comparison audits, sweeps, and their artifacts.

The identifier manifests are not medical data and contain no image or mask
content. Dataset names are references only; users must obtain data through
authorized sources under the datasets' own terms.

## Provenance assessment

The upstream boundary is fixed at the commit above. Modified upstream files
and additions made after that commit are enumerated in
[`MODIFICATIONS.md`](MODIFICATIONS.md). The earlier Category-C conclusion was
overly conservative because it treated local Git `untracked` status as if it
were proof that redistribution rights were absent. Git tracking status is a
version-control fact, not a license classification.

For this release, the added ACDC, BTCV, ISIC2018, preprocessing, audit wiring,
configuration, and runtime-support files are treated as project-side release
modifications. No included file carries a contrary third-party copyright or
license marker. This supports Category A under the user's stated release
authority and the upstream grant for Complementary Material.

The residual risk remains **moderate**, rather than low, because the local
additions do not each carry individual authorship records and because the root
license is a responsible-AI license rather than a conventional permissive
software license. The publisher should retain its authorship history. If any
local addition was copied from another source, that source and its license must
be added before publication.

## Third-party assessment

The included code retains or depends on MIT-licensed material from:

- CompVis taming-transformers, resolved at
  `3ba01b241669f5ade541ce990f7650a3b8f65318`;
- OpenAI CLIP, resolved at
  `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`;
- OpenAI guided-diffusion and improved-diffusion utilities; and
- Phil Wang's x-transformers and denoising-diffusion-pytorch implementations.

The corresponding notices are preserved in [`LICENSES/`](LICENSES/) and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). External Python packages
are not relicensed by this repository and remain subject to their own terms.

The focused inventory does **not** include
`ldm/modules/image_degradation/`, ImageNet degradation code, BlindSR, BSRGAN,
Real-ESRGAN degradation implementations, or DPM-Solver. The prior report's
BlindSR concern therefore does not apply to the files actually distributed.
Restoring any excluded component would require a new third-party license
review.

## Audit implementation boundary

The three supplied audit descriptions were used only to identify and preserve
the already-implemented experiment semantics. They were not used to re-create
the algorithms. The retained implementation exposes:

- `full_diffusion`;
- `train_random_yt`;
- `train_shuffle_yt`; and
- `core_no_diff`.

The release preparation work organizes those existing branches across BTCV,
ACDC, and ISIC2018, cleans the public packaging boundary, and corrects
validation/test and portability wiring. It does not claim a new upstream
algorithmic authorship position.

## Publication conditions

Before making a public repository, perform one final inventory check and
confirm all of the following:

1. `LICENSE`, `UPSTREAM.md`, `MODIFICATIONS.md`,
   `THIRD_PARTY_NOTICES.md`, and `LICENSES/` are present;
2. no model or optimizer checkpoint is tracked;
3. no dataset, medical image, mask, prediction, result, or paper asset is
   tracked;
4. no private absolute path, username, credential, or local service token is
   present;
5. generated Python caches and transient test output are absent;
6. modified-file notices and upstream attributions remain prominent; and
7. technical validation results are reported separately and honestly.

The Category A decision is a redistribution assessment, not a claim that all
technical tests have passed. GPU smoke-test status and reproducibility evidence
must be taken from the final technical validation documents rather than
inferred from this report.

## Final classification

**Category A — conditionally permitted public modified-code release, with
moderate residual risk.**

The release may contain code, configurations, split identifiers,
documentation, and license records. It must not contain weights, checkpoints,
datasets, medical assets, or unrelated copyrighted publication assets. The
complete CreativeML Open RAIL-M, attribution, and modification record must be
preserved in every redistributed copy.

This report documents a practical release-compliance review and is not legal
advice.
