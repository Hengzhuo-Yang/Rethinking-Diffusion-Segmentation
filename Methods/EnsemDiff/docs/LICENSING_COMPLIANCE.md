# Licensing Compliance

## Decision

The release is classified as:

**A — publication of a modified source release is permitted, subject to the conditions below.**

No reviewed governing license imposes a non-commercial or research-only restriction, so category B does not apply. Category C is not assigned because the direct redistribution chain consists of the JuliaWolleb MIT release, OpenAI MIT releases, OpenAI Baselines MIT material, and Apache-2.0-licensed TransUNet list artifacts. The no-license DDPM ancestor is disclosed as residual provenance while the release relies on OpenAI's later MIT distribution of its PyTorch ports.

This is a source-provenance and license-compliance assessment, not a guarantee of zero legal risk.

## License matrix

| Material | Source and version | Governing terms for this release | Preserved text |
| --- | --- | --- | --- |
| Primary EnsemDiff upstream | JuliaWolleb/Diffusion-based-Segmentation 676f214035e90edd0357f51feab45841e4aefcfb | MIT, Copyright (c) 2023 JuliaWolleb | LICENSE; LICENSES/JuliaWolleb-Diffusion-based-Segmentation-MIT.txt |
| Diffusion framework lineage | OpenAI improved-diffusion 1bc7bbbdc414d83d4abf2ad8cc1446dc36c4e4d5 and guided-diffusion 22e0df8183507e13a7813f8d38d51b072ca1e67c | MIT, Copyright (c) 2021 OpenAI | LICENSES/OpenAI-improved-guided-CLIP-MIT.txt |
| Logger | OpenAI Baselines ea25b9e8b234e6ee1bca43083f8f3cf974143998 | MIT, Copyright (c) 2017 OpenAI | LICENSES/OpenAI-Baselines-MIT.txt |
| AttentionPool2d ancestor | OpenAI CLIP, main-branch path recorded upstream | MIT, Copyright (c) 2021 OpenAI | LICENSES/OpenAI-improved-guided-CLIP-MIT.txt |
| BTCV case-list artifacts | Beckschen/TransUNet 02ef0010b36eb8328b5e689eadaf613602edf9b8 | Apache License 2.0 | LICENSES/TransUNet-Apache-2.0.txt |
| Ho DDPM ancestor | hojonathanho/diffusion 1e0dceb3b3495bbe19116a5e1b3596cd0706c543 | No separate public license found; OpenAI MIT port is the direct grant relied upon | Provenance links retained; no invented license |
| Release-local additions | Absent from JuliaWolleb baseline | Conditional on contributor or owner having sufficient rights and approving distribution under the stated release terms | Confirmation required before publication |

## Verified license copies

The following files were compared with the official fixed-source texts and matched exactly after normalizing line endings:

- LICENSE against JuliaWolleb/Diffusion-based-Segmentation at 676f214
- LICENSES/JuliaWolleb-Diffusion-based-Segmentation-MIT.txt against the same source
- LICENSES/OpenAI-improved-guided-CLIP-MIT.txt against OpenAI improved-diffusion at 1bc7bb, guided-diffusion at 22e0df, and CLIP's published MIT text
- LICENSES/OpenAI-Baselines-MIT.txt against OpenAI Baselines at ea25b9e8
- LICENSES/TransUNet-Apache-2.0.txt against TransUNet at 02ef0010

No separate NOTICE file was found in the reviewed JuliaWolleb, OpenAI improved-diffusion, OpenAI guided-diffusion, OpenAI Baselines, OpenAI CLIP, or TransUNet source snapshots. Consequently, no upstream NOTICE text was available to copy. THIRD_PARTY_NOTICES.md records the attributions relevant to this release.

## Governing versus historical licenses

The current primary baseline commit 676f214 uses the JuliaWolleb MIT license. An Apache License 2.0 file existed in earlier commits of that repository, was deleted at 2891a01, and was replaced by MIT at 676f214. That history is recorded for traceability; it is not used to call the current primary upstream Apache-licensed or automatically dual-licensed.

The TransUNet Apache License 2.0 has a different and narrow role: it applies to the BTCV manifests derived from TransUNet's Synapse split lists. It does not replace the MIT terms applying to the diffusion framework.

## Conditions that must remain satisfied

1. Distribute the root LICENSE and all four files currently in LICENSES.
2. Distribute UPSTREAM.md, MODIFICATIONS.md, and THIRD_PARTY_NOTICES.md with the source.
3. Retain the existing Baselines, CLIP, and Ho/DDPM source references in the affected Python files.
4. Continue to identify the BTCV manifests as transformed TransUNet list artifacts and retain their modification notice.
5. Do not describe the entire combined repository as solely original work of the release maintainer.
6. Obtain and retain confirmation that the contributor or project owner authored each release-local addition or otherwise has authority to distribute it under the release terms.
7. Do not include medical data, masks, model weights, checkpoints, paper PDFs, logs, caches, or other assets merely because code can load them.
8. Review licenses separately before vendoring any currently external runtime dependency.
9. Preserve the unchanged official README at code/README.md as an upstream record.

## Apache-2.0 handling for BTCV manifests

The TransUNet-derived BTCV files are machine-readable lists containing one case identifier per line. Adding comment headers could change downstream parsers, so the prominent modification statement is provided in MODIFICATIONS.md and THIRD_PARTY_NOTICES.md and names every affected manifest. The unmodified Apache License 2.0 text is included. The reviewed TransUNet snapshot contained no separate NOTICE file.

Downstream redistributors who change these manifests must preserve the Apache license and provide an equally clear modification notice. They should not imply that TransUNet authors endorse this validation/test repartition.

## Release-local authorship confirmation

Git status and history can establish that a file was absent from the JuliaWolleb baseline; they cannot establish who authored it. Before public publication, the project owner should record a confirmation covering at least:

- ACDC, BTCV, and ISIC2018 data loaders;
- validation and visualization support;
- preprocessing and evaluation scripts;
- checkpoint validation entry points;
- fixed manifests and split transformations;
- environment and release configuration;
- release documentation and tests.

The confirmation should state either that the material is original and may be released under the intended terms, or identify each external source and its compatible license. Any unsupported copied material must be removed or separately authorized.

## Residual risks

- The Ho DDPM repository cited by three source files had no public license at the referenced snapshot. The direct OpenAI PyTorch implementations were published under MIT, which is the practical redistribution grant relied upon, but the ancestor should remain disclosed.
- The OpenAI CLIP attribution records a main-branch path rather than a fixed commit. The license text is known and preserved, but exact source-version pinning is unavailable from the inherited comment.
- Dependency versions may change; source-only use does not authorize vendoring future package contents.
- Dataset access conditions, privacy rules, challenge terms, and trained-weight terms remain outside this source-code license decision.

Subject to the listed conditions, the release is suitable for a public source repository under category A.
