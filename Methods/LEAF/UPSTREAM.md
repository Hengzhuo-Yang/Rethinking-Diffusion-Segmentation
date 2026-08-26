# Upstream provenance

## Project identity

This release is a source redistribution of **LEAF: Latent Diffusion with Efficient Encoder Distillation for Aligned Features in Medical Image Segmentation**. The local upstream remote recorded during release preparation was:

- repository: `https://github.com/Pearisli/LEAF.git`
- upstream base commit: `adb6ae37e641124107dc606c3642e406da8c0559`
- root license at that revision: MIT, reproduced in `LICENSE` and `LICENSES/LEAF-MIT.txt`

The upstream README uses the spelling `https://github.com/lispear/LEAF.git`. The immutable commit identifier above, rather than owner-name capitalization or redirect behavior, is the reference used for comparison.

## Dirty-worktree disclosure

The local `official_code` directory used as the functional source was based on the commit above but had uncommitted research and audit changes. It was therefore a **dirty working tree**, not a byte-for-byte clean checkout of that commit. This release does not attribute those local changes to the upstream authors and does not describe the release tree as an unmodified upstream snapshot.

The provenance relationship is:

1. upstream LEAF commit `adb6ae37e641124107dc606c3642e406da8c0559` supplies the identifiable base;
2. selected local, experimentally used changes supply the audit implementations and dataset/evaluation support;
3. release-only changes make paths, splits, documentation, tests, and licensing suitable for a public source package;
4. excluded experimental artifacts remain outside this release.

`code/README.md` is the upstream official README preserved as received. It is retained for historical fidelity and is not the release's environment or command guide. Any release-level README is separate and may describe the verified local environment and corrected workflow.

## Source categories

- **Upstream LEAF:** the model, training, and support code inherited from the commit above, including locally modified derivatives.
- **Third-party derived:** the files mapped in `THIRD_PARTY_NOTICES.md`; their own licenses and attribution continue to apply.
- **Locally added or modified:** controlled audit modes, dataset preparation and evaluation, fixed split handling, tests, portable configuration, and release documentation.
- **Generated metadata/configuration:** manifests and configuration templates that contain no dataset samples or model parameters.
- **Excluded:** data, weights, checkpoints, samples, logs, papers, caches, secrets, and out-of-scope experiments.

See `MODIFICATIONS.md` for the change classes and `RELEASE_COMPLIANCE_REPORT.md` for the release boundary.
