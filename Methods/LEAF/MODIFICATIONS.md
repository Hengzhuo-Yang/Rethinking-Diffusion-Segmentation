# Modifications from upstream LEAF

## Baseline and interpretation

The comparison baseline is LEAF commit `adb6ae37e641124107dc606c3642e406da8c0559`. The local `official_code` source was dirty, so this document distinguishes upstream inheritance, controlled research modifications, and release-only cleanup. It does not claim that every locally present change was committed by the upstream authors.

`code/README.md` is deliberately preserved as the official upstream README. Release instructions and corrected split semantics belong in separate release documentation.

## Controlled mechanism audits

The public package retains the full LEAF condition and three falsification-oriented conditions:

- **Full:** the LEAF diffusion path used as the reference condition.
- **Random-Yt:** replaces only the training-time noisy-mask-latent input with an independent Gaussian tensor of matching shape, dtype, and device; it does not replace the original objective target.
- **Shuffle-Yt:** uses a no-self-match permutation to select another case's clean mask latent, then re-noises it with the current sample's timestep and noise; it changes only the training-time input and does not shuffle the objective target.
- **Core-No-Diff:** removes the noised-mask input, timestep conditioning, forward noising, and reverse sampler from the main path while retaining an image-conditioned latent segmentation objective for the controlled comparison.

The principal implementation areas are `code/train.py`, `code/leaf/core_no_diff.py`, `code/leaf/__init__.py`, the twelve dataset/condition configuration files, the three evaluation scripts, and the audit tests. These are experimental audit changes, not hardware compatibility changes and not a claim of a new upstream LEAF method.

## Dataset and evaluation corrections

Release organization separates training, checkpoint selection, and final testing:

- BTCV validation for best-checkpoint selection is restricted to the fixed validation cases `case0008` and `case0001`; the final test split is evaluated only after loading that selected checkpoint.
- ACDC and ISIC2018 validation are used for checkpoint selection and are not reused as final test sets.
- Dataset adapters, preprocessing scripts, manifests, and evaluation entry points make split roles explicit and test for overlap/leakage.
- Preprocessors and the formal pipeline lock the fixed manifests by fingerprint and validate physical cache ownership and counts before model execution.
- Validation-selection metadata records the exact validation split, and final evaluation fails if its selected EMA weights are missing.

This replaces historical local shortcuts such as BTCV test-10-percent selection and quick final tests that reused validation data.

## Compatibility and public-release changes

Release-only work includes:

- portable paths and configuration entry points instead of private absolute paths;
- CUDA-oriented preflight and explicit failure behavior for formal GPU entry points rather than silent CPU fallback;
- preservation of the locally used mixed-precision/TF32 behavior where documented, without describing it as an algorithmic contribution;
- preprocessing and evaluation commands for BTCV, ACDC, and ISIC2018;
- fixed-split validation, metric, audit-branch, and GPU smoke-test infrastructure;
- global seed initialization and stable filename ordering for repeatable data traversal;
- provenance, license, exclusion, environment, and code-structure documentation.

The model architecture is not reformatted or broadly rewritten for release aesthetics. Third-party-derived model files keep their separate license lineage as described in `THIRD_PARTY_NOTICES.md`.

## Third-party-derived files

- `code/leaf/autoencoder.py`: CompVis latent-diffusion lineage, MIT.
- `code/leaf/unet.py`: OpenAI guided-diffusion lineage, MIT, with the retained Jonathan Ho historical source citation.
- `code/src/util/seeding.py`: Marigold lineage, Apache-2.0, with the Bingxin Ke/ETH Zurich header retained.

## Items deliberately not carried forward

The source release omits the `noise_coupled` experiment, the different-sampling-steps audit, obsolete split shortcuts, weights and model assets, datasets and medical images, run outputs, logs, caches, paper PDFs, cloud bundles, archives, and unknown-provenance binaries. These omissions define the public package boundary; they do not alter the read-only source project outside this release directory.
