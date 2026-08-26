# Fixed data splits

The distributed manifests contain only case, subject, or image identifiers. They contain no image data, masks, absolute paths, or derived statistics. Every condition and seed must use these exact files.

## BTCV/Synapse

The source convention is the common 18-case training group and 12-case held-out pool. The held-out pool is divided at case level; it is not resampled by slice.

- Training, 18 cases and 2,211 slices: `case0005`, `case0006`, `case0007`, `case0009`, `case0010`, `case0021`, `case0023`, `case0024`, `case0026`, `case0027`, `case0028`, `case0030`, `case0031`, `case0033`, `case0034`, `case0037`, `case0039`, `case0040`.
- Validation, 2 cases and 295 slices: `case0008` (148) and `case0001` (147).
- Test, 10 cases and 1,273 slices: `case0022`, `case0038`, `case0036`, `case0032`, `case0002`, `case0029`, `case0003`, `case0004`, `case0025`, `case0035`.

The preprocessor stores training under `BTCV/train` and the 12-case pool under `BTCV/heldout`. For compatibility with the already validated local cache, the loader may read the same pool from the legacy physical directory `BTCV/test`; the semantic role is always determined by the required validation or test case manifest. It never reads `test_10pct_seed23`.

Manifests:

- `manifests/btcv/train_cases.txt`
- `manifests/btcv/validation_cases.txt`
- `manifests/btcv/test_cases.txt`

## ACDC

The split is fixed at subject level, including every labelled cardiac frame and every slice for a subject.

- Training: 70 subjects, 1,304 slices.
- Validation: 10 subjects, 182 slices.
- Test: 20 subjects, 416 slices.

The exact recovered subject IDs are in:

- `manifests/acdc/training_subjects.txt`
- `manifests/acdc/validation_subjects.txt`
- `manifests/acdc/test_subjects.txt`

The physical directories are `ACDC/training`, `ACDC/validation`, and `ACDC/testing`. No slice-level random split is allowed.

## ISIC2018 Task 1

The official image-level partitions are preserved without drawing a new validation set from training.

- Training: 2,594 images.
- Validation: 100 images.
- Test: 1,000 images.

Manifests:

- `manifests/isic2018/training_images.txt`
- `manifests/isic2018/validation_images.txt`
- `manifests/isic2018/test_images.txt`

The physical directories are `ISIC18/training`, `ISIC18/validation`, and `ISIC18/testing`.

## Enforced role contract

1. Training items are the only items used for gradient updates, training augmentation, and training statistics.
2. Validation runs without gradient updates and is the only source for mean-Dice best-checkpoint selection.
3. Test is not accessible to the training entry. Final-test entries require a validation-selection metadata file and load its checkpoint.
4. Each condition and each seed uses the same three manifests.
5. No cross-sample normalization statistic is fitted outside training. The implemented image scaling is per item and does not estimate statistics from validation or test.
6. Test does not determine an epoch, loss weight, hyperparameter, architecture, scheduler, or early-stop decision.

## Validation result

`tests/validate_splits.py` was executed against the local preprocessed datasets on 2026-08-22. It reported PASS for manifest uniqueness, pairwise disjointness, image/mask pairing, case/subject/image filtering, and all target counts:

| Dataset | Train | Validation | Test |
|---|---:|---:|---:|
| BTCV | 2,211 | 295 | 1,273 |
| ACDC | 1,304 | 182 | 416 |
| ISIC2018 | 2,594 | 100 | 1,000 |

The validator observed the obsolete 10% BTCV directory in the read-only local cache and explicitly reported it as present but excluded.
