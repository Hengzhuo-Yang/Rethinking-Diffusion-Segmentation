# Fixed Data Splits

The ID-only files under code/manifests are the source of truth. Preprocessors,
loaders, the release pipeline, and split tests all consume or validate these
same files. No dataset payload is distributed.

## Summary

| Dataset | Training | Validation | Final test | Unit |
| --- | ---: | ---: | ---: | --- |
| BTCV / Synapse | 18 / 2,211 | 2 / 295 | 10 / 1,273 | cases / 2D slices |
| ACDC | 70 / 1,304 | 10 / 182 | 20 / 416 | subjects / 2D slices |
| ISIC2018 Task 1 | 2,594 | 100 | 1,000 | images |

Every dataset has pairwise-disjoint training, validation, and final-test IDs.
Validation is used only for checkpoint selection. Final test is held back until
best_model.pt has been selected and recorded in best_checkpoint.json.

## BTCV / Synapse

Manifest files:

- code/manifests/btcv/train_cases.txt
- code/manifests/btcv/val_cases.txt
- code/manifests/btcv/test_cases.txt

Training cases:

    case0031 case0007 case0009 case0005 case0026 case0039
    case0024 case0034 case0033 case0030 case0023 case0040
    case0010 case0021 case0006 case0027 case0028 case0037

Validation cases, in manifest order:

    case0008 case0001

Final-test cases, in manifest order:

    case0022 case0038 case0036 case0032 case0002
    case0029 case0003 case0004 case0025 case0035

The original 12-case held-out pool is intentionally divided into the two-case
validation partition and ten-case final-test partition above. The old
10-percent/quick subset is excluded and must not be used to select a public
checkpoint. The BTCV preprocessor rejects legacy quick/10-percent directories,
requires exactly the 30 fixed raw cases, and checks 2,211/295/1,273 slices.

## ACDC

Manifest files:

- code/manifests/acdc/train_patients.txt
- code/manifests/acdc/val_patients.txt
- code/manifests/acdc/test_patients.txt

The fixed subject allocation is 70/10/20. Validation subjects are:

    patient019 patient021 patient029 patient033 patient041
    patient050 patient061 patient071 patient076 patient080

Final-test subjects are:

    patient002 patient003 patient008 patient009 patient012
    patient014 patient017 patient024 patient042 patient048
    patient049 patient053 patient055 patient064 patient067
    patient079 patient081 patient088 patient092 patient095

The remaining IDs named by train_patients.txt form the training partition. The
ACDC preprocessor validates the full subject set and expected slice totals
before any optional smoke-test limit is applied.

## ISIC2018 Task 1

Manifest files:

- code/manifests/isic2018/training.txt
- code/manifests/isic2018/validation.txt
- code/manifests/isic2018/testing.txt

These lists preserve the official Task 1 archive partitions: 2,594 training
images, 100 validation images, and 1,000 testing images. There is no random
re-splitting. The preprocessor compares archive IDs to the fixed lists before
writing arrays.

## Expected preprocessed layout

Each dataset output root must contain:

    DATA_ROOT/
      training/
        ITEM_ID/
          ITEM_ID_image.npy
          ITEM_ID_seg.npy
      validation/
        ...
      testing/
        ...
      manifest.csv
      summary.json

For BTCV and ACDC, ITEM_ID denotes a 2D slice directory and the case or subject
is encoded in its name. For ISIC2018, it denotes the official image ID. The
public loaders filter against the corresponding ID-only manifest and fail if a
listed case, subject, or image is missing.

## Validation

Run the manifest-only checks before obtaining or preprocessing any data:

    python tests/validate_splits.py

The validator also supports local preprocessed roots; see its command-line help:

    python tests/validate_splits.py --help

Never edit a manifest to accommodate an incomplete local copy. Repair the local
dataset or explicitly define a new, clearly named experimental protocol that is
not represented as this release's benchmark.
