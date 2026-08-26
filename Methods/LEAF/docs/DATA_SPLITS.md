# Fixed Data Splits

The ID-only files under `code/manifests/` are the source of truth. They contain
no image, label, patient metadata, or local path. All conditions and seeds must
use the same files.

| Dataset | Training | Validation | Final test | Unit |
| --- | ---: | ---: | ---: | --- |
| BTCV/Synapse | 18 / 2,211 | 2 / 295 | 10 / 1,273 | cases / slices |
| ACDC | 70 / 1,304 | 10 / 182 | 20 / 416 | subjects / slices |
| ISIC2018 Task 1 | 2,594 | 100 | 1,000 | images |

Every split is disjoint. Augmentation is training-only. Validation selects the
best checkpoint by mean Dice. Test is used once after that checkpoint is
loaded.

## BTCV/Synapse

Manifests are `code/manifests/btcv/{train,val,test}.txt`.

Training:

    case0031 case0007 case0009 case0005 case0026 case0039
    case0024 case0034 case0033 case0030 case0023 case0040
    case0010 case0021 case0006 case0027 case0028 case0037

Validation:

    case0008 case0001

Final test:

    case0022 case0038 case0036 case0032 case0002
    case0029 case0003 case0004 case0025 case0035

The conventional 18-case training partition is unchanged. The 12-case
held-out pool is divided at case level: two cases select the checkpoint and
the other ten are final test only.

## ACDC

Manifests are `code/manifests/acdc/{train,val,test}.txt`.

Validation:

    patient019 patient021 patient029 patient033 patient041
    patient050 patient061 patient071 patient076 patient080

Final test:

    patient002 patient003 patient008 patient009 patient012
    patient014 patient017 patient024 patient042 patient048
    patient049 patient053 patient055 patient064 patient067
    patient079 patient081 patient088 patient092 patient095

The other 70 IDs in `train.txt` are training subjects. The allocation was
recovered from the MT-UNet/CASCADE ACDC distribution used by the formal local
experiments. All labelled cardiac frames and slices from one subject remain
together.

## ISIC2018 Task 1

Manifests are `code/manifests/isic2018/{train,val,test}.txt`. They preserve the
official 2,594/100/1,000 archive partitions. The preprocessor compares image
and mask archive members against the fixed IDs and does not resplit training.

## Preprocessed layout

BTCV uses `BTCV/train`, `BTCV/val`, and `BTCV/test`. ACDC uses
`ACDC/training`, `ACDC/validation`, and `ACDC/testing`. ISIC2018 uses
`ISIC18/training`, `ISIC18/validation`, and `ISIC18/testing`. Each split has
paired `images/` and `masks/`; ACDC also has `label_maps/`.

## Validation

```powershell
python tests\validate_splits.py
python tests\validate_splits.py --data-root btcv=<BTCV_CACHE_ROOT> --data-root acdc=<ACDC_CACHE_ROOT> --data-root isic2018=<ISIC2018_CACHE_ROOT>
```

The validator checks exact canonical hashes, format, counts, duplicates,
cross-split overlap, image/mask pairing, sample counts, and owner membership.
Do not edit a manifest to accommodate a partial local dataset.
