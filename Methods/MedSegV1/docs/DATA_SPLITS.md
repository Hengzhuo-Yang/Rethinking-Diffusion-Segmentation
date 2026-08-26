# Fixed data splits

No images or masks are distributed. The repository contains ID-only manifests
under `manifests/`; every condition and seed uses exactly those files.

## Role policy

```text
train -> gradient updates and training statistics
val   -> periodic inference and mean validation Dice checkpoint selection
best  -> checkpoint selected independently for each training seed
test  -> one final inference after loading that seed's best checkpoint
```

Test data are never used for checkpoint selection, early stopping, loss design,
hyperparameters or model design. Augmentation is restricted to training, and no
cross-sample normalization statistics are computed from validation or test.

## BTCV / Synapse

The common TransUNet arrangement supplies 18 training cases and a 12-case
held-out pool. The release partitions the held-out pool at case level:

- Validation: `case0008` (148 slices), `case0001` (147 slices), total 295.
- Test: `case0022`, `case0038`, `case0036`, `case0032`, `case0002`,
  `case0029`, `case0003`, `case0004`, `case0025`, `case0035`, total 1,273.
- Training: the locally recorded 18 IDs in `manifests/btcv/train_cases.txt`,
  total 2,211 slices.

The historical slice-level `test_10pct_seed23` subset is excluded. The release
preprocessor creates `train`, `val` and `test` directories directly. The loader
also enforces the ID manifest, so a legacy combined held-out directory can be
read without allowing validation/test case overlap.

## ACDC

The locally used `cascade_70_10_20` subject split is preserved:

- Training: 70 subjects, 1,304 labelled ED/ES slices.
- Validation: 10 subjects, 182 slices.
- Test: 20 subjects, 416 slices.

Every cardiac frame and slice for a patient remains in the same partition. The
exact patient IDs are in `manifests/acdc/`; no slice-level random split is used.

## ISIC2018 Task 1

The official image-level archives are preserved without re-splitting:

- Training: 2,594 images.
- Validation: 100 images.
- Test: 1,000 images.

The exact image IDs are in `manifests/isic2018/`. Each listed image must have a
mask with the same stem.

## Validation command

Manifest-only validation:

```powershell
python tests\validate_splits.py
```

Validation against prepared local data:

```powershell
python tests\validate_splits.py `
  --btcv-root <processed-btcv-root> `
  --acdc-root <processed-acdc-root> `
  --isic2018-root <processed-isic2018-root>
```

The check fails on wrong unit or sample counts, duplicate IDs, partition overlap,
missing masks, unexpected filenames or manifest/data mismatch.

