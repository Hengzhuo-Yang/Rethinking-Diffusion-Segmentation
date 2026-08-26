# Split usage map

| Dataset | Task | Semantic partition | Physical location | Manifest field/file | Dataloader | Entry point | Device | Allowed use |
|---|---|---|---|---|---|---|---|---|
| BTCV | Training | train | `BTCV/train` | `train_manifest`; `manifests/btcv/train_cases.txt` | `create_dataset` -> `BTCVDataset` | `train_cDal_monu_and_lung.py:391` | `cuda:0` | Gradient updates and training statistics. |
| BTCV | Selection | validation | `BTCV/heldout`, or legacy physical `BTCV/test` | `val_manifest`; `manifests/btcv/validation_cases.txt` | `create_dataset` -> `BTCVDataset` | validation call at `train_cDal_monu_and_lung.py:709` | `cuda:0` | Mean validation Dice and best-checkpoint selection only. |
| BTCV | Final inference | test | same 12-case physical pool, disjoint case filter | `--manifest`; `manifests/btcv/test_cases.txt` | `create_dataset` -> `BTCVDataset` | `evaluate_btcv_cdal.py:71` | `cuda:0` | Load validation-selected best checkpoint and compute final metrics. |
| ACDC | Training | train | `ACDC/training` | `train_manifest`; `manifests/acdc/training_subjects.txt` | `create_dataset` -> `ACDCDataset` | `train_cDal_monu_and_lung.py:391` | `cuda:0` | Gradient updates and training statistics. |
| ACDC | Selection | validation | `ACDC/validation` | `val_manifest`; `manifests/acdc/validation_subjects.txt` | `create_dataset` -> `ACDCDataset` | validation call at `train_cDal_monu_and_lung.py:709` | `cuda:0` | Mean validation Dice and best-checkpoint selection only. |
| ACDC | Final inference | test | `ACDC/testing` | `--manifest`; `manifests/acdc/test_subjects.txt` | `create_dataset` -> `ACDCDataset` | `evaluate_acdc_cdal.py:70` | `cuda:0` | Load validation-selected best checkpoint and compute final metrics. |
| ISIC2018 | Training | train | `ISIC18/training` | `train_manifest`; `manifests/isic2018/training_images.txt` | `create_dataset` -> `ISIC2018Dataset` | `train_cDal_monu_and_lung.py:391` | `cuda:0` | Gradient updates and training statistics. |
| ISIC2018 | Selection | validation | `ISIC18/validation` | `val_manifest`; `manifests/isic2018/validation_images.txt` | `create_dataset` -> `ISIC2018Dataset` | validation call at `train_cDal_monu_and_lung.py:709` | `cuda:0` | Mean validation Dice and best-checkpoint selection only. |
| ISIC2018 | Final inference | test | `ISIC18/testing` | `--manifest`; `manifests/isic2018/test_images.txt` | `create_dataset` -> `ISIC2018Dataset` | `evaluate_isic2018_cdal.py:70` | `cuda:0` | Load validation-selected best checkpoint and compute final metrics. |

The shared loader factory is `code/preprocess_dataset/dataset.py:4`. The dataset classes begin at `BTCV.py:30`, `ACDC.py:36`, and `ISIC2018.py:34`. Formal device enforcement is `code/utils.py:41`. Final-test metadata validation is `code/release_validation.py:14`.

Best checkpoints are written by `save_best_checkpoint` at `code/train_cDal_monu_and_lung.py:345` only after the validation call returns a Dice value that improves the current best. Metadata binds the checkpoint to the dataset, audit mode, semantic validation split, and validation-manifest hash. The three final-test entries reject validation as a final split and reject a checkpoint without valid selection metadata.
