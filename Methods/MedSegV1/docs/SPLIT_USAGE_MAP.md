# Split usage map

| Dataset | Task | Partition | Path and manifest fields | Dataloader | Entry point | Device | Allowed use |
|---|---|---|---|---|---|---|---|
| BTCV | Training | train | `--data_dir`, `--data_manifest manifests/btcv/train_cases.txt` | `BTCVDataset` | `segmentation_train.py` | cuda:0 | Gradient updates |
| BTCV | Selection | val | `--val_data_dir`, `--val_manifest manifests/btcv/val_cases.txt` | `BTCVDataset` through `ValidationRunner` | `segmentation_train.py` | cuda:0 | Select best checkpoint by validation Dice |
| BTCV | Final inference | test | `--data_dir`, `--data_manifest manifests/btcv/test_cases.txt` | `BTCVDataset` | `segmentation_sample.py` | cuda:0 | Final metrics after loading best |
| ACDC | Training | training | `--data_dir`, `--acdc_split training`, `--data_manifest manifests/acdc/train_subjects.txt` | `ACDCDataset` | `segmentation_train.py` | cuda:0 | Gradient updates |
| ACDC | Selection | validation | `--val_data_dir`, `--val_acdc_split validation`, `--val_manifest manifests/acdc/val_subjects.txt` | `ACDCDataset` through `ValidationRunner` | `segmentation_train.py` | cuda:0 | Select best checkpoint by validation Dice |
| ACDC | Final inference | testing | `--data_dir`, `--acdc_split testing`, `--data_manifest manifests/acdc/test_subjects.txt` | `ACDCDataset` | `segmentation_sample.py` | cuda:0 | Final metrics after loading best |
| ISIC2018 | Training | training | `--data_dir`, `--data_manifest manifests/isic2018/train_images.txt` | `ISICDataset` | `segmentation_train.py` | cuda:0 | Gradient updates |
| ISIC2018 | Selection | validation | `--val_data_dir`, `--val_manifest manifests/isic2018/val_images.txt` | `ISICDataset` through `ValidationRunner` | `segmentation_train.py` | cuda:0 | Select best checkpoint by validation Dice |
| ISIC2018 | Final inference | testing | `--data_dir`, `--data_manifest manifests/isic2018/test_images.txt` | `ISICDataset` | `segmentation_sample.py` | cuda:0 | Final metrics after loading best |

`run_experiment.py` resolves these fields from `configs/experiments.json` and
records the resolved configuration in the output directory. The final-test stage
derives the checkpoint filename from `best_checkpoint_meta.json` and does not
accept a validation or test metric as a replacement selection source.

