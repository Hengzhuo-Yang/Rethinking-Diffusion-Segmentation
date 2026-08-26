# Split Usage Map

The public protocol is:

    fixed training manifest
        -> optimization checkpoints
        -> fixed validation manifest
        -> best_checkpoint.json + best_model.pt
        -> fixed final-test manifest
        -> samples and final metrics

There is no arrow from final-test metrics back to model choice, hyperparameter
choice, stopping time, seed choice, or condition definition.

## Partition responsibilities

| Partition | Allowed uses | Forbidden uses | Enforced by |
| --- | --- | --- | --- |
| Training | Gradient updates, augmentation, optimizer state | Reported final evaluation | Training manifest passed to the dataset loader |
| Validation | Periodic inference and Dice-based checkpoint selection | Gradient updates; final claims presented as test | Validation manifest, deterministic validation seed, TrainLoop best-checkpoint metadata |
| Final test | One post-selection sampling/evaluation pass | Checkpoint selection, early stopping, tuning, reranking, repeated best-of-test trials | Test manifest, explicit best_model.pt load, metadata checks, non-overwriting final_test directory |

## Exact stage-to-loader map

All manifest paths below are repository-relative under `code/manifests/`.
`release_pipeline.py` derives separate `training/`, `validation/`, and
`testing/` paths from `--data-root`; it never reuses one ambiguous evaluation
path. Every model stage first passes the RTX 5090 preflight.

| Dataset | Task | Partition | Manifest and path fields | Dataloader | Entry point | Model device | Allowed use |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BTCV | Training | train | `btcv/train_cases.txt`; `--data-root/training` → `--data_dir`; `--train_manifest` | `BTCVDataset(test_flag=False, manifest_path=...)` | `release_pipeline.py train` → `segmentation_train.py` | `cuda:0` RTX 5090 | Gradient update and training statistics |
| BTCV | Selection | val | `btcv/val_cases.txt`; `--data-root/validation` → `--val_data_dir`; `--val_manifest` | `BTCVDataset(test_flag=False, manifest_path=...)` in `create_validation_fn` | periodic validation inside `segmentation_train.py` | `cuda:0` RTX 5090 | Mean validation Dice and best-checkpoint selection only |
| BTCV | Final inference | test | `btcv/test_cases.txt`; `--data-root/testing` → sample `--data_dir`; `--manifest` | `BTCVDataset(test_flag=True, manifest_path=...)` | `release_pipeline.py test` → `segmentation_sample.py` → `evaluate_btcv_samples.py` | `cuda:0` RTX 5090; saved-tensor metric aggregation on CPU | One final inference/evaluation after loading `best_model.pt` |
| ACDC | Training | train | `acdc/train_patients.txt`; `--data-root/training` → `--data_dir`; `--train_manifest` | `ACDCDataset(test_flag=False, manifest_path=...)` | `release_pipeline.py train` → `segmentation_train.py` | `cuda:0` RTX 5090 | Gradient update and training statistics |
| ACDC | Selection | val | `acdc/val_patients.txt`; `--data-root/validation` → `--val_data_dir`; `--val_manifest` | `ACDCDataset(test_flag=False, manifest_path=...)` in `create_validation_fn` | periodic validation inside `segmentation_train.py` | `cuda:0` RTX 5090 | Mean validation Dice and best-checkpoint selection only |
| ACDC | Final inference | test | `acdc/test_patients.txt`; `--data-root/testing` → sample `--data_dir`; `--manifest` | `ACDCDataset(test_flag=True, manifest_path=...)` | `release_pipeline.py test` → `segmentation_sample.py` → `evaluate_acdc_samples.py` | `cuda:0` RTX 5090; saved-tensor metric aggregation on CPU | One final inference/evaluation after loading `best_model.pt` |
| ISIC2018 | Training | train | `isic2018/training.txt`; `--data-root/training` → `--data_dir`; `--train_manifest` | `ISIC2018Dataset(test_flag=False, manifest_path=...)` | `release_pipeline.py train` → `segmentation_train.py` | `cuda:0` RTX 5090 | Gradient update and training statistics |
| ISIC2018 | Selection | val | `isic2018/validation.txt`; `--data-root/validation` → `--val_data_dir`; `--val_manifest` | `ISIC2018Dataset(test_flag=False, manifest_path=...)` in `create_validation_fn` | periodic validation inside `segmentation_train.py` | `cuda:0` RTX 5090 | Mean validation Dice and best-checkpoint selection only |
| ISIC2018 | Final inference | test | `isic2018/testing.txt`; `--data-root/testing` → sample `--data_dir`; `--manifest` | `ISIC2018Dataset(test_flag=True, manifest_path=...)` | `release_pipeline.py test` → `segmentation_sample.py` → `evaluate_isic2018_samples.py` | `cuda:0` RTX 5090; saved-tensor metric aggregation on CPU | One final inference/evaluation after loading `best_model.pt` |

## Training and checkpoint selection

release_pipeline.py passes both the training and validation manifests to
segmentation_train.py. The training loader can see only training IDs. At every
configured validation interval, TrainLoop snapshots the Python, NumPy, PyTorch,
and CUDA random-number-generator states, runs validation with the fixed
validation seed, restores the training states, and compares the returned Dice
value with the previous best.

On improvement, TrainLoop writes:

- best_model.pt: the selected model state;
- best_checkpoint.json: selected step, metric name/value, source checkpoint,
  validation partition, validation manifest, training and validation seeds,
  and condition.

Ordinary savedmodel and EMA files remain restart artifacts. The public final
test does not discover or rank them.

## Final test gate

Before sampling, release_pipeline.py requires:

1. best_checkpoint.json exists;
2. selection_partition is exactly validation;
3. checkpoint is exactly best_model.pt;
4. the recorded condition matches the requested condition;
5. the recorded validation manifest is this release's fixed manifest;
6. best_model.pt exists;
7. the fixed test manifest and preprocessed testing directory exist; and
8. the final_test directory is empty.

It then calls segmentation_sample.py with an explicit model path, test manifest,
condition, and seed, followed by the dataset-specific evaluator. The pipeline
records a final_test_request.json status artifact. A non-empty final_test
directory is not overwritten; use a new run root for a genuinely independent
run.

## Excluded leakage paths

- BTCV's historic 10-percent/quick held-out subset is not published as an
  entry point and is rejected by the fixed preprocessor.
- ACDC and ISIC2018 validation data are not relabeled as final test.
- The dedicated different-sampling-steps audit is outside this release.
- Historic private checkpoints chosen with obsolete partitions are not shipped
  and must not be reported as results from this corrected protocol.
