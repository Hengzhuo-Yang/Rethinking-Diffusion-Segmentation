# Split Usage Map

## Stage contract

| Stage | Allowed use | Forbidden use |
| --- | --- | --- |
| Train | Gradient updates, train-only augmentation, optimizer and EMA state | Evaluation claims from held-out test |
| Validation | CUDA inference, mean Dice, best-checkpoint selection | Gradient updates or final-test claims |
| Test | One CUDA evaluation after loading the recorded best checkpoint | Selection, tuning, early stopping, reranking, or best-of-test trials |

Runtime loaders read physically separated directories produced from fixed
manifests. `release_pipeline.py` enforces the higher-level state transition.

| Dataset | Task | Manifest | Physical directory | Loader / entry | Device | Use |
| --- | --- | --- | --- | --- | --- | --- |
| BTCV | Training | `btcv/train.txt` | `BTCV/train` | `BTCV` via `load_custom_dataset` | `cuda:0` | Optimization |
| BTCV | Selection | `btcv/val.txt` | `BTCV/val` | YAML `validation_split: val` and `train.py` | `cuda:0` | Mean validation Dice |
| BTCV | Final | `btcv/test.txt` | `BTCV/test` | `release_pipeline.py test` -> `evaluate_btcv_leaf.py` | `cuda:0` | Final Dice/IoU |
| ACDC | Training | `acdc/train.txt` | `ACDC/training` | `ACDC` via `load_custom_dataset` | `cuda:0` | Optimization |
| ACDC | Selection | `acdc/val.txt` | `ACDC/validation` | YAML `validation_split: validation` and `train.py` | `cuda:0` | Mean validation Dice |
| ACDC | Final | `acdc/test.txt` | `ACDC/testing` | `release_pipeline.py test` -> `evaluate_acdc_leaf.py` | `cuda:0` | Final class metrics |
| ISIC2018 | Training | `isic2018/train.txt` | `ISIC18/training` | `ISIC` via `load_custom_dataset` | `cuda:0` | Optimization |
| ISIC2018 | Selection | `isic2018/val.txt` | `ISIC18/validation` | YAML `validation_split: validation` and `train.py` | `cuda:0` | Mean validation Dice |
| ISIC2018 | Final | `isic2018/test.txt` | `ISIC18/testing` | `release_pipeline.py test` -> `evaluate_isic2018_leaf.py` | `cuda:0` | Final Dice/IoU |

Manifest paths are relative to `code/manifests/`.

## Enforcement

1. Preprocessors load `train.txt`, `val.txt`, and `test.txt` from
   repository-relative defaults and verify counts and disjointness.
2. `tests/validate_splits.py` locks exact IDs and can validate a cache.
3. YAML files name only the validation partition.
4. `train.py` checks the expected validation alias and saves only on improved
   validation Dice.
5. `best_checkpoint.json` records validation selection and metric.
6. `release_pipeline.py` verifies metadata and condition before resolving the
   checkpoint.
7. Final evaluation explicitly requests `test` or `testing` and will not
   overwrite `final_test/`.
8. Formal evaluators call `require_rtx5090`; model inference cannot fall back
   to CPU.

BTCV validation and test may originate from one historical held-out storage
pool, but manifest-filtered case roles are distinct. A parent-directory name
never overrides the manifest role.
