# Split usage map

## Required information flow

```text
training partition
    -> optimizer updates
    -> validation partition only
    -> choose maximum val_avg_dice
    -> record best_checkpoint.json
    -> load that exact checkpoint
    -> one-time final test partition
```

Test metrics must not influence epoch, checkpoint, condition, seed, threshold,
preprocessing choice, or any other hyperparameter.

## Configuration mapping

| Dataset | Training dataset | Loss validation dataset | Metric validation dataset | Final test dataset |
|---|---|---|---|---|
| BTCV | `BTCVTrain` (`train`) | `BTCVValidation` (`validation`) | `BTCVValidationEval(split=validation)` | `BTCVValidationEval(split=test)` |
| ACDC | `ACDCTrain` (`train`) | `ACDCValidation` (`validation`) | `ACDCFullLabelSliceEval(split=validation)` | `ACDCFullLabelSliceEval(split=test)` |
| ISIC2018 | `ISIC2018Train` (`training`) | `ISIC2018Validation` (`validation`) | `ISIC2018ValidationEval` (`validation`) | `ISIC2018Test` (`testing`) |

Every formal experiment YAML contains all four data-module keys:
`train`, `validation`, `metric_validation`, and `test`.

## What each key may do

| Key | Allowed use | Forbidden use |
|---|---|---|
| `train` | gradient updates, training augmentation, training losses | validation selection or final claims |
| `validation` | Lightning validation loss/diagnostics | final reporting as test |
| `metric_validation` | periodic segmentation Dice and best-checkpoint selection | optimizer updates or final test claim |
| `test` | one-time inference after checkpoint selection | checkpoint/hyperparameter/condition selection |

`SDSeg.log_dice(data=None)` resolves `metric_validation`, not `test`. The
checkpoint callback monitors `val_avg_dice` in maximize mode and writes
`best_checkpoint.json` with `selection_partition=validation` and
`metric_dataset_key=metric_validation`.

## Evaluation CLI aliases

| CLI dataset | Partition | Intended use |
|---|---|---|
| `btcv-b-val` | BTCV validation | validation-only evaluation/diagnostics |
| `btcv-b` | BTCV test | final test only |
| `acdc-val` | ACDC validation | validation-only evaluation/diagnostics |
| `acdc` | ACDC test | final test only |
| `isic2018-val` | ISIC2018 validation | validation-only evaluation/diagnostics |
| `isic2018` | ISIC2018 testing | final test only |

The old BTCV quick/random-slice alias is not part of this map.

## Final-test gate

`scripts/slice2seg.py` requires both `--ckpt` and `--selection-metadata` and
checks that:

- selection partition is `validation`;
- metric dataset key is `metric_validation`;
- monitor is `val_avg_dice`;
- mode is `max`;
- `best_model_path` is present;
- the requested checkpoint is exactly the recorded best checkpoint;
- evaluation is requested once (`--times=1`);
- the output directory is empty;
- CUDA is available and the exact formal GPU name matches.

A missing or mismatched proof is a hard failure, not a warning.
