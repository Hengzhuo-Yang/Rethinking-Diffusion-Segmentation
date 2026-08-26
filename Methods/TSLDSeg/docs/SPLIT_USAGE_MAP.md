# Split usage map

```text
fixed train manifest
        │
        ▼
50,000-step training ────────────────┐
                                     │ model state candidates
fixed validation manifest            ▼
        └── DDIM-10 or direct ── val_avg_dice ── select top-1 checkpoint
                                                        │
                                                        ▼
                                             explicit best checkpoint
                                                        │
fixed test manifest                                      ▼
        └── explicit slice2seg inference ── test_* metrics / final report
```

Code bindings:

| Stage | Dataset key | Metric namespace | Permitted purpose |
|---|---|---|---|
| Optimization | `train` | `train/*` | Gradient updates only |
| Validation loss | `validation` | `val/*` | No gradients; training-form target view |
| Model selection | `validation_metrics` | `val_avg_dice`, `val_avg_iou` | Evaluation-label view of the same val manifest; choose one best checkpoint |
| Final evaluation | explicit test loader/manifest | `test_avg_dice`, `test_avg_iou` | One final report after selection |

`LatentDiffusion.log_dice(data=None)` now reads
`trainer.datamodule.datasets["validation_metrics"]` and forces the `val`
prefix. `validation` and `validation_metrics` point to the same fixed IDs and
physical partition; the separate views retain the original [-1,1] training
target needed for validation loss while exposing full class-index labels for
Dice/IoU checkpoint selection.
`scripts/slice2seg.py` requires an explicit data directory, manifest, config,
and checkpoint and calls the same metric routine with `metric_prefix="test"`.

Formal training commands use `--no-test True`. If a caller deliberately enables
Lightning's test phase after training, `main.py` requests `ckpt_path="best"`;
the documented final segmentation report still uses the explicit inference CLI.
