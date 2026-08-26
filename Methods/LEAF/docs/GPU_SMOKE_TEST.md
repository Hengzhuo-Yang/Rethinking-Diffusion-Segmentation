# RTX 5090 GPU Smoke Test

## Purpose

`tests/gpu_smoke_12.py` is a minimal execution test, not a performance
benchmark and not a paper-result evaluation. It exercises the real LEAF model,
loss, DINOv2 alignment branch, audit branch, fixed data manifests, validation
selection rule, checkpoint serialization, and post-checkpoint inference on an
NVIDIA GeForce RTX 5090.

The runner always performs the strict preflight first and never switches to a
CPU model path.

## Coverage per combination

Each selected dataset-condition pair must complete:

1. fixed train manifest loading and a real training batch;
2. CUDA model/input forward and the configured LEAF loss;
3. CUDA backward, gradient clipping, AdamW step, LR-scheduler step, and EMA
   update when enabled;
4. execution of the selected Full, Random-Yt, Shuffle-Yt, or Core-No-Diff
   branch;
5. fixed validation manifest loading, CUDA forward, and mean Dice computation;
6. temporary best-checkpoint selection using validation Dice;
7. checkpoint save, deliberate in-memory parameter modification, reload, and
   exact parameter restoration check;
8. fixed test manifest loading and CUDA inference from the reloaded best
   checkpoint;
9. CUDA device evidence and peak allocated memory capture;
10. removal of the temporary checkpoint and per-combination work directory.

Random-Yt and Shuffle-Yt change only the training-time `Y_t` input. They retain
the original target, timestep, loss, validation, and inference behavior.
Shuffle-Yt uses a no-self-match batch permutation and therefore requires a
smoke batch of at least two. Core-No-Diff executes the real image-only latent
model with no `Y_t`, timestep, forward noising, or reverse sampler.

## Fixed manifests

All four conditions use the same files:

| Dataset | Train | Validation | Test |
| --- | --- | --- | --- |
| BTCV | `code/manifests/btcv/train.txt` | `code/manifests/btcv/val.txt` | `code/manifests/btcv/test.txt` |
| ACDC | `code/manifests/acdc/train.txt` | `code/manifests/acdc/val.txt` | `code/manifests/acdc/test.txt` |
| ISIC2018 | `code/manifests/isic2018/train.txt` | `code/manifests/isic2018/val.txt` | `code/manifests/isic2018/test.txt` |

The manifests contain canonical IDs only. The runner maps each real PNG sample
back to its case, patient, or image ID and fails if a manifest ID is absent.
Cross-split overlap is rejected before model execution.

The validated BTCV cache has physically separate `BTCV/val` and `BTCV/test`
directories. Validation contains only its two fixed cases, and test contains
only its ten fixed cases; the runner rejects missing or cross-split ownership.

## Commands

Run all twelve combinations:

```powershell
conda activate leaf-rtx5090
$env:TORCH_HOME = '<LOCAL_DINOV2_CACHE>'
python tests\gpu_smoke_12.py `
  --dataset all `
  --condition all `
  --data-root btcv='<BTCV_PREPROCESSED_ROOT>' `
  --data-root acdc='<ACDC_PREPROCESSED_ROOT>' `
  --data-root isic2018='<ISIC2018_PREPROCESSED_ROOT>' `
  --pretrained-root '<LEAF_ASSETS>' `
  --work-dir '<DEDICATED_TEMP_DIR>' `
  --device cuda:0
```

Run one combination:

```powershell
python tests\gpu_smoke_12.py `
  --dataset btcv `
  --condition shuffle-yt `
  --data-root btcv='<BTCV_PREPROCESSED_ROOT>' `
  --pretrained-root '<LEAF_ASSETS>' `
  --work-dir '<DEDICATED_TEMP_DIR>' `
  --device cuda:0
```

If torch hub cannot use the existing `TORCH_HOME`, pass
`--dinov2-repo '<LOCAL_DINOV2_CHECKOUT>'`. The DINOv2 model is real and its
alignment forward is mandatory; the runner does not substitute a stub.

The command prints one JSON report to standard output. Progress messages go to
standard error. A zero exit code means every selected combination is `PASS`.
Missing data, weights, DINOv2 resources, or GPU runtime evidence are reported as
`BLOCKED`; code or semantic assertion failures are reported as `FAIL`.

## Formal versus smoke parameters

The runner reads each formal YAML without writing it. Only the independent
smoke command shortens execution:

| Setting | Formal run | Smoke run |
| --- | --- | --- |
| Training length | YAML `max_train_steps` | one optimizer step |
| Training batch | YAML `train_batch_size` | 2 by default |
| Validation batch | YAML `test_batch_size` | 1 |
| Test batch | YAML `test_batch_size` | 1 |
| Workers | YAML `num_workers` | 0 |
| Validation batches | full validation manifest | one real manifest sample |
| Test batches | full test manifest | one real manifest sample |
| Diffusion inference steps | existing one-step LEAF validation/evaluation | 1 |
| Core-No-Diff inference steps | no sampler | 0 |

Model structure, pretrained initialization, target, timestep construction,
audit definition, L1 plus DINOv2 alignment loss, BF16 mode, seed, EMA policy,
validation Dice checkpoint selection, and test-after-best-load order are not
changed.

## Twelve-combination status

The final physical-cache run completed in 63.427 seconds: 12 PASS, 0 FAIL, and
0 BLOCKED. Every reported coverage flag was `true`, including real train,
validation, and test manifest reads; CUDA forward/loss/backward/optimizer/EMA;
validation Dice selection; checkpoint save/reload; post-reload test inference;
and temporary-artifact cleanup.

| Dataset | Condition | Forward/loss | Backward/optimizer | Validation/metric | Best save/load | Test inference | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BTCV | Full | PASS | PASS | PASS | PASS | PASS | PASS |
| BTCV | Random-Yt | PASS | PASS | PASS | PASS | PASS | PASS |
| BTCV | Shuffle-Yt | PASS | PASS | PASS | PASS | PASS | PASS |
| BTCV | Core-No-Diff | PASS | PASS | PASS | PASS | PASS | PASS |
| ACDC | Full | PASS | PASS | PASS | PASS | PASS | PASS |
| ACDC | Random-Yt | PASS | PASS | PASS | PASS | PASS | PASS |
| ACDC | Shuffle-Yt | PASS | PASS | PASS | PASS | PASS | PASS |
| ACDC | Core-No-Diff | PASS | PASS | PASS | PASS | PASS | PASS |
| ISIC2018 | Full | PASS | PASS | PASS | PASS | PASS | PASS |
| ISIC2018 | Random-Yt | PASS | PASS | PASS | PASS | PASS | PASS |
| ISIC2018 | Shuffle-Yt | PASS | PASS | PASS | PASS | PASS | PASS |
| ISIC2018 | Core-No-Diff | PASS | PASS | PASS | PASS | PASS | PASS |

Peak allocated CUDA memory was approximately 8014.9--8018.7 MiB for Full,
Random-Yt, and Shuffle-Yt, and 8157.6 MiB for Core-No-Diff.

## Cleanup

Each combination receives a uniquely named child directory under the explicit
`--work-dir`. Cleanup refuses to act if that directory is not a direct child of
the specified work root. The temporary best checkpoint is deleted with that
child directory. If the runner created an otherwise empty work root, it removes
that empty directory as well. A cleanup failure changes the combination status
to `FAIL`.

No smoke checkpoint, prediction, copied data, cache, log, or local absolute
path belongs in the public repository.

