# GPU smoke-test protocol

GPU smoke tests are release gates, not substitutes for full training. A smoke
run may shorten dataset length or batch count, but it must preserve the formal
split identities, audit branch, CUDA-only behavior, checkpoint-selection
direction, and direct one-step evaluation semantics.

## Required test sequence per matrix row

Each of the 12 dataset/condition rows must independently demonstrate:

1. the training manifest is the fixed formal manifest;
2. a training batch loads (at least two samples for Shuffle-Yt);
3. the model, inputs, targets, loss, and outputs are on CUDA;
4. forward, finite loss, backward, and one optimizer update complete;
5. the validation manifest is the fixed validation manifest;
6. validation forward and metric computation complete;
7. a temporary best-checkpoint artifact can be saved and reloaded, with metadata
   proving selection by maximum validation Dice;
8. the fixed test manifest loads and test inference completes;
9. the configured audit branch is observed and matches the row;
10. there is no train/validation/test leakage and no CPU fallback.

The temporary checkpoint must be deleted after the row. No smoke checkpoint,
medical image, prediction, or result is part of the source release.

## Commands

Run from `code/` after preparing the local data and pretrained roots:

```bash
python tests/test_release_static.py -v
python tests/gpu_preflight.py --help
python tests/gpu_preflight.py \
  --config configs/experiments/btcv/full.yaml \
  --pretrained-root <PRETRAINED_ROOT> \
  --cuda-device 0 \
  --expected-gpu-name "NVIDIA GeForce RTX 5090"
python tests/validate_splits.py --data-root <DATA_ROOT>
python tests/gpu_smoke_12.py --help
python tests/gpu_smoke_12.py \
  --data-root <DATA_ROOT> \
  --pretrained-root <PRETRAINED_ROOT> \
  --report <LOCAL_REPORT_DIR>/gpu-smoke-12.json \
  --max-eval-items 1 \
  --expected-gpu-name "NVIDIA GeForce RTX 5090"
```

The command runs all 12 rows by default and writes the required JSON report
atomically. `--max-eval-items=1` limits validation/test work for smoke only; it
does not modify the formal YAMLs. The optional `--only-dataset` and
`--only-condition` filters are for diagnosis and do not establish a 12-row pass.
Do not infer a pass from the script's presence or from a prior report against a
different release tree.

## Authoritative release run

The final full-matrix smoke run started at `2026-07-20T06:19:41Z` and completed
at `2026-07-20T06:21:25Z`. Its report records:

| Field | Recorded value |
|---|---|
| GPU | `NVIDIA GeForce RTX 5090` |
| Compute capability | `12.0` |
| PyTorch | `2.11.0+cu128` |
| PyTorch CUDA runtime | `12.8` |
| Matrix result | 12 completed, 12 `pass`, 0 `fail` |
| Completion flag | `all_12_passed=true` |
| Smoke flag | `is_smoke_test=true` |
| CPU fallback | `cpu_fallback=false` |
| Per-row limits | train `2`, validation `1`, test `1` |
| Validation/test inference | `direct`, requested steps `1` |
| Maximum row peak CUDA allocation | `8.67 GiB` |
| Sum of row `elapsed_seconds` | `102.72 s` |

All 12 rows passed the CUDA batch placement, finite loss, backward, finite
gradient, optimizer-step, audit-branch, validation inference, strict
full-state checkpoint reload, temporary-file cleanup, test inference,
validation-only selection, and test-isolation checks.

The elapsed sum and peak allocation describe this smoke validation only; they
are not performance benchmarks. The local generated report has SHA-256
`d2221ad34f3829f9598e09529ba4b0b8c8d7b6c85c6ce91a01684d5fa9545bc6`.
The report was generated and verified locally, and its SHA-256 is recorded
above. The original JSON was deleted after its evidence was summarized and was
never included in the source-code release package.

## Evidence schema

The JSON report uses lowercase `pass` and `fail` status values. Report-level
fields are:

| Field | Recorded content |
|---|---|
| `status` | overall lowercase `pass` or `fail` |
| `is_smoke_test`, `cpu_fallback`, `seed` | smoke/run boundary flags and seed |
| `started_utc`, `completed_utc` | report time window |
| `expected_combinations`, `full_release_matrix` | requested matrix scope |
| `completed_combinations`, `passed_combinations`, `failed_combinations`, `all_12_passed` | completion summary |
| `gpu` | `name`, `capability`, `torch_version`, and `torch_cuda_runtime` |
| `split_validation` | bundled and processed split validation report |
| `results` | the row records described below |

Each successful row contains the actual fields below:

| Field | Recorded content |
|---|---|
| `status`, `is_smoke_test`, `cpu_fallback`, `seed` | lowercase status and row boundary flags |
| `dataset`, `condition`, `audit_mode` | registered matrix identity and observed branch |
| `config`, `config_sha256` | release-relative configuration and its SHA-256 |
| `formal_parameters` | fixed seed, batch size, workers, max steps, learning rate, and LR-scaling flag |
| `smoke_limits` | `train_items`, `validation_items`, and `test_items` |
| `dataset_lengths` | full `train`, `metric_validation`, and `test` loader lengths |
| `manifest_sha256` | hashes for `train`, `metric_validation`, and `test` manifests |
| `train` | loss, loss dictionary, gradient count, optimizer name/step, CUDA loss device, and batch tensor devices |
| `audit` | observed mode flags and condition-specific branch evidence |
| `validation`, `test` | item count, sampler, requested steps, and finite Dice/IoU summaries |
| `checkpoint` | full-model-state scope, size, key count, strict reload, validation metadata, and cleanup flag |
| `selection_partition`, `final_partition`, `test_used_for_selection` | validation/test isolation evidence |
| `cuda_peak_memory_bytes`, `elapsed_seconds` | per-row resource observations |

A failed row instead records lowercase `status=fail`, its matrix/config identity,
`error_type`, `error`, and `cuda_peak_memory_bytes`. There are no independent
report fields named `device_name`, `train_manifest`, `validation_manifest`, or
`test_manifest`; device evidence is nested under `gpu`/`train`, while manifest
evidence is represented by `split_validation`, `manifest_sha256`, and
`dataset_lengths`.

## Matrix completion table

The authoritative status comes from the report identified above.

| Dataset | Full | Random-Yt | Shuffle-Yt | Core-No-Diff |
|---|---|---|---|---|
| BTCV | PASS | PASS | PASS | PASS |
| ACDC | PASS | PASS | PASS | PASS |
| ISIC2018 | PASS | PASS | PASS | PASS |

For this authoritative run, all 12 row statuses are lowercase `pass`, the
report status is `pass`, and `all_12_passed=true`. Future modifications require
a new complete report; a missing, stale, filtered, or failing report cannot
inherit this result.

## Formal-run boundary

A smoke limit must be recorded as `is_smoke_test=true`. Smoke metrics cannot be
reported as formal validation or test performance. Full experiments still
require `100000` maximum steps, the complete partitions, validation-only best
checkpoint selection, and one-time final test evaluation.
