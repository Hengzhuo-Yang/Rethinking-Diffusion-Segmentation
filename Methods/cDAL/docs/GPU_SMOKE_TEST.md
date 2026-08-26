# RTX 5090 GPU smoke test

## Result

Preflight and all 12 dataset-condition combinations passed on 2026-08-22. The result is an execution and split-role check, not a performance claim.

Preflight evidence:

- GPU: NVIDIA GeForce RTX 5090, `cuda:0`, capability 12.0.
- CUDA tensor creation, 1024×1024 matrix multiplication, and backward: PASS.
- cDAL model parameters, condition input, noisy-mask input, and output on CUDA: PASS.
- cDAL model forward: PASS.
- Peak preflight allocated memory: 150,088,192 bytes.
- No CPU fallback.

## Combination matrix

| Dataset | Condition | Train batch | Forward | Loss | Backward | Optimizer | Validation/metric | Best save | Reload | Test inference | Audit branch | Peak MiB | Status |
|---|---|---:|---|---|---|---|---|---|---|---|---|---:|---|
| BTCV | Full | 1 | yes | CUDA | yes | yes | yes | yes | yes | yes | yes | 1875.2 | PASS |
| BTCV | Random-Yt | 1 | yes | CUDA | yes | yes | yes | yes | yes | yes | yes | 1875.4 | PASS |
| BTCV | Shuffle-Yt | 2 | yes | CUDA | yes | yes | yes | yes | yes | yes | yes | 3436.1 | PASS |
| BTCV | Core-No-Diff | 1 | yes | CUDA | yes | yes | yes | yes | yes | yes | yes | 1171.3 | PASS |
| ACDC | Full | 1 | yes | CUDA | yes | yes | yes | yes | yes | yes | yes | 1882.6 | PASS |
| ACDC | Random-Yt | 1 | yes | CUDA | yes | yes | yes | yes | yes | yes | yes | 1883.4 | PASS |
| ACDC | Shuffle-Yt | 2 | yes | CUDA | yes | yes | yes | yes | yes | yes | yes | 3459.8 | PASS |
| ACDC | Core-No-Diff | 1 | yes | CUDA | yes | yes | yes | yes | yes | yes | yes | 1172.3 | PASS |
| ISIC2018 | Full | 1 | yes | CUDA | yes | yes | yes | yes | yes | yes | yes | 1875.2 | PASS |
| ISIC2018 | Random-Yt | 1 | yes | CUDA | yes | yes | yes | yes | yes | yes | yes | 1875.4 | PASS |
| ISIC2018 | Shuffle-Yt | 2 | yes | CUDA | yes | yes | yes | yes | yes | yes | yes | 3436.1 | PASS |
| ISIC2018 | Core-No-Diff | 1 | yes | CUDA | yes | yes | yes | yes | yes | yes | yes | 1171.3 | PASS |

For every row, the model parameter, training input, training output, loss, validation input/output, and test input/output devices were `cuda:0`. Full/Random-Yt/Shuffle-Yt used the fixed four-step reverse path for validation and test; Core-No-Diff executed its actual direct core path. Random-Yt and Shuffle-Yt were exercised in their training scope and used unchanged Full inference as designed.

## Covered workflow

1. Load the fixed training manifest and a real preprocessed batch.
2. Execute the requested condition branch on the RTX 5090.
3. Compute the real clean-mask objective, backward, and optimizer step.
4. Load the fixed validation manifest, run model inference, and compute the configured Dice/IoU metric.
5. Promote the result through the normal best-checkpoint code and write selection metadata.
6. Construct a fresh model in the final-test process, validate metadata, and reload the checkpoint.
7. Load the fixed test manifest and perform test inference on the RTX 5090.

The smoke runner used one optimizer step, one validation item, one test item, ensemble size 1, and validation/test batch size 1. Only Shuffle-Yt required train batch size 2. Formal defaults remain 10,000 steps, ensemble size 5, and batch size 4.

## Reproduction command

```powershell
python tests/gpu_smoke_matrix.py --btcv-root data_preprocessed/btcv_synapse_cdal_binary_png --acdc-root data_preprocessed/acdc_mt_unet_cascade_cdal_png --isic-root data_preprocessed/isic2018_task1_cdal_png
```

## Cleanup

The runner removed its work directory after collection. No temporary checkpoint, medical image, per-item metric file, log, prediction, or copied dataset remains in the release. The reusable test program and this sanitized evidence remain. There are no unresolved smoke-test failures.
