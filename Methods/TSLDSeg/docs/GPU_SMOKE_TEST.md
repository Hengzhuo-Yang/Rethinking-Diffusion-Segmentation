# RTX 5090 end-to-end GPU smoke-test record

Date: 2026-08-22. Runtime: Python 3.10.20, PyTorch 2.11.0+cu128,
CUDA 12.8, cuDNN 9.19.0, NVIDIA GeForce RTX 5090 (`sm_120`). Formal seed 23;
real preprocessed samples; formal batch size 12; smoke training batch size 2;
smoke validation/test batch size 1.

Every row performed all of the following on `cuda:0`:

1. instantiated the release model and read the fixed train manifest;
2. ran a real training forward, unchanged loss, backward, AdamW step, and
   LambdaLR step;
3. proved that TAM, HSEM, MCF, and the UNet path executed;
4. read the fixed validation manifest, ran a validation-loss forward, then
   computed `val_avg_dice` from the evaluation-label view of that same split;
5. selected and saved one temporary best checkpoint from that validation
   metric, then strictly reloaded its complete state;
6. read the fixed test manifest and ran test inference from the reloaded
   checkpoint;
7. removed the temporary checkpoint before continuing.

Diffusion rows used DDIM-10 for both validation and test. Core-no-diff used its
existing direct inference path. Model parameters, train tensors, loss, UNet
inputs, and UNet outputs were all asserted to be on `cuda:0`; no CPU model
fallback occurred.

| Dataset | Condition | Train loss | Parameters | UNet input | Peak CUDA MiB | Val smoke Dice | Test smoke Dice | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| BTCV | full diffusion | 1.917803 | 422,729,440 | 8 | 19,802.9 | 0.000000 | 0.000000 | PASS |
| BTCV | random Y_t | 9.645808 | 422,729,440 | 8 | 7,114.9 | 0.000000 | 0.000000 | PASS |
| BTCV | shuffled Y_t | 1.923901 | 422,729,440 | 8 | 7,114.9 | 0.000000 | 0.000000 | PASS |
| BTCV | core no diffusion | 8.240461 | 422,722,528 | 4 | 7,105.2 | 0.000000 | 0.000000 | PASS |
| ACDC | full diffusion | 1.800738 | 422,732,512 | 8 | 7,115.0 | 0.000000 | 0.000000 | PASS |
| ACDC | random Y_t | 10.031679 | 422,732,512 | 8 | 7,115.0 | 0.000000 | 0.000000 | PASS |
| ACDC | shuffled Y_t | 1.729989 | 422,732,512 | 8 | 7,115.0 | 0.000000 | 0.000000 | PASS |
| ACDC | core no diffusion | 8.061776 | 422,725,600 | 4 | 7,112.0 | 0.000000 | 0.000000 | PASS |
| ISIC2018 | full diffusion | 2.295659 | 422,729,440 | 8 | 7,114.9 | 0.000000 | 0.001640 | PASS |
| ISIC2018 | random Y_t | 9.645808 | 422,729,440 | 8 | 7,114.9 | 0.000000 | 0.000000 | PASS |
| ISIC2018 | shuffled Y_t | 2.707431 | 422,729,440 | 8 | 7,114.9 | 0.000000 | 0.159602 | PASS |
| ISIC2018 | core no diffusion | 6.985487 | 422,722,528 | 4 | 7,112.0 | 0.001884 | 0.185701 | PASS |

The first row includes one-time cuDNN benchmark/workspace allocation. Exact
checkpoint sizes ranged from 2,871,451,581 to 2,871,538,401 bytes; all were
strictly loaded and deleted. Full/random/shuffle parameter counts were
identical within every dataset. Core-no-diff intentionally has 6,912 fewer
parameters because its existing structural audit changes the UNet input
convolution from 8 to 4 channels.

For every shuffled-Y_t training pass, the no-self-match permutation was
`[1, 0]`; current timestep/noise were reused and Y_t shape, dtype, and device
matched the reference. All input-side audit traces reported
`target_changed=false`, `loss_changed=false`, and
`model_output_changed=false`. All temporary checkpoint and JSON artifacts were
removed after recording this table.

The Dice values above are execution checks from an upstream initialization
after only one optimizer step. They are not convergence results, benchmark
performance, or evidence for any audit conclusion. Full 50,000-step training,
validation selection across training, and one final full-split test remain the
formal experiment.
