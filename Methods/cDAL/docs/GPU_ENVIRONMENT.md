# Validated GPU environment

## Selection evidence

The source notes mentioned a proposed `cdal5090` environment, but that environment was not present. Completed formal run logs identified `medsegv1`, and direct checks in that environment successfully imported the project, detected the RTX 5090, executed CUDA operations, and ran the cDAL model. It was therefore selected under the evidence priority rule. Merely active shell environments were not used as proof.

The release environment name `cdal-rtx5090` is generic; it reconstructs the required subset of the validated `medsegv1` packages without exposing a local installation prefix.

## Recorded values

| Field | Validated value |
|---|---|
| Operating system | Windows 10 API build `10.0.26200` |
| Conda | 26.1.1 |
| Validation environment | `medsegv1` |
| Python | 3.10.20 |
| PyTorch | 2.7.0+cu128 |
| PyTorch CUDA runtime | 12.8 |
| cuDNN | 90701 (9.7.1) |
| NVIDIA driver | 591.86 |
| `nvidia-smi` maximum CUDA compatibility | 13.1 |
| Device count / index | 1 / `cuda:0` |
| GPU | NVIDIA GeForce RTX 5090 |
| Capability | 12.0, queried at runtime |
| PyTorch architecture list | `sm_50`, `sm_60`, `sm_61`, `sm_70`, `sm_75`, `sm_80`, `sm_86`, `sm_90`, `sm_100`, `sm_120` |
| Default/project dtype | float32 |
| AMP / GradScaler | not used |
| Matrix-multiply TF32 | false |
| cuDNN TF32 | true |
| cuDNN benchmark | false |
| cuDNN deterministic | false |
| Deterministic algorithms | false |

These flags were read from the runtime; the release does not silently enable or disable them.

## Optional CUDA extensions

The repository contains CUDA/C++ sources for fused bias activation and upfirdn2d. On the validated Windows installation, both optional JIT extensions were unavailable and the locally retained PyTorch fallback executed on CUDA tensors. Preflight and all 12 smoke combinations passed with:

- `fused_bias_act_extension = pytorch_cuda_fallback`
- `upfirdn2d_extension = pytorch_cuda_fallback`

This is not a CPU model fallback. Inputs, parameters, outputs, losses, backward, and optimizer updates stayed on `cuda:0`.

## Creation and activation

```powershell
conda env create -f environment.yml
conda activate cdal-rtx5090
python tests/gpu_preflight.py
```

Equivalent explicit PyTorch installation:

```powershell
conda create -n cdal-rtx5090 python=3.10.20 pip=26.1.2 -c conda-forge
conda activate cdal-rtx5090
python -m pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python tests/gpu_preflight.py
```

NVIDIA drivers are host software and are intentionally absent from Conda/Python dependency files.

## Scope statement

This release is validated only for the user's Windows + Conda + PyTorch CUDA environment on an NVIDIA GeForce RTX 5090. CPU-only execution, silent CPU fallback, the paper authors' older Quadro RTX 6000 environment, other GPUs, and other CUDA/PyTorch combinations are not validated and are not adaptation targets.
