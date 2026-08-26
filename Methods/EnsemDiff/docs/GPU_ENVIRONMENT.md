# GPU Environment

## Supported platform

This release has one supported execution target:

- Windows
- Conda
- NVIDIA GeForce RTX 5090

CPU execution, non-NVIDIA accelerators, other NVIDIA GPU models, Linux, macOS, and pip-only environments are outside the supported release configuration. Formal training, validation, and testing must stop with an error if CUDA is unavailable or the detected GPU is not an NVIDIA GeForce RTX 5090. They must never continue by silently falling back to the CPU.

CPU work remains normal for file decoding, preprocessing, dataloader workers,
checkpoint serialization, and metric aggregation. The prohibition concerns a
CPU substitute for model forward, loss/backward, optimizer parameters,
validation inference, final inference, or reverse diffusion.

## Create the environment

From the repository root, create the pinned Conda environment:

```powershell
conda env create -f environment.yml
conda activate ensemdiff
```

The root `environment.yml` is the canonical Conda specification. It creates the `ensemdiff` environment with Python 3.10.20 and installs the root `requirements.txt`. The requirements file selects the official PyTorch CUDA 12.8 wheel index and pins the public Python dependencies.

Before running any formal experiment, execute the GPU preflight from the repository root:

```powershell
python tests/gpu_preflight.py
```

A failed preflight is a hard stop. Do not bypass it by selecting the CPU or a different GPU model.

## Why this environment was selected

Local training launchers, reproduction notes, and successful BTCV, ACDC, and
ISIC2018 run records consistently select the Conda environment named
`ensemdiff`. That evidence was checked before considering the currently active
shell or unrelated environments. A read-only probe of `ensemdiff` then passed
project imports, exact RTX 5090 detection, CUDA arithmetic/backward, and the
formal model forward on the first evidence-backed candidate. No environment
was created, upgraded, downgraded, or otherwise modified during verification.

## Verified configuration

The following configuration was measured directly on the supported machine on 2026-07-19:

| Component | Verified value |
| --- | --- |
| Operating system | Microsoft Windows 11 Pro, build 26200 |
| Conda | 26.1.1; environment `ensemdiff` |
| GPU | NVIDIA GeForce RTX 5090 |
| GPU count | 1 |
| Physical / logical CUDA index | 0 / `cuda:0` |
| Compute capability | 12.0 |
| PyTorch architecture support | `sm_120` present |
| NVIDIA driver | 591.86 |
| Driver-reported CUDA compatibility | 13.1 |
| Python | 3.10.20 |
| PyTorch | 2.7.0+cu128 |
| torchvision | 0.22.0+cu128 |
| torchaudio | 2.7.0+cu128 |
| PyTorch CUDA runtime | 12.8 |
| cuDNN | 9.7.1 (`90701`) |

The CUDA 13.1 value displayed by `nvidia-smi` is the maximum CUDA compatibility level reported by the installed NVIDIA driver. It is not the CUDA runtime embedded in PyTorch. This release uses the PyTorch CUDA 12.8 wheels, so `torch.version.cuda` correctly reports 12.8.

## Numerical execution policy

The formal train, validation, and test paths use FP32 (`use_fp16=False`). The measured process defaults were:

| Setting | Verified value |
| --- | --- |
| CUDA AMP autocast available | `True` |
| CUDA AMP autocast enabled | `False` |
| Default CUDA autocast dtype | `float16` |
| `torch.backends.cuda.matmul.allow_tf32` | `False` |
| `torch.backends.cudnn.allow_tf32` | `True` |
| `torch.backends.cudnn.benchmark` | `False` |
| `torch.backends.cudnn.deterministic` | `False` |
| Deterministic algorithms enabled | `False` |
| Float32 matmul precision | `highest` |

These values describe the verified release environment. In particular, the project does not claim bitwise-deterministic CUDA results. Changing precision, AMP, TF32, cuDNN, or deterministic-algorithm settings creates a different execution configuration and must be documented with the resulting experiment.

The public orchestrator sets `CUDA_VISIBLE_DEVICES` from `--cuda-device`, so
the selected physical device becomes logical `cuda:0`. It also sets
`PYTHONDONTWRITEBYTECODE=1` to avoid release-tree caches and preserves the local
Windows compatibility setting `KMP_DUPLICATE_LIB_OK=TRUE`. These process
settings do not enable CPU fallback or change AMP/TF32/dtype policy.

## CUDA implementation boundary

The repository contains no custom CUDA or C++ extension sources and no extension build hooks. It does not invoke `nvcc`, `torch.utils.cpp_extension`, Triton, or CuPy to build project-specific GPU kernels. RTX 5090 and `sm_120` support therefore comes from the pinned official PyTorch CUDA 12.8 wheels.

## Validation evidence and scope

The supported environment passed both of the following CUDA-only checks without a CPU fallback:

- An FP32 CUDA matrix multiplication and backward pass with finite, nonzero gradients.
- A release-project import and a forward pass through the formal BTCV Full UNet configuration on a 224×224 CUDA input, with a finite CUDA FP32 output.

This document records the environment measured on 2026-07-19. It is the reproducible public release target, but it is not a claim that every earlier private experiment was executed with an identically frozen package, driver, or runtime state. Historical logs that do not record complete environment metadata cannot establish exact historical versions.
