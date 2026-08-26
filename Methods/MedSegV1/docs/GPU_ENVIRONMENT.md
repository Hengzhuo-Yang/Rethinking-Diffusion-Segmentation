# RTX 5090 environment

## Verified local environment

- Operating system: Windows build 26200 (64-bit; Python platform string `Windows-10-10.0.26200-SP0`)
- GPU: NVIDIA GeForce RTX 5090, device `cuda:0`
- GPU capability: 12.0 (`sm_120` present in the PyTorch architecture list)
- GPU memory reported by the driver: 32,607 MiB
- NVIDIA driver: 591.86
- Conda: 26.1.1
- Verified local environment name: `medsegv1`
- Python: 3.10.20
- PyTorch: 2.7.0+cu128
- torchvision: 0.22.0
- torchaudio: 2.7.0
- PyTorch CUDA runtime: 12.8
- cuDNN: 9.7.1 (`90701`)
- Model dtype: float32
- AMP: disabled
- TF32 matmul: disabled at runtime
- TF32 cuDNN: enabled at runtime
- cuDNN benchmark: disabled
- cuDNN deterministic: disabled
- Custom CUDA extensions: none found in this release

The `medsegv1` environment was selected because project creation scripts name it,
formal run commands invoke it, it contains the recorded PyTorch/CUDA versions,
and it passed CUDA tensor operations plus a formal MedSegDiff V1 forward on the
RTX 5090. The active shell environment was not used as selection evidence.

## Rebuild

The public environment uses a generic name and does not contain a local prefix:

```powershell
conda env create -f environment.yml
conda activate medsegdiff-audit-rtx5090
python -m pip install -r requirements-torch-cu128.txt
```

The CUDA PyTorch command uses the same official cu128 wheel channel and versions
as the verified local environment. The NVIDIA driver is a system prerequisite,
not a Conda or pip package.

Run the fail-fast check before training:

```powershell
python tests\gpu_preflight.py
```

This release is verified only for the recorded Windows, Conda and RTX 5090
configuration. CPU-only execution, other GPUs and the paper's historical
CUDA/PyTorch environment are outside the supported target and are not offered as
fallback paths.
