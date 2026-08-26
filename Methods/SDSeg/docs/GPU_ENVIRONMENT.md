# Proven GPU environment

This document records the existing environment used as the local source of
truth. It is an observation, not an instruction to mutate that environment.

## Hardware and driver

| Field | Observed value |
|---|---|
| GPU | `NVIDIA GeForce RTX 5090` |
| GPUs visible | `1` |
| VRAM reported by `nvidia-smi` | `32607 MiB` |
| NVIDIA driver | `591.86` |
| Driver-reported CUDA compatibility | `13.1` |
| CUDA compute capability | `12.0` |

The driver-reported CUDA value is not the CUDA runtime bundled with PyTorch.
The installed PyTorch wheel is built for CUDA `12.8`.

## Core software

| Component | Observed value |
|---|---|
| Conda environment name | `sdseg` |
| Python | `3.10.20` |
| PyTorch | `2.11.0+cu128` |
| torchvision | `0.26.0+cu128` |
| `torch.version.cuda` | `12.8` |
| cuDNN runtime integer | `91900` (9.19.0) |
| PyTorch Lightning | `1.9.5` |
| torchmetrics | `0.11.4` |
| lightning-utilities | `0.15.3` |
| NumPy | `1.26.4` |
| Pillow | `12.2.0` |
| SciPy | `1.15.3` |
| OmegaConf | `2.3.0` |
| einops | `0.3.0` |
| kornia | `0.6.0` |
| albumentations | `2.0.8` |
| transformers | `4.19.2` |
| nibabel | `5.4.2` |
| h5py | `3.16.0` |
| pandas | `2.3.3` |
| matplotlib | `3.10.9` |

The focused install snapshot is at `requirements.txt`. The two editable
dependencies observed in the environment are pinned there by upstream revision:

- CompVis `taming-transformers` at
  `3ba01b241669f5ade541ce990f7650a3b8f65318`;
- OpenAI `CLIP` at `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`.

## CUDA/backend observations

| Setting | Observed/default value |
|---|---|
| CUDA available | true |
| Current CUDA device | `0` |
| Compiled architecture list | `sm_75`, `sm_80`, `sm_86`, `sm_90`, `sm_100`, `sm_120` |
| Matmul TF32 at plain import | false |
| cuDNN TF32 at plain import | true |
| cuDNN deterministic at plain import | false |
| cuDNN benchmark at plain import | false |
| Formal config `trainer.benchmark` | true |
| Explicit mixed precision | not enabled |
| Formal default floating dtype | float32 |

Record the effective settings again inside every run because framework startup
and trainer configuration can change backend flags after import.

## Fail-closed hardware policy

Formal entry points do not silently fall back to CPU. The default exact device
name is `NVIDIA GeForce RTX 5090`; a mismatch blocks a formal run. The GPU
preflight checks CUDA availability, device identity and capability, compiled
architecture support, formal configuration invariants, pretrained inputs, model
instantiation and placement, and a finite CUDA forward probe. The 12-condition
GPU smoke test separately adds backward execution, an optimizer update,
validation and test inference, the checkpoint save/reload cycle, split isolation,
and confirmation that no CPU fallback occurred.

## Environment caveats

- Both `opencv-python==4.11.0.86` and
  `opencv-python-headless==4.13.0.92` are installed in the observed environment;
  they provide overlapping `cv2` modules. The snapshot records this rather than
  silently choosing a different state.
- `scikit-image==0.24.0` appears in the broad environment inventory, but a direct
  import failed because `lazy_loader` was absent. The release runtime does not
  import scikit-image, so it is not listed as a validated runtime dependency.
- No environment creation or package update was performed as part of formal
  validation. New users should create `sdseg-release` separately from
  `environment.yml`.

These caveats must not be converted into a claim that an untested newly solved
environment is identical to the proven one.
