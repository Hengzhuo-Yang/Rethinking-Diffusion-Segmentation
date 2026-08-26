# Validated GPU environment

The release target is the single local environment that was actually exercised.
No CPU or cross-platform fallback is claimed.

| Item | Validated value |
|---|---|
| Operating system | Windows |
| Existing Conda environment used for validation | `sdseg` |
| Python | 3.10.20 |
| PyTorch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| PyTorch CUDA runtime | 12.8 |
| cuDNN | 9.19.0 (`91900`) |
| GPU | NVIDIA GeForce RTX 5090 |
| Compute capability | 12.0 (`sm_120`) |
| `torch.cuda.get_arch_list()` | `sm_75`, `sm_80`, `sm_86`, `sm_90`, `sm_100`, `sm_120` |
| `torch.cuda.is_available()` | `true` |
| CUDA device count/current index | 1 / 0 |
| GPU memory reported by driver | 32,607 MiB |
| NVIDIA driver | 591.86 |
| `nvidia-smi` CUDA compatibility | 13.1 |
| `nvidia-smi` result | device 0 identified as NVIDIA GeForce RTX 5090; query succeeded |

The driver-reported CUDA compatibility level and PyTorch's compiled CUDA
runtime are different fields; PyTorch in this environment is the cu128 build.

Important package versions are pinned in `requirements.txt`. The environment
also pins the exact locally inspected commits of taming-transformers and CLIP.

## Precision and backend behavior

- Training configs do not set Lightning AMP/precision; formal training is the
  PyTorch Lightning 1.9.5 default FP32 path.
- Segmentation validation/inference in `log_dice` uses CUDA autocast, preserving
  the validated source behavior.
- Configs set `benchmark: true`; Lightning enables cuDNN benchmarking during
  training.
- At plain PyTorch import on the validated environment:
  `cuda.matmul.allow_tf32=false`, `cudnn.allow_tf32=true`, and
  `cudnn.deterministic=false`.
- No custom CUDA extension is built by this release.

The pre-existing `sdseg` environment did not contain pytest and was not modified.
The 25 pure unit/static checks used pytest 8.4.2 from the machine's existing test
tooling while running under the `sdseg` Python 3.10.20 interpreter and importing
the project/runtime dependencies from `sdseg`. The release requirements include
pytest so a newly reconstructed environment has the runner directly. All CUDA
preflight and 12-cell model execution used the unmodified `sdseg` environment.

`gpu_preflight.py` verifies exact device name/capability and performs a CUDA
matrix multiplication plus backward pass. The training and inference entry
points repeat the same fail-fast gate.
