# GPU Environment

## Supported platform

This release has one supported execution target:

- Windows with the public `leaf-rtx5090` Conda environment
- NVIDIA GeForce RTX 5090
- CUDA-enabled PyTorch; CPU training and inference are not supported

Runtime evidence was captured with the pre-existing local `leaf` interpreter.
The reproducible public environment created by `environment.yml` is named
`leaf-rtx5090`; public commands below use that name. The historical upstream
PyTorch 2.4 environment is not the release baseline for the RTX 5090.

## Version baseline

The public RTX 5090 environment pins the following packages:

| Component | Version or build source |
| --- | --- |
| Conda environment | `leaf-rtx5090` |
| PyTorch | `2.7.0` from the CUDA 12.8 wheel index |
| torchvision | `0.22.0` |
| torchaudio | `2.7.0` |
| accelerate | `1.3.0` |
| diffusers | `0.31.0` |
| timm | `1.0.15` |
| NumPy | `2.3.0` |
| SciPy | `1.15.3` |
| OmegaConf | `2.3.0` |
| Pillow | `11.2.1` |

The Python patch version, NVIDIA driver, cuDNN build, and detected compute
capability below come from the strict runtime preflight rather than inference
from package filenames.

## Runtime evidence

| Field | Captured value |
| --- | --- |
| Strict preflight status | `PASS` |
| Python version | `3.11.11` |
| `torch.__version__` | `2.7.0+cu128` |
| `torch.version.cuda` | `12.8` |
| `torch.backends.cudnn.version()` | `90701` |
| NVIDIA driver | `591.86` |
| GPU name | `NVIDIA GeForce RTX 5090` |
| Reported GPU memory | `32607 MiB` |
| Device capability | `12.0` |
| PyTorch architecture support | includes `sm_120` |
| CUDA device count and selected index | `1`, index `0` (`cuda:0`) |

The preflight completed CUDA tensor matmul/backward and a convolutional model
forward, backward, and optimizer step. All model, tensor, loss, and gradient
device checks passed, and no CPU fallback was used.

Re-run with the public environment:

```powershell
conda activate leaf-rtx5090
python tests\gpu_preflight.py --device cuda:0
```

The command prints JSON and exits nonzero unless all of the following are true:

- CUDA is available;
- the selected device name is exactly `NVIDIA GeForce RTX 5090`;
- `nvidia-smi` reports the required GPU and driver;
- a CUDA tensor matrix multiplication and backward pass complete;
- a CUDA convolutional model completes forward, backward, and an optimizer
  step;
- model parameters, inputs, outputs, losses, and gradients remain on the
  requested CUDA device.

There is no CPU fallback.

## Precision and backend settings

All twelve formal configurations specify BF16 mixed precision. The project
does not define or compile a custom CUDA extension.

The training entry point explicitly sets both of these flags to `True` before
training and its in-process validation:

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

The standalone evaluation programs do not set either TF32 flag. They inherit
the fresh Python process defaults. Consequently, training/in-process
validation and standalone final evaluation do not have an explicitly identical
TF32 policy. The GPU smoke runner records the initial backend values, uses the
training policy for the training step and validation checkpoint selection, and
restores the initial values before the post-checkpoint test inference. It does
not silently make the two policies equal.

`torch.backends.cudnn.benchmark` and
`torch.backends.cudnn.deterministic` are not changed by the LEAF entry points;
their runtime values are recorded by the preflight. No deterministic mode is
claimed unless the runtime evidence reports it.

AMP behavior also follows the existing code:

- the training optimizer path uses the configuration's BF16 mixed precision;
- Core-No-Diff validation and standalone evaluation use an explicit BF16
  autocast context;
- full-diffusion in-training validation calls CUDA autocast without an explicit
  dtype and therefore uses the active PyTorch CUDA autocast default.

These are execution details of the reproduced implementation, not new method
contributions.

## Local resources

Pretrained LEAF VAE/U-Net weights and the DINOv2 cache are read-only inputs.
They are not included in this repository. Set `TORCH_HOME` to an existing
DINOv2 cache or pass a local DINOv2 checkout to the smoke runner. Do not copy
weights, datasets, or cache files into the release tree.

