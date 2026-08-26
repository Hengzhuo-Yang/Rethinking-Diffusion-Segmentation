"""Strict CUDA runtime checks for the RTX 5090 release target."""

from __future__ import annotations

import torch


EXPECTED_GPU_NAME = "NVIDIA GeForce RTX 5090"
EXPECTED_CAPABILITY = (12, 0)


def require_rtx5090(device_index: int = 0) -> dict[str, object]:
    """Fail fast instead of silently falling back to CPU or another GPU."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this release has no CPU fallback")
    if device_index < 0 or device_index >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA device {device_index} is unavailable; found {torch.cuda.device_count()} device(s)"
        )
    name = torch.cuda.get_device_name(device_index)
    capability = tuple(torch.cuda.get_device_capability(device_index))
    if name != EXPECTED_GPU_NAME or capability != EXPECTED_CAPABILITY:
        raise RuntimeError(
            "This validated release targets exactly "
            f"{EXPECTED_GPU_NAME} compute capability {EXPECTED_CAPABILITY}; "
            f"found {name!r} capability {capability}"
        )
    device = torch.device("cuda", device_index)
    probe = torch.randn((32, 32), device=device, requires_grad=True)
    loss = (probe @ probe.T).square().mean()
    loss.backward()
    torch.cuda.synchronize(device)
    return {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_name": name,
        "compute_capability": list(capability),
        "device_index": device_index,
        "cuda_probe_backward_passed": probe.grad is not None,
    }
