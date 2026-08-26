"""Fail-fast RTX 5090 and MedSegDiff V1 CUDA preflight."""

import argparse
import json
import platform
import sys
from pathlib import Path

import torch


RELEASE_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = RELEASE_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT))

from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults


def tensor_devices(value):
    if torch.is_tensor(value):
        return [str(value.device)]
    if isinstance(value, (tuple, list)):
        result = []
        for item in value:
            result.extend(tensor_devices(item))
        return result
    if isinstance(value, dict):
        result = []
        for item in value.values():
            result.extend(tensor_devices(item))
        return result
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--required-name", default="NVIDIA GeForce RTX 5090")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU fallback is not permitted")
    if torch.cuda.device_count() < 1:
        raise RuntimeError("No CUDA devices were detected")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    name = torch.cuda.get_device_name(0)
    if name != args.required_name:
        raise RuntimeError(f"Expected {args.required_name!r} at cuda:0, found {name!r}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    matrix = torch.randn(1024, 1024, device=device, requires_grad=True)
    matrix_loss = (matrix @ matrix.transpose(0, 1)).square().mean()
    matrix_loss.backward()
    if matrix.grad is None or matrix.grad.device.type != "cuda":
        raise RuntimeError("CUDA backward did not produce a CUDA gradient")

    model_config = model_and_diffusion_defaults()
    model_config.update(
        image_size=256,
        num_channels=128,
        num_res_blocks=2,
        num_heads=1,
        in_ch=4,
        num_seg_classes=2,
        num_mask_channels=1,
        class_cond=False,
        learn_sigma=True,
        use_scale_shift_norm=False,
        attention_resolutions="16",
        diffusion_steps=1000,
        noise_schedule="linear",
        rescale_learned_sigmas=False,
        rescale_timesteps=False,
        version="1",
        use_fp16=False,
        dpm_solver=False,
    )
    model, _diffusion = create_model_and_diffusion(**model_config)
    model.to(device)
    model.eval()
    model_devices = sorted({str(parameter.device) for parameter in model.parameters()})
    if model_devices != ["cuda:0"]:
        raise RuntimeError(f"Project model parameters are not exclusively on cuda:0: {model_devices}")

    project_input = torch.randn(1, 4, 256, 256, device=device)
    timestep = torch.zeros(1, dtype=torch.long, device=device)
    with torch.inference_mode():
        project_output = model(project_input, timestep)
    output_devices = sorted(set(tensor_devices(project_output)))
    if output_devices != ["cuda:0"]:
        raise RuntimeError(f"Project model output left CUDA: {output_devices}")
    if torch.cuda.memory_allocated(device) <= 0:
        raise RuntimeError("CUDA reports zero allocated memory after project model forward")

    report = {
        "status": "PASS",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "device_name": name,
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "arch_list": torch.cuda.get_arch_list(),
        "device": str(device),
        "matrix_forward": True,
        "matrix_backward": True,
        "project_model": "MedSegDiff V1 binary 256x256",
        "project_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "model_parameter_devices": model_devices,
        "input_device": str(project_input.device),
        "output_devices": output_devices,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
        "amp_enabled": False,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "silent_cpu_fallback": False,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
