import argparse
import importlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT))

import torch

from score_sde.models.ncsnpp_generator_adagn import NCSNpp
from utils import EXPECTED_GPU_NAME, dev


def nvidia_driver_version():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip().splitlines()[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU fallback is disabled")
    if torch.cuda.device_count() < 1:
        raise RuntimeError("No CUDA devices were detected")
    device = dev(args.device)
    torch.cuda.reset_peak_memory_stats(device)

    left = torch.randn(1024, 1024, device=device, requires_grad=True)
    right = torch.randn(1024, 1024, device=device)
    matrix_output = (left @ right).square().mean()
    matrix_output.backward()
    if left.grad is None or left.grad.device != device:
        raise RuntimeError("CUDA backward did not produce a CUDA gradient")

    config = json.loads((CODE_ROOT / "parameters_btcv.json").read_text(encoding="utf-8-sig"))
    model = NCSNpp(SimpleNamespace(**config)).to(device)
    model.eval()
    noisy_mask = torch.randn(1, 1, 256, 256, device=device)
    timestep = torch.zeros(1, dtype=torch.long, device=device)
    condition = torch.randn(1, 1, 256, 256, device=device)
    latent = torch.randn(1, int(config["nz"]), device=device)
    with torch.no_grad():
        model_output = model(noisy_mask, timestep, condition, latent)
    if next(model.parameters()).device != device:
        raise RuntimeError("Project model parameters are not on the required CUDA device")
    if model_output.device != device:
        raise RuntimeError("Project model output is not on the required CUDA device")
    if torch.cuda.memory_allocated(device) <= 0:
        raise RuntimeError("CUDA reports no allocated memory after model execution")

    fused_module = importlib.import_module("score_sde.op.fused_act")
    upfirdn_module = importlib.import_module("score_sde.op.upfirdn2d")
    capability = torch.cuda.get_device_capability(device)
    report = {
        "status": "PASS",
        "operating_system": platform.platform(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "pytorch_cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_current_device": torch.cuda.current_device(),
        "cuda_device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "expected_gpu_name": EXPECTED_GPU_NAME,
        "gpu_capability": f"{capability[0]}.{capability[1]}",
        "torch_cuda_arch_list": torch.cuda.get_arch_list(),
        "nvidia_driver": nvidia_driver_version(),
        "default_dtype": str(torch.get_default_dtype()),
        "amp_enabled_by_project": False,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "matrix_forward": True,
        "matrix_backward": True,
        "model_parameter_device": str(next(model.parameters()).device),
        "model_input_device": str(condition.device),
        "model_output_device": str(model_output.device),
        "model_forward": True,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "fused_bias_act_extension": "compiled" if fused_module.fused is not None else "pytorch_cuda_fallback",
        "upfirdn2d_extension": "compiled" if upfirdn_module.upfirdn2d_op is not None else "pytorch_cuda_fallback",
    }
    if report["gpu_name"] != EXPECTED_GPU_NAME:
        raise RuntimeError(f"Unexpected GPU: {report['gpu_name']}")
    if args.json:
        output = Path(args.json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
