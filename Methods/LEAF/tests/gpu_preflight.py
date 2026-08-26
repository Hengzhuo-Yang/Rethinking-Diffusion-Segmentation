"""Strict CUDA preflight for the supported LEAF release environment."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


EXPECTED_GPU_NAME = "NVIDIA GeForce RTX 5090"


def _nvidia_smi_inventory() -> list[dict[str, str]]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise RuntimeError("nvidia-smi is required but was not found on PATH")
    command = [
        executable,
        "--query-gpu=uuid,name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    records: list[dict[str, str]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            raise RuntimeError(f"Unexpected nvidia-smi record: {line!r}")
        records.append(
            {
                "uuid": fields[0],
                "name": fields[1],
                "driver_version": fields[2],
                "memory_total_mib": fields[3],
            }
        )
    if not records:
        raise RuntimeError("nvidia-smi returned no GPU records")
    return records


def _cuda_device(device_spec: str) -> torch.device:
    device = torch.device(device_spec)
    if device.type != "cuda":
        raise ValueError(f"CPU fallback is forbidden; expected a CUDA device, got {device_spec!r}")
    index = torch.cuda.current_device() if device.index is None else device.index
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError(
            f"CUDA device index {index} is outside the detected range "
            f"0..{torch.cuda.device_count() - 1}"
        )
    return torch.device("cuda", index)


def _all_parameter_devices(module: nn.Module) -> list[str]:
    return sorted({str(parameter.device) for parameter in module.parameters()})


def run_preflight(device_spec: str = "cuda:0") -> dict[str, Any]:
    """Run the strict preflight and return a JSON-serializable result."""

    started = time.time()
    result: dict[str, Any] = {
        "schema_version": 1,
        "test": "leaf_gpu_preflight",
        "expected_gpu_name": EXPECTED_GPU_NAME,
        "requested_device": device_spec,
        "status": "FAIL",
        "stage": "environment",
        "checks": {
            "cuda_available": False,
            "exact_gpu_name": False,
            "nvidia_smi": False,
            "tensor_matmul": False,
            "tensor_backward": False,
            "tiny_model_forward": False,
            "tiny_model_backward": False,
            "tiny_model_optimizer_step": False,
            "no_cpu_fallback": True,
        },
    }

    try:
        result["environment"] = {
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version.split()[0],
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "pytorch": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "tf32_cuda_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
        }
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; CPU execution is not permitted")
        result["checks"]["cuda_available"] = True

        device = _cuda_device(device_spec)
        torch.cuda.set_device(device)
        properties = torch.cuda.get_device_properties(device)
        device_name = torch.cuda.get_device_name(device)
        capability = torch.cuda.get_device_capability(device)
        arch_list = torch.cuda.get_arch_list() if hasattr(torch.cuda, "get_arch_list") else []
        result["device"] = {
            "index": device.index,
            "current_device": torch.cuda.current_device(),
            "device": str(device),
            "name": device_name,
            "capability": list(capability),
            "architecture_list": arch_list,
            "total_memory_bytes": int(properties.total_memory),
            "multi_processor_count": int(properties.multi_processor_count),
        }
        if device_name != EXPECTED_GPU_NAME:
            raise RuntimeError(
                f"Unsupported GPU {device_name!r}; this release requires exactly {EXPECTED_GPU_NAME!r}"
            )
        result["checks"]["exact_gpu_name"] = True

        inventory = _nvidia_smi_inventory()
        result["nvidia_smi"] = inventory
        if not any(record["name"] == EXPECTED_GPU_NAME for record in inventory):
            raise RuntimeError(
                f"nvidia-smi did not report the required GPU {EXPECTED_GPU_NAME!r}"
            )
        result["checks"]["nvidia_smi"] = True

        torch.cuda.reset_peak_memory_stats(device)
        torch.manual_seed(1337)
        torch.cuda.manual_seed_all(1337)

        result["stage"] = "tensor_matmul_backward"
        lhs = torch.randn(256, 256, device=device, dtype=torch.float32, requires_grad=True)
        rhs = torch.randn(256, 256, device=device, dtype=torch.float32, requires_grad=True)
        product = lhs @ rhs
        if product.device != device or not torch.isfinite(product).all():
            raise RuntimeError("CUDA matrix multiplication produced an invalid result or wrong device")
        result["checks"]["tensor_matmul"] = True
        matmul_loss = product.square().mean()
        matmul_loss.backward()
        if lhs.grad is None or rhs.grad is None:
            raise RuntimeError("CUDA matrix multiplication backward did not create gradients")
        if lhs.grad.device != device or rhs.grad.device != device:
            raise RuntimeError("CUDA matrix multiplication gradients left the requested device")
        if not torch.isfinite(lhs.grad).all() or not torch.isfinite(rhs.grad).all():
            raise RuntimeError("CUDA matrix multiplication backward produced non-finite gradients")
        result["checks"]["tensor_backward"] = True
        result["tensor_evidence"] = {
            "lhs_device": str(lhs.device),
            "rhs_device": str(rhs.device),
            "output_device": str(product.device),
            "loss_device": str(matmul_loss.device),
            "gradient_device": str(lhs.grad.device),
        }

        result["stage"] = "tiny_model"
        model = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(8, 4, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(4, 2),
        ).to(device)
        if _all_parameter_devices(model) != [str(device)]:
            raise RuntimeError(f"Tiny model parameters are not exclusively on {device}")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
        inputs = torch.randn(2, 3, 32, 32, device=device)
        targets = torch.tensor([0, 1], device=device)
        first_parameter = next(model.parameters())
        before_step = first_parameter.detach().clone()
        logits = model(inputs)
        if logits.device != device or logits.shape != (2, 2):
            raise RuntimeError("Tiny model forward produced an invalid shape or wrong device")
        result["checks"]["tiny_model_forward"] = True
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        if first_parameter.grad is None or first_parameter.grad.device != device:
            raise RuntimeError("Tiny model backward did not produce a CUDA gradient")
        result["checks"]["tiny_model_backward"] = True
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if torch.equal(before_step, first_parameter.detach()):
            raise RuntimeError("Tiny model optimizer step did not update its CUDA parameter")
        result["checks"]["tiny_model_optimizer_step"] = True
        torch.cuda.synchronize(device)
        result["tiny_model_evidence"] = {
            "parameter_devices": _all_parameter_devices(model),
            "input_device": str(inputs.device),
            "output_device": str(logits.device),
            "loss_device": str(loss.device),
            "optimizer_parameter_device": str(first_parameter.device),
        }
        result["peak_cuda_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        result["stage"] = "complete"
        result["status"] = "PASS"
    except Exception as exc:  # The JSON result is the diagnostic interface.
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
    finally:
        result["duration_seconds"] = round(time.time() - started, 6)

    return result


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strict NVIDIA GeForce RTX 5090 CUDA preflight for LEAF."
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path for the same JSON document printed to stdout.",
    )
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    result = run_preflight(args.device)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.json_out is not None:
        output_path = args.json_out.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
