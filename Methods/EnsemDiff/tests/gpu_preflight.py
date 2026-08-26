#!/usr/bin/env python3
"""Fail-closed CUDA acceptance check for the public EnsemDiff release.

This program intentionally has no CPU execution path.  Importing the module is
safe on machines without PyTorch/CUDA so that ordinary static unit tests and
``--help`` do not probe the GPU; GPU imports happen only inside ``run_preflight``.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any


TEST_ROOT = Path(__file__).resolve().parent
RELEASE_ROOT = TEST_ROOT.parent
CODE_ROOT = RELEASE_ROOT / "code"
sys.dont_write_bytecode = True

EXPECTED_GPU_NAME = "NVIDIA GeForce RTX 5090"
EXPECTED_CAPABILITY = (12, 0)
EXPECTED_ARCH = "sm_120"
CUDA_DEVICE_INDEX = 0
CUDA_DEVICE = "cuda:0"

# Exact model/diffusion values emitted by code/scripts/release_pipeline.py.
FORMAL_MODEL_CONFIG: dict[str, Any] = {
    "image_size": 256,
    "class_cond": False,
    "learn_sigma": True,
    "num_channels": 128,
    "num_res_blocks": 2,
    "channel_mult": "1,1,2,2,4,4",
    "num_heads": 1,
    "num_head_channels": -1,
    "num_heads_upsample": -1,
    "attention_resolutions": "16",
    "dropout": 0.0,
    "diffusion_steps": 1000,
    "noise_schedule": "linear",
    "use_kl": False,
    "predict_xstart": False,
    "rescale_timesteps": False,
    "rescale_learned_sigmas": False,
    "use_checkpoint": False,
    "use_scale_shift_norm": False,
    "resblock_updown": False,
    "use_fp16": False,
    "use_new_attention_order": False,
}

PREFLIGHT_DATASET_CONFIG = {
    "image_channels": 1,
    "mask_channels": 1,
    "num_seg_classes": 2,
}

PASS = "PASS"
FAIL = "FAIL"
BLOCKED = "BLOCKED"


class _CheckFailure(RuntimeError):
    """Internal marker; its message is never copied into public JSON."""


def _require(condition: bool, check: str) -> None:
    if not condition:
        raise _CheckFailure(check)


def create_formal_model_and_diffusion(
    dataset_config: dict[str, int],
    *,
    core_no_diff: bool,
    timestep_respacing: str = "",
):
    """Create the canonical release model/diffusion without changing config."""
    os.environ["ENSEMDIFF_USE_VISDOM"] = "0"
    os.environ["ENSEMDIFF_SAMPLE_PROGRESS"] = "0"
    if str(CODE_ROOT) not in sys.path:
        sys.path.insert(0, str(CODE_ROOT))
    from guided_diffusion.script_util import create_model_and_diffusion

    return create_model_and_diffusion(
        image_channels=dataset_config["image_channels"],
        mask_channels=dataset_config["mask_channels"],
        num_seg_classes=dataset_config["num_seg_classes"],
        timestep_respacing=timestep_respacing,
        core_no_diff=core_no_diff,
        **FORMAL_MODEL_CONFIG,
    )


def _base_result() -> dict[str, Any]:
    return {
        "program": "gpu_preflight",
        "status": BLOCKED,
        "device": {
            "index": CUDA_DEVICE_INDEX,
            "logical_device": CUDA_DEVICE,
            "expected_name": EXPECTED_GPU_NAME,
            "expected_capability": list(EXPECTED_CAPABILITY),
            "expected_arch": EXPECTED_ARCH,
        },
        "checks": {},
        "memory": {
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
        },
    }


def run_preflight() -> dict[str, Any]:
    """Run all checks and return sanitized, JSON-serializable evidence."""
    result = _base_result()
    checks = result["checks"]
    stage = "torch_import"
    torch = None
    model = None
    diffusion = None
    left = None
    right = None
    product = None
    model_input = None
    timesteps = None
    model_output = None

    try:
        import torch as imported_torch

        torch = imported_torch
        checks[stage] = True
        result["runtime"] = {
            "torch_version": str(torch.__version__),
            "cuda_runtime": str(torch.version.cuda),
        }

        stage = "cuda_available"
        _require(bool(torch.cuda.is_available()), stage)
        checks[stage] = True

        stage = "device_count"
        device_count = int(torch.cuda.device_count())
        result["device"]["device_count"] = device_count
        _require(device_count >= 1, stage)
        checks[stage] = True

        torch.cuda.set_device(CUDA_DEVICE_INDEX)
        device = torch.device(CUDA_DEVICE)

        stage = "selected_device"
        _require(int(torch.cuda.current_device()) == CUDA_DEVICE_INDEX, stage)
        checks[stage] = True

        stage = "exact_device_name"
        actual_name = str(torch.cuda.get_device_name(CUDA_DEVICE_INDEX))
        result["device"]["actual_name"] = actual_name
        _require(actual_name == EXPECTED_GPU_NAME, stage)
        checks[stage] = True

        stage = "compute_capability"
        capability = tuple(
            int(value) for value in torch.cuda.get_device_capability(CUDA_DEVICE_INDEX)
        )
        result["device"]["actual_capability"] = list(capability)
        _require(capability == EXPECTED_CAPABILITY, stage)
        checks[stage] = True

        stage = "compiled_arch_list"
        arch_list = sorted(str(value) for value in torch.cuda.get_arch_list())
        result["device"]["compiled_arch_list"] = arch_list
        _require(EXPECTED_ARCH in arch_list, stage)
        checks[stage] = True

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(CUDA_DEVICE_INDEX)

        stage = "cuda_matmul"
        left = torch.randn(64, 64, device=device, dtype=torch.float32, requires_grad=True)
        right = torch.randn(64, 64, device=device, dtype=torch.float32, requires_grad=True)
        product = (left @ right).square().mean()
        _require(product.device == device, stage)
        _require(bool(torch.isfinite(product).item()), stage)
        checks[stage] = True

        stage = "cuda_backward"
        product.backward()
        _require(left.grad is not None and left.grad.device == device, stage)
        _require(right.grad is not None and right.grad.device == device, stage)
        _require(bool(torch.isfinite(left.grad).all().item()), stage)
        _require(bool(torch.isfinite(right.grad).all().item()), stage)
        checks[stage] = True

        stage = "formal_model_create"
        model, diffusion = create_formal_model_and_diffusion(
            PREFLIGHT_DATASET_CONFIG,
            core_no_diff=False,
            timestep_respacing="",
        )
        model = model.to(device)
        model.eval()
        _require(int(diffusion.num_timesteps) == 1000, stage)
        checks[stage] = True

        stage = "formal_model_parameters_cuda0"
        parameters = list(model.parameters())
        buffers = list(model.buffers())
        _require(bool(parameters), stage)
        _require(all(parameter.device == device for parameter in parameters), stage)
        _require(all(buffer.device == device for buffer in buffers), stage)
        _require(all(parameter.dtype == torch.float32 for parameter in parameters), stage)
        checks[stage] = True

        stage = "formal_model_forward_cuda0"
        model_input = torch.randn(
            1,
            PREFLIGHT_DATASET_CONFIG["image_channels"]
            + PREFLIGHT_DATASET_CONFIG["mask_channels"],
            224,
            224,
            device=device,
            dtype=torch.float32,
        )
        timesteps = torch.zeros(1, device=device, dtype=torch.long)
        _require(model_input.device == device and timesteps.device == device, stage)
        with torch.no_grad():
            model_output = model(model_input, timesteps)
        _require(model_output.device == device, stage)
        _require(model_output.dtype == torch.float32, stage)
        _require(tuple(model_output.shape) == (1, 2, 224, 224), stage)
        torch.cuda.synchronize(CUDA_DEVICE_INDEX)
        checks[stage] = True

        result["formal_model"] = {
            "parameter_count": int(sum(parameter.numel() for parameter in parameters)),
            "input_shape": list(model_input.shape),
            "input_dtype": str(model_input.dtype),
            "output_shape": list(model_output.shape),
            "output_dtype": str(model_output.dtype),
            "diffusion_steps": int(diffusion.num_timesteps),
            "use_fp16": False,
        }

        stage = "cuda_memory_nonzero"
        memory = {
            "allocated_bytes": int(torch.cuda.memory_allocated(CUDA_DEVICE_INDEX)),
            "reserved_bytes": int(torch.cuda.memory_reserved(CUDA_DEVICE_INDEX)),
            "peak_allocated_bytes": int(
                torch.cuda.max_memory_allocated(CUDA_DEVICE_INDEX)
            ),
            "peak_reserved_bytes": int(
                torch.cuda.max_memory_reserved(CUDA_DEVICE_INDEX)
            ),
        }
        result["memory"] = memory
        _require(memory["allocated_bytes"] > 0, stage)
        _require(memory["peak_allocated_bytes"] > 0, stage)
        checks[stage] = True

        result["status"] = PASS
        return result
    except Exception as error:  # fail closed; never expose raw exception text/paths
        checks.setdefault(stage, False)
        blocking_stages = {
            "torch_import",
            "cuda_available",
            "device_count",
            "selected_device",
            "exact_device_name",
            "compute_capability",
            "compiled_arch_list",
        }
        result["status"] = BLOCKED if stage in blocking_stages else FAIL
        result["error"] = {
            "check": stage,
            "kind": type(error).__name__,
            "message": "GPU preflight check did not pass.",
        }
        return result
    finally:
        model_output = None
        timesteps = None
        model_input = None
        diffusion = None
        model = None
        product = None
        right = None
        left = None
        gc.collect()
        if torch is not None and bool(torch.cuda.is_available()):
            torch.cuda.empty_cache()


def create_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description=(
            "Fail closed unless cuda:0 is the exact RTX 5090 target and passes "
            "CUDA arithmetic, backward, formal-UNet, and memory checks."
        )
    )


def main() -> int:
    create_parser().parse_args()
    result = run_preflight()
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    if result["status"] == PASS:
        return 0
    if result["status"] == BLOCKED:
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
