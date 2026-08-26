#!/usr/bin/env python3
"""Fail-closed ten-check CUDA preflight for a formal SDSeg configuration."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch
from omegaconf import OmegaConf


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from ldm.util import instantiate_from_config  # noqa: E402


EXPECTED_AUDIT_MODES = {
    "full": "full_diffusion",
    "random-yt": "train_random_yt",
    "shuffle-yt": "train_shuffle_yt",
    "core-no-diff": "core_no_diff",
}


@dataclass
class Check:
    number: int
    name: str
    status: str
    detail: dict | str


def _path_label(path: Path) -> str:
    try:
        return path.resolve().relative_to(CODE_ROOT).as_posix()
    except ValueError:
        return path.name


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--pretrained-root", required=True, type=Path)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--expected-gpu-name", default="NVIDIA GeForce RTX 5090")
    parser.add_argument("--expected-capability", default="12.0")
    parser.add_argument("--report", type=Path)
    return parser


def run_preflight(args: argparse.Namespace) -> dict:
    checks: list[Check] = []
    state: dict = {"config": None, "model": None, "device": None}

    def run(name: str, operation: Callable[[], dict | str]) -> None:
        number = len(checks) + 1
        try:
            detail = operation()
            checks.append(Check(number, name, "pass", detail))
        except Exception as exc:
            checks.append(Check(number, name, "fail", f"{type(exc).__name__}: {exc}"))

    def cuda_available() -> dict:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; CPU fallback is prohibited.")
        return {
            "torch_version": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        }

    def select_device() -> dict:
        count = torch.cuda.device_count()
        if args.cuda_device < 0 or args.cuda_device >= count:
            raise ValueError(f"CUDA device {args.cuda_device} is outside available range 0..{count - 1}")
        torch.cuda.set_device(args.cuda_device)
        state["device"] = torch.device("cuda", args.cuda_device)
        return {"device_count": count, "selected_device": args.cuda_device}

    def exact_name() -> dict:
        if state["device"] is None:
            raise RuntimeError("CUDA device selection failed.")
        actual = torch.cuda.get_device_name(args.cuda_device)
        if actual != args.expected_gpu_name:
            raise RuntimeError(f"Expected {args.expected_gpu_name!r}, found {actual!r}")
        return {"name": actual}

    def capability() -> dict:
        expected = tuple(int(part) for part in args.expected_capability.split(".", 1))
        actual = tuple(torch.cuda.get_device_capability(args.cuda_device))
        if actual != expected:
            raise RuntimeError(f"Expected CUDA capability {expected}, found {actual}")
        return {"capability": f"{actual[0]}.{actual[1]}"}

    def compiled_arch() -> dict:
        arches = list(torch.cuda.get_arch_list())
        expected = "sm_" + args.expected_capability.replace(".", "")
        if expected not in arches:
            raise RuntimeError(f"Torch build lacks {expected}; compiled arches={arches}")
        return {"required_arch": expected, "compiled_arches": arches}

    def config_invariants() -> dict:
        config_path = args.config.expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        os.environ["SDSEG_PRETRAINED_ROOT"] = str(args.pretrained_root.expanduser().resolve())
        config = OmegaConf.load(config_path)
        config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        state["config"] = config
        mode = str(config.model.params.audit_mode)
        if mode not in set(EXPECTED_AUDIT_MODES.values()):
            raise ValueError(f"Unsupported release audit_mode={mode!r}")
        invariants = {
            "base_learning_rate": (float(config.model.base_learning_rate), 1e-5),
            "batch_size": (int(config.data.params.batch_size), 4),
            "num_workers": (int(config.data.params.num_workers), 8),
            "max_steps": (int(config.lightning.trainer.max_steps), 100000),
            "random_seed": (int(config.model.params.random_seed), 23),
            "monitor": (str(config.model.params.monitor), "val_avg_dice"),
        }
        mismatches = {
            key: {"actual": actual, "expected": expected}
            for key, (actual, expected) in invariants.items()
            if actual != expected
        }
        if mismatches:
            raise ValueError(f"Formal config parameter drift: {mismatches}")
        if "metric_validation" not in config.data.params or "test" not in config.data.params:
            raise ValueError("Config must define separate metric_validation and test datasets.")
        return {
            "config": _path_label(config_path),
            "audit_mode": mode,
            **{key: actual for key, (actual, _) in invariants.items()},
        }

    def pretrained_inputs() -> dict:
        config = state["config"]
        if config is None:
            raise RuntimeError("Config loading failed.")
        paths = [
            Path(str(config.model.params.ckpt_path)),
            Path(str(config.model.params.first_stage_config.params.ckpt_path)),
            Path(str(config.model.params.cond_stage_config.params.ckpt_path)),
        ]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing required pretrained files: {[p.name for p in missing]}")
        return {
            "files": [path.name for path in paths],
            "sizes_bytes": [path.stat().st_size for path in paths],
            "publication_policy": "inputs only; checkpoints are excluded from the release",
        }

    def instantiate_model() -> dict:
        config = state["config"]
        if config is None:
            raise RuntimeError("Config loading failed.")
        model = instantiate_from_config(config.model)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        if parameter_count <= 0 or trainable_count <= 0:
            raise RuntimeError(
                f"Invalid model parameter counts: total={parameter_count}, trainable={trainable_count}"
            )
        state["model"] = model
        return {"parameter_count": parameter_count, "trainable_parameter_count": trainable_count}

    def move_model() -> dict:
        model = state["model"]
        device = state["device"]
        if model is None or device is None:
            raise RuntimeError("Model instantiation or CUDA selection failed.")
        model.to(device)
        parameter_devices = {str(parameter.device) for parameter in model.parameters()}
        buffer_devices = {str(buffer.device) for buffer in model.buffers()}
        expected_device = str(device)
        if parameter_devices != {expected_device}:
            raise RuntimeError(f"Model parameters are not exclusively on {expected_device}: {parameter_devices}")
        if any(not item.startswith("cuda") for item in buffer_devices):
            raise RuntimeError(f"Model has non-CUDA buffers: {buffer_devices}")
        return {"parameter_devices": sorted(parameter_devices), "buffer_devices": sorted(buffer_devices)}

    def forward_probe() -> dict:
        model = state["model"]
        config = state["config"]
        device = state["device"]
        if model is None or config is None or device is None:
            raise RuntimeError("Prerequisite model/config/device check failed.")
        model.eval()
        unet = model.model.diffusion_model
        in_channels = int(unet.in_channels)
        image_size = int(config.model.params.image_size)
        latent_channels = int(config.model.params.channels)
        batch = 1
        probe = torch.randn(batch, in_channels, image_size, image_size, device=device)
        timesteps = torch.zeros(batch, dtype=torch.long, device=device)
        class_ids = None
        if getattr(unet, "num_classes", None) is not None:
            class_ids = torch.ones(batch, dtype=torch.long, device=device)
        torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            if str(config.model.params.audit_mode) == "core_no_diff":
                output = model._forward_unet_without_timestep(probe, cls_id=class_ids)
            elif class_ids is None:
                output = unet(probe, timesteps)
            else:
                output = unet(probe, timesteps, y=class_ids)
        expected_shape = (batch, latent_channels, image_size, image_size)
        if tuple(output.shape) != expected_shape:
            raise RuntimeError(f"UNet output shape {tuple(output.shape)} != {expected_shape}")
        if output.device != device or not torch.isfinite(output).all():
            raise RuntimeError("UNet probe output is not finite on the selected CUDA device.")
        return {
            "input_shape": list(probe.shape),
            "output_shape": list(output.shape),
            "input_device": str(probe.device),
            "output_device": str(output.device),
            "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
        }

    operations = [
        ("cuda_available_no_cpu_fallback", cuda_available),
        ("cuda_device_selection", select_device),
        ("exact_gpu_identity", exact_name),
        ("exact_compute_capability", capability),
        ("torch_sm120_support", compiled_arch),
        ("formal_config_parameters", config_invariants),
        ("pretrained_inputs_present", pretrained_inputs),
        ("model_instantiation_and_parameter_count", instantiate_model),
        ("all_model_state_on_selected_cuda", move_model),
        ("cuda_model_input_output_probe", forward_probe),
    ]
    for name, operation in operations:
        run(name, operation)

    try:
        del state["model"]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    failed = [check for check in checks if check.status != "pass"]
    return {
        "status": "pass" if not failed else "fail",
        "cuda_only": True,
        "expected_gpu_name": args.expected_gpu_name,
        "check_count": len(checks),
        "passed": len(checks) - len(failed),
        "failed": len(failed),
        "checks": [check.__dict__ for check in checks],
    }


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_preflight(args)
    if args.report:
        _write_json(args.report, report)
    stream = sys.stdout if report["status"] == "pass" else sys.stderr
    print(json.dumps(report, indent=2, sort_keys=True), file=stream)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
