#!/usr/bin/env python3
"""Real, fail-closed GPU smoke acceptance for all 3 x 4 release runs.

The smoke length is one optimizer step, one validation sample, and one final
test sample per combination.  Scientific configuration is not reduced: the
canonical UNet and loss are used, validation is 100-step DDPM, final test is
1000-step DDPM, and non-core final test uses five ensemble samples.

PyTorch and product modules are imported only after argument parsing and the
GPU preflight.  This keeps ``--help`` and ordinary static unit tests GPU-free.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import gc
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


TEST_ROOT = Path(__file__).resolve().parent
RELEASE_ROOT = TEST_ROOT.parent
CODE_ROOT = RELEASE_ROOT / "code"
MANIFEST_ROOT = CODE_ROOT / "manifests"
TRAIN_UTIL_PATH = CODE_ROOT / "guided_diffusion" / "train_util.py"
GAUSSIAN_DIFFUSION_PATH = (
    CODE_ROOT / "guided_diffusion" / "gaussian_diffusion.py"
)
sys.dont_write_bytecode = True

sys.path.insert(0, str(TEST_ROOT))
import gpu_preflight  # noqa: E402  (stdlib-only module at import time)


PASS = "PASS"
FAIL = "FAIL"
BLOCKED = "BLOCKED"
SEED = 10
VALIDATION_SEED = 10
TRAIN_LR = 0.0001
WEIGHT_DECAY = 0.0
VALIDATION_RESPACING = "100"
TEST_RESPACING = "1000"
VALIDATION_ENSEMBLE = 1
NON_CORE_TEST_ENSEMBLE = 5

CONDITIONS = {
    "full": "none",
    "random-yt": "train_random_yt",
    "shuffle-yt": "train_shuffle_yt",
    "core-no-diff": "core_no_diff",
}

EXPECTED_AUDIT_EFFECTS = {
    "none": "reference_q_sample_y_t",
    "train_random_yt": "replace_y_t_input_with_independent_gaussian",
    "train_shuffle_yt": "replace_y_t_input_with_batch_derangement",
    "core_no_diff": "direct_segmentation_without_diffusion",
}

DATASETS: dict[str, dict[str, Any]] = {
    "btcv": {
        "image_channels": 1,
        "mask_channels": 1,
        "num_seg_classes": 2,
        "loader": "BTCVDataset",
        "partition_totals": {
            "training": 2211,
            "validation": 295,
            "testing": 1273,
        },
        "manifests": {
            "training": "train_cases.txt",
            "validation": "val_cases.txt",
            "testing": "test_cases.txt",
        },
    },
    "acdc": {
        "image_channels": 1,
        "mask_channels": 4,
        "num_seg_classes": 4,
        "loader": "ACDCDataset",
        "partition_totals": {
            "training": 1304,
            "validation": 182,
            "testing": 416,
        },
        "manifests": {
            "training": "train_patients.txt",
            "validation": "val_patients.txt",
            "testing": "test_patients.txt",
        },
    },
    "isic2018": {
        "image_channels": 3,
        "mask_channels": 1,
        "num_seg_classes": 2,
        "loader": "ISIC2018Dataset",
        "partition_totals": {
            "training": 2594,
            "validation": 100,
            "testing": 1000,
        },
        "manifests": {
            "training": "training.txt",
            "validation": "validation.txt",
            "testing": "testing.txt",
        },
    },
}

COMBINATIONS = tuple(
    (dataset, condition)
    for dataset in DATASETS
    for condition in CONDITIONS
)

SMOKE_BATCH_SIZE = {
    "full": 1,
    "random-yt": 1,
    "shuffle-yt": 2,
    "core-no-diff": 1,
}

BEST_METADATA_KEYS = {
    "checkpoint",
    "source_checkpoint",
    "step",
    "metric_name",
    "metric_value",
    "selection_partition",
    "validation_manifest",
    "validation_seed",
    "training_seed",
    "audit_mode",
}


class InputBlocked(RuntimeError):
    """A required manifest/preprocessed-data condition was not met."""


class AcceptanceFailure(RuntimeError):
    """A real GPU acceptance stage ran but did not satisfy its assertions."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise AcceptanceFailure(code)


def _parse_data_roots(values: list[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        dataset, separator, raw_path = value.partition("=")
        dataset = dataset.strip().lower()
        if not separator or not raw_path.strip():
            raise argparse.ArgumentTypeError(
                "Each --data-root must use DATASET=PATH syntax."
            )
        if dataset not in DATASETS:
            raise argparse.ArgumentTypeError(
                "DATASET must be one of: btcv, acdc, isic2018."
            )
        if dataset in roots:
            raise argparse.ArgumentTypeError(
                f"Duplicate --data-root entry for {dataset}."
            )
        roots[dataset] = Path(raw_path).expanduser().resolve()
    if set(roots) != set(DATASETS):
        missing = sorted(set(DATASETS) - set(roots))
        raise argparse.ArgumentTypeError(
            "Exactly one root is required for each dataset; missing: "
            + ", ".join(missing)
        )
    return roots


def _manifest_path(dataset: str, split: str) -> Path:
    return MANIFEST_ROOT / dataset / DATASETS[dataset]["manifests"][split]


def _manifest_display_path(dataset: str, split: str) -> str:
    path = _manifest_path(dataset, split)
    return path.relative_to(RELEASE_ROOT).as_posix()


def _read_manifest_evidence(dataset: str) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    split_ids: dict[str, set[str]] = {}
    for split in ("training", "validation", "testing"):
        path = _manifest_path(dataset, split)
        if not path.is_file():
            raise InputBlocked("fixed_manifest_missing")
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise InputBlocked("fixed_manifest_not_utf8") from error
        identifiers = [line.strip() for line in text.splitlines() if line.strip()]
        if not identifiers or len(identifiers) != len(set(identifiers)):
            raise InputBlocked("fixed_manifest_empty_or_duplicate")
        split_ids[split] = set(identifiers)
        evidence[split] = {
            "release_path": _manifest_display_path(dataset, split),
            "entries": len(identifiers),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    if (
        split_ids["training"] & split_ids["validation"]
        or split_ids["training"] & split_ids["testing"]
        or split_ids["validation"] & split_ids["testing"]
    ):
        raise InputBlocked("fixed_manifest_partition_overlap")
    return evidence


def _trainloop_best_schema() -> set[str]:
    """Extract TrainLoop._save_best_checkpoint's metadata keys with stdlib AST."""
    tree = ast.parse(TRAIN_UTIL_PATH.read_text(encoding="utf-8"), str(TRAIN_UTIL_PATH))
    for class_node in tree.body:
        if not isinstance(class_node, ast.ClassDef) or class_node.name != "TrainLoop":
            continue
        for function_node in class_node.body:
            if (
                not isinstance(function_node, ast.FunctionDef)
                or function_node.name != "_save_best_checkpoint"
            ):
                continue
            for node in ast.walk(function_node):
                if not isinstance(node, ast.Assign):
                    continue
                if not any(
                    isinstance(target, ast.Name) and target.id == "metadata"
                    for target in node.targets
                ):
                    continue
                if not isinstance(node.value, ast.Dict):
                    continue
                keys = {
                    key.value
                    for key in node.value.keys
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                }
                return keys
    raise AcceptanceFailure("trainloop_best_schema_not_found")


def _audit_branch_contract() -> set[str]:
    """Extract explicit audit-mode comparisons from the formal training loss."""
    tree = ast.parse(
        GAUSSIAN_DIFFUSION_PATH.read_text(encoding="utf-8"),
        str(GAUSSIAN_DIFFUSION_PATH),
    )
    target = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "training_losses_segmentation"
        ):
            target = node
            break
    if target is None:
        raise AcceptanceFailure("formal_training_loss_not_found")

    modes: set[str] = set()
    for node in ast.walk(target):
        if not isinstance(node, ast.Compare) or len(node.ops) != 1:
            continue
        if not isinstance(node.ops[0], ast.Eq):
            continue
        if not isinstance(node.left, ast.Name) or node.left.id != "audit_mode":
            continue
        if len(node.comparators) != 1:
            continue
        comparator = node.comparators[0]
        if isinstance(comparator, ast.Constant) and isinstance(comparator.value, str):
            modes.add(comparator.value)
    return modes


def _import_runtime() -> dict[str, Any]:
    # These switches affect only visualization/log chatter, never computation.
    os.environ["ENSEMDIFF_USE_VISDOM"] = "0"
    os.environ["ENSEMDIFF_SAMPLE_PROGRESS"] = "0"
    if str(CODE_ROOT) not in sys.path:
        sys.path.insert(0, str(CODE_ROOT))

    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset

    from guided_diffusion import logger
    from guided_diffusion.acdcloader import ACDCDataset
    from guided_diffusion.btcvloader import BTCVDataset
    from guided_diffusion.isicloader import ISIC2018Dataset
    from guided_diffusion.resample import create_named_schedule_sampler
    from guided_diffusion.script_util import create_gaussian_diffusion
    from guided_diffusion.validation_util import validate_segmentation

    return {
        "np": np,
        "torch": torch,
        "DataLoader": DataLoader,
        "Subset": Subset,
        "logger": logger,
        "loaders": {
            "BTCVDataset": BTCVDataset,
            "ACDCDataset": ACDCDataset,
            "ISIC2018Dataset": ISIC2018Dataset,
        },
        "create_named_schedule_sampler": create_named_schedule_sampler,
        "create_gaussian_diffusion": create_gaussian_diffusion,
        "validate_segmentation": validate_segmentation,
    }


def _set_seed(runtime: dict[str, Any], seed: int) -> None:
    np = runtime["np"]
    torch = runtime["torch"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _tf32_state(torch) -> dict[str, bool]:
    return {
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }


def _memory(torch) -> dict[str, int]:
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(0)),
        "reserved_bytes": int(torch.cuda.memory_reserved(0)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
    }


def _device_evidence(preflight: dict[str, Any]) -> dict[str, Any]:
    device = preflight.get("device", {})
    return {
        "logical_device": gpu_preflight.CUDA_DEVICE,
        "index": gpu_preflight.CUDA_DEVICE_INDEX,
        "name": device.get("actual_name"),
        "capability": device.get("actual_capability"),
        "compiled_arch": gpu_preflight.EXPECTED_ARCH,
    }


def _build_dataset(runtime: dict[str, Any], dataset: str, root: Path, split: str):
    directory = root / split
    manifest = _manifest_path(dataset, split)
    if not root.is_dir() or not directory.is_dir():
        raise InputBlocked("preprocessed_split_directory_missing")
    loader_class = runtime["loaders"][DATASETS[dataset]["loader"]]
    kwargs: dict[str, Any] = {
        "directory": directory,
        "test_flag": False,
        "manifest_path": manifest,
    }
    if dataset == "acdc":
        kwargs["num_classes"] = DATASETS[dataset]["num_seg_classes"]
    try:
        return loader_class(**kwargs)
    except (OSError, ValueError, RuntimeError) as error:
        raise InputBlocked("manifest_locked_dataset_load_failed") from error


def _one_batch_loader(runtime: dict[str, Any], dataset_object, batch_size: int):
    if len(dataset_object) < batch_size:
        raise InputBlocked("manifest_partition_too_small_for_smoke_batch")
    subset = runtime["Subset"](dataset_object, list(range(batch_size)))
    return runtime["DataLoader"](
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True,
        pin_memory=False,
    )


def _validate_batch_shape(
    image,
    target,
    config: dict[str, Any],
    batch_size: int,
) -> None:
    _require(image.ndim == 4 and target.ndim == 4, "batch_rank")
    _require(image.shape[0] == target.shape[0] == batch_size, "batch_size")
    _require(image.shape[1] == config["image_channels"], "image_channels")
    _require(target.shape[1] == config["mask_channels"], "mask_channels")
    _require(tuple(image.shape[-2:]) == (224, 224), "image_spatial_shape")
    _require(tuple(target.shape[-2:]) == (224, 224), "target_spatial_shape")


def _formal_eval_diffusion(runtime: dict[str, Any], respacing: str):
    config = gpu_preflight.FORMAL_MODEL_CONFIG
    return runtime["create_gaussian_diffusion"](
        steps=config["diffusion_steps"],
        learn_sigma=config["learn_sigma"],
        noise_schedule=config["noise_schedule"],
        use_kl=config["use_kl"],
        predict_xstart=config["predict_xstart"],
        rescale_timesteps=config["rescale_timesteps"],
        rescale_learned_sigmas=config["rescale_learned_sigmas"],
        timestep_respacing=respacing,
    )


def _assert_model_cuda0_float32(torch, model) -> None:
    parameters = list(model.parameters())
    _require(bool(parameters), "model_has_no_parameters")
    device = torch.device(gpu_preflight.CUDA_DEVICE)
    _require(all(parameter.device == device for parameter in parameters), "parameter_device")
    _require(all(parameter.dtype == torch.float32 for parameter in parameters), "parameter_dtype")
    _require(all(buffer.device == device for buffer in model.buffers()), "buffer_device")


def _train_one_step(
    runtime: dict[str, Any],
    model,
    diffusion,
    loader,
    config: dict[str, Any],
    condition: str,
) -> tuple[dict[str, Any], Any]:
    torch = runtime["torch"]
    device = torch.device(gpu_preflight.CUDA_DEVICE)
    batch_size = SMOKE_BATCH_SIZE[condition]
    audit_mode = CONDITIONS[condition]
    _set_seed(runtime, SEED)
    model.train()
    _assert_model_cuda0_float32(torch, model)
    model_was_training = bool(model.training)
    _require(model_was_training, "audit_requires_training_mode")

    image, target = next(iter(loader))
    _validate_batch_shape(image, target, config, batch_size)
    image = image.to(device=device, dtype=torch.float32)
    target = target.to(device=device, dtype=torch.float32)
    x_start = torch.cat((image, target), dim=1)
    _require(x_start.device == device, "training_input_device")

    if audit_mode == "core_no_diff":
        timesteps = torch.zeros(batch_size, device=device, dtype=torch.long)
        weights = torch.ones(batch_size, device=device, dtype=torch.float32)
    else:
        sampler = runtime["create_named_schedule_sampler"](
            "uniform", diffusion, gpu_preflight.FORMAL_MODEL_CONFIG["diffusion_steps"]
        )
        timesteps, weights = sampler.sample(batch_size, device)
    _require(timesteps.device == device and timesteps.dtype == torch.long, "timestep_device")
    _require(weights.device == device and weights.dtype == torch.float32, "weight_device")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=TRAIN_LR, weight_decay=WEIGHT_DECAY
    )
    probe_candidates = [
        (parameter, parameter.detach().clone())
        for parameter in list(model.parameters())[-6:]
        if parameter.requires_grad
    ]
    optimizer.zero_grad(set_to_none=True)
    terms, model_output = diffusion.training_losses_segmentation(
        model,
        None,
        x_start,
        timesteps,
        model_kwargs={},
        audit_mode=audit_mode,
    )
    expected_loss_terms = (
        {"ce", "dice", "loss"}
        if audit_mode == "core_no_diff"
        else {"mse", "vb", "loss"}
    )
    _require(set(terms) == expected_loss_terms, "formal_loss_terms")
    _require("loss" in terms and tuple(terms["loss"].shape) == (batch_size,), "loss_shape")
    _require(model_output.device == device, "training_output_device")
    _require(model_output.dtype == torch.float32, "training_output_dtype")
    loss = (terms["loss"] * weights).mean()
    _require(loss.device == device, "training_loss_device")
    _require(bool(torch.isfinite(loss).item()), "training_loss_finite")
    loss.backward()

    gradient_probe = None
    for parameter, before in probe_candidates:
        if parameter.grad is None:
            continue
        if not bool(torch.isfinite(parameter.grad).all().item()):
            raise AcceptanceFailure("training_gradient_not_finite")
        if bool((parameter.grad != 0).any().item()):
            gradient_probe = (parameter, before)
            break
    _require(gradient_probe is not None, "training_nonzero_gradient")
    optimizer.step()
    probe_parameter, probe_before = gradient_probe
    _require(not torch.equal(probe_parameter.detach(), probe_before), "optimizer_parameter_update")
    torch.cuda.synchronize(0)

    output = {
        "batch_size": batch_size,
        "loss": float(loss.detach().item()),
        "loss_terms": sorted(terms),
        "timesteps": [int(value) for value in timesteps.detach().tolist()],
        "input_shape": list(x_start.shape),
        "output_shape": list(model_output.shape),
        "dtype": "torch.float32",
        "amp_enabled": False,
        "optimizer": "AdamW",
        "optimizer_updated_parameter": True,
        "model_parameter_device": gpu_preflight.CUDA_DEVICE,
        "input_device": str(x_start.device),
        "output_device": str(model_output.device),
        "loss_device": str(loss.device),
        "forward_completed": True,
        "backward_completed": True,
        "optimizer_step_completed": True,
        "requested_audit_mode": audit_mode,
        "expected_branch_effect": EXPECTED_AUDIT_EFFECTS[audit_mode],
        "audit_branch_executed": True,
        "audit_source_contract_verified": True,
        "model_training": model_was_training,
        "memory": _memory(torch),
    }
    return output, optimizer


def _tensor_values(torch, value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensor_values(torch, item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensor_values(torch, item)


def _run_metric_stage(
    runtime: dict[str, Any],
    *,
    model,
    diffusion,
    loader,
    config: dict[str, Any],
    audit_mode: str,
    ensemble: int,
    expected_steps: int,
    output_csv: Path,
    step: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    torch = runtime["torch"]
    expected_ensemble = 1 if audit_mode == "core_no_diff" else ensemble
    expected_forward_calls = (
        1
        if audit_mode == "core_no_diff"
        else expected_steps * expected_ensemble
    )
    tracker = {
        "pre_calls": 0,
        "post_calls": 0,
        "input_tensor_count": 0,
        "output_tensor_count": 0,
        "all_inputs_cuda0": True,
        "all_outputs_cuda0": True,
    }

    def forward_pre_hook(_module, inputs):
        tensors = list(_tensor_values(torch, inputs))
        tracker["pre_calls"] += 1
        tracker["input_tensor_count"] += len(tensors)
        if not tensors or any(
            str(tensor.device) != gpu_preflight.CUDA_DEVICE
            for tensor in tensors
        ):
            tracker["all_inputs_cuda0"] = False

    def forward_post_hook(_module, _inputs, output):
        tensors = list(_tensor_values(torch, output))
        tracker["post_calls"] += 1
        tracker["output_tensor_count"] += len(tensors)
        if not tensors or any(
            str(tensor.device) != gpu_preflight.CUDA_DEVICE
            for tensor in tensors
        ):
            tracker["all_outputs_cuda0"] = False

    pre_handle = model.register_forward_pre_hook(forward_pre_hook)
    post_handle = model.register_forward_hook(forward_post_hook)
    try:
        row = runtime["validate_segmentation"](
            model=model,
            diffusion=diffusion,
            dataloader=loader,
            step=step,
            image_channels=config["image_channels"],
            mask_channels=config["mask_channels"],
            num_seg_classes=config["num_seg_classes"],
            num_ensemble=ensemble,
            use_ddim=False,
            output_csv=output_csv,
            audit_mode=audit_mode,
        )
    finally:
        pre_handle.remove()
        post_handle.remove()

    _require(tracker["pre_calls"] == expected_forward_calls, "forward_pre_call_count")
    _require(tracker["post_calls"] == expected_forward_calls, "forward_post_call_count")
    _require(bool(tracker["all_inputs_cuda0"]), "metric_model_input_device")
    _require(bool(tracker["all_outputs_cuda0"]), "metric_model_output_device")
    _require(output_csv.is_file() and output_csv.stat().st_size > 0, "metric_csv_missing")
    _require(int(row["num_slices"]) == 1, "metric_slice_count")
    _require(int(row["sampler_steps"]) == expected_steps, "metric_sampler_steps")
    _require(int(row["num_ensemble"]) == expected_ensemble, "metric_ensemble")
    _require(row["use_ddim"] is False, "ddpm_required")
    _require(math.isfinite(float(row["dice"])), "metric_dice_finite")
    _require(math.isfinite(float(row["iou"])), "metric_iou_finite")
    torch.cuda.synchronize(0)
    return {
        "dice": float(row["dice"]),
        "iou": float(row["iou"]),
        "num_slices": int(row["num_slices"]),
        "num_ensemble": int(row["num_ensemble"]),
        "sampler_steps": int(row["sampler_steps"]),
        "use_ddim": False,
        "forward_calls": int(tracker["post_calls"]),
        "expected_forward_calls": expected_forward_calls,
        "forward_pre_hook_calls": int(tracker["pre_calls"]),
        "model_input_device": gpu_preflight.CUDA_DEVICE,
        "model_output_device": gpu_preflight.CUDA_DEVICE,
        "all_model_inputs_cuda0": bool(tracker["all_inputs_cuda0"]),
        "all_model_outputs_cuda0": bool(tracker["all_outputs_cuda0"]),
        "memory": _memory(torch),
    }, row


def _save_best_like_trainloop(
    torch,
    *,
    model,
    row: dict[str, Any],
    directory: Path,
    validation_manifest: Path,
    audit_mode: str,
) -> tuple[Path, Path, dict[str, Any]]:
    metric = float(row["dice"])
    _require(math.isfinite(metric), "best_metric_not_finite")
    checkpoint_path = directory / "best_model.pt"
    metadata_path = directory / "best_checkpoint.json"
    state_dict = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
    }
    torch.save(state_dict, checkpoint_path)
    del state_dict
    metadata = {
        "checkpoint": "best_model.pt",
        "source_checkpoint": "savedmodel000001.pt",
        "step": 1,
        "metric_name": "dice",
        "metric_value": metric,
        "selection_partition": "validation",
        "validation_manifest": str(validation_manifest.resolve()),
        "validation_seed": VALIDATION_SEED,
        "training_seed": SEED,
        "audit_mode": audit_mode,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return checkpoint_path, metadata_path, metadata


def _verify_best_save(
    *,
    checkpoint_path: Path,
    metadata_path: Path,
    metadata: dict[str, Any],
    validation_manifest: Path,
    audit_mode: str,
) -> dict[str, Any]:
    _require(checkpoint_path.is_file() and checkpoint_path.stat().st_size > 0, "best_model_missing")
    _require(metadata_path.is_file() and metadata_path.stat().st_size > 0, "best_metadata_missing")
    persisted = json.loads(metadata_path.read_text(encoding="utf-8"))
    _require(set(persisted) == BEST_METADATA_KEYS, "best_metadata_keys")
    _require(set(metadata) == BEST_METADATA_KEYS, "best_helper_metadata_keys")
    _require(persisted == metadata, "best_metadata_round_trip")
    _require(persisted["checkpoint"] == "best_model.pt", "best_checkpoint_name")
    _require(persisted["selection_partition"] == "validation", "best_partition")
    _require(persisted["metric_name"] == "dice", "best_metric_name")
    _require(persisted["training_seed"] == SEED, "best_training_seed")
    _require(persisted["validation_seed"] == VALIDATION_SEED, "best_validation_seed")
    _require(persisted["audit_mode"] == audit_mode, "best_audit_mode")
    _require(
        Path(persisted["validation_manifest"]).resolve() == validation_manifest.resolve(),
        "best_validation_manifest",
    )
    return {
        "schema_keys": sorted(persisted),
        "checkpoint_format": "state_dict",
        "selection_partition": "validation",
        "metric_name": "dice",
        "schema_source": "TrainLoop._save_best_checkpoint",
    }


def _reload_best(torch, model, checkpoint_path: Path) -> dict[str, Any]:
    device = torch.device(gpu_preflight.CUDA_DEVICE)
    saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
    expected_keys = set(model.state_dict())
    _require(isinstance(saved, dict) and set(saved) == expected_keys, "checkpoint_state_keys")
    incompatible = model.load_state_dict(saved, strict=True)
    _require(not incompatible.missing_keys and not incompatible.unexpected_keys, "checkpoint_strict_load")
    del saved
    _assert_model_cuda0_float32(torch, model)
    return {
        "strict": True,
        "state_dict_keys": len(expected_keys),
        "map_location": gpu_preflight.CUDA_DEVICE,
    }


def _cleanup_cuda(runtime: dict[str, Any] | None) -> None:
    gc.collect()
    if runtime is not None:
        torch = runtime["torch"]
        if bool(torch.cuda.is_available()):
            torch.cuda.empty_cache()


def _run_combo(
    runtime: dict[str, Any],
    preflight: dict[str, Any],
    roots: dict[str, Path],
    dataset: str,
    condition: str,
) -> dict[str, Any]:
    torch = runtime["torch"]
    config = DATASETS[dataset]
    audit_mode = CONDITIONS[condition]
    core_no_diff = audit_mode == "core_no_diff"
    stage_name = "manifest_locked_data"
    stages: list[dict[str, Any]] = []
    record: dict[str, Any] = {
        "record_type": "gpu_smoke_combination",
        "dataset": dataset,
        "condition": condition,
        "audit_mode": audit_mode,
        "status": FAIL,
        "device": _device_evidence(preflight),
        "manifests": {},
        "configuration": {
            "formal_unet": True,
            "training_seed": SEED,
            "validation_seed": VALIDATION_SEED,
            "dtype": "torch.float32",
            "amp_enabled": False,
            "tf32_mutated": False,
            "validation_ddpm_steps": 0 if core_no_diff else 100,
            "test_ddpm_steps": 0 if core_no_diff else 1000,
            "validation_ensemble": 1,
            "test_ensemble": 1 if core_no_diff else NON_CORE_TEST_ENSEMBLE,
            "training_batch_size": SMOKE_BATCH_SIZE[condition],
            "smoke_optimizer_steps": 1,
            "smoke_validation_samples": 1,
            "smoke_test_samples": 1,
        },
        "stages": stages,
    }
    model = None
    diffusion = None
    validation_diffusion = None
    test_diffusion = None
    optimizer = None
    reloaded_model = None
    train_dataset = None
    validation_dataset = None
    test_dataset = None
    tf32_before = _tf32_state(torch)
    combo_started = time.perf_counter()

    try:
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(0)

        manifests = _read_manifest_evidence(dataset)
        record["manifests"] = manifests
        train_dataset = _build_dataset(runtime, dataset, roots[dataset], "training")
        validation_dataset = _build_dataset(
            runtime, dataset, roots[dataset], "validation"
        )
        test_dataset = _build_dataset(runtime, dataset, roots[dataset], "testing")
        train_loader = _one_batch_loader(
            runtime, train_dataset, SMOKE_BATCH_SIZE[condition]
        )
        validation_loader = _one_batch_loader(runtime, validation_dataset, 1)
        test_loader = _one_batch_loader(runtime, test_dataset, 1)
        observed_partition_totals = {
            "training": len(train_dataset),
            "validation": len(validation_dataset),
            "testing": len(test_dataset),
        }
        expected_partition_totals = dict(config["partition_totals"])
        if observed_partition_totals != expected_partition_totals:
            raise InputBlocked("manifest_partition_total_mismatch")
        stages.append(
            {
                "name": stage_name,
                "status": PASS,
                "partition_totals_expected": expected_partition_totals,
                "partition_totals_observed": observed_partition_totals,
                "partition_totals_match": True,
            }
        )

        stage_name = "formal_gpu_train_step"
        model, diffusion = gpu_preflight.create_formal_model_and_diffusion(
            config,
            core_no_diff=core_no_diff,
            timestep_respacing="",
        )
        model = model.to(torch.device(gpu_preflight.CUDA_DEVICE))
        train_evidence, optimizer = _train_one_step(
            runtime, model, diffusion, train_loader, config, condition
        )
        stages.append({"name": stage_name, "status": PASS, **train_evidence})

        stage_name = "validation_forward_and_metric"
        validation_diffusion = (
            None
            if core_no_diff
            else _formal_eval_diffusion(runtime, VALIDATION_RESPACING)
        )
        _set_seed(runtime, VALIDATION_SEED)
        with tempfile.TemporaryDirectory(prefix="ensemdiff-gpu-smoke-") as temporary:
            temporary_root = Path(temporary)
            with runtime["logger"].scoped_configure(
                dir=str(temporary_root / "logger"), format_strs=[]
            ):
                validation_evidence, validation_row = _run_metric_stage(
                    runtime,
                    model=model,
                    diffusion=validation_diffusion,
                    loader=validation_loader,
                    config=config,
                    audit_mode=audit_mode,
                    ensemble=VALIDATION_ENSEMBLE,
                    expected_steps=0 if core_no_diff else 100,
                    output_csv=temporary_root / "validation.csv",
                    step=1,
                )
            stages.append(
                {"name": stage_name, "status": PASS, **validation_evidence}
            )

            stage_name = "temporary_best_save_schema"
            checkpoint_path, metadata_path, metadata = _save_best_like_trainloop(
                torch,
                model=model,
                row=validation_row,
                directory=temporary_root,
                validation_manifest=_manifest_path(dataset, "validation"),
                audit_mode=audit_mode,
            )
            best_evidence = _verify_best_save(
                checkpoint_path=checkpoint_path,
                metadata_path=metadata_path,
                metadata=metadata,
                validation_manifest=_manifest_path(dataset, "validation"),
                audit_mode=audit_mode,
            )
            stages.append({"name": stage_name, "status": PASS, **best_evidence})

            # Release the live training graph/model before creating the strict
            # reload target.  The only surviving model source is best_model.pt.
            optimizer = None
            validation_diffusion = None
            diffusion = None
            model = None
            _cleanup_cuda(runtime)

            stage_name = "best_checkpoint_reload_cuda0"
            reloaded_model, test_diffusion = (
                gpu_preflight.create_formal_model_and_diffusion(
                    config,
                    core_no_diff=core_no_diff,
                    timestep_respacing=TEST_RESPACING,
                )
            )
            reloaded_model = reloaded_model.to(
                torch.device(gpu_preflight.CUDA_DEVICE)
            )
            reload_evidence = _reload_best(
                torch, reloaded_model, checkpoint_path
            )
            stages.append(
                {"name": stage_name, "status": PASS, **reload_evidence}
            )

            stage_name = "final_test_inference_and_metric"
            _set_seed(runtime, SEED)
            with runtime["logger"].scoped_configure(
                dir=str(temporary_root / "test_logger"), format_strs=[]
            ):
                test_evidence, _ = _run_metric_stage(
                    runtime,
                    model=reloaded_model,
                    diffusion=None if core_no_diff else test_diffusion,
                    loader=test_loader,
                    config=config,
                    audit_mode=audit_mode,
                    ensemble=(
                        1 if core_no_diff else NON_CORE_TEST_ENSEMBLE
                    ),
                    expected_steps=0 if core_no_diff else 1000,
                    output_csv=temporary_root / "test.csv",
                    step=2,
                )
            stages.append({"name": stage_name, "status": PASS, **test_evidence})

        stage_name = "configuration_invariants"
        tf32_after = _tf32_state(torch)
        _require(tf32_after == tf32_before, "tf32_state_changed")
        _require(
            gpu_preflight.FORMAL_MODEL_CONFIG["use_fp16"] is False,
            "formal_use_fp16_changed",
        )
        stages.append(
            {
                "name": stage_name,
                "status": PASS,
                "tf32_before": tf32_before,
                "tf32_after": tf32_after,
                "amp_enabled": False,
                "dtype": "torch.float32",
            }
        )
        record["memory"] = _memory(torch)
        _require(record["memory"]["peak_allocated_bytes"] > 0, "combo_gpu_memory_zero")
        record["elapsed_sec"] = round(time.perf_counter() - combo_started, 3)
        record["status"] = PASS
        return record
    except InputBlocked as error:
        stages.append(
            {
                "name": stage_name,
                "status": BLOCKED,
                "error": {
                    "kind": type(error).__name__,
                    "code": str(error),
                },
            }
        )
        record["elapsed_sec"] = round(time.perf_counter() - combo_started, 3)
        record["status"] = BLOCKED
        return record
    except Exception as error:  # continue remaining combinations; sanitized
        stages.append(
            {
                "name": stage_name,
                "status": FAIL,
                "error": {
                    "kind": type(error).__name__,
                    "code": (
                        str(error)
                        if isinstance(error, AcceptanceFailure)
                        else "gpu_acceptance_stage_failed"
                    ),
                },
            }
        )
        record["elapsed_sec"] = round(time.perf_counter() - combo_started, 3)
        record["status"] = FAIL
        return record
    finally:
        reloaded_model = None
        test_diffusion = None
        optimizer = None
        validation_diffusion = None
        diffusion = None
        model = None
        test_dataset = None
        validation_dataset = None
        train_dataset = None
        _cleanup_cuda(runtime)


def _blocked_records(
    preflight: dict[str, Any],
    *,
    status: str,
    stage: str,
    code: str,
) -> list[dict[str, Any]]:
    records = []
    for dataset, condition in COMBINATIONS:
        core_no_diff = CONDITIONS[condition] == "core_no_diff"
        records.append(
            {
                "record_type": "gpu_smoke_combination",
                "dataset": dataset,
                "condition": condition,
                "audit_mode": CONDITIONS[condition],
                "status": status,
                "device": _device_evidence(preflight),
                "manifests": {
                    split: {
                        "release_path": _manifest_display_path(dataset, split)
                    }
                    for split in ("training", "validation", "testing")
                },
                "configuration": {
                    "formal_unet": True,
                    "training_batch_size": SMOKE_BATCH_SIZE[condition],
                    "validation_ddpm_steps": 0 if core_no_diff else 100,
                    "test_ddpm_steps": 0 if core_no_diff else 1000,
                    "test_ensemble": 1 if core_no_diff else 5,
                    "amp_enabled": False,
                    "tf32_mutated": False,
                },
                "stages": [
                    {
                        "name": stage,
                        "status": status,
                        "error": {"code": code},
                    }
                ],
            }
        )
    return records


def _emit(record: dict[str, Any]) -> None:
    print(
        json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        flush=True,
    )


def _write_report(path: Path, document: dict[str, Any]) -> None:
    report = path.expanduser().resolve()
    if not report.parent.is_dir():
        raise FileNotFoundError("report_parent_missing")
    with report.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the 12 real EnsemDiff GPU smoke combinations. Supply exactly "
            "one preprocessed root for btcv, acdc, and isic2018."
        )
    )
    parser.add_argument(
        "--data-root",
        action="append",
        required=True,
        metavar="DATASET=PATH",
        help=(
            "Preprocessed dataset root containing training/validation/testing; "
            "repeat for btcv, acdc, and isic2018."
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        help=(
            "Optional new JSON report path. The parent must exist and an existing "
            "file is never overwritten. By default no persistent artifact is made."
        ),
    )
    return parser


def main() -> int:
    run_started = time.perf_counter()
    parser = create_parser()
    args = parser.parse_args()
    try:
        roots = _parse_data_roots(args.data_root)
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))

    os.environ["ENSEMDIFF_USE_VISDOM"] = "0"
    os.environ["ENSEMDIFF_SAMPLE_PROGRESS"] = "0"
    preflight = gpu_preflight.run_preflight()
    records: list[dict[str, Any]]
    runtime = None

    if preflight["status"] != PASS:
        records = _blocked_records(
            preflight,
            status=BLOCKED if preflight["status"] == BLOCKED else FAIL,
            stage="gpu_preflight",
            code=str(preflight.get("error", {}).get("check", "preflight_failed")),
        )
    else:
        try:
            schema = _trainloop_best_schema()
            _require(schema == BEST_METADATA_KEYS, "trainloop_best_schema_mismatch")
            audit_modes = _audit_branch_contract()
            _require(
                audit_modes
                == {
                    "core_no_diff",
                    "train_random_yt",
                    "train_shuffle_yt",
                },
                "formal_audit_branch_contract_mismatch",
            )
            runtime = _import_runtime()
        except Exception as error:
            records = _blocked_records(
                preflight,
                status=FAIL,
                stage="acceptance_program_static_contract",
                code=(
                    str(error)
                    if isinstance(error, AcceptanceFailure)
                    else "static_contract_failed"
                ),
            )
        else:
            records = []
            for dataset, condition in COMBINATIONS:
                record = _run_combo(
                    runtime, preflight, roots, dataset, condition
                )
                records.append(record)
                _emit(record)

    # Preflight/static-contract failures still emit one JSON record per combo.
    if not (preflight["status"] == PASS and runtime is not None):
        for record in records:
            _emit(record)

    counts = {
        status: sum(record["status"] == status for record in records)
        for status in (PASS, FAIL, BLOCKED)
    }
    overall = PASS if counts[PASS] == len(COMBINATIONS) else (
        FAIL if counts[FAIL] else BLOCKED
    )
    summary = {
        "record_type": "gpu_smoke_summary",
        "status": overall,
        "expected_combinations": len(COMBINATIONS),
        "counts": counts,
        "preflight": preflight,
        "persistent_artifacts": [] if args.report is None else ["requested_report"],
        "elapsed_sec": round(time.perf_counter() - run_started, 3),
        "temporary_artifacts_cleaned": True,
    }
    _emit(summary)

    if args.report is not None:
        try:
            _write_report(
                args.report,
                {"summary": summary, "combinations": records},
            )
        except Exception as error:
            _emit(
                {
                    "record_type": "gpu_smoke_report_error",
                    "status": FAIL,
                    "error": {
                        "kind": type(error).__name__,
                        "code": "optional_report_write_failed",
                    },
                }
            )
            return 1

    _cleanup_cuda(runtime)
    if overall == PASS:
        return 0
    if overall == BLOCKED:
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
