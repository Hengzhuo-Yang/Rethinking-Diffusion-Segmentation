#!/usr/bin/env python3
"""Run the release's 3-dataset x 4-condition CUDA smoke matrix.

The smoke limits runtime data to two training samples and a configurable
number of validation/test samples.  It never changes the formal YAMLs.  Each
combination performs a real CUDA training forward, loss, backward and AdamW
step; validation inference; a temporary full-state checkpoint save/reload;
and test inference.  A JSON report is written atomically and temporary
checkpoints are deleted even when a combination fails.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from ldm.util import instantiate_from_config, load_torch_checkpoint  # noqa: E402
from validate_splits import SPECS, validate_release_splits  # noqa: E402


EXPECTED_GPU = "NVIDIA GeForce RTX 5090"
SEED = 23
DATASETS = ("btcv", "acdc", "isic2018")
CONDITIONS = ("full", "random-yt", "shuffle-yt", "core-no-diff")
AUDIT_MODES = {
    "full": "full_diffusion",
    "random-yt": "train_random_yt",
    "shuffle-yt": "train_shuffle_yt",
    "core-no-diff": "core_no_diff",
}
NUM_CLASSES = {"btcv": 2, "acdc": 4, "isic2018": 2}


def _json_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _write_json_atomic(path: Path, payload: dict) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_value(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _scalar(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _finite_metric_list(values, name: str) -> list[float]:
    result = [float(value) for value in values]
    if not result or not all(math.isfinite(value) for value in result):
        raise RuntimeError(f"{name} is empty or non-finite: {result}")
    return result


def _normalise_optimizer(configured):
    if isinstance(configured, torch.optim.Optimizer):
        return configured
    if isinstance(configured, tuple):
        configured = configured[0]
    if isinstance(configured, list):
        if not configured:
            raise RuntimeError("configure_optimizers returned an empty list")
        candidate = configured[0]
        if isinstance(candidate, torch.optim.Optimizer):
            return candidate
        if isinstance(candidate, dict) and isinstance(candidate.get("optimizer"), torch.optim.Optimizer):
            return candidate["optimizer"]
    if isinstance(configured, dict) and isinstance(configured.get("optimizer"), torch.optim.Optimizer):
        return configured["optimizer"]
    raise TypeError(f"Unsupported configure_optimizers result: {type(configured).__name__}")


def _move_batch_to_device(value, device: torch.device):
    """Recursively mirror Lightning's batch-to-device transfer for manual smoke steps."""

    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move_batch_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_batch_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_batch_to_device(item, device) for item in value)
    return value


def _load_config(dataset: str, condition: str, pretrained_root: Path):
    path = CODE_ROOT / "configs" / "experiments" / dataset / f"{condition}.yaml"
    if not path.is_file():
        raise FileNotFoundError(path)
    os.environ["SDSEG_PRETRAINED_ROOT"] = str(pretrained_root.expanduser().resolve())
    config = OmegaConf.load(path)
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    expected_mode = AUDIT_MODES[condition]
    invariants = {
        "audit_mode": (str(config.model.params.audit_mode), expected_mode),
        "random_seed": (int(config.model.params.random_seed), SEED),
        "num_classes": (int(config.model.params.num_classes), NUM_CLASSES[dataset]),
        "batch_size": (int(config.data.params.batch_size), 4),
        "num_workers": (int(config.data.params.num_workers), 8),
        "max_steps": (int(config.lightning.trainer.max_steps), 100000),
        "learning_rate": (float(config.model.base_learning_rate), 1e-5),
        "monitor": (str(config.model.params.monitor), "val_avg_dice"),
    }
    errors = [
        f"{key}: {actual!r} != {expected!r}"
        for key, (actual, expected) in invariants.items()
        if actual != expected
    ]
    if errors:
        raise ValueError("Formal config drift: " + "; ".join(errors))
    return path, config


def _instantiate_dataset(config, split: str):
    dataset_config = OmegaConf.create(
        OmegaConf.to_container(config.data.params[split], resolve=True)
    )
    dataset_config.params.num_classes = int(config.model.params.num_classes)
    return instantiate_from_config(dataset_config)


def _expected_lengths(dataset: str) -> dict[str, int]:
    return {
        "train": SPECS[dataset]["train"].item_count,
        "metric_validation": SPECS[dataset]["validation"].item_count,
        "test": SPECS[dataset]["test"].item_count,
    }


def _audit_evidence(model, condition: str, loss_dict: dict, train_batch_size: int) -> dict:
    shuffle_flag = _scalar(loss_dict.get("train/audit_train_shuffle_yt", 0.0))
    random_flag = _scalar(loss_dict.get("train/audit_train_random_yt", 0.0))
    core_flag = _scalar(loss_dict.get("train/audit_core_no_diff", 0.0))
    evidence = {
        "audit_mode": str(model.audit_mode),
        "train_shuffle_yt_flag": shuffle_flag,
        "train_random_yt_flag": random_flag,
        "core_no_diff_flag": core_flag,
    }
    if str(model.audit_mode) != AUDIT_MODES[condition]:
        raise RuntimeError(f"Instantiated audit mode does not match {condition}")
    if condition == "random-yt":
        if random_flag != 1.0 or shuffle_flag != 0.0:
            raise RuntimeError(f"random-yt branch flags are wrong: {evidence}")
    elif condition == "shuffle-yt":
        if shuffle_flag != 1.0 or random_flag != 0.0:
            raise RuntimeError(f"shuffle-yt branch flags are wrong: {evidence}")
        permutation = model._last_train_shuffle_yt_perm
        if permutation is None or int(permutation.numel()) != train_batch_size:
            raise RuntimeError("shuffle-yt did not record a batch permutation")
        indices = torch.arange(train_batch_size)
        if torch.any(permutation.cpu() == indices):
            raise RuntimeError(f"shuffle-yt permutation is not a derangement: {permutation.tolist()}")
        evidence["shuffle_permutation"] = permutation.tolist()
        evidence["shuffle_derangement"] = True
    elif condition == "core-no-diff":
        if core_flag != 1.0 or random_flag != 0.0 or shuffle_flag != 0.0:
            raise RuntimeError(f"core-no-diff branch flags are wrong: {evidence}")
        actual_channels = int(model.model.diffusion_model.in_channels)
        if actual_channels != int(configured_channels := model.channels):
            raise RuntimeError(
                f"core-no-diff main input must be image-only: {actual_channels} != {configured_channels}"
            )
        evidence["main_core_input_channels"] = actual_channels
        evidence["original_diffusion_input_channels"] = int(model._core_no_diff_original_in_channels)
        evidence["timestep_passed_to_main_core"] = False
    else:
        if random_flag != 0.0 or shuffle_flag != 0.0 or core_flag != 0.0:
            raise RuntimeError(f"full baseline unexpectedly activated an audit branch: {evidence}")
    return evidence


def _metric_smoke(model, dataset, max_items: int) -> dict:
    subset_size = min(int(max_items), len(dataset))
    if subset_size <= 0:
        raise RuntimeError("Metric smoke subset is empty")
    loader = DataLoader(
        Subset(dataset, range(subset_size)),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    metrics, _, details = model.log_dice(
        data=loader,
        sampler_name="direct",
        ddim_steps=1,
        ddim_eta=0.0,
        return_details=True,
    )
    dice = _finite_metric_list(metrics["val_avg_dice"], "val_avg_dice")
    iou = _finite_metric_list(metrics["val_avg_iou"], "val_avg_iou")
    if int(details.get("requested_steps", 1)) != 1:
        raise RuntimeError(f"Metric smoke did not use one requested step: {details}")
    return {
        "items": subset_size,
        "sampler": str(details.get("sampler_name", "direct")),
        "requested_steps": int(details.get("requested_steps", 1)),
        "mean_dice": float(np.mean(dice)),
        "mean_iou": float(np.mean(iou)),
        "dice": dice,
        "iou": iou,
    }


def _checkpoint_roundtrip(model, directory: Path, metadata: dict) -> dict:
    checkpoint = directory / "smoke_best.ckpt"
    selection = directory / "best_checkpoint.json"
    payload = {"state_dict": model.state_dict(), "global_step": 1}
    torch.save(payload, checkpoint)
    del payload
    size = checkpoint.stat().st_size
    selection_payload = {
        "selection_partition": "validation",
        "metric_dataset_key": "metric_validation",
        "monitor": "val_avg_dice",
        "mode": "max",
        "best_model_path": checkpoint.name,
        "best_model_score": metadata["validation"]["mean_dice"],
        "audit_mode": metadata["audit_mode"],
        "metric_validation_dataset_class": metadata["metric_validation_dataset_class"],
        "num_classes": metadata["num_classes"],
        "random_seed": SEED,
        "smoke_only": True,
    }
    selection.write_text(json.dumps(selection_payload, indent=2), encoding="utf-8")

    restored = load_torch_checkpoint(str(checkpoint), map_location="cpu")
    if "state_dict" not in restored:
        raise RuntimeError("Temporary checkpoint reload has no state_dict")
    missing, unexpected = model.load_state_dict(restored["state_dict"], strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Strict temporary checkpoint reload mismatch: missing={missing}, unexpected={unexpected}"
        )
    restored_keys = len(restored["state_dict"])
    del restored
    return {
        "scope": "full_model_state_dict",
        "size_bytes": size,
        "state_dict_keys": restored_keys,
        "strict_reload": True,
        "selection_partition": "validation",
        "metric_dataset_key": "metric_validation",
        "temporary_files_cleaned": True,
    }


def _run_combination(
    dataset_name: str,
    condition: str,
    data_root: Path,
    pretrained_root: Path,
    max_eval_items: int,
) -> dict:
    started = time.time()
    _seed_everything(SEED)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    config_path, config = _load_config(dataset_name, condition, pretrained_root)
    os.environ["SDSEG_DATA_ROOT"] = str(data_root.expanduser().resolve())

    datasets = {
        split: _instantiate_dataset(config, split)
        for split in ("train", "metric_validation", "test")
    }
    expected_lengths = _expected_lengths(dataset_name)
    actual_lengths = {split: len(value) for split, value in datasets.items()}
    if actual_lengths != expected_lengths:
        raise RuntimeError(f"Dataset lengths differ: {actual_lengths} != {expected_lengths}")

    train_batch_size = 2
    train_loader = DataLoader(
        Subset(datasets["train"], range(train_batch_size)),
        batch_size=train_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    batch = next(iter(train_loader))
    if int(batch["image"].shape[0]) != train_batch_size:
        raise RuntimeError("Training smoke batch was not collated to size two")

    model = instantiate_from_config(config.model)
    model.learning_rate = float(config.model.base_learning_rate)
    device = torch.device("cuda", 0)
    model.to(device)
    if {parameter.device.type for parameter in model.parameters()} != {"cuda"}:
        raise RuntimeError("Model parameters did not move completely to CUDA")
    batch = _move_batch_to_device(batch, device)
    batch_tensor_devices = {
        key: str(value.device)
        for key, value in batch.items()
        if isinstance(value, torch.Tensor)
    }
    if not batch_tensor_devices:
        raise RuntimeError("Training smoke batch has no top-level tensors")
    if set(batch_tensor_devices.values()) != {str(device)}:
        raise RuntimeError(
            f"Training smoke batch tensors are not all on {device}: {batch_tensor_devices}"
        )
    model.train()

    optimizer = _normalise_optimizer(model.configure_optimizers())
    optimizer.zero_grad(set_to_none=True)
    loss, loss_dict = model.shared_step(batch)
    if not torch.is_tensor(loss) or not torch.isfinite(loss):
        raise RuntimeError(f"Training loss is not a finite CUDA tensor: {loss}")
    if loss.device.type != "cuda":
        raise RuntimeError(f"Training loss is not on CUDA: {loss.device}")
    loss.backward()
    gradient_tensors = 0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        gradient_tensors += 1
        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError("Non-finite gradient detected")
    if gradient_tensors == 0:
        raise RuntimeError("No gradients were produced by the training smoke batch")
    optimizer.step()
    audit = _audit_evidence(model, condition, loss_dict, train_batch_size)

    model.eval()
    with torch.inference_mode():
        validation = _metric_smoke(model, datasets["metric_validation"], max_eval_items)
    metric_class = datasets["metric_validation"].__class__
    metric_class_name = f"{metric_class.__module__}.{metric_class.__name__}"
    checkpoint_metadata = {
        "validation": validation,
        "audit_mode": str(model.audit_mode),
        "metric_validation_dataset_class": metric_class_name,
        "num_classes": int(model.num_classes),
    }

    temp_parent = CODE_ROOT.parent
    temporary_path = Path(tempfile.mkdtemp(prefix=".gpu_smoke_", dir=str(temp_parent)))
    try:
        checkpoint = _checkpoint_roundtrip(model, temporary_path, checkpoint_metadata)
        with torch.inference_mode():
            test = _metric_smoke(model, datasets["test"], max_eval_items)
    finally:
        shutil.rmtree(temporary_path, ignore_errors=False)

    result = {
        "status": "pass",
        "is_smoke_test": True,
        "cpu_fallback": False,
        "seed": SEED,
        "dataset": dataset_name,
        "condition": condition,
        "audit_mode": str(model.audit_mode),
        "config": config_path.relative_to(CODE_ROOT).as_posix(),
        "config_sha256": _sha256(config_path),
        "formal_parameters": {
            "seed": SEED,
            "batch_size": 4,
            "num_workers": 8,
            "max_steps": 100000,
            "learning_rate": 1e-5,
            "scale_lr": False,
        },
        "smoke_limits": {"train_items": train_batch_size, "validation_items": validation["items"], "test_items": test["items"]},
        "dataset_lengths": actual_lengths,
        "manifest_sha256": {
            split: SPECS[dataset_name]["validation" if split == "metric_validation" else split].sha256
            for split in ("train", "metric_validation", "test")
        },
        "train": {
            "loss": float(loss.detach().cpu().item()),
            "loss_dict": {key: _scalar(value) for key, value in loss_dict.items()},
            "gradient_tensor_count": gradient_tensors,
            "optimizer": optimizer.__class__.__name__,
            "optimizer_step": True,
            "device": str(loss.device),
            "batch_tensor_devices": batch_tensor_devices,
        },
        "audit": audit,
        "validation": validation,
        "checkpoint": checkpoint,
        "test": test,
        "selection_partition": "validation",
        "final_partition": "test",
        "test_used_for_selection": False,
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated(0)),
        "elapsed_seconds": time.time() - started,
    }
    del optimizer, model, batch, datasets
    gc.collect()
    torch.cuda.empty_cache()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--pretrained-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--max-eval-items", type=int, default=1)
    parser.add_argument("--expected-gpu-name", default=EXPECTED_GPU)
    parser.add_argument("--only-dataset", choices=DATASETS)
    parser.add_argument("--only-condition", choices=CONDITIONS)
    parser.add_argument("--stop-on-failure", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_eval_items < 1:
        raise ValueError("--max-eval-items must be at least 1")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is disabled.")
    torch.cuda.set_device(0)
    gpu_name = torch.cuda.get_device_name(0)
    if gpu_name != args.expected_gpu_name:
        raise RuntimeError(f"Expected GPU {args.expected_gpu_name!r}, found {gpu_name!r}")
    capability = torch.cuda.get_device_capability(0)
    if capability != (12, 0):
        raise RuntimeError(f"Expected RTX 5090 compute capability (12, 0), found {capability}")

    split_report = validate_release_splits(args.data_root)
    datasets = [args.only_dataset] if args.only_dataset else list(DATASETS)
    conditions = [args.only_condition] if args.only_condition else list(CONDITIONS)
    matrix = [(dataset, condition) for dataset in datasets for condition in conditions]
    results = []
    report = {
        "status": "running",
        "is_smoke_test": True,
        "cpu_fallback": False,
        "seed": SEED,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "expected_combinations": len(matrix),
        "full_release_matrix": len(matrix) == 12,
        "gpu": {
            "name": gpu_name,
            "capability": list(capability),
            "torch_version": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
        },
        "split_validation": split_report,
        "results": results,
    }
    _write_json_atomic(args.report, report)

    for index, (dataset_name, condition) in enumerate(matrix, start=1):
        print(f"[gpu-smoke] {index}/{len(matrix)} {dataset_name}/{condition}", flush=True)
        try:
            result = _run_combination(
                dataset_name,
                condition,
                args.data_root,
                args.pretrained_root,
                args.max_eval_items,
            )
        except Exception as exc:
            failed_config = CODE_ROOT / "configs" / "experiments" / dataset_name / f"{condition}.yaml"
            result = {
                "status": "fail",
                "is_smoke_test": True,
                "cpu_fallback": False,
                "seed": SEED,
                "dataset": dataset_name,
                "condition": condition,
                "audit_mode": AUDIT_MODES[condition],
                "config": failed_config.relative_to(CODE_ROOT).as_posix(),
                "config_sha256": _sha256(failed_config) if failed_config.is_file() else None,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated(0)),
            }
            gc.collect()
            torch.cuda.empty_cache()
        results.append(result)
        report["results"] = results
        report["completed_combinations"] = len(results)
        report["passed_combinations"] = sum(item["status"] == "pass" for item in results)
        report["failed_combinations"] = sum(item["status"] != "pass" for item in results)
        _write_json_atomic(args.report, report)
        if result["status"] != "pass" and args.stop_on_failure:
            break

    passed = sum(item["status"] == "pass" for item in results)
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    report["completed_combinations"] = len(results)
    report["passed_combinations"] = passed
    report["failed_combinations"] = len(results) - passed
    report["status"] = "pass" if passed == len(matrix) and len(results) == len(matrix) else "fail"
    report["all_12_passed"] = bool(len(matrix) == 12 and passed == 12)
    _write_json_atomic(args.report, report)
    print(json.dumps({key: report[key] for key in ("status", "completed_combinations", "passed_combinations", "failed_combinations", "all_12_passed")}, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
