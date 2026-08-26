"""End-to-end RTX 5090 smoke matrix for three datasets and four conditions.

Each condition performs one real optimizer step, validation loss/metric
inference, temporary validation-selected checkpoint save/reload, and test
inference. Temporary checkpoints are removed before the process exits.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import tempfile
import time
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = RELEASE_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from ldm.audit import normalize_audit_mode
from ldm.data.manifest import canonical_sample_owner, read_manifest_ids
from ldm.runtime import require_rtx5090
from ldm.util import instantiate_from_config


SEED = 23
DDIM_STEPS = 10
DATASETS = {
    "btcv": {
        "config": "btcv-cls2-ldm-kl-8.yaml",
        "paths": {
            "train": ("BTCV", "train"),
            "validation": ("BTCV", "test"),
            "validation_metrics": ("BTCV", "test"),
            "test": ("BTCV", "test"),
        },
    },
    "acdc": {
        "config": "acdc-cls4-ldm-kl-8.yaml",
        "paths": {
            "train": ("ACDC", "training"),
            "validation": ("ACDC", "validation"),
            "validation_metrics": ("ACDC", "validation"),
            "test": ("ACDC", "testing"),
        },
    },
    "isic2018": {
        "config": "isic-ldm-kl-8.yaml",
        "paths": {
            "train": ("ISIC18", "training"),
            "validation": ("ISIC18", "validation"),
            "validation_metrics": ("ISIC18", "validation"),
            "test": ("ISIC18", "testing"),
        },
    },
}
CONDITIONS = {
    "full_diffusion": "none",
    "train_random_yt": "train_random_yt",
    "train_shuffle_yt": "train_shuffle_yt",
    "core_no_diff": "core_no_diff",
}
MANIFEST_SPLITS = {
    "train": "train",
    "validation": "val",
    "validation_metrics": "val",
    "test": "test",
}


def _parse_mapping(values: list[str], allowed: set[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        key, separator, raw_path = value.partition("=")
        key = key.lower().strip()
        if not separator or key not in allowed or not raw_path.strip():
            raise ValueError(f"Invalid mapping {value!r}; expected NAME=PATH")
        result[key] = Path(raw_path.strip()).expanduser().resolve()
    missing = sorted(allowed - set(result))
    if missing:
        raise ValueError(f"Missing data roots: {missing}")
    return result


def _seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _data_dir(root: Path, components: tuple[str, ...]) -> Path:
    candidates = [root.joinpath(*components)]
    if root.name.casefold() == components[0].casefold():
        candidates.insert(0, root.joinpath(*components[1:]))
    if root.name.casefold() == components[-1].casefold():
        candidates.insert(0, root)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Cannot resolve data directory below supplied root for {'/'.join(components)}"
    )


def _config(dataset: str, data_root: Path, ldm_ckpt: Path, vae_ckpt: Path, mode: str):
    path = CODE_ROOT / "configs" / "latent-diffusion" / DATASETS[dataset]["config"]
    config = OmegaConf.load(path)
    for phase, components in DATASETS[dataset]["paths"].items():
        config.data.params[phase].params.data_root = str(_data_dir(data_root, components))
    config.model.params.ckpt_path = str(ldm_ckpt)
    config.model.params.first_stage_config.params.ckpt_path = str(vae_ckpt)
    config.model.params.cond_stage_config.params.ckpt_path = str(vae_ckpt)
    config.model.params.audit_mode = mode
    return config


def _manifest_record(dataset: str, phase: str, dataset_object) -> dict:
    split = MANIFEST_SPLITS[phase]
    configured = Path(dataset_object.manifest_path).as_posix()
    expected_suffix = f"manifests/{dataset}/{split}.txt"
    if not configured.endswith(expected_suffix):
        raise RuntimeError(
            f"{dataset}/{phase} used {configured!r}, expected {expected_suffix!r}"
        )
    expected_owners = set(read_manifest_ids(dataset_object.manifest_path, dataset))
    observed_owners = {
        canonical_sample_owner(dataset, path) for path in dataset_object.image_paths
    }
    if observed_owners != expected_owners:
        raise RuntimeError(
            f"{dataset}/{phase} owners differ from its fixed {split} manifest"
        )
    return {
        "phase": phase,
        "split": split,
        "manifest": expected_suffix,
        "owners": len(observed_owners),
        "samples": len(dataset_object),
    }


def _datasets(config, dataset: str):
    result = {
        phase: instantiate_from_config(config.data.params[phase])
        for phase in MANIFEST_SPLITS
    }
    expected_modes = {
        "train": "train",
        "validation": "val",
        "validation_metrics": "test",
        "test": "test",
    }
    for phase, expected_mode in expected_modes.items():
        if getattr(result[phase], "mode", None) != expected_mode:
            raise RuntimeError(
                f"{dataset}/{phase} expected loader mode {expected_mode!r}, "
                f"got {getattr(result[phase], 'mode', None)!r}"
            )
    records = {
        phase: _manifest_record(dataset, phase, object_)
        for phase, object_ in result.items()
    }
    if records["validation"]["manifest"] != records["validation_metrics"]["manifest"]:
        raise RuntimeError("Validation loss and validation metric views must use one manifest")
    return result, records


def _cuda_batch(dataset, device: torch.device, batch_size: int):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    batch = next(iter(loader))
    batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    tensor_devices = {
        key: str(value.device)
        for key, value in batch.items()
        if isinstance(value, torch.Tensor)
    }
    if not tensor_devices or any(not value.startswith("cuda:") for value in tensor_devices.values()):
        raise RuntimeError(f"Model batch contains a non-CUDA tensor: {tensor_devices}")
    return batch, tensor_devices


def _foreground_loader(dataset, dataset_name: str):
    selected = None
    for index in range(len(dataset)):
        sample = dataset[index]
        segmentation = np.asarray(sample["segmentation"])
        if np.any(segmentation > 0):
            selected = index
            break
    if selected is None:
        raise RuntimeError("No foreground sample is available for a finite smoke metric")
    owner = canonical_sample_owner(dataset_name, dataset.image_paths[selected])
    loader = DataLoader(Subset(dataset, [selected]), batch_size=1, shuffle=False, num_workers=0)
    return loader, owner


def _parameter_count(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _assert_model_cuda(model) -> str:
    devices = {str(parameter.device) for parameter in model.parameters()}
    if not devices or any(not device.startswith("cuda:") for device in devices):
        raise RuntimeError(f"Model parameters are not entirely on CUDA: {sorted(devices)}")
    return ",".join(sorted(devices))


def _first_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _register_path_hooks(model):
    seen = {"tam": False, "hsem": False, "mcf": False, "unet": False}
    evidence = {"input_channels": [], "input_devices": [], "output_devices": []}

    def mark(name):
        def hook(_module, _args, _output):
            seen[name] = True
        return hook

    def unet_hook(_module, args, output):
        seen["unet"] = True
        input_tensor = _first_tensor(args)
        output_tensor = _first_tensor(output)
        if input_tensor is not None:
            evidence["input_channels"].append(int(input_tensor.shape[1]))
            input_device = str(input_tensor.device)
            if input_device not in evidence["input_devices"]:
                evidence["input_devices"].append(input_device)
        if output_tensor is not None:
            output_device = str(output_tensor.device)
            if output_device not in evidence["output_devices"]:
                evidence["output_devices"].append(output_device)

    handles = [
        model.cond_stage_model.oeem.register_forward_hook(mark("tam")),
        model.cond_stage_model.hsem.register_forward_hook(mark("hsem")),
        model.cond_stage_model.hff.register_forward_hook(mark("mcf")),
        model.model.diffusion_model.input_blocks[0].register_forward_hook(unet_hook),
    ]
    return seen, evidence, handles


def _register_unet_hook(model):
    evidence = {"calls": 0, "input_devices": [], "output_devices": []}

    def hook(_module, args, output):
        evidence["calls"] += 1
        input_tensor = _first_tensor(args)
        output_tensor = _first_tensor(output)
        if input_tensor is not None:
            input_device = str(input_tensor.device)
            if input_device not in evidence["input_devices"]:
                evidence["input_devices"].append(input_device)
        if output_tensor is not None:
            output_device = str(output_tensor.device)
            if output_device not in evidence["output_devices"]:
                evidence["output_devices"].append(output_device)

    handle = model.model.diffusion_model.input_blocks[0].register_forward_hook(hook)
    return evidence, handle


def _optimizer_and_scheduler(model):
    configured = model.configure_optimizers()
    scheduler = None
    if isinstance(configured, tuple):
        optimizers, schedulers = configured
        optimizer = optimizers[0]
        if schedulers:
            scheduler = schedulers[0].get("scheduler", schedulers[0])
    elif isinstance(configured, list):
        optimizer = configured[0]
    else:
        optimizer = configured
    if not isinstance(optimizer, torch.optim.AdamW):
        raise RuntimeError(f"Expected formal AdamW optimizer, got {type(optimizer).__name__}")
    return optimizer, scheduler


def _train_step(model, batch, condition: str, device: torch.device):
    model.train()
    model.zero_grad(set_to_none=True)
    _seed()
    seen, evidence, handles = _register_path_hooks(model)
    optimizer, scheduler = _optimizer_and_scheduler(model)
    try:
        loss, loss_dict = model.shared_step(batch)
        if loss.device != device or not torch.isfinite(loss):
            raise RuntimeError(f"Invalid CUDA training loss: {loss}")
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise RuntimeError("Missing or non-finite trainable gradients")
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        step_values = [
            state.get("step") for state in optimizer.state.values() if "step" in state
        ]
        if not step_values or not all(float(value) >= 1.0 for value in step_values):
            raise RuntimeError("AdamW optimizer state does not prove an optimizer step")
        model.on_train_batch_end()
        torch.cuda.synchronize(device)
    finally:
        for handle in handles:
            handle.remove()

    if not all(seen.values()):
        raise RuntimeError(f"Auxiliary/core path did not fully execute: {seen}")
    expected_channels = 4 if condition == "core_no_diff" else 8
    if not evidence["input_channels"] or evidence["input_channels"][-1] != expected_channels:
        raise RuntimeError(
            f"{condition} expected {expected_channels} UNet channels, "
            f"observed {evidence['input_channels']}"
        )
    all_devices = evidence["input_devices"] + evidence["output_devices"]
    if not all_devices or any(not value.startswith("cuda:") for value in all_devices):
        raise RuntimeError(f"Training core used a non-CUDA device: {evidence}")

    trace = dict(getattr(model, "audit_trace", {}))
    mode = CONDITIONS[condition]
    if trace.get("audit_mode") != mode:
        raise RuntimeError(f"Audit trace mismatch for {condition}: {trace}")
    if condition == "train_shuffle_yt":
        permutation = trace.get("shuffle_permutation")
        if permutation is None or any(index == source for index, source in enumerate(permutation)):
            raise RuntimeError("train_shuffle_yt did not enforce a no-self-match derangement")
    if condition in {"train_random_yt", "train_shuffle_yt"}:
        if not trace.get(f"{condition}_active"):
            raise RuntimeError(f"{condition} branch was not active")
        for invariant in ("target_changed", "loss_changed", "model_output_changed"):
            if trace.get(invariant) is not False:
                raise RuntimeError(f"{condition} violated {invariant}=false")
    if condition == "core_no_diff" and not trace.get("audit_active"):
        raise RuntimeError("core_no_diff branch was not active")

    result = {
        "loss": float(loss.detach().cpu()),
        "loss_device": str(loss.device),
        "forward": True,
        "backward": True,
        "optimizer_step": True,
        "optimizer": type(optimizer).__name__,
        "scheduler": type(scheduler).__name__ if scheduler is not None else None,
        "auxiliary_modules_executed": seen,
        "unet": evidence,
        "audit_trace": trace,
        "loss_keys": sorted(loss_dict),
    }
    model.zero_grad(set_to_none=True)
    del gradients, optimizer, scheduler
    _clear_cuda()
    return result


def _validation_loss(model, batch, device: torch.device):
    model.eval()
    evidence, handle = _register_unet_hook(model)
    try:
        with torch.no_grad():
            loss, _ = model.shared_step(batch)
        torch.cuda.synchronize(device)
    finally:
        handle.remove()
    if loss.device != device or not torch.isfinite(loss):
        raise RuntimeError(f"Invalid validation loss: {loss}")
    devices = evidence["input_devices"] + evidence["output_devices"]
    if evidence["calls"] < 1 or any(not value.startswith("cuda:") for value in devices):
        raise RuntimeError(f"Validation forward did not remain on CUDA: {evidence}")
    return float(loss.detach().cpu()), evidence


def _metric_inference(model, loader, prefix: str, sampler: str, device: torch.device):
    model.eval()
    evidence, handle = _register_unet_hook(model)
    try:
        metrics, _ = model.log_dice(
            data=loader,
            save_dir=None,
            ddim_steps=DDIM_STEPS,
            sampler_name=sampler,
            metric_prefix=prefix,
        )
        torch.cuda.synchronize(device)
    finally:
        handle.remove()
    values = np.asarray(metrics[f"{prefix}_avg_dice"], dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise RuntimeError(f"{prefix} Dice has no finite foreground value: {values}")
    devices = evidence["input_devices"] + evidence["output_devices"]
    if evidence["calls"] < 1 or any(not value.startswith("cuda:") for value in devices):
        raise RuntimeError(f"{prefix} inference did not remain on CUDA: {evidence}")
    return float(finite.mean()), evidence


def _model_probe(model) -> torch.Tensor:
    weight = model.model.diffusion_model.input_blocks[0][0].weight
    return weight.detach().reshape(-1)[:32].float().cpu().clone()


def _save_best(model, validation_metric: float, checkpoint: Path) -> int:
    if not np.isfinite(validation_metric):
        raise RuntimeError("Cannot select a best checkpoint from a non-finite validation metric")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "monitor": "val_avg_dice",
            "score": float(validation_metric),
            "seed": SEED,
        },
        checkpoint,
    )
    size = checkpoint.stat().st_size
    if size <= 0:
        raise RuntimeError("Temporary best checkpoint is empty")
    return size


def _reload(config, checkpoint: Path, expected_probe: torch.Tensor, device: torch.device):
    model = instantiate_from_config(config.model)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("monitor") != "val_avg_dice":
        raise RuntimeError("Temporary checkpoint did not retain validation monitor metadata")
    incompatible = model.load_state_dict(payload["state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Strict checkpoint reload failed: {incompatible}")
    del payload
    if not torch.equal(_model_probe(model), expected_probe):
        raise RuntimeError("Reloaded model probe differs from the saved model")
    model = model.to(device)
    _assert_model_cuda(model)
    return model


def _clear_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _run_condition(
    dataset: str,
    condition: str,
    config,
    checkpoint: Path,
    smoke_batch_size: int,
    device: torch.device,
):
    started = time.perf_counter()
    datasets, manifests = _datasets(config, dataset)
    train_batch, train_input_devices = _cuda_batch(
        datasets["train"], device, smoke_batch_size
    )
    validation_batch, validation_input_devices = _cuda_batch(
        datasets["validation"], device, 1
    )
    validation_loader, validation_owner = _foreground_loader(
        datasets["validation_metrics"], dataset
    )
    test_loader, test_owner = _foreground_loader(datasets["test"], dataset)

    _seed()
    model = instantiate_from_config(config.model).to(device)
    model.learning_rate = float(config.model.base_learning_rate)
    model.audit_mode = normalize_audit_mode(CONDITIONS[condition])
    model_parameter_device = _assert_model_cuda(model)
    parameter_count = _parameter_count(model)
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    torch.cuda.reset_peak_memory_stats(device)

    training = _train_step(model, train_batch, condition, device)
    validation_loss, validation_forward = _validation_loss(model, validation_batch, device)
    sampler = "direct" if condition == "core_no_diff" else "ddim"
    validation_metric, validation_inference = _metric_inference(
        model, validation_loader, "val", sampler, device
    )
    probe = _model_probe(model)
    checkpoint_size = _save_best(model, validation_metric, checkpoint)

    del train_batch, validation_batch, model
    _clear_cuda()

    loaded_model = _reload(config, checkpoint, probe, device)
    test_metric, test_inference = _metric_inference(
        loaded_model, test_loader, "test", sampler, device
    )
    peak_memory = round(torch.cuda.max_memory_allocated(device) / 2**20, 1)
    del loaded_model
    _clear_cuda()
    checkpoint.unlink()
    if checkpoint.exists():
        raise RuntimeError("Temporary checkpoint cleanup failed")

    return {
        "dataset": dataset,
        "condition": condition,
        "audit_mode": CONDITIONS[condition],
        "status": "PASS",
        "formal_batch_size": int(config.data.params.batch_size),
        "smoke_train_batch_size": smoke_batch_size,
        "smoke_validation_batch_size": 1,
        "smoke_test_batch_size": 1,
        "ddim_steps": None if sampler == "direct" else DDIM_STEPS,
        "sampler": sampler,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "model_parameter_device": model_parameter_device,
        "train_input_devices": train_input_devices,
        "validation_input_devices": validation_input_devices,
        "training": training,
        "validation_forward": True,
        "validation_loss": validation_loss,
        "validation_cuda_evidence": validation_forward,
        "validation_metric_name": "val_avg_dice",
        "validation_metric": validation_metric,
        "validation_inference_cuda_evidence": validation_inference,
        "best_checkpoint_selected_from_validation": True,
        "checkpoint_saved": True,
        "checkpoint_size_bytes": checkpoint_size,
        "checkpoint_loaded_strict": True,
        "checkpoint_removed": True,
        "test_inference": True,
        "test_metric_name": "test_avg_dice",
        "test_metric": test_metric,
        "test_inference_cuda_evidence": test_inference,
        "validation_sample_owner": validation_owner,
        "test_sample_owner": test_owner,
        "manifests": manifests,
        "peak_cuda_memory_mib": peak_memory,
        "duration_seconds": round(time.perf_counter() - started, 3),
        "device": str(device),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", action="append", required=True, metavar="DATASET=PATH")
    parser.add_argument("--ldm-ckpt", required=True)
    parser.add_argument("--vae-ckpt", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--json-output")
    args = parser.parse_args()
    if args.batch_size <= 1:
        raise ValueError("GPU matrix requires batch_size > 1 for train_shuffle_yt")

    runtime = require_rtx5090(0)
    device = torch.device("cuda", 0)
    torch.backends.cudnn.benchmark = True
    roots = _parse_mapping(args.data_root, set(DATASETS))
    ldm_ckpt = Path(args.ldm_ckpt).expanduser().resolve()
    vae_ckpt = Path(args.vae_ckpt).expanduser().resolve()
    for checkpoint in (ldm_ckpt, vae_ckpt):
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing required pretrained checkpoint: {checkpoint}")

    rows = []
    with tempfile.TemporaryDirectory(prefix=".gpu_smoke_", dir=RELEASE_ROOT) as temp:
        temp_root = Path(temp)
        for dataset in DATASETS:
            dataset_rows = []
            for condition, mode in CONDITIONS.items():
                config = _config(dataset, roots[dataset], ldm_ckpt, vae_ckpt, mode)
                checkpoint = temp_root / f"{dataset}-{condition}-best.ckpt"
                row = _run_condition(
                    dataset,
                    condition,
                    config,
                    checkpoint,
                    args.batch_size,
                    device,
                )
                dataset_rows.append(row)
                rows.append(row)
                print(f"PASS {dataset}/{condition}")
            input_audit_counts = {
                row["parameter_count"]
                for row in dataset_rows
                if row["condition"] != "core_no_diff"
            }
            if len(input_audit_counts) != 1:
                raise RuntimeError(
                    f"Full/random/shuffle parameter counts differ for {dataset}: "
                    f"{sorted(input_audit_counts)}"
                )

    if len(rows) != 12 or any(row["status"] != "PASS" for row in rows):
        raise RuntimeError("The complete 12-condition GPU matrix did not pass")
    leftovers = list(RELEASE_ROOT.glob(".gpu_smoke_*"))
    if leftovers:
        raise RuntimeError("GPU smoke temporary directory cleanup failed")

    payload = {
        "runtime": runtime,
        "formal_seed": SEED,
        "formal_batch_size": 12,
        "smoke_train_batch_size": args.batch_size,
        "smoke_validation_batch_size": 1,
        "smoke_test_batch_size": 1,
        "matrix_status": "PASS",
        "conditions_passed": len(rows),
        "coverage": [
            "train_manifest",
            "training_forward_loss_backward_optimizer_step",
            "validation_manifest_forward_metric",
            "validation_selected_checkpoint_save_reload",
            "test_manifest_inference_from_reloaded_checkpoint",
            "audit_branch_and_cuda_device_evidence",
        ],
        "results": rows,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if args.json_output:
        output = Path(args.json_output).expanduser()
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite GPU smoke JSON: {output}")
        output.write_text(encoded + "\n", encoding="utf-8")
        print(f"GPU_SMOKE_JSON_SAVED {args.json_output}")
    else:
        print("GPU_SMOKE_JSON_BEGIN")
        print(encoded)
        print("GPU_SMOKE_JSON_END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
