"""Run the twelve real-data RTX 5090 smoke paths and clean temporary artifacts."""

import argparse
import json
import random
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F


RELEASE_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = RELEASE_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT))

from guided_diffusion.acdcloader import ACDCDataset, labels_from_foreground_channels
from guided_diffusion.btcvloader import BTCVDataset
from guided_diffusion.isicloader import ISICDataset
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults


DATASETS = ("btcv", "acdc", "isic2018")
CONDITIONS = ("full", "train_random_yt", "train_shuffle_yt", "core_no_diff")
AUDIT_MODES = {
    "full": "none",
    "train_random_yt": "train_random_yt",
    "train_shuffle_yt": "train_shuffle_yt",
    "core_no_diff": "core_no_diff",
}
EXPECTED_SAMPLES = {
    "btcv": {"train": 2211, "val": 295, "test": 1273},
    "acdc": {"train": 1304, "val": 182, "test": 416},
    "isic2018": {"train": 2594, "val": 100, "test": 1000},
}


def require_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU fallback is not permitted")
    if torch.cuda.device_count() < 1:
        raise RuntimeError("No CUDA device was detected")
    torch.cuda.set_device(0)
    name = torch.cuda.get_device_name(0)
    if name != "NVIDIA GeForce RTX 5090":
        raise RuntimeError(f"Expected NVIDIA GeForce RTX 5090 at cuda:0, found {name!r}")
    return torch.device("cuda:0"), name


def read_config():
    return json.loads((CODE_ROOT / "configs" / "experiments.json").read_text(encoding="utf-8"))


def existing(root, candidates):
    for candidate in candidates:
        path = Path(root) / candidate
        if path.is_dir():
            return path
    raise FileNotFoundError(f"No expected path under {root}: {candidates}")


def split_paths(dataset, root):
    root = Path(root)
    if dataset == "btcv":
        train = existing(root, ("BTCV/train", "BTCV/training", "train"))
        heldout = existing(root, ("BTCV/test", "BTCV/testing", "test"))
        val = root / "BTCV/val"
        return {"train": train, "val": val if val.is_dir() else heldout, "test": heldout}
    if dataset == "acdc":
        return {"train": root, "val": root, "test": root}
    if dataset == "isic2018":
        return {
            "train": existing(root, ("ISIC18/training", "training")),
            "val": existing(root, ("ISIC18/validation", "validation")),
            "test": existing(root, ("ISIC18/testing", "testing")),
        }
    raise ValueError(dataset)


def manifest_for(config, dataset, split):
    path = RELEASE_ROOT / config["datasets"][dataset][f"{split}_manifest"]
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def loader_args(dataset_config, manifest, acdc_split):
    return SimpleNamespace(
        image_size=256,
        num_seg_classes=dataset_config["num_seg_classes"],
        num_mask_channels=dataset_config["num_mask_channels"],
        acdc_split=acdc_split,
        acdc_skip_empty=False,
        data_manifest=str(manifest),
    )


def build_dataset(config, dataset, split, path):
    dataset_config = config["datasets"][dataset]
    manifest = manifest_for(config, dataset, split)
    args = loader_args(
        dataset_config,
        manifest,
        {"train": "training", "val": "validation", "test": "testing"}[split],
    )
    if dataset == "btcv":
        result = BTCVDataset(path, image_size=256, mode=split, data_manifest=manifest)
    elif dataset == "acdc":
        result = ACDCDataset(args, path, mode=split)
    else:
        result = ISICDataset(args, path, mode=split, image_size=256)
    expected = EXPECTED_SAMPLES[dataset][split]
    if len(result) != expected:
        raise AssertionError(f"{dataset}/{split}: expected {expected} samples, loader returned {len(result)}")
    return result


def sample_unit(dataset, name):
    stem = Path(str(name)).stem
    if dataset == "btcv":
        return stem.split("_", 1)[0]
    if dataset == "acdc":
        return stem.split("_", 1)[0]
    return stem


def pick_indices(dataset, data, count):
    foreground = getattr(data, "foreground_pixels", [1] * len(data))
    chosen = []
    units = set()
    for index, pixels in enumerate(foreground):
        if int(pixels) <= 0:
            continue
        _image, _mask, name = data[index]
        unit = sample_unit(dataset, name)
        if unit in units and len(units) < count:
            continue
        chosen.append(index)
        units.add(unit)
        if len(chosen) == count:
            return chosen
    for index in range(len(data)):
        if index not in chosen:
            chosen.append(index)
        if len(chosen) == count:
            return chosen
    raise RuntimeError(f"Could not select {count} smoke samples from {dataset}")


def collate_indices(data, indices, device):
    rows = [data[index] for index in indices]
    images = torch.stack([row[0] for row in rows]).to(device=device, dtype=torch.float32)
    masks = torch.stack([row[1] for row in rows]).to(device=device, dtype=torch.float32)
    names = [str(row[2]) for row in rows]
    return images, masks, names


def model_config(dataset_config):
    config = model_and_diffusion_defaults()
    config.update(
        image_size=256,
        num_channels=128,
        num_res_blocks=2,
        num_heads=1,
        in_ch=3 + dataset_config["num_mask_channels"],
        num_seg_classes=dataset_config["num_seg_classes"],
        num_mask_channels=dataset_config["num_mask_channels"],
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
    return config


def prediction_from_logits(logits):
    if logits.shape[1] == 1:
        return torch.sigmoid(logits)
    if logits.shape[1] == 2:
        return F.softmax(logits, dim=1)[:, 1:2]
    return F.softmax(logits, dim=1)[:, 1:]


def inference(model, diffusion, images, masks, audit_mode, smoke_sampling_steps):
    model.eval()
    with torch.no_grad():
        if audit_mode == "core_no_diff":
            prediction = prediction_from_logits(model.core_no_diff_logits(images))
            branch = "image_only_direct"
        else:
            input_tensor = torch.cat((images, torch.zeros_like(masks)), dim=1)
            sample, _x_noisy, _image, _cal, _cal_out = diffusion.p_sample_loop_known(
                model,
                tuple(input_tensor.shape),
                input_tensor,
                step=smoke_sampling_steps,
                clip_denoised=True,
                model_kwargs={},
            )
            prediction = sample
            branch = "reverse_diffusion"
    if prediction.device.type != "cuda":
        raise RuntimeError(f"Inference output left CUDA: {prediction.device}")
    return prediction, branch


def metric(dataset, prediction, target):
    if dataset != "acdc":
        pred = prediction[:, :1] > 0.5
        truth = target[:, :1] > 0.5
        intersection = (pred & truth).sum(dtype=torch.float64)
        denominator = pred.sum(dtype=torch.float64) + truth.sum(dtype=torch.float64)
        return float(((2.0 * intersection + 1e-6) / (denominator + 1e-6)).item())
    pred_labels = labels_from_foreground_channels(prediction, num_classes=4, threshold=0.5)
    target_labels = labels_from_foreground_channels(target, num_classes=4, threshold=0.5)
    values = []
    for class_id in range(1, 4):
        pred = pred_labels == class_id
        truth = target_labels == class_id
        if truth.sum() == 0:
            continue
        intersection = (pred & truth).sum(dtype=torch.float64)
        denominator = pred.sum(dtype=torch.float64) + truth.sum(dtype=torch.float64)
        values.append((2.0 * intersection + 1e-6) / (denominator + 1e-6))
    if not values:
        raise RuntimeError("ACDC smoke sample has no foreground metric observation")
    return float(torch.stack(values).mean().item())


def run_one(config, dataset, condition, paths, device, gpu_name, smoke_sampling_steps):
    started = time.perf_counter()
    audit_mode = AUDIT_MODES[condition]
    dataset_config = config["datasets"][dataset]
    train_data = build_dataset(config, dataset, "train", paths["train"])
    val_data = build_dataset(config, dataset, "val", paths["val"])
    test_data = build_dataset(config, dataset, "test", paths["test"])
    train_batch_size = 2 if condition == "train_shuffle_yt" else 1
    train_indices = pick_indices(dataset, train_data, train_batch_size)
    val_indices = pick_indices(dataset, val_data, 1)
    test_indices = pick_indices(dataset, test_data, 1)

    torch.manual_seed(23)
    torch.cuda.manual_seed_all(23)
    np.random.seed(23)
    random.seed(23)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    model, diffusion = create_model_and_diffusion(**model_config(dataset_config))
    model.to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.0)
    images, masks, names = collate_indices(train_data, train_indices, device)
    x_start = torch.cat((images, masks), dim=1)
    timesteps = None if audit_mode == "core_no_diff" else torch.arange(train_batch_size, device=device) % 1000

    model.train()
    optimizer.zero_grad(set_to_none=True)
    terms, output = diffusion.training_losses_segmentation(
        model,
        None,
        x_start,
        timesteps,
        model_kwargs={},
        audit_mode=audit_mode,
    )
    loss = terms["loss"].mean()
    if loss.device.type != "cuda" or output.device.type != "cuda":
        raise RuntimeError(f"Training tensors left CUDA: loss={loss.device}, output={output.device}")
    loss.backward()
    active_gradients = sum(
        parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
    )
    if active_gradients == 0:
        raise RuntimeError("No finite model gradients were produced")
    optimizer.step()

    val_images, val_masks, _val_names = collate_indices(val_data, val_indices, device)
    val_prediction, val_branch = inference(
        model, diffusion, val_images, val_masks, audit_mode, smoke_sampling_steps
    )
    val_metric = metric(dataset, val_prediction, val_masks)
    if not np.isfinite(val_metric):
        raise RuntimeError(f"Validation metric is not finite: {val_metric}")

    with tempfile.TemporaryDirectory(prefix="medsegdiff_smoke_") as tmp:
        checkpoint = Path(tmp) / "validation_selected_best.pt"
        torch.save(model.state_dict(), checkpoint)
        if not checkpoint.is_file():
            raise RuntimeError("Temporary best checkpoint was not saved")
        reloaded, reloaded_diffusion = create_model_and_diffusion(**model_config(dataset_config))
        state_dict = torch.load(checkpoint, map_location="cpu")
        reloaded.load_state_dict(state_dict)
        reloaded.to(device)
        test_images, test_masks, _test_names = collate_indices(test_data, test_indices, device)
        test_prediction, test_branch = inference(
            reloaded, reloaded_diffusion, test_images, test_masks, audit_mode, smoke_sampling_steps
        )
        test_metric = metric(dataset, test_prediction, test_masks)
        if not np.isfinite(test_metric):
            raise RuntimeError(f"Test metric is not finite: {test_metric}")
        del reloaded, reloaded_diffusion, state_dict, test_prediction

    report = {
        "dataset": dataset,
        "condition": condition,
        "audit_mode": audit_mode,
        "status": "PASS",
        "gpu_name": gpu_name,
        "cuda_device": str(device),
        "model_parameter_device": str(next(model.parameters()).device),
        "input_device": str(images.device),
        "loss_device": str(loss.device),
        "parameter_count": parameter_count,
        "training_batch_size": train_batch_size,
        "training_samples_from_distinct_units": len({sample_unit(dataset, name) for name in names}) == train_batch_size,
        "forward": True,
        "loss": True,
        "backward": True,
        "optimizer_step": True,
        "validation_forward": True,
        "validation_metric": val_metric,
        "validation_branch": val_branch,
        "checkpoint_saved_by_validation_metric": True,
        "checkpoint_reloaded": True,
        "test_inference": True,
        "test_metric": test_metric,
        "test_branch": test_branch,
        "train_manifest": config["datasets"][dataset]["train_manifest"],
        "val_manifest": config["datasets"][dataset]["val_manifest"],
        "test_manifest": config["datasets"][dataset]["test_manifest"],
        "smoke_sampling_steps": 0 if audit_mode == "core_no_diff" else smoke_sampling_steps,
        "formal_sampling_steps": 0 if audit_mode == "core_no_diff" else dataset_config["final_sampling_steps"],
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
        "temporary_checkpoint_cleaned": True,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    del model, diffusion, optimizer, images, masks, x_start, output, loss
    torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--btcv-root", type=Path, required=True)
    parser.add_argument("--acdc-root", type=Path, required=True)
    parser.add_argument("--isic2018-root", type=Path, required=True)
    parser.add_argument("--smoke-sampling-steps", type=int, default=1)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.smoke_sampling_steps < 1:
        raise ValueError("--smoke-sampling-steps must be at least 1")

    device, gpu_name = require_gpu()
    config = read_config()
    roots = {"btcv": args.btcv_root, "acdc": args.acdc_root, "isic2018": args.isic2018_root}
    paths = {dataset: split_paths(dataset, root) for dataset, root in roots.items()}
    results = []
    for dataset in DATASETS:
        for condition in CONDITIONS:
            try:
                result = run_one(
                    config,
                    dataset,
                    condition,
                    paths[dataset],
                    device,
                    gpu_name,
                    args.smoke_sampling_steps,
                )
            except Exception as exc:
                result = {
                    "dataset": dataset,
                    "condition": condition,
                    "audit_mode": AUDIT_MODES[condition],
                    "status": "FAIL",
                    "failure_type": type(exc).__name__,
                    "error": str(exc),
                }
                torch.cuda.empty_cache()
            results.append(result)
            print(json.dumps(result, indent=2), flush=True)

    payload = {
        "hardware": gpu_name,
        "device": str(device),
        "smoke_sampling_steps": args.smoke_sampling_steps,
        "formal_configuration_unchanged": True,
        "results": results,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    failures = [row for row in results if row["status"] != "PASS"]
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
