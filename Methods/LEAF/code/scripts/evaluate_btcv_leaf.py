"""Evaluate a LEAF checkpoint on the LEAF-owned BTCV PNG cache."""

import argparse
import csv
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import torch
from diffusers import DDIMScheduler
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader

from leaf import AutoencoderKL, CoreNoDiffSegmentor, LatentEncoder, LeafOutput, LeafPipeline, UNetModel
from src.data import dataset_name_class_dict
from src.util.runtime import normalize_audit_mode, require_rtx5090, resolve_from_code_root
from src.util.seeding import generate_seed_sequence


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    mask_values = torch.stack([example["mask_values"] for example in examples])
    return {
        "pixel_values": pixel_values.to(memory_format=torch.contiguous_format).float(),
        "mask_values": mask_values.to(memory_format=torch.contiguous_format).float(),
    }


def dice_iou(pred, target):
    pred = pred.bool()
    target = target.bool()
    intersection = torch.logical_and(pred, target).sum().item()
    pred_sum = pred.sum().item()
    target_sum = target.sum().item()
    if target_sum == 0:
        return None, None, int(pred_sum), int(target_sum)
    dice_den = pred_sum + target_sum
    iou_den = pred_sum + target_sum - intersection
    return (2.0 * intersection) / dice_den, intersection / iou_den, int(pred_sum), int(target_sum)


def mean(values):
    valid = [value for value in values if value is not None]
    return float(sum(valid) / len(valid)) if valid else 0.0


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_pipeline(cfg, checkpoint_dir, device, dtype, prefer_ema=True):
    checkpoint_dir = Path(checkpoint_dir)
    if prefer_ema and not (checkpoint_dir / "unet_ema").is_dir():
        raise FileNotFoundError(
            f"EMA-selected final evaluation requires checkpoint subfolder: {checkpoint_dir / 'unet_ema'}"
        )
    unet_subfolder = "unet_ema" if prefer_ema else "unet"
    vae = AutoencoderKL.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="vae", local_files_only=True)
    unet = UNetModel.from_pretrained(checkpoint_dir, subfolder=unet_subfolder, local_files_only=True)
    latent_encoder = LatentEncoder.from_pretrained(checkpoint_dir, subfolder="latent_encoder", local_files_only=True)
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0015,
        beta_end=0.0155,
        prediction_type=cfg.prediction_type,
        clip_sample=False,
    )
    pipeline = LeafPipeline(vae=vae, unet=unet, latent_encoder=latent_encoder, scheduler=scheduler)
    pipeline = pipeline.to(device)
    pipeline.vae.to(device, dtype=dtype)
    pipeline.unet.to(device, dtype=dtype)
    pipeline.latent_encoder.to(device, dtype=dtype)
    pipeline.set_progress_bar_config(disable=True)
    pipeline.unet.eval()
    pipeline.latent_encoder.eval()
    return pipeline, unet_subfolder


def load_core_no_diff_model(cfg, checkpoint_dir, device, dtype, prefer_ema=True):
    checkpoint_dir = Path(checkpoint_dir)
    if prefer_ema and not (checkpoint_dir / "core_no_diff_segmentor_ema").is_dir():
        raise FileNotFoundError(
            "EMA-selected final evaluation requires checkpoint subfolder: "
            f"{checkpoint_dir / 'core_no_diff_segmentor_ema'}"
        )
    model_subfolder = "core_no_diff_segmentor_ema" if prefer_ema else "core_no_diff_segmentor"
    model = CoreNoDiffSegmentor.from_pretrained(
        checkpoint_dir,
        subfolder=model_subfolder,
        local_files_only=True,
    )
    vae = AutoencoderKL.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="vae", local_files_only=True)
    model.to(device, dtype=dtype)
    vae.to(device, dtype=dtype)
    model.eval()
    vae.eval()
    return model, vae, model_subfolder


def save_mask_png(mask, path):
    mask = mask.detach().cpu().clone().squeeze().numpy().astype(np.uint8) * 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(path)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--data-root", help="Override dataset-specific base_data_dir")
    parser.add_argument("--pretrained-path", help="Override local LEAF VAE/U-Net assets")
    parser.add_argument("--split", default="test", choices=["train", "val", "validation", "test"])
    parser.add_argument("--sample_dir", required=True)
    parser.add_argument("--summary_csv", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--per_slice_csv", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_inference_steps", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prefer_ema", action="store_true")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.data_root:
        cfg.base_data_dir = args.data_root
    if args.pretrained_path:
        cfg.pretrained_model_name_or_path = args.pretrained_path
    cfg.base_data_dir = resolve_from_code_root(cfg.base_data_dir, CODE_ROOT)
    cfg.pretrained_model_name_or_path = resolve_from_code_root(cfg.pretrained_model_name_or_path, CODE_ROOT)
    audit_mode = normalize_audit_mode(cfg.get("audit_mode", "none"))
    split = "val" if args.split == "validation" else args.split
    device = require_rtx5090(args.device)
    weight_dtype = torch.float32
    if device.type == "cuda" and cfg.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif device.type == "cuda" and cfg.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    data_cls = dataset_name_class_dict[cfg.dataset_name]
    dataset_dir = Path(cfg.base_data_dir) / cfg.dataset_name
    dataset = data_cls(dataset_dir=str(dataset_dir), split=split, resolution=cfg.resolution, seed=cfg.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    if audit_mode == "core_no_diff":
        model, vae, model_subfolder = load_core_no_diff_model(cfg, args.checkpoint_dir, device, weight_dtype, args.prefer_ema)
        pipeline = None
        unet_subfolder = ""
    else:
        pipeline, unet_subfolder = load_pipeline(cfg, args.checkpoint_dir, device, weight_dtype, args.prefer_ema)
        model = None
        vae = None
        model_subfolder = unet_subfolder
    primary_model = model if model is not None else pipeline.unet
    model_device = str(next(primary_model.parameters()).device)
    torch.cuda.reset_peak_memory_stats(device)
    seed_sequence = generate_seed_sequence(cfg.seed, len(dataloader))
    sample_dir = Path(args.sample_dir)
    rows = []
    autocast_ctx = torch.autocast(device.type, dtype=weight_dtype) if device.type == "cuda" else nullcontext()
    inference_seconds = 0.0

    index_offset = 0
    for batch in dataloader:
        rgb = batch["pixel_values"].to(device)
        target = torch.mean(batch["mask_values"].to(device), dim=1, keepdim=True) > args.threshold
        seed = seed_sequence.pop()
        generator = None
        if seed is not None and device.type == "cuda":
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)

        torch.cuda.synchronize(device)
        inference_started = time.perf_counter()
        with autocast_ctx:
            if audit_mode == "core_no_diff":
                pred_mask_latent, _ = model(rgb)
                mask_pred = vae.decode(pred_mask_latent / model.scaling_factor)
                mask_pred = torch.clamp(mask_pred, -1.0, 1.0)
                mask_pred = (mask_pred + 1.0) / 2.0
                pred = torch.mean(mask_pred, dim=1, keepdim=True) > args.threshold
            else:
                pipe_out: LeafOutput = pipeline(
                    rgb,
                    num_inference_steps=args.num_inference_steps,
                    generator=generator,
                    show_progress_bar=False,
                )
                pred = torch.mean(pipe_out.mask_pred, dim=1, keepdim=True) > args.threshold
        torch.cuda.synchronize(device)
        inference_seconds += time.perf_counter() - inference_started
        if rgb.device.type != "cuda" or pred.device.type != "cuda":
            raise RuntimeError("BTCV final inference left CUDA; CPU fallback is forbidden.")
        for batch_idx in range(pred.shape[0]):
            dataset_index = index_offset + batch_idx
            image_path = Path(dataset.image_filenames[dataset_index])
            slice_id = image_path.stem
            pred_i = pred[batch_idx : batch_idx + 1]
            target_i = target[batch_idx : batch_idx + 1]
            dsc, iou, pred_pixels, gt_pixels = dice_iou(pred_i, target_i)
            save_mask_png(pred_i, sample_dir / f"{slice_id}_pred.png")
            rows.append(
                {
                    "slice_id": slice_id,
                    "image": str(image_path),
                    "dice": dsc,
                    "iou": iou,
                    "pred_pixels": pred_pixels,
                    "gt_pixels": gt_pixels,
                    "is_empty_gt": int(gt_pixels == 0),
                    "is_empty_pred": int(pred_pixels == 0),
                }
            )
        index_offset += pred.shape[0]

    nonempty_rows = [row for row in rows if not row["is_empty_gt"]]
    summary = {
        "config": str(Path(args.config).resolve()),
        "checkpoint_dir": str(Path(args.checkpoint_dir).resolve()),
        "audit_mode": audit_mode,
        "unet_subfolder": unet_subfolder,
        "model_subfolder": model_subfolder,
        "split": split,
        "sample_dir": str(sample_dir.resolve()),
        "num_slices": len(rows),
        "num_nonempty_gt": len(nonempty_rows),
        "num_empty_gt": len(rows) - len(nonempty_rows),
        "num_empty_pred": sum(row["is_empty_pred"] for row in rows),
        "num_inference_steps": 0 if audit_mode == "core_no_diff" else args.num_inference_steps,
        "sampling_steps_main_core": 0 if audit_mode == "core_no_diff" else args.num_inference_steps,
        "gpu_name": torch.cuda.get_device_name(device),
        "cuda_device": str(device),
        "model_device": model_device,
        "inference_time_total_s": inference_seconds,
        "inference_time_per_slice_s": inference_seconds / max(len(rows), 1),
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "threshold": args.threshold,
        "dice": mean([row["dice"] for row in rows]),
        "iou": mean([row["iou"] for row in rows]),
        "dice_nonempty_gt": mean([row["dice"] for row in nonempty_rows]),
        "iou_nonempty_gt": mean([row["iou"] for row in nonempty_rows]),
        "metric_policy": "skip_empty_ground_truth_slices",
    }

    if rows:
        write_csv(args.per_slice_csv, rows, list(rows[0].keys()))
    else:
        raise RuntimeError(f"No rows evaluated for split={split}")
    write_csv(args.summary_csv, [summary], list(summary.keys()))
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
