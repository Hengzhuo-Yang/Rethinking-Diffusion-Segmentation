"""Evaluate EnsemDiff ISIC2018 Task 1 sampled lesion masks.

The sampler writes one tensor per image and ensemble member:
    <image_id>_output0, <image_id>_output1, ...

ISIC2018 Task 1 is binary lesion segmentation, so metrics are computed for the
lesion foreground class. The ensemble prediction is the mean of sampled lesion
scores thresholded at 0.5 by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch


def load_target(case_dir: Path) -> torch.Tensor:
    seg_paths = sorted(case_dir.glob("*_seg.npy"))
    if not seg_paths:
        raise FileNotFoundError(f"No *_seg.npy found under {case_dir}")
    target = torch.as_tensor(np.load(seg_paths[0]), dtype=torch.float32)
    if target.ndim == 3 and target.shape[0] == 1:
        target = target[0]
    elif target.ndim == 3 and target.shape[-1] == 1:
        target = target[..., 0]
    elif target.ndim != 2:
        raise ValueError(f"Unexpected target shape at {seg_paths[0]}: {tuple(target.shape)}")
    return target > 0.5


def load_sample_scores(path: Path, image_channels: int) -> torch.Tensor:
    try:
        sample = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        sample = torch.load(path, map_location="cpu")
    if isinstance(sample, np.ndarray):
        sample = torch.as_tensor(sample)
    if not torch.is_tensor(sample):
        raise TypeError(f"Expected tensor or ndarray in {path}, got {type(sample)}")

    sample = sample.float()
    if sample.ndim == 4 and sample.shape[0] == 1:
        sample = sample[0]
    if sample.ndim == 2:
        return sample
    if sample.ndim != 3:
        raise ValueError(f"Unexpected prediction shape at {path}: {tuple(sample.shape)}")

    if sample.shape[0] == 1:
        return sample[0]
    if sample.shape[0] == image_channels + 1:
        return sample[-1]
    if sample.shape[-1] == 1:
        return sample[..., 0]
    if sample.shape[-1] == image_channels + 1:
        return sample[..., -1]
    raise ValueError(
        f"Expected binary mask scores or image+mask channels at {path}; "
        f"got {tuple(sample.shape)}"
    )


def dice_iou(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, float, int, int]:
    pred = pred.bool()
    target = target.bool()
    intersection = torch.logical_and(pred, target).sum().item()
    pred_sum = pred.sum().item()
    target_sum = target.sum().item()
    dice_den = pred_sum + target_sum
    iou_den = pred_sum + target_sum - intersection
    dice = 1.0 if dice_den == 0 else (2.0 * intersection) / dice_den
    iou = 1.0 if iou_den == 0 else intersection / iou_den
    return float(dice), float(iou), int(pred_sum), int(target_sum)


def mean(values: list[float]) -> float:
    valid = [value for value in values if math.isfinite(value)]
    return float(sum(valid) / len(valid)) if valid else 0.0


def std(values: list[float]) -> float:
    valid = [value for value in values if math.isfinite(value)]
    if len(valid) <= 1:
        return 0.0
    avg = sum(valid) / len(valid)
    return float((sum((value - avg) ** 2 for value in valid) / (len(valid) - 1)) ** 0.5)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Preprocessed ISIC2018 split directory.")
    parser.add_argument("--sample_dir", required=True, help="Directory with sampled masks.")
    parser.add_argument("--num_ensemble", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--image_channels", type=int, default=3)
    parser.add_argument(
        "--per_image_csv",
        default="",
        help="Per-image output CSV. Defaults to <sample_dir>/isic2018_metrics_per_image.csv.",
    )
    parser.add_argument(
        "--summary_json",
        default="",
        help="Summary JSON path. Defaults to <sample_dir>/isic2018_metrics_summary.json.",
    )
    parser.add_argument(
        "--summary_csv",
        default="",
        help="Summary CSV path. Defaults to <sample_dir>/isic2018_metrics_summary.csv.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    sample_dir = Path(args.sample_dir).expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Missing data_dir: {data_dir}")
    if not sample_dir.is_dir():
        raise FileNotFoundError(f"Missing sample_dir: {sample_dir}")

    rows = []
    for case_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        image_id = case_dir.name
        target = load_target(case_dir)
        samples = []
        row = {"image_id": image_id}
        for idx in range(args.num_ensemble):
            sample_path = sample_dir / f"{image_id}_output{idx}"
            if not sample_path.exists():
                raise FileNotFoundError(f"Missing sample: {sample_path}")
            scores = load_sample_scores(sample_path, args.image_channels)
            pred = scores > args.threshold
            dsc, iou, pred_pixels, gt_pixels = dice_iou(pred, target)
            row[f"sample{idx}_dice"] = dsc
            row[f"sample{idx}_iou"] = iou
            row[f"sample{idx}_pred_pixels"] = pred_pixels
            row["gt_pixels"] = gt_pixels
            samples.append(scores)

        ensemble_scores = torch.stack(samples, dim=0).mean(dim=0)
        ensemble_pred = ensemble_scores > args.threshold
        dsc, iou, pred_pixels, gt_pixels = dice_iou(ensemble_pred, target)
        row["ensemble_dice"] = dsc
        row["ensemble_iou"] = iou
        row["ensemble_pred_pixels"] = pred_pixels
        row["gt_pixels"] = gt_pixels
        row["is_empty_gt"] = int(gt_pixels == 0)
        row["is_empty_ensemble_pred"] = int(pred_pixels == 0)
        rows.append(row)

    if not rows:
        raise RuntimeError(f"No ISIC2018 case directories found under {data_dir}")

    per_image_csv = (
        Path(args.per_image_csv)
        if args.per_image_csv
        else sample_dir / "isic2018_metrics_per_image.csv"
    )
    summary_json = (
        Path(args.summary_json)
        if args.summary_json
        else sample_dir / "isic2018_metrics_summary.json"
    )
    summary_csv = (
        Path(args.summary_csv)
        if args.summary_csv
        else sample_dir / "isic2018_metrics_summary.csv"
    )

    fieldnames = list(rows[0].keys())
    write_csv(per_image_csv, rows, fieldnames)

    summary = {
        "data_dir": str(data_dir),
        "sample_dir": str(sample_dir),
        "num_images": len(rows),
        "num_empty_gt": sum(row["is_empty_gt"] for row in rows),
        "num_empty_ensemble_pred": sum(row["is_empty_ensemble_pred"] for row in rows),
        "num_ensemble": args.num_ensemble,
        "threshold": args.threshold,
        "image_channels": args.image_channels,
        "task": "binary lesion segmentation",
        "sample0_dice": mean([row["sample0_dice"] for row in rows]),
        "sample0_iou": mean([row["sample0_iou"] for row in rows]),
        "sample0_dice_std": std([row["sample0_dice"] for row in rows]),
        "sample0_iou_std": std([row["sample0_iou"] for row in rows]),
        "ensemble_dice": mean([row["ensemble_dice"] for row in rows]),
        "ensemble_iou": mean([row["ensemble_iou"] for row in rows]),
        "ensemble_dice_std": std([row["ensemble_dice"] for row in rows]),
        "ensemble_iou_std": std([row["ensemble_iou"] for row in rows]),
    }
    for idx in range(args.num_ensemble):
        summary[f"sample{idx}_dice"] = mean([row[f"sample{idx}_dice"] for row in rows])
        summary[f"sample{idx}_iou"] = mean([row[f"sample{idx}_iou"] for row in rows])

    write_csv(summary_csv, [summary], list(summary.keys()))
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
