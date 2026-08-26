"""Evaluate EnsemDiff BTCV sampled masks with Dice and IoU.

The sampler saves one tensor per slice and ensemble member:
    <slice_id>_output0, <slice_id>_output1, ...
This script compares those samples with each slice directory's *_seg.npy.
Slices with empty ground-truth masks are skipped for Dice/IoU aggregation.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch


def load_target(slice_dir):
    seg_paths = sorted(Path(slice_dir).glob("*_seg.npy"))
    if not seg_paths:
        raise FileNotFoundError(f"No *_seg.npy found under {slice_dir}")
    return torch.as_tensor(np.load(seg_paths[0]), dtype=torch.float32).squeeze()


def load_sample(path):
    sample = torch.load(path, map_location="cpu")
    if isinstance(sample, np.ndarray):
        sample = torch.as_tensor(sample)
    if not torch.is_tensor(sample):
        raise TypeError(f"Expected tensor or ndarray in {path}, got {type(sample)}")
    return sample.float().squeeze()


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="BTCV test slice directory.")
    parser.add_argument("--sample_dir", required=True, help="Directory with sampled masks.")
    parser.add_argument("--num_ensemble", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--per_slice_csv",
        default="",
        help="Per-slice output CSV. Defaults to <sample_dir>/metrics_per_slice.csv.",
    )
    parser.add_argument(
        "--summary_json",
        default="",
        help="Summary JSON path. Defaults to <sample_dir>/metrics_summary.json.",
    )
    parser.add_argument(
        "--summary_csv",
        default="",
        help="Summary CSV path. Defaults to <sample_dir>/metrics_summary.csv.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    sample_dir = Path(args.sample_dir).expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Missing data_dir: {data_dir}")
    if not sample_dir.is_dir():
        raise FileNotFoundError(f"Missing sample_dir: {sample_dir}")

    per_slice_csv = (
        Path(args.per_slice_csv)
        if args.per_slice_csv
        else sample_dir / "metrics_per_slice.csv"
    )
    summary_json = (
        Path(args.summary_json)
        if args.summary_json
        else sample_dir / "metrics_summary.json"
    )
    summary_csv = (
        Path(args.summary_csv)
        if args.summary_csv
        else sample_dir / "metrics_summary.csv"
    )

    rows = []
    for slice_dir in sorted([path for path in data_dir.iterdir() if path.is_dir()]):
        slice_id = slice_dir.name
        target = load_target(slice_dir)
        samples = []
        row = {"slice_id": slice_id}
        for idx in range(args.num_ensemble):
            sample_path = sample_dir / f"{slice_id}_output{idx}"
            if not sample_path.exists():
                raise FileNotFoundError(f"Missing sample: {sample_path}")
            sample = load_sample(sample_path)
            pred = sample > args.threshold
            dsc, iou, pred_pixels, gt_pixels = dice_iou(pred, target > 0.5)
            row[f"sample{idx}_dice"] = dsc
            row[f"sample{idx}_iou"] = iou
            row[f"sample{idx}_pred_pixels"] = pred_pixels
            row["gt_pixels"] = gt_pixels
            samples.append(sample)

        ensemble_mean = torch.stack(samples, dim=0).mean(dim=0)
        ensemble_pred = ensemble_mean > args.threshold
        dsc, iou, pred_pixels, gt_pixels = dice_iou(ensemble_pred, target > 0.5)
        row["ensemble_dice"] = dsc
        row["ensemble_iou"] = iou
        row["ensemble_pred_pixels"] = pred_pixels
        row["gt_pixels"] = gt_pixels
        row["is_empty_gt"] = int(gt_pixels == 0)
        row["is_empty_ensemble_pred"] = int(pred_pixels == 0)
        rows.append(row)

    if not rows:
        raise RuntimeError(f"No slice directories found under {data_dir}")

    nonempty_rows = [row for row in rows if not row["is_empty_gt"]]
    summary = {
        "data_dir": str(data_dir),
        "sample_dir": str(sample_dir),
        "num_slices": len(rows),
        "num_nonempty_gt": len(nonempty_rows),
        "num_empty_gt": len(rows) - len(nonempty_rows),
        "num_empty_ensemble_pred": sum(row["is_empty_ensemble_pred"] for row in rows),
        "num_ensemble": args.num_ensemble,
        "threshold": args.threshold,
        "sample0_dice": mean([row["sample0_dice"] for row in rows]),
        "sample0_iou": mean([row["sample0_iou"] for row in rows]),
        "ensemble_dice": mean([row["ensemble_dice"] for row in rows]),
        "ensemble_iou": mean([row["ensemble_iou"] for row in rows]),
        "metric_policy": "skip_empty_ground_truth_slices",
        "sample0_dice_nonempty_gt": mean([row["sample0_dice"] for row in nonempty_rows]),
        "sample0_iou_nonempty_gt": mean([row["sample0_iou"] for row in nonempty_rows]),
        "ensemble_dice_nonempty_gt": mean([row["ensemble_dice"] for row in nonempty_rows]),
        "ensemble_iou_nonempty_gt": mean([row["ensemble_iou"] for row in nonempty_rows]),
    }
    for idx in range(args.num_ensemble):
        summary[f"sample{idx}_dice"] = mean([row[f"sample{idx}_dice"] for row in rows])
        summary[f"sample{idx}_iou"] = mean([row[f"sample{idx}_iou"] for row in rows])

    fieldnames = list(rows[0].keys())
    write_csv(per_slice_csv, rows, fieldnames)
    write_csv(summary_csv, [summary], list(summary.keys()))
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
