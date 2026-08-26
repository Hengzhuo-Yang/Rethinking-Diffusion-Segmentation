import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


def binary_metrics(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not gt.any():
        return None, None
    intersection = np.logical_and(pred, gt).sum(dtype=np.float64)
    pred_sum = pred.sum(dtype=np.float64)
    gt_sum = gt.sum(dtype=np.float64)
    union = np.logical_or(pred, gt).sum(dtype=np.float64)
    dice = (2.0 * intersection) / (pred_sum + gt_sum + 1e-10)
    iou = intersection / (union + 1e-10)
    return float(dice), float(iou)


def read_binary_mask(path, threshold):
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr > threshold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="BTCV split directory containing images/ and masks/.")
    parser.add_argument("--sample_dir", required=True, help="slice2seg output seed directory containing *-logits.png.")
    parser.add_argument("--summary_csv", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--per_slice_csv", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    mask_dir = data_dir / "masks"
    sample_dir = Path(args.sample_dir)
    threshold_value = int(round(args.threshold * 255.0))

    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Missing BTCV mask directory: {mask_dir}")
    if not sample_dir.is_dir():
        raise FileNotFoundError(f"Missing sample directory: {sample_dir}")

    rows = []
    dice_values = []
    iou_values = []
    missing = []
    empty_gt = 0
    evaluated = 0

    for mask_path in sorted(mask_dir.glob("*.png")):
        pred_path = sample_dir / f"{mask_path.stem}-logits.png"
        if not pred_path.is_file():
            missing.append(str(pred_path))
            continue
        gt = read_binary_mask(mask_path, 0)
        pred = read_binary_mask(pred_path, threshold_value)
        dice, iou = binary_metrics(pred, gt)
        status = "evaluated"
        if dice is None:
            empty_gt += 1
            status = "skipped_empty_gt"
        else:
            evaluated += 1
            dice_values.append(dice)
            iou_values.append(iou)
        rows.append({
            "slice": mask_path.name,
            "prediction": str(pred_path),
            "gt_foreground_pixels": int(gt.sum()),
            "pred_foreground_pixels": int(pred.sum()),
            "dice": "" if dice is None else f"{dice:.10f}",
            "iou": "" if iou is None else f"{iou:.10f}",
            "status": status,
        })

    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} prediction files; first missing: {missing[0]}")
    if evaluated == 0:
        raise RuntimeError("No non-empty GT BTCV slices were available for Dice/IoU evaluation.")

    summary = {
        "data_dir": str(data_dir),
        "sample_dir": str(sample_dir),
        "threshold": args.threshold,
        "metric_policy": "binary foreground Dice/IoU; skip empty-GT foreground slices; exclude background",
        "num_slices": len(rows),
        "evaluated_non_empty_gt_slices": evaluated,
        "skipped_empty_gt_slices": empty_gt,
        "dice_mean": float(np.mean(dice_values)),
        "iou_mean": float(np.mean(iou_values)),
    }

    Path(args.per_slice_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.per_slice_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "slice",
            "prediction",
            "gt_foreground_pixels",
            "pred_foreground_pixels",
            "dice",
            "iou",
            "status",
        ])
        writer.writeheader()
        writer.writerows(rows)

    with open(args.summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
