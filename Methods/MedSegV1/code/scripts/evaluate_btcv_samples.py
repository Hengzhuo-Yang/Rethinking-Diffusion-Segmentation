import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


def load_binary(path, threshold=127):
    arr = np.asarray(Image.open(path).convert("L"))
    return arr > threshold


def dice_iou(pred, target):
    intersection = np.logical_and(pred, target).sum(dtype=np.float64)
    pred_sum = pred.sum(dtype=np.float64)
    target_sum = target.sum(dtype=np.float64)
    if target_sum == 0:
        return None, None, int(pred_sum), int(target_sum)
    denom = pred_sum + target_sum
    union = pred_sum + target_sum - intersection
    dice = float(2.0 * intersection / denom) if denom > 0 else 0.0
    iou = float(intersection / union) if union > 0 else 0.0
    return dice, iou, int(pred_sum), int(target_sum)


def mean_skip_none(values):
    valid = [value for value in values if value is not None]
    return float(np.mean(valid)) if valid else float("nan")


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="BTCV split directory containing images/ and masks/.")
    parser.add_argument("--sample_dir", required=True, help="Directory containing *_pred.png predictions.")
    parser.add_argument("--summary_csv", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--per_slice_csv", required=True)
    parser.add_argument("--pred_threshold", type=float, default=0.5)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    sample_dir = Path(args.sample_dir)
    mask_paths = sorted((data_dir / "masks").glob("*.png"))
    if not mask_paths:
        raise RuntimeError(f"No BTCV masks found under {data_dir / 'masks'}")

    rows = []
    pred_threshold = int(round(args.pred_threshold * 255.0))
    for mask_path in mask_paths:
        slice_id = mask_path.stem
        pred_path = sample_dir / f"{slice_id}_pred.png"
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing prediction for {slice_id}: {pred_path}")
        pred = load_binary(pred_path, threshold=pred_threshold)
        target = load_binary(mask_path)
        dsc, iou, pred_pixels, gt_pixels = dice_iou(pred, target)
        rows.append(
            {
                "slice_id": slice_id,
                "prediction": str(pred_path),
                "mask": str(mask_path),
                "dice": dsc,
                "iou": iou,
                "pred_pixels": pred_pixels,
                "gt_pixels": gt_pixels,
                "is_empty_gt": int(gt_pixels == 0),
                "is_empty_pred": int(pred_pixels == 0),
            }
        )

    nonempty_rows = [row for row in rows if not row["is_empty_gt"]]
    summary = {
        "dataset": "BTCV",
        "task": "binary_foreground_segmentation",
        "num_slices": len(rows),
        "num_nonempty_gt": len(nonempty_rows),
        "num_empty_gt": len(rows) - len(nonempty_rows),
        "num_empty_pred": sum(row["is_empty_pred"] for row in rows),
        "dice": mean_skip_none([row["dice"] for row in rows]),
        "iou": mean_skip_none([row["iou"] for row in rows]),
        "dice_nonempty_gt": mean_skip_none([row["dice"] for row in nonempty_rows]),
        "iou_nonempty_gt": mean_skip_none([row["iou"] for row in nonempty_rows]),
        "pred_threshold": args.pred_threshold,
        "metric_policy": "skip_empty_ground_truth_slices",
        "sample_dir": str(sample_dir.resolve()),
        "data_dir": str(data_dir.resolve()),
        "per_slice_csv": str(Path(args.per_slice_csv).resolve()),
    }

    write_csv(Path(args.per_slice_csv), rows, list(rows[0].keys()))
    write_csv(Path(args.summary_csv), [summary], list(summary.keys()))
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
