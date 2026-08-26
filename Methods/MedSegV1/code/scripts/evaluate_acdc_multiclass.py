import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from guided_diffusion.acdcloader import (
    ACDC_CLASS_NAMES,
    labels_from_foreground_channels,
    multiclass_dice_iou,
)


def mean_valid(values):
    valid = [value for value in values if value is not None]
    return float(sum(valid) / len(valid)) if valid else 0.0


def evaluate_file(path, threshold, num_classes):
    with np.load(path) as data:
        prob = data["prob"]
        if "target_labels" in data:
            target_labels = data["target_labels"].astype(np.uint8)
        elif "target_mask" in data:
            target_labels = labels_from_foreground_channels(
                data["target_mask"],
                num_classes=num_classes,
                threshold=threshold,
            )
        else:
            raise ValueError(f"{path} does not contain target_labels or target_mask")
        source = str(data["source"]) if "source" in data else path.stem

    pred_labels = labels_from_foreground_channels(
        prob,
        num_classes=num_classes,
        threshold=threshold,
    )
    class_rows = multiclass_dice_iou(pred_labels, target_labels, num_classes=num_classes)
    slice_dice = mean_valid([row["dice"] for row in class_rows])
    slice_iou = mean_valid([row["iou"] for row in class_rows])
    nonempty = [row for row in class_rows if not row["is_empty_gt"]]
    rows = []
    for row in class_rows:
        rows.append(
            {
                "file": path.name,
                "source": source,
                **row,
                "slice_dice": slice_dice,
                "slice_iou": slice_iou,
                "num_nonempty_gt_classes": len(nonempty),
                "is_empty_gt_foreground": int(len(nonempty) == 0),
            }
        )
    return rows


def evaluate_dir(pred_dir, threshold, num_classes):
    pred_dir = Path(pred_dir)
    files = sorted(pred_dir.glob("*_pred.npz"))
    if not files:
        raise FileNotFoundError(f"No *_pred.npz files found in {pred_dir}")

    rows = []
    for path in files:
        rows.extend(evaluate_file(path, threshold, num_classes))

    valid_rows = [row for row in rows if not row["is_empty_gt"]]
    summary = {
        "dataset": "ACDC",
        "num_classes": int(num_classes),
        "foreground_classes": json.dumps(
            {key: value for key, value in ACDC_CLASS_NAMES.items() if key > 0},
            sort_keys=True,
        ),
        "dice": mean_valid([row["dice"] for row in valid_rows]),
        "iou": mean_valid([row["iou"] for row in valid_rows]),
        "num_slices": len(files),
        "num_metric_observations": len(valid_rows),
        "num_empty_gt_foreground_slices": len(
            {
                row["file"]
                for row in rows
                if row["is_empty_gt_foreground"]
            }
        ),
        "num_empty_pred_class_observations": sum(row["is_empty_pred"] for row in rows),
        "threshold": float(threshold),
        "metric_policy": (
            "foreground Dice/IoU over ACDC classes 1..3; skip empty-GT "
            "slice/class observations; background class 0 is excluded."
        ),
        "dir": str(pred_dir),
    }
    for class_id in range(1, int(num_classes)):
        class_rows = [row for row in rows if row["class_id"] == class_id and not row["is_empty_gt"]]
        summary[f"class{class_id}_name"] = ACDC_CLASS_NAMES[class_id]
        summary[f"class{class_id}_dice"] = mean_valid([row["dice"] for row in class_rows])
        summary[f"class{class_id}_iou"] = mean_valid([row["iou"] for row in class_rows])
        summary[f"class{class_id}_nonempty_gt_observations"] = len(class_rows)
    return summary, rows


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"No rows to write to {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = create_argparser().parse_args()
    summary, rows = evaluate_dir(args.pred_dir, args.threshold, args.num_classes)

    print("ACDC multi-class Dice/IoU summary")
    print(
        f"dice={summary['dice']:.4f} iou={summary['iou']:.4f} "
        f"observations={summary['num_metric_observations']}"
    )
    for class_id in range(1, args.num_classes):
        print(
            f"class{class_id} {summary[f'class{class_id}_name']}: "
            f"dice={summary[f'class{class_id}_dice']:.4f} "
            f"iou={summary[f'class{class_id}_iou']:.4f}"
        )

    if args.out_csv:
        write_csv(args.out_csv, [summary])
        detail_csv = Path(args.out_csv).with_name(Path(args.out_csv).stem + "_slices.csv")
        write_csv(detail_csv, rows)
    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": summary,
            "metric_policy": summary["metric_policy"],
        }
        out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", required=True, help="Directory with *_pred.npz files.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--out_csv", default="", help="Optional summary CSV path.")
    parser.add_argument("--out_json", default="", help="Optional summary JSON path.")
    return parser


if __name__ == "__main__":
    main()
