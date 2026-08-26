import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


CLASS_NAMES = {
    1: "rv",
    2: "myocardium",
    3: "lv",
}


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


def read_label_map(path, num_classes):
    arr = np.asarray(Image.open(path))
    if arr.ndim == 2:
        labels = arr
    elif arr.ndim == 3 and arr.shape[2] >= num_classes:
        labels = arr[:, :, :num_classes].argmax(axis=2)
    elif arr.ndim == 3 and arr.shape[2] >= 3 and np.array_equal(arr[:, :, 0], arr[:, :, 1]) and np.array_equal(arr[:, :, 0], arr[:, :, 2]):
        labels = arr[:, :, 0]
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        foreground = arr[:, :, :3] > 0
        if np.any(foreground.sum(axis=2) > 1):
            raise ValueError(f"RGB ACDC mask has overlapping foreground classes: {path}")
        labels = np.zeros(arr.shape[:2], dtype=np.uint8)
        labels[foreground[:, :, 0]] = 1
        labels[foreground[:, :, 1]] = 2
        labels[foreground[:, :, 2]] = 3
    else:
        raise ValueError(f"Unsupported ACDC mask shape {arr.shape}: {path}")

    labels = labels.astype(np.int64)
    unique = np.unique(labels)
    if np.any((unique < 0) | (unique >= num_classes)):
        raise ValueError(f"Label values outside 0..{num_classes - 1} in {path}: {unique.tolist()}")
    return labels


def case_id_from_stem(stem):
    parts = stem.split("_")
    if len(parts) >= 2 and parts[0].lower().startswith("patient"):
        return "_".join(parts[:2])
    return parts[0]


def mean_or_none(values):
    return None if not values else float(np.mean(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="ACDC split directory containing label_maps/ or masks/.")
    parser.add_argument("--sample_dir", required=True, help="slice2seg output seed directory containing *-logits.png.")
    parser.add_argument("--summary_csv", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--per_slice_csv", required=True)
    parser.add_argument("--per_case_csv", default=None)
    parser.add_argument("--num_classes", type=int, default=4)
    args = parser.parse_args()

    if args.num_classes != 4:
        raise ValueError("ACDC evaluation expects num_classes=4: background + RV + myocardium + LV.")

    data_dir = Path(args.data_dir)
    label_dir = data_dir / "label_maps"
    if not label_dir.is_dir():
        label_dir = data_dir / "masks"
    sample_dir = Path(args.sample_dir)

    if not label_dir.is_dir():
        raise FileNotFoundError(f"Missing ACDC label_maps or masks directory under: {data_dir}")
    if not sample_dir.is_dir():
        raise FileNotFoundError(f"Missing sample directory: {sample_dir}")

    rows = []
    missing = []
    class_dice = defaultdict(list)
    class_iou = defaultdict(list)
    case_class_dice = defaultdict(lambda: defaultdict(list))
    case_class_iou = defaultdict(lambda: defaultdict(list))
    evaluated = 0
    skipped_empty_gt = 0

    for label_path in sorted(label_dir.glob("*.png")):
        pred_path = sample_dir / f"{label_path.stem}-logits.png"
        if not pred_path.is_file():
            missing.append(str(pred_path))
            continue

        gt = read_label_map(label_path, args.num_classes)
        pred = read_label_map(pred_path, args.num_classes)
        if pred.shape != gt.shape:
            raise ValueError(f"Prediction shape {pred.shape} != GT shape {gt.shape}: {pred_path}")

        case_id = case_id_from_stem(label_path.stem)
        for class_id in range(1, args.num_classes):
            gt_cls = gt == class_id
            pred_cls = pred == class_id
            dice, iou = binary_metrics(pred_cls, gt_cls)
            status = "evaluated"
            if dice is None:
                skipped_empty_gt += 1
                status = "skipped_empty_gt"
            else:
                evaluated += 1
                class_dice[class_id].append(dice)
                class_iou[class_id].append(iou)
                case_class_dice[case_id][class_id].append(dice)
                case_class_iou[case_id][class_id].append(iou)

            rows.append({
                "slice": label_path.name,
                "case_id": case_id,
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "prediction": str(pred_path),
                "gt_pixels": int(gt_cls.sum()),
                "pred_pixels": int(pred_cls.sum()),
                "dice": "" if dice is None else f"{dice:.10f}",
                "iou": "" if iou is None else f"{iou:.10f}",
                "status": status,
            })

    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} prediction files; first missing: {missing[0]}")
    if evaluated == 0:
        raise RuntimeError("No non-empty ACDC foreground class observations were available for Dice/IoU evaluation.")

    dice_values = [value for values in class_dice.values() for value in values]
    iou_values = [value for values in class_iou.values() for value in values]
    summary = {
        "data_dir": str(data_dir),
        "sample_dir": str(sample_dir),
        "num_classes": args.num_classes,
        "class_ids": "1=RV,2=myocardium,3=LV",
        "metric_policy": "foreground class Dice/IoU; skip empty-GT class observations; exclude background",
        "num_slices": len(list(label_dir.glob("*.png"))),
        "evaluated_non_empty_gt_class_observations": evaluated,
        "skipped_empty_gt_class_observations": skipped_empty_gt,
        "dice_mean": float(np.mean(dice_values)),
        "iou_mean": float(np.mean(iou_values)),
    }
    for class_id in range(1, args.num_classes):
        summary[f"dice_mean_class_{class_id}_{CLASS_NAMES[class_id]}"] = mean_or_none(class_dice[class_id])
        summary[f"iou_mean_class_{class_id}_{CLASS_NAMES[class_id]}"] = mean_or_none(class_iou[class_id])

    case_rows = []
    for case_id in sorted(case_class_dice):
        row = {"case_id": case_id}
        dice_case_values = []
        iou_case_values = []
        for class_id in range(1, args.num_classes):
            dice_mean = mean_or_none(case_class_dice[case_id][class_id])
            iou_mean = mean_or_none(case_class_iou[case_id][class_id])
            row[f"dice_class_{class_id}_{CLASS_NAMES[class_id]}"] = "" if dice_mean is None else f"{dice_mean:.10f}"
            row[f"iou_class_{class_id}_{CLASS_NAMES[class_id]}"] = "" if iou_mean is None else f"{iou_mean:.10f}"
            if dice_mean is not None:
                dice_case_values.append(dice_mean)
            if iou_mean is not None:
                iou_case_values.append(iou_mean)
        row["dice_macro"] = "" if not dice_case_values else f"{np.mean(dice_case_values):.10f}"
        row["iou_macro"] = "" if not iou_case_values else f"{np.mean(iou_case_values):.10f}"
        case_rows.append(row)

    Path(args.per_slice_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.per_slice_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "slice",
            "case_id",
            "class_id",
            "class_name",
            "prediction",
            "gt_pixels",
            "pred_pixels",
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

    if args.per_case_csv:
        case_fields = ["case_id"]
        for class_id in range(1, args.num_classes):
            case_fields.extend([
                f"dice_class_{class_id}_{CLASS_NAMES[class_id]}",
                f"iou_class_{class_id}_{CLASS_NAMES[class_id]}",
            ])
        case_fields.extend(["dice_macro", "iou_macro"])
        Path(args.per_case_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.per_case_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=case_fields)
            writer.writeheader()
            writer.writerows(case_rows)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
