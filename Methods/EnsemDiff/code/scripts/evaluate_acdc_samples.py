"""Evaluate EnsemDiff ACDC samples with volume-level Dice and IoU.

The sampler writes one tensor per slice and ensemble member:
    <slice_id>_output0, <slice_id>_output1, ...

ACDC metrics are reported over foreground classes only:
    1: right ventricle, 2: myocardium, 3: left ventricle.
Slices are grouped back into patient-frame volumes using manifest.csv when
available, then Dice/IoU are computed per volume and class.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


CLASS_NAMES = {
    1: "right_ventricle",
    2: "myocardium",
    3: "left_ventricle",
}


def load_target(slice_dir: Path, num_classes: int) -> torch.Tensor:
    label_paths = sorted(slice_dir.glob("*_label.npy"))
    if label_paths:
        target = torch.as_tensor(np.load(label_paths[0]))
        return target.long().clamp(min=0, max=num_classes - 1)

    seg_paths = sorted(slice_dir.glob("*_seg.npy"))
    if not seg_paths:
        raise FileNotFoundError(f"No *_label.npy or *_seg.npy found under {slice_dir}")
    target = torch.as_tensor(np.load(seg_paths[0]))
    if target.ndim == 2:
        return target.long().clamp(min=0, max=num_classes - 1)
    if target.ndim == 3 and target.shape[0] == num_classes:
        return target.argmax(dim=0).long()
    if target.ndim == 3 and target.shape[-1] == num_classes:
        return target.argmax(dim=-1).long()
    raise ValueError(f"Unexpected target shape at {seg_paths[0]}: {tuple(target.shape)}")


def load_prediction_scores(
    path: Path,
    *,
    num_classes: int,
    image_channels: int,
) -> torch.Tensor:
    sample = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sample, np.ndarray):
        sample = torch.as_tensor(sample)
    if not torch.is_tensor(sample):
        raise TypeError(f"Expected tensor or ndarray in {path}, got {type(sample)}")

    sample = sample.float()
    if sample.ndim == 4 and sample.shape[0] == 1:
        sample = sample[0]
    if sample.ndim == 2:
        if num_classes != 2:
            raise ValueError(
                f"Binary prediction at {path} cannot be evaluated as {num_classes} classes."
            )
        return torch.stack((1.0 - sample, sample), dim=0)
    if sample.ndim != 3:
        raise ValueError(f"Unexpected prediction shape at {path}: {tuple(sample.shape)}")

    if sample.shape[0] == num_classes:
        return sample
    if sample.shape[0] == image_channels + num_classes:
        return sample[-num_classes:]
    if sample.shape[-1] == num_classes:
        return sample.permute(2, 0, 1)
    if sample.shape[-1] == image_channels + num_classes:
        return sample[..., -num_classes:].permute(2, 0, 1)
    if sample.shape[0] == 1 and num_classes == 2:
        foreground = sample[0]
        return torch.stack((1.0 - foreground, foreground), dim=0)

    raise ValueError(
        f"Expected {num_classes} mask channels, or {image_channels + num_classes} "
        f"image+mask channels, at {path}; got {tuple(sample.shape)}."
    )


def prediction_from_scores(scores: torch.Tensor, threshold: float) -> torch.Tensor:
    if scores.shape[0] == 2:
        return (scores[1] > threshold).long()
    return scores.argmax(dim=0).long()


def dice_iou_for_class(
    pred: torch.Tensor,
    target: torch.Tensor,
    class_id: int,
) -> tuple[float | None, float | None, int, int]:
    pred_bin = pred == class_id
    target_bin = target == class_id
    intersection = torch.logical_and(pred_bin, target_bin).sum().item()
    pred_sum = pred_bin.sum().item()
    target_sum = target_bin.sum().item()
    if target_sum == 0:
        return None, None, int(pred_sum), int(target_sum)
    dice_den = pred_sum + target_sum
    iou_den = pred_sum + target_sum - intersection
    return (
        (2.0 * intersection) / dice_den,
        intersection / iou_den if iou_den else None,
        int(pred_sum),
        int(target_sum),
    )


def clean_float(value: float | None) -> float | None:
    if value is None:
        return None
    if not math.isfinite(value):
        return None
    return float(value)


def mean(values: list[float | None]) -> float:
    valid = [value for value in values if value is not None and math.isfinite(value)]
    return float(sum(valid) / len(valid)) if valid else 0.0


def std(values: list[float | None]) -> float:
    valid = [value for value in values if value is not None and math.isfinite(value)]
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


def find_manifest(data_dir: Path, manifest_path: str) -> Path | None:
    if manifest_path:
        path = Path(manifest_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing manifest: {path}")
        return path

    candidates = [
        data_dir.parent / "manifest.csv",
        data_dir / "manifest.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def parse_slice_id(slice_id: str) -> tuple[str, int]:
    if "_z" not in slice_id:
        return slice_id, 0
    case_id, z_text = slice_id.rsplit("_z", 1)
    try:
        z_index = int(z_text)
    except ValueError:
        z_index = 0
    return case_id, z_index


def load_slice_records(data_dir: Path, manifest_path: Path | None) -> list[dict]:
    slice_dirs = {path.name: path for path in data_dir.iterdir() if path.is_dir()}
    if not slice_dirs:
        raise RuntimeError(f"No slice directories found under {data_dir}")

    records = []
    seen = set()
    if manifest_path is not None:
        with manifest_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                slice_id = row.get("slice_id", "")
                if slice_id not in slice_dirs:
                    continue
                case_id = row.get("frame_id") or parse_slice_id(slice_id)[0]
                z_index_text = row.get("z_index", "")
                try:
                    z_index = int(z_index_text)
                except ValueError:
                    z_index = parse_slice_id(slice_id)[1]
                records.append(
                    {
                        "slice_id": slice_id,
                        "case_id": case_id,
                        "z_index": z_index,
                        "path": slice_dirs[slice_id],
                    }
                )
                seen.add(slice_id)

    for slice_id, slice_dir in slice_dirs.items():
        if slice_id in seen:
            continue
        case_id, z_index = parse_slice_id(slice_id)
        records.append(
            {
                "slice_id": slice_id,
                "case_id": case_id,
                "z_index": z_index,
                "path": slice_dir,
            }
        )

    return sorted(records, key=lambda item: (item["case_id"], item["z_index"], item["slice_id"]))


def evaluate_case(
    case_id: str,
    records: list[dict],
    *,
    sample_dir: Path,
    num_classes: int,
    image_channels: int,
    num_ensemble: int,
    threshold: float,
) -> dict:
    targets = []
    sample_scores = [[] for _ in range(num_ensemble)]

    for record in sorted(records, key=lambda item: (item["z_index"], item["slice_id"])):
        slice_id = record["slice_id"]
        targets.append(load_target(record["path"], num_classes))
        for idx in range(num_ensemble):
            sample_path = sample_dir / f"{slice_id}_output{idx}"
            if not sample_path.exists():
                raise FileNotFoundError(f"Missing sample: {sample_path}")
            sample_scores[idx].append(
                load_prediction_scores(
                    sample_path,
                    num_classes=num_classes,
                    image_channels=image_channels,
                )
            )

    target_volume = torch.stack(targets, dim=0)
    ensemble_scores = torch.stack(
        [torch.stack(scores, dim=0) for scores in sample_scores],
        dim=0,
    ).mean(dim=0)
    pred_volume = torch.stack(
        [prediction_from_scores(ensemble_scores[z_index], threshold) for z_index in range(ensemble_scores.shape[0])],
        dim=0,
    )

    row = {
        "case_id": case_id,
        "num_slices": len(records),
    }
    case_dice = []
    case_iou = []
    for class_id in range(1, num_classes):
        label = CLASS_NAMES.get(class_id, f"class{class_id}")
        dsc, iou, pred_voxels, gt_voxels = dice_iou_for_class(
            pred_volume,
            target_volume,
            class_id,
        )
        dsc = clean_float(dsc)
        iou = clean_float(iou)
        row[f"{label}_dice"] = dsc
        row[f"{label}_iou"] = iou
        row[f"{label}_pred_voxels"] = pred_voxels
        row[f"{label}_gt_voxels"] = gt_voxels
        case_dice.append(dsc)
        case_iou.append(iou)

    row["mean_foreground_dice"] = mean(case_dice)
    row["mean_foreground_iou"] = mean(case_iou)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Preprocessed ACDC split directory.")
    parser.add_argument("--sample_dir", required=True, help="Directory with sampled masks.")
    parser.add_argument("--manifest_csv", default="", help="Optional manifest.csv path.")
    parser.add_argument("--num_ensemble", type=int, default=5)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--image_channels", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--per_case_csv",
        default="",
        help="Per-volume output CSV. Defaults to <sample_dir>/acdc_metrics_per_case.csv.",
    )
    parser.add_argument(
        "--output_csv",
        default="",
        help="Backward-compatible alias for --per_case_csv.",
    )
    parser.add_argument(
        "--summary_json",
        default="",
        help="Summary JSON path. Defaults to <sample_dir>/acdc_metrics_summary.json.",
    )
    parser.add_argument(
        "--summary_csv",
        default="",
        help="Summary CSV path. Defaults to <sample_dir>/acdc_metrics_summary.csv.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    sample_dir = Path(args.sample_dir).expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Missing data_dir: {data_dir}")
    if not sample_dir.is_dir():
        raise FileNotFoundError(f"Missing sample_dir: {sample_dir}")

    manifest_path = find_manifest(data_dir, args.manifest_csv)
    records = load_slice_records(data_dir, manifest_path)
    grouped = defaultdict(list)
    for record in records:
        grouped[record["case_id"]].append(record)

    rows = []
    for case_id in sorted(grouped):
        rows.append(
            evaluate_case(
                case_id,
                grouped[case_id],
                sample_dir=sample_dir,
                num_classes=args.num_classes,
                image_channels=args.image_channels,
                num_ensemble=args.num_ensemble,
                threshold=args.threshold,
            )
        )

    per_case_csv = (
        Path(args.per_case_csv)
        if args.per_case_csv
        else Path(args.output_csv)
        if args.output_csv
        else sample_dir / "acdc_metrics_per_case.csv"
    )
    summary_json = (
        Path(args.summary_json)
        if args.summary_json
        else sample_dir / "acdc_metrics_summary.json"
    )
    summary_csv = (
        Path(args.summary_csv)
        if args.summary_csv
        else sample_dir / "acdc_metrics_summary.csv"
    )

    fieldnames = list(rows[0].keys())
    write_csv(per_case_csv, rows, fieldnames)

    summary = {
        "data_dir": str(data_dir),
        "sample_dir": str(sample_dir),
        "manifest_csv": str(manifest_path) if manifest_path else "",
        "num_cases": len(rows),
        "num_slices": len(records),
        "num_ensemble": args.num_ensemble,
        "num_classes": args.num_classes,
        "image_channels": args.image_channels,
        "threshold": args.threshold,
        "mean_foreground_dice": mean([row["mean_foreground_dice"] for row in rows]),
        "mean_foreground_iou": mean([row["mean_foreground_iou"] for row in rows]),
        "mean_foreground_dice_std": std([row["mean_foreground_dice"] for row in rows]),
        "mean_foreground_iou_std": std([row["mean_foreground_iou"] for row in rows]),
    }
    for class_id in range(1, args.num_classes):
        label = CLASS_NAMES.get(class_id, f"class{class_id}")
        dice_values = [row[f"{label}_dice"] for row in rows]
        iou_values = [row[f"{label}_iou"] for row in rows]
        summary[f"{label}_dice"] = mean(dice_values)
        summary[f"{label}_iou"] = mean(iou_values)
        summary[f"{label}_dice_std"] = std(dice_values)
        summary[f"{label}_iou_std"] = std(iou_values)
        summary[f"{label}_nonempty_gt_cases"] = sum(
            value is not None for value in dice_values
        )

    write_csv(summary_csv, [summary], list(summary.keys()))
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
