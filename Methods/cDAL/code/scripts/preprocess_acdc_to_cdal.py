"""Preprocess raw ACDC NIfTI frames into a cDAL 2D PNG cache.

Masks remain multi-class through three foreground channels:
  R = right ventricle, G = myocardium, B = left ventricle.
Background is represented by all-zero foreground channels.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image


CLASS_NAMES = {
    0: "background",
    1: "right_ventricle",
    2: "myocardium",
    3: "left_ventricle",
}

SPLIT_DIR_NAMES = {
    "training": "training",
    "validation": "validation",
    "testing": "testing",
}


def _resample_constants():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.BILINEAR, Image.Resampling.NEAREST
    return Image.BILINEAR, Image.NEAREST


BILINEAR, NEAREST = _resample_constants()


def find_acdc_training_root(raw_root: str | Path) -> Path:
    root = Path(raw_root).expanduser().resolve()
    candidates = [
        root,
        root / "database",
        root / "ACDC" / "database",
        root / "data" / "ACDC" / "database",
    ]
    for candidate in candidates:
        training = candidate / "training"
        if training.is_dir() and any(training.glob("patient*/Info.cfg")):
            return training
    raise FileNotFoundError(f"Could not find ACDC database/training under {root}")


def read_patient_list(path: Path) -> list[str]:
    patients = []
    seen = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        patient = line.split("_frame", 1)[0]
        if patient not in seen:
            patients.append(patient)
            seen.add(patient)
    return patients


def load_split(split_root: str | Path) -> dict[str, list[str]]:
    root = Path(split_root).expanduser().resolve()
    split = {
        "training": read_patient_list(root / "training_subjects.txt"),
        "validation": read_patient_list(root / "validation_subjects.txt"),
        "testing": read_patient_list(root / "test_subjects.txt"),
    }
    all_patients = [patient for patients in split.values() for patient in patients]
    if len(all_patients) != len(set(all_patients)):
        raise ValueError(f"Duplicate patients found in split root: {root}")
    return split


def normalize_mri_frame_to_uint8(volume: np.ndarray, lower: float, upper: float) -> np.ndarray:
    volume = np.asarray(volume, dtype=np.float32)
    finite = volume[np.isfinite(volume)]
    foreground = finite[finite > 0]
    reference = foreground if foreground.size else finite
    if reference.size == 0:
        return np.zeros_like(volume, dtype=np.uint8)
    lo, hi = np.percentile(reference, [lower, upper])
    if hi <= lo:
        return np.zeros_like(volume, dtype=np.uint8)
    volume = np.clip(volume, lo, hi)
    volume = (volume - lo) / (hi - lo)
    return np.round(volume * 255.0).astype(np.uint8)


def resize_image(slice_2d: np.ndarray, output_size: int) -> np.ndarray:
    image = Image.fromarray(slice_2d.astype(np.uint8))
    image = image.resize((output_size, output_size), resample=BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def resize_label(slice_2d: np.ndarray, output_size: int) -> np.ndarray:
    image = Image.fromarray(slice_2d.astype(np.uint8))
    image = image.resize((output_size, output_size), resample=NEAREST)
    return np.asarray(image, dtype=np.uint8)


def foreground_rgb_from_label(label: np.ndarray, num_classes: int) -> np.ndarray:
    if num_classes != 4:
        raise ValueError("ACDC cDAL foreground RGB encoding expects num_classes=4")
    label = np.asarray(label, dtype=np.uint8)
    mask_rgb = np.zeros((*label.shape, 3), dtype=np.uint8)
    for class_id in range(1, num_classes):
        mask_rgb[..., class_id - 1] = np.where(label == class_id, 255, 0).astype(np.uint8)
    return mask_rgb


def frame_image_paths(patient_dir: Path) -> list[Path]:
    paths = []
    for path in sorted(patient_dir.glob(f"{patient_dir.name}_frame*.nii.gz")):
        if path.name.endswith("_gt.nii.gz"):
            continue
        paths.append(path)
    if not paths:
        raise FileNotFoundError(f"No labelled ACDC frame images found in {patient_dir}")
    return paths


def frame_id(path: Path) -> str:
    if path.name.endswith(".nii.gz"):
        return path.name[:-7]
    return path.stem


def preprocess_frame(
    *,
    image_path: Path,
    label_path: Path,
    split: str,
    output_root: Path,
    output_size: int,
    clip_lower: float,
    clip_upper: float,
    num_classes: int,
) -> list[dict[str, str | int]]:
    image_volume = nib.load(str(image_path)).get_fdata()
    label_volume = nib.load(str(label_path)).get_fdata()
    if image_volume.shape != label_volume.shape:
        raise ValueError(
            f"Shape mismatch for {image_path.name}: {image_volume.shape} vs {label_volume.shape}"
        )

    labels = np.rint(label_volume).astype(np.uint8)
    if labels.max() >= num_classes:
        raise ValueError(
            f"Unexpected ACDC label value {int(labels.max())} in {label_path}; num_classes={num_classes}"
        )
    image_volume = normalize_mri_frame_to_uint8(image_volume, clip_lower, clip_upper)
    fid = frame_id(image_path)
    split_dir = SPLIT_DIR_NAMES[split]
    image_dir = output_root / "ACDC" / split_dir / "images"
    mask_dir = output_root / "ACDC" / split_dir / "masks"
    label_map_dir = output_root / "ACDC" / split_dir / "label_maps"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    label_map_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for z in range(image_volume.shape[2]):
        slice_id = f"{fid}_z{z:03d}"
        image_slice = resize_image(image_volume[:, :, z], output_size)
        image_rgb = np.repeat(image_slice[:, :, None], 3, axis=2)
        label_slice = resize_label(labels[:, :, z], output_size)
        mask_rgb = foreground_rgb_from_label(label_slice, num_classes)

        image_out = image_dir / f"{slice_id}.png"
        mask_out = mask_dir / f"{slice_id}.png"
        label_map_out = label_map_dir / f"{slice_id}.png"
        Image.fromarray(image_rgb, mode="RGB").save(image_out)
        Image.fromarray(mask_rgb, mode="RGB").save(mask_out)
        Image.fromarray(label_slice.astype(np.uint8), mode="L").save(label_map_out)

        class_pixels = {
            f"class{class_id}_pixels": int((label_slice == class_id).sum())
            for class_id in range(1, num_classes)
        }
        rows.append(
            {
                "split": split_dir,
                "patient_id": image_path.parent.name,
                "frame_id": fid,
                "z_index": z,
                "slice_id": slice_id,
                "image": str(image_out),
                "mask": str(mask_out),
                "label_map": str(label_map_out),
                "empty_foreground": int(sum(class_pixels.values()) == 0),
                **class_pixels,
            }
        )
    return rows


def summarize_rows(rows: list[dict[str, str | int]], num_classes: int) -> dict[str, dict[str, int]]:
    summary = {}
    for split in SPLIT_DIR_NAMES.values():
        split_rows = [row for row in rows if row["split"] == split]
        summary[split] = {
            "slices": len(split_rows),
            "empty_foreground_slices": sum(int(row["empty_foreground"]) for row in split_rows),
        }
        for class_id in range(1, num_classes):
            key = f"class{class_id}_pixels"
            summary[split][key] = sum(int(row[key]) for row in split_rows)
            summary[split][f"class{class_id}_nonempty_slices"] = sum(
                int(row[key]) > 0 for row in split_rows
            )
    return summary


def preprocess(args: argparse.Namespace) -> None:
    training_root = find_acdc_training_root(args.raw_root)
    split = load_split(args.split_root)
    output_root = Path(args.output_root).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output root is not empty, pass --overwrite to replace: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.max_patients_per_split > 0:
        split = {key: patients[: args.max_patients_per_split] for key, patients in split.items()}

    rows = []
    patient_counts = {key: len(value) for key, value in split.items()}
    slice_counts = {SPLIT_DIR_NAMES[key]: 0 for key in split}
    for split_name, patient_ids in split.items():
        for patient_index, patient_id in enumerate(patient_ids, 1):
            patient_dir = training_root / patient_id
            if not patient_dir.is_dir():
                raise FileNotFoundError(f"Missing ACDC patient directory: {patient_dir}")
            for image_path in frame_image_paths(patient_dir):
                gt_path = image_path.with_name(f"{frame_id(image_path)}_gt.nii.gz")
                if not gt_path.exists():
                    raise FileNotFoundError(f"Missing ACDC GT for {image_path}: {gt_path}")
                frame_rows = preprocess_frame(
                    image_path=image_path,
                    label_path=gt_path,
                    split=split_name,
                    output_root=output_root,
                    output_size=args.output_size,
                    clip_lower=args.clip_lower,
                    clip_upper=args.clip_upper,
                    num_classes=args.num_classes,
                )
                rows.extend(frame_rows)
                slice_counts[SPLIT_DIR_NAMES[split_name]] += len(frame_rows)
            print(
                f"{SPLIT_DIR_NAMES[split_name]}: processed {patient_index}/{len(patient_ids)} "
                f"{patient_id} -> {slice_counts[SPLIT_DIR_NAMES[split_name]]} slices so far"
            )

    fieldnames = [
        "split",
        "patient_id",
        "frame_id",
        "z_index",
        "slice_id",
        "image",
        "mask",
        "label_map",
        "empty_foreground",
    ] + [f"class{class_id}_pixels" for class_id in range(1, args.num_classes)]
    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "raw_training_root": str(training_root),
        "split_root": str(Path(args.split_root).expanduser().resolve()),
        "output_root": str(output_root),
        "output_size": args.output_size,
        "intensity_clip_percentiles": [args.clip_lower, args.clip_upper],
        "normalization": "per labelled ED/ES frame, nonzero percentile clip, scaled to uint8 PNG",
        "num_classes": args.num_classes,
        "class_names": CLASS_NAMES,
        "patient_counts": patient_counts,
        "slice_counts": slice_counts,
        "split_summaries": summarize_rows(rows, args.num_classes),
        "total_slices": len(rows),
        "manifest": str(manifest_path),
        "label_policy": (
            "ACDC labels are preserved as foreground RGB channels: "
            "R=RV, G=myocardium, B=LV; all-zero foreground channels are background."
        ),
        "split_note": "Uses the fixed 70/10/20 subject manifests distributed with this release.",
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", default="raw/ACDC/database")
    parser.add_argument("--split_root", default="manifests/acdc")
    parser.add_argument("--output_root", default="data_preprocessed/acdc_mt_unet_cascade_cdal_png")
    parser.add_argument("--output_size", type=int, default=256)
    parser.add_argument("--clip_lower", type=float, default=1.0)
    parser.add_argument("--clip_upper", type=float, default=99.0)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--max_patients_per_split", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    preprocess(build_argparser().parse_args())
