"""Preprocess official ACDC NIfTI frames into fixed SDSeg PNG splits.

Raw data remains outside the repository. Labels remain class indices 0..3.
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
    "training": "train",
    "validation": "validation",
    "testing": "test",
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
    patients: list[str] = []
    seen: set[str] = set()
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
        "training": read_patient_list(root / "train_patients.txt"),
        "validation": read_patient_list(root / "validation_patients.txt"),
        "testing": read_patient_list(root / "test_patients.txt"),
    }
    all_patients = [patient for patients in split.values() for patient in patients]
    if len(all_patients) != len(set(all_patients)):
        raise ValueError(f"Duplicate patients found in split root: {root}")
    return split


def normalize_mri_frame(volume: np.ndarray, lower: float, upper: float) -> np.ndarray:
    volume = np.asarray(volume, dtype=np.float32)
    finite = volume[np.isfinite(volume)]
    foreground = finite[finite > 0]
    reference = foreground if foreground.size else finite
    if reference.size == 0:
        return np.zeros_like(volume, dtype=np.float32)
    lo, hi = np.percentile(reference, [lower, upper])
    if hi <= lo:
        return np.zeros_like(volume, dtype=np.float32)
    volume = np.clip(volume, lo, hi)
    volume = (volume - lo) / (hi - lo)
    return (volume * 2.0 - 1.0).astype(np.float32)


def resize_image_to_uint8(slice_2d: np.ndarray, output_size: int) -> np.ndarray:
    uint8 = np.round(np.clip((slice_2d + 1.0) * 127.5, 0, 255)).astype(np.uint8)
    image = Image.fromarray(uint8)
    image = image.resize((output_size, output_size), resample=BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def resize_label(slice_2d: np.ndarray, output_size: int) -> np.ndarray:
    image = Image.fromarray(slice_2d.astype(np.uint8))
    image = image.resize((output_size, output_size), resample=NEAREST)
    return np.asarray(image, dtype=np.uint8)


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
            f"Shape mismatch for {image_path.name}: "
            f"{image_volume.shape} vs {label_volume.shape}"
        )

    labels = np.rint(label_volume).astype(np.uint8)
    if labels.max() >= num_classes:
        raise ValueError(
            f"Unexpected ACDC label value {int(labels.max())} in {label_path}; "
            f"num_classes={num_classes}"
        )

    image_volume = normalize_mri_frame(image_volume, clip_lower, clip_upper)
    fid = frame_id(image_path)
    split_dir = SPLIT_DIR_NAMES[split]
    image_dir = output_root / split_dir / "images"
    mask_dir = output_root / split_dir / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | int]] = []
    for z in range(image_volume.shape[2]):
        slice_id = f"{fid}_z{z:03d}"
        image_slice = resize_image_to_uint8(image_volume[:, :, z], output_size)
        image_rgb = np.repeat(image_slice[:, :, None], 3, axis=2).astype(np.uint8)
        label_slice = resize_label(labels[:, :, z], output_size)

        image_out = image_dir / f"{slice_id}.png"
        mask_out = mask_dir / f"{slice_id}.png"
        Image.fromarray(image_rgb, mode="RGB").save(image_out)
        Image.fromarray(label_slice.astype(np.uint8), mode="L").save(mask_out)

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
                "image": image_out.relative_to(output_root).as_posix(),
                "mask": mask_out.relative_to(output_root).as_posix(),
                "empty_foreground": int(sum(class_pixels.values()) == 0),
                **class_pixels,
            }
        )

    return rows


def summarize_rows(rows: list[dict[str, str | int]], num_classes: int) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
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
    if int(args.num_classes) != 4:
        raise ValueError("ACDC preprocessing expects num_classes=4.")

    training_root = find_acdc_training_root(args.raw_root)
    split = load_split(args.split_root)
    output_root = Path(args.output_root).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output root is not empty, pass --overwrite to replace: {output_root}")
        for name in ("train", "validation", "test"):
            path = output_root / name
            if path.exists():
                shutil.rmtree(path)
        for name in ("manifest.csv", "summary.json"):
            path = output_root / name
            if path.exists():
                path.unlink()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.max_patients_per_split > 0:
        split = {
            key: patients[: args.max_patients_per_split]
            for key, patients in split.items()
        }

    rows: list[dict[str, str | int]] = []
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
        "empty_foreground",
    ] + [f"class{class_id}_pixels" for class_id in range(1, args.num_classes)]
    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
                "output_size": args.output_size,
        "intensity_clip_percentiles": [args.clip_lower, args.clip_upper],
        "normalization": "per labelled ED/ES frame, nonzero percentile clip; PNG loader scales images to [-1, 1]",
        "num_classes": args.num_classes,
        "class_names": CLASS_NAMES,
        "patient_counts": patient_counts,
        "slice_counts": slice_counts,
        "split_summaries": summarize_rows(rows, args.num_classes),
        "total_slices": len(rows),
        "manifest": "manifest.csv",
        "label_policy": (
            "ACDC labels are preserved as grayscale class-index masks 0..3. "
            "The SDSeg loader expands them to the 3-channel class-conditional representation at load time."
        ),
        "split_note": "Uses the bundled fixed 70/10/20 patient-ID manifests.",
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    manifest_default = Path(__file__).resolve().parents[1] / "manifests" / "acdc"
    parser.add_argument("--raw-root", "--raw_root", dest="raw_root", required=True,
                        help="Official ACDC database or a parent containing database/training.")
    parser.add_argument("--split-root", "--split_root", dest="split_root", default=str(manifest_default),
                        help="Directory containing the bundled patient-ID manifests.")
    parser.add_argument("--output-root", "--output_root", dest="output_root", required=True)
    parser.add_argument("--output-size", "--output_size", dest="output_size", type=int, default=256)
    parser.add_argument("--clip-lower", "--clip_lower", dest="clip_lower", type=float, default=1.0)
    parser.add_argument("--clip-upper", "--clip_upper", dest="clip_upper", type=float, default=99.0)
    parser.add_argument("--num-classes", "--num_classes", dest="num_classes", type=int, default=4)
    parser.add_argument("--max-patients-per-split", "--max_patients_per_split",
                        dest="max_patients_per_split", type=int, default=0,
                        help="Smoke-only per-split limit; 0 processes all patients.")
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    preprocess(build_argparser().parse_args())
