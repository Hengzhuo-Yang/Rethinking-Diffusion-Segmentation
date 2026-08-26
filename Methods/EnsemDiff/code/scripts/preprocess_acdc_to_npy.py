"""Preprocess ACDC NIfTI frames using the repository's fixed subject split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

from fixed_split_manifests import (
    assert_exact_id_set,
    load_fixed_split_manifests,
    prepare_empty_output_directory,
)


CLASS_NAMES = {
    0: "background",
    1: "right_ventricle",
    2: "myocardium",
    3: "left_ventricle",
}
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST_ROOT = SCRIPT_DIR.parent / "manifests" / "acdc"
MANIFEST_FILENAMES = {
    "training": "train_patients.txt",
    "validation": "val_patients.txt",
    "testing": "test_patients.txt",
}
EXPECTED_PATIENT_COUNTS = {"training": 70, "validation": 10, "testing": 20}
EXPECTED_SLICE_COUNTS = {"training": 1304, "validation": 182, "testing": 416}


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


def load_split(manifest_root: str | Path) -> dict[str, list[str]]:
    return load_fixed_split_manifests(
        manifest_root,
        manifest_filenames=MANIFEST_FILENAMES,
        expected_counts=EXPECTED_PATIENT_COUNTS,
        id_pattern=r"patient\d{3}",
        dataset_name="ACDC",
    )


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
    return ((volume - lo) / (hi - lo) * 2.0 - 1.0).astype(np.float32)


def resize_image(slice_2d: np.ndarray, output_size: int) -> np.ndarray:
    image = Image.fromarray(slice_2d.astype(np.float32))
    image = image.resize((output_size, output_size), resample=BILINEAR)
    return np.asarray(image, dtype=np.float32)


def resize_label(slice_2d: np.ndarray, output_size: int) -> np.ndarray:
    image = Image.fromarray(slice_2d.astype(np.uint8))
    image = image.resize((output_size, output_size), resample=NEAREST)
    return np.asarray(image, dtype=np.uint8)


def one_hot(label: np.ndarray, num_classes: int) -> np.ndarray:
    label = np.asarray(label, dtype=np.uint8)
    label = np.clip(label, 0, num_classes - 1)
    return np.eye(num_classes, dtype=np.uint8)[label].transpose(2, 0, 1)


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


def validate_acdc_source(
    training_root: Path,
    split: dict[str, list[str]],
) -> dict[str, int]:
    expected_patients = [patient for patients in split.values() for patient in patients]
    actual_patients = {
        path.name
        for path in training_root.glob("patient*")
        if path.is_dir() and (path / "Info.cfg").is_file()
    }
    assert_exact_id_set(
        actual_patients,
        expected_patients,
        dataset_name="ACDC",
        split_name="all fixed partitions",
        source_name=str(training_root),
    )

    slice_counts = {split_name: 0 for split_name in split}
    for split_name, patient_ids in split.items():
        for patient_id in patient_ids:
            patient_dir = training_root / patient_id
            for image_path in frame_image_paths(patient_dir):
                gt_path = image_path.with_name(f"{frame_id(image_path)}_gt.nii.gz")
                if not gt_path.is_file():
                    raise FileNotFoundError(f"Missing ACDC GT for {image_path}: {gt_path}")
                image_shape = nib.load(str(image_path)).shape
                label_shape = nib.load(str(gt_path)).shape
                if image_shape != label_shape:
                    raise ValueError(
                        f"Shape mismatch for {image_path.name}: {image_shape} vs {label_shape}"
                    )
                if len(image_shape) != 3:
                    raise ValueError(
                        f"Expected a 3D labelled ACDC frame at {image_path}, got {image_shape}"
                    )
                slice_counts[split_name] += int(image_shape[2])

    if slice_counts != EXPECTED_SLICE_COUNTS:
        raise ValueError(
            "ACDC source slice counts do not match the fixed release split: "
            f"expected {EXPECTED_SLICE_COUNTS}, found {slice_counts}"
        )
    return slice_counts


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

    image_volume = normalize_mri_frame(image_volume, clip_lower, clip_upper)
    labels = np.rint(label_volume).astype(np.uint8)
    fid = frame_id(image_path)
    rows = []

    for z in range(image_volume.shape[2]):
        slice_id = f"{fid}_z{z:03d}"
        slice_dir = output_root / split / slice_id
        slice_dir.mkdir(parents=True, exist_ok=True)

        image_slice = resize_image(image_volume[:, :, z], output_size)
        label_slice = resize_label(labels[:, :, z], output_size)
        label_one_hot = one_hot(label_slice, num_classes)

        image_out = slice_dir / f"{slice_id}_image.npy"
        label_out = slice_dir / f"{slice_id}_seg.npy"
        label_map_out = slice_dir / f"{slice_id}_label.npy"
        np.save(image_out, image_slice.astype(np.float32))
        np.save(label_out, label_one_hot)
        np.save(label_map_out, label_slice.astype(np.uint8))

        class_pixels = {
            f"class{class_id}_pixels": int((label_slice == class_id).sum())
            for class_id in range(1, num_classes)
        }
        rows.append(
            {
                "split": split,
                "patient_id": image_path.parent.name,
                "frame_id": fid,
                "z_index": z,
                "slice_id": slice_id,
                "image": image_out.relative_to(output_root).as_posix(),
                "label": label_out.relative_to(output_root).as_posix(),
                "label_map": label_map_out.relative_to(output_root).as_posix(),
                **class_pixels,
            }
        )
    return rows


def preprocess(args: argparse.Namespace) -> None:
    training_root = find_acdc_training_root(args.raw_root)
    fixed_split = load_split(args.manifest_root)
    source_slice_counts = validate_acdc_source(training_root, fixed_split)
    output_root = prepare_empty_output_directory(
        args.output_root,
        dataset_name="ACDC",
    )

    split = fixed_split
    if args.max_patients_per_split > 0:
        split = {
            key: patients[: args.max_patients_per_split]
            for key, patients in fixed_split.items()
        }

    rows = []
    counts = {key: 0 for key in split}
    patient_counts = {key: len(value) for key, value in split.items()}

    for split_name, patient_ids in split.items():
        for patient_index, patient_id in enumerate(patient_ids, 1):
            patient_dir = training_root / patient_id
            frame_paths = frame_image_paths(patient_dir)
            for image_path in frame_paths:
                gt_path = image_path.with_name(f"{frame_id(image_path)}_gt.nii.gz")
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
                counts[split_name] += len(frame_rows)
            print(
                f"{split_name}: processed {patient_index}/{len(patient_ids)} "
                f"{patient_id} -> {counts[split_name]} slices so far"
            )

    fieldnames = [
        "split",
        "patient_id",
        "frame_id",
        "z_index",
        "slice_id",
        "image",
        "label",
        "label_map",
    ] + [f"class{class_id}_pixels" for class_id in range(1, args.num_classes)]
    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "dataset": "ACDC",
        "fixed_split_manifest_files": MANIFEST_FILENAMES,
        "fixed_patient_counts": EXPECTED_PATIENT_COUNTS,
        "fixed_slice_counts": EXPECTED_SLICE_COUNTS,
        "validated_source_slice_counts": source_slice_counts,
        "processed_patient_counts": patient_counts,
        "processed_slice_counts": counts,
        "processed_total_slices": len(rows),
        "output_size": args.output_size,
        "intensity_clip_percentiles": [args.clip_lower, args.clip_upper],
        "normalization": "per labelled ED/ES frame, nonzero percentile clip, scaled to [-1, 1]",
        "num_classes": args.num_classes,
        "class_names": CLASS_NAMES,
        "manifest": manifest_path.name,
        "split_note": (
            "Fixed MT-UNet/CASCADE subject split: 70 training patients, "
            "10 validation patients, and 20 testing patients."
        ),
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_root",
        "--raw-root",
        required=True,
        help="ACDC database path or a parent directory containing database/training.",
    )
    parser.add_argument(
        "--manifest_root",
        "--manifest-root",
        "--split_root",
        dest="manifest_root",
        default=str(DEFAULT_MANIFEST_ROOT),
        help="Fixed ID-only split directory (defaults to code/manifests/acdc).",
    )
    parser.add_argument(
        "--output_root",
        "--output-root",
        required=True,
        help="Destination for generated ACDC arrays and metadata.",
    )
    parser.add_argument("--output_size", "--output-size", type=int, default=224)
    parser.add_argument("--clip_lower", "--clip-lower", type=float, default=1.0)
    parser.add_argument("--clip_upper", "--clip-upper", type=float, default=99.0)
    parser.add_argument("--num_classes", "--num-classes", type=int, default=4)
    parser.add_argument(
        "--max_patients_per_split",
        "--max-patients-per-split",
        type=int,
        default=0,
        help="Optional smoke-test limit applied only after full split/source validation.",
    )
    return parser


if __name__ == "__main__":
    preprocess(build_argparser().parse_args())


