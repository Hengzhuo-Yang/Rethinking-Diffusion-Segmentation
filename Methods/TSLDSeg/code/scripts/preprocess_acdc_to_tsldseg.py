"""Preprocess raw ACDC NIfTI frames into the fixed 70/10/20 TSLDSeg splits.

Release modification derived from the locally validated preprocessing lineage;
see MODIFICATIONS.md, DATA_SPLITS.md, and THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image


CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_ROOT = CODE_ROOT / "manifests" / "acdc"
SPLIT_DIRS = {"train": "training", "val": "validation", "test": "testing"}
EXPECTED_PATIENT_COUNTS = {"train": 70, "val": 10, "test": 20}
EXPECTED_SLICE_COUNTS = {"train": 1304, "val": 182, "test": 416}
CLASS_NAMES = {
    0: "background",
    1: "right_ventricle",
    2: "myocardium",
    3: "left_ventricle",
}


def _resampling():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.BILINEAR, Image.Resampling.NEAREST
    return Image.BILINEAR, Image.NEAREST


BILINEAR, NEAREST = _resampling()


def find_training_root(raw_root: str | Path) -> Path:
    root = Path(raw_root).expanduser().resolve()
    candidates = [root, root / "database", root / "ACDC" / "database"]
    for candidate in candidates:
        training = candidate / "training"
        if training.is_dir() and any(training.glob("patient*/Info.cfg")):
            return training
    raise FileNotFoundError(f"Could not find ACDC database/training under {root}")


def read_fixed_splits(manifest_root: str | Path) -> dict[str, list[str]]:
    root = Path(manifest_root).expanduser().resolve()
    result = {}
    for split, expected_count in EXPECTED_PATIENT_COUNTS.items():
        path = root / f"{split}.txt"
        identifiers = path.read_text(encoding="utf-8").splitlines()
        if len(identifiers) != expected_count or len(identifiers) != len(set(identifiers)):
            raise ValueError(
                f"{path} requires {expected_count} unique patient IDs, found {len(identifiers)}"
            )
        if any(not identifier.startswith("patient") or len(identifier) != 10 for identifier in identifiers):
            raise ValueError(f"Invalid ACDC patient ID in {path}")
        result[split] = identifiers
    owners = [identifier for identifiers in result.values() for identifier in identifiers]
    expected = {f"patient{number:03d}" for number in range(1, 101)}
    if len(owners) != len(set(owners)) or set(owners) != expected:
        raise ValueError("ACDC manifests must partition patient001 through patient100 without overlap")
    return result


def normalize_frame(volume: np.ndarray, lower: float, upper: float) -> np.ndarray:
    volume = np.asarray(volume, dtype=np.float32)
    finite = volume[np.isfinite(volume)]
    foreground = finite[finite > 0]
    reference = foreground if foreground.size else finite
    if reference.size == 0:
        return np.zeros_like(volume, dtype=np.uint8)
    low_value, high_value = np.percentile(reference, [lower, upper])
    if high_value <= low_value:
        return np.zeros_like(volume, dtype=np.uint8)
    volume = np.clip(volume, low_value, high_value)
    volume = (volume - low_value) / (high_value - low_value)
    return np.round(volume * 255.0).astype(np.uint8)


def resize(array: np.ndarray, size: int, resample: int) -> np.ndarray:
    image = Image.fromarray(array.astype(np.uint8))
    return np.asarray(image.resize((size, size), resample=resample), dtype=np.uint8)


def frame_paths(patient_dir: Path) -> list[Path]:
    paths = [
        path
        for path in sorted(patient_dir.glob(f"{patient_dir.name}_frame*.nii.gz"))
        if not path.name.endswith("_gt.nii.gz")
    ]
    if not paths:
        raise FileNotFoundError(f"No labelled ACDC frames found in {patient_dir}")
    return paths


def frame_id(path: Path) -> str:
    return path.name[:-7] if path.name.endswith(".nii.gz") else path.stem


def foreground_rgb(label: np.ndarray) -> np.ndarray:
    mask = np.zeros((*label.shape, 3), dtype=np.uint8)
    for class_id in range(1, 4):
        mask[..., class_id - 1] = np.where(label == class_id, 255, 0).astype(np.uint8)
    return mask


def preprocess_frame(
    image_path: Path,
    label_path: Path,
    *,
    logical_split: str,
    output_root: Path,
    output_size: int,
    clip_lower: float,
    clip_upper: float,
) -> list[dict[str, object]]:
    image = nib.load(str(image_path)).get_fdata()
    label = np.rint(nib.load(str(label_path)).get_fdata()).astype(np.uint8)
    if image.shape != label.shape:
        raise ValueError(f"Shape mismatch for {image_path.name}: {image.shape} vs {label.shape}")
    if label.max() >= 4:
        raise ValueError(f"Unexpected ACDC label {int(label.max())} in {label_path}")
    image = normalize_frame(image, clip_lower, clip_upper)
    physical_split = SPLIT_DIRS[logical_split]
    image_dir = output_root / "ACDC" / physical_split / "images"
    mask_dir = output_root / "ACDC" / physical_split / "masks"
    label_dir = output_root / "ACDC" / physical_split / "label_maps"
    for directory in (image_dir, mask_dir, label_dir):
        directory.mkdir(parents=True, exist_ok=True)

    identifier = frame_id(image_path)
    rows = []
    for z_index in range(image.shape[2]):
        sample_id = f"{identifier}_z{z_index:03d}"
        image_slice = resize(image[:, :, z_index], output_size, BILINEAR)
        image_rgb = np.repeat(image_slice[:, :, None], 3, axis=2)
        label_slice = resize(label[:, :, z_index], output_size, NEAREST)
        mask_rgb = foreground_rgb(label_slice)
        image_out = image_dir / f"{sample_id}.png"
        mask_out = mask_dir / f"{sample_id}.png"
        label_out = label_dir / f"{sample_id}.png"
        Image.fromarray(image_rgb, mode="RGB").save(image_out)
        Image.fromarray(mask_rgb, mode="RGB").save(mask_out)
        Image.fromarray(label_slice, mode="L").save(label_out)
        class_pixels = {
            f"class{class_id}_pixels": int((label_slice == class_id).sum())
            for class_id in range(1, 4)
        }
        rows.append(
            {
                "split": logical_split,
                "patient_id": image_path.parent.name,
                "frame_id": identifier,
                "z_index": z_index,
                "sample_id": sample_id,
                "image": image_out.relative_to(output_root).as_posix(),
                "mask": mask_out.relative_to(output_root).as_posix(),
                "label_map": label_out.relative_to(output_root).as_posix(),
                "empty_foreground": int(sum(class_pixels.values()) == 0),
                **class_pixels,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--manifest_root", default=str(DEFAULT_MANIFEST_ROOT))
    parser.add_argument("--output_size", type=int, default=256)
    parser.add_argument("--clip_lower", type=float, default=1.0)
    parser.add_argument("--clip_upper", type=float, default=99.0)
    args = parser.parse_args()

    training_root = find_training_root(args.raw_root)
    fixed_splits = read_fixed_splits(args.manifest_root)
    output_root = Path(args.output_root).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output root must be absent or empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for logical_split, patient_ids in fixed_splits.items():
        for index, patient_id in enumerate(patient_ids, start=1):
            patient_dir = training_root / patient_id
            if not patient_dir.is_dir():
                raise FileNotFoundError(f"Missing ACDC patient directory: {patient_dir}")
            for image_path in frame_paths(patient_dir):
                label_path = image_path.with_name(f"{frame_id(image_path)}_gt.nii.gz")
                if not label_path.is_file():
                    raise FileNotFoundError(f"Missing ACDC ground truth: {label_path}")
                rows.extend(
                    preprocess_frame(
                        image_path,
                        label_path,
                        logical_split=logical_split,
                        output_root=output_root,
                        output_size=args.output_size,
                        clip_lower=args.clip_lower,
                        clip_upper=args.clip_upper,
                    )
                )
            print(f"{logical_split}: processed {index}/{len(patient_ids)} {patient_id}")

    counts = {
        split: sum(row["split"] == split for row in rows)
        for split in SPLIT_DIRS
    }
    if counts != EXPECTED_SLICE_COUNTS:
        raise ValueError(
            f"Generated ACDC slice counts changed: expected {EXPECTED_SLICE_COUNTS}, found {counts}"
        )
    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "dataset": "ACDC",
        "task": "four-class 2D segmentation including background",
        "class_names": CLASS_NAMES,
        "output_size": args.output_size,
        "intensity_clip_percentiles": [args.clip_lower, args.clip_upper],
        "patient_counts": EXPECTED_PATIENT_COUNTS,
        "slice_counts": counts,
        "manifest": manifest_path.name,
        "fixed_id_manifests": "release code/manifests/acdc/{train,val,test}.txt",
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
