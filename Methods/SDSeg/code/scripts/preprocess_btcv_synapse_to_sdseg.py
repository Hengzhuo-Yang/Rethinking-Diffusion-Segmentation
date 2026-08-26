"""Preprocess BTCV NIfTI volumes into fixed, case-level SDSeg splits.

The release split is train=18 cases, validation=2 cases, and test=10 cases.
Images are stored as 16-bit PNGs and class-index masks as grayscale PNGs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
from PIL import Image
from tqdm import tqdm


VALIDATION_VOLUME_IDS = {"0001", "0008"}
TEST_VOLUME_IDS = {
    "0002", "0003", "0004", "0022", "0025", "0029", "0032", "0035", "0036", "0038",
}


def find_training_root(raw_root: str | Path) -> Path:
    root = Path(raw_root).expanduser().resolve()
    candidates = [
        root,
        root / "RawData" / "Training",
        root / "raw" / "RawData" / "Training",
        root / "BTCV" / "raw" / "RawData" / "Training",
    ]
    for candidate in candidates:
        if (candidate / "img").is_dir() and (candidate / "label").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find BTCV RawData/Training/img and label directories under "
        f"{root}"
    )


def volume_id(path: Path) -> str:
    match = re.search(r"(\d+)", path.name)
    if not match:
        raise ValueError(f"Could not parse BTCV volume id from {path.name}")
    return match.group(1)


def label_path_for(image_path: Path, label_dir: Path) -> Path:
    return label_dir / image_path.name.replace("img", "label", 1)


def label_range_detector(gts: np.ndarray) -> tuple[int, int, int, int]:
    x_list = []
    y_list = []
    for slice_idx in range(gts.shape[0]):
        if gts[slice_idx, :, :].sum() > 0:
            x_list.append(slice_idx)
    for slice_idx in range(gts.shape[1]):
        if gts[:, slice_idx, :].sum() > 0:
            y_list.append(slice_idx)
    if not x_list or not y_list:
        return 0, 0, 0, 0
    return x_list[0], gts.shape[0] - x_list[-1], y_list[0], gts.shape[1] - y_list[-1]


def split_for(image_path: Path) -> str:
    vid = volume_id(image_path)
    if vid in VALIDATION_VOLUME_IDS:
        return "validation"
    if vid in TEST_VOLUME_IDS:
        return "test"
    return "train"


def encode_image_u16(slice_img: np.ndarray) -> np.ndarray:
    slice_img = np.clip(slice_img.astype(np.float32), -175.0, 250.0)
    return np.round(((slice_img + 175.0) / 425.0) * 65535.0).astype(np.uint16)


def clean_output(output_root: Path) -> None:
    for name in ["train", "validation", "test"]:
        if not name:
            continue
        path = output_root / name
        if path.exists():
            shutil.rmtree(path)
    for name in ["manifest.csv", "summary.json"]:
        path = output_root / name
        if path.exists():
            path.unlink()


def preprocess_volume(
    *,
    image_path: Path,
    label_path: Path,
    output_root: Path,
    output_size: int,
) -> list[dict[str, str | int]]:
    image = nib.load(str(image_path)).get_fdata()
    label = nib.load(str(label_path)).get_fdata()
    if image.shape != label.shape:
        raise ValueError(f"Shape mismatch for {image_path.name}: {image.shape} vs {label.shape}")

    split = split_for(image_path)
    image_dir = output_root / split / "images"
    mask_dir = output_root / split / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    min_edge = min(label_range_detector(label))
    pixels_count = np.zeros(14, dtype=np.int64)
    rows = []
    vid = volume_id(image_path)
    for slice_idx in range(image.shape[2]):
        slice_img = image[:, :, slice_idx].astype(np.float32)
        slice_img = cv2.resize(slice_img, (output_size, output_size))
        image_u16 = encode_image_u16(slice_img)

        slice_label = label[:, :, slice_idx].astype(np.uint8)
        slice_label = cv2.resize(
            slice_label,
            (output_size, output_size),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.uint8)

        for class_id in range(14):
            pixels_count[class_id] += int((slice_label == class_id).sum())

        slice_id = f"vol{vid}_{slice_idx:03d}"
        image_out = image_dir / f"{slice_id}.png"
        mask_out = mask_dir / f"{slice_id}.png"
        Image.fromarray(image_u16, mode="I;16").save(image_out)
        Image.fromarray(slice_label, mode="L").save(mask_out)

        rows.append(
            {
                "split": split,
                "volume": vid,
                "slice": slice_idx,
                "slice_id": slice_id,
                "image": image_out.relative_to(output_root).as_posix(),
                "mask": mask_out.relative_to(output_root).as_posix(),
                "foreground_pixels": int((slice_label > 0).sum()),
                "min_label_edge": int(min_edge),
            }
        )
    return rows



def preprocess(args: argparse.Namespace) -> None:
    training_root = find_training_root(args.raw_root)
    image_dir = training_root / "img"
    label_dir = training_root / "label"
    output_root = Path(args.output_root).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output root is not empty, pass --overwrite to replace cache files: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        clean_output(output_root)

    image_paths = sorted(image_dir.glob("img*.nii.gz"))
    if args.max_volumes > 0:
        image_paths = image_paths[: args.max_volumes]
    if not image_paths:
        raise RuntimeError(f"No BTCV images found under {image_dir}")

    rows = []
    for image_path in tqdm(image_paths, desc="processing BTCV volumes"):
        label_path = label_path_for(image_path, label_dir)
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing BTCV label for {image_path}: {label_path}")
        rows.extend(
            preprocess_volume(
                image_path=image_path,
                label_path=label_path,
                output_root=output_root,
                output_size=args.output_size,
            )
        )


    fieldnames = [
        "split",
        "volume",
        "slice",
        "slice_id",
        "image",
        "mask",
        "foreground_pixels",
        "min_label_edge",
    ]
    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    split_counts = {}
    for split in ("train", "validation", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        split_counts[split] = {
            "slices": len(split_rows),
            "volumes": len(set(row["volume"] for row in split_rows)),
            "empty_foreground_slices": sum(int(row["foreground_pixels"]) == 0 for row in split_rows),
        }
    expected = {
        "train": {"slices": 2211, "volumes": 18},
        "validation": {"slices": 295, "volumes": 2},
        "test": {"slices": 1273, "volumes": 10},
    }
    if args.max_volumes == 0:
        for split, counts in expected.items():
            for key, value in counts.items():
                if split_counts[split][key] != value:
                    raise ValueError(
                        f"BTCV {split} {key} mismatch: {split_counts[split][key]} != {value}"
                    )
    summary = {
        "dataset": "BTCV",
        "output_size": args.output_size,
        "storage": "16-bit PNG images plus grayscale class-index PNG masks",
        "split_policy": "Fixed case-level 18/2/10 train/validation/test split.",
        "preprocessing": {
            "ct_window": [-175.0, 250.0],
            "image_resize": "cv2.resize default interpolation",
            "label_resize": "cv2.INTER_NEAREST",
            "loader_image_formula": "(((hu + 125) / 400) * 2) - 1",
            "label_policy": "preserve class-index labels 0..13; loader maps labels as configured",
        },
        "split_counts": split_counts,
        "total_manifest_rows": len(rows),
        "manifest": "manifest.csv",
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", "--raw_root", dest="raw_root", required=True,
                        help="BTCV RawData/Training or a parent containing it.")
    parser.add_argument("--output-root", "--output_root", dest="output_root", required=True,
                        help="Empty output directory for the generated BTCV cache.")
    parser.add_argument("--output-size", "--output_size", dest="output_size", type=int, default=256)
    parser.add_argument("--max-volumes", "--max_volumes", dest="max_volumes", type=int, default=0,
                        help="Smoke-only volume limit; 0 processes all volumes.")
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    preprocess(build_argparser().parse_args())
