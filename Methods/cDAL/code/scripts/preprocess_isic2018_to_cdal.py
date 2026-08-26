"""Preprocess official ISIC2018 Task 1 zips into a cDAL 2D PNG cache.

The script preserves the official training, validation, and testing partitions.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image


SPLIT_ZIPS = {
    "training": {
        "image_zip": "ISIC2018_Task1-2_Training_Input.zip",
        "mask_zip": "ISIC2018_Task1_Training_GroundTruth.zip",
    },
    "validation": {
        "image_zip": "ISIC2018_Task1-2_Validation_Input.zip",
        "mask_zip": "ISIC2018_Task1_Validation_GroundTruth.zip",
    },
    "testing": {
        "image_zip": "ISIC2018_Task1-2_Test_Input.zip",
        "mask_zip": "ISIC2018_Task1_Test_GroundTruth.zip",
    },
}

CLASS_NAMES = {
    0: "background",
    1: "lesion",
}


def _resample_constants():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.BILINEAR, Image.Resampling.NEAREST
    return Image.BILINEAR, Image.NEAREST


BILINEAR, NEAREST = _resample_constants()


def image_id_from_member(member: str) -> str:
    stem = Path(member).stem
    return re.sub(r"[_-]segmentation$", "", stem, flags=re.IGNORECASE)


def collect_members(zip_path: Path, extensions: tuple[str, ...]) -> dict[str, str]:
    members: dict[str, str] = {}
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            if member.endswith("/"):
                continue
            if Path(member).suffix.lower() not in extensions:
                continue
            image_id = image_id_from_member(member)
            if image_id in members:
                raise ValueError(
                    f"Duplicate ISIC2018 id {image_id} in {zip_path.name}: "
                    f"{members[image_id]} and {member}"
                )
            members[image_id] = member
    return members


def read_rgb_png(archive: zipfile.ZipFile, member: str, output_size: int) -> np.ndarray:
    with archive.open(member) as handle:
        image = Image.open(BytesIO(handle.read())).convert("RGB")
    image = image.resize((output_size, output_size), resample=BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def read_binary_mask_png(archive: zipfile.ZipFile, member: str, output_size: int) -> np.ndarray:
    with archive.open(member) as handle:
        mask = Image.open(BytesIO(handle.read())).convert("L")
    mask = mask.resize((output_size, output_size), resample=NEAREST)
    mask_array = np.asarray(mask, dtype=np.uint8)
    return np.where(mask_array > 127, 255, 0).astype(np.uint8)


def preprocess_split(
    *,
    split: str,
    raw_root: Path,
    output_root: Path,
    output_size: int,
    max_items: int,
) -> list[dict[str, str | int]]:
    split_config = SPLIT_ZIPS[split]
    image_zip = raw_root / split_config["image_zip"]
    mask_zip = raw_root / split_config["mask_zip"]
    if not image_zip.is_file():
        raise FileNotFoundError(f"Missing ISIC2018 image zip: {image_zip}")
    if not mask_zip.is_file():
        raise FileNotFoundError(f"Missing ISIC2018 mask zip: {mask_zip}")

    image_members = collect_members(image_zip, (".jpg", ".jpeg", ".png"))
    mask_members = collect_members(mask_zip, (".png",))
    image_ids = sorted(image_members)
    if max_items > 0:
        image_ids = image_ids[:max_items]
    if not image_ids:
        raise RuntimeError(f"No ISIC2018 images found in {image_zip}")

    missing_masks = [image_id for image_id in image_ids if image_id not in mask_members]
    if missing_masks:
        preview = ", ".join(missing_masks[:5])
        raise FileNotFoundError(
            f"{len(missing_masks)} ISIC2018 {split} masks are missing in "
            f"{mask_zip.name}; first missing ids: {preview}"
        )

    image_dir = output_root / "ISIC18" / split / "images"
    mask_dir = output_root / "ISIC18" / split / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | int]] = []
    with zipfile.ZipFile(image_zip) as image_archive, zipfile.ZipFile(mask_zip) as mask_archive:
        for index, image_id in enumerate(image_ids, 1):
            image = read_rgb_png(image_archive, image_members[image_id], output_size)
            mask = read_binary_mask_png(mask_archive, mask_members[image_id], output_size)

            image_out = image_dir / f"{image_id}.png"
            mask_out = mask_dir / f"{image_id}.png"
            Image.fromarray(image, mode="RGB").save(image_out)
            Image.fromarray(mask, mode="L").save(mask_out)

            rows.append(
                {
                    "split": split,
                    "image_id": image_id,
                    "image": str(image_out),
                    "mask": str(mask_out),
                    "foreground_pixels": int((mask > 0).sum()),
                    "output_size": output_size,
                    "source_image_zip": str(image_zip),
                    "source_image_member": image_members[image_id],
                    "source_mask_zip": str(mask_zip),
                    "source_mask_member": mask_members[image_id],
                }
            )
            if index % 100 == 0 or index == len(image_ids):
                print(f"{split}: processed {index}/{len(image_ids)} images")
    return rows


def summarize_rows(rows: list[dict[str, str | int]]) -> tuple[dict[str, int], dict[str, int]]:
    counts = {}
    empty_foreground_counts = {}
    for split in SPLIT_ZIPS:
        split_rows = [row for row in rows if row["split"] == split]
        counts[split] = len(split_rows)
        empty_foreground_counts[split] = sum(int(row["foreground_pixels"]) == 0 for row in split_rows)
    return counts, empty_foreground_counts


def preprocess(args: argparse.Namespace) -> None:
    raw_root = Path(args.raw_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output root is not empty, pass --overwrite to replace: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | int]] = []
    for split in args.splits:
        split_rows = preprocess_split(
            split=split,
            raw_root=raw_root,
            output_root=output_root,
            output_size=args.output_size,
            max_items=args.max_items_per_split,
        )
        rows.extend(split_rows)
        split_list = output_root / f"{split}.txt"
        split_list.write_text(
            "\n".join(str(row["image_id"]) for row in split_rows) + ("\n" if split_rows else ""),
            encoding="utf-8",
        )

    fieldnames = [
        "split",
        "image_id",
        "image",
        "mask",
        "foreground_pixels",
        "output_size",
        "source_image_zip",
        "source_image_member",
        "source_mask_zip",
        "source_mask_member",
    ]
    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    counts, empty_foreground_counts = summarize_rows(rows)
    summary = {
        "dataset": "ISIC2018 Task 1",
        "task": "binary lesion segmentation",
        "class_names": CLASS_NAMES,
        "split_policy": "official ISIC2018 Task 1 training/validation/testing zip split, matching LEAF",
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "output_size": args.output_size,
        "image_channels": 3,
        "condition_image_channels_used_by_cdal": 1,
        "mask_channels": 1,
        "num_seg_classes": 2,
        "image_preprocessing": "RGB image resized to 256x256 with bilinear interpolation; cDAL loader converts to grayscale condition in [-1, 1]",
        "mask_preprocessing": "grayscale mask resized with nearest-neighbor interpolation and thresholded at >127 to {0, 255}; cDAL loader maps foreground to +1 and background to -1",
        "counts": counts,
        "empty_foreground_counts": empty_foreground_counts,
        "manifest": str(manifest_path),
        "source_zips": SPLIT_ZIPS,
        "label_policy": "ISIC2018 Task 1 lesion masks are binary foreground/background.",
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_root",
        default="raw/ISIC2018",
        help="Directory containing the official ISIC2018 Task 1 zip files.",
    )
    parser.add_argument(
        "--output_root",
        default="data_preprocessed/isic2018_task1_cdal_png",
        help="cDAL-owned ISIC2018 preprocessed output directory.",
    )
    parser.add_argument("--output_size", type=int, default=256)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=sorted(SPLIT_ZIPS),
        default=["training", "validation", "testing"],
    )
    parser.add_argument(
        "--max_items_per_split",
        type=int,
        default=0,
        help="Optional smoke-test limit per split. 0 processes all images.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    preprocess(build_argparser().parse_args())
