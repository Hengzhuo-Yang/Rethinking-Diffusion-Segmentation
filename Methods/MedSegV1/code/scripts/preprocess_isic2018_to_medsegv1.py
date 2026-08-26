"""Preprocess shared ISIC2018 Task 1 zips into a MedSegV1-owned PNG cache."""

import argparse
import csv
import io
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


SPLITS = {
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


def _resample_constants():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.BILINEAR, Image.Resampling.NEAREST
    return Image.BILINEAR, Image.NEAREST


BILINEAR, NEAREST = _resample_constants()


def image_id_from_member(member):
    return Path(member).stem


def mask_id_from_member(member):
    stem = Path(member).stem
    suffix = "_segmentation"
    return stem[: -len(suffix)] if stem.endswith(suffix) else stem


def zip_members_by_id(zip_path, kind):
    suffixes = (".jpg", ".jpeg", ".png")
    id_fn = image_id_from_member if kind == "image" else mask_id_from_member
    with zipfile.ZipFile(zip_path) as archive:
        members = [
            name
            for name in archive.namelist()
            if not name.endswith("/") and Path(name).suffix.lower() in suffixes
        ]
    by_id = {}
    for member in sorted(members):
        image_id = id_fn(member)
        if image_id in by_id:
            raise ValueError(f"Duplicate {kind} id {image_id} in {zip_path}")
        by_id[image_id] = member
    if not by_id:
        raise RuntimeError(f"No {kind} files found in {zip_path}")
    return by_id


def read_image(archive, member, output_size):
    with archive.open(member) as handle:
        image = Image.open(io.BytesIO(handle.read())).convert("RGB")
    image = image.resize((output_size, output_size), resample=BILINEAR)
    return image


def read_mask(archive, member, output_size):
    with archive.open(member) as handle:
        mask = Image.open(io.BytesIO(handle.read())).convert("L")
    mask = mask.resize((output_size, output_size), resample=NEAREST)
    arr = np.asarray(mask, dtype=np.uint8)
    arr = np.where(arr > 127, 255, 0).astype(np.uint8)
    return Image.fromarray(arr), int((arr > 0).sum())


def preprocess_split(raw_root, output_root, split, output_size, max_items):
    spec = SPLITS[split]
    image_zip = raw_root / spec["image_zip"]
    mask_zip = raw_root / spec["mask_zip"]
    if not image_zip.is_file():
        raise FileNotFoundError(f"Missing ISIC2018 image zip: {image_zip}")
    if not mask_zip.is_file():
        raise FileNotFoundError(f"Missing ISIC2018 mask zip: {mask_zip}")

    images_by_id = zip_members_by_id(image_zip, "image")
    masks_by_id = zip_members_by_id(mask_zip, "mask")
    missing_masks = sorted(set(images_by_id) - set(masks_by_id))
    missing_images = sorted(set(masks_by_id) - set(images_by_id))
    if missing_masks or missing_images:
        raise RuntimeError(
            f"ISIC2018 {split} image/mask mismatch: "
            f"missing_masks={len(missing_masks)} missing_images={len(missing_images)}"
        )

    image_ids = sorted(images_by_id)
    if max_items:
        image_ids = image_ids[:max_items]

    image_dir = output_root / "ISIC18" / split / "images"
    mask_dir = output_root / "ISIC18" / split / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with zipfile.ZipFile(image_zip) as image_archive, zipfile.ZipFile(mask_zip) as mask_archive:
        for image_id in image_ids:
            image_member = images_by_id[image_id]
            mask_member = masks_by_id[image_id]
            image = read_image(image_archive, image_member, output_size)
            mask, foreground_pixels = read_mask(mask_archive, mask_member, output_size)
            image_out = image_dir / f"{image_id}.png"
            mask_out = mask_dir / f"{image_id}.png"
            image.save(image_out)
            mask.save(mask_out)
            rows.append(
                {
                    "split": split,
                    "image_id": image_id,
                    "image": str(image_out),
                    "mask": str(mask_out),
                    "foreground_pixels": foreground_pixels,
                    "output_size": output_size,
                    "source_image_zip": str(image_zip),
                    "source_image_member": image_member,
                    "source_mask_zip": str(mask_zip),
                    "source_mask_member": mask_member,
                }
            )
    print(f"{split}: wrote {len(rows)} image/mask pairs")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", default="../../data/ISIC2018/raw")
    parser.add_argument("--output_root", default="../data_preprocessed/isic2018_task1_medsegv1_png")
    parser.add_argument("--output_size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_items_per_split", type=int, default=0)
    args = parser.parse_args()

    raw_root = Path(args.raw_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Missing shared ISIC2018 raw root: {raw_root}")
    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output root is not empty, pass --overwrite to replace: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for split in ("training", "validation", "testing"):
        rows.extend(
            preprocess_split(
                raw_root=raw_root,
                output_root=output_root,
                split=split,
                output_size=args.output_size,
                max_items=args.max_items_per_split,
            )
        )

    manifest_path = output_root / "manifest.csv"
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
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    counts = {split: sum(row["split"] == split for row in rows) for split in SPLITS}
    empty_counts = {
        split: sum(row["split"] == split and int(row["foreground_pixels"]) == 0 for row in rows)
        for split in SPLITS
    }
    summary = {
        "dataset": "ISIC2018 Task 1",
        "task": "binary lesion segmentation",
        "class_names": {"0": "background", "1": "lesion"},
        "split_policy": "official ISIC2018 Task 1 training/validation/testing zip split, matching LEAF",
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "output_size": args.output_size,
        "image_channels": 3,
        "mask_channels": 1,
        "num_seg_classes": 2,
        "image_normalization": "RGB uint8 PNG; MedSegV1 loader scales to [0, 1]",
        "mask_normalization": "grayscale PNG thresholded at >127 to {0, 255}; loader scales to {0, 1}",
        "counts": counts,
        "empty_foreground_counts": empty_counts,
        "manifest": str(manifest_path),
        "source_zips": SPLITS,
        "label_policy": "ISIC2018 Task 1 lesion masks are binary foreground/background.",
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
