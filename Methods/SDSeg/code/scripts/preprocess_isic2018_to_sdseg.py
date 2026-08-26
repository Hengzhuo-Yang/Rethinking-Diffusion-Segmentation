"""Preprocess official ISIC2018 Task 1 archives into fixed SDSeg splits.

Raw archives remain outside the repository. Generated paths are relative.
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


def read_rgb_image(archive: zipfile.ZipFile, member: str, output_size: int) -> np.ndarray:
    with archive.open(member) as handle:
        image = Image.open(BytesIO(handle.read())).convert("RGB")
    image = image.resize((output_size, output_size), resample=BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def read_binary_mask(archive: zipfile.ZipFile, member: str, output_size: int) -> np.ndarray:
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

    image_dir = output_root / split / "images"
    mask_dir = output_root / split / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | int]] = []
    with zipfile.ZipFile(image_zip) as image_archive, zipfile.ZipFile(mask_zip) as mask_archive:
        for index, image_id in enumerate(image_ids, 1):
            image = read_rgb_image(image_archive, image_members[image_id], output_size)
            mask = read_binary_mask(mask_archive, mask_members[image_id], output_size)

            image_out = image_dir / f"{image_id}.png"
            mask_out = mask_dir / f"{image_id}.png"
            Image.fromarray(image, mode="RGB").save(image_out)
            Image.fromarray(mask, mode="L").save(mask_out)

            rows.append(
                {
                    "split": split,
                    "image_id": image_id,
                    "image": image_out.relative_to(output_root).as_posix(),
                    "mask": mask_out.relative_to(output_root).as_posix(),
                    "foreground_pixels": int((mask > 0).sum()),
                    "empty_foreground": int((mask > 0).sum() == 0),
                    "output_size": output_size,
                    "source_image_zip": image_zip.name,
                    "source_image_member": image_members[image_id],
                    "source_mask_zip": mask_zip.name,
                    "source_mask_member": mask_members[image_id],
                }
            )
            if index % 100 == 0 or index == len(image_ids):
                print(f"{split}: processed {index}/{len(image_ids)} images")
    return rows


def summarize_rows(rows: list[dict[str, str | int]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for split in SPLIT_ZIPS:
        split_rows = [row for row in rows if row["split"] == split]
        summary[split] = {
            "images": len(split_rows),
            "empty_foreground_images": sum(int(row["empty_foreground"]) for row in split_rows),
            "foreground_pixels": sum(int(row["foreground_pixels"]) for row in split_rows),
        }
    return summary


def preprocess(args: argparse.Namespace) -> None:
    if int(args.num_classes) != 2:
        raise ValueError("ISIC2018 Task 1 preprocessing expects num_classes=2.")

    raw_root = Path(args.raw_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output root is not empty, pass --overwrite to replace: {output_root}")
        for name in SPLIT_ZIPS:
            path = output_root / name
            if path.exists():
                shutil.rmtree(path)
        for name in ("manifest.csv", "summary.json", "training.txt", "validation.txt", "testing.txt"):
            path = output_root / name
            if path.exists():
                path.unlink()
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
        "empty_foreground",
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

    summary = {
        "dataset": "ISIC2018 Task 1",
        "task": "binary lesion segmentation",
        "class_names": CLASS_NAMES,
        "split_policy": "official ISIC2018 Task 1 training/validation/testing zip split, matching LEAF",
        "output_size": args.output_size,
        "image_channels": 3,
        "mask_channels": 1,
        "num_seg_classes": args.num_classes,
        "image_normalization": "RGB resized with bilinear interpolation and stored as uint8 PNG; SDSeg loader scales to [-1, 1]",
        "mask_normalization": "nearest-neighbor binary mask thresholded at >127 to {0, 255}; SDSeg loader expands to repeated RGB labels",
        "split_summaries": summarize_rows(rows),
        "total_images": len(rows),
        "manifest": "manifest.csv",
        "source_zips": SPLIT_ZIPS,
        "label_policy": "ISIC2018 Task 1 lesion masks are binary foreground/background.",
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", "--raw_root", dest="raw_root", required=True,
                        help="Directory containing official ISIC2018 Task 1 archives.")
    parser.add_argument("--output-root", "--output_root", dest="output_root", required=True)
    parser.add_argument("--output-size", "--output_size", dest="output_size", type=int, default=256)
    parser.add_argument("--num-classes", "--num_classes", dest="num_classes", type=int, default=2)
    parser.add_argument("--splits", nargs="+", choices=sorted(SPLIT_ZIPS),
                        default=["training", "validation", "testing"])
    parser.add_argument("--max-items-per-split", "--max_items_per_split",
                        dest="max_items_per_split", type=int, default=0,
                        help="Smoke-only per-split limit; 0 processes all images.")
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    preprocess(build_argparser().parse_args())
