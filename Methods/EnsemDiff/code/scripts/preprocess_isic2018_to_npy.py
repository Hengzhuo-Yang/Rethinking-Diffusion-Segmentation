"""Preprocess ISIC 2018 Task 1 using the repository's fixed image splits."""

from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from fixed_split_manifests import (
    assert_exact_id_set,
    load_fixed_split_manifests,
    prepare_empty_output_directory,
)


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
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST_ROOT = SCRIPT_DIR.parent / "manifests" / "isic2018"
MANIFEST_FILENAMES = {
    "training": "training.txt",
    "validation": "validation.txt",
    "testing": "testing.txt",
}
EXPECTED_IMAGE_COUNTS = {"training": 2594, "validation": 100, "testing": 1000}


def _resample_constants():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.BILINEAR, Image.Resampling.NEAREST
    return Image.BILINEAR, Image.NEAREST


BILINEAR, NEAREST = _resample_constants()


def load_split(manifest_root: str | Path) -> dict[str, list[str]]:
    return load_fixed_split_manifests(
        manifest_root,
        manifest_filenames=MANIFEST_FILENAMES,
        expected_counts=EXPECTED_IMAGE_COUNTS,
        id_pattern=r"ISIC_\d{7}",
        dataset_name="ISIC2018",
    )


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


def read_rgb_image(
    archive: zipfile.ZipFile,
    member: str,
    output_size: int,
) -> np.ndarray:
    with archive.open(member) as handle:
        image = Image.open(BytesIO(handle.read())).convert("RGB")
    image = image.resize((output_size, output_size), resample=BILINEAR)
    array = np.asarray(image, dtype=np.float32)
    return (array / 127.5) - 1.0


def read_binary_mask(
    archive: zipfile.ZipFile,
    member: str,
    output_size: int,
) -> np.ndarray:
    with archive.open(member) as handle:
        mask = Image.open(BytesIO(handle.read())).convert("L")
    mask = mask.resize((output_size, output_size), resample=NEAREST)
    array = np.asarray(mask, dtype=np.uint8)
    return (array > 127).astype(np.uint8)


def preprocess_split(
    *,
    split: str,
    expected_ids: list[str],
    raw_root: Path,
    output_root: Path,
    output_size: int,
    max_items: int,
) -> list[dict[str, str | int]]:
    config = SPLIT_ZIPS[split]
    image_zip = raw_root / config["image_zip"]
    mask_zip = raw_root / config["mask_zip"]
    if not image_zip.is_file():
        raise FileNotFoundError(f"Missing ISIC2018 image zip: {image_zip}")
    if not mask_zip.is_file():
        raise FileNotFoundError(f"Missing ISIC2018 mask zip: {mask_zip}")

    image_members = collect_members(image_zip, (".jpg", ".jpeg", ".png"))
    mask_members = collect_members(mask_zip, (".png",))
    assert_exact_id_set(
        image_members,
        expected_ids,
        dataset_name="ISIC2018",
        split_name=split,
        source_name=image_zip.name,
    )
    assert_exact_id_set(
        mask_members,
        expected_ids,
        dataset_name="ISIC2018",
        split_name=split,
        source_name=mask_zip.name,
    )

    image_ids = list(expected_ids)
    if max_items > 0:
        image_ids = image_ids[:max_items]

    rows: list[dict[str, str | int]] = []
    split_dir = output_root / split
    with zipfile.ZipFile(image_zip) as image_archive, zipfile.ZipFile(mask_zip) as mask_archive:
        for index, image_id in enumerate(image_ids, 1):
            case_dir = split_dir / image_id
            case_dir.mkdir(parents=True, exist_ok=True)
            image = read_rgb_image(image_archive, image_members[image_id], output_size)
            mask_member = mask_members[image_id]
            mask = read_binary_mask(mask_archive, mask_member, output_size)

            image_out = case_dir / f"{image_id}_image.npy"
            mask_out = case_dir / f"{image_id}_seg.npy"
            np.save(image_out, image.astype(np.float32))
            np.save(mask_out, mask.astype(np.uint8))

            rows.append(
                {
                    "split": split,
                    "image_id": image_id,
                    "image": image_out.relative_to(output_root).as_posix(),
                    "label": mask_out.relative_to(output_root).as_posix(),
                    "foreground_pixels": int(mask.sum()),
                    "output_size": output_size,
                    "source_image_zip": image_zip.name,
                    "source_image_member": image_members[image_id],
                    "source_mask_zip": mask_zip.name,
                    "source_mask_member": mask_member,
                }
            )
            if index % 100 == 0 or index == len(image_ids):
                print(f"{split}: processed {index}/{len(image_ids)} images")

    return rows


def preprocess(args: argparse.Namespace) -> None:
    raw_root = Path(args.raw_root).expanduser().resolve()
    fixed_split = load_split(args.manifest_root)
    output_root = prepare_empty_output_directory(
        args.output_root,
        dataset_name="ISIC2018",
    )

    all_rows = []
    counts = {}
    foreground_empty = {}
    for split in args.splits:
        rows = preprocess_split(
            split=split,
            expected_ids=fixed_split[split],
            raw_root=raw_root,
            output_root=output_root,
            output_size=args.output_size,
            max_items=args.max_items_per_split,
        )
        all_rows.extend(rows)
        counts[split] = len(rows)
        foreground_empty[split] = sum(int(row["foreground_pixels"]) == 0 for row in rows)
        processed_list = output_root / f"{split}.txt"
        processed_list.write_text(
            "\n".join(str(row["image_id"]) for row in rows) + ("\n" if rows else ""),
            encoding="utf-8",
        )

    fieldnames = [
        "split",
        "image_id",
        "image",
        "label",
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
        writer.writerows(all_rows)

    summary = {
        "dataset": "ISIC2018 Task 1",
        "task": "binary lesion segmentation",
        "class_names": {"0": "background", "1": "lesion"},
        "split_policy": "repository-owned fixed official ISIC2018 Task 1 split",
        "fixed_split_manifest_files": MANIFEST_FILENAMES,
        "fixed_image_counts": EXPECTED_IMAGE_COUNTS,
        "processed_counts": counts,
        "processed_splits": list(args.splits),
        "output_size": args.output_size,
        "image_channels": 3,
        "mask_channels": 1,
        "num_seg_classes": 2,
        "image_normalization": "RGB uint8 scaled to [-1, 1]",
        "mask_normalization": "PNG grayscale thresholded at >127 to {0, 1}",
        "empty_foreground_counts": foreground_empty,
        "manifest": manifest_path.name,
        "source_zips": SPLIT_ZIPS,
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
        help="Directory containing the official ISIC2018 Task 1 zip files.",
    )
    parser.add_argument(
        "--manifest_root",
        "--manifest-root",
        dest="manifest_root",
        default=str(DEFAULT_MANIFEST_ROOT),
        help="Fixed ID-only split directory (defaults to code/manifests/isic2018).",
    )
    parser.add_argument(
        "--output_root",
        "--output-root",
        required=True,
        help="Destination for generated ISIC2018 arrays and metadata.",
    )
    parser.add_argument("--output_size", "--output-size", type=int, default=224)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=sorted(SPLIT_ZIPS),
        default=["training", "validation", "testing"],
    )
    parser.add_argument(
        "--max_items_per_split",
        "--max-items-per-split",
        type=int,
        default=0,
        help="Optional smoke-test limit applied only after full zip/manifest validation.",
    )
    return parser


if __name__ == "__main__":
    preprocess(build_argparser().parse_args())


