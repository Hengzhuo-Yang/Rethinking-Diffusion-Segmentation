"""Preprocess shared raw ISIC2018 Task 1 zips into a LEAF-owned PNG cache.

Raw and output roots are supplied explicitly; raw data is never copied into the repository.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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

DEFAULT_MANIFEST_ROOT = Path(__file__).resolve().parents[1] / "manifests" / "isic2018"
MANIFEST_NAMES = {"training": "train.txt", "validation": "val.txt", "testing": "test.txt"}
EXPECTED_COUNTS = {"training": 2594, "validation": 100, "testing": 1000}
EXPECTED_MANIFEST_FINGERPRINTS = {
    "training": "b6a11fa50b1cf0363160258b5cac96ccae382ea524e34434c3212d1fbb1e4775",
    "validation": "44c159239176fe8f63c1efd5f40b5cf66fcafcfd61571c684cd8467f912a366c",
    "testing": "6fb4df9f361085d70966ce9098133a6491b756869717af4eaef4ca687686e4f7",
}


def _manifest_fingerprint(identifiers: list[str]) -> str:
    canonical = "\n".join(identifiers) + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

CLASS_NAMES = {
    0: "background",
    1: "lesion",
}


def _resample_constants():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.BILINEAR, Image.Resampling.NEAREST
    return Image.BILINEAR, Image.NEAREST


BILINEAR, NEAREST = _resample_constants()


def load_manifests(manifest_root: str | Path) -> dict[str, set[str]]:
    root = Path(manifest_root).expanduser().resolve()
    manifests = {}
    for split, filename in MANIFEST_NAMES.items():
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing fixed ISIC2018 manifest: {path}")
        ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if _manifest_fingerprint(ids) != EXPECTED_MANIFEST_FINGERPRINTS[split]:
            raise ValueError(
                f"ISIC2018 {split} manifest differs from the fixed release manifest: {path}"
            )
        if len(ids) != EXPECTED_COUNTS[split] or len(ids) != len(set(ids)):
            raise ValueError(f"ISIC2018 {split} manifest count/uniqueness mismatch: {path}")
        manifests[split] = set(ids)
    if sum(map(len, manifests.values())) != len(set().union(*manifests.values())):
        raise ValueError("ISIC2018 fixed manifests overlap across train/val/test.")
    return manifests

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
    expected_ids: set[str],
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
    if set(image_ids) != expected_ids:
        missing = sorted(expected_ids - set(image_ids))
        unexpected = sorted(set(image_ids) - expected_ids)
        raise ValueError(
            f"ISIC2018 {split} zip IDs do not match fixed manifest; missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
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
                    "image": image_out.relative_to(output_root).as_posix(),
                    "mask": mask_out.relative_to(output_root).as_posix(),
                    "foreground_pixels": int((mask > 0).sum()),
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


def summarize_rows(rows: list[dict[str, str | int]]) -> tuple[dict[str, int], dict[str, int]]:
    counts = {}
    empty_foreground_counts = {}
    for split in SPLIT_ZIPS:
        split_rows = [row for row in rows if row["split"] == split]
        counts[split] = len(split_rows)
        empty_foreground_counts[split] = sum(
            int(row["foreground_pixels"]) == 0 for row in split_rows
        )
    return counts, empty_foreground_counts


def preprocess(args: argparse.Namespace) -> None:
    raw_root = Path(args.raw_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output root is not empty, pass --overwrite to replace: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifests = load_manifests(args.manifest_root)
    rows: list[dict[str, str | int]] = []
    for split in SPLIT_ZIPS:
        split_rows = preprocess_split(
            split=split,
            raw_root=raw_root,
            output_root=output_root,
            output_size=args.output_size,
            expected_ids=manifests[split],
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
    if counts != EXPECTED_COUNTS:
        raise RuntimeError(f"ISIC2018 counts differ from release contract: {counts}")
    summary = {
        "dataset": "ISIC2018 Task 1",
        "task": "binary lesion segmentation",
        "class_names": CLASS_NAMES,
        "split_policy": "official ISIC2018 Task 1 training/validation/testing zip split, matching EnsemDiff",
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "output_size": args.output_size,
        "image_channels": 3,
        "mask_channels": 1,
        "num_seg_classes": 2,
        "image_normalization": "RGB uint8 PNG; LEAF loader scales to [0, 1]",
        "mask_normalization": "grayscale PNG thresholded at >127 to {0, 255}; LEAF loader scales to {0, 1}",
        "counts": counts,
        "empty_foreground_counts": empty_foreground_counts,
        "manifest": str(manifest_path),
        "manifest_root": str(Path(args.manifest_root).expanduser().resolve()),
        "source_zips": SPLIT_ZIPS,
        "label_policy": "ISIC2018 Task 1 lesion masks are binary foreground/background.",
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", required=True, help="Directory containing official Task 1 zip files.")
    parser.add_argument("--output_root", required=True, help="New LEAF-owned ISIC2018 output directory.")
    parser.add_argument(
        "--manifest_root", default=str(DEFAULT_MANIFEST_ROOT),
        help="Directory containing fixed train.txt, val.txt, and test.txt image manifests.",
    )
    parser.add_argument("--output_size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    preprocess(build_argparser().parse_args())
