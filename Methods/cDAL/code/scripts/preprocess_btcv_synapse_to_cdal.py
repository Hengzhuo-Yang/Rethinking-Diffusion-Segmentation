"""Preprocess BTCV/Synapse NIfTI volumes into the fixed cDAL PNG layout."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image


def _resample_constants():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.BILINEAR, Image.Resampling.NEAREST
    return Image.BILINEAR, Image.NEAREST


BILINEAR, NEAREST = _resample_constants()


def read_case_manifest(path: str | Path) -> list[str]:
    values = []
    seen = set()
    for raw_line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.search(r"(\d+)", line)
        if not match:
            raise ValueError(f"Could not parse BTCV case id from {line!r}")
        case_id = match.group(1).zfill(4)
        if case_id in seen:
            raise ValueError(f"Duplicate BTCV case id {case_id} in {path}")
        seen.add(case_id)
        values.append(case_id)
    return values


def load_fixed_split(args: argparse.Namespace) -> dict[str, list[str]]:
    split = {
        "train": read_case_manifest(args.train_manifest),
        "validation": read_case_manifest(args.validation_manifest),
        "test": read_case_manifest(args.test_manifest),
    }
    expected_counts = {"train": 18, "validation": 2, "test": 10}
    for name, count in expected_counts.items():
        if len(split[name]) != count:
            raise ValueError(f"BTCV {name} manifest must contain {count} cases")
    all_ids = [case_id for values in split.values() for case_id in values]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("BTCV case manifests overlap")
    return split


def find_training_root(raw_root: str | Path) -> Path:
    root = Path(raw_root).expanduser().resolve()
    for candidate in (root, root / "RawData" / "Training", root / "raw" / "RawData" / "Training"):
        if (candidate / "img").is_dir() and (candidate / "label").is_dir():
            return candidate
    raise FileNotFoundError("Could not find BTCV RawData/Training/img and label directories")


def volume_id(path: Path) -> str:
    match = re.search(r"(\d+)", path.name)
    if not match:
        raise ValueError(f"Could not parse BTCV volume id from {path.name}")
    return match.group(1).zfill(4)


def normalize_ct_to_uint8(image: np.ndarray, mode: str) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    if mode == "sdseg_2d":
        low, high = -175.0, 250.0
    elif mode == "synapse_window":
        low, high = -125.0, 275.0
    else:
        raise ValueError(f"Unknown window mode: {mode}")
    image = np.clip(image, low, high)
    image = (image - low) / (high - low)
    return np.round(image * 255.0).astype(np.uint8)


def transfer_to_synapse8(labels: np.ndarray) -> np.ndarray:
    labels = labels.copy()
    labels[labels == 5] = 0
    labels[labels == 6] = 5
    labels[labels == 7] = 6
    labels[labels == 8] = 7
    labels[labels == 9] = 0
    labels[labels == 10] = 0
    labels[labels == 11] = 8
    labels[labels == 12] = 0
    labels[labels == 13] = 0
    return labels


def resize(array: np.ndarray, output_size: int, resample) -> np.ndarray:
    image = Image.fromarray(array.astype(np.uint8))
    image = image.resize((output_size, output_size), resample=resample)
    return np.asarray(image, dtype=np.uint8)


def preprocess_volume(
    image_path: Path,
    label_path: Path,
    role: str,
    output_root: Path,
    output_size: int,
    window_mode: str,
) -> list[dict[str, str | int]]:
    image = nib.load(str(image_path)).get_fdata()
    label = nib.load(str(label_path)).get_fdata()
    if image.shape != label.shape:
        raise ValueError(f"Shape mismatch for {image_path.name}: {image.shape} vs {label.shape}")

    case_id = volume_id(image_path)
    physical_split = "train" if role == "train" else "heldout"
    image_dir = output_root / "BTCV" / physical_split / "images"
    mask_dir = output_root / "BTCV" / physical_split / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for slice_index in range(image.shape[2]):
        slice_id = f"vol{case_id}_{slice_index:03d}"
        image_slice = resize(
            normalize_ct_to_uint8(image[:, :, slice_index], window_mode),
            output_size,
            BILINEAR,
        )
        image_rgb = np.repeat(image_slice[:, :, None], 3, axis=2)
        label_slice = resize(label[:, :, slice_index], output_size, NEAREST)
        label_slice = transfer_to_synapse8(label_slice)
        mask = np.where(label_slice > 0, 255, 0).astype(np.uint8)

        image_out = image_dir / f"{slice_id}.png"
        mask_out = mask_dir / f"{slice_id}.png"
        Image.fromarray(image_rgb, mode="RGB").save(image_out)
        Image.fromarray(mask, mode="L").save(mask_out)
        rows.append(
            {
                "role": role,
                "physical_split": physical_split,
                "case_id": f"case{case_id}",
                "slice_index": slice_index,
                "slice_id": slice_id,
                "image": str(image_out.relative_to(output_root)),
                "mask": str(mask_out.relative_to(output_root)),
                "foreground_pixels": int((mask > 0).sum()),
            }
        )
    return rows


def preprocess(args: argparse.Namespace) -> None:
    training_root = find_training_root(args.raw_root)
    split = load_fixed_split(args)
    role_by_id = {
        case_id: role
        for role, case_ids in split.items()
        for case_id in case_ids
    }
    output_root = Path(args.output_root).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output root is not empty, pass --overwrite to replace: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    image_paths = sorted((training_root / "img").glob("img*.nii.gz"))
    available_ids = {volume_id(path) for path in image_paths}
    required_ids = set(role_by_id)
    if missing := sorted(required_ids - available_ids):
        raise FileNotFoundError(f"Missing BTCV raw case(s): {', '.join(missing)}")

    rows = []
    for image_path in image_paths:
        case_id = volume_id(image_path)
        if case_id not in role_by_id:
            continue
        label_path = training_root / "label" / image_path.name.replace("img", "label", 1)
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing BTCV label for {image_path.name}")
        volume_rows = preprocess_volume(
            image_path,
            label_path,
            role_by_id[case_id],
            output_root,
            args.output_size,
            args.window_mode,
        )
        rows.extend(volume_rows)
        print(f"{role_by_id[case_id]}: {image_path.name} -> {len(volume_rows)} slices")

    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    role_counts = {
        role: sum(row["role"] == role for row in rows)
        for role in ("train", "validation", "test")
    }
    summary = {
        "dataset": "BTCV/Synapse binary 2D",
        "output_size": args.output_size,
        "window_mode": args.window_mode,
        "case_ids": {role: [f"case{value}" for value in values] for role, values in split.items()},
        "case_counts": {role: len(values) for role, values in split.items()},
        "slice_counts": role_counts,
        "physical_layout": {"train": "BTCV/train", "heldout_pool": "BTCV/heldout"},
        "label_policy": "BTCV labels mapped to Synapse 8 organs, then collapsed to binary foreground/background.",
        "manifest": "manifest.csv",
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", default="raw/BTCV")
    parser.add_argument("--output_root", default="data_preprocessed/btcv_synapse_cdal_binary_png")
    parser.add_argument("--train_manifest", default="manifests/btcv/train_cases.txt")
    parser.add_argument("--validation_manifest", default="manifests/btcv/validation_cases.txt")
    parser.add_argument("--test_manifest", default="manifests/btcv/test_cases.txt")
    parser.add_argument("--output_size", type=int, default=256)
    parser.add_argument("--window_mode", choices=["sdseg_2d", "synapse_window"], default="sdseg_2d")
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    preprocess(build_argparser().parse_args())
