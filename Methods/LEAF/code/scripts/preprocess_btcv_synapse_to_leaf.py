"""Preprocess shared BTCV/Synapse NIfTI data into a LEAF-owned 2D PNG cache."""

import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image


DEFAULT_MANIFEST_ROOT = Path(__file__).resolve().parents[1] / "manifests" / "btcv"
EXPECTED_SLICE_COUNTS = {"train": 2211, "val": 295, "test": 1273}
EXPECTED_MANIFEST_FINGERPRINTS = {
    "train": "dfdb73f71f36503c06c569760e77ab15fa67cdce84341127e553b327e2219c71",
    "val": "47ed3d1d80a76914ffa3bb75ecc33f40b9b8ae7c113d852fe32f3aff5031efc3",
    "test": "fa5ebbed2ed11e94f6a7dce1c89ff9c03bf4d9cf2e893bf28da9cd0f9f193b3c",
}


def _manifest_fingerprint(identifiers):
    canonical = "\n".join(identifiers) + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _resample_constants():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.BILINEAR, Image.Resampling.NEAREST
    return Image.BILINEAR, Image.NEAREST


BILINEAR, NEAREST = _resample_constants()


def find_training_root(raw_root):
    root = Path(raw_root).expanduser().resolve()
    candidates = [
        root,
        root / "RawData" / "Training",
        root / "raw" / "RawData" / "Training",
    ]
    for candidate in candidates:
        if (candidate / "img").is_dir() and (candidate / "label").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find BTCV RawData/Training/img and label directories under "
        f"{root}"
    )


def volume_id(path):
    match = re.search(r"(\d+)", path.name)
    if not match:
        raise ValueError(f"Could not parse BTCV volume id from {path.name}")
    return match.group(1)


def label_path_for(image_path, label_dir):
    return label_dir / image_path.name.replace("img", "label", 1)


def normalize_ct_to_uint8(image, mode):
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


def transfer_to_synapse8(gts):
    gts = gts.copy()
    gts[gts == 5] = 0
    gts[gts == 6] = 5
    gts[gts == 7] = 6
    gts[gts == 8] = 7
    gts[gts == 9] = 0
    gts[gts == 10] = 0
    gts[gts == 11] = 8
    gts[gts == 12] = 0
    gts[gts == 13] = 0
    return gts


def resize_image(slice_2d, output_size):
    image = Image.fromarray(slice_2d.astype(np.uint8))
    image = image.resize((output_size, output_size), resample=BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def resize_label(slice_2d, output_size):
    image = Image.fromarray(slice_2d.astype(np.uint8))
    image = image.resize((output_size, output_size), resample=NEAREST)
    return np.asarray(image, dtype=np.uint8)


def load_partitions(manifest_root):
    root = Path(manifest_root).expanduser().resolve()
    partitions = {}
    for split in ("train", "val", "test"):
        manifest = root / f"{split}.txt"
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing fixed BTCV manifest: {manifest}")
        case_ids = [line.strip() for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        if _manifest_fingerprint(case_ids) != EXPECTED_MANIFEST_FINGERPRINTS[split]:
            raise ValueError(f"BTCV {split} manifest differs from the fixed release manifest: {manifest}")
        if len(case_ids) != len(set(case_ids)):
            raise ValueError(f"Duplicate BTCV IDs in {manifest}")
        if any(re.fullmatch(r"case\d{4}", case_id) is None for case_id in case_ids):
            raise ValueError(f"Invalid BTCV case ID in {manifest}")
        partitions[split] = {case_id.removeprefix("case") for case_id in case_ids}
    all_ids = set().union(*partitions.values())
    if sum(len(ids) for ids in partitions.values()) != len(all_ids):
        raise ValueError("BTCV fixed manifests overlap across train/val/test.")
    return partitions


def split_for(image_path, partitions):
    vid = volume_id(image_path)
    matches = [split for split, ids in partitions.items() if vid in ids]
    if len(matches) != 1:
        raise ValueError(f"BTCV volume {vid} is assigned to {len(matches)} fixed partitions.")
    return matches[0]


def preprocess_volume(image_path, label_path, split, output_root, output_size, window_mode):
    image = nib.load(str(image_path)).get_fdata()
    label = nib.load(str(label_path)).get_fdata()
    if image.shape != label.shape:
        raise ValueError(f"Shape mismatch for {image_path.name}: {image.shape} vs {label.shape}")

    vid = volume_id(image_path)
    image_dir = output_root / "BTCV" / split / "images"
    mask_dir = output_root / "BTCV" / split / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for slice_index in range(image.shape[2]):
        slice_id = f"vol{vid}_{slice_index:03d}"

        image_slice = normalize_ct_to_uint8(image[:, :, slice_index], window_mode)
        image_slice = resize_image(image_slice, output_size)
        image_rgb = np.repeat(image_slice[:, :, None], 3, axis=2)

        label_slice = resize_label(label[:, :, slice_index], output_size)
        label_slice = transfer_to_synapse8(label_slice)
        label_slice = np.where(label_slice > 0, 255, 0).astype(np.uint8)

        image_out = image_dir / f"{slice_id}.png"
        label_out = mask_dir / f"{slice_id}.png"
        Image.fromarray(image_rgb, mode="RGB").save(image_out)
        Image.fromarray(label_slice, mode="L").save(label_out)

        rows.append(
            {
                "split": split,
                "volume": vid,
                "slice": slice_index,
                "image": image_out.relative_to(output_root).as_posix(),
                "mask": label_out.relative_to(output_root).as_posix(),
                "foreground_pixels": int((label_slice > 0).sum()),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", required=True, help="BTCV RawData/Training or a parent containing it.")
    parser.add_argument("--output_root", required=True, help="New LEAF-owned preprocessed output directory.")
    parser.add_argument(
        "--manifest_root",
        default=str(DEFAULT_MANIFEST_ROOT),
        help="Directory containing fixed train.txt, val.txt, and test.txt case manifests.",
    )
    parser.add_argument("--output_size", type=int, default=256)
    parser.add_argument("--window_mode", choices=["sdseg_2d", "synapse_window"], default="sdseg_2d")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    training_root = find_training_root(args.raw_root)
    img_dir = training_root / "img"
    label_dir = training_root / "label"
    output_root = Path(args.output_root).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output root is not empty, pass --overwrite to replace: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(img_dir.glob("img*.nii.gz"))
    if not image_paths:
        raise RuntimeError(f"No BTCV images found under {img_dir}")

    partitions = load_partitions(args.manifest_root)
    rows = []
    counts = {"train": 0, "val": 0, "test": 0}
    volumes = {"train": set(), "val": set(), "test": set()}
    for image_path in image_paths:
        label_path = label_path_for(image_path, label_dir)
        if not label_path.exists():
            raise FileNotFoundError(f"Missing BTCV label for {image_path}: {label_path}")
        split = split_for(image_path, partitions)
        volume_rows = preprocess_volume(image_path, label_path, split, output_root, args.output_size, args.window_mode)
        rows.extend(volume_rows)
        counts[split] += len(volume_rows)
        volumes[split].add(volume_id(image_path))
        print(f"{split}: {image_path.name} -> {len(volume_rows)} slices")

    if counts != EXPECTED_SLICE_COUNTS:
        raise RuntimeError(
            f"BTCV slice counts differ from the fixed release contract: {counts} != {EXPECTED_SLICE_COUNTS}"
        )
    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "volume", "slice", "image", "mask", "foreground_pixels"],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "raw_root": str(training_root),
        "output_root": str(output_root),
        "output_size": args.output_size,
        "window_mode": args.window_mode,
        "manifest_root": str(Path(args.manifest_root).expanduser().resolve()),
        "volumes": {key: sorted(value) for key, value in volumes.items()},
        "counts": counts,
        "total_slices": len(rows),
        "total_unique_source_slices": len(rows),
        "manifest": str(manifest_path),
        "label_policy": "BTCV labels mapped to Synapse 8 organs, then collapsed to binary foreground/background.",
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
