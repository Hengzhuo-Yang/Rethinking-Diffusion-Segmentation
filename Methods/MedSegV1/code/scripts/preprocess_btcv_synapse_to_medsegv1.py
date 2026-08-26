"""Preprocess shared BTCV/Synapse NIfTI data into a MedSegV1-owned 2D PNG cache."""

import argparse
import csv
import json
import re
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image


VALIDATION_VOLUMES = {
    "0001",
    "0008",
}

TEST_VOLUMES = {
    "0002",
    "0003",
    "0004",
    "0022",
    "0025",
    "0029",
    "0032",
    "0035",
    "0036",
    "0038",
}

TRAIN_VOLUMES = {
    "0005",
    "0006",
    "0007",
    "0009",
    "0010",
    "0021",
    "0023",
    "0024",
    "0026",
    "0027",
    "0028",
    "0030",
    "0031",
    "0033",
    "0034",
    "0037",
    "0039",
    "0040",
}

EXPECTED_CASE_COUNTS = {"train": 18, "val": 2, "test": 10}
EXPECTED_SLICE_COUNTS = {"train": 2211, "val": 295, "test": 1273}


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


def split_for(image_path):
    vid = volume_id(image_path)
    if vid in VALIDATION_VOLUMES:
        return "val"
    if vid in TEST_VOLUMES:
        return "test"
    if vid in TRAIN_VOLUMES:
        return "train"
    raise ValueError(f"BTCV volume {vid} is not assigned to the fixed public split")


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
                "image": str(image_out),
                "mask": str(label_out),
                "foreground_pixels": int((label_slice > 0).sum()),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", default="../../data/BTCV/raw/RawData/Training")
    parser.add_argument("--output_root", default="../data_preprocessed/btcv_synapse_medsegv1_binary_png")
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
            raise FileExistsError(
                f"Output root is not empty, pass --overwrite to replace: {output_root}"
            )
        if output_root == output_root.parent or len(output_root.parts) < 3:
            raise ValueError(
                f"Refusing to replace an unsafe output root: {output_root}"
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(img_dir.glob("img*.nii.gz"))
    if not image_paths:
        raise RuntimeError(f"No BTCV images found under {img_dir}")

    expected_volumes = TRAIN_VOLUMES | VALIDATION_VOLUMES | TEST_VOLUMES
    found_volumes = {volume_id(path) for path in image_paths}
    if found_volumes != expected_volumes:
        missing = sorted(expected_volumes - found_volumes)
        unexpected = sorted(found_volumes - expected_volumes)
        raise RuntimeError(
            "BTCV source cases do not match the fixed 30-case public split: "
            f"missing={missing}, unexpected={unexpected}"
        )

    rows = []
    counts = {"train": 0, "val": 0, "test": 0}
    volumes = {"train": set(), "val": set(), "test": set()}
    for image_path in image_paths:
        label_path = label_path_for(image_path, label_dir)
        if not label_path.exists():
            raise FileNotFoundError(f"Missing BTCV label for {image_path}: {label_path}")
        split = split_for(image_path)
        volume_rows = preprocess_volume(image_path, label_path, split, output_root, args.output_size, args.window_mode)
        rows.extend(volume_rows)
        counts[split] += len(volume_rows)
        volumes[split].add(volume_id(image_path))
        print(f"{split}: {image_path.name} -> {len(volume_rows)} slices")

    actual_case_counts = {key: len(value) for key, value in volumes.items()}
    if actual_case_counts != EXPECTED_CASE_COUNTS:
        raise RuntimeError(
            f"BTCV fixed split case-count mismatch: expected={EXPECTED_CASE_COUNTS}, "
            f"actual={actual_case_counts}"
        )
    if counts != EXPECTED_SLICE_COUNTS:
        raise RuntimeError(
            f"BTCV fixed split slice-count mismatch: expected={EXPECTED_SLICE_COUNTS}, "
            f"actual={counts}"
        )

    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "volume", "slice", "image", "mask", "foreground_pixels"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "raw_root": str(training_root),
        "output_root": str(output_root),
        "output_size": args.output_size,
        "window_mode": args.window_mode,
        "split_policy": "fixed_case_level_train_val_test",
        "train_volumes": sorted(TRAIN_VOLUMES),
        "validation_volumes": sorted(VALIDATION_VOLUMES),
        "test_volumes": sorted(TEST_VOLUMES),
        "volumes": {key: sorted(value) for key, value in volumes.items()},
        "case_counts": actual_case_counts,
        "counts": counts,
        "total_slices": len(rows),
        "manifest": str(manifest_path),
        "label_policy": (
            "BTCV binary foreground uses raw labels {1,2,3,4,6,7,8,11}; "
            "background uses raw label 0 plus {5,9,10,12,13}. "
            "Implementation maps to Synapse8 labels, then collapses >0 to foreground."
        ),
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
