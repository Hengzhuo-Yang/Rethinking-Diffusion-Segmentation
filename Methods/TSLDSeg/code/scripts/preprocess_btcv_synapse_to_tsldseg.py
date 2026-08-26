"""Preprocess BTCV/Synapse NIfTI volumes into the fixed TSLDSeg PNG splits.

Release modification derived from the locally validated preprocessing lineage;
see MODIFICATIONS.md, DATA_SPLITS.md, and THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image


SPLIT_IDS = {
    "train": {
        "0005", "0006", "0007", "0009", "0010", "0021", "0023", "0024",
        "0026", "0027", "0028", "0030", "0031", "0033", "0034", "0037",
        "0039", "0040",
    },
    "val": {"0008", "0001"},
    "test": {
        "0022", "0038", "0036", "0032", "0002", "0029", "0003", "0004",
        "0025", "0035",
    },
}
EXPECTED_SLICE_COUNTS = {"train": 2211, "val": 295, "test": 1273}


def _resampling():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.BILINEAR, Image.Resampling.NEAREST
    return Image.BILINEAR, Image.NEAREST


BILINEAR, NEAREST = _resampling()


def find_training_root(raw_root: str | Path) -> Path:
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
        f"Could not find BTCV RawData/Training/img and label under {root}"
    )


def volume_id(path: Path) -> str:
    match = re.search(r"(\d+)", path.name)
    if match is None:
        raise ValueError(f"Cannot parse BTCV volume ID from {path.name}")
    return f"{int(match.group(1)):04d}"


def split_for(path: Path) -> str:
    identifier = volume_id(path)
    matches = [split for split, identifiers in SPLIT_IDS.items() if identifier in identifiers]
    if len(matches) != 1:
        raise ValueError(f"BTCV volume {identifier} is not assigned to exactly one fixed split")
    return matches[0]


def normalize_ct_to_uint8(image: np.ndarray, window_mode: str) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    if window_mode == "sdseg_2d":
        low, high = -175.0, 250.0
    elif window_mode == "synapse_window":
        low, high = -125.0, 275.0
    else:
        raise ValueError(f"Unknown window mode: {window_mode}")
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


def resize(array: np.ndarray, size: int, resample: int) -> np.ndarray:
    image = Image.fromarray(array.astype(np.uint8))
    return np.asarray(image.resize((size, size), resample=resample), dtype=np.uint8)


def preprocess_volume(
    image_path: Path,
    label_path: Path,
    split: str,
    output_root: Path,
    output_size: int,
    window_mode: str,
) -> list[dict[str, object]]:
    image = nib.load(str(image_path)).get_fdata()
    label = nib.load(str(label_path)).get_fdata()
    if image.shape != label.shape:
        raise ValueError(f"Shape mismatch for {image_path.name}: {image.shape} vs {label.shape}")

    identifier = volume_id(image_path)
    # The 12 held-out cases share one physical pool. ID manifests enforce the
    # logical 2-case validation / 10-case test partition without duplicating PNGs.
    physical_split = "train" if split == "train" else "test"
    image_dir = output_root / "BTCV" / physical_split / "images"
    mask_dir = output_root / "BTCV" / physical_split / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for slice_index in range(image.shape[2]):
        sample_id = f"vol{identifier}_{slice_index:03d}"
        image_slice = resize(
            normalize_ct_to_uint8(image[:, :, slice_index], window_mode),
            output_size,
            BILINEAR,
        )
        image_rgb = np.repeat(image_slice[:, :, None], 3, axis=2)
        label_slice = transfer_to_synapse8(
            resize(label[:, :, slice_index], output_size, NEAREST)
        )
        mask = np.where(label_slice > 0, 255, 0).astype(np.uint8)
        image_out = image_dir / f"{sample_id}.png"
        mask_out = mask_dir / f"{sample_id}.png"
        Image.fromarray(image_rgb, mode="RGB").save(image_out)
        Image.fromarray(mask, mode="L").save(mask_out)
        rows.append(
            {
                "split": split,
                "case_id": f"case{identifier}",
                "slice_index": slice_index,
                "sample_id": sample_id,
                "image": image_out.relative_to(output_root).as_posix(),
                "mask": mask_out.relative_to(output_root).as_posix(),
                "foreground_pixels": int((mask > 0).sum()),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--output_size", type=int, default=256)
    parser.add_argument(
        "--window_mode", choices=["sdseg_2d", "synapse_window"], default="sdseg_2d"
    )
    args = parser.parse_args()

    training_root = find_training_root(args.raw_root)
    output_root = Path(args.output_root).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output root must be absent or empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    image_paths = sorted((training_root / "img").glob("img*.nii.gz"))
    observed_ids = {volume_id(path) for path in image_paths}
    expected_ids = set().union(*SPLIT_IDS.values())
    if observed_ids != expected_ids:
        raise ValueError(
            "BTCV raw case IDs differ from the fixed 30-case partition: "
            f"missing={sorted(expected_ids - observed_ids)}, "
            f"unexpected={sorted(observed_ids - expected_ids)}"
        )

    rows: list[dict[str, object]] = []
    for image_path in image_paths:
        label_path = training_root / "label" / image_path.name.replace("img", "label", 1)
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing BTCV label: {label_path}")
        split = split_for(image_path)
        volume_rows = preprocess_volume(
            image_path, label_path, split, output_root, args.output_size, args.window_mode
        )
        rows.extend(volume_rows)
        print(f"{split}: {image_path.name} -> {len(volume_rows)} slices")

    counts = {split: sum(row["split"] == split for row in rows) for split in SPLIT_IDS}
    if counts != EXPECTED_SLICE_COUNTS:
        raise ValueError(
            f"Generated BTCV slice counts changed: expected {EXPECTED_SLICE_COUNTS}, found {counts}"
        )
    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "dataset": "BTCV/Synapse binary 2D",
        "output_size": args.output_size,
        "window_mode": args.window_mode,
        "fixed_case_ids": {
            split: [f"case{identifier}" for identifier in sorted(identifiers)]
            for split, identifiers in SPLIT_IDS.items()
        },
        "slice_counts": counts,
        "manifest": manifest_path.name,
        "label_policy": "Synapse 8-organ mapping collapsed to foreground/background",
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
