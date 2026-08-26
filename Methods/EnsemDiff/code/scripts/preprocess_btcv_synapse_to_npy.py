"""Preprocess the fixed BTCV/Synapse train/validation/test partitions."""

import argparse
import csv
import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image


DEFAULT_MANIFEST_DIR = Path(__file__).resolve().parents[1] / "manifests" / "btcv"
MANIFEST_FILES = {
    "training": "train_cases.txt",
    "validation": "val_cases.txt",
    "testing": "test_cases.txt",
}
EXPECTED_CASE_COUNTS = {"training": 18, "validation": 2, "testing": 10}
EXPECTED_SLICE_COUNTS = {"training": 2211, "validation": 295, "testing": 1273}
CASE_ID_RE = re.compile(r"case(\d{4})$")


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
    return f"{int(match.group(1)):04d}"


def load_split_manifests(manifest_dir):
    manifest_dir = Path(manifest_dir).expanduser().resolve()
    split_cases = {}
    owner = {}

    for split, filename in MANIFEST_FILES.items():
        manifest_path = manifest_dir / filename
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing BTCV {split} manifest: {manifest_path}")

        cases = []
        for line_number, raw_line in enumerate(
            manifest_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            case_id = raw_line.strip()
            if not case_id or case_id.startswith("#"):
                continue
            match = CASE_ID_RE.fullmatch(case_id)
            if match is None:
                raise ValueError(
                    f"Invalid BTCV case ID at {manifest_path}:{line_number}: {case_id!r}"
                )
            volume = match.group(1)
            if volume in owner:
                raise ValueError(
                    f"BTCV case {case_id} appears in both {owner[volume]} and {split}"
                )
            owner[volume] = split
            cases.append(case_id)

        expected = EXPECTED_CASE_COUNTS[split]
        if len(cases) != expected:
            raise ValueError(
                f"BTCV {split} manifest must contain {expected} cases, found {len(cases)}"
            )
        split_cases[split] = cases

    if len(owner) != sum(EXPECTED_CASE_COUNTS.values()):
        raise ValueError("BTCV manifests must contain 30 unique cases")
    return split_cases, owner


def validate_raw_cases(image_paths, case_owner):
    images_by_volume = {}
    for image_path in image_paths:
        vid = volume_id(image_path)
        if vid in images_by_volume:
            raise ValueError(
                f"Multiple BTCV images resolve to case{vid}: "
                f"{images_by_volume[vid]} and {image_path}"
            )
        images_by_volume[vid] = image_path

    expected = set(case_owner)
    available = set(images_by_volume)
    missing = sorted(expected - available)
    unexpected = sorted(available - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing=" + ",".join(f"case{vid}" for vid in missing))
        if unexpected:
            details.append("unexpected=" + ",".join(f"case{vid}" for vid in unexpected))
        raise ValueError("BTCV raw cases do not match the fixed manifests: " + "; ".join(details))
    return images_by_volume


def prepare_output_root(output_root):
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    legacy = [
        path
        for path in output_root.iterdir()
        if "10pct" in path.name.lower() or "quick" in path.name.lower()
    ]
    if legacy:
        names = ", ".join(path.name for path in sorted(legacy))
        raise RuntimeError(
            "Refusing to preprocess beside legacy BTCV 10pct/quick partitions: " + names
        )

    populated = []
    for split in MANIFEST_FILES:
        split_dir = output_root / split
        if split_dir.exists() and any(split_dir.iterdir()):
            populated.append(split_dir)
    if populated:
        names = ", ".join(str(path) for path in populated)
        raise RuntimeError(
            "BTCV output split directories must be empty before preprocessing: " + names
        )

    for split in MANIFEST_FILES:
        (output_root / split).mkdir(parents=True, exist_ok=True)
    return output_root


def label_path_for(image_path, label_dir):
    return label_dir / image_path.name.replace("img", "label", 1)


def normalize_ct(image, mode):
    image = image.astype(np.float32, copy=False)
    if mode == "sdseg_2d":
        image = np.clip(image, -175.0, 250.0)
        return (((image + 125.0) / 400.0) * 2.0) - 1.0
    if mode == "synapse_window":
        image = np.clip(image, -125.0, 275.0)
        return (((image + 125.0) / 400.0) * 2.0) - 1.0
    raise ValueError(f"Unknown window mode: {mode}")


def transfer_to_9(gts):
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
    image = Image.fromarray(slice_2d.astype(np.float32))
    image = image.resize((output_size, output_size), resample=BILINEAR)
    return np.asarray(image, dtype=np.float32)


def resize_label(slice_2d, output_size):
    image = Image.fromarray(slice_2d.astype(np.uint8))
    image = image.resize((output_size, output_size), resample=NEAREST)
    return np.asarray(image, dtype=np.uint8)


def preprocess_volume(image_path, label_path, split, output_root, output_size, window_mode):
    image = nib.load(str(image_path)).get_fdata()
    label = nib.load(str(label_path)).get_fdata()
    if image.shape != label.shape:
        raise ValueError(f"Shape mismatch for {image_path.name}: {image.shape} vs {label.shape}")

    vid = volume_id(image_path)
    split_dir = output_root / split
    rows = []
    for slice_index in range(image.shape[2]):
        slice_id = f"vol{vid}_{slice_index:03d}"
        slice_dir = split_dir / slice_id
        slice_dir.mkdir(parents=True, exist_ok=True)

        image_slice = normalize_ct(image[:, :, slice_index], window_mode)
        image_slice = resize_image(image_slice, output_size)

        label_slice = resize_label(label[:, :, slice_index], output_size)
        label_slice = transfer_to_9(label_slice)
        label_slice = np.where(label_slice > 0, 1, 0).astype(np.uint8)

        image_out = slice_dir / f"{slice_id}_image.npy"
        label_out = slice_dir / f"{slice_id}_seg.npy"
        np.save(image_out, image_slice.astype(np.float32))
        np.save(label_out, label_slice)

        rows.append(
            {
                "split": split,
                "volume": vid,
                "slice": slice_index,
                "image": image_out.relative_to(output_root).as_posix(),
                "label": label_out.relative_to(output_root).as_posix(),
                "foreground_pixels": int(label_slice.sum()),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_root",
        required=True,
        help="Path to BTCV RawData/Training or any parent containing it.",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Empty destination for the training/validation/testing directories.",
    )
    parser.add_argument(
        "--manifest_dir",
        default=DEFAULT_MANIFEST_DIR,
        help=(
            "Directory containing train_cases.txt, val_cases.txt, and test_cases.txt "
            "(default: repository code/manifests/btcv)."
        ),
    )
    parser.add_argument("--output_size", type=int, default=224)
    parser.add_argument(
        "--window_mode",
        choices=["sdseg_2d", "synapse_window"],
        default="sdseg_2d",
        help="sdseg_2d matches SDSeg data/synapse/nii2format.py.",
    )
    args = parser.parse_args()

    split_cases, case_owner = load_split_manifests(args.manifest_dir)
    training_root = find_training_root(args.raw_root)
    img_dir = training_root / "img"
    label_dir = training_root / "label"
    output_root = prepare_output_root(args.output_root)

    image_paths = sorted(img_dir.glob("img*.nii.gz"))
    if not image_paths:
        raise RuntimeError(f"No BTCV images found under {img_dir}")
    images_by_volume = validate_raw_cases(image_paths, case_owner)

    rows = []
    counts = {split: 0 for split in MANIFEST_FILES}
    for vid in sorted(images_by_volume):
        image_path = images_by_volume[vid]
        label_path = label_path_for(image_path, label_dir)
        if not label_path.exists():
            raise FileNotFoundError(f"Missing BTCV label for {image_path}: {label_path}")
        split = case_owner[vid]
        volume_rows = preprocess_volume(
            image_path=image_path,
            label_path=label_path,
            split=split,
            output_root=output_root,
            output_size=args.output_size,
            window_mode=args.window_mode,
        )
        rows.extend(volume_rows)
        counts[split] += len(volume_rows)
        print(f"{split}: {image_path.name} -> {len(volume_rows)} slices")

    if counts != EXPECTED_SLICE_COUNTS:
        raise ValueError(
            "BTCV slice counts do not match the fixed release split: "
            f"expected={EXPECTED_SLICE_COUNTS}, observed={counts}"
        )

    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "volume", "slice", "image", "label", "foreground_pixels"],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "output_size": args.output_size,
        "window_mode": args.window_mode,
        "split_cases": split_cases,
        "case_counts": EXPECTED_CASE_COUNTS,
        "counts": counts,
        "total_slices": len(rows),
        "manifest": manifest_path.name,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
