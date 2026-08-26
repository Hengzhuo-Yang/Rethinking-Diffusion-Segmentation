"""Validate public split manifests and, optionally, prepared dataset trees."""

import argparse
import json
import re
from pathlib import Path


EXPECTED = {
    "btcv": {"train": (18, 2211), "val": (2, 295), "test": (10, 1273)},
    "acdc": {"train": (70, 1304), "val": (10, 182), "test": (20, 416)},
    "isic2018": {"train": (2594, 2594), "val": (100, 100), "test": (1000, 1000)},
}

MANIFEST_NAMES = {
    "btcv": {"train": "train_cases.txt", "val": "val_cases.txt", "test": "test_cases.txt"},
    "acdc": {"train": "train_subjects.txt", "val": "val_subjects.txt", "test": "test_subjects.txt"},
    "isic2018": {"train": "train_images.txt", "val": "val_images.txt", "test": "test_images.txt"},
}


def read_ids(path):
    values = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(values) != len(set(values)):
        raise AssertionError(f"Duplicate IDs in {path}")
    return values


def load_manifests(root):
    result = {}
    for dataset, split_names in MANIFEST_NAMES.items():
        result[dataset] = {}
        for split, filename in split_names.items():
            ids = read_ids(Path(root) / dataset / filename)
            expected_units = EXPECTED[dataset][split][0]
            if len(ids) != expected_units:
                raise AssertionError(
                    f"{dataset}/{split}: expected {expected_units} IDs, found {len(ids)}"
                )
            result[dataset][split] = ids
        sets = {split: set(ids) for split, ids in result[dataset].items()}
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = sets[left] & sets[right]
            if overlap:
                raise AssertionError(f"{dataset}: {left}/{right} leakage: {sorted(overlap)[:5]}")
    return result


def paired_pngs(directory, allowed_ids, id_from_stem):
    directory = Path(directory)
    images = directory / "images"
    masks = directory / "masks"
    if not images.is_dir() or not masks.is_dir():
        raise FileNotFoundError(f"Expected images/ and masks/ under {directory}")
    selected = []
    missing_masks = []
    observed_ids = set()
    allowed = set(allowed_ids)
    for image in sorted(images.glob("*.png")):
        unit_id = id_from_stem(image.stem)
        if unit_id not in allowed:
            continue
        observed_ids.add(unit_id)
        mask = masks / image.name
        if not mask.is_file():
            missing_masks.append(image.name)
        selected.append(image)
    if missing_masks:
        raise AssertionError(f"Missing masks under {directory}: {missing_masks[:3]}")
    if observed_ids != allowed:
        missing = sorted(allowed - observed_ids)
        extra = sorted(observed_ids - allowed)
        raise AssertionError(f"Manifest/data ID mismatch under {directory}: missing={missing[:5]} extra={extra[:5]}")
    return len(selected)


def btcv_id(stem):
    match = re.match(r"vol(\d{4})_", stem)
    if not match:
        raise AssertionError(f"Unexpected BTCV slice name: {stem}")
    return match.group(1)


def acdc_id(stem):
    match = re.match(r"(patient\d{3})_", stem)
    if not match:
        raise AssertionError(f"Unexpected ACDC slice name: {stem}")
    return match.group(1)


def resolve(root, candidates):
    root = Path(root)
    for candidate in candidates:
        path = root / candidate
        if path.is_dir():
            return path
    raise FileNotFoundError(f"None of the expected dataset paths exist under {root}: {candidates}")


def validate_btcv(root, manifests):
    train_dir = resolve(root, ("BTCV/train", "BTCV/training", "train", "training"))
    heldout_dir = resolve(root, ("BTCV/test", "BTCV/testing", "test", "testing"))
    val_dir = resolve(root, ("BTCV/val", "BTCV/validation", str(heldout_dir.relative_to(root))))
    test_dir = resolve(root, ("BTCV/test", "BTCV/testing", "test", "testing"))
    dirs = {"train": train_dir, "val": val_dir, "test": test_dir}
    return {
        split: paired_pngs(dirs[split], manifests[split], btcv_id)
        for split in ("train", "val", "test")
    }


def validate_acdc(root, manifests):
    dirs = {
        "train": resolve(root, ("training", "train", "ACDC/training", "ACDC/train")),
        "val": resolve(root, ("validation", "val", "ACDC/validation", "ACDC/val")),
        "test": resolve(root, ("testing", "test", "ACDC/testing", "ACDC/test")),
    }
    return {
        split: paired_pngs(dirs[split], manifests[split], acdc_id)
        for split in ("train", "val", "test")
    }


def validate_isic(root, manifests):
    dirs = {
        "train": resolve(root, ("ISIC18/training", "training", "ISIC18/train", "train")),
        "val": resolve(root, ("ISIC18/validation", "validation", "ISIC18/val", "val")),
        "test": resolve(root, ("ISIC18/testing", "testing", "ISIC18/test", "test")),
    }
    return {
        split: paired_pngs(dirs[split], manifests[split], lambda stem: stem)
        for split in ("train", "val", "test")
    }


def check_sample_counts(dataset, counts):
    for split, count in counts.items():
        expected = EXPECTED[dataset][split][1]
        if count != expected:
            raise AssertionError(f"{dataset}/{split}: expected {expected} samples, found {count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", type=Path, default=Path(__file__).resolve().parents[1] / "manifests")
    parser.add_argument("--btcv-root", type=Path)
    parser.add_argument("--acdc-root", type=Path)
    parser.add_argument("--isic2018-root", type=Path)
    args = parser.parse_args()

    manifests = load_manifests(args.manifest_root)
    report = {
        dataset: {
            split: {"units": len(ids), "expected_samples": EXPECTED[dataset][split][1]}
            for split, ids in splits.items()
        }
        for dataset, splits in manifests.items()
    }
    if args.btcv_root:
        counts = validate_btcv(args.btcv_root, manifests["btcv"])
        check_sample_counts("btcv", counts)
        for split, count in counts.items():
            report["btcv"][split]["observed_samples"] = count
    if args.acdc_root:
        counts = validate_acdc(args.acdc_root, manifests["acdc"])
        check_sample_counts("acdc", counts)
        for split, count in counts.items():
            report["acdc"][split]["observed_samples"] = count
    if args.isic2018_root:
        counts = validate_isic(args.isic2018_root, manifests["isic2018"])
        check_sample_counts("isic2018", counts)
        for split, count in counts.items():
            report["isic2018"][split]["observed_samples"] = count
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
