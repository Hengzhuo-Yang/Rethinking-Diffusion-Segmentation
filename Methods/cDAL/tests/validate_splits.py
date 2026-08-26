import argparse
import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

EXPECTED = {
    "btcv": {
        "train": (18, 2211),
        "validation": (2, 295),
        "test": (10, 1273),
    },
    "acdc": {
        "train": (70, 1304),
        "validation": (10, 182),
        "test": (20, 416),
    },
    "isic2018": {
        "train": (2594, 2594),
        "validation": (100, 100),
        "test": (1000, 1000),
    },
}

MANIFESTS = {
    "btcv": {
        "train": "manifests/btcv/train_cases.txt",
        "validation": "manifests/btcv/validation_cases.txt",
        "test": "manifests/btcv/test_cases.txt",
    },
    "acdc": {
        "train": "manifests/acdc/training_subjects.txt",
        "validation": "manifests/acdc/validation_subjects.txt",
        "test": "manifests/acdc/test_subjects.txt",
    },
    "isic2018": {
        "train": "manifests/isic2018/training_images.txt",
        "validation": "manifests/isic2018/validation_images.txt",
        "test": "manifests/isic2018/test_images.txt",
    },
}


def read_manifest(relative_path):
    path = REPO_ROOT / relative_path
    values = [
        Path(line.strip()).stem
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate entries in {relative_path}")
    return values


def validate_manifest_sets(dataset):
    values = {role: read_manifest(path) for role, path in MANIFESTS[dataset].items()}
    for role, items in values.items():
        expected_ids = EXPECTED[dataset][role][0]
        if len(items) != expected_ids:
            raise ValueError(f"{dataset} {role}: expected {expected_ids} IDs, found {len(items)}")
    roles = list(values)
    for index, left in enumerate(roles):
        for right in roles[index + 1:]:
            overlap = set(values[left]) & set(values[right])
            if overlap:
                raise ValueError(f"{dataset} leakage between {left} and {right}: {sorted(overlap)[:5]}")
    return values


def paired_png_names(root):
    image_dir = root / "images"
    mask_dir = root / "masks"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Missing images/masks directories under {root}")
    images = {path.name for path in image_dir.glob("*.png")}
    masks = {path.name for path in mask_dir.glob("*.png")}
    if images != masks:
        raise ValueError(
            f"Image/mask mismatch under {root.name}: images_only={len(images - masks)}, masks_only={len(masks - images)}"
        )
    return images


def validate_btcv(root, manifests):
    dataset_root = root / "BTCV" if (root / "BTCV").is_dir() else root
    train_names = paired_png_names(dataset_root / "train")
    heldout_dir = dataset_root / "heldout"
    if not heldout_dir.is_dir():
        heldout_dir = dataset_root / "test"
    heldout_names = paired_png_names(heldout_dir)

    def ids(items):
        return {re.search(r"(\d+)", item).group(1).zfill(4) for item in items}

    expected_ids = {
        role: ids(items)
        for role, items in manifests.items()
    }
    names_by_role = {
        "train": {name for name in train_names if re.match(r"vol(\d{4})", Path(name).stem).group(1) in expected_ids["train"]},
        "validation": {name for name in heldout_names if re.match(r"vol(\d{4})", Path(name).stem).group(1) in expected_ids["validation"]},
        "test": {name for name in heldout_names if re.match(r"vol(\d{4})", Path(name).stem).group(1) in expected_ids["test"]},
    }
    return names_by_role, (dataset_root / "test_10pct_seed23").exists()


def validate_acdc(root, manifests):
    dataset_root = root / "ACDC" if (root / "ACDC").is_dir() else root
    physical = {"train": "training", "validation": "validation", "test": "testing"}
    result = {}
    for role, dirname in physical.items():
        names = paired_png_names(dataset_root / dirname)
        allowed = set(manifests[role])
        result[role] = {name for name in names if name.split("_frame", 1)[0] in allowed}
    return result


def validate_isic(root, manifests):
    dataset_root = root / "ISIC18" if (root / "ISIC18").is_dir() else root
    physical = {"train": "training", "validation": "validation", "test": "testing"}
    result = {}
    for role, dirname in physical.items():
        names = paired_png_names(dataset_root / dirname)
        allowed = set(manifests[role])
        result[role] = {name for name in names if Path(name).stem in allowed}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--btcv-root")
    parser.add_argument("--acdc-root")
    parser.add_argument("--isic-root")
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    report = {"status": "PASS", "datasets": {}}
    roots = {"btcv": args.btcv_root, "acdc": args.acdc_root, "isic2018": args.isic_root}
    for dataset in EXPECTED:
        manifests = validate_manifest_sets(dataset)
        dataset_report = {
            "manifest_counts": {role: len(values) for role, values in manifests.items()},
            "manifest_disjoint": True,
            "data_checked": bool(roots[dataset]),
        }
        if roots[dataset]:
            if dataset == "btcv":
                names, old_quick_subset_present = validate_btcv(Path(roots[dataset]), manifests)
                dataset_report["legacy_test_10pct_present_but_excluded"] = old_quick_subset_present
            elif dataset == "acdc":
                names = validate_acdc(Path(roots[dataset]), manifests)
            else:
                names = validate_isic(Path(roots[dataset]), manifests)
            slice_counts = {role: len(items) for role, items in names.items()}
            for role, count in slice_counts.items():
                expected = EXPECTED[dataset][role][1]
                if count != expected:
                    raise ValueError(f"{dataset} {role}: expected {expected} samples, found {count}")
            dataset_report["sample_counts"] = slice_counts
            dataset_report["image_mask_pairs_complete"] = True
        report["datasets"][dataset] = dataset_report

    if args.json:
        output = Path(args.json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
