"""Build public ID-only split manifests from local preprocessing manifests."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


BTCV_VAL_CASES = ("0008", "0001")
BTCV_TEST_CASES = (
    "0022",
    "0038",
    "0036",
    "0032",
    "0002",
    "0029",
    "0003",
    "0004",
    "0025",
    "0035",
)

EXPECTED = {
    "btcv": {
        "train": {"units": 18, "samples": 2211},
        "val": {"units": 2, "samples": 295},
        "test": {"units": 10, "samples": 1273},
    },
    "acdc": {
        "train": {"units": 70, "samples": 1304},
        "val": {"units": 10, "samples": 182},
        "test": {"units": 20, "samples": 416},
    },
    "isic2018": {
        "train": {"units": 2594, "samples": 2594},
        "val": {"units": 100, "samples": 100},
        "test": {"units": 1000, "samples": 1000},
    },
}


def read_rows(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_ids(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


def require_counts(dataset, split, units, samples):
    expected = EXPECTED[dataset][split]
    actual = {"units": len(units), "samples": samples}
    if actual != expected:
        raise ValueError(
            f"{dataset}/{split} mismatch: expected {expected}, observed {actual}"
        )


def build_btcv(path, out_root):
    rows = [row for row in read_rows(path) if row["split"] in {"train", "test"}]
    by_volume = Counter(row["volume"].zfill(4) for row in rows)
    train_cases = sorted({row["volume"].zfill(4) for row in rows if row["split"] == "train"})
    val_cases = list(BTCV_VAL_CASES)
    test_cases = list(BTCV_TEST_CASES)
    if set(val_cases) & set(test_cases):
        raise ValueError("BTCV validation and test cases overlap")
    if set(train_cases) & (set(val_cases) | set(test_cases)):
        raise ValueError("BTCV training cases overlap the held-out pool")
    split_cases = {"train": train_cases, "val": val_cases, "test": test_cases}
    result = {}
    for split, cases in split_cases.items():
        samples = sum(by_volume[case] for case in cases)
        require_counts("btcv", split, cases, samples)
        write_ids(out_root / "btcv" / f"{split}_cases.txt", cases)
        result[split] = {"ids": cases, "units": len(cases), "samples": samples}
    return result


def build_acdc(path, out_root):
    source_to_release = {"training": "train", "validation": "val", "testing": "test"}
    grouped = defaultdict(list)
    for row in read_rows(path):
        grouped[source_to_release[row["split"]]].append(row)
    result = {}
    for split in ("train", "val", "test"):
        rows = grouped[split]
        subjects = sorted({row["patient_id"] for row in rows})
        require_counts("acdc", split, subjects, len(rows))
        write_ids(out_root / "acdc" / f"{split}_subjects.txt", subjects)
        result[split] = {"ids": subjects, "units": len(subjects), "samples": len(rows)}
    return result


def build_isic(path, out_root):
    source_to_release = {"training": "train", "validation": "val", "testing": "test"}
    grouped = defaultdict(list)
    for row in read_rows(path):
        if row["split"] in source_to_release:
            grouped[source_to_release[row["split"]]].append(row["image_id"])
    result = {}
    for split in ("train", "val", "test"):
        image_ids = sorted(grouped[split])
        if len(image_ids) != len(set(image_ids)):
            raise ValueError(f"Duplicate ISIC2018 image IDs in {split}")
        require_counts("isic2018", split, image_ids, len(image_ids))
        write_ids(out_root / "isic2018" / f"{split}_images.txt", image_ids)
        result[split] = {
            "ids": image_ids,
            "units": len(image_ids),
            "samples": len(image_ids),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--btcv-manifest", type=Path, required=True)
    parser.add_argument("--acdc-manifest", type=Path, required=True)
    parser.add_argument("--isic2018-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "role_policy": {
            "train": "gradient updates and training statistics only",
            "val": "checkpoint selection only",
            "test": "one final evaluation after loading the validation-selected checkpoint",
        },
        "datasets": {
            "btcv": build_btcv(args.btcv_manifest, args.output_root),
            "acdc": build_acdc(args.acdc_manifest, args.output_root),
            "isic2018": build_isic(args.isic2018_manifest, args.output_root),
        },
    }
    (args.output_root / "split_spec.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: {s: {"units": v["units"], "samples": v["samples"]} for s, v in d.items()} for k, d in summary["datasets"].items()}, indent=2))


if __name__ == "__main__":
    main()
