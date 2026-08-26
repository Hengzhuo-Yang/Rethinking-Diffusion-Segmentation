"""Validate the release's fixed ID manifests and optional PNG caches."""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_ROOT = RELEASE_ROOT / "code" / "manifests"
SPLITS = ("train", "val", "test")
SPECS = {
    "btcv": {
        "dataset_dir": "BTCV",
        "output_dirs": {"train": "train", "val": "val", "test": "test"},
        "counts": {"train": 18, "val": 2, "test": 10},
        "sample_counts": {"train": 2211, "val": 295, "test": 1273},
        "pattern": r"case\d{4}",
        "sample_pattern": r"vol(?P<owner>\d{4})_\d{3}",
        "fingerprints": {
            "train": "dfdb73f71f36503c06c569760e77ab15fa67cdce84341127e553b327e2219c71",
            "val": "47ed3d1d80a76914ffa3bb75ecc33f40b9b8ae7c113d852fe32f3aff5031efc3",
            "test": "fa5ebbed2ed11e94f6a7dce1c89ff9c03bf4d9cf2e893bf28da9cd0f9f193b3c",
        },
    },
    "acdc": {
        "dataset_dir": "ACDC",
        "output_dirs": {"train": "training", "val": "validation", "test": "testing"},
        "counts": {"train": 70, "val": 10, "test": 20},
        "sample_counts": {"train": 1304, "val": 182, "test": 416},
        "pattern": r"patient\d{3}",
        "sample_pattern": r"(?P<owner>patient\d{3})_frame\d+_z\d{3}",
        "fingerprints": {
            "train": "bfc7abb8764d17f50615d15c26653e6e395205a71aa60712a4e3cbfe4c4946b8",
            "val": "4d1799fbc83e1d6fcdf000f5e5c4d9b4efae840e306bae95fdb8105180a6b6aa",
            "test": "7b2aac91c98e719a7f41300eccf8d622c0fc98754c7b8e6639ce5bbf563dd89b",
        },
    },
    "isic2018": {
        "dataset_dir": "ISIC18",
        "output_dirs": {"train": "training", "val": "validation", "test": "testing"},
        "counts": {"train": 2594, "val": 100, "test": 1000},
        "sample_counts": {"train": 2594, "val": 100, "test": 1000},
        "pattern": r"ISIC_\d{7}",
        "sample_pattern": r"(?P<owner>ISIC_\d{7})",
        "fingerprints": {
            "train": "b6a11fa50b1cf0363160258b5cac96ccae382ea524e34434c3212d1fbb1e4775",
            "val": "44c159239176fe8f63c1efd5f40b5cf66fcafcfd61571c684cd8467f912a366c",
            "test": "6fb4df9f361085d70966ce9098133a6491b756869717af4eaef4ca687686e4f7",
        },
    },
}

BTCV_EXPECTED = {
    "train": {
        "case0005", "case0006", "case0007", "case0009", "case0010", "case0021",
        "case0023", "case0024", "case0026", "case0027", "case0028", "case0030",
        "case0031", "case0033", "case0034", "case0037", "case0039", "case0040",
    },
    "val": {"case0008", "case0001"},
    "test": {
        "case0022", "case0038", "case0036", "case0032", "case0002",
        "case0029", "case0003", "case0004", "case0025", "case0035",
    },
}


class SplitValidationError(RuntimeError):
    """Raised when fixed split evidence is inconsistent."""


def _fingerprint(identifiers: list[str]) -> str:
    canonical = "\n".join(identifiers) + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _read_manifest(dataset: str, split: str) -> list[str]:
    spec = SPECS[dataset]
    path = MANIFEST_ROOT / dataset / f"{split}.txt"
    if not path.is_file():
        raise SplitValidationError(f"Missing {dataset} {split} manifest: {path}")
    identifiers = path.read_text(encoding="utf-8").splitlines()
    for line_number, identifier in enumerate(identifiers, start=1):
        if not identifier or identifier != identifier.strip() or identifier.startswith("#"):
            raise SplitValidationError(
                f"{path}:{line_number} is not a single ID-only manifest line"
            )
        if re.fullmatch(spec["pattern"], identifier) is None:
            raise SplitValidationError(
                f"{path}:{line_number} contains invalid ID {identifier!r}"
            )
    expected_count = spec["counts"][split]
    if len(identifiers) != expected_count:
        raise SplitValidationError(
            f"{dataset} {split} requires {expected_count} IDs, found {len(identifiers)}"
        )
    if len(set(identifiers)) != len(identifiers):
        raise SplitValidationError(f"{dataset} {split} contains duplicate IDs")
    observed_hash = _fingerprint(identifiers)
    expected_hash = spec["fingerprints"][split]
    if observed_hash != expected_hash:
        raise SplitValidationError(
            f"{dataset} {split} IDs/order changed: expected {expected_hash}, found {observed_hash}"
        )
    return identifiers


def validate_release_manifests() -> dict[str, dict[str, list[str]]]:
    """Check exact IDs, counts, duplicates, and cross-partition leakage."""

    result = {}
    for dataset, spec in SPECS.items():
        owners = {}
        split_ids = {}
        for split in SPLITS:
            identifiers = _read_manifest(dataset, split)
            split_ids[split] = identifiers
            for identifier in identifiers:
                previous = owners.get(identifier)
                if previous is not None:
                    raise SplitValidationError(
                        f"{dataset} ID {identifier} leaks across {previous} and {split}"
                    )
                owners[identifier] = split
        expected_total = sum(spec["counts"].values())
        if len(owners) != expected_total:
            raise SplitValidationError(
                f"{dataset} requires {expected_total} unique IDs, found {len(owners)}"
            )
        if dataset == "btcv":
            for split in SPLITS:
                if set(split_ids[split]) != BTCV_EXPECTED[split]:
                    raise SplitValidationError(f"BTCV {split} case allocation changed")
        if dataset == "acdc":
            expected = {f"patient{number:03d}" for number in range(1, 101)}
            if set(owners) != expected:
                raise SplitValidationError(
                    "ACDC manifests must partition patient001 through patient100"
                )
        result[dataset] = split_ids
    return result


def _resolve_dataset_root(dataset: str, root: Path) -> Path:
    root = root.expanduser().resolve()
    nested = root / SPECS[dataset]["dataset_dir"]
    return nested if nested.is_dir() else root


def _sample_owner(dataset: str, stem: str) -> str:
    match = re.fullmatch(SPECS[dataset]["sample_pattern"], stem)
    if match is None:
        raise SplitValidationError(f"Unexpected {dataset} PNG stem: {stem}")
    owner = match.group("owner")
    return f"case{owner}" if dataset == "btcv" else owner


def validate_preprocessed_root(
    dataset: str,
    root: Path,
    manifests: dict[str, list[str]],
) -> dict[str, int]:
    """Check cache pairings, expected sample counts, and owner isolation."""

    if dataset not in SPECS:
        raise SplitValidationError(f"Unknown dataset {dataset!r}")
    spec = SPECS[dataset]
    dataset_root = _resolve_dataset_root(dataset, root)
    if not dataset_root.is_dir():
        raise SplitValidationError(f"Missing {dataset} cache root: {dataset_root}")

    observed_counts = {}
    for split in SPLITS:
        split_root = dataset_root / spec["output_dirs"][split]
        image_dir = split_root / "images"
        mask_dir = split_root / "masks"
        if not image_dir.is_dir() or not mask_dir.is_dir():
            raise SplitValidationError(f"Missing {dataset} {split} images/masks directories")
        image_stems = {
            path.stem for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() == ".png"
        }
        mask_stems = {
            path.stem for path in mask_dir.iterdir() if path.is_file() and path.suffix.lower() == ".png"
        }
        if image_stems != mask_stems:
            missing_masks = sorted(image_stems - mask_stems)
            missing_images = sorted(mask_stems - image_stems)
            raise SplitValidationError(
                f"{dataset} {split} image/mask mismatch: "
                f"missing_masks={missing_masks[:5]}, missing_images={missing_images[:5]}"
            )
        expected_samples = spec["sample_counts"][split]
        if len(image_stems) != expected_samples:
            raise SplitValidationError(
                f"{dataset} {split} requires {expected_samples} samples, found {len(image_stems)}"
            )
        observed_owners = {_sample_owner(dataset, stem) for stem in image_stems}
        expected_owners = set(manifests[split])
        if observed_owners != expected_owners:
            missing = sorted(expected_owners - observed_owners)
            unexpected = sorted(observed_owners - expected_owners)
            raise SplitValidationError(
                f"{dataset} {split} cache does not match manifest: "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}"
            )
        if dataset == "acdc":
            label_map_dir = split_root / "label_maps"
            label_stems = {
                path.stem
                for path in label_map_dir.iterdir()
                if path.is_file() and path.suffix.lower() == ".png"
            } if label_map_dir.is_dir() else set()
            if label_stems != image_stems:
                raise SplitValidationError(f"ACDC {split} label maps do not match images")
        observed_counts[split] = len(image_stems)
    return observed_counts


def _parse_data_roots(values: list[str]) -> dict[str, Path]:
    roots = {}
    for value in values:
        dataset, separator, path = value.partition("=")
        dataset = dataset.strip().lower()
        if not separator or dataset not in SPECS or not path.strip():
            raise SplitValidationError(
                f"Invalid --data-root {value!r}; expected one of "
                f"{', '.join(SPECS)}=PATH"
            )
        if dataset in roots:
            raise SplitValidationError(f"Duplicate --data-root for {dataset}")
        roots[dataset] = Path(path.strip())
    return roots


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        action="append",
        default=[],
        metavar="DATASET=PATH",
        help="Optionally validate a generated PNG cache; may be repeated.",
    )
    args = parser.parse_args(argv)
    try:
        manifests = validate_release_manifests()
        for dataset, root in _parse_data_roots(args.data_root).items():
            validate_preprocessed_root(dataset, root, manifests[dataset])
    except (OSError, SplitValidationError) as error:
        print(f"split validation failed: {error}", file=sys.stderr)
        return 1
    for dataset, spec in SPECS.items():
        counts = spec["counts"]
        print(f"{dataset}: train={counts['train']} val={counts['val']} test={counts['test']}")
    print("split validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
