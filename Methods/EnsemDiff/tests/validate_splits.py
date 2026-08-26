#!/usr/bin/env python3
"""Validate fixed release manifests and optional preprocessed dataset roots."""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_ROOT = RELEASE_ROOT / "code" / "manifests"
SPLITS = ("training", "validation", "testing")

SPECS = {
    "btcv": {
        "manifest_dir": "btcv",
        "manifest_files": {
            "training": "train_cases.txt",
            "validation": "val_cases.txt",
            "testing": "test_cases.txt",
        },
        "id_pattern": r"case\d{4}",
        "id_counts": {"training": 18, "validation": 2, "testing": 10},
        "slice_counts": {"training": 2211, "validation": 295, "testing": 1273},
        "fingerprints": {
            "training": "dfdb73f71f36503c06c569760e77ab15fa67cdce84341127e553b327e2219c71",
            "validation": "47ed3d1d80a76914ffa3bb75ecc33f40b9b8ae7c113d852fe32f3aff5031efc3",
            "testing": "fa5ebbed2ed11e94f6a7dce1c89ff9c03bf4d9cf2e893bf28da9cd0f9f193b3c",
        },
        "member_pattern": r"vol(?P<number>\d{4})_\d{3}",
        "required_suffixes": ("image", "seg"),
    },
    "acdc": {
        "manifest_dir": "acdc",
        "manifest_files": {
            "training": "train_patients.txt",
            "validation": "val_patients.txt",
            "testing": "test_patients.txt",
        },
        "id_pattern": r"patient\d{3}",
        "id_counts": {"training": 70, "validation": 10, "testing": 20},
        "slice_counts": {"training": 1304, "validation": 182, "testing": 416},
        "fingerprints": {
            "training": "bfc7abb8764d17f50615d15c26653e6e395205a71aa60712a4e3cbfe4c4946b8",
            "validation": "4d1799fbc83e1d6fcdf000f5e5c4d9b4efae840e306bae95fdb8105180a6b6aa",
            "testing": "7b2aac91c98e719a7f41300eccf8d622c0fc98754c7b8e6639ce5bbf563dd89b",
        },
        "member_pattern": r"(?P<owner>patient\d{3})_frame\d+_z\d{3}",
        "required_suffixes": ("image", "seg", "label"),
    },
    "isic2018": {
        "manifest_dir": "isic2018",
        "manifest_files": {
            "training": "training.txt",
            "validation": "validation.txt",
            "testing": "testing.txt",
        },
        "id_pattern": r"ISIC_\d{7}",
        "id_counts": {"training": 2594, "validation": 100, "testing": 1000},
        "slice_counts": {"training": 2594, "validation": 100, "testing": 1000},
        "fingerprints": {
            "training": "b6a11fa50b1cf0363160258b5cac96ccae382ea524e34434c3212d1fbb1e4775",
            "validation": "44c159239176fe8f63c1efd5f40b5cf66fcafcfd61571c684cd8467f912a366c",
            "testing": "6fb4df9f361085d70966ce9098133a6491b756869717af4eaef4ca687686e4f7",
        },
        "member_pattern": r"(?P<owner>ISIC_\d{7})",
        "required_suffixes": ("image", "seg"),
    },
}

BTCV_EXPECTED = {
    "training": {
        "case0005", "case0006", "case0007", "case0009", "case0010", "case0021",
        "case0023", "case0024", "case0026", "case0027", "case0028", "case0030",
        "case0031", "case0033", "case0034", "case0037", "case0039", "case0040",
    },
    "validation": {"case0001", "case0008"},
    "testing": {
        "case0002", "case0003", "case0004", "case0022", "case0025",
        "case0029", "case0032", "case0035", "case0036", "case0038",
    },
}


class SplitValidationError(RuntimeError):
    """Raised when a fixed manifest or preprocessed split is inconsistent."""


def _canonical_fingerprint(identifiers: list[str]) -> str:
    canonical = "\n".join(identifiers) + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _read_manifest(
    path: Path,
    *,
    dataset: str,
    split: str,
    expected_count: int,
    id_pattern: str,
    expected_fingerprint: str,
) -> list[str]:
    if not path.is_file():
        raise SplitValidationError(f"Missing {dataset} {split} manifest: {path}")

    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise SplitValidationError(
                f"{path}:{line_number} is blank; release manifests must be ID-only"
            )
        if line != line.strip():
            raise SplitValidationError(
                f"{path}:{line_number} contains surrounding whitespace"
            )
        if line.startswith("#"):
            raise SplitValidationError(
                f"{path}:{line_number} is a comment; release manifests must be ID-only"
            )

    identifiers = lines
    if len(identifiers) != expected_count:
        raise SplitValidationError(
            f"{dataset} {split} must contain {expected_count} IDs, "
            f"found {len(identifiers)} in {path}"
        )

    invalid = [
        identifier
        for identifier in identifiers
        if re.fullmatch(id_pattern, identifier) is None
    ]
    if invalid:
        raise SplitValidationError(
            f"{dataset} {split} contains invalid IDs: {', '.join(invalid[:5])}"
        )

    if len(set(identifiers)) != len(identifiers):
        duplicates = sorted(
            identifier
            for identifier in set(identifiers)
            if identifiers.count(identifier) > 1
        )
        raise SplitValidationError(
            f"{dataset} {split} contains duplicate IDs: {', '.join(duplicates[:5])}"
        )

    fingerprint = _canonical_fingerprint(identifiers)
    if fingerprint != expected_fingerprint:
        raise SplitValidationError(
            f"{dataset} {split} IDs or ordering changed unexpectedly: "
            f"expected sha256 {expected_fingerprint}, found {fingerprint}"
        )
    return identifiers


def validate_release_manifests() -> dict[str, dict[str, list[str]]]:
    """Validate exact fixed IDs, counts, uniqueness, and cross-split isolation."""

    all_manifests: dict[str, dict[str, list[str]]] = {}
    for dataset, spec in SPECS.items():
        manifest_dir = MANIFEST_ROOT / spec["manifest_dir"]
        split_ids: dict[str, list[str]] = {}
        owners: dict[str, str] = {}

        for split in SPLITS:
            identifiers = _read_manifest(
                manifest_dir / spec["manifest_files"][split],
                dataset=dataset,
                split=split,
                expected_count=spec["id_counts"][split],
                id_pattern=spec["id_pattern"],
                expected_fingerprint=spec["fingerprints"][split],
            )
            split_ids[split] = identifiers
            for identifier in identifiers:
                previous = owners.get(identifier)
                if previous is not None:
                    raise SplitValidationError(
                        f"{dataset} ID {identifier} appears in both {previous} and {split}"
                    )
                owners[identifier] = split

        expected_total = sum(spec["id_counts"].values())
        if len(owners) != expected_total:
            raise SplitValidationError(
                f"{dataset} must contain {expected_total} unique IDs, found {len(owners)}"
            )

        if dataset == "btcv":
            for split in SPLITS:
                actual = set(split_ids[split])
                if actual != BTCV_EXPECTED[split]:
                    missing = sorted(BTCV_EXPECTED[split] - actual)
                    unexpected = sorted(actual - BTCV_EXPECTED[split])
                    raise SplitValidationError(
                        f"BTCV {split} case set changed: "
                        f"missing={missing}, unexpected={unexpected}"
                    )
        elif dataset == "acdc":
            expected_subjects = {f"patient{number:03d}" for number in range(1, 101)}
            if set(owners) != expected_subjects:
                missing = sorted(expected_subjects - set(owners))
                unexpected = sorted(set(owners) - expected_subjects)
                raise SplitValidationError(
                    "ACDC manifests must partition patient001 through patient100: "
                    f"missing={missing[:5]}, unexpected={unexpected[:5]}"
                )

        all_manifests[dataset] = split_ids
    return all_manifests


def _owner_from_member(dataset: str, member_name: str, pattern: str) -> str:
    match = re.fullmatch(pattern, member_name)
    if match is None:
        raise SplitValidationError(
            f"Unexpected {dataset} preprocessed member directory: {member_name}"
        )
    if dataset == "btcv":
        return f"case{match.group('number')}"
    return match.group("owner")


def validate_preprocessed_root(
    dataset: str,
    root: Path,
    manifests: dict[str, list[str]],
) -> dict[str, int]:
    """Validate one preprocessed dataset root without importing array libraries."""

    spec = SPECS[dataset]
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise SplitValidationError(f"{dataset} data root is not a directory: {root}")

    observed_counts: dict[str, int] = {}
    for split in SPLITS:
        split_dir = root / split
        if not split_dir.is_dir():
            raise SplitValidationError(
                f"Missing {dataset} preprocessed {split} directory: {split_dir}"
            )

        members = sorted(split_dir.iterdir(), key=lambda path: path.name)
        non_directories = [path.name for path in members if not path.is_dir()]
        if non_directories:
            raise SplitValidationError(
                f"{dataset} {split} contains non-directory members: "
                f"{', '.join(non_directories[:5])}"
            )

        expected_slices = spec["slice_counts"][split]
        if len(members) != expected_slices:
            raise SplitValidationError(
                f"{dataset} {split} must contain {expected_slices} sample directories, "
                f"found {len(members)} in {split_dir}"
            )

        observed_owners: set[str] = set()
        for member in members:
            owner = _owner_from_member(dataset, member.name, spec["member_pattern"])
            observed_owners.add(owner)

            expected_files = {
                f"{member.name}_{suffix}.npy"
                for suffix in spec["required_suffixes"]
            }
            actual_children = list(member.iterdir())
            nested = [path.name for path in actual_children if not path.is_file()]
            actual_files = {path.name for path in actual_children if path.is_file()}
            if nested or actual_files != expected_files:
                raise SplitValidationError(
                    f"{member} must contain exactly {sorted(expected_files)}; "
                    f"found files={sorted(actual_files)}, nested={sorted(nested)}"
                )

        expected_owners = set(manifests[split])
        if observed_owners != expected_owners:
            missing = sorted(expected_owners - observed_owners)
            unexpected = sorted(observed_owners - expected_owners)
            raise SplitValidationError(
                f"{dataset} {split} data members do not match the fixed manifest: "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}"
            )
        observed_counts[split] = len(members)
    return observed_counts


def _parse_data_roots(values: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        dataset, separator, raw_path = value.partition("=")
        dataset = dataset.strip().lower()
        if not separator or not dataset or not raw_path.strip():
            raise SplitValidationError(
                f"Invalid --data-root {value!r}; expected DATASET=PATH"
            )
        if dataset not in SPECS:
            raise SplitValidationError(
                f"Unknown dataset {dataset!r}; choose from {', '.join(SPECS)}"
            )
        if dataset in parsed:
            raise SplitValidationError(f"Duplicate --data-root for {dataset}")
        parsed[dataset] = Path(raw_path.strip())
    return parsed


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate all fixed release manifests. Optionally repeat --data-root "
            "DATASET=PATH to validate preprocessed directory membership and counts."
        )
    )
    parser.add_argument(
        "--data-root",
        action="append",
        default=[],
        metavar="DATASET=PATH",
        help="Dataset-specific preprocessed root; may be repeated.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    try:
        manifests = validate_release_manifests()
        data_roots = _parse_data_roots(args.data_root)
        for dataset in SPECS:
            counts = SPECS[dataset]["id_counts"]
            print(
                f"{dataset} manifests: "
                + ", ".join(f"{split}={counts[split]}" for split in SPLITS)
            )
        for dataset, root in data_roots.items():
            counts = validate_preprocessed_root(dataset, root, manifests[dataset])
            print(
                f"{dataset} data: "
                + ", ".join(f"{split}={counts[split]}" for split in SPLITS)
            )
    except (OSError, SplitValidationError) as error:
        print(f"split validation failed: {error}", file=sys.stderr)
        return 1
    print("split validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
