"""Validate fixed release manifests and, optionally, preprocessed PNG caches."""

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
        "pool_dirs": {"train": "train", "val": "test", "test": "test"},
        "id_counts": {"train": 18, "val": 2, "test": 10},
        "sample_counts": {"train": 2211, "val": 295, "test": 1273},
        "id_pattern": r"case\d{4}",
        "sample_pattern": r"vol(?P<owner>\d{4})_\d{3}",
        "fingerprints": {
            "train": "dfdb73f71f36503c06c569760e77ab15fa67cdce84341127e553b327e2219c71",
            "val": "47ed3d1d80a76914ffa3bb75ecc33f40b9b8ae7c113d852fe32f3aff5031efc3",
            "test": "fa5ebbed2ed11e94f6a7dce1c89ff9c03bf4d9cf2e893bf28da9cd0f9f193b3c",
        },
    },
    "acdc": {
        "dataset_dir": "ACDC",
        "pool_dirs": {"train": "training", "val": "validation", "test": "testing"},
        "id_counts": {"train": 70, "val": 10, "test": 20},
        "sample_counts": {"train": 1304, "val": 182, "test": 416},
        "id_pattern": r"patient\d{3}",
        "sample_pattern": r"(?P<owner>patient\d{3})_frame\d+_z\d{3}",
        "fingerprints": {
            "train": "bfc7abb8764d17f50615d15c26653e6e395205a71aa60712a4e3cbfe4c4946b8",
            "val": "4d1799fbc83e1d6fcdf000f5e5c4d9b4efae840e306bae95fdb8105180a6b6aa",
            "test": "7b2aac91c98e719a7f41300eccf8d622c0fc98754c7b8e6639ce5bbf563dd89b",
        },
    },
    "isic2018": {
        "dataset_dir": "ISIC18",
        "pool_dirs": {"train": "training", "val": "validation", "test": "testing"},
        "id_counts": {"train": 2594, "val": 100, "test": 1000},
        "sample_counts": {"train": 2594, "val": 100, "test": 1000},
        "id_pattern": r"ISIC_\d{7}",
        "sample_pattern": r"(?P<owner>ISIC_\d{7})",
        "fingerprints": {
            "train": "b6a11fa50b1cf0363160258b5cac96ccae382ea524e34434c3212d1fbb1e4775",
            "val": "44c159239176fe8f63c1efd5f40b5cf66fcafcfd61571c684cd8467f912a366c",
            "test": "6fb4df9f361085d70966ce9098133a6491b756869717af4eaef4ca687686e4f7",
        },
    },
}


class SplitValidationError(RuntimeError):
    pass


def _fingerprint(identifiers: list[str]) -> str:
    return hashlib.sha256(("\n".join(identifiers) + "\n").encode("utf-8")).hexdigest()


def _read_manifest(dataset: str, split: str) -> list[str]:
    spec = SPECS[dataset]
    path = MANIFEST_ROOT / dataset / f"{split}.txt"
    if not path.is_file():
        raise SplitValidationError(f"Missing manifest: {path}")
    identifiers = path.read_text(encoding="utf-8").splitlines()
    if len(identifiers) != spec["id_counts"][split]:
        raise SplitValidationError(
            f"{dataset}/{split}: expected {spec['id_counts'][split]} IDs, found {len(identifiers)}"
        )
    if len(identifiers) != len(set(identifiers)):
        raise SplitValidationError(f"{dataset}/{split}: duplicate IDs")
    for identifier in identifiers:
        if identifier != identifier.strip() or re.fullmatch(spec["id_pattern"], identifier) is None:
            raise SplitValidationError(f"{dataset}/{split}: invalid ID {identifier!r}")
    observed_hash = _fingerprint(identifiers)
    if observed_hash != spec["fingerprints"][split]:
        raise SplitValidationError(
            f"{dataset}/{split}: fingerprint changed ({observed_hash})"
        )
    return identifiers


def validate_release_manifests() -> dict[str, dict[str, list[str]]]:
    result = {}
    for dataset, spec in SPECS.items():
        split_ids = {split: _read_manifest(dataset, split) for split in SPLITS}
        owners: dict[str, str] = {}
        for split, identifiers in split_ids.items():
            for identifier in identifiers:
                if identifier in owners:
                    raise SplitValidationError(
                        f"{dataset}: {identifier} leaks across {owners[identifier]} and {split}"
                    )
                owners[identifier] = split
        if len(owners) != sum(spec["id_counts"].values()):
            raise SplitValidationError(f"{dataset}: unique ID total changed")
        if dataset == "acdc" and set(owners) != {
            f"patient{number:03d}" for number in range(1, 101)
        }:
            raise SplitValidationError("ACDC manifests do not partition patient001..patient100")
        result[dataset] = split_ids
    return result


def _dataset_root(dataset: str, root: Path) -> Path:
    root = root.expanduser().resolve()
    nested = root / SPECS[dataset]["dataset_dir"]
    return nested if nested.is_dir() else root


def _owner(dataset: str, stem: str) -> str:
    match = re.fullmatch(SPECS[dataset]["sample_pattern"], stem)
    if match is None:
        raise SplitValidationError(f"Unexpected {dataset} PNG stem: {stem}")
    owner = match.group("owner")
    return f"case{owner}" if dataset == "btcv" else owner


def _png_stems(directory: Path) -> set[str]:
    if not directory.is_dir():
        raise SplitValidationError(f"Missing directory: {directory}")
    return {
        path.stem for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    }


def validate_preprocessed_root(
    dataset: str,
    root: Path,
    manifests: dict[str, list[str]],
) -> dict[str, int]:
    spec = SPECS[dataset]
    dataset_root = _dataset_root(dataset, root)
    selected_by_split: dict[str, set[str]] = {}
    observed_counts = {}
    checked_pools: set[Path] = set()
    for split in SPLITS:
        pool = dataset_root / spec["pool_dirs"][split]
        image_stems = _png_stems(pool / "images")
        mask_stems = _png_stems(pool / "masks")
        if image_stems != mask_stems:
            raise SplitValidationError(f"{dataset}/{split}: image-mask pairing mismatch")
        if dataset == "acdc" and _png_stems(pool / "label_maps") != image_stems:
            raise SplitValidationError(f"acdc/{split}: image-label_map pairing mismatch")
        checked_pools.add(pool)
        expected_owners = set(manifests[split])
        selected = {stem for stem in image_stems if _owner(dataset, stem) in expected_owners}
        observed_owners = {_owner(dataset, stem) for stem in selected}
        if observed_owners != expected_owners:
            raise SplitValidationError(f"{dataset}/{split}: cache owners differ from manifest")
        expected_samples = spec["sample_counts"][split]
        if len(selected) != expected_samples:
            raise SplitValidationError(
                f"{dataset}/{split}: expected {expected_samples} samples, found {len(selected)}"
            )
        selected_by_split[split] = selected
        observed_counts[split] = len(selected)
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if selected_by_split[left] & selected_by_split[right]:
            raise SplitValidationError(f"{dataset}: sample leakage across {left}/{right}")
    return observed_counts


def _data_roots(values: list[str]) -> dict[str, Path]:
    roots = {}
    for value in values:
        dataset, separator, raw_path = value.partition("=")
        dataset = dataset.lower().strip()
        if not separator or dataset not in SPECS or not raw_path.strip():
            raise SplitValidationError(f"Invalid --data-root {value!r}; expected DATASET=PATH")
        roots[dataset] = Path(raw_path.strip())
    return roots


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", action="append", default=[], metavar="DATASET=PATH")
    args = parser.parse_args(argv)
    try:
        manifests = validate_release_manifests()
        for dataset, root in _data_roots(args.data_root).items():
            counts = validate_preprocessed_root(dataset, root, manifests[dataset])
            print(f"{dataset} cache: {counts}")
    except (OSError, SplitValidationError) as error:
        print(f"split validation failed: {error}", file=sys.stderr)
        return 1
    for dataset, spec in SPECS.items():
        print(f"{dataset} IDs: {spec['id_counts']}; samples: {spec['sample_counts']}")
    print("split validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
