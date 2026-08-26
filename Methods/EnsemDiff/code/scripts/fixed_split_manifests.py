"""Utilities for fixed dataset splits and fail-closed preprocessing outputs."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path


def read_id_manifest(
    path: str | Path,
    *,
    expected_count: int,
    id_pattern: str,
    dataset_name: str,
    split_name: str,
) -> list[str]:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing {dataset_name} {split_name} manifest: {manifest_path}"
        )

    identifiers = [
        line.strip()
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(identifiers) != expected_count:
        raise ValueError(
            f"{dataset_name} {split_name} manifest must contain exactly "
            f"{expected_count} IDs, found {len(identifiers)} in {manifest_path}"
        )

    invalid = [
        identifier
        for identifier in identifiers
        if re.fullmatch(id_pattern, identifier) is None
    ]
    if invalid:
        preview = ", ".join(invalid[:5])
        raise ValueError(
            f"Invalid {dataset_name} IDs in {manifest_path}; first invalid IDs: {preview}"
        )

    seen: set[str] = set()
    duplicates: list[str] = []
    for identifier in identifiers:
        if identifier in seen and identifier not in duplicates:
            duplicates.append(identifier)
        seen.add(identifier)
    if duplicates:
        preview = ", ".join(duplicates[:5])
        raise ValueError(
            f"Duplicate {dataset_name} IDs in {manifest_path}; first duplicates: {preview}"
        )
    return identifiers


def load_fixed_split_manifests(
    manifest_root: str | Path,
    *,
    manifest_filenames: Mapping[str, str],
    expected_counts: Mapping[str, int],
    id_pattern: str,
    dataset_name: str,
) -> dict[str, list[str]]:
    root = Path(manifest_root).expanduser().resolve()
    expected_splits = set(expected_counts)
    if set(manifest_filenames) != expected_splits:
        raise ValueError(
            f"{dataset_name} manifest filename mapping must define exactly "
            f"{sorted(expected_splits)}"
        )

    splits = {
        split_name: read_id_manifest(
            root / manifest_filenames[split_name],
            expected_count=expected_counts[split_name],
            id_pattern=id_pattern,
            dataset_name=dataset_name,
            split_name=split_name,
        )
        for split_name in expected_counts
    }

    owners: dict[str, str] = {}
    leakage: list[tuple[str, str, str]] = []
    for split_name, identifiers in splits.items():
        for identifier in identifiers:
            previous_split = owners.get(identifier)
            if previous_split is not None:
                leakage.append((identifier, previous_split, split_name))
            else:
                owners[identifier] = split_name
    if leakage:
        preview = ", ".join(
            f"{identifier} ({left}/{right})"
            for identifier, left, right in leakage[:5]
        )
        raise ValueError(f"Cross-split {dataset_name} leakage detected: {preview}")
    return splits


def assert_exact_id_set(
    actual_ids: Iterable[str],
    expected_ids: Iterable[str],
    *,
    dataset_name: str,
    split_name: str,
    source_name: str,
) -> None:
    actual = set(actual_ids)
    expected = set(expected_ids)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if not missing and not unexpected:
        return

    details = []
    if missing:
        details.append(f"missing {len(missing)} ({', '.join(missing[:5])})")
    if unexpected:
        details.append(f"unexpected {len(unexpected)} ({', '.join(unexpected[:5])})")
    raise ValueError(
        f"{dataset_name} {split_name} IDs in {source_name} do not match the fixed "
        f"manifest: {'; '.join(details)}"
    )


def prepare_empty_output_directory(
    path: str | Path,
    *,
    dataset_name: str,
) -> Path:
    """Create an output directory or reject it when any prior content exists."""
    output_root = Path(path).expanduser().resolve()
    if output_root.exists():
        if not output_root.is_dir():
            raise NotADirectoryError(
                f"{dataset_name} output path is not a directory: {output_root}"
            )
        existing = sorted(entry.name for entry in output_root.iterdir())
        if existing:
            preview = ", ".join(existing[:5])
            raise FileExistsError(
                f"{dataset_name} output directory must be empty: {output_root}; "
                f"existing entries include: {preview}"
            )
    else:
        output_root.mkdir(parents=True)
    return output_root
