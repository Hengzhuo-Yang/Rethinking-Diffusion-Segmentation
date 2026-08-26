"""Fixed-manifest helpers for the public TSLDSeg audit release.

Release modification; see MODIFICATIONS.md and LICENSES/. Manifests contain
only public dataset identifiers, never local filesystem paths.
"""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
_ID_PATTERNS = {
    "btcv": re.compile(r"case\d{4}"),
    "acdc": re.compile(r"patient\d{3}"),
    "isic2018": re.compile(r"ISIC_\d{7}"),
}


def _manifest_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved.resolve()


def read_manifest_ids(path: str | Path, dataset: str) -> list[str]:
    """Read and strictly validate one ID-only split manifest."""

    dataset = dataset.lower()
    if dataset not in _ID_PATTERNS:
        raise ValueError(f"Unsupported manifest dataset: {dataset!r}")
    manifest = _manifest_path(path)
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing fixed split manifest: {manifest}")
    identifiers = manifest.read_text(encoding="utf-8").splitlines()
    if not identifiers:
        raise ValueError(f"Fixed split manifest is empty: {manifest}")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"Fixed split manifest contains duplicate IDs: {manifest}")
    for line_number, identifier in enumerate(identifiers, start=1):
        if identifier != identifier.strip() or not identifier:
            raise ValueError(f"{manifest}:{line_number} is not one clean ID")
        if _ID_PATTERNS[dataset].fullmatch(identifier) is None:
            raise ValueError(
                f"{manifest}:{line_number} has invalid {dataset} ID {identifier!r}"
            )
    return identifiers


def canonical_sample_owner(dataset: str, sample_path: str | Path) -> str:
    """Map a preprocessed PNG filename to its case/patient/image owner."""

    stem = Path(sample_path).stem
    dataset = dataset.lower()
    if dataset == "btcv":
        match = re.search(r"(?:case|vol)(\d{1,4})(?:_|$)", stem, flags=re.IGNORECASE)
        if match:
            return f"case{int(match.group(1)):04d}"
    elif dataset == "acdc":
        match = re.search(r"patient\d{3}", stem, flags=re.IGNORECASE)
        if match:
            return match.group(0).lower()
    elif dataset == "isic2018":
        match = re.search(r"ISIC_\d{7}", stem, flags=re.IGNORECASE)
        if match:
            return match.group(0).upper()
    else:
        raise ValueError(f"Unsupported manifest dataset: {dataset!r}")
    raise ValueError(f"Cannot map {dataset} sample filename to an owner ID: {stem!r}")


def filter_image_paths(
    image_paths: list[Path],
    *,
    manifest_path: str | Path | None,
    dataset: str,
) -> list[Path]:
    """Filter a physical data pool to the exact IDs in a fixed manifest."""

    if manifest_path is None:
        return image_paths
    identifiers = read_manifest_ids(manifest_path, dataset)
    allowed = set(identifiers)
    selected = [
        path for path in image_paths
        if canonical_sample_owner(dataset, path) in allowed
    ]
    observed = {canonical_sample_owner(dataset, path) for path in selected}
    missing = sorted(allowed - observed)
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} IDs in {manifest_path} are absent from the data pool; "
            f"first missing ID: {missing[0]}"
        )
    if not selected:
        raise RuntimeError(f"Manifest selected no {dataset} samples: {manifest_path}")
    return selected
