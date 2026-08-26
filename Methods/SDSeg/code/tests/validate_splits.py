#!/usr/bin/env python3
"""Validate the immutable SDSeg release splits and an optional processed cache.

The bundled text files are the release's source of truth.  The SHA-256 checks
below make an accidental edit to any ID list visible; the explicit BTCV and
ACDC sets also make the clinically important case-level partition readable in
code.  When ``--data-root`` is supplied, every processed image/mask pair and
its case identifier is checked against the corresponding manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


CODE_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_ROOT = CODE_ROOT / "manifests"


BTCV_TRAIN = (
    "case0005", "case0006", "case0007", "case0009", "case0010", "case0021",
    "case0023", "case0024", "case0026", "case0027", "case0028", "case0030",
    "case0031", "case0033", "case0034", "case0037", "case0039", "case0040",
)
BTCV_VALIDATION = ("case0001", "case0008")
BTCV_TEST = (
    "case0002", "case0003", "case0004", "case0022", "case0025", "case0029",
    "case0032", "case0035", "case0036", "case0038",
)

ACDC_TRAIN = (
    "patient001", "patient004", "patient005", "patient006", "patient007",
    "patient010", "patient011", "patient013", "patient015", "patient016",
    "patient018", "patient020", "patient022", "patient023", "patient025",
    "patient026", "patient027", "patient028", "patient030", "patient031",
    "patient032", "patient034", "patient035", "patient036", "patient037",
    "patient038", "patient039", "patient040", "patient043", "patient044",
    "patient045", "patient046", "patient047", "patient051", "patient052",
    "patient054", "patient056", "patient057", "patient058", "patient059",
    "patient060", "patient062", "patient063", "patient065", "patient066",
    "patient068", "patient069", "patient070", "patient072", "patient073",
    "patient074", "patient075", "patient077", "patient078", "patient082",
    "patient083", "patient084", "patient085", "patient086", "patient087",
    "patient089", "patient090", "patient091", "patient093", "patient094",
    "patient096", "patient097", "patient098", "patient099", "patient100",
)
ACDC_VALIDATION = (
    "patient019", "patient021", "patient029", "patient033", "patient041",
    "patient050", "patient061", "patient071", "patient076", "patient080",
)
ACDC_TEST = (
    "patient002", "patient003", "patient008", "patient009", "patient012",
    "patient014", "patient017", "patient024", "patient042", "patient048",
    "patient049", "patient053", "patient055", "patient064", "patient067",
    "patient079", "patient081", "patient088", "patient092", "patient095",
)


@dataclass(frozen=True)
class SplitSpec:
    manifest: str
    directory: str
    item_count: int
    case_count: int
    sha256: str
    ids: tuple[str, ...] | None = None


SPECS: dict[str, dict[str, SplitSpec]] = {
    "btcv": {
        "train": SplitSpec(
            "btcv/train_cases.txt", "train", 2211, 18,
            "9571fab0d21ff9da6be3d847b3423f70f599a2cff34eeed3637729f86c1a9229",
            BTCV_TRAIN,
        ),
        "validation": SplitSpec(
            "btcv/validation_cases.txt", "validation", 295, 2,
            "54937730dd706df2f829478609b0cc4f032db91f3a543c1ef5db21879a88e10d",
            BTCV_VALIDATION,
        ),
        "test": SplitSpec(
            "btcv/test_cases.txt", "test", 1273, 10,
            "a94a1230626cf591938afe4c6b32faaab0bce1f20cd3dd1b1d6c77631c67dc21",
            BTCV_TEST,
        ),
    },
    "acdc": {
        "train": SplitSpec(
            "acdc/train_patients.txt", "train", 1304, 70,
            "bfc7abb8764d17f50615d15c26653e6e395205a71aa60712a4e3cbfe4c4946b8",
            ACDC_TRAIN,
        ),
        "validation": SplitSpec(
            "acdc/validation_patients.txt", "validation", 182, 10,
            "4d1799fbc83e1d6fcdf000f5e5c4d9b4efae840e306bae95fdb8105180a6b6aa",
            ACDC_VALIDATION,
        ),
        "test": SplitSpec(
            "acdc/test_patients.txt", "test", 416, 20,
            "7b2aac91c98e719a7f41300eccf8d622c0fc98754c7b8e6639ce5bbf563dd89b",
            ACDC_TEST,
        ),
    },
    "isic2018": {
        "train": SplitSpec(
            "isic2018/training.txt", "training", 2594, 2594,
            "76585038230cf3bd26a2ed60d093f3c87201831fd8b9fc505b37a7ce5d37ed69",
        ),
        "validation": SplitSpec(
            "isic2018/validation.txt", "validation", 100, 100,
            "622367fc2722edf9847a1ae8084d9579b6fdc14ad750aac7f1fbe2f3f0dbb088",
        ),
        "test": SplitSpec(
            "isic2018/testing.txt", "testing", 1000, 1000,
            "6dc64664060f69472971d205862a877005720b1518f48aca25a87c7498588137",
        ),
    },
}


ID_PATTERNS = {
    "btcv": re.compile(r"(?:case|vol)(\d{4})", re.IGNORECASE),
    "acdc": re.compile(r"(patient\d{3})", re.IGNORECASE),
    "isic2018": re.compile(r"(ISIC_\d{7})", re.IGNORECASE),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_ids(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing release split manifest: {path}")
    raw = path.read_text(encoding="utf-8").splitlines()
    ids = tuple(line.strip() for line in raw if line.strip())
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs in {path}")
    return ids


def _normalise_id(dataset: str, value: str) -> str:
    match = ID_PATTERNS[dataset].search(value)
    if not match:
        raise ValueError(f"Could not parse a {dataset} ID from {value!r}")
    if dataset == "btcv":
        return f"case{match.group(1)}"
    if dataset == "acdc":
        return match.group(1).lower()
    return match.group(1).upper()


def _paired_pngs(split_root: Path) -> tuple[list[Path], list[Path]]:
    image_dir = split_root / "images"
    mask_dir = split_root / "masks"
    if not mask_dir.is_dir():
        mask_dir = split_root / "label_maps"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(
            f"Expected images/ and masks/ (or label_maps/) under {split_root}"
        )
    images = sorted(image_dir.glob("*.png"))
    masks = sorted(mask_dir.glob("*.png"))
    image_names = {path.name for path in images}
    mask_names = {path.name for path in masks}
    if image_names != mask_names:
        only_images = sorted(image_names - mask_names)[:3]
        only_masks = sorted(mask_names - image_names)[:3]
        raise ValueError(
            f"Unpaired images/masks in {split_root}: "
            f"images_only={only_images}, masks_only={only_masks}"
        )
    return images, masks


def _validate_manifest_files() -> tuple[dict, dict[str, dict[str, tuple[str, ...]]]]:
    report: dict = {"manifest_root": "manifests", "datasets": {}}
    loaded: dict[str, dict[str, tuple[str, ...]]] = {}
    for dataset, split_specs in SPECS.items():
        loaded[dataset] = {}
        report["datasets"][dataset] = {}
        for split, spec in split_specs.items():
            path = MANIFEST_ROOT / spec.manifest
            ids = _read_ids(path)
            loaded[dataset][split] = ids
            actual_hash = _sha256(path)
            if actual_hash != spec.sha256:
                raise ValueError(
                    f"Canonical manifest hash mismatch for {spec.manifest}: "
                    f"{actual_hash} != {spec.sha256}"
                )
            if len(ids) != spec.case_count:
                raise ValueError(
                    f"{dataset}/{split} ID count mismatch: {len(ids)} != {spec.case_count}"
                )
            if spec.ids is not None and ids != spec.ids:
                raise ValueError(f"Exact ID/order mismatch in {spec.manifest}")
            pattern = ID_PATTERNS[dataset]
            invalid = [item for item in ids if pattern.fullmatch(item) is None]
            if invalid:
                raise ValueError(f"Invalid IDs in {spec.manifest}: {invalid[:3]}")
            report["datasets"][dataset][split] = {
                "manifest": spec.manifest,
                "sha256": actual_hash,
                "case_count": len(ids),
                "expected_processed_items": spec.item_count,
            }

        split_sets = {name: set(values) for name, values in loaded[dataset].items()}
        names = tuple(split_sets)
        for index, left in enumerate(names):
            for right in names[index + 1:]:
                overlap = split_sets[left] & split_sets[right]
                if overlap:
                    raise ValueError(
                        f"{dataset} split leakage between {left} and {right}: {sorted(overlap)[:3]}"
                    )
    return report, loaded


def _validate_processed_data(
    data_root: Path,
    report: dict,
    loaded: dict[str, dict[str, tuple[str, ...]]],
) -> None:
    resolved = data_root.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Processed data root does not exist: {resolved}")
    report["processed_data_checked"] = True

    for dataset, split_specs in SPECS.items():
        dataset_root = resolved / dataset
        if not dataset_root.is_dir():
            raise FileNotFoundError(f"Missing processed dataset directory: {dataset_root}")
        observed_by_split: dict[str, set[str]] = {}
        for split, spec in split_specs.items():
            split_root = dataset_root / spec.directory
            # The proven local BTCV cache predates the public split layout: its
            # test/ directory contains all 12 held-out cases.  The release
            # loaders filter that combined directory by immutable manifests.
            # Accept both layouts, then enforce the exact filtered slice count.
            if dataset == "btcv" and not split_root.is_dir() and split in {"validation", "test"}:
                split_root = dataset_root / "test"
            images, masks = _paired_pngs(split_root)
            expected_ids = set(loaded[dataset][split])
            if dataset == "btcv":
                images = [
                    path for path in images
                    if _normalise_id(dataset, path.stem) in expected_ids
                ]
                masks = [
                    path for path in masks
                    if _normalise_id(dataset, path.stem) in expected_ids
                ]
            if len(images) != spec.item_count or len(masks) != spec.item_count:
                raise ValueError(
                    f"{dataset}/{split} processed count mismatch: "
                    f"images={len(images)}, masks={len(masks)}, expected={spec.item_count}"
                )
            observed_ids = {_normalise_id(dataset, path.stem) for path in images}
            if observed_ids != expected_ids:
                raise ValueError(
                    f"{dataset}/{split} processed IDs differ from manifest: "
                    f"missing={sorted(expected_ids - observed_ids)[:3]}, "
                    f"unexpected={sorted(observed_ids - expected_ids)[:3]}"
                )
            observed_by_split[split] = observed_ids
            report["datasets"][dataset][split]["processed_items"] = len(images)
            report["datasets"][dataset][split]["processed_case_count"] = len(observed_ids)

        names = tuple(observed_by_split)
        for index, left in enumerate(names):
            for right in names[index + 1:]:
                overlap = observed_by_split[left] & observed_by_split[right]
                if overlap:
                    raise ValueError(
                        f"Processed {dataset} leakage between {left} and {right}: "
                        f"{sorted(overlap)[:3]}"
                    )


def validate_release_splits(data_root: Path | None = None) -> dict:
    """Return a JSON-serialisable report or raise on the first violation."""

    report, loaded = _validate_manifest_files()
    report["processed_data_checked"] = False
    if data_root is not None:
        _validate_processed_data(Path(data_root), report, loaded)
    report["status"] = "pass"
    report["partition_policy"] = "case-level for BTCV/ACDC; official split for ISIC2018"
    return report


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Optional parent containing btcv/, acdc/, and isic2018/ processed caches.",
    )
    parser.add_argument("--report", type=Path, help="Optional JSON report path.")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = validate_release_splits(args.data_root)
    except Exception as exc:
        report = {"status": "fail", "error_type": type(exc).__name__, "error": str(exc)}
        if args.report:
            _write_json(args.report, report)
        print(json.dumps(report, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    if args.report:
        _write_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
