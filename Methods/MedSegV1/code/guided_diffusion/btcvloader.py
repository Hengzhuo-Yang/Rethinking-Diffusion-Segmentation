import csv
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _resample_constants():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.BILINEAR, Image.Resampling.NEAREST
    return Image.BILINEAR, Image.NEAREST


BILINEAR, NEAREST = _resample_constants()

_CSV_ID_HEADERS = {
    "id",
    "ids",
    "case",
    "case_id",
    "volume",
    "volume_id",
}
_BTCV_CASE_FROM_STEM = re.compile(r"^vol(?P<case_id>\d{4})(?:_|$)")
_BTCV_MANIFEST_ID = re.compile(r"^(?:vol)?(?P<case_id>\d{4})$", re.IGNORECASE)


def _read_id_manifest(data_manifest):
    """Read a one-dimensional, ID-only manifest."""
    manifest_path = Path(data_manifest).expanduser()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"BTCV data manifest not found: {manifest_path}")

    suffix = manifest_path.suffix.lower()
    if suffix == ".json":
        with manifest_path.open("r", encoding="utf-8-sig") as handle:
            values = json.load(handle)
        if isinstance(values, dict):
            if "ids" not in values:
                raise ValueError(
                    f"BTCV JSON manifest must be a list or contain an 'ids' list: {manifest_path}"
                )
            values = values["ids"]
        if not isinstance(values, list):
            raise ValueError(f"BTCV JSON manifest must contain an ID list: {manifest_path}")
    elif suffix in {".txt", ".list"}:
        with manifest_path.open("r", encoding="utf-8-sig") as handle:
            values = [
                line.strip()
                for line in handle
                if line.strip() and not line.lstrip().startswith("#")
            ]
    elif suffix == ".csv":
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = []
            for row_number, row in enumerate(csv.reader(handle), 1):
                if not row or not any(cell.strip() for cell in row):
                    continue
                if len(row) != 1:
                    raise ValueError(
                        f"BTCV CSV manifest row {row_number} must contain exactly one ID column: "
                        f"{manifest_path}"
                    )
                rows.append(row[0].strip())
        if rows and rows[0].casefold() in _CSV_ID_HEADERS:
            rows = rows[1:]
        values = rows
    else:
        raise ValueError(
            f"Unsupported BTCV data manifest format '{suffix}'; use JSON, TXT, or CSV: "
            f"{manifest_path}"
        )

    ids = []
    for position, value in enumerate(values, 1):
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ValueError(
                f"BTCV manifest ID {position} must be a string or integer: {manifest_path}"
            )
        value = str(value).strip()
        if not value or value in {".", ".."} or "/" in value or chr(92) in value:
            raise ValueError(
                f"BTCV manifest ID {position} is not a valid ID-only value: {value!r}"
            )
        ids.append(value)
    if not ids:
        raise ValueError(f"BTCV data manifest is empty: {manifest_path}")
    return manifest_path, ids


def _filter_pairs_by_manifest(pairs, data_manifest):
    manifest_path, raw_ids = _read_id_manifest(data_manifest)
    requested = []
    seen = set()
    duplicate_ids = set()
    for raw_id in raw_ids:
        match = _BTCV_MANIFEST_ID.fullmatch(raw_id)
        if match is None:
            raise ValueError(
                f"Invalid BTCV case ID {raw_id!r} in {manifest_path}; expected XXXX or volXXXX"
            )
        case_id = match.group("case_id")
        if case_id in seen:
            duplicate_ids.add(case_id)
        seen.add(case_id)
        requested.append(case_id)
    if duplicate_ids:
        raise ValueError(
            f"Duplicate BTCV case IDs in {manifest_path}: {', '.join(sorted(duplicate_ids))}"
        )
    requested_ids = set(requested)

    indexed_pairs = []
    for pair in pairs:
        stem = pair[0].stem
        match = _BTCV_CASE_FROM_STEM.match(stem)
        if match is None:
            raise ValueError(
                f"Cannot extract a BTCV case ID from image stem {stem!r}; expected volXXXX_..."
            )
        indexed_pairs.append((match.group("case_id"), pair))

    available_ids = {case_id for case_id, _ in indexed_pairs}
    missing_ids = sorted(requested_ids - available_ids)
    if missing_ids:
        raise FileNotFoundError(
            f"BTCV manifest IDs not found under the physical image directory "
            f"({manifest_path}): {', '.join(missing_ids)}"
        )

    filtered = [pair for case_id, pair in indexed_pairs if case_id in requested_ids]
    observed_ids = {
        _BTCV_CASE_FROM_STEM.match(image_path.stem).group("case_id")
        for image_path, _ in filtered
    }
    missing_after_filter = sorted(requested_ids - observed_ids)
    extra_after_filter = sorted(observed_ids - requested_ids)
    if missing_after_filter or extra_after_filter:
        raise RuntimeError(
            "BTCV manifest filtering coverage mismatch: "
            f"missing={missing_after_filter}, extra={extra_after_filter}"
        )
    return filtered, manifest_path


class BTCVDataset(Dataset):
    """LEAF-style BTCV/Synapse binary 2D PNG dataset."""

    def __init__(self, data_path, image_size=256, mode="Training", data_manifest=None):
        super().__init__()
        self.data_path = os.path.expanduser(str(data_path))
        self.image_size = int(image_size)
        self.mode = mode

        root = Path(self.data_path)
        image_dir = root / "images"
        mask_dir = root / "masks"
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Missing BTCV image directory: {image_dir}")
        if not mask_dir.is_dir():
            raise FileNotFoundError(f"Missing BTCV mask directory: {mask_dir}")

        images = sorted(image_dir.glob("*.png"))
        masks_by_stem = {path.stem: path for path in sorted(mask_dir.glob("*.png"))}
        pairs = []
        missing = []
        for image_path in images:
            mask_path = masks_by_stem.get(image_path.stem)
            if mask_path is None:
                missing.append(image_path.name)
                continue
            pairs.append((image_path, mask_path))
        if missing:
            raise FileNotFoundError(
                f"Missing BTCV masks for {len(missing)} images, first missing image: {missing[0]}"
            )
        if not pairs:
            raise RuntimeError(f"No BTCV PNG image/mask pairs found under {root}")

        self.data_manifest = None
        if data_manifest is not None and str(data_manifest).strip():
            pairs, manifest_path = _filter_pairs_by_manifest(pairs, data_manifest)
            self.data_manifest = str(manifest_path)

        self.name_list = [str(image_path) for image_path, _ in pairs]
        self.label_list = [str(mask_path) for _, mask_path in pairs]
        self.foreground_pixels = []
        for mask_path in self.label_list:
            mask = Image.open(mask_path).convert("L")
            self.foreground_pixels.append(int((np.asarray(mask) > 127).sum()))

    def __len__(self):
        return len(self.name_list)

    def _load_image(self, path):
        image = Image.open(path).convert("RGB")
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), resample=BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    def _load_mask(self, path):
        mask = Image.open(path).convert("L")
        if mask.size != (self.image_size, self.image_size):
            mask = mask.resize((self.image_size, self.image_size), resample=NEAREST)
        arr = (np.asarray(mask, dtype=np.uint8) > 127).astype(np.float32)
        return torch.from_numpy(arr[None, ...]).contiguous()

    def __getitem__(self, index):
        image_path = self.name_list[index]
        mask_path = self.label_list[index]
        return self._load_image(image_path), self._load_mask(mask_path), image_path
