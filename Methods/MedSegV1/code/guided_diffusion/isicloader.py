import csv
import json
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
    "image",
    "image_id",
}


def _read_id_manifest(data_manifest):
    """Read a one-dimensional, ID-only manifest."""
    manifest_path = Path(data_manifest).expanduser()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"ISIC2018 data manifest not found: {manifest_path}")

    suffix = manifest_path.suffix.lower()
    if suffix == ".json":
        with manifest_path.open("r", encoding="utf-8-sig") as handle:
            values = json.load(handle)
        if isinstance(values, dict):
            if "ids" not in values:
                raise ValueError(
                    f"ISIC2018 JSON manifest must be a list or contain an 'ids' list: "
                    f"{manifest_path}"
                )
            values = values["ids"]
        if not isinstance(values, list):
            raise ValueError(f"ISIC2018 JSON manifest must contain an ID list: {manifest_path}")
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
                        f"ISIC2018 CSV manifest row {row_number} must contain exactly one ID "
                        f"column: {manifest_path}"
                    )
                rows.append(row[0].strip())
        if rows and rows[0].casefold() in _CSV_ID_HEADERS:
            rows = rows[1:]
        values = rows
    else:
        raise ValueError(
            f"Unsupported ISIC2018 data manifest format '{suffix}'; use JSON, TXT, or CSV: "
            f"{manifest_path}"
        )

    ids = []
    for position, value in enumerate(values, 1):
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ValueError(
                f"ISIC2018 manifest ID {position} must be a string or integer: {manifest_path}"
            )
        value = str(value).strip()
        if not value or value in {".", ".."} or "/" in value or chr(92) in value:
            raise ValueError(
                f"ISIC2018 manifest ID {position} is not a valid ID-only value: {value!r}"
            )
        ids.append(value)
    if not ids:
        raise ValueError(f"ISIC2018 data manifest is empty: {manifest_path}")
    duplicate_ids = sorted({value for value in ids if ids.count(value) > 1})
    if duplicate_ids:
        raise ValueError(
            f"Duplicate ISIC2018 IDs in {manifest_path}: {', '.join(duplicate_ids)}"
        )
    return manifest_path, ids


def _filter_pairs_by_manifest(pairs, data_manifest):
    manifest_path, requested = _read_id_manifest(data_manifest)
    requested_ids = set(requested)
    indexed_pairs = []
    for pair in pairs:
        stem = pair[0].stem
        matches = [
            item_id
            for item_id in requested
            if stem == item_id or stem.startswith(f"{item_id}_")
        ]
        if len(matches) > 1:
            raise ValueError(
                f"ISIC2018 image stem {stem!r} ambiguously matches manifest IDs "
                f"{matches} in {manifest_path}"
            )
        if matches:
            indexed_pairs.append((matches[0], pair))

    observed_ids = {item_id for item_id, _ in indexed_pairs}
    missing_ids = sorted(requested_ids - observed_ids)
    extra_ids = sorted(observed_ids - requested_ids)
    if missing_ids or extra_ids:
        raise FileNotFoundError(
            "ISIC2018 manifest coverage mismatch against the physical image directory "
            f"({manifest_path}): missing={missing_ids}, extra={extra_ids}"
        )
    return [pair for _, pair in indexed_pairs], manifest_path


class ISICDataset(Dataset):
    """ISIC2018 Task 1 binary lesion segmentation PNG dataset."""

    def __init__(
        self,
        args,
        data_path,
        transform=None,
        mode="Training",
        plane=False,
        image_size=None,
        data_manifest=None,
    ):
        super().__init__()
        _ = transform, mode, plane
        self.data_path = Path(data_path).expanduser()
        self.image_size = int(image_size or getattr(args, "image_size", 256))
        self.num_seg_classes = int(getattr(args, "num_seg_classes", 2))
        self.num_mask_channels = int(getattr(args, "num_mask_channels", 1))
        if self.num_seg_classes != 2 or self.num_mask_channels != 1:
            raise ValueError(
                "ISIC2018 is binary lesion segmentation and expects "
                "--num_seg_classes 2 and --num_mask_channels 1."
            )

        image_dir = self.data_path / "images"
        mask_dir = self.data_path / "masks"
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Missing ISIC2018 image directory: {image_dir}")
        if not mask_dir.is_dir():
            raise FileNotFoundError(f"Missing ISIC2018 mask directory: {mask_dir}")

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
                f"Missing ISIC2018 masks for {len(missing)} images, first missing image: {missing[0]}"
            )
        if not pairs:
            raise RuntimeError(f"No ISIC2018 PNG image/mask pairs found under {self.data_path}")

        manifest_value = (
            data_manifest
            if data_manifest is not None
            else getattr(args, "data_manifest", None)
        )
        self.data_manifest = None
        if manifest_value is not None and str(manifest_value).strip():
            pairs, manifest_path = _filter_pairs_by_manifest(pairs, manifest_value)
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
