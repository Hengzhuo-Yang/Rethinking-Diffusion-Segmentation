import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


ACDC_CLASS_NAMES = {
    0: "background",
    1: "right_ventricle",
    2: "myocardium",
    3: "left_ventricle",
}

_CSV_ID_HEADERS = {
    "id",
    "ids",
    "case",
    "case_id",
    "subject",
    "subject_id",
    "patient",
    "patient_id",
}


def _read_id_manifest(data_manifest):
    """Read a one-dimensional, ID-only manifest."""
    manifest_path = Path(data_manifest).expanduser()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"ACDC data manifest not found: {manifest_path}")

    suffix = manifest_path.suffix.lower()
    if suffix == ".json":
        with manifest_path.open("r", encoding="utf-8-sig") as handle:
            values = json.load(handle)
        if isinstance(values, dict):
            if "ids" not in values:
                raise ValueError(
                    f"ACDC JSON manifest must be a list or contain an 'ids' list: {manifest_path}"
                )
            values = values["ids"]
        if not isinstance(values, list):
            raise ValueError(f"ACDC JSON manifest must contain an ID list: {manifest_path}")
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
                        f"ACDC CSV manifest row {row_number} must contain exactly one ID column: "
                        f"{manifest_path}"
                    )
                rows.append(row[0].strip())
        if rows and rows[0].casefold() in _CSV_ID_HEADERS:
            rows = rows[1:]
        values = rows
    else:
        raise ValueError(
            f"Unsupported ACDC data manifest format '{suffix}'; use JSON, TXT, or CSV: "
            f"{manifest_path}"
        )

    ids = []
    for position, value in enumerate(values, 1):
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ValueError(
                f"ACDC manifest ID {position} must be a string or integer: {manifest_path}"
            )
        value = str(value).strip()
        if not value or value in {".", ".."} or "/" in value or chr(92) in value:
            raise ValueError(
                f"ACDC manifest ID {position} is not a valid ID-only value: {value!r}"
            )
        ids.append(value)
    if not ids:
        raise ValueError(f"ACDC data manifest is empty: {manifest_path}")
    duplicate_ids = sorted({value for value in ids if ids.count(value) > 1})
    if duplicate_ids:
        raise ValueError(
            f"Duplicate ACDC IDs in {manifest_path}: {', '.join(duplicate_ids)}"
        )
    return manifest_path, ids


def labels_from_foreground_channels(mask, num_classes=4, threshold=0.5):
    """Convert C-1 foreground channels to class-index labels."""
    if int(num_classes) != 4:
        raise ValueError(f"ACDC expects num_classes=4, got {num_classes}")

    if torch.is_tensor(mask):
        if mask.ndim == 4:
            if mask.shape[1] != num_classes - 1:
                raise ValueError(
                    f"Expected {num_classes - 1} foreground channels, got {tuple(mask.shape)}"
                )
            foreground_score, foreground_index = torch.max(mask, dim=1)
            return torch.where(
                foreground_score > threshold,
                foreground_index.long() + 1,
                torch.zeros_like(foreground_index, dtype=torch.long),
            )
        if mask.ndim == 3:
            return mask.round().long().clamp(min=0, max=num_classes - 1)
        raise ValueError(f"Expected tensor shape [B,C,H,W] or [B,H,W], got {tuple(mask.shape)}")

    mask = np.asarray(mask)
    if mask.ndim == 3:
        if mask.shape[0] == num_classes - 1:
            channels = mask
        elif mask.shape[-1] == num_classes - 1:
            channels = np.moveaxis(mask, -1, 0)
        else:
            raise ValueError(
                f"Expected {num_classes - 1} foreground channels, got shape {mask.shape}"
            )
        foreground_index = np.argmax(channels, axis=0)
        foreground_score = np.max(channels, axis=0)
        return np.where(foreground_score > threshold, foreground_index + 1, 0).astype(np.uint8)
    if mask.ndim == 2:
        return np.rint(mask).clip(0, num_classes - 1).astype(np.uint8)
    raise ValueError(f"Expected array shape [C,H,W], [H,W,C], or [H,W], got {mask.shape}")


def multiclass_dice_iou(pred_labels, target_labels, num_classes=4):
    rows = []
    for class_id in range(1, int(num_classes)):
        pred = pred_labels == class_id
        target = target_labels == class_id
        intersection = np.logical_and(pred, target).sum(dtype=np.float64)
        pred_sum = pred.sum(dtype=np.float64)
        target_sum = target.sum(dtype=np.float64)
        if target_sum == 0:
            dice = None
            iou = None
        else:
            dice_den = pred_sum + target_sum
            iou_den = pred_sum + target_sum - intersection
            dice = 0.0 if dice_den == 0 else float((2.0 * intersection) / dice_den)
            iou = 0.0 if iou_den == 0 else float(intersection / iou_den)
        rows.append(
            {
                "class_id": class_id,
                "class_name": ACDC_CLASS_NAMES[class_id],
                "dice": dice,
                "iou": iou,
                "pred_pixels": int(pred_sum),
                "gt_pixels": int(target_sum),
                "intersection": float(intersection),
                "is_empty_gt": int(target_sum == 0),
                "is_empty_pred": int(pred_sum == 0),
            }
        )
    return rows


def _split_candidates(root, split):
    split_key = (split or "").strip().lower()
    aliases = {
        "train": ("training", "train"),
        "training": ("training", "train"),
        "val": ("validation", "val"),
        "valid": ("validation", "val"),
        "validation": ("validation", "val"),
        "test": ("testing", "test"),
        "testing": ("testing", "test"),
    }.get(split_key, (split_key,))
    roots = [root]
    if (root / "ACDC").is_dir():
        roots.insert(0, root / "ACDC")
    return [base / name for base in roots for name in aliases if name]


def _filter_files_by_manifest(files, data_manifest):
    manifest_path, requested = _read_id_manifest(data_manifest)
    requested_ids = set(requested)
    indexed_files = []
    for path in files:
        matches = [
            item_id
            for item_id in requested
            if path.stem == item_id or path.stem.startswith(f"{item_id}_")
        ]
        if len(matches) > 1:
            raise ValueError(
                f"ACDC image stem {path.stem!r} ambiguously matches manifest IDs "
                f"{matches} in {manifest_path}"
            )
        if matches:
            indexed_files.append((matches[0], path))

    observed_ids = {item_id for item_id, _ in indexed_files}
    missing_ids = sorted(requested_ids - observed_ids)
    extra_ids = sorted(observed_ids - requested_ids)
    if missing_ids or extra_ids:
        raise FileNotFoundError(
            "ACDC manifest coverage mismatch against the physical image directory "
            f"({manifest_path}): missing={missing_ids}, extra={extra_ids}"
        )
    return [path for _, path in indexed_files], manifest_path


class ACDCDataset(Dataset):
    """ACDC multi-class 2D PNG loader for MedSegDiff V1.

    Masks are RGB foreground channels:
      R = right ventricle, G = myocardium, B = left ventricle.
    Background is the all-zero foreground state.
    """

    def __init__(
        self,
        args,
        data_path,
        transform=None,
        mode="Training",
        data_manifest=None,
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.transform = transform
        self.mode = mode
        self.image_size = int(getattr(args, "image_size", 256))
        self.num_seg_classes = int(getattr(args, "num_seg_classes", 4))
        if self.num_seg_classes != 4:
            raise ValueError(
                "ACDC loader expects num_seg_classes=4: background, RV, myocardium, LV."
            )
        self.num_mask_channels = self.num_seg_classes - 1
        split = getattr(args, "acdc_split", None)
        if split is None:
            split = "training" if mode.lower().startswith("train") else "testing"

        self.files = self._collect_files(self.data_path, split)
        if not self.files:
            raise FileNotFoundError(
                f"No ACDC PNG slices found under {self.data_path} for split '{split}'. "
                "Run scripts/prepare_acdc.py first or point --data_dir to the processed root."
            )

        self.skip_empty = bool(getattr(args, "acdc_skip_empty", False))
        if self.skip_empty:
            self.files = [path for path in self.files if self._has_foreground(path)]
            if not self.files:
                raise FileNotFoundError(f"No ACDC foreground slices under {self.data_path}.")

        manifest_value = (
            data_manifest
            if data_manifest is not None
            else getattr(args, "data_manifest", None)
        )
        self.data_manifest = None
        if manifest_value is not None and str(manifest_value).strip():
            self.files, manifest_path = _filter_files_by_manifest(
                self.files, manifest_value
            )
            self.data_manifest = str(manifest_path)

        self.foreground_pixels = [int(self._load_mask(path).sum()) for path in self.files]

    def _collect_files(self, root, split):
        for candidate in _split_candidates(root, split):
            image_dir = candidate / "images"
            if image_dir.is_dir():
                files = sorted(image_dir.glob("*.png"))
                if files:
                    return files
        return []

    def _mask_path(self, image_path):
        mask_path = image_path.parent.parent / "masks" / image_path.name
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing ACDC mask for image {image_path}: {mask_path}")
        return mask_path

    def _load_mask(self, image_path):
        mask = np.asarray(Image.open(self._mask_path(image_path)).convert("RGB"))
        mask = (mask > 127).astype(np.float32)
        if np.any(mask.sum(axis=2) > 1.0):
            raise ValueError(f"ACDC mask has overlapping foreground classes: {self._mask_path(image_path)}")
        return np.moveaxis(mask, -1, 0)

    def _has_foreground(self, image_path):
        return bool(self._load_mask(image_path).sum() > 0)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        image = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
        mask = self._load_mask(path)

        image = torch.from_numpy(np.moveaxis(image, -1, 0))
        mask = torch.from_numpy(mask)

        image = F.interpolate(
            image[None, ...],
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )[0]
        mask = F.interpolate(
            mask[None, ...],
            size=(self.image_size, self.image_size),
            mode="nearest",
        )[0]
        mask = torch.where(mask > 0.5, 1.0, 0.0)

        return image.float(), mask.float(), path.stem
