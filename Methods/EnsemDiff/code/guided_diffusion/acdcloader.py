import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _load_image(path):
    array = np.load(path)
    tensor = torch.as_tensor(array, dtype=torch.float32)
    if tensor.ndim == 2:
        tensor = tensor[None, ...]
    elif tensor.ndim == 3 and tensor.shape[-1] == 1:
        tensor = tensor.permute(2, 0, 1)
    elif tensor.ndim != 3:
        raise ValueError(f"Expected 2D or 3D image npy at {path}, got {tensor.shape}")
    return tensor


def _load_label(path, num_classes):
    array = np.load(path)
    tensor = torch.as_tensor(array)
    if tensor.ndim == 2:
        labels = tensor.long().clamp(min=0, max=num_classes - 1)
        return F.one_hot(labels, num_classes=num_classes).permute(2, 0, 1).float()
    if tensor.ndim == 3 and tensor.shape[0] == num_classes:
        return tensor.float()
    if tensor.ndim == 3 and tensor.shape[-1] == num_classes:
        return tensor.permute(2, 0, 1).float()
    raise ValueError(
        f"Expected label map or {num_classes}-channel one-hot npy at {path}, "
        f"got {tensor.shape}"
    )


def _crop_to_224(tensor):
    h, w = tensor.shape[-2:]
    if (h, w) == (224, 224):
        return tensor
    if h < 224 or w < 224:
        raise ValueError(f"Expected at least 224x224 spatial size, got {(h, w)}")
    top = (h - 224) // 2
    left = (w - 224) // 2
    return tensor[..., top : top + 224, left : left + 224]


class ACDCDataset(torch.utils.data.Dataset):
    """ACDC 2D slices for EnsemDiff multi-class segmentation.

    The preprocessor writes one directory per slice with:
      *_image.npy: single-channel cardiac MRI slice, resized to 224x224.
      *_seg.npy: 4-channel one-hot target mask for background/RV/Myo/LV.
    """

    def __init__(self, directory, test_flag=True, num_classes=4, manifest_path=""):
        super().__init__()
        self.directory = Path(os.path.expanduser(directory))
        self.test_flag = test_flag
        self.num_classes = num_classes
        if not self.directory.exists():
            raise FileNotFoundError(f"ACDC data directory does not exist: {self.directory}")

        allowed_patients = None
        if manifest_path:
            manifest = Path(os.path.expanduser(manifest_path))
            if not manifest.is_file():
                raise FileNotFoundError(f"ACDC manifest does not exist: {manifest}")
            patient_ids = [
                line.strip()
                for line in manifest.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(patient_ids) != len(set(patient_ids)):
                raise ValueError(f"Duplicate ACDC patient IDs in manifest: {manifest}")
            allowed_patients = set(patient_ids)

        self.database = []
        observed_patients = set()
        for image_path in sorted(self.directory.rglob("*_image.npy")):
            match = re.search(r"patient\d{3}", image_path.name)
            if not match:
                raise ValueError(f"Could not parse ACDC patient ID from {image_path}")
            patient_id = match.group(0)
            if allowed_patients is not None and patient_id not in allowed_patients:
                continue
            seg_path = image_path.with_name(image_path.name.replace("_image.npy", "_seg.npy"))
            if not test_flag and not seg_path.exists():
                raise FileNotFoundError(f"Missing ACDC label for {image_path}: {seg_path}")
            self.database.append({"image": image_path, "seg": seg_path})
            observed_patients.add(patient_id)

        if allowed_patients is not None:
            missing = sorted(allowed_patients - observed_patients)
            if missing:
                raise RuntimeError(
                    f"ACDC manifest patients not found below {self.directory}: {missing}"
                )

        if not self.database:
            raise RuntimeError(f"No ACDC *_image.npy slices found under {self.directory}")

    def __getitem__(self, index):
        record = self.database[index]
        image = _crop_to_224(_load_image(record["image"]))
        if self.test_flag:
            return image, str(record["image"])

        label = _crop_to_224(_load_label(record["seg"], self.num_classes))
        return image, label

    def __len__(self):
        return len(self.database)
