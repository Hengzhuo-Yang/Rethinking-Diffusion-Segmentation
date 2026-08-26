import os
import re
from pathlib import Path

import numpy as np
import torch


def _load_npy(path):
    array = np.load(path)
    tensor = torch.as_tensor(array, dtype=torch.float32)
    if tensor.ndim == 2:
        tensor = tensor[None, ...]
    elif tensor.ndim == 3 and tensor.shape[-1] == 1:
        tensor = tensor.permute(2, 0, 1)
    elif tensor.ndim != 3:
        raise ValueError(f"Expected 2D or 3D npy tensor at {path}, got {tensor.shape}")
    return tensor


def _crop_to_224(tensor):
    h, w = tensor.shape[-2:]
    if (h, w) == (224, 224):
        return tensor
    if h < 224 or w < 224:
        raise ValueError(f"Expected at least 224x224 spatial size, got {(h, w)}")
    top = (h - 224) // 2
    left = (w - 224) // 2
    return tensor[..., top : top + 224, left : left + 224]


class BTCVDataset(torch.utils.data.Dataset):
    """BTCV/Synapse 2D slices for EnsemDiff binary segmentation.

    The preprocessor writes one directory per slice with:
      *_image.npy: single-channel CT slice, normalized as configured.
      *_seg.npy: binary target mask, 0/1.
    """

    def __init__(self, directory, test_flag=True, manifest_path=""):
        super().__init__()
        self.directory = Path(os.path.expanduser(directory))
        self.test_flag = test_flag
        if not self.directory.exists():
            raise FileNotFoundError(f"BTCV data directory does not exist: {self.directory}")

        allowed_cases = None
        if manifest_path:
            manifest = Path(os.path.expanduser(manifest_path))
            if not manifest.is_file():
                raise FileNotFoundError(f"BTCV manifest does not exist: {manifest}")
            case_ids = [
                line.strip()
                for line in manifest.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(case_ids) != len(set(case_ids)):
                raise ValueError(f"Duplicate BTCV case IDs in manifest: {manifest}")
            allowed_cases = set(case_ids)
        self.database = []
        observed_cases = set()
        for image_path in sorted(self.directory.rglob("*_image.npy")):
            match = re.search(r"vol(\d{4})_", image_path.name)
            if not match:
                raise ValueError(f"Could not parse BTCV case ID from {image_path}")
            case_id = f"case{match.group(1)}"
            if allowed_cases is not None and case_id not in allowed_cases:
                continue
            seg_path = image_path.with_name(image_path.name.replace("_image.npy", "_seg.npy"))
            if not test_flag and not seg_path.exists():
                raise FileNotFoundError(f"Missing BTCV label for {image_path}: {seg_path}")
            self.database.append({"image": image_path, "seg": seg_path})
            observed_cases.add(case_id)

        if allowed_cases is not None:
            missing = sorted(allowed_cases - observed_cases)
            if missing:
                raise RuntimeError(
                    f"BTCV manifest cases not found below {self.directory}: {missing}"
                )

        if not self.database:
            raise RuntimeError(f"No BTCV *_image.npy slices found under {self.directory}")

    def __getitem__(self, index):
        record = self.database[index]
        image = _crop_to_224(_load_npy(record["image"]))
        if self.test_flag:
            return image, str(record["image"])

        label = _crop_to_224(_load_npy(record["seg"]))
        label = torch.where(label > 0, 1, 0).float()
        return image, label

    def __len__(self):
        return len(self.database)
