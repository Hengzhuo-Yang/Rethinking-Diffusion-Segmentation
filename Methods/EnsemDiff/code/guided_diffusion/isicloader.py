import os
from pathlib import Path

import numpy as np
import torch


def _load_image(path):
    array = np.load(path)
    tensor = torch.as_tensor(array, dtype=torch.float32)
    if tensor.ndim == 2:
        tensor = tensor[None, ...]
    elif tensor.ndim == 3 and tensor.shape[-1] in (1, 3):
        tensor = tensor.permute(2, 0, 1)
    elif tensor.ndim != 3:
        raise ValueError(f"Expected 2D or 3D image npy at {path}, got {tensor.shape}")
    return tensor


def _load_binary_mask(path):
    array = np.load(path)
    tensor = torch.as_tensor(array, dtype=torch.float32)
    if tensor.ndim == 2:
        tensor = tensor[None, ...]
    elif tensor.ndim == 3 and tensor.shape[0] == 1:
        pass
    elif tensor.ndim == 3 and tensor.shape[-1] == 1:
        tensor = tensor.permute(2, 0, 1)
    else:
        raise ValueError(f"Expected binary mask npy at {path}, got {tensor.shape}")
    return (tensor > 0.5).float()


def _crop_to_224(tensor):
    h, w = tensor.shape[-2:]
    if (h, w) == (224, 224):
        return tensor
    if h < 224 or w < 224:
        raise ValueError(f"Expected at least 224x224 spatial size, got {(h, w)}")
    top = (h - 224) // 2
    left = (w - 224) // 2
    return tensor[..., top : top + 224, left : left + 224]


class ISIC2018Dataset(torch.utils.data.Dataset):
    """ISIC 2018 Task 1 2D lesion segmentation for EnsemDiff.

    The preprocessor writes one directory per image with:
      *_image.npy: RGB dermoscopic image, normalized to [-1, 1].
      *_seg.npy: single-channel binary lesion mask, 0/1.
    """

    def __init__(self, directory, test_flag=True, manifest_path=""):
        super().__init__()
        self.directory = Path(os.path.expanduser(directory))
        self.test_flag = test_flag
        if not self.directory.exists():
            raise FileNotFoundError(f"ISIC2018 data directory does not exist: {self.directory}")

        allowed_images = None
        if manifest_path:
            manifest = Path(os.path.expanduser(manifest_path))
            if not manifest.is_file():
                raise FileNotFoundError(f"ISIC2018 manifest does not exist: {manifest}")
            image_ids = [
                line.strip()
                for line in manifest.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(image_ids) != len(set(image_ids)):
                raise ValueError(f"Duplicate ISIC2018 image IDs in manifest: {manifest}")
            allowed_images = set(image_ids)

        self.database = []
        observed_images = set()
        for image_path in sorted(self.directory.rglob("*_image.npy")):
            image_id = image_path.name[: -len("_image.npy")]
            if allowed_images is not None and image_id not in allowed_images:
                continue
            seg_path = image_path.with_name(image_path.name.replace("_image.npy", "_seg.npy"))
            if not test_flag and not seg_path.exists():
                raise FileNotFoundError(f"Missing ISIC2018 label for {image_path}: {seg_path}")
            self.database.append({"image": image_path, "seg": seg_path})
            observed_images.add(image_id)

        if allowed_images is not None:
            missing = sorted(allowed_images - observed_images)
            if missing:
                preview = ", ".join(missing[:10])
                raise RuntimeError(
                    f"{len(missing)} ISIC2018 manifest IDs were not found below "
                    f"{self.directory}; first missing: {preview}"
                )

        if not self.database:
            raise RuntimeError(f"No ISIC2018 *_image.npy files found under {self.directory}")

    def __getitem__(self, index):
        record = self.database[index]
        image = _crop_to_224(_load_image(record["image"]))
        if self.test_flag:
            return image, str(record["image"])

        label = _crop_to_224(_load_binary_mask(record["seg"]))
        return image, label

    def __len__(self):
        return len(self.database)
