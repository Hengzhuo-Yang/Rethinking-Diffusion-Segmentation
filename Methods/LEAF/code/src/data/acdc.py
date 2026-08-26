import os
import random
from pathlib import Path

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset


class ACDC(Dataset):
    """LEAF-owned ACDC multi-class cache.

    Expected layout:
      <dataset_dir>/training/images/*.png
      <dataset_dir>/training/masks/*.png
      <dataset_dir>/validation/images/*.png
      <dataset_dir>/validation/masks/*.png
      <dataset_dir>/testing/images/*.png
      <dataset_dir>/testing/masks/*.png

    Masks are RGB PNGs with three foreground channels:
      R = right ventricle, G = myocardium, B = left ventricle.
    Background is represented by all foreground channels equal to zero.
    """

    SPLIT_ALIASES = {
        "train": "training",
        "training": "training",
        "val": "validation",
        "valid": "validation",
        "validation": "validation",
        "test": "testing",
        "testing": "testing",
    }

    def __init__(
        self,
        dataset_dir: str,
        split: str,
        resolution: int = 256,
        seed: int = 42,
        num_seg_classes: int = 4,
    ) -> None:
        super().__init__()
        if int(num_seg_classes) != 4:
            raise ValueError(
                "ACDC loader expects num_seg_classes=4: background, RV, myocardium, LV."
            )

        split_key = str(split).lower()
        if split_key not in self.SPLIT_ALIASES:
            raise ValueError(f"Unsupported ACDC split={split!r}")
        split_dir_name = self.SPLIT_ALIASES[split_key]

        self.dataset_dir = Path(dataset_dir)
        self.split = split_dir_name
        self.seed = seed
        self.resolution = int(resolution)
        self.image_dir = self.dataset_dir / split_dir_name / "images"
        self.mask_dir = self.dataset_dir / split_dir_name / "masks"
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Missing ACDC image directory: {self.image_dir}")
        if not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Missing ACDC mask directory: {self.mask_dir}")

        base_filenames = sorted(
            path.stem for path in self.image_dir.iterdir() if path.suffix.lower() == ".png"
        )
        if not base_filenames:
            raise RuntimeError(f"No ACDC PNG images found under {self.image_dir}")

        self.image_filenames = [str(self.image_dir / f"{name}.png") for name in base_filenames]
        self.mask_filenames = [str(self.mask_dir / f"{name}.png") for name in base_filenames]
        missing_masks = [path for path in self.mask_filenames if not os.path.exists(path)]
        if missing_masks:
            raise FileNotFoundError(f"Missing ACDC masks, first missing path: {missing_masks[0]}")

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize((self.resolution, self.resolution)),
                transforms.ToTensor(),
            ]
        )
        self.mask_resize = transforms.Resize(
            (self.resolution, self.resolution),
            transforms.InterpolationMode.NEAREST_EXACT,
        )

    def __len__(self):
        return len(self.image_filenames)

    def _flip(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if random.random() > 0.5:
            image, mask = F.hflip(image), F.hflip(mask)
        if random.random() > 0.5:
            image, mask = F.vflip(image), F.vflip(mask)
        return image, mask

    def __getitem__(self, index: int):
        image = Image.open(self.image_filenames[index]).convert("RGB")
        mask = Image.open(self.mask_filenames[index]).convert("RGB")

        image_tensor = self.image_transforms(image)
        mask = self.mask_resize(mask)
        mask_tensor = F.pil_to_tensor(mask).float() / 255.0
        mask_tensor = torch.where(mask_tensor > 0.5, 1.0, 0.0)
        if torch.any(mask_tensor.sum(dim=0) > 1.0):
            raise ValueError(
                f"ACDC mask has overlapping foreground classes: {self.mask_filenames[index]}"
            )

        if self.split == "training":
            image_tensor, mask_tensor = self._flip(image_tensor, mask_tensor)

        return {
            "pixel_values": image_tensor,
            "mask_values": mask_tensor,
        }
