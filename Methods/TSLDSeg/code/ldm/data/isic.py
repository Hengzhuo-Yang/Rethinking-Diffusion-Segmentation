import os
from pathlib import Path

# Release modification: fixed ID manifests and portable paths; see MODIFICATIONS.md.

import numpy as np
import PIL
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from ldm.data.manifest import filter_image_paths


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT.parent


def _first_existing(candidates):
    for path in candidates:
        if path.is_dir():
            return str(path)
    return str(candidates[0])


def _default_isic_root(split):
    env_root = os.environ.get("TSLDSEG_ISIC2018_PREPROCESSED_ROOT") or os.environ.get("TSLDSEG_ISIC_PREPROCESSED_ROOT")
    candidates = []
    if env_root:
        candidates.append(Path(env_root) / "ISIC18" / split)
    candidates.append(REPO_ROOT / "data" / "ISIC18" / split)
    return _first_existing(candidates)


class ISICBase(Dataset):
    """ISIC2018 Task 1 binary lesion segmentation PNG cache.

    Expected split layout:
      <root>/images/*.png
      <root>/masks/*.png

    Training/validation masks are binary foreground/background masks expanded to
    three channels for the KL-f8 first-stage VAE and mapped to [-1, 1]. Test
    masks stay in {0, 1} so Dice/IoU evaluation can threshold foreground only.
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

    CLASS_NAMES = {
        0: "background",
        1: "lesion",
    }

    def __init__(self, data_root=None, size=256, interpolation="nearest", mode=None,
                 num_classes=2, year=18, manifest_path=None):
        self.mode = mode
        self.num_classes = int(num_classes)
        self.year = int(year)
        self.size = int(size)
        self.interpolation = dict(nearest=PIL.Image.NEAREST)[interpolation]
        self.manifest_path = manifest_path
        if self.mode not in ["train", "val", "test"]:
            raise NotImplementedError(f"Only support ISIC split modes train/val/test; got {self.mode}")
        if self.year != 18:
            raise ValueError(f"This local reproduction uses ISIC2018 Task 1 only; got year={self.year}")
        if self.num_classes != 2:
            raise ValueError(
                "ISIC2018 Task 1 is binary lesion segmentation including background. "
                f"Use num_classes=2, got {self.num_classes}."
            )

        self.data_root = str(data_root)
        root = Path(self.data_root)
        self.image_dir = root / "images"
        self.mask_dir = root / "masks"
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Missing ISIC2018 image directory: {self.image_dir}")
        if not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Missing ISIC2018 mask directory: {self.mask_dir}")

        image_paths = filter_image_paths(
            sorted(self.image_dir.glob("*.png")),
            manifest_path=self.manifest_path,
            dataset="isic2018",
        )
        if not image_paths:
            raise FileNotFoundError(f"No ISIC2018 PNG images found under {self.image_dir}")

        self.image_paths = [str(path) for path in image_paths]
        self.mask_paths = [str(self.mask_dir / path.name) for path in image_paths]
        missing = [path for path in self.mask_paths if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing ISIC2018 mask for image, first missing path: {missing[0]}")

        self.data_paths = self.mask_paths
        self.labels = {
            "file_path_": list(self.data_paths),
            "image_path_": list(self.image_paths),
            "mask_path_": list(self.mask_paths),
        }
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ])
        print(f"[Dataset]: ISIC2018 Task 1 binary with {self.num_classes} classes, in {self.mode} mode")

    def __len__(self):
        return len(self.data_paths)

    def _resize_image(self, image, interpolation):
        if image.size == (self.size, self.size):
            return image
        return image.resize((self.size, self.size), resample=interpolation)

    def _load_image_png(self, path):
        image = Image.open(path).convert("RGB")
        image = self._resize_image(image, PIL.Image.BILINEAR)
        return (np.asarray(image, dtype=np.float32) / 127.5 - 1.0).astype(np.float32)

    def _load_binary_mask(self, path):
        mask = Image.open(path).convert("L")
        mask = self._resize_image(mask, self.interpolation)
        arr = np.asarray(mask, dtype=np.uint8)
        return np.where(arr > 127, 1, 0).astype(np.uint8)

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        mask = self._load_binary_mask(example["mask_path_"])
        image = self._load_image_png(example["image_path_"])

        mask_tensor = torch.tensor(mask[None, :, :])
        image_tensor = torch.tensor(image.transpose([2, 0, 1]))
        if self.mode == "train":
            state = torch.get_rng_state()
            mask_tensor = self.transform(mask_tensor)
            torch.set_rng_state(state)
            image_tensor = self.transform(image_tensor)

        mask = mask_tensor[0].numpy().astype(np.uint8)
        image = image_tensor.permute(1, 2, 0).numpy().astype(np.float32)
        segmentation = np.repeat(mask[:, :, None], 3, axis=2).astype(np.float32)
        if self.mode != "test":
            segmentation = (segmentation * 2.0) - 1.0

        example["class_id"] = np.array([-1], dtype=np.int64)
        example["image"] = image
        example["segmentation"] = segmentation.astype(np.float32)

        image_min, image_max = float(example["image"].min()), float(example["image"].max())
        if not np.isfinite(image_min) or not np.isfinite(image_max):
            raise ValueError(f"Non-finite ISIC2018 image values: {(image_min, image_max)}")
        seg_min, seg_max = float(example["segmentation"].min()), float(example["segmentation"].max())
        if self.mode == "test":
            if seg_min < 0 or seg_max > 1:
                raise ValueError(f"ISIC2018 test mask outside binary range: {(seg_min, seg_max)}")
        elif seg_min < -1 or seg_max > 1:
            raise ValueError(f"ISIC2018 train/val mask outside [-1, 1]: {(seg_min, seg_max)}")
        return example


class ISICTrain(ISICBase):
    def __init__(self, data_root=None, split="training", **kwargs):
        split = self.SPLIT_ALIASES.get(str(split).lower(), split)
        super().__init__(data_root=data_root or _default_isic_root(split), mode="train", **kwargs)


class ISICValidation(ISICBase):
    def __init__(self, data_root=None, split="validation", **kwargs):
        split = self.SPLIT_ALIASES.get(str(split).lower(), split)
        super().__init__(data_root=data_root or _default_isic_root(split), mode="val", **kwargs)


class ISICTest(ISICBase):
    def __init__(self, data_root=None, split="testing", **kwargs):
        split = self.SPLIT_ALIASES.get(str(split).lower(), split)
        super().__init__(data_root=data_root or _default_isic_root(split), mode="test", **kwargs)
