import glob
import os
from pathlib import Path

import numpy as np
import PIL
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


REPO_ROOT = Path(__file__).resolve().parents[2]


def _first_existing(candidates):
    for path in candidates:
        if path.is_dir():
            return str(path)
    return str(candidates[0])


def _default_isic2018_root(split):
    data_root = Path(os.environ.get("SDSEG_DATA_ROOT", REPO_ROOT / "data"))
    candidates = [data_root / "isic2018" / split]
    return _first_existing(candidates)


class ISIC2018Base(Dataset):
    """SDSeg-owned ISIC2018 Task 1 binary segmentation cache.

    Expected layout:
      data/isic2018/training/images/*.png and masks/*.png
      data/isic2018/validation/images/*.png and masks/*.png
      data/isic2018/testing/images/*.png and masks/*.png

    Images are stored as RGB uint8 PNGs and scaled to [-1, 1] at load time.
    PNG masks store binary labels as 0 or 255 and are expanded to SDSeg's
    VAE-compatible 3-channel segmentation representation.
    """

    def __init__(self, data_root, size=256, interpolation="nearest", mode=None, num_classes=2):
        if int(num_classes) != 2:
            raise ValueError("ISIC2018 loader expects num_classes=2: background and lesion.")
        if mode not in ["train", "val", "test"]:
            raise ValueError(f"Unsupported ISIC2018 mode={mode!r}")
        self.data_root = data_root
        self.mode = mode
        self.num_classes = int(num_classes)
        self.size = int(size)
        self.interpolation = dict(nearest=PIL.Image.NEAREST)[interpolation]
        image_dir = Path(self.data_root) / "images"
        mask_dir = Path(self.data_root) / "masks"
        if image_dir.is_dir() and mask_dir.is_dir():
            image_paths = sorted(image_dir.glob("*.png"))
            mask_paths = [mask_dir / image_path.name for image_path in image_paths]
            missing = [path for path in mask_paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Missing ISIC2018 mask for image, first missing path: {missing[0]}")
            self.image_paths = [str(path) for path in image_paths]
            self.mask_paths = [str(path) for path in mask_paths]
            self.data_paths = self.mask_paths
        else:
            self.data_paths = sorted(glob.glob(os.path.join(self.data_root, "*.png")))
            self.mask_paths = self.data_paths
            self.image_paths = [path.replace(".png", ".npy") for path in self.mask_paths]
        if not self.data_paths:
            raise RuntimeError(f"No ISIC2018 mask samples found under {self.data_root}")
        self.labels = {
            "file_path_": [path for path in self.data_paths],
            "image_path_": [path for path in self.image_paths],
            "mask_path_": [path for path in self.mask_paths],
        }
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ])
        print(f"[Dataset]: ISIC2018 with {self.num_classes} classes, in {self.mode} mode, n={len(self.data_paths)}")

    def __len__(self):
        return len(self.data_paths)

    @staticmethod
    def _load_binary_label(path):
        segmentation = np.array(Image.open(path))
        if segmentation.ndim == 3:
            if not np.all(segmentation[:, :, 0] == segmentation[:, :, 1]) or not np.all(segmentation[:, :, 0] == segmentation[:, :, 2]):
                raise ValueError(f"ISIC2018 binary mask RGB channels disagree: {path}")
            segmentation = segmentation[:, :, 0]
        if segmentation.ndim != 2:
            raise ValueError(f"ISIC2018 mask must be 2D or repeated RGB, got shape={segmentation.shape}: {path}")
        return (segmentation > 127).astype(np.uint8)

    @staticmethod
    def _load_image(path):
        if str(path).lower().endswith(".npy"):
            return np.load(path).astype(np.float32)
        image = Image.open(path).convert("RGB")
        return (np.asarray(image, dtype=np.float32) / 127.5 - 1.0).astype(np.float32)

    @staticmethod
    def _resize_image(image, size):
        uint8 = np.round(np.clip((image + 1.0) * 127.5, 0, 255)).astype(np.uint8)
        pil_image = Image.fromarray(uint8, mode="RGB")
        pil_image = pil_image.resize((size, size), resample=PIL.Image.BILINEAR)
        return (np.asarray(pil_image, dtype=np.float32) / 127.5 - 1.0).astype(np.float32)

    @staticmethod
    def _resize_label(label, size):
        pil_label = Image.fromarray((label * 255).astype(np.uint8), mode="L")
        pil_label = pil_label.resize((size, size), resample=PIL.Image.NEAREST)
        return (np.asarray(pil_label, dtype=np.uint8) > 127).astype(np.uint8)

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        mask_path = example["mask_path_"]
        image_path = example["image_path_"]
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Missing ISIC2018 image array for mask {mask_path}: {image_path}")

        label_map = self._load_binary_label(mask_path)
        image = self._load_image(image_path)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"ISIC2018 image must be HxWx3, got shape={image.shape}: {image_path}")
        if image.shape[:2] != label_map.shape:
            raise ValueError(
                f"ISIC2018 image/mask shape mismatch: image={image.shape}, "
                f"mask={label_map.shape}, path={mask_path}"
            )
        if image.shape[0] != self.size or image.shape[1] != self.size:
            image = self._resize_image(image, self.size)
            label_map = self._resize_label(label_map, self.size)

        label_rgb = np.repeat(label_map[:, :, None], 3, axis=2).astype(np.float32)
        image_t = torch.tensor(image.transpose([2, 0, 1]))
        label_t = torch.tensor(label_rgb.transpose([2, 0, 1]))
        if self.mode == "train":
            state = torch.get_rng_state()
            label_t = self.transform(label_t)
            torch.set_rng_state(state)
            image_t = self.transform(image_t)

        label_rgb = np.array(label_t.permute(1, 2, 0), dtype=np.float32)
        image = np.array(image_t.permute(1, 2, 0), dtype=np.float32)
        if np.max(image) > 1.001 or np.min(image) < -1.001:
            raise ValueError(
                f"ISIC2018 image must be normalized to [-1, 1], got "
                f"{image.min()}..{image.max()} from {image_path}"
            )
        if not set(np.unique(label_rgb).astype(int).tolist()).issubset({0, 1}):
            raise ValueError(f"ISIC2018 masks must be binary after loading: {mask_path}")

        example["class_id"] = np.array([-1])
        example["image"] = image
        if self.mode == "test":
            example["segmentation"] = label_rgb
        else:
            example["segmentation"] = (label_rgb * 2.0) - 1.0
        return example


class ISIC2018Train(ISIC2018Base):
    def __init__(self, **kwargs):
        super().__init__(data_root=_default_isic2018_root("training"), mode="train", **kwargs)


class ISIC2018Validation(ISIC2018Base):
    def __init__(self, **kwargs):
        super().__init__(data_root=_default_isic2018_root("validation"), mode="val", **kwargs)


class ISIC2018ValidationEval(ISIC2018Base):
    def __init__(self, **kwargs):
        super().__init__(data_root=_default_isic2018_root("validation"), mode="test", **kwargs)


class ISIC2018Test(ISIC2018Base):
    def __init__(self, **kwargs):
        super().__init__(data_root=_default_isic2018_root("testing"), mode="test", **kwargs)
