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


def _default_acdc_root(split):
    env_root = os.environ.get("TSLDSEG_ACDC_PREPROCESSED_ROOT")
    candidates = []
    if env_root:
        candidates.append(Path(env_root) / "ACDC" / split)
    candidates.append(REPO_ROOT / "data" / "ACDC" / split)
    return _first_existing(candidates)


class ACDCBase(Dataset):
    """ACDC multi-class dataset using MT-UNet/CASCADE 70/10/20 PNG cache."""

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
        1: "right_ventricle",
        2: "myocardium",
        3: "left_ventricle",
    }

    def __init__(self, data_root, size=256, interpolation="nearest", mode=None,
                 num_classes=4, manifest_path=None):
        self.mode = mode
        self.num_classes = int(num_classes)
        self.data_root = str(data_root)
        self.size = int(size)
        self.interpolation = dict(nearest=PIL.Image.NEAREST)[interpolation]
        self.manifest_path = manifest_path
        if self.mode not in ["train", "val", "test"]:
            raise NotImplementedError(f"Only support ACDC split modes train/val/test; got {self.mode}")
        if self.num_classes != 4:
            raise ValueError(
                "ACDC is a 4-class segmentation dataset including background. "
                f"Use num_classes=4, got {self.num_classes}."
            )

        root = Path(self.data_root)
        self.image_dir = root / "images"
        self.label_dir = root / "label_maps"
        self.mask_dir = root / "masks"
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Missing ACDC image directory: {self.image_dir}")
        if not self.label_dir.is_dir() and not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Missing ACDC label_maps or masks directory under: {root}")

        image_paths = filter_image_paths(
            sorted(self.image_dir.glob("*.png")),
            manifest_path=self.manifest_path,
            dataset="acdc",
        )
        if not image_paths:
            raise FileNotFoundError(f"No ACDC PNG images found under {self.image_dir}")

        self.image_paths = [str(path) for path in image_paths]
        self.mask_paths = [str(self._mask_path_for_image(path)) for path in image_paths]
        missing = [path for path in self.mask_paths if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing ACDC label map for image, first missing path: {missing[0]}")

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

        print(f"[Dataset]: ACDC multi-class with {self.num_classes} classes, in {self.mode} mode")

    def __len__(self):
        return len(self.data_paths)

    def _mask_path_for_image(self, image_path):
        label_path = self.label_dir / image_path.name
        if label_path.is_file():
            return label_path
        return self.mask_dir / image_path.name

    def _resize_image(self, image, interpolation):
        if image.size == (self.size, self.size):
            return image
        return image.resize((self.size, self.size), resample=interpolation)

    def _load_image_png(self, path):
        image = Image.open(path).convert("RGB")
        image = self._resize_image(image, PIL.Image.BILINEAR)
        return (np.asarray(image, dtype=np.float32) / 127.5 - 1.0).astype(np.float32)

    def _load_label_map(self, path):
        image = Image.open(path)
        image = self._resize_image(image, self.interpolation)
        arr = np.asarray(image)
        if arr.ndim == 2:
            label = arr.astype(np.uint8)
        elif arr.ndim == 3 and arr.shape[2] >= 3:
            rgb = arr[:, :, :3]
            if np.all(rgb[:, :, 0] == rgb[:, :, 1]) and np.all(rgb[:, :, 0] == rgb[:, :, 2]):
                label = rgb[:, :, 0].astype(np.uint8)
            else:
                fg = rgb > 127
                overlap = fg.sum(axis=2) > 1
                if np.any(overlap):
                    raise ValueError(f"ACDC RGB mask has overlapping foreground classes: {path}")
                label = np.zeros(rgb.shape[:2], dtype=np.uint8)
                label[fg[:, :, 0]] = 1
                label[fg[:, :, 1]] = 2
                label[fg[:, :, 2]] = 3
        else:
            raise ValueError(f"ACDC mask must be 2D label map or RGB foreground mask, got shape={arr.shape}: {path}")

        unique = set(np.unique(label).tolist())
        allowed = set(range(self.num_classes))
        if not unique.issubset(allowed):
            raise ValueError(f"ACDC label values outside 0..3 in {path}: {sorted(unique)}")
        return label.astype(np.uint8)

    def _map_segmentation(self, label_map):
        present_classes = sorted(int(v) for v in np.unique(label_map))
        class_id = int(np.random.choice(np.array(present_classes, dtype=np.int64), size=1)[0])
        if class_id != 0:
            segmentation = label_map == class_id
        else:
            segmentation = label_map != class_id
        segmentation = np.repeat(segmentation[:, :, None], 3, axis=2)
        return segmentation.astype(np.uint8), np.array([class_id], dtype=np.int64)

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        label_map = self._load_label_map(example["mask_path_"])
        image = self._load_image_png(example["image_path_"])

        label_tensor = torch.tensor(label_map[None, :, :])
        image_tensor = torch.tensor(image.transpose([2, 0, 1]))
        if self.mode == "train":
            state = torch.get_rng_state()
            label_tensor = self.transform(label_tensor)
            torch.set_rng_state(state)
            image_tensor = self.transform(image_tensor)

        label_map = label_tensor[0].numpy().astype(np.uint8)
        image = image_tensor.permute(1, 2, 0).numpy().astype(np.float32)

        if self.mode == "test":
            segmentation = np.repeat(label_map[:, :, None], 3, axis=2).astype(np.float32)
            class_id = np.array([0], dtype=np.int64)
        else:
            segmentation, class_id = self._map_segmentation(label_map)
            segmentation = (segmentation.astype(np.float32) * 2.0) - 1.0

        example["class_id"] = class_id
        example["image"] = image
        example["segmentation"] = segmentation.astype(np.float32)

        image_min, image_max = float(example["image"].min()), float(example["image"].max())
        if not np.isfinite(image_min) or not np.isfinite(image_max):
            raise ValueError(f"Non-finite ACDC image values: {(image_min, image_max)}")
        seg_min, seg_max = float(example["segmentation"].min()), float(example["segmentation"].max())
        if self.mode == "test":
            if seg_min < 0 or seg_max > self.num_classes - 1:
                raise ValueError(f"ACDC test mask outside class range: {(seg_min, seg_max)}")
        elif seg_min < -1 or seg_max > 1:
            raise ValueError(f"ACDC train/val mask outside [-1, 1]: {(seg_min, seg_max)}")
        return example


class ACDCTrain(ACDCBase):
    def __init__(self, data_root=None, **kwargs):
        super().__init__(data_root=data_root or _default_acdc_root("training"), mode="train", **kwargs)


class ACDCValidation(ACDCBase):
    def __init__(self, data_root=None, split="validation", **kwargs):
        split = self.SPLIT_ALIASES.get(str(split).lower(), split)
        super().__init__(data_root=data_root or _default_acdc_root(split), mode="val", **kwargs)


class ACDCValidationEval(ACDCBase):
    def __init__(self, data_root=None, split="testing", **kwargs):
        split = self.SPLIT_ALIASES.get(str(split).lower(), split)
        super().__init__(data_root=data_root or _default_acdc_root(split), mode="test", **kwargs)
