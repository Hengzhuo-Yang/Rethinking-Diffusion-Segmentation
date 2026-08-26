import os
import re
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


def _default_acdc_root(split):
    data_root = Path(os.environ.get("SDSEG_DATA_ROOT", REPO_ROOT / "data"))
    candidates = [data_root / "acdc" / split]
    return _first_existing(candidates)


class ACDCBase(Dataset):
    """SDSeg-owned ACDC cache.

    2D split layout:
      data/acdc/train/images/*.png and masks/*.png
      data/acdc/validation/images/*.png and masks/*.png
      data/acdc/test/images/*.png and masks/*.png

    Volume split layout:
      volume evaluation is reconstructed from split PNG slices, grouped by
      frame id before the `_zNNN` suffix.

    PNG masks store class-index labels 0..3. The class-conditional SDSeg
    training path converts each sample to a binary mask for one selected class
    without collapsing the dataset definition to binary.
    """

    def __init__(self, data_root, size=256, interpolation="nearest", mode=None, num_classes=4):
        if int(num_classes) != 4:
            raise ValueError("ACDC loader expects num_classes=4: background, RV, myocardium, LV.")
        self.mode = mode
        self.num_classes = int(num_classes)
        self.data_root = data_root
        self.size = int(size)
        self.interpolation = dict(nearest=PIL.Image.NEAREST)[interpolation]
        if mode not in ["train", "val", "test", "test_vol"]:
            raise ValueError(f"Unsupported ACDC mode={mode!r}")

        self.root_path = Path(self.data_root)
        self.volume_from_slices = False
        self.volume_samples = []
        if mode == "test_vol":
            self._init_volume_paths()
        else:
            self._init_slice_paths()
        if not self.data_paths:
            raise RuntimeError(f"No ACDC samples found under {self.data_root}")

        self.labels = {
            "file_path_": [path for path in self.data_paths],
            "image_path_": [path for path in self.image_paths],
            "mask_path_": [path for path in self.mask_paths],
        }
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ])
        print(f"[Dataset]: ACDC with {self.num_classes} classes, in {self.mode} mode, n={len(self.data_paths)}")

    def __len__(self):
        return len(self.data_paths)

    @staticmethod
    def _case_id_from_slice(path: Path) -> str:
        return re.sub(r"_z\d+$", "", path.stem)

    @staticmethod
    def _slice_index(path: Path) -> int:
        match = re.search(r"_z(\d+)$", path.stem)
        if not match:
            raise ValueError(f"Could not parse ACDC z-index from slice name: {path}")
        return int(match.group(1))

    def _init_slice_paths(self):
        image_dir = self.root_path / "images"
        mask_dir = self.root_path / "label_maps"
        if not mask_dir.is_dir():
            mask_dir = self.root_path / "masks"
        if image_dir.is_dir() and mask_dir.is_dir():
            image_paths = sorted(image_dir.glob("*.png"))
            mask_paths = [mask_dir / image_path.name for image_path in image_paths]
            missing = [path for path in mask_paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Missing ACDC mask for image, first missing path: {missing[0]}")
            self.image_paths = [str(path) for path in image_paths]
            self.mask_paths = [str(path) for path in mask_paths]
            self.data_paths = self.mask_paths
            return

        mask_paths = sorted(self.root_path.glob("*.png"))
        self.mask_paths = [str(path) for path in mask_paths]
        self.image_paths = [str(path.with_suffix(".npy")) for path in mask_paths]
        self.data_paths = self.mask_paths

    def _init_volume_paths(self):
        npz_paths = sorted(self.root_path.glob("*.npz"))
        if npz_paths:
            self.image_paths = [str(path) for path in npz_paths]
            self.mask_paths = [str(path) for path in npz_paths]
            self.data_paths = [str(path) for path in npz_paths]
            return

        image_dir = self.root_path / "images"
        mask_dir = self.root_path / "label_maps"
        if not mask_dir.is_dir():
            mask_dir = self.root_path / "masks"
        if not image_dir.is_dir() or not mask_dir.is_dir():
            self.image_paths = []
            self.mask_paths = []
            self.data_paths = []
            return

        grouped = {}
        for image_path in sorted(image_dir.glob("*.png")):
            mask_path = mask_dir / image_path.name
            if not mask_path.is_file():
                raise FileNotFoundError(f"Missing ACDC volume mask for image: {mask_path}")
            case_id = self._case_id_from_slice(image_path)
            grouped.setdefault(case_id, []).append((self._slice_index(image_path), image_path, mask_path))

        self.volume_from_slices = True
        self.volume_samples = []
        for case_id in sorted(grouped):
            items = sorted(grouped[case_id], key=lambda item: item[0])
            image_paths = [item[1] for item in items]
            mask_paths = [item[2] for item in items]
            self.volume_samples.append({"case_id": case_id, "image_paths": image_paths, "mask_paths": mask_paths})
        self.data_paths = [str(sample["image_paths"][0]) for sample in self.volume_samples]
        self.image_paths = self.data_paths
        self.mask_paths = [str(sample["mask_paths"][0]) for sample in self.volume_samples]

    @staticmethod
    def _validate_label_values(segmentation, num_classes, path):
        if segmentation.size == 0:
            raise ValueError(f"Empty ACDC segmentation loaded from {path}")
        min_label = int(np.min(segmentation))
        max_label = int(np.max(segmentation))
        if min_label < 0 or max_label >= num_classes:
            raise ValueError(
                f"ACDC label values must be in [0, {num_classes - 1}], "
                f"got min={min_label}, max={max_label}, path={path}"
            )

    @staticmethod
    def _class_conditional_mask(segmentation, class_id):
        if int(class_id) != 0:
            return segmentation == int(class_id)
        return segmentation != 0

    @staticmethod
    def _load_image_png(path):
        image = Image.open(path).convert("RGB")
        return (np.asarray(image, dtype=np.float32) / 127.5 - 1.0).astype(np.float32)

    @staticmethod
    def _load_label_png(path):
        segmentation = np.array(Image.open(path))
        if segmentation.ndim == 2:
            return segmentation.astype(np.uint8)
        if segmentation.ndim == 3:
            if not np.all(segmentation[:, :, 0] == segmentation[:, :, 1]) or not np.all(segmentation[:, :, 0] == segmentation[:, :, 2]):
                raise ValueError(f"ACDC class-index PNG channels disagree: {path}")
            return segmentation[:, :, 0].astype(np.uint8)
        raise ValueError(f"ACDC mask must be 2D or repeated RGB, got shape={segmentation.shape}: {path}")

    def _load_slice(self, example):
        mask_path = example["mask_path_"]
        image_path = example["image_path_"]
        label_map = self._load_label_png(mask_path)
        if str(image_path).lower().endswith(".npy"):
            image = np.load(image_path)
        else:
            image = self._load_image_png(image_path)
        segmentation = np.repeat(label_map[:, :, None], 3, axis=2)
        if image.shape[:2] != segmentation.shape[:2]:
            raise ValueError(
                f"ACDC image/mask shape mismatch: image={image.shape}, "
                f"mask={segmentation.shape}, path={mask_path}"
            )

        segmentation_t = torch.tensor(segmentation.transpose([2, 0, 1]))
        image_t = torch.tensor(image.transpose([2, 0, 1]))
        if self.mode == "train":
            state = torch.get_rng_state()
            segmentation_t = self.transform(segmentation_t)
            torch.set_rng_state(state)
            image_t = self.transform(image_t)

        segmentation = np.array(segmentation_t.permute(1, 2, 0))
        image = np.array(image_t.permute(1, 2, 0), dtype=np.float32)
        label_map = segmentation[:, :, 0].astype(np.uint8)
        self._validate_label_values(label_map, self.num_classes, mask_path)

        exist_class = np.array(sorted(set(label_map.flatten().astype(int))), dtype=np.int64)
        class_id = np.random.choice(exist_class, size=1, p=None).astype(np.int64)
        mask = self._class_conditional_mask(label_map, class_id[0])

        example["class_id"] = class_id
        example["segmentation"] = ((mask.astype(np.float32) * 2.0) - 1.0)
        example["segmentation"] = np.repeat(example["segmentation"][:, :, None], 3, axis=2)
        example["image"] = image
        if np.max(example["image"]) > 1.001 or np.min(example["image"]) < -1.001:
            raise ValueError(
                f"ACDC image must be normalized to [-1, 1], got "
                f"{example['image'].min()}..{example['image'].max()} from {image_path}"
            )
        return example

    def _load_volume(self, example):
        sample = np.load(example["file_path_"], allow_pickle=False)
        image = sample["image"].astype(np.float32)
        segmentation = sample["label"].astype(np.uint8)
        if image.shape != segmentation.shape:
            raise ValueError(
                f"ACDC volume image/mask shape mismatch: image={image.shape}, "
                f"mask={segmentation.shape}, path={example['file_path_']}"
            )
        self._validate_label_values(segmentation, self.num_classes, example["file_path_"])
        example["image"] = image
        example["segmentation"] = segmentation
        return example

    def _load_volume_from_slices(self, i):
        sample = self.volume_samples[i]
        image_slices = []
        label_slices = []
        for image_path, mask_path in zip(sample["image_paths"], sample["mask_paths"]):
            image = self._load_image_png(image_path)
            label = self._load_label_png(mask_path)
            if image.shape[2] != 3:
                raise ValueError(f"ACDC slice image must be RGB, got shape={image.shape}: {image_path}")
            image_slices.append(image[:, :, 0].astype(np.float32))
            label_slices.append(label.astype(np.uint8))
        image_volume = np.stack(image_slices, axis=2).astype(np.float32)
        label_volume = np.stack(label_slices, axis=2).astype(np.uint8)
        self._validate_label_values(label_volume, self.num_classes, sample["case_id"])
        return {
            "file_path_": str(sample["image_paths"][0]),
            "image": image_volume,
            "segmentation": label_volume,
        }

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        if self.mode == "test_vol":
            if self.volume_from_slices:
                return self._load_volume_from_slices(i)
            return self._load_volume(example)
        return self._load_slice(example)


class ACDCTrain(ACDCBase):
    def __init__(self, **kwargs):
        super().__init__(data_root=_default_acdc_root("train"), mode="train", **kwargs)


class ACDCValidation(ACDCBase):
    def __init__(self, **kwargs):
        super().__init__(data_root=_default_acdc_root("validation"), mode="val", **kwargs)


class ACDCTest(ACDCBase):
    def __init__(self, **kwargs):
        super().__init__(data_root=_default_acdc_root("test"), mode="test", **kwargs)



class ACDCFullLabelSliceEval(ACDCBase):
    SPLIT_TO_ROOT = {
        "validation": _default_acdc_root("validation"),
        "val": _default_acdc_root("validation"),
        "test": _default_acdc_root("test"),
        "testing": _default_acdc_root("test"),
    }

    def __init__(self, split="test", **kwargs):
        split_key = str(split).lower()
        if split_key not in self.SPLIT_TO_ROOT:
            raise ValueError(f"Unsupported ACDC 2D eval split={split!r}")
        super().__init__(data_root=self.SPLIT_TO_ROOT[split_key], mode="test", **kwargs)

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        mask_path = example["mask_path_"]
        image_path = example["image_path_"]
        label_map = self._load_label_png(mask_path)
        if str(image_path).lower().endswith(".npy"):
            image = np.load(image_path).astype(np.float32)
        else:
            image = self._load_image_png(image_path)
        if image.shape[:2] != label_map.shape:
            raise ValueError(
                f"ACDC image/mask shape mismatch: image={image.shape}, "
                f"mask={label_map.shape}, path={mask_path}"
            )
        self._validate_label_values(label_map, self.num_classes, mask_path)
        example["class_id"] = np.array([-1], dtype=np.int64)
        example["segmentation"] = np.repeat(label_map[:, :, None], 3, axis=2).astype(np.uint8)
        example["image"] = image
        return example

class ACDCValidationVolume(ACDCBase):
    SPLIT_TO_ROOT = {
        "validation": _default_acdc_root("validation"),
        "val": _default_acdc_root("validation"),
        "test": _default_acdc_root("test"),
        "testing": _default_acdc_root("test"),
    }

    def __init__(self, split="testing", **kwargs):
        split_key = str(split).lower()
        if split_key not in self.SPLIT_TO_ROOT:
            raise ValueError(f"Unsupported ACDC volume split={split!r}")
        super().__init__(data_root=self.SPLIT_TO_ROOT[split_key], mode="test_vol", **kwargs)
