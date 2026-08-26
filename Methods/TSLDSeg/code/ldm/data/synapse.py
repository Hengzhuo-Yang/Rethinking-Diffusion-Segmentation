import glob
import os
from pathlib import Path

# Release modification: fixed ID manifests and portable paths; see MODIFICATIONS.md.

import nibabel as nib
import numpy as np
import PIL
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from ldm.data.manifest import filter_image_paths


TEST_VOLUME_IDS = {
    "0001",
    "0002",
    "0003",
    "0004",
    "0008",
    "0022",
    "0025",
    "0029",
    "0032",
    "0035",
    "0036",
    "0038",
}

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT.parent

COLOR_MAP = np.array([
    [0., 0., 0.],
    [255., 0., 0.],
    [0., 255., 0.],
    [0., 0., 255.],
    [255., 255., 0.],
    [0., 255., 255.],
    [255., 0., 255.],
    [255., 239., 213.],
    [0., 0., 205.],
    [205., 133., 63.],
    [210., 180., 140.],
    [102., 205., 170.],
    [0., 0., 128.],
    [0., 139., 139.],
])


def colorize(seg, num_classes=14):
    """Colorize a segmentation map for visual logging."""
    if num_classes == 2:
        return seg * 255
    for idx in range(1, 14):
        seg[seg[:, :, 0] == idx] = COLOR_MAP[idx]
    return seg


def _first_existing(candidates):
    for path in candidates:
        if path.is_dir():
            return str(path)
    return str(candidates[0])


def _default_btcv_root(split):
    env_root = os.environ.get("TSLDSEG_BTCV_PREPROCESSED_ROOT")
    candidates = []
    if env_root:
        candidates.append(Path(env_root) / "BTCV" / split)
    candidates.append(REPO_ROOT / "data" / "BTCV" / split)
    return _first_existing(candidates)


def _default_btcv_raw_root():
    env_root = os.environ.get("TSLDSEG_BTCV_RAW_ROOT") or os.environ.get("SDSEG_BTCV_RAW_ROOT")
    candidates = []
    if env_root:
        candidates.append(Path(env_root))
    candidates.extend([
        PROJECT_ROOT.parent / "data" / "BTCV" / "raw" / "RawData" / "Training",
        PROJECT_ROOT / "data" / "BTCV" / "raw" / "RawData" / "Training",
        REPO_ROOT / "data" / "btcv" / "raw" / "RawData" / "Training",
        REPO_ROOT / "data" / "synapse" / "test_vol",
    ])
    return _first_existing(candidates)


class SynapseBase(Dataset):
    """BTCV/Synapse dataset used by the original SDSeg/TSLDSeg configs.

    For binary BTCV reproduction, labels are mapped to the common Synapse
    eight-organ set and then collapsed to foreground/background.
    """

    def __init__(self, data_root, size=256, interpolation="nearest", mode=None,
                 num_classes=2, manifest_path=None):
        self.mode = mode
        self.num_classes = num_classes
        self.data_root = str(data_root)
        self.size = size
        self.interpolation = dict(nearest=PIL.Image.NEAREST)[interpolation]
        self.manifest_path = manifest_path
        print(f"[Dataset]: BTCV/Synapse with {self.num_classes} classes, in {self.mode} mode")
        if self.mode not in ["train", "val", "test", "test_vol"]:
            raise NotImplementedError(f"Only support dataset split: train, val, test, test_vol; got {self.mode}")
        if self.num_classes < 2:
            raise ValueError(f"num_classes must include background and at least one foreground, got {self.num_classes}")

        self.image_paths = []
        self.mask_paths = []
        if self.mode == "test_vol":
            self.data_paths = self._find_volume_paths()
            self.image_paths = self.data_paths
            self.mask_paths = [self._label_path_for_volume(path) for path in self.data_paths]
        else:
            self._init_slice_paths()
        self._length = len(self.data_paths)
        if self._length == 0:
            raise FileNotFoundError(f"No BTCV/Synapse samples found under {self.data_root}")

        self.labels = {
            "file_path_": list(self.data_paths),
            "image_path_": list(self.image_paths),
            "mask_path_": list(self.mask_paths),
        }
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ])

    def __len__(self):
        return self._length

    def _init_slice_paths(self):
        root = Path(self.data_root)
        image_dir = root / "images"
        mask_dir = root / "masks"
        if image_dir.is_dir() and mask_dir.is_dir():
            image_paths = filter_image_paths(
                sorted(image_dir.glob("*.png")),
                manifest_path=self.manifest_path,
                dataset="btcv",
            )
            mask_paths = [mask_dir / image_path.name for image_path in image_paths]
            missing = [path for path in mask_paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Missing BTCV mask for image, first missing path: {missing[0]}")
            self.image_paths = [str(path) for path in image_paths]
            self.mask_paths = [str(path) for path in mask_paths]
            self.data_paths = self.mask_paths
            return

        self.data_paths = sorted(glob.glob(os.path.join(self.data_root, "*.png")))
        self.mask_paths = self.data_paths
        self.image_paths = [path.replace("png", "npy") for path in self.mask_paths]

    @staticmethod
    def _volume_id(path):
        digits = "".join(ch for ch in Path(path).stem if ch.isdigit())
        return digits

    def _find_volume_paths(self):
        local_paths = sorted(glob.glob(os.path.join(self.data_root, "img*.nii.gz")))
        if local_paths:
            return local_paths

        raw_root = Path(_default_btcv_raw_root())
        img_dir = raw_root / "img"
        label_dir = raw_root / "label"
        if img_dir.is_dir() and label_dir.is_dir():
            return [
                str(path)
                for path in sorted(img_dir.glob("img*.nii.gz"))
                if self._volume_id(path) in TEST_VOLUME_IDS
            ]
        return []

    @staticmethod
    def _label_path_for_volume(image_path):
        path = Path(image_path)
        if path.parent.name == "img":
            return str(path.parent.parent / "label" / path.name.replace("img", "label", 1))
        return str(path).replace("img", "label")

    @staticmethod
    def _load_mask_png(path):
        segmentation = np.array(Image.open(path))
        if segmentation.ndim == 2:
            return segmentation.astype(np.uint8)
        if segmentation.ndim == 3:
            if not np.all(segmentation[:, :, 0] == segmentation[:, :, 1]) or not np.all(segmentation[:, :, 0] == segmentation[:, :, 2]):
                raise ValueError(f"BTCV class-index PNG channels disagree: {path}")
            return segmentation[:, :, 0].astype(np.uint8)
        raise ValueError(f"BTCV mask must be 2D or repeated RGB, got shape={segmentation.shape}: {path}")

    @staticmethod
    def _load_image_png(path):
        image = Image.open(path).convert("RGB")
        return (np.asarray(image, dtype=np.float32) / 127.5 - 1.0).astype(np.float32)

    @staticmethod
    def transfer_to_9(gts):
        gts = np.asarray(gts).copy()
        gts[gts == 5] = 0
        gts[gts == 6] = 5
        gts[gts == 7] = 6
        gts[gts == 8] = 7
        gts[gts == 9] = 0
        gts[gts == 10] = 0
        gts[gts == 11] = 8
        gts[gts == 12] = 0
        gts[gts == 13] = 0
        return gts

    def _map_segmentation(self, segmentation):
        if self.num_classes == 2:
            segmentation = self.transfer_to_9(segmentation)
            return np.where(segmentation > 0, 1, 0).astype(np.uint8), np.array([-1], dtype=np.int64)
        if self.num_classes == 9:
            segmentation = self.transfer_to_9(segmentation)
        exist_class = sorted(list(set(segmentation.flatten())))
        class_id = np.random.choice(np.array(exist_class), size=1, p=None).astype(np.int64)
        if class_id != 0:
            segmentation = (segmentation == class_id)
        else:
            segmentation = (segmentation != class_id)
        return segmentation.astype(np.uint8), class_id

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)

        if self.mode == "test_vol":
            image = nib.load(example["file_path_"]).get_fdata()
            segmentation = nib.load(example["mask_path_"]).get_fdata()
            image[image < -125] = -125
            image[image > 275] = 275
            image = (image - image.min()) / max(image.max() - image.min(), 1e-8)
            image = (image * 2) - 1
            segmentation, _ = self._map_segmentation(segmentation)
            example["image"] = image.astype(np.float32)
            example["segmentation"] = segmentation.astype(np.float32)
            return example

        segmentation = self._load_mask_png(example["mask_path_"])
        image_path = example["image_path_"]
        if image_path.lower().endswith(".npy"):
            image = np.load(image_path)
        else:
            image = self._load_image_png(image_path)

        segmentation = np.repeat(segmentation[:, :, None], 3, axis=2)
        segmentation = torch.tensor(segmentation.transpose([2, 0, 1]))
        image = torch.tensor(image.transpose([2, 0, 1]))

        if self.mode == "train":
            state = torch.get_rng_state()
            segmentation = self.transform(segmentation)
            torch.set_rng_state(state)
            image = self.transform(image)

        segmentation = np.array(segmentation.permute(1, 2, 0))
        image = np.array(image.permute(1, 2, 0))
        segmentation, class_id = self._map_segmentation(segmentation)

        example["class_id"] = class_id
        example["image"] = image.astype(np.float32)
        if self.mode == "test":
            example["segmentation"] = segmentation.astype(np.float32)
        else:
            example["segmentation"] = ((segmentation.astype(np.float32) * 2) - 1)

        image_min, image_max = float(example["image"].min()), float(example["image"].max())
        if not np.isfinite(image_min) or not np.isfinite(image_max):
            raise ValueError(f"Non-finite BTCV image values: {(image_min, image_max)}")
        seg_min, seg_max = float(example["segmentation"].min()), float(example["segmentation"].max())
        if self.mode == "test":
            if seg_min < 0 or seg_max > self.num_classes - 1:
                raise ValueError(f"BTCV test mask outside class range: {(seg_min, seg_max)}")
        elif seg_min < -1 or seg_max > 1:
            raise ValueError(f"BTCV train/val mask outside [-1, 1]: {(seg_min, seg_max)}")
        return example


class SynapseTrain(SynapseBase):
    def __init__(self, data_root=None, **kwargs):
        super().__init__(data_root=data_root or _default_btcv_root("train"), mode="train", **kwargs)


class SynapseValidation(SynapseBase):
    def __init__(self, data_root=None, split="test", **kwargs):
        super().__init__(data_root=data_root or _default_btcv_root(split), mode="val", **kwargs)


class SynapseValidationEval(SynapseBase):
    def __init__(self, data_root=None, split="test", **kwargs):
        super().__init__(data_root=data_root or _default_btcv_root(split), mode="test", **kwargs)


class SynapseValidationVolume(SynapseBase):
    def __init__(self, data_root=None, **kwargs):
        super().__init__(data_root=data_root or str(Path(_default_btcv_raw_root())), mode="test_vol", **kwargs)


class SynapseValidationVolume4test(SynapseBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
