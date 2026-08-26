# MODIFICATION NOTICE (release prepared 2026-07-19): this file differs from
# upstream SDSeg commit 0b0aa388a5e2def75abfbef90d7bcfc5c16f2704.
# Changes implement the fixed BTCV split and release data-loading contract.
# See the method-root MODIFICATIONS.md.

import os
import re
import sys
from pathlib import Path

import numpy as np
import PIL
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import torch.nn.functional as F
import nibabel as nib


TRAIN_VOLUME_IDS = {
    "0005", "0006", "0007", "0009", "0010", "0021", "0023", "0024", "0026",
    "0027", "0028", "0030", "0031", "0033", "0034", "0037", "0039", "0040",
}
VALIDATION_VOLUME_IDS = {"0001", "0008"}
TEST_VOLUME_IDS = {
    "0002", "0003", "0004", "0022", "0025", "0029", "0032", "0035", "0036", "0038",
}
ALL_EVAL_VOLUME_IDS = VALIDATION_VOLUME_IDS | TEST_VOLUME_IDS


REPO_ROOT = Path(__file__).resolve().parents[2]


def _first_existing(candidates):
    for path in candidates:
        if path.is_dir():
            return str(path)
    return str(candidates[0])


def _default_btcv_root(split):
    data_root = Path(os.environ.get("SDSEG_DATA_ROOT", REPO_ROOT / "data"))
    candidates = [data_root / "btcv" / split]
    # Historical caches stored all 12 held-out cases under test/. Filtering
    # below still enforces the fixed 2/10 validation/test partition.
    if split in {"validation", "test"}:
        candidates.append(data_root / "btcv" / "test")
    return _first_existing(candidates)


COLOR_MAP = np.array([
            [  0.,   0.,   0.],
            [255.,   0.,   0.],
            [  0., 255.,   0.],
            [  0.,   0., 255.],
            [255., 255.,   0.],
            [  0., 255., 255.],
            [255.,   0., 255.],
            [255., 239., 213.],
            [  0.,   0., 205.],
            [205., 133.,  63.],
            [210., 180., 140.],
            [102., 205., 170.],
            [  0.,   0., 128.],
            [  0., 139., 139.],
        ])



def colorize(seg, num_classes=14):
    """ seg (H W C)"""
    if num_classes == 2:
        return seg * 255
    for idx in range(1, 14):
        seg[seg[:, :, 0] == idx] = COLOR_MAP[idx]
    return seg


class SynapseBase(Dataset):
    """BTCV Dataset Base (historically named Synapse in SDSeg)
    Notes:
        - `segmentation` is for the diffusion training stage (range binary -1 and 1)
        - `image` is for conditional signal to guided final seg-map (range -1 to 1)
    TODO:
        - extend to multi-label segmentation.
        - extend to fit 13 organs and 8 organs.
    """
    def __init__(self, data_root, size=256, interpolation="nearest", mode=None,
                 num_classes=2, case_ids=None):
        self.mode = mode
        self.num_classes = num_classes
        self.case_ids = None if case_ids is None else {str(case_id).zfill(4) for case_id in case_ids}
        print(f"[Dataset]: BTCV with {self.num_classes} classes, in {self.mode} mode")
        assert mode in ["train", "val", "test", "test_vol"]

        self.data_root = data_root
        self.image_paths = []
        self.mask_paths = []
        if mode == "test_vol":
            self.data_paths = self._find_volume_paths()
            self.image_paths = self.data_paths
            self.mask_paths = [path.replace("img", "label") for path in self.data_paths]
        else:
            self._init_slice_paths()
        self._length = len(self.data_paths)

        self.labels = dict(
            # relative_file_path_=[l for l in self.data_paths],
            file_path_=[path for path in self.data_paths],
            image_path_=[path for path in self.image_paths],
            mask_path_=[path for path in self.mask_paths],
        )
        self.size = size
        self.interpolation = dict(nearest=PIL.Image.NEAREST)[interpolation]   # for segmentation slice
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            # transforms.CenterCrop(size=(256, 256)),
            # transforms.Resize(size=(256, 256), interpolation=self.interpolation),
            # transforms.RandomResizedCrop(size=(256, 256),
            #                              scale=(0.2, 1),
            #                              interpolation=self.interpolation),
        ])

    def __len__(self):
        return self._length

    def _init_slice_paths(self):
        root = Path(self.data_root)
        image_dir = root / "images"
        mask_dir = root / "masks"
        if image_dir.is_dir() and mask_dir.is_dir():
            image_paths = sorted(image_dir.glob("*.png"))
            if self.case_ids is not None:
                image_paths = [
                    path for path in image_paths
                    if self._slice_volume_id(path) in self.case_ids
                ]
            mask_paths = [mask_dir / image_path.name for image_path in image_paths]
            missing = [path for path in mask_paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Missing BTCV mask for image, first missing path: {missing[0]}")
            self.image_paths = [str(path) for path in image_paths]
            self.mask_paths = [str(path) for path in mask_paths]
            self.data_paths = self.mask_paths
            return

        self.data_paths = sorted(glob.glob(os.path.join(self.data_root, "*.png")))
        if self.case_ids is not None:
            self.data_paths = [
                path for path in self.data_paths
                if self._slice_volume_id(path) in self.case_ids
            ]
        self.mask_paths = self.data_paths
        self.image_paths = [path.replace("png", "npy") for path in self.mask_paths]

    @staticmethod
    def _slice_volume_id(path):
        stem = Path(path).stem
        match = re.search(r"(?:vol|case)(\d{4})", stem, flags=re.IGNORECASE)
        if match:
            return match.group(1)
        digits = "".join(ch for ch in stem if ch.isdigit())
        return digits[:4]

    @staticmethod
    def _volume_id(path):
        digits = "".join(ch for ch in Path(path).stem if ch.isdigit())
        return digits

    def _find_volume_paths(self):
        local_paths = sorted(glob.glob(os.path.join(self.data_root, "img*.nii.gz")))
        if local_paths:
            return local_paths

        candidates = []
        env_root = os.environ.get("SDSEG_BTCV_RAW_ROOT")
        if env_root:
            candidates.append(Path(env_root))
        candidates.append(REPO_ROOT / "data" / "btcv" / "raw" / "RawData" / "Training")
        for candidate in candidates:
            img_dir = candidate / "img"
            label_dir = candidate / "label"
            if img_dir.is_dir() and label_dir.is_dir():
                paths = [
                    path for path in sorted(img_dir.glob("img*.nii.gz"))
                    if self._volume_id(path) in TEST_VOLUME_IDS
                ]
                if paths:
                    return [str(path) for path in paths]
        return []

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
        raw = np.array(Image.open(path))
        if raw.dtype == np.uint16:
            ct = raw.astype(np.float32) / 65535.0 * 425.0 - 175.0
            image = (((ct + 125.0) / 400.0) * 2.0) - 1.0
            return np.repeat(image[:, :, None], 3, axis=2).astype(np.float32)
        image = Image.open(path).convert("RGB")
        return (np.asarray(image, dtype=np.float32) / 127.5 - 1.0).astype(np.float32)

    @staticmethod
    def _label_path_for_volume(image_path):
        path = Path(image_path)
        if path.parent.name == "img":
            return str(path.parent.parent / "label" / path.name.replace("img", "label", 1))
        return str(path).replace("img", "label")

    def __getitem__(self, i):
        # read segmentation and images
        example = dict((k, self.labels[k][i]) for k in self.labels)

        if self.mode == "test_vol":     # 3-D volume
            image = nib.load(example["file_path_"]).get_fdata()
            segmentation = nib.load(self._label_path_for_volume(example["file_path_"])).get_fdata()

            image[image < -125] = -125  # window-level window width
            image[image > 275] = 275
            image = (image - image.min()) / (image.max() - image.min())     # [-125, 275] -> [0, 1]
            image = (image * 2) - 1     # [0, 1] -> [-1, 1]

            if self.num_classes == 2:
                segmentation = self.transfer_to_9(segmentation) # 14 -> 9 -> 2
                segmentation = np.where(segmentation > 0, 1, 0)  # TODO: extend to multi-label segmentation
            elif self.num_classes == 9:
                segmentation = self.transfer_to_9(segmentation)
            else:
                pass

            example["image"] = image
            example["segmentation"] = segmentation
            # example["segmentation_onehot"] = \
            #     F.one_hot(torch.tensor(segmentation)[:, :, 0].long(), num_classes=self.num_classes).numpy()
            return example

        segmentation = self._load_mask_png(example["mask_path_"])
        image_path = example["image_path_"]
        if image_path.lower().endswith(".npy"):
            image = np.load(image_path)    # same name, different postfix
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

        segmentation = np.array(segmentation.permute(1, 2, 0))      # h w c
        image = np.array(image.permute(1, 2, 0))        # h w c

        if self.num_classes == 2:
            segmentation = self.transfer_to_9(segmentation) # 14 -> 9 -> 2
            segmentation = np.where(segmentation > 0, 1, 0)
            class_id = np.array([-1]) # doesn't matter for binary seg
        else:
            if self.num_classes == 9:
                segmentation = self.transfer_to_9(segmentation)
            # # handle segmentation map
            # example["segmentation_onehot"] = \
            #     F.one_hot(torch.tensor(segmentation)[:, :, 0].long(), num_classes=self.num_classes).numpy()

            # handle random class
            exist_class = sorted(list(set(segmentation.flatten())))
            class_id = np.random.choice(np.array(exist_class), size=1,
                                        p=None).astype(np.int64)

            # # choose class from id (3 channel, 1 class)
            if class_id != 0:
                segmentation = (segmentation == class_id)   # for multi, get a random class (existed) except 0
            else:
                segmentation = (segmentation != class_id)   # (empty slice) or (not empty slice & class_id==0)

        example["class_id"] = class_id
        example["image"] = image     # range from -1 to 1, np.float32
        if self.mode == "test":
            example["segmentation"] = segmentation.astype(np.float32)
        else:
            # turn segmentation map [0, 1] -> [-1, 1] for diffusion training/validation
            example["segmentation"] = ((segmentation.astype(np.float32) * 2) - 1)
        image_min, image_max = float(example["image"].min()), float(example["image"].max())
        assert np.isfinite(image_min) and np.isfinite(image_max), (image_min, image_max)
        if self.mode == "test":
            seg_min, seg_max = float(example["segmentation"].min()), float(example["segmentation"].max())
            assert 0 <= seg_min and seg_max <= self.num_classes - 1, (seg_min, seg_max)
        else:
            seg_min, seg_max = float(example["segmentation"].min()), float(example["segmentation"].max())
            assert -1 <= seg_min and seg_max <= 1, (seg_min, seg_max)
        return example

    @staticmethod
    def transfer_to_9(gts):
        # 0 1 2 3 4 5 6 7 8 9 10 11 12 13
        # 0 1 2 3 4   6 7 8      11
        # 0 1 2 3 4   5 6 7      8
        # extract the 8 target classes (total 9 classes)for training
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

class SynapseTrain(SynapseBase):
    def __init__(self, **kwargs):
        super().__init__(data_root=_default_btcv_root("train"), mode="train",
                         case_ids=TRAIN_VOLUME_IDS, **kwargs)


class SynapseValidation(SynapseBase):
    def __init__(self, split="validation", **kwargs):
        if split != "validation":
            raise ValueError(f"BTCV training validation split must be validation, got {split!r}")
        super().__init__(data_root=_default_btcv_root(split), mode="val",
                         case_ids=VALIDATION_VOLUME_IDS, **kwargs)


class SynapseValidationEval(SynapseBase):
    def __init__(self, split="test", **kwargs):
        if split not in {"validation", "test"}:
            raise ValueError(f"BTCV evaluation split must be validation or test, got {split!r}")
        case_ids = VALIDATION_VOLUME_IDS if split == "validation" else TEST_VOLUME_IDS
        super().__init__(data_root=_default_btcv_root(split), mode="test",
                         case_ids=case_ids, **kwargs)


class SynapseValidationVolume(SynapseBase):
    def __init__(self, **kwargs):
        super().__init__(data_root=str(REPO_ROOT / "data" / "btcv" / "test_vol"), mode="test_vol", **kwargs)

class SynapseValidationVolume4test(SynapseBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


# from Swin-UNETR:

# "validation": [
#         {
#             "image": "imagesTr/img0035.nii.gz",
#             "label": "labelsTr/label0035.nii.gz"
#         },
#         {
#             "image": "imagesTr/img0036.nii.gz",
#             "label": "labelsTr/label0036.nii.gz"
#         },
#         {
#             "image": "imagesTr/img0037.nii.gz",
#             "label": "labelsTr/label0037.nii.gz"
#         },
#         {
#             "image": "imagesTr/img0038.nii.gz",
#             "label": "labelsTr/label0038.nii.gz"
#         },
#         {
#             "image": "imagesTr/img0039.nii.gz",
#             "label": "labelsTr/label0039.nii.gz"
#         },
#         {
#             "image": "imagesTr/img0040.nii.gz",
#             "label": "labelsTr/label0040.nii.gz"
#         }
