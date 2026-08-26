import os
from pathlib import Path
import re

import numpy as np
import torch
from PIL import Image

from preprocess_dataset.transforms import Compose, ToPILImage, ToTensor, Resize, RandomAffine, Normalize
from preprocess_dataset.manifest import read_manifest


def get_btcv_transform(image_resize):
    transform_train = Compose([
        ToPILImage(),
        Resize((image_resize, image_resize)),
        RandomAffine(int(22), scale=(float(0.75), float(1.25))),
        ToTensor(),
        Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
    ])
    transform_test = Compose([
        ToPILImage(),
        Resize((image_resize, image_resize)),
        ToTensor(),
        Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
    ])
    return transform_train, transform_test


class BTCVDataset(torch.utils.data.Dataset):
    """2D binary BTCV/Synapse PNG cache for cDAL.

    Expected layout:
      <root>/BTCV/<split>/images/*.png
      <root>/BTCV/<split>/masks/*.png

    The masks are binary foreground/background masks. Foreground is encoded as
    +1 and background as -1, matching the cDAL MoNuSeg convention and LEAF's
    foreground Dice/IoU policy.
    """

    foreground_positive = True
    skip_empty_gt_for_metrics = True

    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        train=False,
        loader=None,
        image_size=256,
        fold=0,
        split=None,
        manifest_path=None,
    ):
        self.root = Path(root)
        self.image_size = image_size
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.fold = fold

        split_key = str(split or ("train" if train else "test")).lower()
        split_aliases = {
            "train": "train",
            "training": "train",
            "val": "validation",
            "valid": "validation",
            "validation": "validation",
            "test": "test",
            "testing": "test",
        }
        if split_key not in split_aliases:
            raise ValueError(f"Unsupported BTCV split={split!r}")
        semantic_split = split_aliases[split_key]
        dataset_root = self.root / "BTCV" if (self.root / "BTCV").is_dir() else self.root
        physical_split = "train" if semantic_split == "train" else "heldout"
        split_root = dataset_root / physical_split
        if semantic_split in {"validation", "test"} and not split_root.is_dir():
            legacy_heldout = dataset_root / "test"
            if legacy_heldout.is_dir():
                physical_split = "test"
                split_root = legacy_heldout
        self.split = semantic_split
        self.physical_split = physical_split
        self.imgs_root = split_root / "images"
        self.masks_root = split_root / "masks"
        if not self.imgs_root.is_dir() or not self.masks_root.is_dir():
            raise FileNotFoundError(
                f"Missing BTCV split directories: images={self.imgs_root}, masks={self.masks_root}"
            )

        self.paths = sorted(path.name for path in self.imgs_root.glob("*.png"))
        manifest = read_manifest(manifest_path)
        if semantic_split in {"validation", "test"} and manifest is None:
            raise ValueError(f"BTCV {semantic_split} requires an explicit case manifest")
        if manifest is not None:
            expected_ids = set()
            for item in manifest:
                match = re.search(r"(\d+)", item)
                expected_ids.add(match.group(1).zfill(4) if match else item)
            self.paths = [
                name
                for name in self.paths
                if (match := re.match(r"vol(\d{4})", Path(name).stem))
                and match.group(1) in expected_ids
            ]
            found_ids = {
                re.match(r"vol(\d{4})", Path(name).stem).group(1)
                for name in self.paths
            }
            missing_ids = sorted(expected_ids - found_ids)
            if missing_ids:
                raise FileNotFoundError(
                    f"BTCV manifest case(s) absent from {split_root}: {', '.join(missing_ids)}"
                )
        if not self.paths:
            raise RuntimeError(f"No BTCV PNG images found under {self.imgs_root}")
        missing = [name for name in self.paths if not (self.masks_root / name).exists()]
        if missing:
            raise FileNotFoundError(f"Missing BTCV mask(s), first missing: {missing[0]}")
        print(
            f"BTCV semantic split {self.split} from physical {self.physical_split}: "
            f"{len(self.paths)} slices"
        )

    def __getitem__(self, index):
        name = self.paths[index]
        img_path = self.imgs_root / name
        mask_path = self.masks_root / name

        img = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
        mask = np.where(mask > 0, 1, 0).astype(np.uint8)

        if self.transform is not None:
            img, mask = self.transform(img, mask)
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask).float()

        img = (img.mean(dim=0, keepdim=True) / 255.0) * 2.0 - 1.0
        mask = (mask > 0).float()
        mask = 2.0 * mask - 1.0
        out_dict = {"conditioned_image": img}
        return mask.unsqueeze(0), out_dict, f"{Path(name).stem}_{index}"

    def __len__(self):
        return len(self.paths)
