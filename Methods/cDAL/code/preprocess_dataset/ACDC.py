from pathlib import Path

import numpy as np
import torch
from PIL import Image

from preprocess_dataset.transforms import Compose, ToPILImage, ToTensor, Resize, RandomAffine, Normalize
from preprocess_dataset.manifest import read_manifest


CLASS_NAMES = {
    0: "background",
    1: "right_ventricle",
    2: "myocardium",
    3: "left_ventricle",
}


def get_acdc_transform(image_resize):
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


class ACDCDataset(torch.utils.data.Dataset):
    """2D multi-class ACDC PNG cache for cDAL.

    Expected layout:
      <root>/ACDC/<split>/images/*.png
      <root>/ACDC/<split>/masks/*.png

    Masks are RGB foreground channels, matching the shared LEAF/CASCADE cache:
    R = right ventricle, G = myocardium, B = left ventricle. Background is all
    zero. cDAL diffuses the three foreground channels in [-1, 1]; metrics
    recover labels 0..3 from those channels and never collapse them to binary.
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

    foreground_positive = True
    skip_empty_gt_for_metrics = True
    num_seg_classes = 4
    num_foreground_channels = 3
    class_names = CLASS_NAMES
    is_multiclass = True

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
        del target_transform, loader
        self.root = Path(root)
        self.image_size = image_size
        self.transform = transform
        self.train = train
        self.fold = fold

        split_key = str(split or ("training" if train else "testing")).lower()
        if split_key not in self.SPLIT_ALIASES:
            raise ValueError(f"Unsupported ACDC split={split!r}")
        split_name = self.SPLIT_ALIASES[split_key]

        dataset_root = self.root / "ACDC" if (self.root / "ACDC").is_dir() else self.root
        split_root = dataset_root / split_name
        self.physical_split = split_name
        self.split = {"training": "train", "validation": "validation", "testing": "test"}[split_name]
        self.imgs_root = split_root / "images"
        self.masks_root = split_root / "masks"
        if not self.imgs_root.is_dir() or not self.masks_root.is_dir():
            raise FileNotFoundError(
                f"Missing ACDC split directories: images={self.imgs_root}, masks={self.masks_root}"
            )

        self.paths = sorted(path.name for path in self.imgs_root.glob("*.png"))
        manifest = read_manifest(manifest_path)
        if manifest is None:
            raise ValueError(f"ACDC {self.split} requires an explicit subject manifest")
        expected_subjects = set(manifest)
        self.paths = [
            name for name in self.paths
            if name.split("_frame", 1)[0] in expected_subjects
        ]
        found_subjects = {name.split("_frame", 1)[0] for name in self.paths}
        missing_subjects = sorted(expected_subjects - found_subjects)
        if missing_subjects:
            raise FileNotFoundError(
                f"ACDC manifest subject(s) absent from {split_root}: {', '.join(missing_subjects)}"
            )
        if not self.paths:
            raise RuntimeError(f"No ACDC PNG images found under {self.imgs_root}")
        missing = [name for name in self.paths if not (self.masks_root / name).exists()]
        if missing:
            raise FileNotFoundError(f"Missing ACDC mask(s), first missing: {missing[0]}")
        print(f"ACDC semantic split {self.split} from physical {self.physical_split}: {len(self.paths)} slices")

    def __getitem__(self, index):
        name = self.paths[index]
        img_path = self.imgs_root / name
        mask_path = self.masks_root / name

        img = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        mask = np.asarray(Image.open(mask_path).convert("RGB"), dtype=np.uint8)

        if self.transform is not None:
            img, mask = self.transform(img, mask)
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask).permute(2, 0, 1).float()

        img = (img.mean(dim=0, keepdim=True) / 255.0) * 2.0 - 1.0
        mask = (mask > 0).float()
        if mask.shape[0] != self.num_foreground_channels:
            raise ValueError(
                f"ACDC mask must have 3 foreground channels, got shape {tuple(mask.shape)} for {mask_path}"
            )
        if torch.any(mask.sum(dim=0) > 1.0):
            raise ValueError(f"ACDC mask has overlapping foreground classes: {mask_path}")

        mask = 2.0 * mask - 1.0
        out_dict = {"conditioned_image": img}
        return mask, out_dict, f"{Path(name).stem}_{index}"

    def __len__(self):
        return len(self.paths)
