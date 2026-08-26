import os

from .base_seg_dataset import BaseSegDataset


class ISIC(BaseSegDataset):
    """LEAF-owned ISIC2018 Task 1 binary segmentation cache.

    Expected layout:
      <dataset_dir>/training/images/*.png
      <dataset_dir>/training/masks/*.png
      <dataset_dir>/validation/images/*.png
      <dataset_dir>/validation/masks/*.png
      <dataset_dir>/testing/images/*.png
      <dataset_dir>/testing/masks/*.png
    """

    SPLIT_ALIASES = {
        "train": ("training", "train"),
        "training": ("training", "train"),
        "val": ("validation", "val"),
        "valid": ("validation", "val"),
        "validation": ("validation", "val"),
        "test": ("testing", "test"),
        "testing": ("testing", "test"),
    }

    def __init__(self, **kwargs) -> None:
        dataset_dir = kwargs.get("dataset_dir")
        split = str(kwargs.get("split")).lower()
        if split not in self.SPLIT_ALIASES:
            raise ValueError(f"Unsupported ISIC2018 split={kwargs.get('split')!r}")
        split_dir, loader_split = self.SPLIT_ALIASES[split]
        kwargs["dataset_dir"] = os.path.join(dataset_dir, split_dir)
        kwargs["split"] = loader_split
        super().__init__(
            rgb_subfolder="images",
            mask_subfolder="masks",
            rgb_name_mode=".png",
            mask_name_mode=".png",
            **kwargs,
        )
