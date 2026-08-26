import os

from .base_seg_dataset import BaseSegDataset


class BTCV(BaseSegDataset):
    """LEAF-owned 2D BTCV/Synapse binary segmentation cache.

    Expected layout:
      <dataset_dir>/<split>/images/*.png
      <dataset_dir>/<split>/masks/*.png
    """

    def __init__(self, **kwargs) -> None:
        dataset_dir = kwargs.get("dataset_dir")
        split = kwargs.get("split")
        if split == "validation":
            split = "val"
        kwargs["split"] = split
        kwargs["dataset_dir"] = os.path.join(dataset_dir, split)
        super().__init__(
            rgb_subfolder="images",
            mask_subfolder="masks",
            rgb_name_mode=".png",
            mask_name_mode=".png",
            **kwargs,
        )
