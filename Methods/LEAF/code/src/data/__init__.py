import os
from .base_seg_dataset import BaseSegDataset
from .acdc import ACDC
from .btcv import BTCV
from .isic import ISIC

dataset_name_class_dict: dict[str, type[BaseSegDataset]] = {
    "ACDC": ACDC,
    "BTCV": BTCV,
    "ISIC18": ISIC,
}

def load_custom_dataset(
    base_data_dir: str,
    dataset_name: str,
    resolution: int = 256,
    seed:int = 42,
    eval_split: str = "test",
    **kwargs
) -> dict:

    data_cls = dataset_name_class_dict[dataset_name]
    dataset_dir = os.path.join(base_data_dir, dataset_name)
    return (
        data_cls(dataset_dir=dataset_dir, split="train", resolution=resolution, seed=seed, **kwargs),
        data_cls(dataset_dir=dataset_dir, split=eval_split,  resolution=resolution, seed=seed, **kwargs),
    )
