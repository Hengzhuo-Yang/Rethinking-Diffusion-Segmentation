from torch.utils.data import DataLoader


def create_dataset(
    data_dir: str,
    mode: str = "train",
    image_size: int = 256,
    dataset_name: str = "monu",
    fold: int = 0,
    manifest_path: str | None = None,
):
    if dataset_name == "monu":
        from preprocess_dataset.MoNu import MonuDataset, get_monu_transform
        dataset_class = MonuDataset
        transform_train, transform_test = get_monu_transform(image_size=image_size)
    elif dataset_name == "lung":
        from preprocess_dataset.Lung import LungDataset, get_lung_transform
        dataset_class = LungDataset
        transform_train, transform_test = get_lung_transform(image_resize=image_size)
    elif dataset_name == "btcv":
        from preprocess_dataset.BTCV import BTCVDataset, get_btcv_transform
        dataset_class = BTCVDataset
        transform_train, transform_test = get_btcv_transform(image_resize=image_size)
    elif dataset_name == "acdc":
        from preprocess_dataset.ACDC import ACDCDataset, get_acdc_transform
        dataset_class = ACDCDataset
        transform_train, transform_test = get_acdc_transform(image_resize=image_size)
    elif dataset_name in ("isic2018", "isic18", "isic"):
        from preprocess_dataset.ISIC2018 import ISIC2018Dataset, get_isic2018_transform
        dataset_class = ISIC2018Dataset
        transform_train, transform_test = get_isic2018_transform(image_resize=image_size)
    else:
        raise ValueError(
            "Dataset name should be one of \"monu\", \"lung\", \"btcv\", \"acdc\", or \"isic2018\", "
            "Unknown dataset: {}".format(dataset_name)
        )

    if dataset_name in ("btcv", "acdc", "isic2018", "isic18", "isic"):
        split = "train" if mode == "train" else mode
        return dataset_class(
            data_dir,
            train=(mode == "train"),
            transform=(transform_train if mode == "train" else transform_test),
            image_size=image_size,
            fold=fold,
            split=split,
            manifest_path=manifest_path,
        )
    if mode == "train":
        return dataset_class(data_dir, train=True, transform=transform_train, image_size=image_size, fold=fold)
    return dataset_class(data_dir, train=False, transform=transform_test, image_size=image_size, fold=fold)


def load_data(
    *,
    data_dir: str,
    batch_size: int,
    image_size: int,
    deterministic=True,
    dataset_name: str = "monu",
    manifest_path: str | None = None,
):
    """
    For a dataset, create a generator over (images, labels) pairs.
    """

    dataset_date = create_dataset(
        data_dir=data_dir,
        image_size=image_size,
        mode="train",
        dataset_name=dataset_name,
        fold=0,
        manifest_path=manifest_path,
    )

    if deterministic:
        loader = DataLoader(dataset_date, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
    else:
        loader = DataLoader(dataset_date, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    while True:
        yield from loader

