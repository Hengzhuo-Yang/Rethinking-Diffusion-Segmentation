from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.isic import ISIC


def write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def test_isic_loader_uses_official_split_dirs_and_binary_masks(tmp_path):
    root = tmp_path / "ISIC18"
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[..., 0] = 100
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 255
    write_png(root / "validation" / "images" / "ISIC_0000001.png", image)
    write_png(root / "validation" / "masks" / "ISIC_0000001.png", mask)

    dataset = ISIC(dataset_dir=str(root), split="validation", resolution=4, num_seg_classes=2)
    example = dataset[0]

    assert example["pixel_values"].shape == (3, 4, 4)
    assert example["mask_values"].shape == (3, 4, 4)
    assert example["mask_values"].max().item() == 1.0
    assert example["mask_values"].min().item() == 0.0
    assert example["mask_values"][:, 1, 1].tolist() == [1.0, 1.0, 1.0]
    assert example["mask_values"][:, 0, 0].tolist() == [0.0, 0.0, 0.0]


def test_isic_loader_rejects_multiclass_config(tmp_path):
    root = tmp_path / "ISIC18"
    write_png(root / "training" / "images" / "ISIC_0000001.png", np.zeros((4, 4, 3), dtype=np.uint8))
    write_png(root / "training" / "masks" / "ISIC_0000001.png", np.zeros((4, 4), dtype=np.uint8))

    try:
        ISIC(dataset_dir=str(root), split="training", resolution=4, num_seg_classes=4)
    except ValueError as exc:
        assert "binary-mask loader" in str(exc)
    else:
        raise AssertionError("ISIC loader accepted a multi-class config")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_isic_loader_uses_official_split_dirs_and_binary_masks(Path(tmp_dir))
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_isic_loader_rejects_multiclass_config(Path(tmp_dir))
    print("isic dataset tests passed")
