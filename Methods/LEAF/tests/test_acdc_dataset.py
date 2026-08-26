from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.acdc import ACDC
from src.util.metric import labels_from_segmentation_tensor


def write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def test_acdc_loader_preserves_three_foreground_channels(tmp_path):
    root = tmp_path / "ACDC"
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    mask = np.zeros((4, 4, 3), dtype=np.uint8)
    mask[0:2, 0:2, 0] = 255
    mask[2:4, 2:4, 2] = 255
    write_png(root / "validation" / "images" / "slice001.png", image)
    write_png(root / "validation" / "masks" / "slice001.png", mask)

    dataset = ACDC(dataset_dir=str(root), split="validation", resolution=4, num_seg_classes=4)
    example = dataset[0]
    labels = labels_from_segmentation_tensor(example["mask_values"].unsqueeze(0), num_classes=4)

    assert example["pixel_values"].shape == (3, 4, 4)
    assert example["mask_values"].shape == (3, 4, 4)
    assert labels[0, 0, 0].item() == 1
    assert labels[0, 3, 3].item() == 3
    assert labels[0, 0, 3].item() == 0


def test_acdc_loader_rejects_binary_class_count(tmp_path):
    root = tmp_path / "ACDC"
    write_png(root / "validation" / "images" / "slice001.png", np.zeros((4, 4, 3), dtype=np.uint8))
    write_png(root / "validation" / "masks" / "slice001.png", np.zeros((4, 4, 3), dtype=np.uint8))

    try:
        ACDC(dataset_dir=str(root), split="validation", resolution=4, num_seg_classes=2)
    except ValueError as exc:
        assert "num_seg_classes=4" in str(exc)
    else:
        raise AssertionError("ACDC loader accepted binary num_seg_classes")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_acdc_loader_preserves_three_foreground_channels(Path(tmp_dir))
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_acdc_loader_rejects_binary_class_count(Path(tmp_dir))
    print("acdc dataset tests passed")
