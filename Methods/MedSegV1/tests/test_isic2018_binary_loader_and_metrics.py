from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
from PIL import Image
import torch

CODE_ROOT = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / "scripts"))

from guided_diffusion.isicloader import ISICDataset
from validation_runner import metric_batch


def write_png_pair(root, image_id, mask_values):
    image_dir = root / "images"
    mask_dir = root / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[..., 0] = 32
    image[..., 1] = 96
    image[..., 2] = 160
    Image.fromarray(image).save(image_dir / f"{image_id}.png")
    Image.fromarray(mask_values.astype(np.uint8)).save(mask_dir / f"{image_id}.png")


def make_args(num_seg_classes=2, num_mask_channels=1):
    return SimpleNamespace(
        image_size=16,
        num_seg_classes=num_seg_classes,
        num_mask_channels=num_mask_channels,
    )


def test_isic2018_loader_binarizes_mask_and_tracks_foreground(tmp_path):
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 1:5] = 255
    write_png_pair(tmp_path, "ISIC_0000000", mask)

    dataset = ISICDataset(make_args(), tmp_path)
    image, target, path = dataset[0]

    assert Path(path).stem == "ISIC_0000000"
    assert image.shape == (3, 16, 16)
    assert target.shape == (1, 16, 16)
    assert set(torch.unique(target).tolist()) == {0.0, 1.0}
    assert dataset.foreground_pixels == [16]


def test_isic2018_loader_rejects_multiclass_configuration(tmp_path):
    write_png_pair(tmp_path, "ISIC_0000000", np.zeros((8, 8), dtype=np.uint8))

    try:
        ISICDataset(make_args(num_seg_classes=4, num_mask_channels=3), tmp_path)
    except ValueError as exc:
        assert "binary lesion segmentation" in str(exc)
    else:
        raise AssertionError("ISICDataset should reject multi-class configuration")


def test_binary_metric_skips_empty_ground_truth_slices():
    pred = torch.zeros((2, 1, 4, 4), dtype=torch.float32)
    target = torch.zeros((2, 1, 4, 4), dtype=torch.float32)
    pred[0, 0, 1:3, 1:3] = 1.0
    pred[1, 0, 1:3, 1:3] = 1.0
    target[1, 0, 1:3, 1:3] = 1.0

    dice, iou, nonempty = metric_batch(pred, target, threshold=0.5)

    assert dice == [1.0]
    assert iou == [1.0]
    assert nonempty == [False, True]


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        test_isic2018_loader_binarizes_mask_and_tracks_foreground(Path(tmp_dir))
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_isic2018_loader_rejects_multiclass_configuration(Path(tmp_dir))
    test_binary_metric_skips_empty_ground_truth_slices()
    print("ISIC2018 binary loader and metric tests passed")
