from pathlib import Path
import importlib.util
import sys

import numpy as np
import pytest
from PIL import Image

CODE_ROOT = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_ROOT))

from scripts.evaluate_isic2018_binary_logits import binary_metrics, read_binary_mask

_ISIC_SPEC = importlib.util.spec_from_file_location(
    "isic_dataset", CODE_ROOT / "ldm" / "data" / "isic.py"
)
_ISIC_MODULE = importlib.util.module_from_spec(_ISIC_SPEC)
_ISIC_SPEC.loader.exec_module(_ISIC_MODULE)
ISICTrain = _ISIC_MODULE.ISICTrain
ISICTest = _ISIC_MODULE.ISICTest


def _write_split(root, mask):
    image_dir = root / "images"
    mask_dir = root / "masks"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    image = np.zeros((*mask.shape, 3), dtype=np.uint8)
    image[:, :, 0] = 64
    image[:, :, 1] = 128
    image[:, :, 2] = 192
    Image.fromarray(image).save(image_dir / "ISIC_0000000.png")
    Image.fromarray(mask.astype(np.uint8)).save(mask_dir / "ISIC_0000000.png")


def test_isic_train_dataset_uses_binary_three_channel_vae_target(tmp_path):
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 1:7] = 255
    _write_split(tmp_path, mask)

    dataset = ISICTrain(data_root=tmp_path, size=8, num_classes=2)
    example = dataset[0]

    assert example["image"].shape == (8, 8, 3)
    assert example["segmentation"].shape == (8, 8, 3)
    assert set(np.unique(example["segmentation"]).astype(int)) == {-1, 1}
    assert example["class_id"].tolist() == [-1]


def test_isic_test_dataset_preserves_binary_eval_mask(tmp_path):
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[1:3, 2:5] = 255
    _write_split(tmp_path, mask)

    dataset = ISICTest(data_root=tmp_path, size=8, num_classes=2)
    example = dataset[0]

    assert example["segmentation"].shape == (8, 8, 3)
    assert set(np.unique(example["segmentation"]).astype(int)) == {0, 1}


def test_isic_binary_loader_rejects_multiclass_request(tmp_path):
    mask = np.zeros((8, 8), dtype=np.uint8)
    _write_split(tmp_path, mask)

    with pytest.raises(ValueError, match="binary lesion"):
        ISICTrain(data_root=tmp_path, size=8, num_classes=4)


def test_isic_binary_metrics_perfect_prediction_and_empty_gt_policy(tmp_path):
    gt = np.zeros((4, 4), dtype=bool)
    gt[1:3, 1:3] = True
    dice, iou = binary_metrics(gt, gt)
    assert dice == pytest.approx(1.0)
    assert iou == pytest.approx(1.0)

    empty = np.zeros((4, 4), dtype=bool)
    dice, iou = binary_metrics(empty, empty)
    assert dice is None
    assert iou is None

    mask_path = tmp_path / "mask.png"
    Image.fromarray((gt.astype(np.uint8) * 255)).save(mask_path)
    assert np.array_equal(read_binary_mask(mask_path, 0), gt)
