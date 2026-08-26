from pathlib import Path
import importlib.util
import sys

import numpy as np
import pytest
from PIL import Image

CODE_ROOT = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_ROOT))

from scripts.evaluate_acdc_multiclass_logits import binary_metrics, read_label_map

_ACDC_SPEC = importlib.util.spec_from_file_location(
    "acdc_dataset", CODE_ROOT / "ldm" / "data" / "acdc.py"
)
_ACDC_MODULE = importlib.util.module_from_spec(_ACDC_SPEC)
_ACDC_SPEC.loader.exec_module(_ACDC_MODULE)
ACDCTrain = _ACDC_MODULE.ACDCTrain
ACDCValidationEval = _ACDC_MODULE.ACDCValidationEval


def _write_split(root, label):
    image_dir = root / "images"
    label_dir = root / "label_maps"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    image = np.zeros((*label.shape, 3), dtype=np.uint8)
    image[:, :, 0] = 64
    image[:, :, 1] = 128
    image[:, :, 2] = 192
    Image.fromarray(image).save(image_dir / "patient001_frame01_slice001.png")
    Image.fromarray(label.astype(np.uint8)).save(label_dir / "patient001_frame01_slice001.png")


def test_acdc_eval_dataset_preserves_class_index_label_map(tmp_path):
    label = np.zeros((8, 8), dtype=np.uint8)
    label[1:3, 1:4] = 1
    label[3:6, 2:5] = 2
    label[6:8, 5:8] = 3
    _write_split(tmp_path, label)

    dataset = ACDCValidationEval(data_root=tmp_path, size=8, num_classes=4)
    example = dataset[0]

    assert example["image"].shape == (8, 8, 3)
    assert example["segmentation"].shape == (8, 8, 3)
    assert set(np.unique(example["segmentation"]).astype(int)) == {0, 1, 2, 3}
    assert np.array_equal(example["segmentation"][:, :, 0], label)
    assert example["class_id"].tolist() == [0]


def test_acdc_train_dataset_rejects_binary_num_classes(tmp_path):
    label = np.zeros((8, 8), dtype=np.uint8)
    _write_split(tmp_path, label)

    with pytest.raises(ValueError, match="4-class"):
        ACDCTrain(data_root=tmp_path, size=8, num_classes=2)


def test_acdc_train_dataset_uses_class_conditional_binary_target(tmp_path):
    label = np.zeros((8, 8), dtype=np.uint8)
    label[2:6, 2:6] = 2
    _write_split(tmp_path, label)

    dataset = ACDCTrain(data_root=tmp_path, size=8, num_classes=4)
    example = dataset[0]

    assert example["image"].shape == (8, 8, 3)
    assert example["segmentation"].shape == (8, 8, 3)
    assert set(np.unique(example["segmentation"]).astype(int)).issubset({-1, 1})
    assert int(example["class_id"][0]) in {0, 2}


def test_acdc_evaluator_reads_class_index_and_rgb_foreground_masks(tmp_path):
    class_index = np.zeros((4, 4), dtype=np.uint8)
    class_index[0:2, 0:2] = 1
    class_index[2:4, 0:2] = 2
    class_index[2:4, 2:4] = 3
    class_index_path = tmp_path / "class_index.png"
    Image.fromarray(class_index).save(class_index_path)

    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[class_index == 1, 0] = 255
    rgb[class_index == 2, 1] = 255
    rgb[class_index == 3, 2] = 255
    rgb_path = tmp_path / "rgb.png"
    Image.fromarray(rgb).save(rgb_path)

    assert np.array_equal(read_label_map(class_index_path, 4), class_index)
    assert np.array_equal(read_label_map(rgb_path, 4), class_index)

    for class_id in (1, 2, 3):
        dice, iou = binary_metrics(class_index == class_id, class_index == class_id)
        assert dice == pytest.approx(1.0)
        assert iou == pytest.approx(1.0)
