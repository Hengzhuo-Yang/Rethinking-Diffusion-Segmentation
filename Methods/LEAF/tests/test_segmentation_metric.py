import torch

from src.data.base_seg_dataset import BaseSegDataset
from src.util.metric import SegmentationMetric


def test_binary_metric_perfect_prediction_is_one():
    target = torch.zeros((2, 1, 4, 4), dtype=torch.long)
    target[0, :, 1:3, 1:3] = 1
    target[1, :, 0:2, 0:2] = 1

    metric = SegmentationMetric(metrics=["dice", "miou"], device=torch.device("cpu"))
    metric.update(target.clone(), target)
    results = metric.compute()

    assert results["dice"] == 1.0
    assert results["miou"] == 1.0


def test_binary_metric_skips_empty_ground_truth_slices():
    target = torch.zeros((2, 1, 4, 4), dtype=torch.long)
    target[0, :, 1:3, 1:3] = 1
    pred = target.clone()
    pred[1] = 1

    metric = SegmentationMetric(metrics=["dice", "miou"], device=torch.device("cpu"))
    metric.update(pred, target)
    results = metric.compute()

    assert results["dice"] == 1.0
    assert results["miou"] == 1.0


def test_multiclass_metric_averages_nonempty_foreground_classes():
    target = torch.zeros((2, 1, 4, 4), dtype=torch.long)
    target[0, :, 0:2, 0:2] = 1
    target[0, :, 2:4, 2:4] = 2
    target[1, :, 1:3, 1:3] = 2
    pred = target.clone()
    pred[1, :, 0:1, 0:1] = 3

    metric = SegmentationMetric(metrics=["dice", "miou"], device=torch.device("cpu"), num_classes=4)
    metric.update(pred, target)
    results = metric.compute()
    counts = metric.summary_counts()

    assert results["dice"] == 1.0
    assert results["miou"] == 1.0
    assert counts["num_slices"] == 2
    assert counts["num_metric_observations"] == 3
    assert counts["num_empty_gt"] == 0


def test_multiclass_foreground_channel_masks_decode_background_and_classes():
    target = torch.zeros((2, 3, 4, 4), dtype=torch.float32)
    target[0, 0, 0:2, 0:2] = 1.0
    target[0, 2, 2:4, 2:4] = 1.0
    pred = target.clone()
    pred[1, 1, 0:1, 0:1] = 1.0

    metric = SegmentationMetric(metrics=["dice", "miou"], device=torch.device("cpu"), num_classes=4)
    metric.update(pred, target)
    results = metric.compute()
    counts = metric.summary_counts()

    assert results["dice"] == 1.0
    assert results["miou"] == 1.0
    assert counts["num_slices"] == 2
    assert counts["num_metric_observations"] == 2
    assert counts["num_empty_gt"] == 1


def test_binary_dataset_loader_rejects_multiclass_config():
    try:
        BaseSegDataset(
            dataset_dir=".",
            split="train",
            rgb_subfolder="images",
            mask_subfolder="masks",
            num_seg_classes=4,
        )
    except ValueError as exc:
        assert "binary-mask loader" in str(exc)
    else:
        raise AssertionError("BaseSegDataset accepted a multi-class config")
