from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_segmentation_metric import (
    test_binary_metric_perfect_prediction_is_one,
    test_binary_metric_skips_empty_ground_truth_slices,
    test_binary_dataset_loader_rejects_multiclass_config,
    test_multiclass_foreground_channel_masks_decode_background_and_classes,
    test_multiclass_metric_averages_nonempty_foreground_classes,
)


if __name__ == "__main__":
    test_binary_metric_perfect_prediction_is_one()
    test_binary_metric_skips_empty_ground_truth_slices()
    test_multiclass_metric_averages_nonempty_foreground_classes()
    test_multiclass_foreground_channel_masks_decode_background_and_classes()
    test_binary_dataset_loader_rejects_multiclass_config()
    print("metric tests passed")
