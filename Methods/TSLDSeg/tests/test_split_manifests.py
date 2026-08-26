from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_splits import SPECS, validate_release_manifests


def test_fixed_manifests_are_complete_unique_and_disjoint():
    manifests = validate_release_manifests()
    for dataset, split_ids in manifests.items():
        assert {split: len(ids) for split, ids in split_ids.items()} == SPECS[dataset]["id_counts"]


def test_btcv_validation_and_test_assignments_are_exact():
    btcv = validate_release_manifests()["btcv"]
    assert btcv["val"] == ["case0008", "case0001"]
    assert btcv["test"] == [
        "case0022", "case0038", "case0036", "case0032", "case0002",
        "case0029", "case0003", "case0004", "case0025", "case0035",
    ]
