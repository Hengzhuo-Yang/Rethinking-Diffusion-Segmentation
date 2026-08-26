"""Static regression tests for the release split contract."""

import re
from pathlib import Path

from validate_splits import BTCV_EXPECTED, SPECS, validate_release_manifests


RELEASE_ROOT = Path(__file__).resolve().parents[1]


def test_fixed_manifests_are_exact_and_disjoint():
    manifests = validate_release_manifests()
    for dataset, spec in SPECS.items():
        observed = manifests[dataset]
        assert {split: len(ids) for split, ids in observed.items()} == spec["counts"]
        assert not (set(observed["train"]) & set(observed["val"]))
        assert not (set(observed["train"]) & set(observed["test"]))
        assert not (set(observed["val"]) & set(observed["test"]))
    assert {split: set(ids) for split, ids in manifests["btcv"].items()} == BTCV_EXPECTED


def test_preprocessors_have_no_local_path_defaults_or_legacy_btcv_subset():
    scripts = RELEASE_ROOT / "code" / "scripts"
    sources = {
        path.name: path.read_text(encoding="utf-8")
        for path in (
            scripts / "preprocess_btcv_synapse_to_leaf.py",
            scripts / "preprocess_acdc_to_leaf.py",
            scripts / "preprocess_isic2018_to_leaf.py",
        )
    }
    for source in sources.values():
        assert re.search(r"(?i)\b[a-z]:[\\/]", source) is None
        assert "DEFAULT_MANIFEST_ROOT" in source
        assert "EXPECTED_MANIFEST_FINGERPRINTS" in source
        assert '"--manifest_root"' in source
    btcv = sources["preprocess_btcv_synapse_to_leaf.py"]
    for forbidden in (
        "test_val_fraction",
        "test_val_split_name",
        "materialize_test_validation_subset",
        "choose_validation_volumes",
        "val_fraction",
        "10pct",
    ):
        assert forbidden not in btcv
