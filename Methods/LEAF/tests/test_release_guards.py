"""Static guards for the formal release lifecycle."""

from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parents[1]


def test_formal_pipeline_validates_cache_and_selection_metadata():
    source = (RELEASE_ROOT / "code" / "scripts" / "release_pipeline.py").read_text(
        encoding="utf-8"
    )
    assert "validate_fixed_cache(args)" in source
    assert "validation_split" in source
    assert "SPLIT_VALIDATOR" in source


def test_final_evaluators_require_requested_ema_weights():
    scripts = RELEASE_ROOT / "code" / "scripts"
    for name in (
        "evaluate_btcv_leaf.py",
        "evaluate_acdc_leaf.py",
        "evaluate_isic2018_leaf.py",
    ):
        source = (scripts / name).read_text(encoding="utf-8")
        assert source.count("EMA-selected final evaluation requires checkpoint subfolder") == 2


def test_training_seed_and_stable_filename_order_are_explicit():
    train_source = (RELEASE_ROOT / "code" / "train.py").read_text(encoding="utf-8")
    data_source = (RELEASE_ROOT / "code" / "src" / "data" / "base_seg_dataset.py").read_text(
        encoding="utf-8"
    )
    assert "set_seed(int(cfg.seed), device_specific=False)" in train_source
    assert "sorted(os.listdir" in data_source


if __name__ == "__main__":
    test_formal_pipeline_validates_cache_and_selection_metadata()
    test_final_evaluators_require_requested_ema_weights()
    test_training_seed_and_stable_filename_order_are_explicit()
    print("release guard tests passed")
