from pathlib import Path

from omegaconf import OmegaConf


RELEASE_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = RELEASE_ROOT / "code"
CONFIG_ROOT = CODE_ROOT / "configs" / "latent-diffusion"
CONFIGS = {
    "btcv": CONFIG_ROOT / "btcv-cls2-ldm-kl-8.yaml",
    "acdc": CONFIG_ROOT / "acdc-cls4-ldm-kl-8.yaml",
    "isic2018": CONFIG_ROOT / "isic-ldm-kl-8.yaml",
}


def test_all_release_configs_use_validation_metric_for_checkpoint_selection():
    for path in CONFIGS.values():
        config = OmegaConf.load(path)
        assert config.model.params.monitor == "val_avg_dice"
        assert config.lightning.modelcheckpoint.params.monitor == "val_avg_dice"
        assert "val_avg_dice" in config.lightning.modelcheckpoint.params.filename
        assert config.lightning.modelcheckpoint.params.save_top_k == 1
        assert config.lightning.modelcheckpoint.params.save_last is False


def test_all_release_configs_bind_fixed_manifests_and_real_test_partitions():
    expected = {
        "btcv": ("btcv/train.txt", "btcv/val.txt", "btcv/test.txt", "/BTCV/test"),
        "acdc": ("acdc/train.txt", "acdc/val.txt", "acdc/test.txt", "/ACDC/testing"),
        "isic2018": (
            "isic2018/train.txt", "isic2018/val.txt", "isic2018/test.txt", "/ISIC18/testing"
        ),
    }
    for dataset, path in CONFIGS.items():
        config = OmegaConf.load(path)
        train_suffix, val_suffix, test_suffix, test_root_suffix = expected[dataset]
        assert config.data.params.train.params.manifest_path.endswith(train_suffix)
        assert config.data.params.validation.params.manifest_path.endswith(val_suffix)
        assert config.data.params.validation_metrics.params.manifest_path.endswith(val_suffix)
        assert config.data.params.validation.params.data_root == (
            config.data.params.validation_metrics.params.data_root
        )
        assert config.data.params.test.params.manifest_path.endswith(test_suffix)
        assert str(config.data.params.test.params.data_root).replace("\\", "/").endswith(
            test_root_suffix
        )


def test_release_has_no_legacy_btcv_slice_subset_reference():
    public_text = "\n".join(
        path.read_text(encoding="utf-8-sig", errors="replace")
        for path in list(CODE_ROOT.rglob("*.py")) + list(CODE_ROOT.rglob("*.yaml"))
    )
    assert "test_10pct" not in public_text
    assert "test_avg_dice" not in "\n".join(
        path.read_text(encoding="utf-8-sig", errors="replace")
        for path in CONFIGS.values()
    )


def test_internal_log_dice_reads_metric_view_of_validation_and_final_cli_uses_test_prefix():
    ddpm = (CODE_ROOT / "ldm" / "models" / "diffusion" / "ddpm.py").read_text(
        encoding="utf-8-sig"
    )
    inference = (CODE_ROOT / "scripts" / "slice2seg.py").read_text(
        encoding="utf-8-sig"
    )
    assert 'datasets["validation_metrics"]' in ddpm
    assert 'metric_prefix = "val"' in ddpm
    assert 'metric_prefix="test"' in inference
