from pathlib import Path
import sys
import tempfile

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train import normalize_audit_mode, random_yt_like, write_audit_metadata


def test_normalize_audit_mode_accepts_train_random_yt():
    cfg = OmegaConf.create({"audit_mode": "TRAIN_RANDOM_YT"})
    assert normalize_audit_mode(cfg) == "train_random_yt"


def test_random_yt_like_preserves_shape_dtype_device_and_not_target():
    generator = torch.Generator(device="cpu")
    generator.manual_seed(123)
    y_t_ref = torch.ones((2, 4, 8, 8), dtype=torch.float32)
    target = torch.full_like(y_t_ref, 0.25)
    timesteps = torch.tensor([10, 20], dtype=torch.long)

    y_t_input = random_yt_like(y_t_ref, generator=generator)

    assert y_t_input.shape == y_t_ref.shape
    assert y_t_input.dtype == y_t_ref.dtype
    assert y_t_input.device == y_t_ref.device
    assert not torch.equal(y_t_input, y_t_ref)
    assert torch.equal(target, torch.full_like(y_t_ref, 0.25))
    assert torch.equal(timesteps, torch.tensor([10, 20], dtype=torch.long))


def test_train_random_yt_metadata_records_objective_preserving_contract(tmp_path):
    cfg = OmegaConf.create(
        {
            "seed": 1337,
            "prediction_type": "sample",
            "use_alignment": True,
            "use_ema": True,
        }
    )
    path = tmp_path / "audit_metadata.json"

    write_audit_metadata(str(path), cfg, "train_random_yt", parameter_count=123)

    text = path.read_text(encoding="utf-8")
    assert '"audit_mode": "train_random_yt"' in text
    assert '"original_target_type": "mask_latent/Y_0"' in text
    assert '"target_changed": false' in text
    assert '"loss_changed": false' in text
    assert '"model_output_changed": false' in text
    assert '"yt_input_replaced_by_independent_gaussian": true' in text
    assert '"inference_changed": false' in text
    assert '"evaluation_changed": false' in text


if __name__ == "__main__":
    test_normalize_audit_mode_accepts_train_random_yt()
    test_random_yt_like_preserves_shape_dtype_device_and_not_target()
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_train_random_yt_metadata_records_objective_preserving_contract(Path(tmp_dir))
    print("train_random_yt audit tests passed")
