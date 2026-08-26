from pathlib import Path
import sys

import torch

CODE_ROOT = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_ROOT))

from ldm.audit import audited_y_t_input, normalize_audit_mode


def test_normalize_audit_mode_accepts_aliases_and_train_random_yt():
    assert normalize_audit_mode(None) == "none"
    assert normalize_audit_mode("full_diffusion") == "none"
    assert normalize_audit_mode("TRAIN_RANDOM_YT") == "train_random_yt"
    assert normalize_audit_mode("core_no_diff") == "core_no_diff"


def test_default_audit_mode_returns_original_y_t_reference():
    y_t_ref = torch.ones((2, 4, 8, 8), dtype=torch.float32)

    y_t_input, trace = audited_y_t_input(y_t_ref, "none", training=True)

    assert y_t_input is y_t_ref
    assert trace["train_random_yt_active"] is False
    assert trace["y_t_input_source"] == "original_q_sample"
    assert trace["target_changed"] is False
    assert trace["loss_changed"] is False


def test_train_random_yt_replaces_only_y_t_input_tensor_contract():
    torch.manual_seed(123)
    y_t_ref = torch.ones((2, 4, 8, 8), dtype=torch.float32)
    target = torch.full_like(y_t_ref, 0.25)
    t = torch.tensor([7, 11], dtype=torch.long)

    y_t_input, trace = audited_y_t_input(y_t_ref, "train_random_yt", training=True)

    assert y_t_input.shape == y_t_ref.shape
    assert y_t_input.dtype == y_t_ref.dtype
    assert y_t_input.device == y_t_ref.device
    assert not torch.equal(y_t_input, y_t_ref)
    assert torch.equal(target, torch.full_like(y_t_ref, 0.25))
    assert torch.equal(t, torch.tensor([7, 11], dtype=torch.long))
    assert trace["train_random_yt_active"] is True
    assert trace["same_shape"] is True
    assert trace["same_dtype"] is True
    assert trace["same_device"] is True
    assert trace["target_changed"] is False
    assert trace["loss_changed"] is False
    assert trace["model_output_changed"] is False
    assert trace["image_condition_changed"] is False
    assert trace["timestep_changed"] is False
    assert trace["epsilon_noise_target_replaced_by_y_t_input_noise"] is False


def test_train_random_yt_is_training_only():
    torch.manual_seed(123)
    y_t_ref = torch.ones((2, 4, 8, 8), dtype=torch.float32)

    y_t_input, trace = audited_y_t_input(y_t_ref, "train_random_yt", training=False)

    assert y_t_input is y_t_ref
    assert trace["train_random_yt_active"] is False
    assert trace["y_t_input_source"] == "original_q_sample"


def test_latent_diffusion_loss_uses_reference_y_t_for_original_loss_contract():
    root = CODE_ROOT
    text = (root / "ldm" / "models" / "diffusion" / "ddpm.py").read_text(encoding="utf-8")
    audit_text = (root / "ldm" / "audit.py").read_text(encoding="utf-8")

    assert "model_output = self.apply_model(x_noisy_input, t, cond)" in text
    assert "target = noise" in text
    assert (
        "loss_seg = self.get_loss_seg_regression(x_start, x_noisy_ref, t, model_output, "
        "seg_loss_type='default').mean([1, 2, 3])"
    ) in text
    assert "epsilon_noise_target_replaced_by_y_t_input_noise" in audit_text


if __name__ == "__main__":
    test_normalize_audit_mode_accepts_aliases_and_train_random_yt()
    test_default_audit_mode_returns_original_y_t_reference()
    test_train_random_yt_replaces_only_y_t_input_tensor_contract()
    test_train_random_yt_is_training_only()
    test_latent_diffusion_loss_uses_reference_y_t_for_original_loss_contract()
    print("train_random_yt audit tests passed")
