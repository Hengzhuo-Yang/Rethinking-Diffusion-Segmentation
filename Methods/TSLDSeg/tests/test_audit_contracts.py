from pathlib import Path
import sys

import pytest
import torch


RELEASE_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = RELEASE_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT))

from ldm.audit import audited_y_t_input, normalize_audit_mode


def test_all_four_release_modes_are_supported():
    assert normalize_audit_mode("full_diffusion") == "none"
    assert normalize_audit_mode("train_random_yt") == "train_random_yt"
    assert normalize_audit_mode("train_shuffle_yt") == "train_shuffle_yt"
    assert normalize_audit_mode("core_no_diff") == "core_no_diff"


def test_shuffle_yt_deranges_x_start_and_reuses_current_t_and_noise():
    torch.manual_seed(23)
    x_start = torch.arange(4, dtype=torch.float32).reshape(4, 1, 1, 1)
    noise = torch.arange(10, 14, dtype=torch.float32).reshape(4, 1, 1, 1)
    t = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    y_t_ref = x_start + noise + t.reshape(4, 1, 1, 1)
    original_target = noise.clone()

    def q_sample_fn(*, x_start, t, noise):
        return x_start + noise + t.reshape(4, 1, 1, 1)

    y_t_input, trace = audited_y_t_input(
        y_t_ref,
        "train_shuffle_yt",
        True,
        x_start=x_start,
        t=t,
        noise=noise,
        q_sample_fn=q_sample_fn,
    )
    permutation = torch.tensor(trace["shuffle_permutation"])
    assert torch.all(permutation != torch.arange(4))
    assert torch.equal(
        y_t_input,
        x_start[permutation] + noise + t.reshape(4, 1, 1, 1),
    )
    assert torch.equal(original_target, noise)
    assert trace["current_timestep_reused_for_y_t_input"] is True
    assert trace["current_noise_reused_for_y_t_input"] is True
    assert trace["target_changed"] is False
    assert trace["loss_changed"] is False


def test_shuffle_yt_rejects_batch_size_one():
    tensor = torch.ones((1, 1, 2, 2))
    with pytest.raises(ValueError, match="batch_size > 1"):
        audited_y_t_input(
            tensor,
            "train_shuffle_yt",
            True,
            x_start=tensor,
            t=torch.tensor([1]),
            noise=tensor,
            q_sample_fn=lambda **kwargs: kwargs["x_start"],
        )


def test_input_side_audits_are_training_only():
    reference = torch.ones((2, 1, 2, 2))
    for mode in ("train_random_yt", "train_shuffle_yt"):
        result, trace = audited_y_t_input(reference, mode, False)
        assert result is reference
        assert trace["audit_active"] is False


def test_ddpm_preserves_existing_audits_and_objective_contracts():
    text = (CODE_ROOT / "ldm" / "models" / "diffusion" / "ddpm.py").read_text(
        encoding="utf-8-sig"
    )
    assert "x_noisy_ref = self.q_sample(x_start=x_start, t=t, noise=noise)" in text
    assert "model_output = self.apply_model(x_noisy_input, t, cond)" in text
    assert "target = noise" in text
    assert "self.get_loss_seg_regression(x_start, x_noisy_ref, t, model_output" in text
    assert "return self._core_no_diff_direct_loss(x_start, model_output, seg_label=seg_label)" in text
    assert "if self._is_core_no_diff()" in text


def test_random_and_shuffle_do_not_reconfigure_unet_channels():
    text = (CODE_ROOT / "ldm" / "models" / "diffusion" / "ddpm.py").read_text(
        encoding="utf-8-sig"
    )
    assert 'normalize_audit_mode(kwargs.get("audit_mode", "none")) != "core_no_diff"' in text
    assert "self._config_set(params, \"in_channels\", image_only_channels)" in text
