from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from guided_diffusion.gaussian_diffusion import ModelMeanType
from guided_diffusion.nn import mean_flat
from guided_diffusion.script_util import create_gaussian_diffusion


class CaptureModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input = None
        self.timesteps = None

    def forward(self, x, timesteps, **_kwargs):
        self.input = x.detach().clone()
        self.timesteps = timesteps.detach().clone()
        return torch.zeros_like(x[:, -1:, ...]), torch.zeros_like(x[:, -1:, ...])


class CoreOnlyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.image = None

    def core_no_diff_logits(self, image):
        self.image = image.detach().clone()
        return torch.zeros_like(image[:, :1, ...])

    def forward(self, *_args, **_kwargs):
        raise AssertionError("core_no_diff must not call timestep-conditioned forward")


def _make_case():
    diffusion = create_gaussian_diffusion(steps=1000, learn_sigma=False)
    x_start = torch.zeros((2, 4, 8, 8), dtype=torch.float32)
    x_start[:, :3, ...] = 0.5
    x_start[0, -1, 2:6, 2:6] = 1.0
    x_start[1, -1, 1:4, 1:4] = 1.0
    t = torch.tensor([2, 7], dtype=torch.long)
    noise = torch.linspace(-1.0, 1.0, steps=2 * 8 * 8, dtype=torch.float32).reshape(2, 1, 8, 8)
    res = torch.where(x_start[:, -1:, ...] > 0, 1, 0)
    y_t_ref = diffusion.q_sample(res, t, noise=noise)
    return diffusion, x_start, t, noise, y_t_ref


def test_default_training_uses_reference_yt_and_original_noise_target():
    diffusion, x_start, t, noise, y_t_ref = _make_case()
    model = CaptureModel()

    terms, _sample = diffusion.training_losses_segmentation(
        model,
        None,
        x_start.clone(),
        t,
        noise=noise,
        audit_mode="none",
    )

    assert diffusion.model_mean_type == ModelMeanType.EPSILON
    assert torch.allclose(model.input[:, -1:, ...], y_t_ref.float())
    assert torch.equal(model.timesteps, t)
    assert torch.allclose(terms["loss_diff"], mean_flat(noise**2))
    assert "loss_cal" not in terms
    assert torch.allclose(terms["loss"], terms["loss_diff"])


def test_train_random_yt_replaces_only_model_input_yt():
    diffusion, x_start, t, noise, y_t_ref = _make_case()
    model = CaptureModel()
    torch.manual_seed(123)

    terms, _sample = diffusion.training_losses_segmentation(
        model,
        None,
        x_start.clone(),
        t,
        noise=noise,
        audit_mode="train_random_yt",
    )

    y_t_input = model.input[:, -1:, ...]
    assert y_t_input.shape == y_t_ref.shape
    assert y_t_input.dtype == y_t_ref.dtype
    assert y_t_input.device == y_t_ref.device
    assert not torch.allclose(y_t_input, y_t_ref)
    assert torch.equal(model.timesteps, t)
    assert torch.allclose(model.input[:, :3, ...], x_start[:, :3, ...])
    assert torch.allclose(terms["loss_diff"], mean_flat(noise**2))
    assert "loss_cal" not in terms


def test_train_shuffle_yt_replaces_only_model_input_yt_with_other_case_mask():
    diffusion, x_start, t, noise, y_t_ref = _make_case()
    model = CaptureModel()
    res = torch.where(x_start[:, -1:, ...] > 0, 1, 0)
    expected = diffusion.q_sample(res[[1, 0]], t, noise=noise)

    terms, _sample = diffusion.training_losses_segmentation(
        model,
        None,
        x_start.clone(),
        t,
        noise=noise,
        audit_mode="train_shuffle_yt",
    )

    y_t_input = model.input[:, -1:, ...]
    assert y_t_input.shape == y_t_ref.shape
    assert y_t_input.dtype == y_t_ref.dtype
    assert y_t_input.device == y_t_ref.device
    assert torch.allclose(y_t_input, expected.float())
    assert not torch.allclose(y_t_input, y_t_ref.float())
    assert torch.equal(model.timesteps, t)
    assert torch.allclose(model.input[:, :3, ...], x_start[:, :3, ...])
    assert torch.allclose(terms["loss_diff"], mean_flat(noise**2))
    assert "loss_cal" not in terms


def test_train_shuffle_yt_requires_batch_size_greater_than_one():
    diffusion, x_start, t, noise, _y_t_ref = _make_case()
    model = CaptureModel()

    try:
        diffusion.training_losses_segmentation(
            model,
            None,
            x_start[:1].clone(),
            t[:1],
            noise=noise[:1],
            audit_mode="train_shuffle_yt",
        )
    except ValueError as exc:
        assert "requires batch_size > 1" in str(exc)
    else:
        raise AssertionError("train_shuffle_yt should reject batch_size=1")


def test_core_no_diff_uses_direct_image_only_logits():
    diffusion, x_start, t, noise, _y_t_ref = _make_case()
    model = CoreOnlyModel()

    terms, logits = diffusion.training_losses_segmentation(
        model,
        None,
        x_start.clone(),
        None,
        noise=noise,
        audit_mode="core_no_diff",
    )

    assert torch.allclose(model.image, x_start[:, :3, ...])
    assert logits.shape == x_start[:, -1:, ...].shape
    assert "loss_cal" not in terms
    assert "loss_diff" not in terms
    assert "loss_bce" in terms
    assert "loss_dice" in terms
    assert "loss_direct" in terms
    assert torch.isfinite(terms["loss"]).all()


if __name__ == "__main__":
    test_default_training_uses_reference_yt_and_original_noise_target()
    test_train_random_yt_replaces_only_model_input_yt()
    test_train_shuffle_yt_replaces_only_model_input_yt_with_other_case_mask()
    test_train_shuffle_yt_requires_batch_size_greater_than_one()
    test_core_no_diff_uses_direct_image_only_logits()
    print("training input audit tests passed")
