from pathlib import Path
import sys

import torch


RELEASE_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = RELEASE_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT))

from guided_diffusion.script_util import create_gaussian_diffusion


class DummyLearnedVarianceModel(torch.nn.Module):
    def __init__(self, num_mask_channels):
        super().__init__()
        self.num_mask_channels = num_mask_channels

    def forward(self, conditioned, timesteps, **_kwargs):
        mask = conditioned[:, -self.num_mask_channels:, ...]
        predicted_noise = torch.zeros_like(mask)
        predicted_variance = torch.zeros_like(mask)
        return torch.cat((predicted_noise, predicted_variance), dim=1), None


def test_dpm_solver_supports_three_mask_channels():
    torch.manual_seed(7)
    num_mask_channels = 3
    diffusion = create_gaussian_diffusion(
        steps=1000,
        learn_sigma=True,
        noise_schedule="linear",
        dpm_solver=True,
        num_mask_channels=num_mask_channels,
    )
    model = DummyLearnedVarianceModel(num_mask_channels)
    image = torch.zeros(2, 3, 8, 8)
    mask_placeholder = torch.zeros(2, num_mask_channels, 8, 8)
    conditioned = torch.cat((image, mask_placeholder), dim=1)

    sample, x_noisy, _image, cal, cal_out = diffusion.p_sample_loop_known(
        model,
        tuple(conditioned.shape),
        conditioned,
        step=2,
        clip_denoised=True,
        model_kwargs={},
        device=torch.device("cpu"),
    )

    assert sample.shape == mask_placeholder.shape
    assert x_noisy.shape == conditioned.shape
    assert torch.isfinite(sample).all()
    assert cal is None
