from pathlib import Path
import sys

import torch
from diffusers import DDIMScheduler
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train import derangement_like_permutation, normalize_audit_mode


def test_derangement_like_permutation_has_no_self_matches():
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1337)

    permutation = derangement_like_permutation(
        batch_size=8,
        device=torch.device("cpu"),
        generator=generator,
    )

    assert sorted(permutation.tolist()) == list(range(8))
    assert not torch.any(permutation == torch.arange(8))


def test_train_shuffle_yt_keeps_target_noise_and_timestep_contract():
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1337)
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0015,
        beta_end=0.0155,
        prediction_type="sample",
    )
    gt_mask_latent = torch.arange(4 * 1 * 2 * 2, dtype=torch.float32).reshape(4, 1, 2, 2)
    noise = torch.randn(gt_mask_latent.shape, generator=generator)
    timesteps = torch.tensor([3, 17, 29, 53]).long()
    original_noise = noise.clone()
    original_timesteps = timesteps.clone()
    target = gt_mask_latent

    y_t_ref = scheduler.add_noise(gt_mask_latent, noise, timesteps)
    permutation = derangement_like_permutation(
        batch_size=gt_mask_latent.shape[0],
        device=torch.device("cpu"),
        generator=generator,
    )
    y_t_input = scheduler.add_noise(gt_mask_latent[permutation], noise, timesteps)

    assert y_t_input.shape == y_t_ref.shape
    assert y_t_input.dtype == y_t_ref.dtype
    assert y_t_input.device == y_t_ref.device
    assert not torch.any(permutation == torch.arange(gt_mask_latent.shape[0]))
    assert torch.equal(target, gt_mask_latent)
    assert torch.equal(noise, original_noise)
    assert torch.equal(timesteps, original_timesteps)
    assert not torch.allclose(y_t_input, y_t_ref)


def test_sample_prediction_scheduler_preserves_unbounded_mask_latents():
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0015,
        beta_end=0.0155,
        prediction_type="sample",
        clip_sample=False,
    )
    scheduler.set_timesteps(1)

    model_pred = torch.tensor([[[[2.5, -3.0], [0.25, -0.5]]]])
    sample = torch.zeros_like(model_pred)
    out = scheduler.step(
        model_pred,
        torch.tensor(999, dtype=torch.long),
        sample,
    ).pred_original_sample

    assert torch.allclose(out, model_pred)


def test_audit_mode_defaults_to_none_and_accepts_train_shuffle_yt():
    assert normalize_audit_mode(OmegaConf.create({})) == "none"
    assert normalize_audit_mode(OmegaConf.create({"audit_mode": "train_shuffle_yt"})) == "train_shuffle_yt"


if __name__ == "__main__":
    test_derangement_like_permutation_has_no_self_matches()
    test_train_shuffle_yt_keeps_target_noise_and_timestep_contract()
    test_audit_mode_defaults_to_none_and_accepts_train_shuffle_yt()
    print("train_shuffle_yt audit tests passed")
