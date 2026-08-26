from pathlib import Path
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from leaf import CoreNoDiffSegmentor


def make_tiny_model():
    return CoreNoDiffSegmentor(
        encoder_ch=32,
        encoder_ch_mult=(1,),
        encoder_num_res_blocks=1,
        encoder_resolution=32,
        unet_image_size=32,
        unet_model_channels=32,
        unet_num_res_blocks=1,
        unet_attention_resolutions=(),
        unet_channel_mult=(1,),
        unet_num_heads=1,
        use_alignment=False,
    )


def test_core_no_diff_forward_is_image_only_and_predicts_latent():
    model = CoreNoDiffSegmentor(
        encoder_ch=32,
        encoder_ch_mult=(1,),
        encoder_num_res_blocks=1,
        encoder_resolution=32,
        unet_image_size=32,
        unet_model_channels=32,
        unet_num_res_blocks=1,
        unet_attention_resolutions=(),
        unet_channel_mult=(1,),
        unet_num_heads=1,
        use_alignment=False,
    )
    image = torch.rand(2, 3, 32, 32)
    pred_mask_latent, z_tilde = model(image)

    assert z_tilde is None
    assert pred_mask_latent.shape == (2, 4, 32, 32)


def test_core_no_diff_rejects_non_image_main_input():
    model = make_tiny_model()
    bad_input = torch.rand(2, 4, 32, 32)

    try:
        model(bad_input)
    except ValueError as exc:
        assert "image-only" in str(exc)
    else:
        raise AssertionError("core_no_diff accepted a non-image main-path input")


def test_core_no_diff_latent_l1_loss_backward():
    model = make_tiny_model()
    image = torch.rand(2, 3, 32, 32)
    target_latent = torch.randn(2, 4, 32, 32)

    pred_mask_latent, _ = model(image)
    loss = F.l1_loss(pred_mask_latent, target_latent)
    loss.backward()

    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())


if __name__ == "__main__":
    test_core_no_diff_forward_is_image_only_and_predicts_latent()
    test_core_no_diff_rejects_non_image_main_input()
    test_core_no_diff_latent_l1_loss_backward()
    print("core_no_diff tests passed")
