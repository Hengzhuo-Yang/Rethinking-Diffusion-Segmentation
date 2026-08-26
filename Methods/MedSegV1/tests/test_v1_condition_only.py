import sys
from pathlib import Path

import torch


CODE_ROOT = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_ROOT))

from guided_diffusion.unet import Generic_UNet


def test_encoder_only_skips_localization_and_keeps_encoder_gradients():
    model = Generic_UNet(
        input_channels=3,
        base_num_features=4,
        num_classes=1,
        num_pool=2,
        highway=False,
    )
    model.eval()

    localization_calls = []
    hooks = [
        module.register_forward_hook(
            lambda _module, _inputs, _output: localization_calls.append(True)
        )
        for module in model.conv_blocks_localization
    ]
    try:
        rng_state = torch.get_rng_state()
        embedding = model(
            torch.randn(1, 3, 32, 32),
            hs=None,
            encoder_only=True,
        )
    finally:
        for hook in hooks:
            hook.remove()

    assert embedding.shape[1] == 512
    assert localization_calls == []

    # With the same transient projection initialization, skipping the decoder
    # must not alter the condition embedding consumed by the diffusion U-Net.
    torch.set_rng_state(rng_state)
    reference_embedding, _cal = model(
        torch.randn(1, 3, 32, 32),
        hs=None,
        encoder_only=False,
    )
    torch.testing.assert_close(embedding, reference_embedding)

    embedding.square().mean().backward()
    assert any(
        parameter.grad is not None
        for parameter in model.conv_blocks_context.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in model.conv_blocks_localization.parameters()
    )
    assert all(
        parameter.grad is None for parameter in model.seg_outputs.parameters()
    )
