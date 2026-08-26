from typing import Optional, Tuple

import torch
import torch.nn as nn
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config

from .autoencoder import AutoencoderKL, LatentEncoder
from .unet import UNetModel


class CoreNoDiffSegmentor(ModelMixin, ConfigMixin):
    """Image-only latent x0 predictor for the LEAF core_no_diff audit.

    This removes the noisy mask latent and timestep from the main path while
    preserving the LEAF x0-style latent mask prediction target.
    """

    @register_to_config
    def __init__(
        self,
        encoder_ch: int = 128,
        encoder_ch_mult: Tuple = (1, 2, 4, 4),
        encoder_num_res_blocks: int = 2,
        encoder_attn_resolutions: Tuple = (),
        encoder_dropout: float = 0.2,
        encoder_in_channels: int = 3,
        encoder_resolution: int = 256,
        encoder_z_channels: int = 4,
        encoder_double_z: bool = True,
        encoder_embed_dim: int = 4,
        unet_image_size: int = 32,
        unet_in_channels: int = 4,
        unet_model_channels: int = 192,
        unet_out_channels: int = 4,
        unet_num_res_blocks: int = 2,
        unet_attention_resolutions: Tuple = (1, 2, 4, 8),
        unet_dropout: float = 0.0,
        unet_channel_mult: Tuple = (1, 2, 2, 4, 4),
        unet_conv_resample: bool = True,
        unet_num_classes=None,
        unet_use_checkpoint: bool = False,
        unet_num_heads: int = 8,
        unet_num_head_channels: int = -1,
        unet_num_heads_upsample: int = -1,
        unet_use_scale_shift_norm: bool = False,
        unet_resblock_updown: bool = False,
        unet_use_new_attention_order: bool = False,
        unet_legacy=True,
        use_alignment: bool = True,
        projector_dim: int = 2048,
        dino_dim: int = 768,
        scaling_factor: float = 0.18215,
    ):
        super().__init__()
        self.scaling_factor = float(scaling_factor)
        self.use_alignment = bool(use_alignment)

        self.latent_encoder = LatentEncoder(
            ch=encoder_ch,
            out_ch=3,
            ch_mult=encoder_ch_mult,
            num_res_blocks=encoder_num_res_blocks,
            attn_resolutions=encoder_attn_resolutions,
            dropout=encoder_dropout,
            in_channels=encoder_in_channels,
            resolution=encoder_resolution,
            z_channels=encoder_z_channels,
            double_z=encoder_double_z,
            embed_dim=encoder_embed_dim,
        )
        self.unet = UNetModel(
            image_size=unet_image_size,
            in_channels=unet_in_channels,
            model_channels=unet_model_channels,
            out_channels=unet_out_channels,
            num_res_blocks=unet_num_res_blocks,
            attention_resolutions=unet_attention_resolutions,
            dropout=unet_dropout,
            channel_mult=unet_channel_mult,
            conv_resample=unet_conv_resample,
            num_classes=unet_num_classes,
            use_checkpoint=unet_use_checkpoint,
            num_heads=unet_num_heads,
            num_head_channels=unet_num_head_channels,
            num_heads_upsample=unet_num_heads_upsample,
            use_scale_shift_norm=unet_use_scale_shift_norm,
            resblock_updown=unet_resblock_updown,
            use_new_attention_order=unet_use_new_attention_order,
            legacy=unet_legacy,
        )

        if self.use_alignment:
            self.projector = nn.Sequential(
                nn.Linear(self.unet.model_channels, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, dino_dim),
            )
        else:
            self.projector = None

    @classmethod
    def from_pretrained_components(
        cls,
        vae: AutoencoderKL,
        unet: UNetModel,
        use_alignment: bool = True,
        scaling_factor: float = 0.18215,
    ) -> "CoreNoDiffSegmentor":
        model = cls(
            encoder_ch=vae.config.ch,
            encoder_ch_mult=tuple(vae.config.ch_mult),
            encoder_num_res_blocks=vae.config.num_res_blocks,
            encoder_attn_resolutions=tuple(vae.config.attn_resolutions),
            encoder_dropout=vae.config.dropout,
            encoder_in_channels=vae.config.in_channels,
            encoder_resolution=vae.config.resolution,
            encoder_z_channels=vae.config.z_channels,
            encoder_double_z=vae.config.double_z,
            encoder_embed_dim=vae.config.embed_dim,
            unet_image_size=unet.config.image_size,
            unet_in_channels=unet.config.in_channels,
            unet_model_channels=unet.config.model_channels,
            unet_out_channels=unet.config.out_channels,
            unet_num_res_blocks=unet.config.num_res_blocks,
            unet_attention_resolutions=tuple(unet.config.attention_resolutions),
            unet_dropout=unet.config.dropout,
            unet_channel_mult=tuple(unet.config.channel_mult),
            unet_conv_resample=unet.config.conv_resample,
            unet_num_classes=unet.config.num_classes,
            unet_use_checkpoint=unet.config.use_checkpoint,
            unet_num_heads=unet.config.num_heads,
            unet_num_head_channels=unet.config.num_head_channels,
            unet_num_heads_upsample=unet.config.num_heads_upsample,
            unet_use_scale_shift_norm=unet.config.use_scale_shift_norm,
            unet_resblock_updown=unet.config.resblock_updown,
            unet_use_new_attention_order=unet.config.use_new_attention_order,
            unet_legacy=unet.config.legacy,
            use_alignment=use_alignment,
            scaling_factor=scaling_factor,
        )
        model.latent_encoder.init_from_pretrained(vae)
        model.unet.load_state_dict(unet.state_dict(), strict=True)
        return model

    def _forward_unet_without_timestep(
        self,
        latent: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden_states = []
        z_tilde = None
        time_embed_dim = self.unet.time_embed[-1].out_features
        emb = torch.zeros(
            latent.shape[0],
            time_embed_dim,
            device=latent.device,
            dtype=latent.dtype,
        )

        sample = latent
        for i, module in enumerate(self.unet.input_blocks):
            sample = module(sample, emb)
            hidden_states.append(sample)

            bsz, c, h, w = sample.shape
            if self.projector is not None and i > 0 and h * w == 256 and z_tilde is None:
                z_tilde = self.projector(sample.view(bsz, c, h * w).transpose(-2, -1))

        sample = self.unet.middle_block(sample, emb)

        for module in self.unet.output_blocks:
            sample = torch.cat([sample, hidden_states.pop()], dim=1)
            sample = module(sample, emb)

        return self.unet.out(sample), z_tilde

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"core_no_diff expects image-only [B, 3, H, W] input, got {tuple(image.shape)}")
        image_norm = image * 2.0 - 1.0
        image_latent = self.latent_encoder(image_norm).mode()
        image_latent = image_latent * self.scaling_factor
        pred_mask_latent, z_tilde = self._forward_unet_without_timestep(image_latent.float())
        return pred_mask_latent, z_tilde
