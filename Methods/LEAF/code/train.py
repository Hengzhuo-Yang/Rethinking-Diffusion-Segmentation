import argparse
import csv
import json
import logging
import math
import os
import shutil
import time
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
from tqdm.auto import tqdm

import tensorboard
from accelerate.logging import get_logger
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from omegaconf import OmegaConf

import diffusers
from diffusers import DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel

from leaf import (
    AutoencoderKL,
    CoreNoDiffSegmentor,
    LatentEncoder,
    UNetModel,
    UNetModelWrapper,
    LeafPipeline,
    LeafOutput
)
from src.data import load_custom_dataset
from src.util.loss import cosine_loss
from src.util.metric import SegmentationMetric, Visualization, labels_from_segmentation_tensor
from src.util.runtime import normalize_audit_mode as normalize_audit_mode_value
from src.util.runtime import require_rtx5090, resolve_from_code_root
from src.util.seeding import generate_seed_sequence

logger = get_logger(__name__, log_level="INFO")
CODE_ROOT = Path(__file__).resolve().parent

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train latent diffusion model with VAE and UNet"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML config file"
    )
    parser.add_argument("--data-root", type=str, help="Override dataset-specific base_data_dir")
    parser.add_argument("--output-dir", type=str, help="Override the run output root")
    parser.add_argument("--pretrained-path", type=str, help="Override local LEAF VAE/U-Net assets")
    parser.add_argument("--job-name", type=str, help="Override the config job name")
    parser.add_argument("--max-train-steps", type=int, help="Explicit smoke/debug override")
    parser.add_argument("--validation-start-steps", type=int, help="Explicit smoke/debug override")
    parser.add_argument("--validation-steps", type=int, help="Explicit smoke/debug override")
    parser.add_argument("--train-batch-size", type=int, help="Explicit smoke/debug override")
    parser.add_argument("--validation-batch-size", type=int, help="Explicit smoke/debug override")
    parser.add_argument("--num-workers", type=int, help="Explicit smoke/debug override")
    return parser.parse_args()

def apply_runtime_overrides(cfg: OmegaConf, args: argparse.Namespace) -> None:
    overrides = {
        "base_data_dir": args.data_root,
        "output_dir": args.output_dir,
        "pretrained_model_name_or_path": args.pretrained_path,
        "job_name": args.job_name,
        "max_train_steps": args.max_train_steps,
        "validation_start_steps": args.validation_start_steps,
        "validation_steps": args.validation_steps,
        "train_batch_size": args.train_batch_size,
        "test_batch_size": args.validation_batch_size,
        "num_workers": args.num_workers,
    }
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value
    for key in ("base_data_dir", "output_dir", "pretrained_model_name_or_path"):
        cfg[key] = resolve_from_code_root(str(cfg[key]), CODE_ROOT)
    if int(cfg.train_batch_size) < 1 or int(cfg.test_batch_size) < 1:
        raise ValueError("Training and validation batch sizes must be positive.")
    if int(cfg.max_train_steps) < 1:
        raise ValueError("max_train_steps must be positive.")
    expected_validation = {"BTCV": "val", "ACDC": "validation", "ISIC18": "validation"}
    expected = expected_validation.get(str(cfg.dataset_name))
    if expected is None or str(cfg.validation_split) != expected:
        raise ValueError(
            f"Release validation split mismatch for {cfg.dataset_name}: "
            f"expected {expected!r}, received {cfg.validation_split!r}."
        )
    cfg.validation_csv = str(Path(cfg.output_dir) / str(cfg.job_name) / "validation_metrics.csv")

def setup_directories(base_output: str, job_name: str, save_visualization: bool = False) -> Dict[str, str]:
    out_run = os.path.join(base_output, job_name)
    dirs = {
        "run": out_run,
        "ckpt": os.path.join(out_run, "checkpoint"),
        "tb": os.path.join(out_run, "tensorboard"),
    }
    if save_visualization:
        dirs["vis"] = os.path.join(out_run, "visualization")
    return dirs

def append_validation_csv(path: str, step: int, metrics: Dict[str, float]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["step"] + sorted(metrics.keys())
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        row = {"step": step}
        row.update({key: metrics[key] for key in sorted(metrics.keys())})
        writer.writerow(row)


def normalize_audit_mode(cfg: OmegaConf) -> str:
    return normalize_audit_mode_value(cfg.get("audit_mode", "none"))


def write_best_checkpoint_metadata(
    run_dir: str,
    checkpoint_dir: str,
    cfg: OmegaConf,
    audit_mode: str,
    step: int,
    validation_dice: float,
) -> None:
    payload = {
        "checkpoint": os.path.relpath(checkpoint_dir, run_dir).replace("\\", "/"),
        "checkpoint_step": int(step),
        "dataset": str(cfg.dataset_name),
        "audit_mode": audit_mode,
        "training_seed": int(cfg.seed),
        "selection_partition": "validation",
        "validation_split": str(cfg.validation_split),
        "selection_metric": "mean_validation_dice",
        "validation_dice": float(validation_dice),
        "validation_csv": "validation_metrics.csv",
    }
    with open(os.path.join(run_dir, "best_checkpoint.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def derangement_like_permutation(
    batch_size: int,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if batch_size <= 1:
        raise ValueError("audit_mode=train_shuffle_yt requires batch_size > 1 for batch-level shuffling.")
    order = torch.randperm(batch_size, device=device, generator=generator)
    shifted_order = torch.roll(order, shifts=1, dims=0)
    permutation = torch.empty_like(order)
    permutation[order] = shifted_order
    if torch.any(permutation == torch.arange(batch_size, device=device)):
        raise RuntimeError("Failed to construct a no-self-match permutation for train_shuffle_yt.")
    return permutation


def random_yt_like(
    y_t_ref: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    y_t_input = torch.randn(
        y_t_ref.shape,
        device=y_t_ref.device,
        dtype=y_t_ref.dtype,
        generator=generator,
    )
    if (
        y_t_input.shape != y_t_ref.shape
        or y_t_input.dtype != y_t_ref.dtype
        or y_t_input.device != y_t_ref.device
    ):
        raise RuntimeError(
            "train_random_yt produced a Y_t input with shape, dtype, or device "
            "different from the original Y_t reference."
        )
    return y_t_input


def prediction_target_type(prediction_type: str) -> str:
    if prediction_type == "epsilon":
        return "epsilon/noise"
    if prediction_type == "sample":
        return "mask_latent/Y_0"
    if prediction_type == "v_prediction":
        return "v_prediction"
    return f"unknown:{prediction_type}"


def write_audit_metadata(
    path: str,
    cfg: OmegaConf,
    audit_mode: str,
    parameter_count: int,
) -> None:
    if audit_mode not in {"train_shuffle_yt", "train_random_yt"}:
        return
    prediction_type = str(cfg.prediction_type)
    target_type = prediction_target_type(prediction_type)
    retained_auxiliary_modules = ["latent_encoder"]
    if bool(cfg.use_alignment):
        retained_auxiliary_modules.append("DINOv2 feature alignment projector and cosine distillation loss")
    if bool(cfg.use_ema):
        retained_auxiliary_modules.append("EMA model tracking")

    metadata = {
        "audit_mode": audit_mode,
        "random_seed": cfg.get("seed", None),
        "shuffle_strategy": "batch_level" if audit_mode == "train_shuffle_yt" else "not_applicable",
        "no_self_match_enforced": audit_mode == "train_shuffle_yt",
        "random_Yt_type": (
            "independent standard Gaussian"
            if audit_mode == "train_random_yt"
            else "not_applicable"
        ),
        "original_output_type": target_type,
        "original_target_type": target_type,
        "original_loss_type": "L1(model_pred, original_target)"
        + (" + lam * cosine_loss(DINOv2(image), z_tilde)" if bool(cfg.use_alignment) else ""),
        "target_changed": False,
        "loss_changed": False,
        "model_output_changed": False,
        "image_shuffled": False,
        "timestep_changed": False,
        "original_target_shuffled_or_modified": False,
        "original_loss_modified": False,
        "yt_input_constructed_from_another_case_mask": audit_mode == "train_shuffle_yt",
        "yt_input_replaced_by_independent_gaussian": audit_mode == "train_random_yt",
        "epsilon_or_noise_target_replaced_from_shuffled_mask": False,
        "epsilon_or_noise_target_replaced_by_random_yt_input_noise": False,
        "validation_changed": False,
        "inference_changed": False,
        "evaluation_changed": False,
        "objective_preserving": True,
        "training_changes": True,
        "retained_auxiliary_modules": retained_auxiliary_modules,
        "modified_auxiliary_modules": [],
        "model_parameter_count": parameter_count,
        "model_parameter_count_identical_to_full_diffusion": True,
        "prediction_type": prediction_type,
        "sampling_steps_for_training_validation": 1,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def write_core_no_diff_metadata(
    path: str,
    cfg: OmegaConf,
    parameter_count_full_diffusion: int,
    parameter_count_core_no_diff: int,
    training_time_seconds: Optional[float] = None,
    gpu_memory_peak_bytes: Optional[int] = None,
) -> None:
    retained_auxiliary_modules = [
        "image latent encoder initialized from LEAF VAE encoder",
        "x0 mask-latent prediction target",
        "L1 latent prediction loss",
        "VAE mask decoder for validation/evaluation",
    ]
    modified_auxiliary_modules = ["denoising U-Net input stem uses image latent only instead of [image latent, noisy mask latent]"]
    if bool(cfg.use_alignment):
        retained_auxiliary_modules.append("DINOv2 feature alignment")
        modified_auxiliary_modules.append("alignment projector attached to image-only latent tokens")
    if bool(cfg.use_ema):
        retained_auxiliary_modules.append("EMA model tracking")

    metadata = {
        "audit_mode": "core_no_diff",
        "counterfactual_type": "discriminative_capacity",
        "objective_preserving": False,
        "random_seed": cfg.get("seed", None),
        "parameter_count_full_diffusion": parameter_count_full_diffusion,
        "parameter_count_core_no_diff": parameter_count_core_no_diff,
        "changed_input_channels": True,
        "changed_prediction_head": False,
        "removed_timestep_embedding": True,
        "removed_noise_branch": True,
        "training_time_seconds": training_time_seconds,
        "gpu_memory_peak_bytes": gpu_memory_peak_bytes,
        "sampling_steps_main_core": 0,
        "main_core_input": "image_only",
        "uses_Y_t_in_main_core": False,
        "uses_timestep_in_main_core": False,
        "uses_q_sample_for_main_core": False,
        "uses_forward_noising_for_main_core": False,
        "uses_reverse_sampler_for_main_core": False,
        "diffusion_loss_used_for_main_core": False,
        "main_core_loss": "L1(predicted_mask_latent, gt_mask_latent)",
        "original_prediction_type": str(cfg.get("prediction_type", "sample")),
        "preserves_x0_latent_prediction_objective": str(cfg.get("prediction_type", "sample")) == "sample",
        "retained_auxiliary_modules": retained_auxiliary_modules,
        "modified_auxiliary_modules": modified_auxiliary_modules,
        "incompatible_auxiliary_modules": [
            "diffusion noisy-mask input branch: requires Y_t",
            "reverse diffusion sampler: requires sampled mask latent state and timestep",
        ],
        "objective_changes": [
            "main objective keeps x0 clean mask latent prediction for prediction_type=sample, but removes noised-mask conditioning and timestep conditioning",
            "main inference changed from reverse denoising to single image-only latent prediction followed by VAE decode",
        ],
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def replace_unet_input(unet: UNetModel, log_update: bool = True):
    _n_convin_out_channel = unet.input_blocks[0][0].out_channels
    _new_conv_in = nn.Conv2d(
        8, _n_convin_out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
    )
    _weight = unet.input_blocks[0][0].weight.clone()
    _bias = unet.input_blocks[0][0].bias.clone()
    _weight = _weight.repeat((1, 2, 1, 1))  # Keep selected channel(s)
    # half the activation magnitude
    _weight *= 0.5
    _new_conv_in.weight = nn.Parameter(_weight)
    _new_conv_in.bias = nn.Parameter(_bias)
    unet.input_blocks[0][0] = _new_conv_in
    unet.register_to_config(in_channels=8)
    if log_update:
        logger.info("Unet config is updated")
    return

@torch.no_grad()
def load_dinov2(model_name: str, resolution: int = 256) -> nn.Module:
    import timm
    encoder = torch.hub.load('facebookresearch/dinov2', model_name)
    del encoder.head  # Remove classification head
    patch_resolution = 16 * (resolution // 256)
    encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
        encoder.pos_embed.data, [patch_resolution, patch_resolution]
    )
    encoder.head = torch.nn.Identity()  # Replace head with identity
    encoder.eval()
    return encoder


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    mask_values = torch.stack([example["mask_values"] for example in examples])
    mask_values = mask_values.to(memory_format=torch.contiguous_format).float()
    return {"pixel_values": pixel_values, "mask_values": mask_values}


@torch.no_grad()
def log_validation(
    vae: AutoencoderKL,
    latent_encoder: AutoencoderKL,
    unet: UNetModel,
    noise_scheduler: DDIMScheduler,
    valid_dataloader: DataLoader,
    cfg: OmegaConf,
    accelerator: Accelerator,
    save_directory: Optional[str],
    weight_dtype: torch.dtype,
    step: int
):
    pipeline = LeafPipeline(
        vae=accelerator.unwrap_model(vae),
        unet=accelerator.unwrap_model(unet),
        latent_encoder=accelerator.unwrap_model(latent_encoder),
        scheduler=noise_scheduler,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    pipeline.unet.eval()
    pipeline.latent_encoder.eval()

    val_seed_ls = generate_seed_sequence(cfg.seed, len(valid_dataloader))
    
    num_seg_classes = int(cfg.get("num_seg_classes", 2))
    metrics = SegmentationMetric(metrics=cfg.metrics, device=accelerator.device, num_classes=num_seg_classes)
    save_visualization = bool(cfg.get("save_visualization", False))
    visualization = Visualization() if save_visualization else None

    with torch.autocast(accelerator.device.type):
        for batch in tqdm(
            valid_dataloader,
            desc="Validating",
            total=len(valid_dataloader),
            # disable=not accelerator.is_local_main_process
        ):
            # Read input image (tensor)
            rgb = batch["pixel_values"]  # [B, 3, H, W]
            # GT mask
            mask_gt = batch["mask_values"].to(accelerator.device)

            # Random number generator
            seed = val_seed_ls.pop()
            if seed is None:
                generator = None
            else:
                generator = torch.Generator(device=accelerator.device)
                generator.manual_seed(seed)
            
            # Predict mask
            with torch.autocast(accelerator.device.type):
                pipe_out: LeafOutput = pipeline(
                    rgb, # don't need to norm outside
                    num_inference_steps=1,
                    generator=generator,
                    show_progress_bar=False,
                )

            mask_pred = pipe_out.mask_pred
            if num_seg_classes == 2:
                mask_pred_index = (torch.mean(mask_pred, dim=1) > 0.5).long()
                mask_gt_index = (torch.mean(mask_gt, dim=1) > 0.5).long()
            else:
                mask_pred_index = labels_from_segmentation_tensor(mask_pred, num_seg_classes)
                mask_gt_index = labels_from_segmentation_tensor(mask_gt, num_seg_classes)
            metrics.update(mask_pred_index, mask_gt_index)

            if visualization is not None:
                mask_pred_vis = (mask_pred_index > 0).long().unsqueeze(1)
                mask_gt_vis = (mask_gt_index > 0).long().unsqueeze(1)
                visualization.update(
                    rgb.cpu(),
                    mask_pred_vis.cpu(),
                    mask_gt_vis.cpu(),
                )

    if visualization is not None and save_directory:
        nrow = min(int(cfg.get("nrow", 6)), visualization.images.shape[0])
        if nrow > 0:
            os.makedirs(save_directory, exist_ok=True)
            grid = visualization.sample(nrow=nrow)
            images = Image.fromarray((grid * 255).astype(np.uint8))
            images.save(os.path.join(save_directory, f"step-{step:06d}.jpg"))

    results = metrics.compute()
    accelerator.log(results, step=step)

    del pipeline
    # torch.cuda.empty_cache()
    return results


@torch.no_grad()
def log_core_no_diff_validation(
    vae: AutoencoderKL,
    model: CoreNoDiffSegmentor,
    valid_dataloader: DataLoader,
    cfg: OmegaConf,
    accelerator: Accelerator,
    save_directory: Optional[str],
    weight_dtype: torch.dtype,
    step: int,
):
    model = accelerator.unwrap_model(model)
    model.eval()
    vae = accelerator.unwrap_model(vae)
    vae.eval()

    num_seg_classes = int(cfg.get("num_seg_classes", 2))
    metrics = SegmentationMetric(metrics=cfg.metrics, device=accelerator.device, num_classes=num_seg_classes)
    save_visualization = bool(cfg.get("save_visualization", False))
    visualization = Visualization() if save_visualization else None
    threshold = float(cfg.get("threshold", 0.5))
    num_cases = 0
    start = time.perf_counter()

    autocast_ctx = (
        torch.autocast(accelerator.device.type, dtype=weight_dtype)
        if accelerator.device.type == "cuda" and weight_dtype != torch.float32
        else nullcontext()
    )
    with autocast_ctx:
        for batch in tqdm(
            valid_dataloader,
            desc="Validating core_no_diff",
            total=len(valid_dataloader),
        ):
            rgb = batch["pixel_values"].to(accelerator.device)
            mask_gt = batch["mask_values"].to(accelerator.device)
            pred_mask_latent, _ = model(rgb)
            mask_pred = vae.decode(pred_mask_latent / model.scaling_factor)
            mask_pred = torch.clamp(mask_pred, -1.0, 1.0)
            mask_pred = (mask_pred + 1.0) / 2.0

            if num_seg_classes == 2:
                mask_pred_index = (torch.mean(mask_pred, dim=1, keepdim=True) > threshold).long()
                mask_gt_index = (torch.mean(mask_gt, dim=1, keepdim=True) > threshold).long()
            else:
                mask_pred_index = labels_from_segmentation_tensor(mask_pred, num_seg_classes)
                mask_gt_index = labels_from_segmentation_tensor(mask_gt, num_seg_classes)
            metrics.update(mask_pred_index, mask_gt_index)
            num_cases += rgb.shape[0]

            if visualization is not None:
                mask_pred_vis = (labels_from_segmentation_tensor(mask_pred_index, num_seg_classes) > 0).long().unsqueeze(1)
                mask_gt_vis = (labels_from_segmentation_tensor(mask_gt_index, num_seg_classes) > 0).long().unsqueeze(1)
                visualization.update(
                    rgb.cpu(),
                    mask_pred_vis.cpu(),
                    mask_gt_vis.cpu(),
                )

    if visualization is not None and save_directory:
        nrow = min(int(cfg.get("nrow", 6)), visualization.images.shape[0])
        if nrow > 0:
            os.makedirs(save_directory, exist_ok=True)
            grid = visualization.sample(nrow=nrow)
            images = Image.fromarray((grid * 255).astype(np.uint8))
            images.save(os.path.join(save_directory, f"step-{step:06d}.jpg"))

    results = metrics.compute()
    accelerator.log(results, step=step)
    elapsed = time.perf_counter() - start
    if num_cases > 0:
        logger.info(
            "core_no_diff validation step=%s inference_time_per_case_s=%.6f",
            step,
            elapsed / num_cases,
        )
    return results


def train_core_no_diff(
    cfg: OmegaConf,
    dirs: Dict[str, str],
    accelerator: Accelerator,
):
    train_start = time.perf_counter()
    num_seg_classes = int(cfg.get("num_seg_classes", 2))
    if num_seg_classes < 2:
        raise ValueError(f"num_seg_classes must be >= 2, got {num_seg_classes}")

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if accelerator.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(accelerator.device)

    vae: AutoencoderKL = AutoencoderKL.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="vae", local_files_only=True
    )
    base_unet = UNetModel.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="unet", local_files_only=True
    )
    core_model = CoreNoDiffSegmentor.from_pretrained_components(
        vae=vae,
        unet=base_unet,
        use_alignment=bool(cfg.use_alignment),
    )
    scaling_factor = core_model.scaling_factor

    full_unet_for_count = UNetModel.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="unet", local_files_only=True
    )
    replace_unet_input(full_unet_for_count, log_update=False)
    full_wrapper_for_count = UNetModelWrapper(full_unet_for_count, cfg.use_alignment)
    full_latent_for_count = LatentEncoder()
    full_latent_for_count.init_from_pretrained(vae)
    parameter_count_full_diffusion = count_parameters(full_wrapper_for_count) + count_parameters(full_latent_for_count)
    parameter_count_core_no_diff = count_parameters(core_model)
    del base_unet, full_unet_for_count, full_wrapper_for_count, full_latent_for_count

    if cfg.use_alignment:
        vision_encoder = load_dinov2(model_name=cfg.vision_encoder_model, resolution=int(cfg.resolution))
        vision_encoder.requires_grad_(False)
        vision_encoder.to(accelerator.device, dtype=weight_dtype)
    else:
        vision_encoder = None

    if cfg.use_ema:
        ema_core_model = deepcopy(core_model)
        ema_core_model: EMAModel = EMAModel(
            ema_core_model.parameters(),
            model_cls=CoreNoDiffSegmentor,
            model_config=ema_core_model.config,
            foreach=True,
        )

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            if cfg.use_ema:
                ema_core_model.save_pretrained(os.path.join(output_dir, "core_no_diff_segmentor_ema"))

            for model in models:
                if isinstance(model, CoreNoDiffSegmentor):
                    model.save_pretrained(os.path.join(output_dir, "core_no_diff_segmentor"))

                weights.pop()

    accelerator.register_save_state_pre_hook(save_model_hook)

    def model_parameters():
        return list(core_model.parameters())

    if accelerator.is_main_process:
        write_core_no_diff_metadata(
            os.path.join(dirs["run"], "audit_metadata.json"),
            cfg,
            parameter_count_full_diffusion=parameter_count_full_diffusion,
            parameter_count_core_no_diff=parameter_count_core_no_diff,
        )
        logger.info(
            "core_no_diff is active: main path predicts clean mask latent from image latent only; "
            "Y_t, timestep sampling/embedding, forward noising for main input, and reverse sampling are disabled."
        )
        logger.info(
            "core_no_diff parameter_count_full_diffusion=%s parameter_count_core_no_diff=%s",
            parameter_count_full_diffusion,
            parameter_count_core_no_diff,
        )

    vae.requires_grad_(False)
    vae.to(accelerator.device, dtype=weight_dtype)
    optimizer = optim.AdamW(model_parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    lr_scheduler = get_scheduler(
        name="constant_with_warmup",
        optimizer=optimizer,
        num_warmup_steps=cfg.num_warmup_steps * accelerator.num_processes,
        num_training_steps=cfg.max_train_steps * accelerator.num_processes,
    )

    logger.info(f"Loading Dataset {cfg.dataset_name}...")
    train_dataset, val_dataset = load_custom_dataset(
        base_data_dir=cfg.base_data_dir,
        dataset_name=cfg.dataset_name,
        resolution=cfg.resolution,
        seed=cfg.seed,
        eval_split=cfg.validation_split,
        num_seg_classes=num_seg_classes,
    )

    imagenet_transforms = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    train_dataloader = DataLoader(
        train_dataset,
        cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )
    val_dataloader = DataLoader(
        val_dataset,
        cfg.test_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )

    core_model, vae, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        core_model, vae, optimizer, train_dataloader, lr_scheduler
    )
    if cfg.use_ema:
        ema_core_model.to(accelerator.device)

    logger.info("Start core_no_diff Training...")
    global_seed_sequence = generate_seed_sequence(
        initial_seed=cfg.seed,
        length=cfg.max_train_steps,
    )
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / accelerator.gradient_accumulation_steps
    )
    max_epoch = math.ceil(cfg.max_train_steps / num_update_steps_per_epoch)

    global_step = 0
    progress_bar = tqdm(
        range(0, cfg.max_train_steps),
        initial=0,
        desc="Training Steps",
        disable=not accelerator.is_local_main_process,
    )
    best_dice_score = float("-inf")
    best_ckpt_step = None
    validation_csv = getattr(
        cfg,
        "validation_csv",
        os.path.join(dirs["run"], "validation_metrics.csv"),
    )

    for epoch in range(max_epoch):
        core_model.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(core_model):
                if cfg.seed is not None:
                    global_seed_sequence.pop()

                rgb: torch.Tensor = batch["pixel_values"].to(accelerator.device)
                mask: torch.Tensor = batch["mask_values"].to(accelerator.device)
                mask_norm = mask * 2.0 - 1.0

                with torch.no_grad():
                    gt_mask_latent = vae.encode(mask_norm.to(weight_dtype)).mode()
                    gt_mask_latent = gt_mask_latent * scaling_factor

                pred_mask_latent, z_tilde = core_model(rgb)
                if pred_mask_latent.shape != gt_mask_latent.shape:
                    raise RuntimeError(
                        f"core_no_diff predicted latent shape {tuple(pred_mask_latent.shape)} does not match "
                        f"target mask latent shape {tuple(gt_mask_latent.shape)}"
                    )
                loss = F.l1_loss(pred_mask_latent.float(), gt_mask_latent.float())

                if cfg.use_alignment:
                    if z_tilde is None:
                        raise RuntimeError("core_no_diff use_alignment=True but model did not return alignment tokens.")
                    with torch.no_grad():
                        rgb_for_dino = F.interpolate(rgb, 224 * (cfg.resolution // 256), mode="bicubic")
                        rgb_for_dino = imagenet_transforms(rgb_for_dino)
                        z: torch.Tensor = vision_encoder.forward_features(rgb_for_dino.to(weight_dtype))["x_norm_patchtokens"]
                    distill_loss = cosine_loss(z.float(), z_tilde.float())
                    loss += cfg.lam * distill_loss.mean()

                avg_loss = accelerator.gather(loss.repeat(rgb.shape[0])).mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model_parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                if cfg.use_ema:
                    ema_core_model.step(core_model.parameters())
                progress_bar.update(1)
                global_step += 1

                if global_step >= cfg.validation_start_steps:
                    if accelerator.is_main_process:
                        if global_step % cfg.validation_steps == 0:
                            if cfg.use_ema:
                                ema_core_model.store(core_model.parameters())
                                ema_core_model.copy_to(core_model.parameters())

                            metrics = log_core_no_diff_validation(
                                vae,
                                core_model,
                                val_dataloader,
                                cfg,
                                accelerator,
                                dirs.get("vis"),
                                weight_dtype,
                                global_step,
                            )
                            append_validation_csv(validation_csv, global_step, metrics)
                            if cfg.use_ema:
                                ema_core_model.restore(core_model.parameters())

                            if best_dice_score < metrics["dice"]:
                                last_best_ckpt_path = (
                                    os.path.join(dirs["ckpt"], f"step-{best_ckpt_step}")
                                    if best_ckpt_step is not None
                                    else None
                                )
                                best_dice_score = metrics["dice"]
                                best_ckpt_step = global_step
                                if accelerator.is_main_process:
                                    if last_best_ckpt_path and os.path.exists(last_best_ckpt_path):
                                        shutil.rmtree(last_best_ckpt_path)
                                    save_path = os.path.join(dirs["ckpt"], f"step-{global_step}")
                                    accelerator.save_state(save_path)
                                    write_best_checkpoint_metadata(
                                        dirs["run"], save_path, cfg, "core_no_diff",
                                        global_step, best_dice_score,
                                    )
                                    logger.info(f"Best Dice Score at step {global_step}: {best_dice_score:.4f}")
                                    logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= cfg.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()

    training_time_seconds = time.perf_counter() - train_start
    gpu_memory_peak_bytes = None
    if accelerator.device.type == "cuda":
        gpu_memory_peak_bytes = int(torch.cuda.max_memory_allocated(accelerator.device))
    if accelerator.is_main_process:
        write_core_no_diff_metadata(
            os.path.join(dirs["run"], "audit_metadata.json"),
            cfg,
            parameter_count_full_diffusion=parameter_count_full_diffusion,
            parameter_count_core_no_diff=parameter_count_core_no_diff,
            training_time_seconds=training_time_seconds,
            gpu_memory_peak_bytes=gpu_memory_peak_bytes,
        )
    if best_ckpt_step is None:
        raise RuntimeError("Training ended before any validation-selected checkpoint was saved.")
    logger.info(f"Best Dice Score: {best_dice_score:.4f}")


def main():

    args = parse_args()
    cfg = OmegaConf.load(args.config)
    apply_runtime_overrides(cfg, args)
    audit_mode = normalize_audit_mode(cfg)
    cfg.audit_mode = audit_mode
    if cfg.seed is not None:
        set_seed(int(cfg.seed), device_specific=False)
    save_visualization = bool(cfg.get("save_visualization", False))

    # Create output directories
    dirs = setup_directories(cfg.output_dir, cfg.job_name, save_visualization=save_visualization)

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with="tensorboard",
        project_config=ProjectConfiguration(project_dir=dirs['run'], logging_dir=dirs['tb'])
    )
    require_rtx5090(accelerator.device)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if accelerator.is_main_process:
        for path in dirs.values():
            os.makedirs(path, exist_ok=False)
        OmegaConf.save(cfg, os.path.join(dirs["run"], "resolved_config.yaml"))
        
        accelerator.init_trackers("tensorboard")

    # Logging configurations
    log_file = os.path.join(dirs['run'], "logging.log")
    log_format = logging.Formatter('%(asctime)s - %(levelname)s -%(filename)s - %(funcName)s >> %(message)s')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_format)
    file_handler.setLevel(logging.INFO)
    logger.logger.addHandler(file_handler)

    logger.info(cfg, main_process_only=True)
    logger.info(accelerator.state, main_process_only=False)
    logger.info(f"audit_mode={audit_mode}", main_process_only=True)
    if accelerator.is_local_main_process:
        diffusers.utils.logging.set_verbosity_info()
    else:
        diffusers.utils.logging.set_verbosity_error()

    # -------------------- Device --------------------
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if audit_mode == "core_no_diff":
        train_core_no_diff(cfg, dirs, accelerator)
        return

    # -------------------- Model --------------------
    unet = UNetModel.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="unet", local_files_only=True
    )
    replace_unet_input(unet)
    unet_wrapper = UNetModelWrapper(unet, cfg.use_alignment)

    vae: AutoencoderKL = AutoencoderKL.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="vae", local_files_only=True
    )

    latent_encoder = LatentEncoder()
    latent_encoder.init_from_pretrained(vae)

    noise_scheduler: DDIMScheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0015,
        beta_end=0.0155,
        prediction_type=cfg.prediction_type,
        clip_sample=False,
    )
    if cfg.use_alignment:
        vision_encoder = load_dinov2(model_name=cfg.vision_encoder_model)
        vision_encoder.requires_grad_(False)
    
    vae.requires_grad_(False)
    latent_encoder.requires_grad_(True)
    unet_wrapper.train()
    latent_encoder.train()

    # Create EMA for the unet.
    if cfg.use_ema:
        ema_unet = deepcopy(unet)
        ema_unet: EMAModel = EMAModel(
            ema_unet.parameters(),
            model_cls=UNetModel,
            model_config=ema_unet.config,
            foreach=True,
        )

    # For mixed precision training, cast non-trainable weights to half-precision
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    if cfg.use_alignment:
        vision_encoder.to(accelerator.device, dtype=weight_dtype)

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            if cfg.use_ema:
                ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

            for i, model in enumerate(models):
                if isinstance(model, UNetModelWrapper):
                    model.unet.save_pretrained(os.path.join(output_dir, "unet"))
                elif isinstance(model, LatentEncoder):
                    model.save_pretrained(os.path.join(output_dir, "latent_encoder"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

    accelerator.register_save_state_pre_hook(save_model_hook)

    # Optimizer and learning rate scheduler
    def model_parameters():
        return list(unet_wrapper.parameters()) + list(latent_encoder.parameters())
    parameter_count = sum(parameter.numel() for parameter in model_parameters())
    if accelerator.is_main_process:
        write_audit_metadata(
            os.path.join(dirs["run"], "audit_metadata.json"),
            cfg,
            audit_mode,
            parameter_count,
        )
        if audit_mode == "train_shuffle_yt":
            logger.info(
                "train_shuffle_yt is active: only the training-time noisy mask latent input is shuffled; "
                "image, timestep, noise, target, loss, validation, inference, and evaluation are unchanged."
            )
        elif audit_mode == "train_random_yt":
            logger.info(
                "train_random_yt is active: original noisy mask latent Y_t_ref is constructed normally, "
                "then only the training-time Y_t input passed to the U-Net is replaced by independent "
                "Gaussian noise. Image, timestep, original noise target, prediction target, loss, "
                "alignment, validation, inference, and evaluation are unchanged."
            )
    optimizer = optim.AdamW(model_parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    lr_scheduler = get_scheduler(
        name="constant_with_warmup",
        optimizer=optimizer,
        num_warmup_steps=cfg.num_warmup_steps * accelerator.num_processes,
        num_training_steps=cfg.max_train_steps * accelerator.num_processes
    )

    # -------------------- Data --------------------
    logger.info(f"Loading Dataset {cfg.dataset_name}...")
    train_dataset, val_dataset = load_custom_dataset(
        base_data_dir=cfg.base_data_dir,
        dataset_name=cfg.dataset_name,
        resolution=cfg.resolution,
        seed=cfg.seed,
        eval_split=cfg.validation_split,
        num_seg_classes=int(cfg.get("num_seg_classes", 2)),
    )

    imagenet_transforms = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    train_dataloader = DataLoader(
        train_dataset,
        cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )
    val_dataloader = DataLoader(
        val_dataset,
        cfg.test_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )

    # Prepare everything with our `accelerator`.
    unet_wrapper, latent_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet_wrapper, latent_encoder, optimizer, train_dataloader, lr_scheduler
    )

    if cfg.use_ema:
        ema_unet.to(accelerator.device)

    # -------------------- Training --------------------
    logger.info("Start Training...")
    global_seed_sequence = generate_seed_sequence(
        initial_seed=cfg.seed,
        length=cfg.max_train_steps,
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / accelerator.gradient_accumulation_steps
    )
    max_epoch = math.ceil(cfg.max_train_steps / num_update_steps_per_epoch)

    global_step = 0
    progress_bar = tqdm(
        range(0, cfg.max_train_steps),
        initial=0,
        desc="Training Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    scaling_factor = 0.18215
    best_dice_score = float("-inf")
    best_ckpt_step = None
    validation_csv = getattr(
        cfg,
        "validation_csv",
        os.path.join(dirs["run"], "validation_metrics.csv"),
    )

    for epoch in range(max_epoch):
        unet_wrapper.train()
        latent_encoder.train()
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):

            with accelerator.accumulate(unet_wrapper):

                # globally consistent random generators
                if cfg.seed is not None:
                    local_seed = global_seed_sequence.pop()
                    rand_num_generator = torch.Generator(device=accelerator.device)
                    rand_num_generator.manual_seed(local_seed)
                else:
                    rand_num_generator = None

                rgb: torch.Tensor = batch["pixel_values"]
                mask: torch.Tensor = batch["mask_values"]

                rgb_norm = rgb * 2.0 - 1.0 # [0, 1] -> [-1, 1]
                mask_norm = mask * 2.0 - 1.0

                # accelerator format code
                rgb_latent = latent_encoder(rgb_norm).mode() # [B, 4, h, w]
                rgb_latent = rgb_latent * scaling_factor

                with torch.no_grad():
                    gt_mask_latent = vae.encode(mask_norm.to(weight_dtype)).mode()
                    gt_mask_latent = gt_mask_latent * scaling_factor

                batch_size = rgb.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    1000,
                    (batch_size,),
                    device=accelerator.device,
                    generator=rand_num_generator,
                ).long()  # [B]

                # Sample noise
                noise = torch.randn(
                    gt_mask_latent.shape,
                    device=accelerator.device,
                    generator=rand_num_generator,
                )  # [B, 4, h, w]

                # Add noise to the latents (diffusion forward process)
                noisy_latents_ref = noise_scheduler.add_noise(
                    gt_mask_latent, noise, timesteps
                )  # [B, 4, h, w]
                noisy_latents_input = noisy_latents_ref

                if audit_mode == "train_shuffle_yt":
                    shuffle_permutation = derangement_like_permutation(
                        batch_size,
                        device=accelerator.device,
                        generator=rand_num_generator,
                    )
                    noisy_latents_input = noise_scheduler.add_noise(
                        gt_mask_latent[shuffle_permutation], noise, timesteps
                    )  # [B, 4, h, w]
                    if (
                        noisy_latents_input.shape != noisy_latents_ref.shape
                        or noisy_latents_input.dtype != noisy_latents_ref.dtype
                        or noisy_latents_input.device != noisy_latents_ref.device
                    ):
                        raise RuntimeError(
                            "train_shuffle_yt produced a Y_t input with shape, dtype, or device "
                            "different from the original Y_t reference."
                        )
                    if global_step == 0 and step == 0 and accelerator.is_main_process:
                        logger.info(
                            "train_shuffle_yt debug: first batch permutation=%s, "
                            "Y_t_input shape=%s dtype=%s device=%s",
                            shuffle_permutation.detach().cpu().tolist(),
                            tuple(noisy_latents_input.shape),
                            noisy_latents_input.dtype,
                            noisy_latents_input.device,
                        )

                if audit_mode == "train_random_yt":
                    noisy_latents_input = random_yt_like(
                        noisy_latents_ref,
                        generator=rand_num_generator,
                    )
                    if global_step == 0 and step == 0 and accelerator.is_main_process:
                        logger.info(
                            "train_random_yt debug: replaced only Y_t input with independent "
                            "Gaussian noise; Y_t_input shape=%s dtype=%s device=%s",
                            tuple(noisy_latents_input.shape),
                            noisy_latents_input.dtype,
                            noisy_latents_input.device,
                        )

                # Concat rgb and depth latents
                cat_latents = torch.cat(
                    [rgb_latent, noisy_latents_input], dim=1
                )  # [B, 8, h, w]
                cat_latents = cat_latents.float()

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "sample":
                    target = gt_mask_latent
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(gt_mask_latent, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
    
                # Predict the noise residual and compute loss
                model_pred, z_tilde = unet_wrapper(cat_latents, timesteps)  # [B, 4, h, w]

                loss = F.l1_loss(model_pred.float(), target.float())

                if cfg.use_alignment:
                    with torch.no_grad():
                        rgb_for_dino = F.interpolate(rgb, 224 * (cfg.resolution // 256), mode="bicubic")
                        rgb_for_dino = imagenet_transforms(rgb_for_dino)
                        z: torch.Tensor = vision_encoder.forward_features(rgb_for_dino.to(weight_dtype))["x_norm_patchtokens"]
                    distill_loss = cosine_loss(z.float(), z_tilde.float())
                    loss += cfg.lam * distill_loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(cfg.train_batch_size)).mean()
                train_loss += avg_loss.item() / cfg.gradient_accumulation_steps
            
                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model_parameters(), 1.0)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if cfg.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                train_loss = 0.0

                if global_step >= cfg.validation_start_steps:
                    
                    if accelerator.is_main_process:
                        if global_step % cfg.validation_steps == 0:
                            if cfg.use_ema:
                                ema_unet.store(unet.parameters())
                                ema_unet.copy_to(unet.parameters())
                            
                            metrics = log_validation(
                                vae,
                                latent_encoder,
                                unet,
                                noise_scheduler,
                                val_dataloader,
                                cfg,
                                accelerator,
                                dirs.get('vis'),
                                weight_dtype,
                                global_step
                            )
                            append_validation_csv(validation_csv, global_step, metrics)
                            if cfg.use_ema:
                                ema_unet.restore(unet.parameters())
                            
                            if best_dice_score < metrics["dice"]:
                                last_best_ckpt_path = (
                                    os.path.join(dirs["ckpt"], f"step-{best_ckpt_step}")
                                    if best_ckpt_step is not None
                                    else None
                                )
                                best_dice_score = metrics["dice"]
                                best_ckpt_step = global_step

                                if accelerator.is_main_process:
                                    if last_best_ckpt_path and os.path.exists(last_best_ckpt_path):
                                        shutil.rmtree(last_best_ckpt_path)
                                    save_path = os.path.join(dirs["ckpt"], f"step-{global_step}")
                                    accelerator.save_state(save_path)
                                    write_best_checkpoint_metadata(
                                        dirs["run"], save_path, cfg, audit_mode,
                                        global_step, best_dice_score,
                                    )
                                    logger.info(f"Best Dice Score at step {global_step}: {best_dice_score:.4f}")
                                    logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= cfg.max_train_steps:
                break

    # Save the final UNet checkpoint
    accelerator.wait_for_everyone()
    accelerator.end_training()

    if best_ckpt_step is None:
        raise RuntimeError("Training ended before any validation-selected checkpoint was saved.")
    logger.info(f"Best Dice Score: {best_dice_score:.4f}")

if __name__ == "__main__":
    main()
