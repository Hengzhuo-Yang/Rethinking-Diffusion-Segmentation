"""Manifest-backed RTX 5090 smoke tests for all LEAF release conditions."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import re
import shutil
import sys
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from diffusers import DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from gpu_preflight import EXPECTED_GPU_NAME, run_preflight
from leaf import (
    AutoencoderKL,
    CoreNoDiffSegmentor,
    LatentEncoder,
    LeafPipeline,
    UNetModel,
    UNetModelWrapper,
)
from src.data import load_custom_dataset
from src.util.loss import cosine_loss
from src.util.metric import SegmentationMetric
from train import (
    collate_fn,
    derangement_like_permutation,
    load_dinov2,
    normalize_audit_mode,
    random_yt_like,
    replace_unet_input,
)


DATASETS: dict[str, dict[str, Any]] = {
    "btcv": {
        "config_stem": "btcv",
        "loader_name": "BTCV",
        "manifest_dir": "btcv",
        "num_seg_classes": 2,
    },
    "acdc": {
        "config_stem": "acdc",
        "loader_name": "ACDC",
        "manifest_dir": "acdc",
        "num_seg_classes": 4,
    },
    "isic2018": {
        "config_stem": "isic2018",
        "loader_name": "ISIC18",
        "manifest_dir": "isic2018",
        "num_seg_classes": 2,
    },
}

CONDITIONS: dict[str, dict[str, str]] = {
    "full": {"config_suffix": "", "audit_mode": "none"},
    "random-yt": {
        "config_suffix": "-train-random-yt",
        "audit_mode": "train_random_yt",
    },
    "shuffle-yt": {
        "config_suffix": "-train-shuffle-yt",
        "audit_mode": "train_shuffle_yt",
    },
    "core-no-diff": {
        "config_suffix": "-core-no-diff",
        "audit_mode": "core_no_diff",
    },
}


def _read_manifest(path: Path, dataset_key: str) -> tuple[list[str], str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing fixed split manifest: {path}")
    raw = path.read_bytes()
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"Manifest must be UTF-8: {path}") from exc
    identifiers = [line.strip() for line in lines if line.strip()]
    if not identifiers:
        raise ValueError(f"Manifest is empty: {path}")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"Manifest contains duplicate IDs: {path}")
    patterns = {
        "btcv": re.compile(r"case\d{4}"),
        "acdc": re.compile(r"patient\d{3}"),
        "isic2018": re.compile(r"ISIC_\d{7}"),
    }
    invalid = [identifier for identifier in identifiers if patterns[dataset_key].fullmatch(identifier) is None]
    if invalid:
        raise ValueError(
            f"Manifest {path} has invalid {dataset_key} IDs; first invalid ID: {invalid[0]!r}"
        )
    return identifiers, hashlib.sha256(raw).hexdigest()


def _canonical_id(dataset_key: str, image_path: str) -> str:
    stem = Path(image_path).stem
    if dataset_key == "btcv":
        match = re.search(r"(?:case|vol)(\d{1,4})(?:_|$)", stem, flags=re.IGNORECASE)
        if match is None:
            raise ValueError(f"Cannot map BTCV sample filename to case ID: {stem!r}")
        return f"case{int(match.group(1)):04d}"
    if dataset_key == "acdc":
        match = re.search(r"patient\d{3}", stem, flags=re.IGNORECASE)
        if match is None:
            raise ValueError(f"Cannot map ACDC sample filename to patient ID: {stem!r}")
        return match.group(0).lower()
    if dataset_key == "isic2018":
        match = re.search(r"ISIC_\d{7}", stem, flags=re.IGNORECASE)
        if match is None:
            raise ValueError(f"Cannot map ISIC2018 sample filename to image ID: {stem!r}")
        return match.group(0).upper()
    raise ValueError(f"Unsupported dataset key: {dataset_key}")


class ManifestDataset(Dataset):
    """Filter a real LEAF dataset using an ID-only fixed split manifest."""

    def __init__(
        self,
        base_dataset: Dataset,
        *,
        dataset_key: str,
        logical_split: str,
        manifest_path: Path,
    ) -> None:
        self.base_dataset = base_dataset
        self.dataset_key = dataset_key
        self.logical_split = logical_split
        self.manifest_path = manifest_path
        self.manifest_ids, self.manifest_sha256 = _read_manifest(manifest_path, dataset_key)
        manifest_set = set(self.manifest_ids)
        image_filenames = getattr(base_dataset, "image_filenames", None)
        if image_filenames is None or len(image_filenames) != len(base_dataset):
            raise TypeError(
                f"{type(base_dataset).__name__} must expose one image_filenames entry per sample"
            )

        selected_indices: list[int] = []
        selected_ids: set[str] = set()
        excluded_sample_count = 0
        for index, image_path in enumerate(image_filenames):
            identifier = _canonical_id(dataset_key, image_path)
            if identifier in manifest_set:
                selected_indices.append(index)
                selected_ids.add(identifier)
            else:
                excluded_sample_count += 1

        missing = sorted(manifest_set - selected_ids)
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} IDs from {manifest_path} are absent from the {logical_split} "
                f"data pool; first missing ID: {missing[0]}"
            )
        if not selected_indices:
            raise RuntimeError(f"Manifest selected no samples for {dataset_key}/{logical_split}")
        self.indices = selected_indices
        self.selected_ids = selected_ids
        self.excluded_sample_count = excluded_sample_count

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.base_dataset[self.indices[index]]

    def evidence(self) -> dict[str, Any]:
        try:
            display_path = self.manifest_path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
        except ValueError:
            display_path = f"<external-manifest>/{self.manifest_path.name}"
        return {
            "logical_split": self.logical_split,
            "manifest": display_path,
            "manifest_sha256": self.manifest_sha256,
            "manifest_id_count": len(self.manifest_ids),
            "selected_sample_count": len(self.indices),
            "excluded_pool_sample_count": self.excluded_sample_count,
        }


def _normalize_data_root(path: Path, loader_name: str) -> Path:
    root = path.expanduser().resolve()
    if (root / loader_name).is_dir():
        return root
    if root.name.lower() == loader_name.lower() and root.is_dir():
        return root.parent
    return root


def _base_datasets(
    dataset_key: str,
    data_root: Path,
    *,
    resolution: int,
    seed: int,
    num_seg_classes: int,
) -> tuple[Dataset, Dataset, Dataset, dict[str, str]]:
    spec = DATASETS[dataset_key]
    loader_name = str(spec["loader_name"])
    base_root = _normalize_data_root(data_root, loader_name)
    common = {
        "base_data_dir": str(base_root),
        "dataset_name": loader_name,
        "resolution": resolution,
        "seed": seed,
        "num_seg_classes": num_seg_classes,
    }
    train_dataset, val_dataset = load_custom_dataset(**common, eval_split="val")
    _, test_dataset = load_custom_dataset(**common, eval_split="test")
    return (
        train_dataset,
        val_dataset,
        test_dataset,
        {"train": "train", "val": "validation", "test": "test"},
    )


def _manifest_datasets(
    dataset_key: str,
    data_root: Path,
    manifest_root: Path,
    *,
    resolution: int,
    seed: int,
    num_seg_classes: int,
) -> tuple[dict[str, ManifestDataset], dict[str, str]]:
    manifest_dir = manifest_root / str(DATASETS[dataset_key]["manifest_dir"])
    manifest_paths = {split: manifest_dir / f"{split}.txt" for split in ("train", "val", "test")}
    manifest_id_sets: dict[str, set[str]] = {}
    for split, path in manifest_paths.items():
        identifiers, _ = _read_manifest(path, dataset_key)
        manifest_id_sets[split] = set(identifiers)
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sorted(manifest_id_sets[left] & manifest_id_sets[right])
        if overlap:
            raise ValueError(
                f"Fixed manifests leak between {left} and {right}; first overlapping ID: {overlap[0]}"
            )

    train_base, val_base, test_base, physical_sources = _base_datasets(
        dataset_key,
        data_root,
        resolution=resolution,
        seed=seed,
        num_seg_classes=num_seg_classes,
    )
    bases = {"train": train_base, "val": val_base, "test": test_base}
    datasets = {
        split: ManifestDataset(
            bases[split],
            dataset_key=dataset_key,
            logical_split=split,
            manifest_path=manifest_paths[split],
        )
        for split in ("train", "val", "test")
    }
    return datasets, physical_sources


def _first_batch(
    dataset: Dataset,
    *,
    batch_size: int,
    train: bool,
    seed: int,
) -> dict[str, torch.Tensor]:
    if len(dataset) < batch_size:
        raise RuntimeError(
            f"Dataset has {len(dataset)} samples but the smoke batch requires {batch_size}"
        )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=train,
        generator=generator if train else None,
    )
    return next(iter(loader))


def _parameter_devices(modules: Iterable[nn.Module]) -> list[str]:
    devices = {
        str(parameter.device)
        for module in modules
        for parameter in module.parameters()
    }
    return sorted(devices)


def _assert_cuda_tensor(tensor: torch.Tensor, device: torch.device, name: str) -> None:
    if tensor.device != device:
        raise RuntimeError(f"{name} is on {tensor.device}, expected {device}; CPU fallback is forbidden")


def _assert_cuda_modules(modules: Iterable[nn.Module], device: torch.device, label: str) -> None:
    devices = _parameter_devices(modules)
    if devices != [str(device)]:
        raise RuntimeError(f"{label} parameter devices are {devices}, expected only {device}")


def _autocast(dtype: torch.dtype | None):
    if dtype is None:
        return torch.autocast("cuda")
    return torch.autocast("cuda", dtype=dtype)


def _dtype_from_config(cfg: Any) -> torch.dtype:
    mixed_precision = str(cfg.mixed_precision).lower()
    if mixed_precision == "bf16":
        return torch.bfloat16
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision in {"no", "none", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported mixed_precision={cfg.mixed_precision!r}")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_local_dinov2(repo: Path, model_name: str, resolution: int) -> nn.Module:
    import timm

    encoder = torch.hub.load(str(repo), model_name, source="local")
    del encoder.head
    patch_resolution = 16 * (resolution // 256)
    encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
        encoder.pos_embed.data,
        [patch_resolution, patch_resolution],
    )
    encoder.head = nn.Identity()
    encoder.eval()
    return encoder


class DinoCache:
    def __init__(self, local_repo: Path | None) -> None:
        self.local_repo = local_repo.expanduser().resolve() if local_repo is not None else None
        self.models: dict[tuple[str, int, str], nn.Module] = {}

    def get(
        self,
        model_name: str,
        resolution: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> nn.Module:
        key = (model_name, resolution, str(dtype))
        if key not in self.models:
            if self.local_repo is None:
                model = load_dinov2(model_name=model_name, resolution=resolution)
            else:
                if not self.local_repo.is_dir():
                    raise FileNotFoundError(f"Missing local DINOv2 repository: {self.local_repo}")
                model = _load_local_dinov2(self.local_repo, model_name, resolution)
            model.requires_grad_(False)
            model.eval()
            model.to(device, dtype=dtype)
            _assert_cuda_modules([model], device, "DINOv2")
            self.models[key] = model
        return self.models[key]

    def clear(self) -> None:
        self.models.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _scheduler(prediction_type: str) -> DDIMScheduler:
    return DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0015,
        beta_end=0.0155,
        prediction_type=prediction_type,
        clip_sample=False,
    )



@torch.no_grad()
def _full_inference(
    *,
    vae: AutoencoderKL,
    unet: UNetModel,
    latent_encoder: LatentEncoder,
    scheduler: DDIMScheduler,
    image: torch.Tensor,
    device: torch.device,
    seed: int,
    autocast_dtype: torch.dtype | None,
) -> tuple[torch.Tensor, list[int]]:
    vae.eval()
    unet.eval()
    latent_encoder.eval()
    pipeline = LeafPipeline(
        vae=vae,
        unet=unet,
        latent_encoder=latent_encoder,
        scheduler=scheduler,
    ).to(device)
    pipeline.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    with _autocast(autocast_dtype):
        output = pipeline(
            image,
            num_inference_steps=1,
            generator=generator,
            show_progress_bar=False,
        )
    prediction = output.mask_pred
    schedule = list(output.actual_timestep_schedule or [])
    del pipeline, output
    return prediction, schedule


@torch.no_grad()
def _core_inference(
    *,
    vae: AutoencoderKL,
    model: CoreNoDiffSegmentor,
    image: torch.Tensor,
    autocast_dtype: torch.dtype,
) -> torch.Tensor:
    vae.eval()
    model.eval()
    with _autocast(autocast_dtype):
        mask_latent, _ = model(image)
        prediction = vae.decode(mask_latent / model.scaling_factor)
        prediction = torch.clamp(prediction, -1.0, 1.0)
        prediction = (prediction + 1.0) / 2.0
    return prediction


def _optimizer_gradient_device(parameters: Iterable[nn.Parameter]) -> str:
    for parameter in parameters:
        if parameter.grad is not None:
            return str(parameter.grad.device)
    raise RuntimeError("Backward completed without producing any optimizer parameter gradient")


def _checkpoint_reload_full(
    *,
    path: Path,
    unet: UNetModel,
    latent_encoder: LatentEncoder,
    audit_mode: str,
    validation_dice: float,
) -> dict[str, Any]:
    reference_parameter = next(unet.parameters())
    reference_values = reference_parameter.detach().reshape(-1)[:64].clone()
    torch.save(
        {
            "kind": "full_diffusion",
            "audit_mode": audit_mode,
            "selection_metric": "dice",
            "selection_value": float(validation_dice),
            "unet": unet.state_dict(),
            "latent_encoder": latent_encoder.state_dict(),
        },
        path,
    )
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError("Temporary best checkpoint was not written")
    with torch.no_grad():
        reference_parameter.add_(1.0)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if loaded.get("selection_metric") != "dice":
        raise RuntimeError("Temporary checkpoint did not preserve validation selection metadata")
    unet.load_state_dict(loaded["unet"], strict=True)
    latent_encoder.load_state_dict(loaded["latent_encoder"], strict=True)
    del loaded
    restored_values = next(unet.parameters()).detach().reshape(-1)[:64]
    if not torch.equal(reference_values, restored_values):
        raise RuntimeError("Temporary best checkpoint reload did not restore U-Net parameters exactly")
    return {
        "saved": True,
        "loaded": True,
        "selection_metric": "dice",
        "selection_value": float(validation_dice),
        "temporary_path": "<temporary>/best.pt",
        "inference_components": ["unet", "latent_encoder"],
    }


def _checkpoint_reload_core(
    *,
    path: Path,
    model: CoreNoDiffSegmentor,
    validation_dice: float,
) -> dict[str, Any]:
    reference_parameter = next(model.parameters())
    reference_values = reference_parameter.detach().reshape(-1)[:64].clone()
    torch.save(
        {
            "kind": "core_no_diff",
            "audit_mode": "core_no_diff",
            "selection_metric": "dice",
            "selection_value": float(validation_dice),
            "core_no_diff_segmentor": model.state_dict(),
        },
        path,
    )
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError("Temporary best checkpoint was not written")
    with torch.no_grad():
        reference_parameter.add_(1.0)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if loaded.get("selection_metric") != "dice":
        raise RuntimeError("Temporary checkpoint did not preserve validation selection metadata")
    model.load_state_dict(loaded["core_no_diff_segmentor"], strict=True)
    del loaded
    restored_values = next(model.parameters()).detach().reshape(-1)[:64]
    if not torch.equal(reference_values, restored_values):
        raise RuntimeError("Temporary best checkpoint reload did not restore core model parameters exactly")
    return {
        "saved": True,
        "loaded": True,
        "selection_metric": "dice",
        "selection_value": float(validation_dice),
        "temporary_path": "<temporary>/best.pt",
        "inference_components": ["core_no_diff_segmentor"],
    }


def _execute_full(
    *,
    cfg: Any,
    audit_mode: str,
    batches: dict[str, dict[str, torch.Tensor]],
    assets_root: Path,
    temp_dir: Path,
    device: torch.device,
    weight_dtype: torch.dtype,
    dino_cache: DinoCache,
    backend_defaults: dict[str, bool],
    result: dict[str, Any],
) -> dict[str, Any]:
    result["stage"] = "model_initialization"
    vae = AutoencoderKL.from_pretrained(
        str(assets_root), subfolder="vae", local_files_only=True
    )
    unet = UNetModel.from_pretrained(
        str(assets_root), subfolder="unet", local_files_only=True
    )
    replace_unet_input(unet, log_update=False)
    unet_wrapper = UNetModelWrapper(unet, bool(cfg.use_alignment))
    latent_encoder = LatentEncoder()
    latent_encoder.init_from_pretrained(vae)

    vae.requires_grad_(False)
    vae.to(device, dtype=weight_dtype)
    unet_wrapper.to(device)
    latent_encoder.to(device)
    trainable_parameters = list(unet_wrapper.parameters()) + list(latent_encoder.parameters())
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
    )
    lr_scheduler = get_scheduler(
        name="constant_with_warmup",
        optimizer=optimizer,
        num_warmup_steps=int(cfg.num_warmup_steps),
        num_training_steps=int(cfg.max_train_steps),
    )
    ema_unet = EMAModel(unet.parameters(), foreach=True) if bool(cfg.use_ema) else None
    if ema_unet is not None:
        ema_unet.to(device)
    vision_encoder = dino_cache.get(
        str(cfg.vision_encoder_model),
        int(cfg.resolution),
        device,
        weight_dtype,
    )
    model_modules = [vae, unet_wrapper, latent_encoder, vision_encoder]
    _assert_cuda_modules(model_modules, device, "LEAF full-diffusion smoke model")

    result["stage"] = "train_forward_loss"
    unet_wrapper.train()
    latent_encoder.train()
    rgb = batches["train"]["pixel_values"].to(device, non_blocking=True)
    mask = batches["train"]["mask_values"].to(device, non_blocking=True)
    _assert_cuda_tensor(rgb, device, "training image")
    _assert_cuda_tensor(mask, device, "training mask")
    generator = torch.Generator(device=device)
    generator.manual_seed(int(cfg.seed))
    imagenet_normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    noise_scheduler = _scheduler(str(cfg.prediction_type))
    scaling_factor = 0.18215
    permutation: list[int] | None = None

    with _autocast(weight_dtype):
        rgb_norm = rgb * 2.0 - 1.0
        mask_norm = mask * 2.0 - 1.0
        rgb_latent = latent_encoder(rgb_norm).mode() * scaling_factor
        with torch.no_grad():
            gt_mask_latent = vae.encode(mask_norm.to(weight_dtype)).mode() * scaling_factor
        batch_size = rgb.shape[0]
        timesteps = torch.randint(
            0,
            1000,
            (batch_size,),
            device=device,
            generator=generator,
        ).long()
        noise = torch.randn(gt_mask_latent.shape, device=device, generator=generator)
        noisy_latents_ref = noise_scheduler.add_noise(gt_mask_latent, noise, timesteps)
        noisy_latents_input = noisy_latents_ref

        if audit_mode == "train_shuffle_yt":
            shuffle = derangement_like_permutation(
                batch_size,
                device=device,
                generator=generator,
            )
            noisy_latents_input = noise_scheduler.add_noise(
                gt_mask_latent[shuffle],
                noise,
                timesteps,
            )
            permutation = shuffle.detach().cpu().tolist()
        elif audit_mode == "train_random_yt":
            noisy_latents_input = random_yt_like(
                noisy_latents_ref,
                generator=generator,
            )
        elif audit_mode != "none":
            raise ValueError(f"Unexpected full-diffusion audit mode: {audit_mode}")

        cat_latents = torch.cat([rgb_latent, noisy_latents_input], dim=1).float()
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "sample":
            target = gt_mask_latent
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(gt_mask_latent, noise, timesteps)
        else:
            raise ValueError(
                f"Unknown prediction type {noise_scheduler.config.prediction_type}"
            )
        model_prediction, alignment_prediction = unet_wrapper(cat_latents, timesteps)
        l1_loss = F.l1_loss(model_prediction.float(), target.float())
        if alignment_prediction is None:
            raise RuntimeError("use_alignment=True but the U-Net wrapper returned no alignment tokens")
        with torch.no_grad():
            dino_image = F.interpolate(
                rgb,
                224 * (int(cfg.resolution) // 256),
                mode="bicubic",
            )
            dino_image = imagenet_normalize(dino_image)
            dino_tokens = vision_encoder.forward_features(
                dino_image.to(weight_dtype)
            )["x_norm_patchtokens"]
        alignment_loss = cosine_loss(
            dino_tokens.float(), alignment_prediction.float()
        ).mean()
        loss = l1_loss + float(cfg.lam) * alignment_loss

    _assert_cuda_tensor(model_prediction, device, "training model output")
    _assert_cuda_tensor(target, device, "training target")
    _assert_cuda_tensor(loss, device, "training loss")
    if audit_mode == "none" and noisy_latents_input.data_ptr() != noisy_latents_ref.data_ptr():
        raise RuntimeError("Full baseline did not preserve the original Y_t input")
    if audit_mode == "train_random_yt" and torch.equal(noisy_latents_input, noisy_latents_ref):
        raise RuntimeError("Random-Yt branch did not replace Y_t")
    if audit_mode == "train_shuffle_yt":
        if permutation is None or any(index == value for index, value in enumerate(permutation)):
            raise RuntimeError("Shuffle-Yt branch did not execute a no-self-match permutation")

    result["stage"] = "train_backward_optimizer"
    loss.backward()
    gradient_device = _optimizer_gradient_device(trainable_parameters)
    if gradient_device != str(device):
        raise RuntimeError(f"Training gradient is on {gradient_device}, expected {device}")
    torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    if ema_unet is not None:
        ema_unet.step(unet.parameters())
    result["coverage"].update(
        {
            "train_forward": True,
            "loss": True,
            "backward": True,
            "optimizer_step": True,
            "audit_branch": True,
        }
    )

    result["stage"] = "validation_forward_metric"
    if ema_unet is not None:
        ema_unet.store(unet.parameters())
        ema_unet.copy_to(unet.parameters())
    val_image = batches["val"]["pixel_values"].to(device, non_blocking=True)
    val_mask = batches["val"]["mask_values"].to(device, non_blocking=True)
    _assert_cuda_tensor(val_image, device, "validation image")
    val_prediction, val_schedule = _full_inference(
        vae=vae,
        unet=unet,
        latent_encoder=latent_encoder,
        scheduler=noise_scheduler,
        image=val_image,
        device=device,
        seed=int(cfg.seed),
        autocast_dtype=None,
    )
    _assert_cuda_tensor(val_prediction, device, "validation output")
    validation_metrics, validation_counts = _metric(
        val_prediction, val_mask, cfg, device
    )
    if "dice" not in validation_metrics or not np.isfinite(validation_metrics["dice"]):
        raise RuntimeError("Validation did not produce a finite mean Dice selection metric")
    result["coverage"].update(
        {"validation_forward": True, "validation_metric": True}
    )

    result["stage"] = "best_checkpoint_save_load"
    checkpoint = _checkpoint_reload_full(
        path=temp_dir / "best.pt",
        unet=unet,
        latent_encoder=latent_encoder,
        audit_mode=audit_mode,
        validation_dice=validation_metrics["dice"],
    )
    if ema_unet is not None:
        ema_unet.restore(unet.parameters())
        loaded = torch.load(temp_dir / "best.pt", map_location="cpu", weights_only=True)
        unet.load_state_dict(loaded["unet"], strict=True)
        latent_encoder.load_state_dict(loaded["latent_encoder"], strict=True)
        del loaded
    result["coverage"].update(
        {"checkpoint_saved": True, "checkpoint_loaded": True}
    )

    result["stage"] = "test_inference"
    torch.backends.cuda.matmul.allow_tf32 = backend_defaults["matmul_tf32"]
    torch.backends.cudnn.allow_tf32 = backend_defaults["cudnn_tf32"]
    test_image = batches["test"]["pixel_values"].to(device, non_blocking=True)
    test_mask = batches["test"]["mask_values"].to(device, non_blocking=True)
    _assert_cuda_tensor(test_image, device, "test image")
    test_prediction, test_schedule = _full_inference(
        vae=vae,
        unet=unet,
        latent_encoder=latent_encoder,
        scheduler=noise_scheduler,
        image=test_image,
        device=device,
        seed=int(cfg.seed),
        autocast_dtype=weight_dtype,
    )
    _assert_cuda_tensor(test_prediction, device, "test output")
    test_metrics, test_counts = _metric(test_prediction, test_mask, cfg, device)
    result["coverage"].update(
        {"test_inference": True, "no_cpu_fallback": True}
    )
    torch.cuda.synchronize(device)

    return {
        "devices": {
            "model_parameters": _parameter_devices(model_modules),
            "train_input": str(rgb.device),
            "train_output": str(model_prediction.device),
            "loss": str(loss.device),
            "gradient": gradient_device,
            "validation_input": str(val_image.device),
            "validation_output": str(val_prediction.device),
            "test_input": str(test_image.device),
            "test_output": str(test_prediction.device),
        },
        "train": {
            "loss": float(loss.detach().float().item()),
            "l1_loss": float(l1_loss.detach().float().item()),
            "alignment_loss": float(alignment_loss.detach().float().item()),
            "optimizer": "AdamW",
            "lr_scheduler": "constant_with_warmup",
        },
        "audit": {
            "configured_mode": audit_mode,
            "executed_branch": "full_diffusion" if audit_mode == "none" else audit_mode,
            "shuffle_permutation": permutation,
            "target_changed": False,
            "validation_changed": False,
            "test_inference_changed": False,
        },
        "validation": {
            "metrics": validation_metrics,
            "counts": validation_counts,
            "actual_timestep_schedule": val_schedule,
        },
        "checkpoint": checkpoint,
        "test": {
            "inference_completed": True,
            "smoke_metrics": test_metrics,
            "counts": test_counts,
            "actual_timestep_schedule": test_schedule,
        },
        "ema_used_for_validation_checkpoint": ema_unet is not None,
    }


def _execute_core(
    *,
    cfg: Any,
    batches: dict[str, dict[str, torch.Tensor]],
    assets_root: Path,
    temp_dir: Path,
    device: torch.device,
    weight_dtype: torch.dtype,
    dino_cache: DinoCache,
    backend_defaults: dict[str, bool],
    result: dict[str, Any],
) -> dict[str, Any]:
    result["stage"] = "model_initialization"
    vae = AutoencoderKL.from_pretrained(
        str(assets_root), subfolder="vae", local_files_only=True
    )
    base_unet = UNetModel.from_pretrained(
        str(assets_root), subfolder="unet", local_files_only=True
    )
    model = CoreNoDiffSegmentor.from_pretrained_components(
        vae=vae,
        unet=base_unet,
        use_alignment=bool(cfg.use_alignment),
    )
    del base_unet
    vae.requires_grad_(False)
    vae.to(device, dtype=weight_dtype)
    model.to(device)
    trainable_parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
    )
    lr_scheduler = get_scheduler(
        name="constant_with_warmup",
        optimizer=optimizer,
        num_warmup_steps=int(cfg.num_warmup_steps),
        num_training_steps=int(cfg.max_train_steps),
    )
    ema_model = EMAModel(model.parameters(), foreach=True) if bool(cfg.use_ema) else None
    if ema_model is not None:
        ema_model.to(device)
    vision_encoder = dino_cache.get(
        str(cfg.vision_encoder_model),
        int(cfg.resolution),
        device,
        weight_dtype,
    )
    model_modules = [vae, model, vision_encoder]
    _assert_cuda_modules(model_modules, device, "LEAF core-no-diff smoke model")

    result["stage"] = "train_forward_loss"
    model.train()
    rgb = batches["train"]["pixel_values"].to(device, non_blocking=True)
    mask = batches["train"]["mask_values"].to(device, non_blocking=True)
    _assert_cuda_tensor(rgb, device, "training image")
    _assert_cuda_tensor(mask, device, "training mask")
    imagenet_normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    with _autocast(weight_dtype):
        mask_norm = mask * 2.0 - 1.0
        with torch.no_grad():
            target_latent = vae.encode(mask_norm.to(weight_dtype)).mode()
            target_latent = target_latent * model.scaling_factor
        model_prediction, alignment_prediction = model(rgb)
        if model_prediction.shape != target_latent.shape:
            raise RuntimeError(
                f"Core-No-Diff output shape {tuple(model_prediction.shape)} "
                f"does not match target {tuple(target_latent.shape)}"
            )
        l1_loss = F.l1_loss(model_prediction.float(), target_latent.float())
        if alignment_prediction is None:
            raise RuntimeError("use_alignment=True but Core-No-Diff returned no alignment tokens")
        with torch.no_grad():
            dino_image = F.interpolate(
                rgb,
                224 * (int(cfg.resolution) // 256),
                mode="bicubic",
            )
            dino_image = imagenet_normalize(dino_image)
            dino_tokens = vision_encoder.forward_features(
                dino_image.to(weight_dtype)
            )["x_norm_patchtokens"]
        alignment_loss = cosine_loss(
            dino_tokens.float(), alignment_prediction.float()
        ).mean()
        loss = l1_loss + float(cfg.lam) * alignment_loss

    _assert_cuda_tensor(model_prediction, device, "training model output")
    _assert_cuda_tensor(target_latent, device, "training target")
    _assert_cuda_tensor(loss, device, "training loss")
    result["stage"] = "train_backward_optimizer"
    loss.backward()
    gradient_device = _optimizer_gradient_device(trainable_parameters)
    if gradient_device != str(device):
        raise RuntimeError(f"Training gradient is on {gradient_device}, expected {device}")
    torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    if ema_model is not None:
        ema_model.step(model.parameters())
    result["coverage"].update(
        {
            "train_forward": True,
            "loss": True,
            "backward": True,
            "optimizer_step": True,
            "audit_branch": True,
        }
    )

    result["stage"] = "validation_forward_metric"
    if ema_model is not None:
        ema_model.store(model.parameters())
        ema_model.copy_to(model.parameters())
    val_image = batches["val"]["pixel_values"].to(device, non_blocking=True)
    val_mask = batches["val"]["mask_values"].to(device, non_blocking=True)
    _assert_cuda_tensor(val_image, device, "validation image")
    val_prediction = _core_inference(
        vae=vae,
        model=model,
        image=val_image,
        autocast_dtype=weight_dtype,
    )
    _assert_cuda_tensor(val_prediction, device, "validation output")
    validation_metrics, validation_counts = _metric(
        val_prediction, val_mask, cfg, device
    )
    if "dice" not in validation_metrics or not np.isfinite(validation_metrics["dice"]):
        raise RuntimeError("Validation did not produce a finite mean Dice selection metric")
    result["coverage"].update(
        {"validation_forward": True, "validation_metric": True}
    )

    result["stage"] = "best_checkpoint_save_load"
    checkpoint = _checkpoint_reload_core(
        path=temp_dir / "best.pt",
        model=model,
        validation_dice=validation_metrics["dice"],
    )
    if ema_model is not None:
        ema_model.restore(model.parameters())
        loaded = torch.load(temp_dir / "best.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(loaded["core_no_diff_segmentor"], strict=True)
        del loaded
    result["coverage"].update(
        {"checkpoint_saved": True, "checkpoint_loaded": True}
    )

    result["stage"] = "test_inference"
    torch.backends.cuda.matmul.allow_tf32 = backend_defaults["matmul_tf32"]
    torch.backends.cudnn.allow_tf32 = backend_defaults["cudnn_tf32"]
    test_image = batches["test"]["pixel_values"].to(device, non_blocking=True)
    test_mask = batches["test"]["mask_values"].to(device, non_blocking=True)
    _assert_cuda_tensor(test_image, device, "test image")
    test_prediction = _core_inference(
        vae=vae,
        model=model,
        image=test_image,
        autocast_dtype=weight_dtype,
    )
    _assert_cuda_tensor(test_prediction, device, "test output")
    test_metrics, test_counts = _metric(test_prediction, test_mask, cfg, device)
    result["coverage"].update(
        {"test_inference": True, "no_cpu_fallback": True}
    )
    torch.cuda.synchronize(device)

    return {
        "devices": {
            "model_parameters": _parameter_devices(model_modules),
            "train_input": str(rgb.device),
            "train_output": str(model_prediction.device),
            "loss": str(loss.device),
            "gradient": gradient_device,
            "validation_input": str(val_image.device),
            "validation_output": str(val_prediction.device),
            "test_input": str(test_image.device),
            "test_output": str(test_prediction.device),
        },
        "train": {
            "loss": float(loss.detach().float().item()),
            "l1_loss": float(l1_loss.detach().float().item()),
            "alignment_loss": float(alignment_loss.detach().float().item()),
            "optimizer": "AdamW",
            "lr_scheduler": "constant_with_warmup",
        },
        "audit": {
            "configured_mode": "core_no_diff",
            "executed_branch": "core_no_diff",
            "uses_y_t": False,
            "uses_timestep": False,
            "uses_reverse_sampler": False,
            "target": "clean mask latent",
        },
        "validation": {
            "metrics": validation_metrics,
            "counts": validation_counts,
            "actual_timestep_schedule": [],
        },
        "checkpoint": checkpoint,
        "test": {
            "inference_completed": True,
            "smoke_metrics": test_metrics,
            "counts": test_counts,
            "actual_timestep_schedule": [],
        },
        "ema_used_for_validation_checkpoint": ema_model is not None,
    }


def _config_path(dataset_key: str, condition: str) -> Path:
    spec = DATASETS[dataset_key]
    suffix = CONDITIONS[condition]["config_suffix"]
    return CODE_ROOT / "config" / f"config-{spec['config_stem']}{suffix}.yaml"


def _reproduction_command(dataset_key: str, condition: str) -> str:
    return (
        "python tests/gpu_smoke_12.py "
        f"--dataset {dataset_key} --condition {condition} "
        f"--data-root {dataset_key}=<PREPROCESSED_ROOT> "
        "--pretrained-root <LEAF_ASSETS> --work-dir <TEMP_DIR>"
    )


def _classify_failure(exc: Exception) -> str:
    if isinstance(exc, (FileNotFoundError, ModuleNotFoundError, ImportError)):
        return "BLOCKED"
    message = str(exc).lower()
    blocked_fragments = (
        "out of memory",
        "cuda is unavailable",
        "nvidia-smi",
        "missing local dinov2",
        "no space left",
    )
    if any(fragment in message for fragment in blocked_fragments):
        return "BLOCKED"
    return "FAIL"


def _new_combination_result(dataset_key: str, condition: str) -> dict[str, Any]:
    return {
        "dataset": dataset_key,
        "condition": condition,
        "status": "BLOCKED",
        "stage": "not_started",
        "config": f"code/config/{_config_path(dataset_key, condition).name}",
        "reproduction_command": _reproduction_command(dataset_key, condition),
        "coverage": {
            "train_manifest_read": False,
            "train_forward": False,
            "loss": False,
            "backward": False,
            "optimizer_step": False,
            "val_manifest_read": False,
            "validation_forward": False,
            "validation_metric": False,
            "checkpoint_saved": False,
            "checkpoint_loaded": False,
            "test_manifest_read": False,
            "test_inference": False,
            "audit_branch": False,
            "no_cpu_fallback": False,
        },
        "temporary_artifacts_cleaned": False,
    }


def _execute_combination(
    *,
    dataset_key: str,
    condition: str,
    data_root: Path,
    manifest_root: Path,
    assets_root: Path,
    temp_dir: Path,
    device: torch.device,
    train_batch_size: int,
    dino_cache: DinoCache,
    backend_defaults: dict[str, bool],
    result: dict[str, Any],
) -> None:
    result["stage"] = "configuration"
    config_path = _config_path(dataset_key, condition)
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing combination config: {config_path}")
    cfg = OmegaConf.load(config_path)
    audit_mode = normalize_audit_mode(cfg)
    expected_mode = CONDITIONS[condition]["audit_mode"]
    if audit_mode != expected_mode:
        raise ValueError(
            f"{config_path.name} resolves audit_mode={audit_mode!r}, expected {expected_mode!r}"
        )
    if str(cfg.dataset_name) != str(DATASETS[dataset_key]["loader_name"]):
        raise ValueError(
            f"{config_path.name} dataset_name={cfg.dataset_name!r} does not match {dataset_key}"
        )
    expected_classes = int(DATASETS[dataset_key]["num_seg_classes"])
    if int(cfg.num_seg_classes) != expected_classes:
        raise ValueError(
            f"{config_path.name} num_seg_classes={cfg.num_seg_classes}, expected {expected_classes}"
        )
    if not bool(cfg.use_alignment):
        raise ValueError("The verified LEAF configuration requires use_alignment=True")
    if str(cfg.mixed_precision).lower() != "bf16":
        raise ValueError(
            f"The verified RTX 5090 configuration requires bf16, got {cfg.mixed_precision!r}"
        )
    if condition == "shuffle-yt" and train_batch_size < 2:
        raise ValueError("Shuffle-Yt requires --train-batch-size >= 2")
    if train_batch_size < 1:
        raise ValueError("--train-batch-size must be positive")
    if not (assets_root / "vae").is_dir() or not (assets_root / "unet").is_dir():
        raise FileNotFoundError(
            "The supplied pretrained root must contain the real LEAF vae/ and unet/ assets"
        )

    cfg.pretrained_model_name_or_path = str(assets_root)
    cfg.base_data_dir = str(data_root)
    result["resolved_audit_mode"] = audit_mode
    result["formal_parameters"] = {
        "seed": int(cfg.seed),
        "resolution": int(cfg.resolution),
        "train_batch_size": int(cfg.train_batch_size),
        "test_batch_size": int(cfg.test_batch_size),
        "mixed_precision": str(cfg.mixed_precision),
        "prediction_type": str(cfg.prediction_type),
        "use_alignment": bool(cfg.use_alignment),
        "use_ema": bool(cfg.use_ema),
    }
    result["smoke_parameters"] = {
        "train_batch_size": train_batch_size,
        "validation_batch_size": 1,
        "test_batch_size": 1,
        "num_workers": 0,
        "training_steps": 1,
        "validation_batches": 1,
        "test_batches": 1,
        "num_inference_steps": 1 if condition != "core-no-diff" else 0,
    }

    result["stage"] = "manifest_and_data_loading"
    datasets, physical_sources = _manifest_datasets(
        dataset_key,
        data_root,
        manifest_root,
        resolution=int(cfg.resolution),
        seed=int(cfg.seed),
        num_seg_classes=int(cfg.num_seg_classes),
    )
    result["manifests"] = {
        split: datasets[split].evidence() for split in ("train", "val", "test")
    }
    result["physical_split_sources"] = physical_sources
    result["coverage"].update(
        {
            "train_manifest_read": True,
            "val_manifest_read": True,
            "test_manifest_read": True,
        }
    )
    _set_seed(int(cfg.seed))
    batches = {
        "train": _first_batch(
            datasets["train"],
            batch_size=train_batch_size,
            train=True,
            seed=int(cfg.seed),
        ),
        "val": _first_batch(
            datasets["val"],
            batch_size=1,
            train=False,
            seed=int(cfg.seed),
        ),
        "test": _first_batch(
            datasets["test"],
            batch_size=1,
            train=False,
            seed=int(cfg.seed),
        ),
    }

    result["stage"] = "cuda_setup"
    torch.cuda.reset_peak_memory_stats(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    result["backend_policy"] = {
        "train_and_training_validation": {
            "matmul_tf32": True,
            "cudnn_tf32": True,
            "amp_dtype": "bfloat16",
        },
        "standalone_test_inference": {
            "matmul_tf32": backend_defaults["matmul_tf32"],
            "cudnn_tf32": backend_defaults["cudnn_tf32"],
            "amp_dtype": "bfloat16",
        },
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }
    weight_dtype = _dtype_from_config(cfg)
    if condition == "core-no-diff":
        evidence = _execute_core(
            cfg=cfg,
            batches=batches,
            assets_root=assets_root,
            temp_dir=temp_dir,
            device=device,
            weight_dtype=weight_dtype,
            dino_cache=dino_cache,
            backend_defaults=backend_defaults,
            result=result,
        )
    else:
        evidence = _execute_full(
            cfg=cfg,
            audit_mode=audit_mode,
            batches=batches,
            assets_root=assets_root,
            temp_dir=temp_dir,
            device=device,
            weight_dtype=weight_dtype,
            dino_cache=dino_cache,
            backend_defaults=backend_defaults,
            result=result,
        )
    result.update(evidence)
    result["peak_cuda_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
    result["stage"] = "complete"
    result["status"] = "PASS"


def _remove_temp_child(path: Path, work_root: Path) -> None:
    resolved_path = path.resolve()
    resolved_root = work_root.resolve()
    if resolved_path.parent != resolved_root:
        raise RuntimeError(
            f"Refusing to clean unexpected smoke path outside the work root: {resolved_path}"
        )
    if resolved_path.exists():
        shutil.rmtree(resolved_path)


def _run_combination(
    *,
    dataset_key: str,
    condition: str,
    data_root: Path,
    manifest_root: Path,
    assets_root: Path,
    work_root: Path,
    device: torch.device,
    train_batch_size: int,
    dino_cache: DinoCache,
    backend_defaults: dict[str, bool],
) -> dict[str, Any]:
    result = _new_combination_result(dataset_key, condition)
    temp_dir = Path(
        tempfile.mkdtemp(
            prefix=f"leaf-gpu-smoke-{dataset_key}-{condition}-",
            dir=str(work_root),
        )
    )
    try:
        _execute_combination(
            dataset_key=dataset_key,
            condition=condition,
            data_root=data_root,
            manifest_root=manifest_root,
            assets_root=assets_root,
            temp_dir=temp_dir,
            device=device,
            train_batch_size=train_batch_size,
            dino_cache=dino_cache,
            backend_defaults=backend_defaults,
            result=result,
        )
    except Exception as exc:
        result["status"] = _classify_failure(exc)
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = backend_defaults["matmul_tf32"]
        torch.backends.cudnn.allow_tf32 = backend_defaults["cudnn_tf32"]
        cleanup_error: Exception | None = None
        try:
            _remove_temp_child(temp_dir, work_root)
        except Exception as exc:
            cleanup_error = exc
        gc.collect()
        torch.cuda.empty_cache()
        if cleanup_error is None:
            result["temporary_artifacts_cleaned"] = True
        else:
            result["temporary_artifacts_cleaned"] = False
            result["status"] = "FAIL"
            result["cleanup_error_type"] = type(cleanup_error).__name__
            result["cleanup_error"] = str(cleanup_error)
    return result


def _parse_data_roots(values: list[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"Invalid --data-root {value!r}; expected DATASET=PATH"
            )
        dataset_key, raw_path = value.split("=", 1)
        dataset_key = dataset_key.strip().lower()
        if dataset_key not in DATASETS:
            raise ValueError(
                f"Unknown --data-root dataset {dataset_key!r}; choose from {sorted(DATASETS)}"
            )
        if dataset_key in roots:
            raise ValueError(f"Duplicate --data-root for {dataset_key}")
        roots[dataset_key] = Path(raw_path).expanduser().resolve()
    return roots


def _selected_values(value: str, choices: Iterable[str]) -> list[str]:
    return list(choices) if value == "all" else [value]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run real one-step LEAF CUDA smoke coverage for one or all "
            "dataset-condition combinations."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=[*DATASETS, "all"],
        default="all",
    )
    parser.add_argument(
        "--condition",
        choices=[*CONDITIONS, "all"],
        default="all",
    )
    parser.add_argument(
        "--data-root",
        action="append",
        default=[],
        metavar="DATASET=PATH",
        help=(
            "Preprocessed root containing the LEAF dataset folder. Repeat for each "
            "selected dataset."
        ),
    )
    parser.add_argument(
        "--pretrained-root",
        type=Path,
        required=True,
        help="Read-only local root containing the real LEAF vae/ and unet/ assets.",
    )
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=CODE_ROOT / "manifests",
        help="Root containing fixed btcv/acdc/isic2018 train/val/test manifests.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="Dedicated parent for per-combination temporary artifacts.",
    )
    parser.add_argument(
        "--dinov2-repo",
        type=Path,
        default=None,
        help="Optional local DINOv2 checkout; otherwise torch.hub uses TORCH_HOME.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=2,
        help="Smoke-only batch size. Shuffle-Yt requires at least 2.",
    )
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    dataset_keys = _selected_values(args.dataset, DATASETS)
    conditions = _selected_values(args.condition, CONDITIONS)
    combinations = [
        (dataset_key, condition)
        for dataset_key in dataset_keys
        for condition in conditions
    ]
    started = time.time()
    report: dict[str, Any] = {
        "schema_version": 1,
        "test": "leaf_gpu_smoke_12",
        "expected_gpu_name": EXPECTED_GPU_NAME,
        "selected_combination_count": len(combinations),
        "status": "BLOCKED",
        "combinations": [],
    }

    try:
        data_roots = _parse_data_roots(args.data_root)
    except ValueError as exc:
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["duration_seconds"] = round(time.time() - started, 6)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1
    missing_roots = [dataset_key for dataset_key in dataset_keys if dataset_key not in data_roots]
    if missing_roots:
        report["error_type"] = "ValueError"
        report["error"] = (
            "Missing --data-root for selected datasets: " + ", ".join(missing_roots)
        )
        report["duration_seconds"] = round(time.time() - started, 6)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1

    preflight = run_preflight(args.device)
    report["preflight"] = preflight
    if preflight["status"] != "PASS":
        for dataset_key, condition in combinations:
            result = _new_combination_result(dataset_key, condition)
            result["stage"] = "gpu_preflight"
            result["error_type"] = preflight.get("error_type", "GPUPreflightError")
            result["error"] = preflight.get("error", "Strict RTX 5090 preflight failed")
            result["temporary_artifacts_cleaned"] = True
            report["combinations"].append(result)
        report["summary"] = {"PASS": 0, "FAIL": 0, "BLOCKED": len(combinations)}
        report["duration_seconds"] = round(time.time() - started, 6)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1

    device_index = int(preflight["device"]["index"])
    device = torch.device("cuda", device_index)
    torch.cuda.set_device(device)
    if torch.cuda.get_device_name(device) != EXPECTED_GPU_NAME:
        raise RuntimeError("GPU identity changed after preflight")
    backend_defaults = {
        "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
    }
    work_root = args.work_dir.expanduser().resolve()
    created_work_root = not work_root.exists()
    work_root.mkdir(parents=True, exist_ok=True)
    assets_root = args.pretrained_root.expanduser().resolve()
    manifest_root = args.manifest_root.expanduser().resolve()
    dino_cache = DinoCache(args.dinov2_repo)

    try:
        for dataset_key, condition in combinations:
            print(
                f"Running {dataset_key} x {condition} on {device}",
                file=sys.stderr,
                flush=True,
            )
            result = _run_combination(
                dataset_key=dataset_key,
                condition=condition,
                data_root=data_roots[dataset_key],
                manifest_root=manifest_root,
                assets_root=assets_root,
                work_root=work_root,
                device=device,
                train_batch_size=args.train_batch_size,
                dino_cache=dino_cache,
                backend_defaults=backend_defaults,
            )
            report["combinations"].append(result)
    finally:
        dino_cache.clear()
        torch.backends.cuda.matmul.allow_tf32 = backend_defaults["matmul_tf32"]
        torch.backends.cudnn.allow_tf32 = backend_defaults["cudnn_tf32"]
        if created_work_root and work_root.is_dir() and not any(work_root.iterdir()):
            work_root.rmdir()

    summary = {
        status: sum(item["status"] == status for item in report["combinations"])
        for status in ("PASS", "FAIL", "BLOCKED")
    }
    report["summary"] = summary
    if summary["FAIL"] > 0:
        report["status"] = "FAIL"
    elif summary["BLOCKED"] > 0:
        report["status"] = "BLOCKED"
    else:
        report["status"] = "PASS"
    report["duration_seconds"] = round(time.time() - started, 6)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


def _metric(
    prediction: torch.Tensor,
    target: torch.Tensor,
    cfg: Any,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, int]]:
    metric = SegmentationMetric(
        metrics=list(cfg.metrics),
        device=device,
        num_classes=int(cfg.num_seg_classes),
    )
    metric.update(prediction, target)
    return metric.compute(), metric.summary_counts()


if __name__ == "__main__":
    raise SystemExit(main())
