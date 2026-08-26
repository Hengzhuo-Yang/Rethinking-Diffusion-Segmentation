"""
Run the same lightweight train-time segmentation validation for one checkpoint.
"""

import argparse
import os
import random
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import torch as th

from guided_diffusion import dist_util, logger
from guided_diffusion.acdcloader import ACDCDataset
from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion.btcvloader import BTCVDataset
from guided_diffusion.isicloader import ISIC2018Dataset
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    create_gaussian_diffusion,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from guided_diffusion.validation_util import validate_segmentation


VALID_AUDIT_MODES = ("none", "train_random_yt", "train_shuffle_yt", "core_no_diff")


def apply_dataset_defaults(args):
    dataset = args.dataset.lower()
    defaults = model_and_diffusion_defaults()
    if dataset == "acdc":
        if args.image_channels == defaults["image_channels"]:
            args.image_channels = 1
        if args.mask_channels == defaults["mask_channels"]:
            args.mask_channels = 4
        if args.num_seg_classes == defaults["num_seg_classes"]:
            args.num_seg_classes = 4
    elif dataset in ("isic", "isic2018"):
        if args.image_channels == defaults["image_channels"]:
            args.image_channels = 3
        if args.mask_channels == defaults["mask_channels"]:
            args.mask_channels = 1
        if args.num_seg_classes == defaults["num_seg_classes"]:
            args.num_seg_classes = 2


def create_dataset(args):
    dataset = args.dataset.lower()
    if dataset in ("brats", "brats2020"):
        return BRATSDataset(args.data_dir, test_flag=False)
    if dataset in ("btcv", "synapse"):
        return BTCVDataset(
            args.data_dir, test_flag=False, manifest_path=args.manifest
        )
    if dataset == "acdc":
        return ACDCDataset(
            args.data_dir,
            test_flag=False,
            num_classes=getattr(args, "num_seg_classes", 4),
            manifest_path=args.manifest,
        )
    if dataset in ("isic", "isic2018"):
        return ISIC2018Dataset(
            args.data_dir, test_flag=False, manifest_path=args.manifest
        )
    raise ValueError(f"Unknown dataset: {args.dataset}")


def create_argparser():
    defaults = dict(
        data_dir="./data/validation",
        dataset="brats",
        manifest="",
        model_path="",
        output_csv="./results/validation_metrics.csv",
        step=0,
        batch_size=1,
        num_workers=0,
        num_ensemble=1,
        timestep_respacing="100",
        use_ddim=False,
        audit_mode="none",
        seed=10,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def main():
    args = create_argparser().parse_args()
    apply_dataset_defaults(args)
    args.audit_mode = args.audit_mode.lower()
    if args.audit_mode not in VALID_AUDIT_MODES:
        raise ValueError(
            f"Unknown audit_mode: {args.audit_mode}. "
            f"Expected one of {VALID_AUDIT_MODES}."
        )
    if not args.model_path:
        raise ValueError("--model_path is required")
    if args.dataset.lower() in ("btcv", "synapse", "acdc", "isic", "isic2018"):
        if not args.manifest:
            raise ValueError("--manifest is required for BTCV, ACDC, and ISIC2018.")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative for reproducible validation.")
    dist_util.setup_dist()
    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    th.cuda.manual_seed_all(args.seed)
    logger.configure(dir=os.getenv("OPENAI_LOGDIR", "./results"))
    logger.log("creating model and validation diffusion...")
    model, _ = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    diffusion = None
    if args.audit_mode != "core_no_diff":
        diffusion = create_gaussian_diffusion(
            steps=args.diffusion_steps,
            learn_sigma=args.learn_sigma,
            noise_schedule=args.noise_schedule,
            use_kl=args.use_kl,
            predict_xstart=args.predict_xstart,
            rescale_timesteps=args.rescale_timesteps,
            rescale_learned_sigmas=args.rescale_learned_sigmas,
            timestep_respacing=args.timestep_respacing,
        )

    dataset = create_dataset(args)
    dataloader = th.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    sampler_steps = 0 if args.audit_mode == "core_no_diff" else diffusion.num_timesteps
    logger.log(
        "checkpoint validation enabled: "
        f"model={args.model_path}, data={args.data_dir}, step={args.step}, "
        f"manifest={args.manifest}, "
        f"slices={len(dataset)}, sampler_steps={sampler_steps}, "
        f"ensemble={args.num_ensemble}, batch_size={args.batch_size}, "
        f"output={args.output_csv}"
    )
    validate_segmentation(
        model=model,
        diffusion=diffusion,
        dataloader=dataloader,
        step=args.step,
        image_channels=args.image_channels,
        mask_channels=args.mask_channels,
        num_seg_classes=args.num_seg_classes,
        num_ensemble=args.num_ensemble,
        use_ddim=args.use_ddim,
        output_csv=args.output_csv,
        audit_mode=args.audit_mode,
    )


if __name__ == "__main__":
    main()
