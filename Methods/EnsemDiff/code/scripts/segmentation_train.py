"""
Train a diffusion model on images.
"""
import sys
import argparse
import json
import os
import random
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
import numpy as np
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion.btcvloader import BTCVDataset
from guided_diffusion.acdcloader import ACDCDataset
from guided_diffusion.isicloader import ISIC2018Dataset
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_gaussian_diffusion,
    create_model_and_diffusion,
    create_model,
    args_to_dict,
    add_dict_to_argparser,
)
import torch as th
from guided_diffusion.train_util import TrainLoop
from guided_diffusion.validation_util import validate_segmentation
from guided_diffusion.visdom_util import make_visdom
viz = make_visdom(port=8850)


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


def create_dataset(args, test_flag, manifest_path=""):
    dataset = args.dataset.lower()
    if dataset in ("brats", "brats2020"):
        return BRATSDataset(args.data_dir, test_flag=test_flag)
    if dataset in ("btcv", "synapse"):
        return BTCVDataset(
            args.data_dir, test_flag=test_flag, manifest_path=manifest_path
        )
    if dataset == "acdc":
        return ACDCDataset(
            args.data_dir,
            test_flag=test_flag,
            num_classes=getattr(args, "num_seg_classes", 4),
            manifest_path=manifest_path,
        )
    if dataset in ("isic", "isic2018"):
        return ISIC2018Dataset(
            args.data_dir, test_flag=test_flag, manifest_path=manifest_path
        )
    raise ValueError(f"Unknown dataset: {args.dataset}")


def create_validation_fn(args, model):
    if not args.val_data_dir or args.val_interval <= 0:
        return None

    val_ds = create_dataset(
        argparse.Namespace(
            dataset=args.dataset,
            data_dir=args.val_data_dir,
            num_seg_classes=args.num_seg_classes,
        ),
        test_flag=False,
        manifest_path=args.val_manifest,
    )
    val_loader = th.utils.data.DataLoader(
        val_ds,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.val_num_workers,
    )
    val_diffusion = None
    if args.audit_mode != "core_no_diff":
        val_diffusion = create_gaussian_diffusion(
            steps=args.diffusion_steps,
            learn_sigma=args.learn_sigma,
            noise_schedule=args.noise_schedule,
            use_kl=args.use_kl,
            predict_xstart=args.predict_xstart,
            rescale_timesteps=args.rescale_timesteps,
            rescale_learned_sigmas=args.rescale_learned_sigmas,
            timestep_respacing=args.val_timestep_respacing,
        )
    val_sampler_steps = 0 if args.audit_mode == "core_no_diff" else val_diffusion.num_timesteps
    logger.log(
        "validation enabled: "
        f"data={args.val_data_dir}, interval={args.val_interval}, "
        f"slices={len(val_ds)}, sampler_steps={val_sampler_steps}, "
        f"ensemble={args.val_num_ensemble}, batch_size={args.val_batch_size}, "
        f"output={args.val_output_csv}"
    )

    def _run_validation(step):
        return validate_segmentation(
            model=model,
            diffusion=val_diffusion,
            dataloader=val_loader,
            step=step,
            image_channels=args.image_channels,
            mask_channels=args.mask_channels,
            num_seg_classes=args.num_seg_classes,
            num_ensemble=args.val_num_ensemble,
            use_ddim=args.val_use_ddim,
            output_csv=args.val_output_csv,
            audit_mode=args.audit_mode,
        )

    return _run_validation


def _dist_rank():
    if not th.distributed.is_available() or not th.distributed.is_initialized():
        return 0
    return th.distributed.get_rank()


def _count_full_diffusion_reference_parameters(args):
    if not args.core_no_diff:
        return None
    ref_model = create_model(
        args.image_size,
        args.num_channels,
        args.num_res_blocks,
        channel_mult=args.channel_mult,
        image_channels=args.image_channels,
        mask_channels=args.mask_channels,
        num_seg_classes=args.num_seg_classes,
        learn_sigma=args.learn_sigma,
        class_cond=args.class_cond,
        use_checkpoint=args.use_checkpoint,
        attention_resolutions=args.attention_resolutions,
        num_heads=args.num_heads,
        num_head_channels=args.num_head_channels,
        num_heads_upsample=args.num_heads_upsample,
        use_scale_shift_norm=args.use_scale_shift_norm,
        dropout=args.dropout,
        resblock_updown=args.resblock_updown,
        use_fp16=args.use_fp16,
        use_new_attention_order=args.use_new_attention_order,
        core_no_diff=False,
    )
    param_count = sum(p.numel() for p in ref_model.parameters())
    del ref_model
    return param_count


def log_training_metadata(args, model, diffusion):
    param_count = sum(p.numel() for p in model.parameters())
    full_diffusion_param_count = _count_full_diffusion_reference_parameters(args)
    full_diffusion_reference_param_count = (
        full_diffusion_param_count if full_diffusion_param_count is not None else param_count
    )
    resolved_config = vars(args).copy()
    random_seed = args.seed if args.seed >= 0 else None
    core_no_diff = args.audit_mode == "core_no_diff"
    train_random_yt = args.audit_mode == "train_random_yt"
    train_shuffle_yt = args.audit_mode == "train_shuffle_yt"
    if core_no_diff:
        original_output_type = "mask/logits/Y0"
        original_target_type = "mask/Y0"
        original_loss_type = "direct Dice+CE/BCE segmentation"
    else:
        original_output_type = "mask/Y0" if args.predict_xstart else "epsilon/noise"
        original_target_type = "mask/Y0" if args.predict_xstart else "epsilon/noise"
        original_loss_type = (
            "rescaled_diffusion_mse"
            if args.rescale_learned_sigmas
            else "diffusion_mse"
        )
        if args.learn_sigma:
            original_output_type += " + learned_variance"
            original_loss_type += " + learned_variance_vb"
    metadata = {
        "audit_mode": args.audit_mode,
        "counterfactual_type": (
            "discriminative_capacity"
            if core_no_diff
            else "Yt_pairing_ablation"
            if train_shuffle_yt
            else "Yt_information_ablation"
            if train_random_yt
            else "not_applicable"
        ),
        "objective_preserving": False if core_no_diff else True,
        "random_seed": random_seed,
        "parameter_count": param_count,
        "parameter_count_full_diffusion": full_diffusion_reference_param_count,
        "parameter_count_core_no_diff": param_count if core_no_diff else None,
        "resolved_config": resolved_config,
        "image_channels": args.image_channels,
        "mask_channels": args.mask_channels,
        "num_seg_classes": args.num_seg_classes,
        "original_output_type": original_output_type,
        "original_target_type": original_target_type,
        "original_loss_type": original_loss_type,
        "target_changed": core_no_diff,
        "loss_changed": core_no_diff,
        "model_output_changed": core_no_diff,
        "model_parameter_count_identical_to_full_diffusion": (
            param_count == full_diffusion_reference_param_count
        ),
        "changed_input_channels": core_no_diff,
        "changed_prediction_head": core_no_diff,
        "removed_timestep_embedding": core_no_diff,
        "removed_noise_branch": core_no_diff,
        "main_core_input": "image_only" if core_no_diff else "image_plus_Yt",
        "uses_Y_t_in_main_core": False if core_no_diff else True,
        "uses_timestep_in_main_core": False if core_no_diff else True,
        "uses_q_sample_for_main_core": False if core_no_diff else True,
        "uses_reverse_sampler_for_main_core": False if core_no_diff else True,
        "diffusion_loss_used_for_main_core": False if core_no_diff else True,
        "main_core_loss": (
            "direct_segmentation_loss" if core_no_diff else "diffusion_mse"
        ),
        "sampling_steps_main_core": 0 if core_no_diff else diffusion.num_timesteps,
        "gpu_memory_allocated_bytes_at_start": th.cuda.memory_allocated(),
        "gpu_memory_reserved_bytes_at_start": th.cuda.memory_reserved(),
        "retained_auxiliary_modules": (
            [
                "UNet encoder/decoder residual path",
                "UNet attention blocks",
                "skip connections",
                "optimizer, scheduler, EMA checkpointing",
                "dataset split, augmentation, and Dice/IoU metrics",
            ]
            if core_no_diff
            else [
                "UNetModel attention/residual/timestep-conditioning blocks",
                "learned variance head" if args.learn_sigma else "fixed variance path",
                "EMA checkpointing",
                "training loss and validation/sampling paths",
            ]
        ),
        "modified_auxiliary_modules": (
            [
                "input stem changed from image-plus-Yt to image-only",
                "ResBlocks built without timestep embedding projections",
                "prediction head interpreted as direct segmentation logits",
                "validation/inference changed to single direct forward pass",
            ]
            if core_no_diff
            else []
        ),
        "incompatible_auxiliary_modules": (
            [
                "reverse diffusion sampler for main-core inference",
                "q_sample-derived noisy-mask input channel",
                "learned variance/noise prediction objective",
            ]
            if core_no_diff
            else []
        ),
        "objective_changes": (
            [
                "main-core diffusion MSE/noise objective replaced with direct Dice+CE/BCE segmentation loss against Y0"
            ]
            if core_no_diff
            else []
        ),
        "image_I_unmodified": True,
        "target_Y0_unmodified": True,
        "timestep_t_unmodified": False if core_no_diff else True,
        "timestep_conditioning_unchanged": False if core_no_diff else True,
        "model_architecture_unchanged": False if core_no_diff else True,
        "loss_target_unchanged": False if core_no_diff else True,
        "diffusion_target_unchanged": False if core_no_diff else True,
        "original_target_unmodified": False if core_no_diff else True,
        "original_loss_unmodified": False if core_no_diff else True,
        "Yt_input_replaced_by_independent_gaussian": train_random_yt,
        "Yt_input_same_shape_dtype_device_as_Yt_ref": (
            True if train_random_yt or train_shuffle_yt else None
        ),
        "Yt_input_constructed_from_Y0": (
            True if train_shuffle_yt else False if train_random_yt else None
        ),
        "Yt_input_uses_current_sample_timestep": train_shuffle_yt,
        "Yt_input_uses_current_sample_epsilon_noise": train_shuffle_yt,
        "loss_uses_original_Yt_ref": False if core_no_diff else True,
        "vb_loss_uses_original_Yt_ref": (
            True if train_shuffle_yt and args.learn_sigma else None
        ),
        "epsilon_noise_target_replaced_by_random_Yt_input_noise": False,
        "Yt_shuffle_strategy": (
            "batch_level_derangement" if train_shuffle_yt else "not_applicable"
        ),
        "no_self_match_enforced": train_shuffle_yt,
        "Yt_source_mask_pairing": (
            "shuffled_batch_mask"
            if train_shuffle_yt
            else "independent_gaussian"
            if train_random_yt
            else "not_applicable"
            if core_no_diff
            else "matched_mask"
        ),
        "Yt_reference_source_mask_pairing": (
            "not_applicable" if core_no_diff else "matched_mask"
        ),
        "Yt_generated_after_shuffle": train_shuffle_yt,
        "dataset_level_shuffle_fallback": "not_implemented" if train_shuffle_yt else "",
        "training_timestep_sampling_unchanged": False if core_no_diff else True,
        "validation_sampling_steps": (
            0 if core_no_diff and args.val_data_dir else args.val_timestep_respacing if args.val_data_dir else ""
        ),
        "diffusion_steps": diffusion.num_timesteps,
    }
    if args.audit_mode == "train_random_yt":
        metadata["random_Yt_type"] = (
            "independent standard Gaussian via torch.randn_like(Y_t_ref)"
        )
        metadata["audit_scope"] = "training input Y_t channel only"
    elif train_shuffle_yt:
        metadata["random_Yt_type"] = (
            "not random; q_sample uses a deranged batch mask with the current "
            "sample timestep and noise target"
        )
        metadata["audit_scope"] = "training input Y_t channel only"
    elif core_no_diff:
        metadata["random_Yt_type"] = "not_applicable"
        metadata["audit_scope"] = "train and validation main segmentation path"
    else:
        metadata["random_Yt_type"] = "not used"
        metadata["audit_scope"] = "full diffusion training"

    logger.log(f"audit_mode = {args.audit_mode}")
    logger.log(f"random_seed = {random_seed if random_seed is not None else 'unset'}")
    logger.log(f"parameter_count = {param_count}")
    logger.log(f"original_output_type = {metadata['original_output_type']}")
    logger.log(f"original_target_type = {metadata['original_target_type']}")
    logger.log(f"original_loss_type = {metadata['original_loss_type']}")
    if core_no_diff:
        logger.log(f"parameter_count_full_diffusion = {full_diffusion_param_count}")
        logger.log(f"parameter_count_core_no_diff = {param_count}")
    logger.log(f"random_Yt_type = {metadata['random_Yt_type']}")
    if train_random_yt:
        logger.log(
            "audit invariant: train_random_yt first constructs the original "
            "q_sample(Y0, t, epsilon_ref) tuple, then replaces only the "
            "training-time Y_t model input with independent standard Gaussian; "
            "image I, timesteps, epsilon/noise target, loss construction, "
            "validation, and sampling are unchanged"
        )
        logger.log("target_changed = false")
        logger.log("loss_changed = false")
        logger.log("model_output_changed = false")
    elif train_shuffle_yt:
        logger.log("Yt_shuffle_strategy = batch_level_derangement")
        logger.log("no_self_match_enforced = true")
        logger.log(
            "audit invariant: train_shuffle_yt shuffles only the mask used to "
            "construct training Y_t; image I, timesteps, noise target, loss, "
            "validation, and sampling are unchanged"
        )
    elif core_no_diff:
        logger.log(
            "audit invariant: core_no_diff main path receives image I only; "
            "Y_t, timestep embeddings, q_sample, diffusion loss, and reverse "
            "sampler are not used for the main core"
        )
        logger.log(
            "objective note: core_no_diff is a discriminative-capacity "
            "counterfactual, not an objective-preserving ablation"
        )
    else:
        logger.log(
            "audit invariant: image I and target Y0 are not modified; "
            "loss target, timestep sampling, validation, and sampling are unchanged"
        )
    logger.log(
        "retained_auxiliary_modules = "
        + ", ".join(metadata["retained_auxiliary_modules"])
    )
    logger.log(
        "modified_auxiliary_modules = "
        + (
            ", ".join(metadata["modified_auxiliary_modules"])
            if metadata["modified_auxiliary_modules"]
            else "none"
        )
    )
    if metadata["incompatible_auxiliary_modules"]:
        logger.log(
            "incompatible_auxiliary_modules = "
            + ", ".join(metadata["incompatible_auxiliary_modules"])
        )

    if _dist_rank() == 0:
        config_path = os.path.join(logger.get_dir(), "resolved_config.json")
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
        logger.log(f"saved resolved config: {config_path}")


def main():
    args = create_argparser().parse_args()
    apply_dataset_defaults(args)
    fixed_manifest_dataset = args.dataset.lower() in (
        "btcv",
        "synapse",
        "acdc",
        "isic",
        "isic2018",
    )
    if fixed_manifest_dataset and not args.train_manifest:
        raise ValueError("--train_manifest is required for this dataset.")
    if fixed_manifest_dataset and bool(args.val_data_dir) != (args.val_interval > 0):
        raise ValueError(
            "--val_data_dir and a positive --val_interval must be enabled together."
        )
    if fixed_manifest_dataset and args.val_data_dir and not args.val_manifest:
        raise ValueError(
            "--val_manifest is required whenever validation is enabled for this dataset."
        )
    args.audit_mode = args.audit_mode.lower()
    if args.audit_mode not in VALID_AUDIT_MODES:
        raise ValueError(
            f"Unknown audit_mode: {args.audit_mode}. "
            f"Expected one of {VALID_AUDIT_MODES}."
        )
    args.core_no_diff = args.audit_mode == "core_no_diff"
    if (
        args.audit_mode in ("train_random_yt", "train_shuffle_yt", "core_no_diff")
        and args.seed < 0
    ):
        raise ValueError(f"{args.audit_mode} runs must pass --seed for audit metadata.")
    effective_microbatch = args.microbatch if args.microbatch > 0 else args.batch_size
    if args.audit_mode == "train_shuffle_yt" and effective_microbatch <= 1:
        raise ValueError(
            "train_shuffle_yt requires an effective training microbatch size > 1 "
            "because dataset-level shuffle fallback is not implemented."
        )
    dist_util.setup_dist()
    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        th.manual_seed(args.seed)
        th.cuda.manual_seed_all(args.seed)
    logger.configure(dir=os.getenv("OPENAI_LOGDIR", "./results"))

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    log_training_metadata(args, model, diffusion)
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion,  maxt=1000)
    validation_fn = create_validation_fn(args, model)

    logger.log("creating data loader...")
    ds = create_dataset(args, test_flag=False, manifest_path=args.train_manifest)
    datal= th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)
    data = iter(datal)


    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        classifier=None,
        data=data,
        dataloader=datal,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        validation_fn=validation_fn,
        validation_interval=args.val_interval,
        audit_mode=args.audit_mode,
        seed=args.seed,
        validation_seed=args.val_seed,
        validation_manifest=args.val_manifest,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="./data/training",
        dataset="brats",
        train_manifest="",
        val_data_dir="",
        val_manifest="",
        val_seed=10,
        val_interval=0,
        val_num_ensemble=1,
        val_batch_size=1,
        val_timestep_respacing="100",
        val_use_ddim=False,
        val_num_workers=0,
        val_output_csv="./results/validation_metrics.csv",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        num_workers=0,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=5000,
        resume_checkpoint='',#'"./results/pretrainedmodel.pt",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        audit_mode="none",
        seed=-1,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
