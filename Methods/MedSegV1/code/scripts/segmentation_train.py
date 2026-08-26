
import sys
import argparse
import json
import random
sys.path.append("../")
sys.path.append("./")
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.bratsloader import BRATSDataset, BRATSDataset3D, CachedBRATSSliceDataset
from guided_diffusion.btcvloader import BTCVDataset
from guided_diffusion.acdcloader import ACDCDataset
from guided_diffusion.isicloader import ISICDataset
from guided_diffusion.custom_dataset_loader import CustomDataset,CustomDataset3D
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
import torch as th
import numpy as np
from pathlib import Path
from guided_diffusion.train_util import TrainLoop
import torchvision.transforms as transforms


def normalize_audit_mode(audit_mode):
    audit_mode = str(audit_mode or "none").strip().lower()
    if audit_mode in {"", "null", "false", "none"}:
        return "none"
    if audit_mode in {"full_diffusion", "original", "default"}:
        return "none"
    allowed_modes = {"none", "train_shuffle_yt", "train_random_yt", "core_no_diff"}
    if audit_mode not in allowed_modes:
        raise ValueError(
            f"Unsupported audit_mode={audit_mode!r}. Supported modes: {sorted(allowed_modes)}"
        )
    return audit_mode


def _diffusion_target_type(diffusion):
    mean_type = getattr(diffusion, "model_mean_type", None)
    name = getattr(mean_type, "name", str(mean_type))
    if name == "EPSILON":
        return "epsilon/noise"
    if name == "START_X":
        return "mask/Y_0"
    if name == "PREVIOUS_X":
        return "posterior_mean/x_{t-1}"
    return f"unknown:{name}"


def write_audit_metadata(out_dir, args, diffusion, parameter_count):
    if args.audit_mode not in {"train_shuffle_yt", "train_random_yt", "core_no_diff"}:
        return
    target_type = _diffusion_target_type(diffusion)
    is_shuffle = args.audit_mode == "train_shuffle_yt"
    is_core = args.audit_mode == "core_no_diff"
    metadata = {
        "audit_mode": args.audit_mode,
        "random_seed": args.seed if args.seed >= 0 else None,
        "shuffle_strategy": "batch_level" if is_shuffle else "not_applicable",
        "no_self_match_enforced": is_shuffle,
        "random_Yt_type": "independent standard Gaussian" if args.audit_mode == "train_random_yt" else "not_applicable",
        "original_output_type": target_type,
        "original_target_type": target_type,
        "original_loss_type": "MSE(model_output, original_target); loss_cal is disabled globally",
        "loss_cal_enabled": False,
        "loss_cal_policy": "never enters total loss for any audit mode or baseline reproduction",
        "target_changed": is_core,
        "loss_changed": is_core,
        "model_output_changed": is_core,
        "image_changed": False,
        "timestep_changed": is_core,
        "timestep_sampling_changed": is_core,
        "original_target_modified": is_core,
        "original_loss_modified": is_core,
        "yt_input_constructed_from_another_case_mask": is_shuffle,
        "yt_input_replaced_by_independent_gaussian": args.audit_mode == "train_random_yt",
        "yt_input_uses_current_sample_timestep": is_shuffle,
        "yt_input_uses_current_sample_original_noise": is_shuffle,
        "image_shuffled": False,
        "original_target_shuffled": False,
        "epsilon_or_noise_target_replaced_by_random_yt_input_noise": False,
        "epsilon_or_noise_target_recomputed_from_shuffled_mask": False,
        "training_changes": True,
        "validation_changed": is_core,
        "inference_changed": is_core,
        "sampling_changed": is_core,
        "evaluation_changed": False,
        "objective_preserving": not is_core,
        "core_no_diff_loss_type": "BCEWithLogits+soft_Dice for one-logit binary; CE+foreground_soft_Dice for 2/K-class logits" if is_core else "not_applicable",
        "core_no_diff_removed": [
            "q_sample(Y_0,t,epsilon)",
            "Y_t model input",
            "timestep conditioning",
            "epsilon/noise/v/score diffusion target",
            "reverse DDPM/DDIM sampling",
        ] if is_core else [],
        "retained_auxiliary_modules": [
            "MedSegDiff V1 highway condition encoder",
            "legacy localization parameters for checkpoint compatibility and explicit core_no_diff only",
            "EMA checkpoint tracking",
        ],
        "modified_auxiliary_modules": [
            "calibration loss disabled globally",
            "normal V1 localization decoder bypassed; inference is sample-only",
        ],
        "model_parameter_count": parameter_count,
        "model_parameter_count_identical_to_full_diffusion": True,
        "data_name": args.data_name,
        "diffusion_steps": args.diffusion_steps,
        "version": args.version,
    }
    out_path = Path(out_dir) / "audit_metadata.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

def create_validation_runner(args):
    if not args.validation_runner:
        return None
    if not args.val_data_dir:
        raise ValueError(
            "--val_data_dir is required when --validation_runner True"
        )

    from validation_runner import ValidationRunner

    validation_args = argparse.Namespace(**vars(args))
    out_dir = Path(args.out_dir)
    validation_args.data_manifest = args.val_manifest
    validation_args.val_checkpoint_dir = args.out_dir
    validation_args.val_out_csv = args.val_out_csv or str(
        out_dir / "validation_metrics.csv"
    )
    validation_args.save_interval = args.save_interval
    validation_args.val_min_checkpoint_age_seconds = 0
    validation_args.val_watch = False
    return ValidationRunner(validation_args)


def main():
    args = create_argparser().parse_args()
    args.audit_mode = normalize_audit_mode(args.audit_mode)
    if args.audit_mode == "train_shuffle_yt":
        if args.batch_size <= 1 or (args.microbatch > 0 and args.microbatch <= 1):
            raise ValueError(
                "audit_mode=train_shuffle_yt requires batch_size and microbatch to be greater than 1 "
                "for batch-level no-self-match shuffling."
            )
    if args.seed >= 0:
        th.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        if th.cuda.is_available():
            th.cuda.manual_seed_all(args.seed)

    dist_util.setup_dist(args)
    logger.configure(dir = args.out_dir)

    logger.log("creating data loader...")

    if args.data_name in {'ISIC', 'ISIC2018'}:
        ds = ISICDataset(args, args.data_dir, image_size=args.image_size)
        args.in_ch = 4
    elif args.data_name == 'BRATS':
        tran_list = [transforms.Resize((args.image_size,args.image_size)),]
        transform_train = transforms.Compose(tran_list)

        if args.brats_cache_dir:
            ds = CachedBRATSSliceDataset(args.brats_cache_dir, test_flag=False)
        else:
            ds = BRATSDataset3D(args.data_dir, transform_train, test_flag=False)
        args.in_ch = 5
    elif args.data_name == 'BTCV':
        ds = BTCVDataset(
            args.data_dir,
            image_size=args.image_size,
            data_manifest=args.data_manifest,
        )
        args.in_ch = 4
    elif args.data_name == 'ACDC':
        if int(args.num_seg_classes) != 4 or int(args.num_mask_channels) != 3:
            raise ValueError("ACDC multi-class path expects --num_seg_classes 4 and --num_mask_channels 3.")
        ds = ACDCDataset(args, args.data_dir, mode='Training')
        args.in_ch = 6
    elif any(Path(args.data_dir).glob("*/*.nii.gz")):
        tran_list = [transforms.Resize((args.image_size,args.image_size)),]
        transform_train = transforms.Compose(tran_list)
        print("Your current directory : ",args.data_dir)
        ds = CustomDataset3D(args, args.data_dir, transform_train)
        args.in_ch = 4
    else:
        tran_list = [transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor(),]
        transform_train = transforms.Compose(tran_list)
        print("Your current directory : ",args.data_dir)
        ds = CustomDataset(args, args.data_dir, transform_train)
        args.in_ch = 4
        
    dataloader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    if args.num_workers > 0:
        dataloader_kwargs["persistent_workers"] = args.persistent_workers
        dataloader_kwargs["prefetch_factor"] = args.prefetch_factor
    datal= th.utils.data.DataLoader(ds, **dataloader_kwargs)
    data = iter(datal)

    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    parameter_count = sum(param.numel() for param in model.parameters())
    logger.log(f"audit_mode={args.audit_mode}")
    logger.log("loss_cal_enabled=False")
    logger.log(f"model_parameter_count={parameter_count}")
    write_audit_metadata(args.out_dir, args, diffusion, parameter_count)
    if args.audit_mode == "train_shuffle_yt":
        logger.log(
            "train_shuffle_yt is active: only the training-time noisy mask input is "
            "constructed from another case's mask; image, timestep, original noise target, "
            "diffusion loss, validation, inference, and evaluation are unchanged. loss_cal is disabled globally."
        )
    elif args.audit_mode == "train_random_yt":
        logger.log(
            "train_random_yt is active: only the training-time noisy mask input is "
            "replaced by independent Gaussian noise; image, timestep, original noise target, "
            "diffusion loss, validation, inference, and evaluation are unchanged. loss_cal is disabled globally."
        )
    elif args.audit_mode == "core_no_diff":
        logger.log(
            "core_no_diff is active: training uses image-only direct segmentation logits "
            "with BCE/CE + soft Dice; q_sample, timestep conditioning, diffusion target MSE, "
            "reverse sampling, and loss_cal are disabled."
        )
    if args.multi_gpu:
        model = th.nn.DataParallel(model,device_ids=[int(id) for id in args.multi_gpu.split(',')])
        model.to(device = th.device('cuda', int(args.gpu_dev)))
    else:
        model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion,  maxt=args.diffusion_steps)
    validation_runner = create_validation_runner(args)
    logger.log(
        "validation_runner="
        f"{'enabled' if validation_runner is not None else 'disabled'}"
    )


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
        validation_runner=validation_runner,
        save_best_only=args.save_best_only,
        best_metric_name=args.best_metric_name,
        audit_mode=args.audit_mode,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_name = 'BRATS',
        data_dir="../dataset/brats2020/training",
        data_manifest="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=5000,
        resume_checkpoint=None, #"/results/pretrainedmodel.pt"
        use_fp16=False,
        fp16_scale_growth=1e-3,
        gpu_dev = "0",
        multi_gpu = None, #"0,1,2"
        out_dir='./results/',
        brats_cache_dir="",
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
        validation_runner=False,
        val_data_dir="",
        val_manifest="",
        val_brats_cache_dir="",
        val_acdc_split="validation",
        val_out_csv="",
        val_dpm_solver=True,
        val_diffusion_steps=20,
        val_sample_steps=0,
        val_num_ensemble=1,
        val_every_n_checkpoints=1,
        val_min_step=5000,
        val_batch_size=1,
        val_num_workers=0,
        val_seed=2021,
        val_pred_threshold=0.5,
        val_use_ddim=False,
        val_clip_denoised=True,
        save_best_only=False,
        best_metric_name="dice_mean",
        audit_mode="none",
        seed=-1,
        acdc_split="training",
        acdc_skip_empty=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
