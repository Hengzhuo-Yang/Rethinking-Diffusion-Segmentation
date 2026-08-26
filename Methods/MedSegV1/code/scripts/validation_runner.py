"""Evaluate checkpoints on every sample in an explicitly provided validation set."""

import argparse
import csv
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch as th
import torchvision.transforms as transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from guided_diffusion.bratsloader import BRATSDataset3D, CachedBRATSSliceDataset
from guided_diffusion.btcvloader import BTCVDataset
from guided_diffusion.acdcloader import ACDCDataset, labels_from_foreground_channels
from guided_diffusion.isicloader import ISICDataset
from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from guided_diffusion.utils import staple


CHECKPOINT_RE = re.compile(r".*?(\d+)\.pt$")


def checkpoint_step(path):
    match = CHECKPOINT_RE.match(path.name)
    return int(match.group(1)) if match else None


def strip_module_prefix(state_dict):
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key[7:] if key.startswith("module.") else key: value for key, value in state_dict.items()}


def load_checkpoint(model, checkpoint_path, device):
    state_dict = th.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(strip_module_prefix(state_dict))
    model.to(device)
    model.eval()


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


def logits_to_prediction(logits):
    if logits.ndim != 4:
        raise ValueError(f"unexpected logits shape: {tuple(logits.shape)}")
    if logits.shape[1] == 1:
        return th.sigmoid(logits)
    if logits.shape[1] == 2:
        return th.softmax(logits, dim=1)[:, 1:2, ...]
    return th.softmax(logits, dim=1)[:, 1:, ...]


def core_no_diff_prediction(model, image):
    module = getattr(model, "module", model)
    if not hasattr(module, "core_no_diff_logits"):
        raise AttributeError(
            "audit_mode=core_no_diff requires the model to implement core_no_diff_logits(image)."
        )
    return logits_to_prediction(module.core_no_diff_logits(image))


def existing_steps(csv_path):
    if not csv_path.exists():
        return set()
    with csv_path.open("r", newline="") as handle:
        return {int(row["step"]) for row in csv.DictReader(handle) if row.get("step")}


def candidate_checkpoints(args, evaluated_steps):
    checkpoint_dir = Path(args.val_checkpoint_dir)
    now = time.time()
    paths = []
    for path in checkpoint_dir.glob("savedmodel*.pt"):
        step = checkpoint_step(path)
        if step is None or step in evaluated_steps or step < args.val_min_step:
            continue
        if args.save_interval > 0 and args.val_every_n_checkpoints > 1:
            interval = args.save_interval * args.val_every_n_checkpoints
            if step % interval != 0:
                continue
        age_seconds = now - path.stat().st_mtime
        if age_seconds < args.val_min_checkpoint_age_seconds:
            continue
        paths.append((step, path))
    return [path for _, path in sorted(paths)]


def build_validation_dataset(args):
    loader_args = argparse.Namespace(**vars(args))
    loader_args.acdc_split = args.val_acdc_split
    loader_args.data_manifest = args.val_manifest
    if args.data_name == "BTCV":
        return BTCVDataset(
            args.val_data_dir,
            image_size=args.image_size,
            mode="Validation",
            data_manifest=args.val_manifest,
        )
    if args.data_name in {"ISIC", "ISIC2018"}:
        return ISICDataset(
            loader_args,
            args.val_data_dir,
            mode="Validation",
            image_size=args.image_size,
        )
    if args.data_name == "ACDC":
        return ACDCDataset(loader_args, args.val_data_dir, mode="Validation")
    if args.val_brats_cache_dir:
        return CachedBRATSSliceDataset(args.val_brats_cache_dir, test_flag=False)
    transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size))])
    return BRATSDataset3D(args.val_data_dir, transform, test_flag=False)


def multiclass_metric_batch(pred, target, threshold, num_classes):
    pred_labels = labels_from_foreground_channels(
        pred,
        num_classes=num_classes,
        threshold=threshold,
    )
    target_labels = labels_from_foreground_channels(
        target,
        num_classes=num_classes,
        threshold=threshold,
    )
    dice_values = []
    iou_values = []
    nonempty_values = []
    for class_id in range(1, int(num_classes)):
        pred_class = pred_labels == class_id
        target_class = target_labels == class_id
        reduce_dims = tuple(range(1, pred_class.ndim))
        intersection = (pred_class & target_class).float().sum(dim=reduce_dims)
        pred_sum = pred_class.float().sum(dim=reduce_dims)
        target_sum = target_class.float().sum(dim=reduce_dims)
        union = pred_sum + target_sum - intersection
        nonempty = target_sum > 0
        dice_denom = pred_sum + target_sum
        dice = th.where(dice_denom > 0, 2 * intersection / dice_denom, th.zeros_like(dice_denom))
        iou = th.where(union > 0, intersection / union, th.zeros_like(union))
        dice_values.extend(dice[nonempty].cpu().tolist())
        iou_values.extend(iou[nonempty].cpu().tolist())
        nonempty_values.extend(nonempty.cpu().tolist())
    return dice_values, iou_values, nonempty_values


def metric_batch(pred, target, threshold, num_classes=None):
    if num_classes is not None and target.ndim == 4 and target.shape[1] == int(num_classes) - 1:
        return multiclass_metric_batch(pred, target, threshold, num_classes)

    pred_bin = (pred > threshold).float()
    target_bin = (target > 0.5).float()
    if target_bin.ndim == 4 and target_bin.shape[1] > 1:
        reduce_dims = tuple(range(2, pred_bin.ndim))
        intersection = (pred_bin * target_bin).sum(dim=reduce_dims)
        pred_sum = pred_bin.sum(dim=reduce_dims)
        target_sum = target_bin.sum(dim=reduce_dims)
        union = pred_sum + target_sum - intersection
        nonempty = target_sum > 0
        if not th.any(nonempty):
            return [], [], nonempty.cpu().flatten().tolist()
        dice_denom = pred_sum + target_sum
        dice = th.where(dice_denom > 0, 2 * intersection / dice_denom, th.zeros_like(dice_denom))
        iou = th.where(union > 0, intersection / union, th.zeros_like(union))
        return dice[nonempty].cpu().tolist(), iou[nonempty].cpu().tolist(), nonempty.cpu().flatten().tolist()

    reduce_dims = tuple(range(1, pred_bin.ndim))
    intersection = (pred_bin * target_bin).sum(dim=reduce_dims)
    pred_sum = pred_bin.sum(dim=reduce_dims)
    target_sum = target_bin.sum(dim=reduce_dims)
    union = pred_sum + target_sum - intersection

    nonempty = target_sum > 0
    if not th.any(nonempty):
        return [], [], nonempty.cpu().tolist()
    dice_denom = pred_sum + target_sum
    dice = th.where(dice_denom > 0, 2 * intersection / dice_denom, th.zeros_like(dice_denom))
    iou = th.where(union > 0, intersection / union, th.zeros_like(union))
    return dice[nonempty].cpu().tolist(), iou[nonempty].cpu().tolist(), nonempty.cpu().tolist()


def metric_policy(args):
    if getattr(args, "data_name", "") == "ACDC":
        return (
            "foreground Dice/IoU over ACDC classes 1..3; skip empty-GT "
            "slice/class observations; background class 0 is excluded"
        )
    if getattr(args, "data_name", "") in {"ISIC", "ISIC2018"}:
        return (
            "ISIC2018 binary lesion Dice/IoU; skip empty-GT foreground slices; "
            "background class 0 is excluded"
        )
    return "skip_empty_ground_truth_slices"


def sample_prediction(model, diffusion, image, args, device):
    image = image.to(device=device, dtype=th.float32)
    if normalize_audit_mode(getattr(args, "audit_mode", "none")) == "core_no_diff":
        with th.no_grad():
            return core_no_diff_prediction(model, image).detach()

    num_mask_channels = int(getattr(args, "num_mask_channels", 1))
    noise_channel = th.randn(
        image.shape[0],
        num_mask_channels,
        image.shape[2],
        image.shape[3],
        device=image.device,
        dtype=image.dtype,
    )
    model_input = th.cat((image, noise_channel), dim=1)
    ensemble = []
    sample_fn = (
        diffusion.ddim_sample_loop_known
        if args.val_use_ddim
        else diffusion.p_sample_loop_known
    )

    with th.no_grad():
        for _ in range(args.val_num_ensemble):
            requested_sample_steps = args.val_sample_steps
            sample_steps = (
                requested_sample_steps
                if requested_sample_steps > 0
                else args.val_diffusion_steps
            )
            sample, _x_noisy, _org, _cal, _cal_out = sample_fn(
                model,
                (model_input.shape[0], model_input.shape[1], args.image_size, args.image_size),
                model_input,
                step=sample_steps,
                clip_denoised=args.val_clip_denoised,
                model_kwargs={},
            )
            if sample.shape[1] == num_mask_channels:
                ensemble.append(sample.detach())
            else:
                ensemble.append(sample[:, -num_mask_channels:, :, :].detach())

    if len(ensemble) == 1:
        return ensemble[0]
    return staple(th.stack(ensemble, dim=0)).squeeze(0)


def evaluate_checkpoint(model, diffusion, loader, checkpoint_path, args, device):
    step = checkpoint_step(checkpoint_path)
    load_checkpoint(model, checkpoint_path, device)

    dice_values = []
    iou_values = []
    nonempty_values = []
    start = time.time()
    for image, target, _path in loader:
        target = target.to(device=device, dtype=th.float32)
        pred = sample_prediction(model, diffusion, image, args, device)
        dice, iou, nonempty = metric_batch(
            pred,
            target,
            args.val_pred_threshold,
            args.num_seg_classes if args.data_name == "ACDC" else None,
        )
        dice_values.extend(dice)
        iou_values.extend(iou)
        nonempty_values.extend(nonempty)

    return {
        "checkpoint": checkpoint_path.name,
        "step": step,
        "num_slices": len(nonempty_values),
        "num_metric_observations": len(dice_values),
        "dice_mean": float(np.mean(dice_values)) if dice_values else float("nan"),
        "dice_nonempty_mean": float(np.mean(dice_values)) if dice_values else float("nan"),
        "iou_mean": float(np.mean(iou_values)) if iou_values else float("nan"),
        "nonempty_slices": int(sum(nonempty_values)),
        "empty_slices": int(len(nonempty_values) - sum(nonempty_values)),
        "diffusion_steps": args.val_diffusion_steps,
        "num_ensemble": args.val_num_ensemble,
        "pred_threshold": args.val_pred_threshold,
        "metric_policy": metric_policy(args),
        "became_best": "",
        "checkpoint_status": "",
        "stable_best_checkpoint": "",
        "elapsed_seconds": round(time.time() - start, 3),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
    }


class ValidationRunner:
    def __init__(self, args):
        self.args = args
        self.device = dist_util.dev()
        self.dataset = build_validation_dataset(args)
        self.loader = th.utils.data.DataLoader(
            self.dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.val_num_workers,
        )
        self.out_csv = Path(args.val_out_csv)

    def should_validate(self, step):
        if step < self.args.val_min_step:
            return False
        if self.args.save_interval > 0 and self.args.val_every_n_checkpoints > 1:
            interval = self.args.save_interval * self.args.val_every_n_checkpoints
            if step % interval != 0:
                return False
        return step not in existing_steps(self.out_csv)

    def evaluate_checkpoint_file(self, model, diffusion, checkpoint_path):
        checkpoint_path = Path(checkpoint_path)
        step = checkpoint_step(checkpoint_path)
        if step is None or not self.should_validate(step):
            return None

        was_training = model.training
        old_dpm_solver = getattr(diffusion, "dpm_solver", None)
        diffusion.dpm_solver = self.args.val_dpm_solver

        torch_rng_state = th.get_rng_state()
        cuda_rng_state = th.cuda.get_rng_state_all() if th.cuda.is_available() else None
        numpy_rng_state = np.random.get_state()
        random_rng_state = random.getstate()
        try:
            th.manual_seed(self.args.val_seed)
            if th.cuda.is_available():
                th.cuda.manual_seed_all(self.args.val_seed)
            np.random.seed(self.args.val_seed)
            random.seed(self.args.val_seed)
            row = evaluate_checkpoint(model, diffusion, self.loader, checkpoint_path, self.args, self.device)
            append_result(self.out_csv, row)
            return row
        finally:
            if old_dpm_solver is not None:
                diffusion.dpm_solver = old_dpm_solver
            if was_training:
                model.train()
            th.set_rng_state(torch_rng_state)
            if cuda_rng_state is not None:
                th.cuda.set_rng_state_all(cuda_rng_state)
            np.random.set_state(numpy_rng_state)
            random.setstate(random_rng_state)
            if th.cuda.is_available():
                th.cuda.empty_cache()

    def evaluate_model(self, model, diffusion, step, checkpoint_name=None):
        if not self.should_validate(step):
            return None

        checkpoint_name = checkpoint_name or f"savedmodel{step:06d}.pt"
        was_training = model.training
        old_dpm_solver = getattr(diffusion, "dpm_solver", None)
        diffusion.dpm_solver = self.args.val_dpm_solver

        dice_values = []
        iou_values = []
        nonempty_values = []
        start = time.time()
        torch_rng_state = th.get_rng_state()
        cuda_rng_state = th.cuda.get_rng_state_all() if th.cuda.is_available() else None
        numpy_rng_state = np.random.get_state()
        random_rng_state = random.getstate()

        try:
            th.manual_seed(self.args.val_seed)
            if th.cuda.is_available():
                th.cuda.manual_seed_all(self.args.val_seed)
            np.random.seed(self.args.val_seed)
            random.seed(self.args.val_seed)
            model.eval()
            for image, target, _path in self.loader:
                target = target.to(device=self.device, dtype=th.float32)
                pred = sample_prediction(model, diffusion, image, self.args, self.device)
                dice, iou, nonempty = metric_batch(
                    pred,
                    target,
                    self.args.val_pred_threshold,
                    self.args.num_seg_classes
                    if self.args.data_name == "ACDC"
                    else None,
                )
                dice_values.extend(dice)
                iou_values.extend(iou)
                nonempty_values.extend(nonempty)
        finally:
            if old_dpm_solver is not None:
                diffusion.dpm_solver = old_dpm_solver
            if was_training:
                model.train()
            th.set_rng_state(torch_rng_state)
            if cuda_rng_state is not None:
                th.cuda.set_rng_state_all(cuda_rng_state)
            np.random.set_state(numpy_rng_state)
            random.setstate(random_rng_state)

        row = {
            "checkpoint": checkpoint_name,
            "step": step,
            "num_slices": len(nonempty_values),
            "num_metric_observations": len(dice_values),
            "dice_mean": float(np.mean(dice_values)) if dice_values else float("nan"),
            "dice_nonempty_mean": float(np.mean(dice_values)) if dice_values else float("nan"),
            "iou_mean": float(np.mean(iou_values)) if iou_values else float("nan"),
            "nonempty_slices": int(sum(nonempty_values)),
            "empty_slices": int(len(nonempty_values) - sum(nonempty_values)),
            "diffusion_steps": self.args.val_diffusion_steps,
            "num_ensemble": self.args.val_num_ensemble,
            "pred_threshold": self.args.val_pred_threshold,
            "metric_policy": metric_policy(self.args),
            "became_best": "",
            "checkpoint_status": "",
            "stable_best_checkpoint": "",
            "elapsed_seconds": round(time.time() - start, 3),
            "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        }
        append_result(self.out_csv, row)
        if th.cuda.is_available():
            th.cuda.empty_cache()
        return row

    def mark_checkpoint_status(self, step, became_best, checkpoint_status, stable_best_checkpoint):
        if not self.out_csv.exists():
            return
        with self.out_csv.open("r", newline="") as handle:
            rows = list(csv.DictReader(handle))
            fieldnames = list(rows[0].keys()) if rows else []
        for field in ("became_best", "checkpoint_status", "stable_best_checkpoint"):
            if field not in fieldnames:
                fieldnames.append(field)
        for row in rows:
            if str(row.get("step")) == str(step):
                row["became_best"] = int(bool(became_best))
                row["checkpoint_status"] = checkpoint_status
                row["stable_best_checkpoint"] = stable_best_checkpoint
        with self.out_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def append_result(csv_path, row):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "checkpoint",
        "step",
        "num_slices",
        "num_metric_observations",
        "dice_mean",
        "dice_nonempty_mean",
        "iou_mean",
        "nonempty_slices",
        "empty_slices",
        "diffusion_steps",
        "num_ensemble",
        "pred_threshold",
        "metric_policy",
        "became_best",
        "checkpoint_status",
        "stable_best_checkpoint",
        "elapsed_seconds",
        "evaluated_at",
    ]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def create_argparser():
    defaults = model_and_diffusion_defaults()
    defaults.update(
        dict(
            val_data_dir="../data_preprocessed/BraTS2021/validation",
            val_manifest="",
            data_name="BRATS",
            val_brats_cache_dir="",
            val_checkpoint_dir="../runs/brats2021_v1b",
            val_out_csv="../runs/brats2021_v1b_validation.csv",
            val_gpu_dev="0",
            image_size=256,
            num_channels=128,
            class_cond=False,
            num_res_blocks=2,
            num_heads=1,
            learn_sigma=True,
            use_scale_shift_norm=False,
            attention_resolutions="16",
            val_dpm_solver=True,
            val_diffusion_steps=20,
            val_sample_steps=0,
            noise_schedule="linear",
            rescale_learned_sigmas=False,
            rescale_timesteps=False,
            version="1",
            val_batch_size=1,
            val_num_workers=0,
            val_num_ensemble=1,
            num_seg_classes=2,
            num_mask_channels=1,
            val_use_ddim=False,
            val_clip_denoised=True,
            val_seed=2021,
            val_pred_threshold=0.5,
            audit_mode="none",
            val_acdc_split="validation",
            save_interval=5000,
            val_every_n_checkpoints=1,
            val_min_step=5000,
            val_min_checkpoint_age_seconds=30,
            val_checkpoint_path="",
            val_watch=False,
            val_poll_seconds=300,
        )
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def main():
    args = create_argparser().parse_args()
    args.audit_mode = normalize_audit_mode(args.audit_mode)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.val_gpu_dev)
    th.manual_seed(args.val_seed)
    np.random.seed(args.val_seed)
    random.seed(args.val_seed)

    if args.data_name in {"BTCV", "ISIC", "ISIC2018"}:
        args.in_ch = 4
    elif args.data_name == "ACDC":
        args.in_ch = 6
    else:
        args.in_ch = 5
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    runner = ValidationRunner(args)
    out_csv = runner.out_csv
    while True:
        evaluated = existing_steps(out_csv)
        if args.val_checkpoint_path:
            checkpoint_path = Path(args.val_checkpoint_path)
            step = checkpoint_step(checkpoint_path)
            checkpoints = [] if step in evaluated else [checkpoint_path]
        else:
            checkpoints = candidate_checkpoints(args, evaluated)
        if not checkpoints:
            print("No new checkpoints to validate.", flush=True)
        for checkpoint_path in checkpoints:
            print(
                f"Validating {checkpoint_path.name} on the complete "
                f"{len(runner.dataset)}-sample validation set.",
                flush=True,
            )
            try:
                row = evaluate_checkpoint(model, diffusion, runner.loader, checkpoint_path, args, runner.device)
            except Exception as exc:
                print(f"Failed to validate {checkpoint_path}: {exc}", flush=True)
                continue
            append_result(out_csv, row)
            print(
                f"step={row['step']} dice_mean={row['dice_mean']:.4f} "
                f"dice_nonempty_mean={row['dice_nonempty_mean']:.4f} "
                f"iou_mean={row['iou_mean']:.4f}",
                flush=True,
            )
            if th.cuda.is_available():
                th.cuda.empty_cache()

        if not args.val_watch:
            break
        time.sleep(args.val_poll_seconds)


if __name__ == "__main__":
    main()
