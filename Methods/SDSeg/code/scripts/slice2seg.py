# MODIFICATION NOTICE (release prepared 2026-07-19): this file differs from
# upstream SDSeg commit 0b0aa388a5e2def75abfbef90d7bcfc5c16f2704.
# Changes add explicit dataset/config/checkpoint inputs, fixed test manifests,
# audit-mode handling, and portable evaluation. See MODIFICATIONS.md.

import argparse, os, sys, glob, csv, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from torch.utils.data import DataLoader, Subset
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config, default, load_torch_checkpoint
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from ldm.data.btcv import BTCVValidationEval
from ldm.data.acdc import ACDCFullLabelSliceEval
from ldm.data.isic2018 import ISIC2018Test, ISIC2018ValidationEval

from scipy.ndimage import zoom

# from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
# from transformers import AutoFeatureExtractor



def prepare_for_first_stage(x, gpu=True):
    x = x.clone().detach()
    if len(x.shape) == 3:
        x = x[None, ...]
    x = rearrange(x, 'b h w c -> b c h w')
    if gpu:
        x = x.to(memory_format=torch.contiguous_format).float().cuda()
    else:
        x = x.float()
    return x


def dice_score(pred, targs):
    assert pred.shape == targs.shape, (pred.shape, targs.shape)
    pred = np.asarray(pred) > 0
    targs = np.asarray(targs) > 0
    if targs.sum() == 0:
        return None
    return (2.0 * np.logical_and(pred, targs).sum()) / (pred.sum() + targs.sum() + 1e-10)


def iou_score(pred, targs):
    pred = np.asarray(pred) > 0
    targs = np.asarray(targs) > 0
    if targs.sum() == 0:
        return None
    intersection = np.logical_and(pred, targs).sum()
    union = pred.sum() + targs.sum() - intersection
    return intersection / (union + 1e-10)



def load_model_from_config(config, ckpt):
    pl_sd = load_torch_checkpoint(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    # print(set(key.split(".")[0] for key in sd.keys()))
    print(f"\033[31m[Model Weights Rewrite]: Loading model from {ckpt}\033[0m")
    m, u = model.load_state_dict(sd, strict=False)
    # if len(m) > 0 and verbose:
    print("\033[31mmissing keys:\033[0m")
    print(m)
    # if len(u) > 0 and verbose:
    print("\033[31munexpected keys:\033[0m")
    print(u)
    # model.cuda()
    model.eval()
    return model, pl_sd


def calculate_volume_dice(**kwargs):
    # inter_list, union_list, pred_sum, gt_sum = kwargs
    inter = sum(kwargs["inter_list"])
    union = sum(kwargs["union_list"])
    if kwargs["gt_sum"] == 0:
        return None
    return 2 * inter / (union + 1e-10)


def to_plain(value):
    if isinstance(value, dict):
        return {str(k): to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_plain(payload), f, indent=2, sort_keys=True)


def write_csv_rows(path, rows):
    rows = [to_plain(row) for row in rows]
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {}
            for key in fieldnames:
                value = row.get(key)
                if isinstance(value, (dict, list)):
                    flat[key] = json.dumps(value, sort_keys=True)
                else:
                    flat[key] = value
            writer.writerow(flat)


def get_model_audit_mode(config):
    return str(config.model.params.get("audit_mode", "full_diffusion"))


def metric_condition_name(audit_mode, sampling_steps=None):
    if sampling_steps is not None:
        return f"sampling_steps_{int(sampling_steps)}"
    return str(audit_mode)


def sorted_dice_class_keys(row):
    return sorted_metric_class_keys(row, "dice")


def sorted_metric_class_keys(row, metric_name):
    def class_index(key):
        try:
            return int(str(key).split("_")[-1])
        except ValueError:
            return 10 ** 9

    return sorted([key for key in row.keys() if str(key).startswith(f"{metric_name}_cls_")], key=class_index)


def class_name_from_dice_key(key, n_class_keys):
    return class_name_from_metric_key(key, n_class_keys, "dice")


def class_name_from_metric_key(key, n_class_keys, metric_name):
    cls_name = str(key).replace(f"{metric_name}_", "")
    if n_class_keys == 1 and cls_name == "cls_1":
        return "foreground"
    return cls_name


def annotate_per_case_metric_rows(rows, audit_mode, checkpoint_path, seed, sampling_steps=None,
                                  sampler=None, audit_condition=None, default_sampling_steps=None,
                                  exact_initial_noise_reuse=None):
    audit_condition = audit_condition or metric_condition_name(audit_mode, sampling_steps)
    annotated = []
    for row in rows:
        item = dict(row)
        item.update({
            "audit_condition": audit_condition,
            "audit_mode": str(audit_mode),
            "checkpoint_path": os.path.abspath(str(checkpoint_path)),
            "seed": int(seed),
            "sampling_steps": int(sampling_steps) if sampling_steps is not None else None,
            "sampler": sampler,
            "default_sampling_steps": int(default_sampling_steps) if default_sampling_steps is not None else None,
            "exact_initial_noise_reuse": exact_initial_noise_reuse,
        })
        if "inference_time_sec" not in item and "inference_time_seconds" in item:
            item["inference_time_sec"] = item["inference_time_seconds"]
        annotated.append(item)
    return annotated


def build_long_per_case_metric_rows(rows, audit_mode, checkpoint_path, seed, sampling_steps=None,
                                    sampler=None, audit_condition=None, default_sampling_steps=None,
                                    exact_initial_noise_reuse=None):
    audit_condition = audit_condition or metric_condition_name(audit_mode, sampling_steps)
    output_rows = []
    for row in rows:
        case_id = row.get("case_id")
        if case_id in (None, ""):
            continue
        inference_time = row.get("inference_time_sec", row.get("inference_time_seconds"))
        common = {
            "case_id": str(case_id),
            "case_index": row.get("case_index"),
            "audit_condition": audit_condition,
            "audit_mode": str(audit_mode),
            "checkpoint_path": os.path.abspath(str(checkpoint_path)),
            "seed": int(seed),
            "sampling_steps": int(sampling_steps) if sampling_steps is not None else None,
            "sampler": sampler,
            "default_sampling_steps": int(default_sampling_steps) if default_sampling_steps is not None else None,
            "exact_initial_noise_reuse": exact_initial_noise_reuse,
            "inference_time_sec": inference_time,
        }
        if row.get("mean_dice") is not None:
            output_rows.append({
                **common,
                "metric_name": "dice",
                "class_name": "mean",
                "metric_value": row.get("mean_dice"),
            })
        if row.get("mean_iou") is not None:
            output_rows.append({
                **common,
                "metric_name": "iou",
                "class_name": "mean",
                "metric_value": row.get("mean_iou"),
            })
        for metric_name in ("dice", "iou"):
            class_keys = sorted_metric_class_keys(row, metric_name)
            for key in class_keys:
                if row.get(key) is None:
                    continue
                output_rows.append({
                    **common,
                    "metric_name": metric_name,
                    "class_name": class_name_from_metric_key(key, len(class_keys), metric_name),
                    "metric_value": row.get(key),
                })
    return output_rows


def write_per_case_metric_exports(outdir, rows, audit_mode, checkpoint_path, seed, sampling_steps=None,
                                  sampler=None, audit_condition=None, default_sampling_steps=None,
                                  exact_initial_noise_reuse=None):
    long_rows = build_long_per_case_metric_rows(
        rows,
        audit_mode=audit_mode,
        checkpoint_path=checkpoint_path,
        seed=seed,
        sampling_steps=sampling_steps,
        sampler=sampler,
        audit_condition=audit_condition,
        default_sampling_steps=default_sampling_steps,
        exact_initial_noise_reuse=exact_initial_noise_reuse,
    )
    wide_rows = annotate_per_case_metric_rows(
        rows,
        audit_mode=audit_mode,
        checkpoint_path=checkpoint_path,
        seed=seed,
        sampling_steps=sampling_steps,
        sampler=sampler,
        audit_condition=audit_condition,
        default_sampling_steps=default_sampling_steps,
        exact_initial_noise_reuse=exact_initial_noise_reuse,
    )
    write_csv_rows(os.path.join(outdir, "per_case_metrics.csv"), long_rows)
    write_csv_rows(os.path.join(outdir, "per_case_metrics_wide.csv"), wide_rows)
    return long_rows


def build_eval_summary(metrics_dict, details, audit_mode, checkpoint_path, seed, sampler,
                       sampling_steps=None, audit_condition=None):
    per_class_dice = {
        f"cls_{idx}": float(value)
        for idx, value in enumerate(metrics_dict["val_avg_dice"], start=1)
    }
    per_class_iou = {
        f"cls_{idx}": float(value)
        for idx, value in enumerate(metrics_dict["val_avg_iou"], start=1)
    }
    per_case_metrics = details.get("per_case_metrics", [])
    per_case_dice = [
        float(row["mean_dice"]) for row in per_case_metrics
        if row.get("mean_dice") is not None
    ]
    per_case_iou = [
        float(row["mean_iou"]) for row in per_case_metrics
        if row.get("mean_iou") is not None
    ]
    return {
        "audit_condition": audit_condition or metric_condition_name(audit_mode, sampling_steps),
        "audit_mode": str(audit_mode),
        "checkpoint_path": os.path.abspath(str(checkpoint_path)),
        "random_seed": int(seed),
        "sampler": sampler,
        "sampling_steps": int(sampling_steps) if sampling_steps is not None else None,
        "mean_dice": float(np.mean(list(per_class_dice.values()))) if per_class_dice else None,
        "mean_iou": float(np.mean(list(per_class_iou.values()))) if per_class_iou else None,
        "per_class_dice": per_class_dice,
        "per_class_iou": per_class_iou,
        "std_dice_across_cases": float(np.std(per_case_dice)) if per_case_dice else None,
        "std_iou_across_cases": float(np.std(per_case_iou)) if per_case_iou else None,
        "n_cases": len(per_case_dice),
        "n_iou_cases": len(per_case_iou),
        "inference_time_per_case_seconds": details.get("inference_time_per_case_seconds"),
        "requested_steps": details.get("requested_steps"),
        "actual_timestep_schedule": details.get("actual_timestep_schedule"),
        "ddim_eta": details.get("ddim_eta"),
    }


def validate_selection_metadata(path, checkpoint_path):
    metadata_path = os.path.abspath(os.path.expanduser(path))
    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("selection_partition") != "validation":
        raise ValueError("Checkpoint metadata must state selection_partition=validation.")
    if metadata.get("metric_dataset_key") != "metric_validation":
        raise ValueError("Checkpoint metadata must state metric_dataset_key=metric_validation.")
    if metadata.get("monitor") != "val_avg_dice" or metadata.get("mode") != "max":
        raise ValueError("Final test requires a checkpoint selected by maximum validation Dice.")
    selected = metadata.get("best_model_path")
    if not selected:
        raise ValueError("Checkpoint metadata does not contain best_model_path.")
    selected_path = os.path.normcase(os.path.abspath(os.path.expanduser(selected)))
    requested_path = os.path.normcase(os.path.abspath(os.path.expanduser(checkpoint_path)))
    if selected_path != requested_path:
        raise ValueError(
            f"Requested checkpoint is not the validation-selected best: {requested_path} != {selected_path}"
        )
    return metadata


def main():
    parser = argparse.ArgumentParser()
    # saving settings
    parser.add_argument("--outdir", type=str, nargs="?", help="dir to write results to",
                        default="outputs/txt2img-samples")
    parser.add_argument("--name", type=str, help="name to call this inference", default="test")
    # sampler settings
    parser.add_argument("--sampler", type=str,
                        choices=["direct", "ddim", "plms"],
                        default="direct",
                        help="the sampler used for sampling", )
    parser.add_argument("--ddim_steps", type=int, default=1, help="number of ddim sampling steps", )
    parser.add_argument("--ddim_eta", type=float, default=0.0,
                        help="ddim eta (eta=0.0 corresponds to deterministic sampling", )
    # dataset settings
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["btcv-b", "btcv-b-val", "acdc", "acdc-val", "isic2018", "isic2018-val"],
        help="Fixed release evaluation split.",
    )
    # sampling settings
    parser.add_argument("--fixed_code", action='store_true',
                        help="if enabled, uses the same starting code across samples ", )
    parser.add_argument("--H", type=int, default=256, help="image height, in pixel space", )
    parser.add_argument("--W", type=int, default=256, help="image width, in pixel space", )
    parser.add_argument("--C", type=int, default=4, help="latent channels", )
    parser.add_argument("--f", type=int, default=8, help="downsampling factor", )
    parser.add_argument("--n_samples", type=int, default=1,
                        help="how many samples to produce for each given prompt. A.k.a. batch size", )
    parser.add_argument("--config", type=str, default="configs/stable-diffusion/v1-inference.yaml",
                        help="path to config which constructs model", )
    parser.add_argument("--ckpt", type=str, default="models/ldm/stable-diffusion-v1/model.ckpt",
                        help="path to checkpoint of model", )
    parser.add_argument("--seed", type=int, default=0,
                        help="the seed (for reproducible sampling)", )
    parser.add_argument("--times", type=int, default=1,
                        help="Must be 1 for one-time final evaluation.")
    parser.add_argument("--save_results", action='store_true',  # will slow down inference
                        help="saving the predictions for the whole test set.", )
    parser.add_argument("--max_cases", type=int, default=0,
                        help="Optional eval limit for smoke tests. 0 evaluates the full split.")
    parser.add_argument("--device", choices=["cuda"], default="cuda",
                        help="CUDA only; CPU fallback is disabled.")
    parser.add_argument("--data-root", required=True,
                        help="Parent directory containing btcv/, acdc/, and isic2018/.")
    parser.add_argument("--selection-metadata", required=True,
                        help="best_checkpoint.json proving validation-only selection.")
    parser.add_argument("--expected-gpu-name", default="NVIDIA GeForce RTX 5090")
    opt = parser.parse_args()
    default_config = parser.get_default("config")
    default_ckpt = parser.get_default("ckpt")
    default_outdir = parser.get_default("outdir")

    
    if opt.times != 1:
        raise ValueError("Final evaluation is one-time only; --times must equal 1.")
    if opt.config == default_config or opt.ckpt == default_ckpt:
        raise ValueError("Explicit --config and validation-selected --ckpt are required.")
    os.environ["SDSEG_DATA_ROOT"] = os.path.abspath(os.path.expanduser(opt.data_root))
    selection_metadata = validate_selection_metadata(opt.selection_metadata, opt.ckpt)

    if opt.dataset == "btcv-b":
        partition = "test"
        dataset = BTCVValidationEval(num_classes=2, split="test")
    elif opt.dataset == "btcv-b-val":
        partition = "validation"
        dataset = BTCVValidationEval(num_classes=2, split="validation")
    elif opt.dataset == "acdc":
        partition = "test"
        dataset = ACDCFullLabelSliceEval(num_classes=4, split="test")
    elif opt.dataset == "acdc-val":
        partition = "validation"
        dataset = ACDCFullLabelSliceEval(num_classes=4, split="validation")
    elif opt.dataset == "isic2018":
        partition = "test"
        dataset = ISIC2018Test(num_classes=2)
    elif opt.dataset == "isic2018-val":
        partition = "validation"
        dataset = ISIC2018ValidationEval(num_classes=2)
    else:
        raise AssertionError(f"Unhandled dataset choice: {opt.dataset}")

    if opt.max_cases > 0:
        max_cases = min(int(opt.max_cases), len(dataset))
        print(f"[slice2seg] Limiting evaluation to {max_cases}/{len(dataset)} cases")
        dataset = Subset(dataset, range(max_cases))

    data = DataLoader(dataset, batch_size=opt.n_samples, shuffle=False)

    config = OmegaConf.load(f"{opt.config}")
    model_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    model_config["model"]["params"].pop("ckpt_path", None)
    model_config["model"]["params"]["cond_stage_config"]["params"].pop("ckpt_path", None)
    model_config["model"]["params"]["first_stage_config"]["params"].pop("ckpt_path", None)

    model, pl_sd = load_model_from_config(model_config, f"{opt.ckpt}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is disabled.")
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    if device_name != opt.expected_gpu_name:
        raise RuntimeError(f"Expected GPU {opt.expected_gpu_name!r}, found {device_name!r}.")
    model = model.to(device)

    if os.path.exists(opt.outdir) and os.listdir(opt.outdir):
        raise FileExistsError(
            f"Evaluation output already exists and is non-empty; refusing to overwrite: {opt.outdir}"
        )
    os.makedirs(opt.outdir, exist_ok=True)

    for idx in range(opt.times):
        if opt.times > 1:   # if test only once, use specified seed.
            opt.seed = idx
        seed_everything(opt.seed)
        print(f"\033[32m seed:{opt.seed}\033[0m")
        
        outpath = os.path.join(opt.outdir, str(opt.seed))
        os.makedirs(outpath, exist_ok=True)

        metrics_dict, _, details = model.log_dice(
            data=data,
            save_dir=outpath if opt.save_results else None,
            ddim_steps=opt.ddim_steps,
            sampler_name=opt.sampler,
            ddim_eta=opt.ddim_eta,
            return_details=True,
        )

        dice_list = metrics_dict["val_avg_dice"]
        iou_list = metrics_dict["val_avg_iou"]
        print(f"\033[31m[Mean Dice][{opt.dataset}][{opt.sampler}]: {sum(dice_list) / len(dice_list)}\033[0m")
        print(f"\033[31m[Mean  IoU][{opt.dataset}][{opt.sampler}]: {sum(iou_list) / len(iou_list)}\033[0m")

        audit_mode = get_model_audit_mode(config)
        audit_condition = metric_condition_name(audit_mode)
        summary = build_eval_summary(
            metrics_dict=metrics_dict,
            details=details,
            audit_mode=audit_mode,
            checkpoint_path=opt.ckpt,
            seed=opt.seed,
            sampler=details.get("sampler_name"),
            audit_condition=audit_condition,
        )
        summary["evaluation_partition"] = partition
        summary["selection_partition"] = selection_metadata["selection_partition"]
        summary["selected_by"] = selection_metadata["monitor"]
        summary["is_smoke_test"] = bool(opt.max_cases > 0)
        write_json(os.path.join(outpath, "metrics_summary.json"), summary)
        write_csv_rows(os.path.join(outpath, "metrics_summary.csv"), [summary])
        write_per_case_metric_exports(
            outpath,
            details.get("per_case_metrics", []),
            audit_mode=audit_mode,
            checkpoint_path=opt.ckpt,
            seed=opt.seed,
            sampler=details.get("sampler_name"),
            audit_condition=audit_condition,
        )
        resolved = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        resolved.inference_seed = int(opt.seed)
        resolved.inference_checkpoint_path = os.path.abspath(str(opt.ckpt))
        resolved.inference_audit_condition = audit_condition
        resolved.inference_partition = partition
        resolved.selection_partition = selection_metadata["selection_partition"]
        resolved.selection_monitor = selection_metadata["monitor"]
        OmegaConf.save(resolved, os.path.join(outpath, "resolved_config.yaml"))

        if opt.times > 1:
            print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
            f" \nEnjoy.")


if __name__ == "__main__":
    main()
