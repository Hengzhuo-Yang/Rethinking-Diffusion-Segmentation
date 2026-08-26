"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import json
import os
from pathlib import Path
import nibabel as nib
import sys
import random
CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
import numpy as np
import time
import torch as th
import torch.distributed as dist
from guided_diffusion import dist_util, logger
from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion.btcvloader import BTCVDataset
from guided_diffusion.acdcloader import ACDCDataset
from guided_diffusion.isicloader import ISIC2018Dataset
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion.gaussian_diffusion import direct_segmentation_prediction
from guided_diffusion.visdom_util import make_visdom
viz = make_visdom(port=8850)


def _artifact_smoke_enabled():
    return os.environ.get("ENSEMDIFF_ARTIFACT_SMOKE_CHECK", "").lower() in {
        "1",
        "true",
        "yes",
    }


def _save_sample_artifact(pred_batch, sample_idx, path, smoke_records, smoke_limit):
    out = pred_batch[sample_idx : sample_idx + 1].detach().cpu().clone()
    th.save(out, path)
    if (
        not _artifact_smoke_enabled()
        or len(smoke_records) >= smoke_limit
        or pred_batch.shape[0] <= 1
    ):
        return

    loaded = th.load(path, map_location="cpu")
    if isinstance(loaded, np.ndarray):
        loaded = th.as_tensor(loaded)
    if not th.is_tensor(loaded):
        raise TypeError(f"Artifact smoke check expected tensor at {path}, got {type(loaded)}")
    if tuple(loaded.shape) != tuple(out.shape):
        raise RuntimeError(
            f"Artifact smoke check shape mismatch at {path}: "
            f"loaded={tuple(loaded.shape)} expected={tuple(out.shape)}"
        )
    if not th.equal(loaded.cpu(), out.cpu()):
        raise RuntimeError(f"Artifact smoke check value mismatch at {path}")

    file_size = os.path.getsize(path)
    artifact_bytes = out.element_size() * out.numel()
    batch_bytes = pred_batch.element_size() * pred_batch.numel()
    if batch_bytes > artifact_bytes and file_size >= batch_bytes * 0.5:
        raise RuntimeError(
            f"Artifact smoke check file is too large at {path}: "
            f"file_size={file_size}, artifact_bytes={artifact_bytes}, "
            f"batch_bytes={batch_bytes}"
        )
    smoke_records.append(
        {
            "path": str(path),
            "shape": list(out.shape),
            "dtype": str(out.dtype),
            "file_size": int(file_size),
            "artifact_tensor_bytes": int(artifact_bytes),
            "batch_tensor_bytes": int(batch_bytes),
            "values_match": True,
        }
    )

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


def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

def dice_score(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()


def create_dataset(args, test_flag):
    dataset = args.dataset.lower()
    if dataset in ("brats", "brats2020"):
        return BRATSDataset(args.data_dir, test_flag=test_flag)
    if dataset in ("btcv", "synapse"):
        return BTCVDataset(
            args.data_dir, test_flag=test_flag, manifest_path=args.manifest
        )
    if dataset == "acdc":
        return ACDCDataset(
            args.data_dir,
            test_flag=test_flag,
            num_classes=getattr(args, "num_seg_classes", 4),
            manifest_path=args.manifest,
        )
    if dataset in ("isic", "isic2018"):
        return ISIC2018Dataset(
            args.data_dir, test_flag=test_flag, manifest_path=args.manifest
        )
    raise ValueError(f"Unknown dataset: {args.dataset}")


def main():
    args = create_argparser().parse_args()
    if not args.model_path:
        raise ValueError("--model_path must explicitly name the validation-selected checkpoint.")
    if (
        args.dataset.lower() in ("btcv", "synapse", "acdc", "isic", "isic2018")
        and not args.manifest
    ):
        raise ValueError("--manifest is required for BTCV, ACDC, and ISIC2018.")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative for reproducible final testing.")
    dist_util.setup_dist()
    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    th.cuda.manual_seed_all(args.seed)
    apply_dataset_defaults(args)
    args.audit_mode = args.audit_mode.lower()
    if args.audit_mode not in (
        "none",
        "train_random_yt",
        "train_shuffle_yt",
        "core_no_diff",
    ):
        raise ValueError(f"Unknown audit_mode: {args.audit_mode}")
    args.core_no_diff = args.audit_mode == "core_no_diff"
    logger.configure(dir=os.getenv("OPENAI_LOGDIR", "./results"))
    os.makedirs(args.output_dir, exist_ok=True)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    ds = create_dataset(args, test_flag=True)
    if args.num_samples <= 0:
        args.num_samples = len(ds)
    datal = th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    data = iter(datal)
    all_images = []
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    if args.core_no_diff:
        metadata = {
            "audit_mode": "core_no_diff",
            "counterfactual_type": "discriminative_capacity",
            "objective_preserving": False,
            "model_path": args.model_path,
            "main_core_input": "image_only",
            "uses_Y_t_in_main_core": False,
            "uses_timestep_in_main_core": False,
            "uses_q_sample_for_main_core": False,
            "uses_reverse_sampler_for_main_core": False,
            "diffusion_loss_used_for_main_core": False,
            "sampling_steps_main_core": 0,
            "gpu_memory_allocated_bytes_at_start": th.cuda.memory_allocated(),
            "gpu_memory_reserved_bytes_at_start": th.cuda.memory_reserved(),
            "num_ensemble_requested": args.num_ensemble,
            "num_ensemble_saved_for_evaluator": max(args.num_ensemble, 1),
            "sampling_note": (
                "Direct image-only prediction is duplicated across output indices "
                "only to preserve the existing evaluator file contract."
            ),
        }
        metadata_path = Path(args.output_dir) / "core_no_diff_sample_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        logger.log(f"saved core_no_diff sample metadata: {metadata_path}")
        logger.log(
            "core_no_diff sampling: using a single image-only forward pass; "
            "DDPM/DDIM reverse sampling is disabled"
        )
    samples_done = 0
    artifact_smoke_records = []
    artifact_smoke_limit = int(os.environ.get("ENSEMDIFF_ARTIFACT_SMOKE_LIMIT", "4"))
    while samples_done < args.num_samples:
        try:
            b, path = next(data)  #should return images from the dataloader "data"
        except StopIteration:
            break
        paths = list(path)
        if args.core_no_diff:
            logger.log("core_no_diff direct inference...")
            start_time = time.time()
            with th.no_grad():
                logits = model(b.to(dist_util.dev()))
                pred = direct_segmentation_prediction(logits).cpu()
            elapsed = time.time() - start_time
            logger.log(
                f"core_no_diff batch inference_sec={elapsed:.4f} "
                f"per_case_sec={elapsed / max(b.shape[0], 1):.6f}"
            )
            for i in range(max(args.num_ensemble, 1)):
                for sample_idx, sample_path in enumerate(paths):
                    slice_ID = Path(sample_path).parent.name
                    _save_sample_artifact(
                        pred,
                        sample_idx,
                        os.path.join(args.output_dir, str(slice_ID)+'_output'+str(i)),
                        artifact_smoke_records,
                        artifact_smoke_limit,
                    )
            samples_done += b.shape[0]
            continue
        c = th.randn(
            b.shape[0],
            args.mask_channels,
            b.shape[-2],
            b.shape[-1],
            dtype=b.dtype,
        )
        img = th.cat((b, c), dim=1)     #add a noise channel$

        for channel in range(img.shape[1]):
            viz.image(
                visualize(img[0, channel, ...]),
                opts=dict(caption="img input" + str(channel)),
            )

        logger.log("sampling...")

        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)


        for i in range(args.num_ensemble):  #this is for the generation of an ensemble of 5 masks.
            model_kwargs = {}
            start.record()
            sample_fn = (
                diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
            )
            sample, x_noisy, org = sample_fn(
                model,
                (
                    b.shape[0],
                    args.image_channels + args.mask_channels,
                    b.shape[-2],
                    b.shape[-1],
                ),
                img,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

            end.record()
            th.cuda.synchronize()
            elapsed_ms = start.elapsed_time(end)
            logger.log(f"sampling_time_ms = {elapsed_ms:.3f}")

            s = th.as_tensor(sample).cpu()
            viz.image(visualize(sample[0, 0, ...]), opts=dict(caption="sampled output"))
            for sample_idx, sample_path in enumerate(paths):
                slice_ID = Path(sample_path).parent.name
                _save_sample_artifact(
                    s,
                    sample_idx,
                    os.path.join(args.output_dir, str(slice_ID)+'_output'+str(i)),
                    artifact_smoke_records,
                    artifact_smoke_limit,
                ) #save the generated mask
        samples_done += b.shape[0]
    if artifact_smoke_records:
        smoke_path = Path(args.output_dir) / "artifact_smoke_check.json"
        smoke_path.write_text(json.dumps(artifact_smoke_records, indent=2), encoding="utf-8")
        logger.log(f"saved artifact smoke check: {smoke_path}")

def create_argparser():
    defaults = dict(
        data_dir="./data/testing",
        dataset="brats",
        manifest="",
        clip_denoised=True,
        num_samples=0,
        batch_size=1,
        num_workers=0,
        use_ddim=False,
        model_path="",
        num_ensemble=5,      #number of samples in the ensemble
        output_dir="./results",
        audit_mode="none",
        seed=10,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":

    main()
