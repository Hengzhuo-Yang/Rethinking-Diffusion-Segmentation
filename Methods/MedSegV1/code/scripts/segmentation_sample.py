

import argparse
import csv
import json
import os
from ssl import OP_NO_TLSv1
import nibabel as nib
# from visdom import Visdom
# viz = Visdom(port=8850)
import sys
import random
sys.path.append(".")
import numpy as np
import time
import torch as th
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
import torch.distributed as dist
from scipy import ndimage
from guided_diffusion import dist_util, logger
from guided_diffusion.bratsloader import BRATSDataset, BRATSDataset3D, CachedBRATSSliceDataset
from guided_diffusion.btcvloader import BTCVDataset
from guided_diffusion.isicloader import ISICDataset
from guided_diffusion.acdcloader import ACDCDataset, labels_from_foreground_channels
from guided_diffusion.custom_dataset_loader import CustomDataset
import torchvision.utils as vutils
from guided_diffusion.utils import staple
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import torchvision.transforms as transforms
from torchsummary import summary
seed=10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

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


def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img


def brats_slice_id(path_str):
    base = Path(path_str).name.split(".nii")[0]
    if "_slice" not in base:
        return base
    subject, slice_id = base.split("_slice", 1)
    parts = subject.split("_")
    if parts[-1].lower() in {"t1", "t1ce", "t1c", "t1gd", "t2", "flair", "t2flair", "seg"}:
        subject = "_".join(parts[:-1])
    return f"{subject}_slice{slice_id}"


def brats_slice_info(path_str):
    path_str = str(path_str)
    if "_slice" not in path_str:
        raise ValueError(f"cannot infer BraTS slice index from path: {path_str}")
    seg_prefix, slice_part = path_str.rsplit("_slice", 1)
    slice_idx = int(Path(slice_part).stem)
    candidates = [seg_prefix + ".nii.gz", seg_prefix + ".nii"]
    seg_path = next((candidate for candidate in candidates if os.path.exists(candidate)), candidates[0])
    case_id = Path(seg_prefix).name
    parts = case_id.split("_")
    if parts[-1].lower() in {"t1", "t1ce", "t1c", "t1gd", "t2", "flair", "t2flair", "seg"}:
        case_id = "_".join(parts[:-1])
    return case_id, slice_idx, seg_path


def prediction_batch(pred):
    pred = pred.detach().float()
    if pred.ndim == 3:
        pred = pred.unsqueeze(1)
    if pred.ndim != 4:
        raise ValueError(f"unexpected prediction batch shape: {tuple(pred.shape)}")
    return pred


def fuse_ensemble(enslist, fusion):
    stacked = th.stack(enslist, dim=0)
    if fusion == "staple":
        return prediction_batch(staple(stacked).squeeze(0))
    if fusion == "mean":
        return prediction_batch(th.mean(stacked, dim=0))
    raise ValueError(f"Unsupported ensemble fusion mode: {fusion}")


def logits_to_prediction(logits, threshold):
    if logits.ndim != 4:
        raise ValueError(f"unexpected logits shape: {tuple(logits.shape)}")
    if logits.shape[1] == 1:
        return th.sigmoid(logits)
    if logits.shape[1] == 2:
        return F.softmax(logits, dim=1)[:, 1:2, ...]
    return F.softmax(logits, dim=1)[:, 1:, ...]


def valid_acdc_prediction(path):
    if not os.path.exists(path):
        return False
    try:
        with np.load(path) as data:
            return "prob" in data and "target_mask" in data and "target_labels" in data
    except Exception:
        return False


def core_no_diff_prediction(model, image, threshold):
    module = getattr(model, "module", model)
    if not hasattr(module, "core_no_diff_logits"):
        raise AttributeError(
            "audit_mode=core_no_diff requires the model to implement core_no_diff_logits(image)."
        )
    logits = module.core_no_diff_logits(image)
    return logits_to_prediction(logits, threshold)


def extract_pred_slice(pred, batch_idx=0):
    pred = prediction_batch(pred).cpu()
    if pred.ndim == 4:
        return pred[batch_idx, 0].numpy()
    if pred.ndim == 2:
        return pred.numpy()
    raise ValueError(f"unexpected prediction shape for metric extraction: {tuple(pred.shape)}")


def resize_binary_stack_to_shape(pred_slices, target_hw):
    tensor = th.from_numpy(pred_slices.astype(np.float32))[:, None, :, :]
    tensor = F.interpolate(tensor, size=target_hw, mode="nearest")
    return tensor[:, 0].numpy() > 0.5


def binary_metrics(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(pred, target).sum(dtype=np.float64)
    pred_sum = pred.sum(dtype=np.float64)
    target_sum = target.sum(dtype=np.float64)
    denom = pred_sum + target_sum
    union = pred_sum + target_sum - intersection
    dice = 1.0 if denom == 0 else float(2.0 * intersection / denom)
    iou = 1.0 if union == 0 else float(intersection / union)
    return dice, iou


def binary_metrics_skip_empty_gt(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(pred, target).sum(dtype=np.float64)
    pred_sum = pred.sum(dtype=np.float64)
    target_sum = target.sum(dtype=np.float64)
    if target_sum == 0:
        return None, None, int(pred_sum), int(target_sum)
    denom = pred_sum + target_sum
    union = pred_sum + target_sum - intersection
    dice = float(2.0 * intersection / denom) if denom > 0 else 0.0
    iou = float(intersection / union) if union > 0 else 0.0
    return dice, iou, int(pred_sum), int(target_sum)


def mean_skip_none(values):
    valid = [value for value in values if value is not None]
    return float(np.mean(valid)) if valid else float("nan")


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def binary_dataset_label(args):
    if args.data_name in {"ISIC", "ISIC2018"}:
        return "ISIC2018"
    return "BTCV"


def binary_metric_stem(args):
    if args.data_name in {"ISIC", "ISIC2018"}:
        return "isic2018"
    return "btcv"


def write_binary_metrics(rows, args):
    if not rows:
        return
    metric_stem = binary_metric_stem(args)
    per_slice_csv = args.metrics_per_slice_csv or args.metrics_out_csv or os.path.join(
        args.out_dir, f"{metric_stem}_metrics_per_slice.csv"
    )
    summary_csv = args.metrics_summary_csv or os.path.join(args.out_dir, f"{metric_stem}_metrics_summary.csv")
    summary_json = args.metrics_summary_json or os.path.join(args.out_dir, f"{metric_stem}_metrics_summary.json")
    nonempty_rows = [row for row in rows if not row["is_empty_gt"]]
    summary = {
        "dataset": binary_dataset_label(args),
        "task": "binary_foreground_segmentation",
        "num_slices": len(rows),
        "num_nonempty_gt": len(nonempty_rows),
        "num_empty_gt": len(rows) - len(nonempty_rows),
        "num_empty_pred": sum(row["is_empty_pred"] for row in rows),
        "dice": mean_skip_none([row["dice"] for row in rows]),
        "iou": mean_skip_none([row["iou"] for row in rows]),
        "dice_nonempty_gt": mean_skip_none([row["dice"] for row in nonempty_rows]),
        "iou_nonempty_gt": mean_skip_none([row["iou"] for row in nonempty_rows]),
        "pred_threshold": args.pred_threshold,
        "diffusion_steps": args.diffusion_steps,
        "sample_steps": args.sample_steps,
        "num_ensemble": args.num_ensemble,
        "ensemble_fusion": args.ensemble_fusion,
        "metric_policy": "skip_empty_ground_truth_slices",
        "per_slice_csv": per_slice_csv,
    }
    write_csv(per_slice_csv, rows, list(rows[0].keys()))
    write_csv(summary_csv, [summary], list(summary.keys()))
    os.makedirs(os.path.dirname(summary_json) or ".", exist_ok=True)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    logger.log(
        f"{summary['dataset']} metrics "
        f"slices={summary['num_slices']} "
        f"nonempty_gt={summary['num_nonempty_gt']} "
        f"dice={summary['dice_nonempty_gt']:.4f} "
        f"iou={summary['iou_nonempty_gt']:.4f}"
    )


def write_btcv_metrics(rows, args):
    write_binary_metrics(rows, args)


def save_binary_prediction_png(pred, path, threshold):
    out = pred.detach().cpu().clone()
    out = (out.squeeze().numpy() > threshold).astype(np.uint8) * 255
    Image.fromarray(out, mode="L").save(path)


def save_btcv_prediction_png(pred, path, threshold):
    save_binary_prediction_png(pred, path, threshold)


def surface(mask):
    if not np.any(mask):
        return mask.astype(bool)
    structure = ndimage.generate_binary_structure(mask.ndim, 1)
    eroded = ndimage.binary_erosion(mask, structure=structure, border_value=0)
    return np.logical_xor(mask, eroded)


def hd95_mm(pred, target, spacing):
    pred = pred.astype(bool)
    target = target.astype(bool)
    pred_empty = not np.any(pred)
    target_empty = not np.any(target)
    if pred_empty and target_empty:
        return 0.0
    if pred_empty or target_empty:
        return float(np.linalg.norm((np.array(pred.shape, dtype=np.float64) - 1.0) * np.array(spacing)))

    pred_surface = surface(pred)
    target_surface = surface(target)
    target_dt = ndimage.distance_transform_edt(~target_surface, sampling=spacing)
    pred_dt = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)
    distances = np.concatenate([target_dt[pred_surface], pred_dt[target_surface]])
    if distances.size == 0:
        return 0.0
    return float(np.percentile(distances, 95))


def write_brats_wt_metrics(case_predictions, args):
    if not case_predictions:
        return

    cache_cases = {}
    if args.brats_cache_dir:
        metadata_path = Path(args.brats_cache_dir) / "metadata.json"
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
            cache_cases = {case["case_id"]: case for case in metadata.get("cases", [])}

    out_csv = args.metrics_out_csv or os.path.join(args.out_dir, "brats_wt_metrics.csv")
    rows = []
    skipped = []
    for case_id in sorted(case_predictions):
        payload = case_predictions[case_id]
        cache_case = cache_cases.get(case_id, {})
        label_orig_file = cache_case.get("label_orig_file")
        if label_orig_file:
            target_zyx = np.load(Path(args.brats_cache_dir) / label_orig_file) > 0
            spacing_zyx = tuple(float(v) for v in cache_case["spacing_zyx"])
            metric_source = "label_orig_npy"
        else:
            seg_img = nib.load(payload["seg_path"])
            target_xyz = seg_img.get_fdata(dtype=np.float32) > 0
            zooms = seg_img.header.get_zooms()[:3]
            target_zyx = np.transpose(target_xyz, (2, 0, 1))
            spacing_zyx = (float(zooms[2]), float(zooms[0]), float(zooms[1]))
            metric_source = "seg_nii"
        expected_slices = target_zyx.shape[0]
        pred_slice_map = payload["pred_slices"]
        missing = sorted(set(range(expected_slices)) - set(pred_slice_map))
        if missing:
            skipped.append({"case_id": case_id, "missing_slices": len(missing)})
            continue

        pred_slices = np.stack([pred_slice_map[idx] for idx in range(expected_slices)], axis=0)
        pred_zyx = resize_binary_stack_to_shape(pred_slices, target_zyx.shape[1:])
        dice, iou = binary_metrics(pred_zyx, target_zyx)
        hd95 = hd95_mm(pred_zyx, target_zyx, spacing_zyx)
        rows.append(
            {
                "case_id": case_id,
                "num_slices": expected_slices,
                "dice_wt": dice,
                "iou_wt": iou,
                "hd95_wt_mm": hd95,
                "pred_voxels": int(pred_zyx.sum()),
                "target_voxels": int(target_zyx.sum()),
                "metric_source": metric_source,
                "seg_path": payload["seg_path"],
            }
        )

    fieldnames = [
        "case_id",
        "num_slices",
        "dice_wt",
        "iou_wt",
        "hd95_wt_mm",
        "pred_voxels",
        "target_voxels",
        "metric_source",
        "seg_path",
    ]
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "metric_region": "WT",
        "label_mapping": "WT = seg > 0",
        "dice_wt_mean": float(np.mean([row["dice_wt"] for row in rows])) if rows else float("nan"),
        "iou_wt_mean": float(np.mean([row["iou_wt"] for row in rows])) if rows else float("nan"),
        "hd95_wt_mm_mean": float(np.mean([row["hd95_wt_mm"] for row in rows])) if rows else float("nan"),
        "num_cases": len(rows),
        "skipped_partial_cases": skipped,
        "metrics_csv": out_csv,
        "metric_source": "label_orig_npy" if any(row["metric_source"] == "label_orig_npy" for row in rows) else "seg_nii",
    }
    summary_path = args.metrics_summary_json or os.path.splitext(out_csv)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.log(
        "WT metrics "
        f"cases={summary['num_cases']} "
        f"dice={summary['dice_wt_mean']:.4f} "
        f"iou={summary['iou_wt_mean']:.4f} "
        f"hd95_mm={summary['hd95_wt_mm_mean']:.4f}"
    )
    if skipped:
        logger.log(f"skipped {len(skipped)} partial BraTS cases when writing metrics")


def main():
    args = create_argparser().parse_args()
    args.audit_mode = normalize_audit_mode(args.audit_mode)
    if args.version != "new" and args.audit_mode != "core_no_diff":
        selected_output = (
            args.acdc_output if args.data_name == "ACDC" else args.binary_output
        )
        if selected_output != "sample":
            raise ValueError(
                "MedSegDiff V1 has no localization/calibration mask output. "
                "Use sample-only inference (--acdc_output sample for ACDC, "
                "or --binary_output sample for other datasets)."
            )
    dist_util.setup_dist(args)
    logger.configure(dir = args.out_dir)

    if args.data_name in {'ISIC', 'ISIC2018'}:
        ds = ISICDataset(args, args.data_dir, mode='Test', image_size=args.image_size)
        args.in_ch = 4
    elif args.data_name == 'BRATS':
        tran_list = [transforms.Resize((args.image_size,args.image_size)),]
        transform_test = transforms.Compose(tran_list)

        if args.brats_cache_dir:
            ds = CachedBRATSSliceDataset(args.brats_cache_dir, test_flag=False)
        else:
            ds = BRATSDataset3D(args.data_dir,transform_test)
        args.in_ch = 5
    elif args.data_name == 'BTCV':
        ds = BTCVDataset(args.data_dir, image_size=args.image_size, mode='Test')
        args.in_ch = 4
    elif args.data_name == 'ACDC':
        if int(args.num_seg_classes) != 4 or int(args.num_mask_channels) != 3:
            raise ValueError("ACDC multi-class path expects --num_seg_classes 4 and --num_mask_channels 3.")
        ds = ACDCDataset(args, args.data_dir, mode='Test')
        args.in_ch = 6
    else:
        tran_list = [transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor()]
        transform_test = transforms.Compose(tran_list)

        ds = CustomDataset(args, args.data_dir, transform_test, mode = 'Test')
        args.in_ch = 4

    datal = th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers)

    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    all_images = []


    state_dict = dist_util.load_state_dict(args.model_path, map_location="cpu")
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # name = k[7:] # remove `module.`
        if 'module.' in k:
            new_state_dict[k[7:]] = v
            # load params
        else:
            new_state_dict = state_dict

    model.load_state_dict(new_state_dict)

    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    sample_steps = args.sample_steps if args.sample_steps > 0 else args.diffusion_steps
    logger.log(f"sampling with diffusion_steps={args.diffusion_steps} sample_steps={sample_steps}")
    logger.log(f"audit_mode={args.audit_mode}")
    if args.audit_mode == "core_no_diff":
        logger.log("core_no_diff sampling uses direct image-only logits; reverse diffusion sampling is skipped.")
    dataset_size = len(ds)
    total_samples = dataset_size if args.num_samples < 0 else min(args.num_samples, dataset_size)
    total_batches = (total_samples + args.batch_size - 1) // args.batch_size
    processed_samples = 0
    brats_metrics = {}
    binary_rows = []
    for batch_idx, (b, m, path) in enumerate(datal):
        if processed_samples >= total_samples:
            break
        remaining = total_samples - processed_samples
        if b.shape[0] > remaining:
            b = b[:remaining]
            m = m[:remaining]
            path = path[:remaining]
        paths = list(path)
        current_batch = b.shape[0]
        c = th.randn(
            b.shape[0],
            int(args.num_mask_channels),
            args.image_size,
            args.image_size,
            dtype=b.dtype,
        )
        img = th.cat((b, c), dim=1)     #add a noise channel$

        logger.log(
            f"sampling batch {batch_idx + 1}/{total_batches} "
            f"slices={processed_samples + 1}-{processed_samples + current_batch}/{total_samples}"
        )

        if args.audit_mode == "core_no_diff":
            with th.no_grad():
                image = b.to(device=dist_util.dev(), dtype=th.float32)
                ensres = prediction_batch(core_no_diff_prediction(model, image, args.pred_threshold).detach())
        else:
            start = th.cuda.Event(enable_timing=True)
            end = th.cuda.Event(enable_timing=True)
            enslist = []

            for i in range(args.num_ensemble):  #this is for the generation of an ensemble of 5 masks.
                model_kwargs = {}
                start.record()
                sample_fn = (
                    diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
                )
                sample, x_noisy, org, cal, cal_out = sample_fn(
                    model,
                    (current_batch, img.shape[1], args.image_size, args.image_size), img,
                    step = sample_steps,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                )

                end.record()
                th.cuda.synchronize()
                print('time for 1 sample', start.elapsed_time(end))  #time measurement for the generation of 1 sample

                co = cal_out.detach() if th.is_tensor(cal_out) else None
                if args.data_name == 'ACDC':
                    if args.acdc_output == "sample":
                        enslist.append(prediction_batch(sample))
                    elif args.acdc_output == "calout":
                        enslist.append(prediction_batch(co))
                    elif args.acdc_output == "cal":
                        enslist.append(prediction_batch(cal))
                    else:
                        raise ValueError(
                            f"Unknown --acdc_output '{args.acdc_output}'. Use sample, calout, or cal."
                        )
                elif args.version == 'new':
                    if args.data_name == 'ACDC':
                        enslist.append(prediction_batch(sample))
                    else:
                        enslist.append(sample[:,-1:,:,:])
                else:
                    if args.binary_output == "sample":
                        enslist.append(prediction_batch(sample))
                    elif args.binary_output == "calout":
                        enslist.append(prediction_batch(co))
                    elif args.binary_output == "cal":
                        enslist.append(prediction_batch(cal))
                    else:
                        raise ValueError(
                            f"Unknown --binary_output '{args.binary_output}'. Use sample, calout, or cal."
                        )

                if args.debug:
                    # print('sample size is',sample.size())
                    # print('org size is',org.size())
                    # print('cal size is',cal.size())
                    if args.data_name in {'ISIC', 'ISIC2018'}:
                        # s = th.tensor(sample)[:,-1,:,:].unsqueeze(1).repeat(1, 3, 1, 1)
                        o = th.tensor(org)[:,:-1,:,:]
                        # co = co.repeat(1, 3, 1, 1)

                        s = sample[:,-1,:,:]
                        batch_len, h, w = s.size()
                        ss = s.clone()
                        ss = ss.view(s.size(0), -1)
                        ss -= ss.min(1, keepdim=True)[0]
                        ss /= ss.max(1, keepdim=True)[0]
                        ss = ss.view(batch_len, h, w)
                        ss = ss.unsqueeze(1).repeat(1, 3, 1, 1)

                        if cal is None:
                            tup = (ss, o)
                        else:
                            c = th.tensor(cal).repeat(1, 3, 1, 1)
                            tup = (ss, o, c)
                    elif args.data_name == 'BRATS':
                        s = th.tensor(sample)[:,-1,:,:].unsqueeze(1)
                        m = th.tensor(m.to(device = 'cuda:0'))[:,0,:,:].unsqueeze(1)
                        o1 = th.tensor(org)[:,0,:,:].unsqueeze(1)
                        o2 = th.tensor(org)[:,1,:,:].unsqueeze(1)
                        o3 = th.tensor(org)[:,2,:,:].unsqueeze(1)
                        o4 = th.tensor(org)[:,3,:,:].unsqueeze(1)
                        if cal is None:
                            tup = (o1/o1.max(),o2/o2.max(),o3/o3.max(),o4/o4.max(),m,s)
                        else:
                            c = th.tensor(cal)
                            tup = (o1/o1.max(),o2/o2.max(),o3/o3.max(),o4/o4.max(),m,s,c,co)
                    for item_idx, path_str in enumerate(paths):
                        if args.data_name in {'ISIC', 'ISIC2018'}:
                            slice_ID = Path(path_str).stem
                        elif args.data_name == 'BRATS':
                            slice_ID = brats_slice_id(path_str)
                        elif args.data_name == 'ACDC':
                            slice_ID = str(path_str)
                        else:
                            slice_ID = Path(path_str).stem
                        compose = th.cat([component[item_idx:item_idx + 1] for component in tup], 0)
                        vutils.save_image(compose, fp = os.path.join(args.out_dir, str(slice_ID)+'_output'+str(i)+".jpg"), nrow = 1, padding = 10)
            ensres = fuse_ensemble(enslist, args.ensemble_fusion)
        for item_idx, path_str in enumerate(paths):
            if args.data_name in {'ISIC', 'ISIC2018'}:
                slice_ID = Path(path_str).stem
            elif args.data_name == 'BRATS':
                slice_ID = brats_slice_id(path_str)
            elif args.data_name == 'ACDC':
                slice_ID = str(path_str)
            else:
                slice_ID = Path(path_str).stem
            vutils.save_image(ensres[item_idx:item_idx + 1], fp = os.path.join(args.out_dir, str(slice_ID)+'_output_ens'+".jpg"), nrow = 1, padding = 10)
            if args.data_name in {'BTCV', 'ISIC', 'ISIC2018'}:
                save_binary_prediction_png(
                    ensres[item_idx:item_idx + 1],
                    os.path.join(args.out_dir, str(slice_ID) + "_pred.png"),
                    args.pred_threshold,
                )
            if args.data_name == 'ACDC':
                prob_tensor = ensres[item_idx:item_idx + 1].detach().cpu().clone().squeeze(0)
                target_tensor = m[item_idx:item_idx + 1].detach().cpu().clone().squeeze(0)
                prob = prob_tensor.numpy().astype(np.float32)
                target_mask = target_tensor.numpy().astype(np.float32)
                pred_labels = labels_from_foreground_channels(
                    prob,
                    num_classes=args.num_seg_classes,
                    threshold=args.acdc_threshold,
                )
                target_labels = labels_from_foreground_channels(
                    target_mask,
                    num_classes=args.num_seg_classes,
                    threshold=args.acdc_threshold,
                )
                np.savez_compressed(
                    os.path.join(args.out_dir, str(slice_ID) + "_pred.npz"),
                    prob=prob,
                    pred_labels=pred_labels.astype(np.uint8),
                    target_mask=target_mask,
                    target_labels=target_labels.astype(np.uint8),
                    output=str("core_direct" if args.audit_mode == "core_no_diff" else args.acdc_output),
                    source=str(slice_ID),
                    num_classes=int(args.num_seg_classes),
                    num_mask_channels=int(args.num_mask_channels),
                )
        if args.compute_metrics and args.data_name == 'BRATS':
            for item_idx, path_str in enumerate(paths):
                case_id, slice_idx, seg_path = brats_slice_info(path_str)
                case_payload = brats_metrics.setdefault(case_id, {"seg_path": seg_path, "pred_slices": {}})
                case_payload["pred_slices"][slice_idx] = extract_pred_slice(ensres, item_idx) > args.pred_threshold
        if args.compute_metrics and args.data_name in {'BTCV', 'ISIC', 'ISIC2018'}:
            for item_idx, path_str in enumerate(paths):
                pred_np = extract_pred_slice(ensres, item_idx) > args.pred_threshold
                target_np = m[item_idx, 0].detach().cpu().numpy() > 0.5
                dice, iou, pred_pixels, gt_pixels = binary_metrics_skip_empty_gt(pred_np, target_np)
                slice_id = Path(path_str).stem
                binary_rows.append(
                    {
                        "slice_id": slice_id,
                        "image": str(path_str),
                        "prediction": os.path.join(args.out_dir, str(slice_id) + "_pred.png"),
                        "dice": dice,
                        "iou": iou,
                        "pred_pixels": pred_pixels,
                        "gt_pixels": gt_pixels,
                        "is_empty_gt": int(gt_pixels == 0),
                        "is_empty_pred": int(pred_pixels == 0),
                    }
                )
        processed_samples += current_batch

    if args.compute_metrics and args.data_name == 'BRATS':
        write_brats_wt_metrics(brats_metrics, args)
    if args.compute_metrics and args.data_name in {'BTCV', 'ISIC', 'ISIC2018'}:
        write_binary_metrics(binary_rows, args)

def create_argparser():
    defaults = dict(
        data_name = 'BRATS',
        data_dir="../dataset/brats2020/testing",
        brats_cache_dir="",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="",         #path to pretrain model
        num_ensemble=5,      #number of samples in the ensemble
        ensemble_fusion="staple",  # staple keeps the original repo behavior; mean disables staple.
        sample_steps=0,      #sampling steps; 0 means use diffusion_steps
        gpu_dev = "0",
        out_dir='./results/',
        multi_gpu = None, #"0,1,2"
        debug = False,
        compute_metrics=True,
        metrics_out_csv="",
        metrics_summary_csv="",
        metrics_summary_json="",
        metrics_per_slice_csv="",
        pred_threshold=0.5,
        num_workers=0,
        audit_mode="none",
        num_seg_classes=2,
        num_mask_channels=1,
        acdc_split="testing",
        acdc_skip_empty=False,
        acdc_threshold=0.5,
        acdc_output="sample",
        binary_output="sample",
        acdc_skip_existing=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":

    main()



