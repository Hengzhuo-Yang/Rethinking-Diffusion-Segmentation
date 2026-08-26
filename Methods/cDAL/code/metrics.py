import csv
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from PIL import Image
from sklearn.metrics import f1_score, jaccard_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import is_dist_avail_and_initialized


DEFAULT_CLASS_NAMES = {
    1: "foreground",
    2: "class2",
    3: "class3",
}


def calculate_metrics(x, gt):
    predict = x.detach().cpu().numpy().astype("uint8")
    target = gt.detach().cpu().numpy().astype("uint8")
    return f1_score(target.flatten(), predict.flatten()), jaccard_score(target.flatten(), predict.flatten())


def calculate_binary_foreground_metrics(x, gt, skip_empty_gt=False):
    pred = x.detach().bool()
    target = gt.detach().bool()
    intersection = torch.logical_and(pred, target).sum().item()
    pred_sum = pred.sum().item()
    target_sum = target.sum().item()
    if target_sum == 0 and skip_empty_gt:
        return None, None, int(pred_sum), int(target_sum)
    dice_den = pred_sum + target_sum
    iou_den = pred_sum + target_sum - intersection
    dice = (2.0 * intersection) / dice_den if dice_den > 0 else 0.0
    iou = intersection / iou_den if iou_den > 0 else 0.0
    return dice, iou, int(pred_sum), int(target_sum)


def labels_from_segmentation_tensor(mask, num_classes, threshold=0.5):
    """Return class-index labels shaped [B, H, W].

    cDAL multi-class ACDC stores C-1 foreground channels, where all-zero means
    background. Binary tensors keep the historical threshold behavior.
    """
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim == 3:
        if num_classes > 2 and mask.shape[0] == num_classes - 1:
            mask = mask.unsqueeze(0)
        elif mask.shape[0] == 1:
            mask = mask.unsqueeze(0)
        else:
            return mask.round().long().clamp(min=0, max=num_classes - 1)
    if mask.ndim != 4:
        raise ValueError(f"Expected segmentation tensor with 2, 3, or 4 dims, got {tuple(mask.shape)}")

    if mask.shape[1] == 1:
        channel = mask[:, 0]
        if num_classes == 2:
            if channel.dtype.is_floating_point:
                return (channel > threshold).long()
            return (channel > 0).long()
        return channel.round().long().clamp(min=0, max=num_classes - 1)
    if mask.shape[1] == num_classes:
        return torch.argmax(mask, dim=1).long()
    if num_classes > 2 and mask.shape[1] == num_classes - 1:
        foreground_score, foreground_index = torch.max(mask, dim=1)
        background = torch.zeros_like(foreground_index)
        return torch.where(foreground_score > threshold, foreground_index.long() + 1, background.long())
    if num_classes == 2:
        channel = mask.mean(dim=1)
        return (channel > threshold).long()
    raise ValueError(
        f"Expected one label channel, {num_classes} class channels, or "
        f"{num_classes - 1} foreground channels, got shape {tuple(mask.shape)}"
    )


def dice_iou_for_label_class(pred_labels, target_labels, class_id):
    pred = pred_labels == class_id
    target = target_labels == class_id
    intersection = torch.logical_and(pred, target).sum().item()
    pred_sum = pred.sum().item()
    target_sum = target.sum().item()
    if target_sum == 0:
        return None, None, int(pred_sum), int(target_sum)
    dice_den = pred_sum + target_sum
    iou_den = pred_sum + target_sum - intersection
    dice = (2.0 * intersection) / dice_den if dice_den > 0 else 0.0
    iou = intersection / iou_den if iou_den > 0 else 0.0
    return dice, iou, int(pred_sum), int(target_sum)


def calculate_multiclass_foreground_metrics(pred, gt, num_classes, threshold=0.5):
    pred_labels = labels_from_segmentation_tensor(pred, num_classes, threshold=threshold)
    gt_labels = labels_from_segmentation_tensor(gt, num_classes, threshold=threshold)
    if pred_labels.shape != gt_labels.shape:
        raise ValueError(f"Prediction and GT label shapes differ: {tuple(pred_labels.shape)} vs {tuple(gt_labels.shape)}")
    dice_values = []
    iou_values = []
    rows = []
    for class_id in range(1, num_classes):
        dice, iou, pred_pixels, gt_pixels = dice_iou_for_label_class(pred_labels, gt_labels, class_id)
        if dice is not None:
            dice_values.append(dice)
            iou_values.append(iou)
        rows.append(
            {
                "class_id": class_id,
                "dice": dice,
                "miou": iou,
                "pred_pixels": pred_pixels,
                "gt_pixels": gt_pixels,
                "metric_included": int(dice is not None),
            }
        )
    return mean_or_zero(dice_values), mean_or_zero(iou_values), rows


def mean_or_zero(values):
    valid = [float(value) for value in values if value is not None]
    return float(sum(valid) / len(valid)) if valid else 0.0


def write_rows_csv(path, rows, fieldnames, append=True):
    if not path or not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not append or not path.exists()
    with path.open("a" if append else "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def set_random_seed_for_iterations(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def _foreground_positive(dataset):
    return dataset.__class__.__name__ == "MonuDataset" or getattr(dataset, "foreground_positive", False)


def _skip_empty_gt(dataset, args):
    if hasattr(args, "metric_skip_empty_gt"):
        return bool(args.metric_skip_empty_gt)
    return getattr(dataset, "skip_empty_gt_for_metrics", False)


def _world_size():
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def _distributed_mean(values, device):
    local = [float(value) for value in values if value is not None]
    if not is_dist_avail_and_initialized():
        return mean_or_zero(local), len(local)
    my_length = len(local)
    length_tensor = torch.tensor(my_length, device=device)
    gathered_lengths = [torch.tensor(0, device=device) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_lengths, length_tensor)
    max_len = int(torch.max(torch.stack(gathered_lengths)).item())
    padded = torch.tensor(local + [-1.0] * (max_len - my_length), device=device)
    gathered = [torch.ones_like(padded) * -1 for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, padded)
    gathered = torch.cat(gathered)
    gathered = gathered[gathered != -1]
    if gathered.numel() == 0:
        return 0.0, 0
    return gathered.mean().item(), int(gathered.numel())


def _save_label_png(label, path):
    array = label.detach().cpu().clone().squeeze().numpy().astype(np.uint8)
    Image.fromarray(array, mode="L").save(path)


def _class_names(dataset, num_classes):
    names = getattr(dataset, "class_names", DEFAULT_CLASS_NAMES)
    return {class_id: names.get(class_id, f"class{class_id}") for class_id in range(1, num_classes)}


def _multiclass_fieldnames(num_classes):
    fields = [
        "step",
        "slice_id",
        "dice",
        "miou",
        "num_nonempty_gt_classes",
        "is_empty_gt_foreground",
        "is_empty_pred_foreground",
    ]
    for class_id in range(1, num_classes):
        fields.extend(
            [
                f"class{class_id}_name",
                f"class{class_id}_dice",
                f"class{class_id}_miou",
                f"class{class_id}_pred_pixels",
                f"class{class_id}_gt_pixels",
                f"class{class_id}_metric_included",
            ]
        )
    return fields


def sampling_major_vote_func(
    pos_coeff,
    sample_from_model,
    netG,
    output_folder,
    dataset,
    logger,
    step,
    args,
    device,
    metrics_csv=None,
    summary_json=None,
):
    is_monu_dataset = dataset.__class__.__name__ == "MonuDataset"
    if is_monu_dataset:
        batch_size = 1
        major_vote_number = 5
    else:
        batch_size = max(1, int(getattr(args, "eval_batch_size", 1)))
        major_vote_number = int(getattr(args, "major_vote_number", 30))
    max_items = max(0, int(getattr(args, "eval_max_items", 0)))
    loader_dataset = dataset
    if max_items > 0:
        loader_dataset = torch.utils.data.Subset(dataset, range(min(max_items, len(dataset))))
    loader = DataLoader(loader_dataset, batch_size=batch_size)
    loader_iter = iter(loader)
    n_rounds = len(loader)

    f1_score_list = []
    miou_list = []
    rows = []
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    foreground_positive = _foreground_positive(dataset)
    skip_empty_gt = _skip_empty_gt(dataset, args)
    num_classes = int(getattr(dataset, "num_seg_classes", 2))
    is_multiclass = num_classes > 2
    class_names = _class_names(dataset, num_classes)
    
    save_visuals = bool(getattr(args, "save_visuals", False))
    model_parameter_device = str(next(netG.parameters()).device)
    if device.type != "cuda" or not model_parameter_device.startswith("cuda:"):
        raise RuntimeError("Validation/test model execution must remain on CUDA")
    input_device = ""
    output_device = ""

    with torch.no_grad():
        for _ in tqdm(range(n_rounds), desc="Generating image samples for Dice and mIoU evaluation."):
            gt_mask, condition_on, name = next(loader_iter)
            set_random_seed_for_iterations(major_vote_number)
            gt_mask = (gt_mask + 1.0) / 2.0
            condition_on = condition_on["conditioned_image"]
            former_frame_for_feature_extraction = condition_on.to(device)
            gt_mask = gt_mask.to(device)
            if former_frame_for_feature_extraction.device.type != "cuda" or gt_mask.device.type != "cuda":
                raise RuntimeError("Validation/test inputs must remain on CUDA")
            input_device = str(former_frame_for_feature_extraction.device)

            if save_visuals and is_multiclass:
                gt_labels = labels_from_segmentation_tensor(gt_mask, num_classes)
                for i in range(gt_labels.shape[0]):
                    _save_label_png(gt_labels[i], output_folder / f"{name[i]}_gt_label.png")
            elif save_visuals:
                for i in range(gt_mask.shape[0]):
                    gt_img = Image.fromarray((gt_mask[i][0].detach().cpu().numpy() * 255).astype(np.uint8))
                    gt_img.save(output_folder / f"{name[i]}_gt.png")

            if is_monu_dataset:
                _, _, W, H = former_frame_for_feature_extraction.shape
                kernel_size = dataset.image_size
                stride = 256
                patches = []

                for y, x in np.ndindex((((W - kernel_size) // stride) + 1, ((H - kernel_size) // stride) + 1)):
                    y = y * stride
                    x = x * stride
                    patches.append(former_frame_for_feature_extraction[0, :, y:min(y + kernel_size, W), x:min(x + kernel_size, H)])
                patches = torch.stack(patches)

                major_vote_list = []
                for _ in range(major_vote_number):
                    x_list = []
                    for index in range(math.ceil(patches.shape[0] / 4)):
                        model_kwargs = {"conditioned_image": patches[index * 4:min((index + 1) * 4, patches.shape[0])]}
                        x_t_1 = torch.randn_like(
                            torch.zeros(
                                model_kwargs["conditioned_image"].shape[0],
                                gt_mask.shape[1],
                                model_kwargs["conditioned_image"].shape[2],
                                model_kwargs["conditioned_image"].shape[3],
                            )
                        ).to(device)
                        y_cond = model_kwargs["conditioned_image"]
                        x = sample_from_model(pos_coeff, netG, args.num_timesteps, x_t_1, y_cond, args)
                        x_list.append(x)
                    out = torch.cat(x_list)

                    output = torch.zeros((former_frame_for_feature_extraction.shape[0], gt_mask.shape[1], former_frame_for_feature_extraction.shape[2], former_frame_for_feature_extraction.shape[3]))
                    idx_sum = torch.zeros((former_frame_for_feature_extraction.shape[0], gt_mask.shape[1], former_frame_for_feature_extraction.shape[2], former_frame_for_feature_extraction.shape[3]))
                    for index, val in enumerate(out):
                        y, x = np.unravel_index(index, (((W - kernel_size) // stride) + 1, ((H - kernel_size) // stride) + 1))
                        y = y * stride
                        x = x * stride
                        idx_sum[0, :, y:min(y + kernel_size, W), x:min(x + kernel_size, H)] += 1
                        output[0, :, y:min(y + kernel_size, W), x:min(x + kernel_size, H)] += val[:, :min(y + kernel_size, W) - y, :min(x + kernel_size, H) - x].cpu().data.numpy()
                    output = output / idx_sum
                    major_vote_list.append(output)
                x = torch.cat(major_vote_list)
            else:
                batch_eval_size = former_frame_for_feature_extraction.shape[0]
                model_kwargs = {
                    "conditioned_image": former_frame_for_feature_extraction.repeat_interleave(
                        major_vote_number,
                        dim=0,
                    )
                }
                x_t_1 = torch.randn_like(
                    torch.zeros(
                        batch_eval_size * major_vote_number,
                        gt_mask.shape[1],
                        model_kwargs["conditioned_image"].shape[2],
                        model_kwargs["conditioned_image"].shape[3],
                    )
                ).to(device)
                y_cond = model_kwargs["conditioned_image"]
                x = sample_from_model(pos_coeff, netG, args.num_timesteps, x_t_1, y_cond, args)

            x = (x + 1.0) / 2.0
            if x.device.type != "cuda":
                raise RuntimeError("Validation/test model output left CUDA")
            output_device = str(x.device)
            if x.shape[2] != gt_mask.shape[2] or x.shape[3] != gt_mask.shape[3]:
                x = F.interpolate(x, gt_mask.shape[2:], mode="bilinear")
            x = torch.clamp(x, 0.0, 1.0)
            if not is_monu_dataset:
                x = x.view(batch_eval_size, major_vote_number, x.shape[1], x.shape[2], x.shape[3]).mean(dim=1)
            else:
                x = x.mean(dim=0, keepdim=True)

            if is_multiclass:
                pred_labels = labels_from_segmentation_tensor(x, num_classes)
                gt_labels = labels_from_segmentation_tensor(gt_mask, num_classes)
                foreground_ids = torch.arange(1, num_classes, device=device)
                for i in range(pred_labels.shape[0]):
                    if save_visuals:
                        _save_label_png(pred_labels[i], output_folder / f"{name[i]}_model_output_label.png")
                    pred_i = pred_labels[i:i + 1]
                    gt_i = gt_labels[i:i + 1]
                    row = {
                        "step": "" if step is None else step,
                        "slice_id": name[i],
                    }
                    slice_dice_values = []
                    slice_iou_values = []
                    pred_foreground = torch.isin(pred_i, foreground_ids)
                    gt_foreground = torch.isin(gt_i, foreground_ids)
                    for class_id in range(1, num_classes):
                        dsc, iou, pred_pixels, gt_pixels = dice_iou_for_label_class(pred_i, gt_i, class_id)
                        row[f"class{class_id}_name"] = class_names[class_id]
                        row[f"class{class_id}_dice"] = "" if dsc is None else dsc
                        row[f"class{class_id}_miou"] = "" if iou is None else iou
                        row[f"class{class_id}_pred_pixels"] = pred_pixels
                        row[f"class{class_id}_gt_pixels"] = gt_pixels
                        row[f"class{class_id}_metric_included"] = int(dsc is not None)
                        if dsc is not None:
                            f1_score_list.append(dsc)
                            miou_list.append(iou)
                            slice_dice_values.append(dsc)
                            slice_iou_values.append(iou)
                    row["dice"] = mean_or_zero(slice_dice_values)
                    row["miou"] = mean_or_zero(slice_iou_values)
                    row["num_nonempty_gt_classes"] = len(slice_dice_values)
                    row["is_empty_gt_foreground"] = int(not bool(gt_foreground.any().item()))
                    row["is_empty_pred_foreground"] = int(not bool(pred_foreground.any().item()))
                    rows.append(row)
                    logger.info(f"{name[i]} iou {row['miou']}, f1_Score {row['dice']}")
            else:
                x = x.round()
                if save_visuals:
                    for i in range(x.shape[0]):
                        out_img = Image.fromarray((x[i][0].detach().cpu().numpy() * 255).astype(np.uint8))
                        out_img.save(output_folder / f"{name[i]}_model_output.png")

                for index, (gt_im, out_im) in enumerate(zip(gt_mask, x)):
                    if foreground_positive:
                        metric_pred = out_im[0]
                        metric_gt = gt_im[0]
                    else:
                        metric_pred = -out_im[0] + 1
                        metric_gt = -gt_im[0] + 1

                    if skip_empty_gt:
                        f1, miou, pred_pixels, gt_pixels = calculate_binary_foreground_metrics(metric_pred, metric_gt, skip_empty_gt=True)
                    else:
                        f1, miou = calculate_metrics(metric_pred, metric_gt)
                        pred_pixels = int(metric_pred.detach().bool().sum().item())
                        gt_pixels = int(metric_gt.detach().bool().sum().item())

                    if f1 is not None:
                        f1_score_list.append(f1)
                        miou_list.append(miou)
                    rows.append(
                        {
                            "step": "" if step is None else step,
                            "slice_id": name[index],
                            "dice": "" if f1 is None else f1,
                            "miou": "" if miou is None else miou,
                            "pred_pixels": pred_pixels,
                            "gt_pixels": gt_pixels,
                            "is_empty_gt": int(gt_pixels == 0),
                            "is_empty_pred": int(pred_pixels == 0),
                            "metric_included": int(f1 is not None),
                        }
                    )
                    logger.info(f"{name[index]} iou {miou}, f1_Score {f1}")

    mean_iou, num_iou = _distributed_mean(miou_list, device)
    mean_f1, num_f1 = _distributed_mean(f1_score_list, device)

    logger.info("measure total avg")
    logger.info(f"mean iou {mean_iou}")
    logger.info(f"mean f1 {mean_f1}")

    if metrics_csv:
        if is_multiclass:
            fieldnames = _multiclass_fieldnames(num_classes)
        else:
            fieldnames = ["step", "slice_id", "dice", "miou", "pred_pixels", "gt_pixels", "is_empty_gt", "is_empty_pred", "metric_included"]
        write_rows_csv(metrics_csv, rows, fieldnames, append=True)
    if summary_json:
        if is_multiclass:
            metric_policy = (
                f"foreground Dice/IoU over classes 1..{num_classes - 1}; "
                "skip empty-GT slice/class observations; background class 0 excluded"
            )
            num_empty_gt = sum(row["is_empty_gt_foreground"] for row in rows)
            num_empty_pred = sum(row["is_empty_pred_foreground"] for row in rows)
        else:
            metric_policy = "skip_empty_ground_truth_slices" if skip_empty_gt else "official_sklearn_all_slices"
            num_empty_gt = sum(row["is_empty_gt"] for row in rows)
            num_empty_pred = sum(row["is_empty_pred"] for row in rows)
        cuda_max_allocated_mib = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
        cuda_max_reserved_mib = float(torch.cuda.max_memory_reserved(device) / (1024 ** 2))
        summary = {
            "step": step,
            "dataset": getattr(args, "dataset", ""),
            "split": getattr(dataset, "split", ""),
            "num_classes": num_classes,
            "major_vote_number": major_vote_number,
            "eval_batch_size": batch_size,
            "eval_max_items": max_items,
            "cuda_max_memory_allocated_mib": cuda_max_allocated_mib,
            "cuda_max_memory_reserved_mib": cuda_max_reserved_mib,
            "gpu_name": torch.cuda.get_device_name(device),
            "cuda_device": str(device),
            "model_parameter_device": model_parameter_device,
            "input_device": input_device,
            "output_device": output_device,
            "save_visuals": save_visuals,
            "audit_mode": getattr(args, "audit_mode", "none"),
            "sampling_steps_main_core": 0 if getattr(args, "audit_mode", "none") == "core_no_diff" else int(getattr(args, "num_timesteps", 0)),
            "uses_reverse_sampler_for_main_core": getattr(args, "audit_mode", "none") != "core_no_diff",
            "uses_non_diffusion_latent": getattr(args, "audit_mode", "none") == "core_no_diff",
            "latent_is_diffusion_noise": False,
            "num_slices": len(rows),
            "num_metric_observations": num_f1,
            "num_empty_gt": num_empty_gt,
            "num_empty_pred": num_empty_pred,
            "metric_policy": metric_policy,
            "dice": mean_f1,
            "miou": mean_iou,
        }
        Path(summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if is_dist_avail_and_initialized():
        dist.barrier()
    return mean_iou, mean_f1
