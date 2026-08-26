import csv
import os
import time
from pathlib import Path

import torch as th

from . import dist_util, logger
from .gaussian_diffusion import direct_segmentation_prediction


def dice_iou_from_binary(pred, target):
    pred = pred.bool()
    target = target.bool()
    intersection = th.logical_and(pred, target).sum().item()
    pred_sum = pred.sum().item()
    target_sum = target.sum().item()
    dice_den = pred_sum + target_sum
    iou_den = pred_sum + target_sum - intersection
    dice = 1.0 if dice_den == 0 else (2.0 * intersection) / dice_den
    iou = 1.0 if iou_den == 0 else intersection / iou_den
    return dice, iou, pred_sum, target_sum


def labels_from_target(target, num_classes):
    if target.shape[1] == 1:
        if num_classes <= 2:
            return (target[:, 0, ...] > 0.5).long()
        return target[:, 0, ...].long().clamp(min=0, max=num_classes - 1)
    if target.shape[1] == num_classes:
        return target.argmax(dim=1).long()
    raise ValueError(
        f"Expected one label channel or {num_classes} one-hot channels, "
        f"got {target.shape[1]} target channels."
    )


def labels_from_prediction(prediction, num_classes):
    if prediction.shape[1] == 1:
        if num_classes <= 2:
            return (prediction[:, 0, ...] > 0.5).long()
        return prediction[:, 0, ...].round().long().clamp(min=0, max=num_classes - 1)
    return prediction.argmax(dim=1).long()


def dice_iou_for_labels(pred_labels, target_labels, num_classes):
    class_metrics = {class_id: {"dice": [], "iou": []} for class_id in range(1, num_classes)}
    sample_metrics = []
    empty_gt = 0
    empty_pred = 0

    for sample_idx in range(pred_labels.shape[0]):
        sample_values = []
        pred_any = pred_labels[sample_idx] > 0
        target_any = target_labels[sample_idx] > 0
        if not target_any.any():
            empty_gt += 1
        if not pred_any.any():
            empty_pred += 1

        for class_id in range(1, num_classes):
            pred = pred_labels[sample_idx] == class_id
            target = target_labels[sample_idx] == class_id
            dice, iou, _, target_sum = dice_iou_from_binary(pred, target)
            target_nonempty = target_sum > 0
            if target_nonempty:
                class_metrics[class_id]["dice"].append(dice)
                class_metrics[class_id]["iou"].append(iou)
            sample_values.append(
                {
                    "dice": dice,
                    "iou": iou,
                    "target_nonempty": target_nonempty,
                }
            )
        sample_metrics.append(sample_values)

    return class_metrics, sample_metrics, empty_gt, empty_pred


def append_validation_row(csv_path, row):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def validate_segmentation(
    *,
    model,
    diffusion,
    dataloader,
    step,
    image_channels,
    mask_channels,
    num_seg_classes,
    num_ensemble,
    use_ddim,
    output_csv,
    audit_mode="none",
):
    was_training = model.training
    model.eval()
    device = dist_util.dev()
    core_no_diff = audit_mode == "core_no_diff"
    sample_fn = None
    if not core_no_diff:
        sample_fn = (
            diffusion.ddim_sample_loop_known if use_ddim else diffusion.p_sample_loop_known
        )

    dices = []
    ious = []
    nonempty_dices = []
    nonempty_ious = []
    class_dices = {class_id: [] for class_id in range(1, num_seg_classes)}
    class_ious = {class_id: [] for class_id in range(1, num_seg_classes)}
    slice_count = 0
    empty_gt = 0
    empty_pred = 0
    start_time = time.time()

    with th.no_grad():
        for image, target in dataloader:
            image = image.to(device)
            target = target.to(device)
            if core_no_diff:
                logits = model(image)
                mean_sample = direct_segmentation_prediction(logits)
            else:
                noisy_mask = th.randn(
                    image.shape[0],
                    mask_channels,
                    image.shape[-2],
                    image.shape[-1],
                    device=device,
                    dtype=image.dtype,
                )
                conditioned = th.cat((image, noisy_mask), dim=1)
                samples = []
                for _ in range(num_ensemble):
                    sample, _, _ = sample_fn(
                        model,
                        (
                            image.shape[0],
                            image_channels + mask_channels,
                            image.shape[-2],
                            image.shape[-1],
                        ),
                        conditioned,
                        clip_denoised=True,
                        model_kwargs={},
                    )
                    samples.append(sample)

                mean_sample = th.stack(samples, dim=0).mean(dim=0)
            pred_labels = labels_from_prediction(mean_sample, num_seg_classes)
            target_labels = labels_from_target(target, num_seg_classes)
            slice_count += pred_labels.shape[0]
            batch_class_metrics, batch_sample_metrics, batch_empty_gt, batch_empty_pred = (
                dice_iou_for_labels(pred_labels, target_labels, num_seg_classes)
            )
            empty_gt += batch_empty_gt
            empty_pred += batch_empty_pred
            for class_id in range(1, num_seg_classes):
                class_dices[class_id].extend(batch_class_metrics[class_id]["dice"])
                class_ious[class_id].extend(batch_class_metrics[class_id]["iou"])
            for sample_values in batch_sample_metrics:
                for value in sample_values:
                    if value["target_nonempty"]:
                        dices.append(value["dice"])
                        ious.append(value["iou"])
                        nonempty_dices.append(value["dice"])
                        nonempty_ious.append(value["iou"])

    elapsed = time.time() - start_time
    count = len(dices)
    row = {
        "step": step,
        "dice": sum(dices) / count if count else 0.0,
        "iou": sum(ious) / count if count else 0.0,
        "dice_nonempty_gt": (
            sum(nonempty_dices) / len(nonempty_dices) if nonempty_dices else 0.0
        ),
        "iou_nonempty_gt": (
            sum(nonempty_ious) / len(nonempty_ious) if nonempty_ious else 0.0
        ),
        "num_slices": slice_count,
        "num_metric_observations": count,
        "num_nonempty_gt": len(nonempty_dices),
        "num_empty_gt": empty_gt,
        "num_empty_pred": empty_pred,
        "num_ensemble": 1 if core_no_diff else num_ensemble,
        "sampler_steps": 0 if core_no_diff else diffusion.num_timesteps,
        "use_ddim": False if core_no_diff else use_ddim,
        "elapsed_sec": round(elapsed, 3),
        "audit_mode": audit_mode,
        "sampling_steps_main_core": 0 if core_no_diff else diffusion.num_timesteps,
        "inference_time_sec_per_case": (
            round(elapsed / slice_count, 6) if slice_count else 0.0
        ),
        "num_seg_classes": num_seg_classes,
        "mask_channels": mask_channels,
    }
    for class_id in range(1, num_seg_classes):
        row[f"class{class_id}_dice"] = (
            sum(class_dices[class_id]) / len(class_dices[class_id])
            if class_dices[class_id]
            else 0.0
        )
        row[f"class{class_id}_iou"] = (
            sum(class_ious[class_id]) / len(class_ious[class_id])
            if class_ious[class_id]
            else 0.0
        )
    append_validation_row(output_csv, row)
    logger.log(
        "validation "
        f"step={step} dice={row['dice']:.4f} iou={row['iou']:.4f} "
        f"nonempty_dice={row['dice_nonempty_gt']:.4f} "
        f"nonempty_iou={row['iou_nonempty_gt']:.4f} "
        f"slices={slice_count} steps={row['sampler_steps']} "
        f"ensemble={row['num_ensemble']} elapsed_sec={row['elapsed_sec']}"
    )
    if was_training:
        model.train()
    return row
