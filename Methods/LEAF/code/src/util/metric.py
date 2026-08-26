import torch
from typing import List, Dict
from torchvision.utils import make_grid


def labels_from_segmentation_tensor(
    mask: torch.Tensor,
    num_classes: int,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Return class-index labels shaped [B, H, W].

    Multi-class LEAF masks can use either C one-hot channels including
    background or C-1 foreground channels where all-zero means background.
    The latter keeps ACDC compatible with the 3-channel LEAF VAE while
    preserving labels 0..3.
    """
    if mask.ndim < 3:
        raise ValueError(f"Expected mask with at least 3 dims, got {tuple(mask.shape)}")
    if mask.ndim >= 4:
        if mask.shape[1] == 1:
            mask = mask[:, 0]
        elif mask.shape[1] == num_classes:
            mask = torch.argmax(mask, dim=1)
        elif num_classes > 2 and mask.shape[1] == num_classes - 1:
            foreground_score, foreground_index = torch.max(mask, dim=1)
            background = torch.zeros_like(foreground_index)
            return torch.where(
                foreground_score > threshold,
                foreground_index.long() + 1,
                background.long(),
            )
        elif num_classes == 2:
            mask = torch.mean(mask, dim=1)
        else:
            raise ValueError(
                f"Expected one label channel, {num_classes} class channels, "
                f"or {num_classes - 1} foreground channels, got shape {tuple(mask.shape)}"
            )

    if num_classes == 2:
        if mask.dtype.is_floating_point:
            return (mask > threshold).long()
        return (mask > 0).long()
    return mask.round().long().clamp(min=0, max=num_classes - 1)

class SegmentationMetric:

    def __init__(
        self,
        metrics: List[str],
        device: torch.device,
        num_classes: int = 2,
        include_background: bool = False,
    ):
        self.metrics = metrics
        unsupported = sorted(set(metrics) - {"dice", "miou", "iou"})
        if unsupported:
            raise ValueError(f"Unsupported segmentation metrics: {unsupported}")
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        self.device = device
        self.num_classes = int(num_classes)
        first_class = 0 if include_background else 1
        self.class_ids = tuple(range(first_class, self.num_classes))
        if not self.class_ids:
            raise ValueError("No classes selected for segmentation metric aggregation")
        self.dice_sum = torch.zeros((), device=device)
        self.iou_sum = torch.zeros((), device=device)
        self.count = torch.zeros((), device=device)
        self.empty_gt_count = torch.zeros((), device=device)
        self.empty_pred_count = torch.zeros((), device=device)
        self.total_slices = torch.zeros((), device=device)

    def update(self, output: torch.Tensor, target: torch.Tensor) -> None:
        output = labels_from_segmentation_tensor(output, self.num_classes).to(self.device)
        target = labels_from_segmentation_tensor(target, self.num_classes).to(self.device)
        if output.shape != target.shape:
            raise ValueError(f"Output and target shapes differ: {tuple(output.shape)} vs {tuple(target.shape)}")

        self.total_slices += output.shape[0]
        foreground_ids = torch.as_tensor(self.class_ids, device=self.device)
        target_foreground = torch.isin(target, foreground_ids)
        output_foreground = torch.isin(output, foreground_ids)
        reduce_dims = tuple(range(1, target.ndim))
        self.empty_gt_count += torch.count_nonzero(target_foreground.sum(dim=reduce_dims) == 0)
        self.empty_pred_count += torch.count_nonzero(output_foreground.sum(dim=reduce_dims) == 0)

        for class_id in self.class_ids:
            output_class = output == class_id
            target_class = target == class_id
            intersection = torch.logical_and(output_class, target_class).sum(dim=reduce_dims).float()
            pred_sum = output_class.sum(dim=reduce_dims).float()
            target_sum = target_class.sum(dim=reduce_dims).float()
            valid = target_sum > 0
            if not torch.any(valid):
                continue

            dice_den = pred_sum + target_sum
            iou_den = pred_sum + target_sum - intersection
            dice = torch.where(dice_den > 0, (2.0 * intersection) / dice_den, torch.zeros_like(dice_den))
            iou = torch.where(iou_den > 0, intersection / iou_den, torch.zeros_like(iou_den))
            self.dice_sum += dice[valid].sum()
            self.iou_sum += iou[valid].sum()
            self.count += valid.sum()
            
    def compute(self) -> Dict[str, float]:
        results = {}
        if self.count.item() == 0:
            mean_dice = torch.zeros((), device=self.device)
            mean_iou = torch.zeros((), device=self.device)
        else:
            mean_dice = self.dice_sum / self.count
            mean_iou = self.iou_sum / self.count
        if "dice" in self.metrics:
            results["dice"] = mean_dice.item()
        if "miou" in self.metrics:
            # Historical column name: for binary LEAF tasks this stores foreground IoU.
            results["miou"] = mean_iou.item()
        if "iou" in self.metrics:
            results["iou"] = mean_iou.item()
        return results

    def summary_counts(self) -> Dict[str, int]:
        return {
            "num_slices": int(self.total_slices.item()),
            "num_metric_observations": int(self.count.item()),
            "num_empty_gt": int(self.empty_gt_count.item()),
            "num_empty_pred": int(self.empty_pred_count.item()),
        }

class Visualization:

    def __init__(self):
        self.images = torch.tensor([])
        self.outputs = torch.tensor([])
        self.targets = torch.tensor([])

    def update(self, images: torch.Tensor, output: torch.Tensor, target: torch.Tensor) -> None:
        output = output.repeat(1, 3, 1, 1)
        target = target.repeat(1, 3, 1, 1)
        self.images = torch.cat([self.images, images.float()], dim=0)
        self.outputs = torch.cat([self.outputs, output.float()], dim=0)
        self.targets = torch.cat([self.targets, target.float()], dim=0)
    
    def sample(self, nrow: int, padding: int = 3):
        def add_padding(image: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.pad(image, (padding, padding, padding, padding), mode="constant", value=1)
        
        target_tensors = [add_padding(tensor) for tensor in [self.images, self.targets, self.outputs]]
        units = [item for i in range(nrow) for item in (target_tensors[0][i], target_tensors[1][i], target_tensors[2][i])]
        grid = make_grid(units, nrow=nrow, normalize=False, padding=3)
        grid_np = grid.permute(1, 2, 0).cpu().numpy()
        return grid_np
