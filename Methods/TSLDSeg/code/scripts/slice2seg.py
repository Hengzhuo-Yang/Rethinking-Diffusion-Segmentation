import argparse
import os
# Release-modified target-dataset CLI; see MODIFICATIONS.md and LICENSES/.
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy
import torch
from einops import rearrange
from omegaconf import OmegaConf
from ptflops import get_model_complexity_info
from pytorch_lightning import seed_everything
from thop import profile
from torch.utils.data import DataLoader

from ldm.util import instantiate_from_config, load_trusted_checkpoint
from ldm.data import ACDCValidationEval, ISICTest, SynapseValidationEval
from ldm.runtime import require_rtx5090


def prepare_for_first_stage(x):
    x = x.clone().detach()
    if len(x.shape) == 3:
        x = x[None, ...]
    x = rearrange(x, 'b h w c -> b c h w')
    return x.to(memory_format=torch.contiguous_format).float().cuda()


def dice_score(pred, targs):
    assert pred.shape == targs.shape, (pred.shape, targs.shape)
    pred = (pred > 0).astype(numpy.float32)
    targs = (targs > 0).astype(numpy.float32)
    if targs.sum() == 0:
        return numpy.nan
    return (2. * (pred * targs).sum()) / (pred.sum() + targs.sum() + 1e-10)


def iou_score(pred, targs):
    assert pred.shape == targs.shape, (pred.shape, targs.shape)
    pred = (pred > 0).astype(numpy.float32)
    targs = (targs > 0).astype(numpy.float32)
    if targs.sum() == 0:
        return numpy.nan
    intersection = (pred * targs).sum()
    union = pred.sum() + targs.sum() - intersection
    return intersection / (union + 1e-10)


def load_model_from_config(config, ckpt):
    pl_sd = load_trusted_checkpoint(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]

    x_grid = "cond_stage_model.oeem.orient_block.gabor_conv.x_grid"
    if x_grid in sd:
        sd[x_grid] = sd[x_grid].clone()
    y_grid = "cond_stage_model.oeem.orient_block.gabor_conv.y_grid"
    if y_grid in sd:
        sd[y_grid] = sd[y_grid].clone()

    model = instantiate_from_config(config.model)
    print(f"\033[31m[Model Weights Rewrite]: Loading model from {ckpt}\033[0m")
    m, u = model.load_state_dict(sd, strict=False)
    print("\033[31mmissing keys:\033[0m")
    print(m)
    print("\033[31munexpected keys:\033[0m")
    print(u)
    model.eval()
    return model, pl_sd


def calculate_volume_dice(**kwargs):
    # inter_list, union_list, pred_sum, gt_sum = kwargs
    inter = sum(kwargs["inter_list"])
    union = sum(kwargs["union_list"])
    if kwargs["pred_sum"] > 0 and kwargs["gt_sum"] > 0:
        return 2 * inter / (union + 1e-10)
    elif kwargs["pred_sum"] > 0 and kwargs["gt_sum"] == 0:
        return 1
    else:
        return 0

def cal_params_flops(model, size):
    input = torch.randn(1, 4, size//8, size//8).cuda()
    c = torch.randn(1, 3, size, size).cuda()
    flops, params = profile(model, inputs=(input, c, numpy.array([-1])))
    print('flops', flops / 1e9)  ## 打印计算量
    print('params', params / 1e6)  ## 打印参数量

    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.2fM" % (total / 1e6))


# Example function to calculate and print GMACs and parameter count for a given model
def print_model_stats(model, input_size=(3, 224, 224)):
    # Print model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model created, param count: {total_params}')

    # Calculate GMACs using ptflops
    macs, params = get_model_complexity_info(model, input_size, as_strings=True, print_per_layer_stat=True)

    # Display GMACs and params
    print(f'Model: {macs} GMACs, {params} parameters')


def main():
    parser = argparse.ArgumentParser()
    # saving settings
    parser.add_argument("--outdir", type=str, required=True, help="directory for predictions")
    parser.add_argument("--name", type=str, help="name to call this inference", default="test")
    # sampler settings
    parser.add_argument("--sampler", type=str,
                        choices=["raw", "direct", "ddim", "plms", "dpm_solver"],
                        help="the sampler used for sampling", )
    parser.add_argument("--ddim_steps", type=int, default=200, help="number of ddim sampling steps", )
    parser.add_argument("--ddim_eta", type=float, default=1.0,
                        help="ddim eta (eta=0.0 corresponds to deterministic sampling", )
    # dataset settings
    parser.add_argument("--dataset", choices=["btcv", "acdc", "isic2018"], required=True)
    parser.add_argument("--data_dir", type=str, required=True,
                        help="physical test-partition image/mask directory", )
    parser.add_argument("--manifest", type=str, required=True,
                        help="fixed test ID manifest", )
    parser.add_argument("--num_classes", type=int, default=2,
                        help="number of classes including background", )
    parser.add_argument("--skip_stats", action='store_true',
                        help="skip FLOPs/parameter profiling after evaluation", )
    # sampling settings
    parser.add_argument("--fixed_code", action='store_true',
                        help="if enabled, uses the same starting code across samples ", )
    parser.add_argument("--H", type=int, default=256, help="image height, in pixel space", )
    parser.add_argument("--W", type=int, default=256, help="image width, in pixel space", )
    parser.add_argument("--C", type=int, default=4, help="latent channels", )
    parser.add_argument("--f", type=int, default=8, help="downsampling factor", )
    parser.add_argument("--n_samples", type=int, default=1,
                        help="how many samples to produce for each given prompt. A.k.a. batch size", )
    parser.add_argument("--config", type=str, required=True,
                        help="path to the matching release config", )
    parser.add_argument("--ckpt", type=str, required=True,
                        help="best validation-selected checkpoint", )
    parser.add_argument("--seed", type=int, default=0,
                        help="the seed (for reproducible sampling)", )
    parser.add_argument("--audit_mode", type=str, default=None,
                        help="optional model.params.audit_mode override, e.g. core_no_diff for core audit checkpoints", )
    parser.add_argument("--times", type=int, default=1,
                        help="times of testing for stability evaluation", )
    parser.add_argument("--save_results", action='store_true',  # will slow down inference
                        help="saving the predictions for the whole test set.", )
    opt = parser.parse_args()
    runtime_info = require_rtx5090(0)
    print(f"RTX 5090 preflight passed: {runtime_info}")

    if opt.dataset == "btcv":
        print("Evaluate on BTCV/Synapse dataset in binary segmentation manner.")
        dataset = SynapseValidationEval(
            data_root=opt.data_dir, manifest_path=opt.manifest,
            size=opt.H, num_classes=opt.num_classes,
        )
    elif opt.dataset == "acdc":
        print("Evaluate on ACDC dataset in multi-class segmentation manner.")
        dataset = ACDCValidationEval(
            data_root=opt.data_dir, manifest_path=opt.manifest,
            size=opt.H, num_classes=opt.num_classes,
        )
    elif opt.dataset == "isic2018":
        print("Evaluate on ISIC2018 Task 1 dataset in binary lesion segmentation manner.")
        if opt.num_classes != 2:
            raise ValueError(f"ISIC2018 Task 1 requires --num_classes 2, got {opt.num_classes}")
        dataset = ISICTest(
            data_root=opt.data_dir, manifest_path=opt.manifest,
            size=opt.H, num_classes=opt.num_classes,
        )
    else:
        raise NotImplementedError(f"Not implement for dataset {opt.dataset}")

    data = DataLoader(dataset, batch_size=opt.n_samples, shuffle=False)

    config = OmegaConf.load(f"{opt.config}")
    if opt.audit_mode is not None:
        config["model"]["params"]["audit_mode"] = opt.audit_mode
    config["model"]["params"].pop("ckpt_path")
    config["model"]["params"]["cond_stage_config"]["params"].pop("ckpt_path")
    config["model"]["params"]["first_stage_config"]["params"].pop("ckpt_path")

    model, pl_sd = load_model_from_config(config, f"{opt.ckpt}")
    device = torch.device("cuda", 0)
    model = model.to(device)

    os.makedirs(opt.outdir, exist_ok=True)


    for idx in range(opt.times):
        if opt.times > 1:   # if test only once, use specified seed.
            opt.seed = idx
        seed_everything(opt.seed)
        print(f"\033[32m seed:{opt.seed}\033[0m")

        outpath = os.path.join(opt.outdir, str(opt.seed))
        os.makedirs(outpath, exist_ok=True)

        start = time.time()
        metrics_dict, _ = model.log_dice(
            data=data,
            save_dir=outpath,
            ddim_steps=opt.ddim_steps,
            sampler_name=opt.sampler or "direct",
            metric_prefix="test",
        )
        print(f"Inference Speed: {len(data) /(time.time() - start)}")

        dice_list = metrics_dict["test_avg_dice"]
        iou_list = metrics_dict["test_avg_iou"]
        print(f"\033[31m[Mean Dice][{opt.dataset}][{opt.sampler or 'direct'}]: {numpy.nanmean(numpy.asarray(dice_list, dtype=numpy.float64))}\033[0m")
        print(f"\033[31m[Mean  IoU][{opt.dataset}][{opt.sampler or 'direct'}]: {numpy.nanmean(numpy.asarray(iou_list, dtype=numpy.float64))}\033[0m")

        if opt.times > 1:
            print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
            f" \nEnjoy.")

    if not opt.skip_stats:
        cal_params_flops(model, 256)


if __name__ == "__main__":
    main()






