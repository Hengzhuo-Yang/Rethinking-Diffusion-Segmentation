import random

from torch import nn
from torch.backends import cudnn
import torch
import shutil
import os
import numpy as np
import torch.distributed as dist


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


# %%

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def set_random_seed_for_iterations(seed):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



EXPECTED_GPU_NAME = "NVIDIA GeForce RTX 5090"


def dev(gpu):
    """Return the required CUDA device and reject every fallback path."""
    gpu = int(gpu)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is disabled for this release")
    if gpu < 0 or gpu >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device index {gpu} is unavailable")
    torch.cuda.set_device(gpu)
    actual_name = torch.cuda.get_device_name(gpu)
    if actual_name != EXPECTED_GPU_NAME:
        raise RuntimeError(
            f"Expected {EXPECTED_GPU_NAME}, found {actual_name!r} on cuda:{gpu}"
        )
    return torch.device(f"cuda:{gpu}")

def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def broadcast_params(params):
    if not is_dist_avail_and_initialized():
        return
    for param in params:
        dist.broadcast(param.data, src=0)


def unwrap_module(module):
    return module.module if hasattr(module, 'module') else module

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname.find("DownConv") == -1 and classname.find("UpConv") == -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

class CustomDDPWrapper(nn.parallel.DistributedDataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

