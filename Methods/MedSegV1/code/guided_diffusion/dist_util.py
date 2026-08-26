"""
Helpers for distributed training.
"""

import io
import os
import socket
import sys

import blobfile as bf
#from mpi4py import MPI
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3
EXPECTED_GPU_NAME = "NVIDIA GeForce RTX 5090"


def require_rtx5090():
    """Return cuda:0 after enforcing the release hardware contract."""
    if not th.cuda.is_available():
        raise RuntimeError(
            "CUDA is required; CPU fallback is disabled for this release."
        )
    if th.cuda.device_count() < 1:
        raise RuntimeError("CUDA reported available but no CUDA devices exist.")
    device = th.device("cuda:0")
    device_name = th.cuda.get_device_name(device).strip()
    if device_name != EXPECTED_GPU_NAME:
        raise RuntimeError(
            "This release is validated only for "
            f"{EXPECTED_GPU_NAME}; cuda:0 is {device_name!r}."
        )
    th.cuda.set_device(device)
    return device


def setup_dist(args):
    """
    Setup a distributed process group.
    """
    if not args.multi_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_dev
    require_rtx5090()
    if dist.is_initialized():
        return

    backend = "gloo" if sys.platform.startswith("win") else "nccl"

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())
    os.environ["MASTER_ADDR"] = '127.0.0.1'#comm.bcast(hostname, root=0)
    os.environ["RANK"] = '0'#str(comm.rank)
    os.environ["WORLD_SIZE"] = '1'#str(comm.size)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=backend, init_method="env://")


def dev():
    """
    Get the device to use for torch.distributed.
    """
    return require_rtx5090()


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    mpigetrank=0
    if mpigetrank==0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
    else:
        data = None
    
    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
