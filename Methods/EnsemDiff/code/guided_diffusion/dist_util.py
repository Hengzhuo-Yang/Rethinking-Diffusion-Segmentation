"""
Helpers for distributed training.
"""

import io
import os
import platform
import socket

import blobfile as bf
#from mpi4py import MPI
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3
EXPECTED_GPU_NAME = "NVIDIA GeForce RTX 5090"
_validated_device = None


def require_target_cuda():
    """Return cuda:0 only after fail-closed RTX 5090 validation."""
    global _validated_device
    if _validated_device is not None:
        return _validated_device
    if not th.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for EnsemDiff training, validation, and inference; "
            "CPU fallback is disabled."
        )
    if th.cuda.device_count() < 1:
        raise RuntimeError("CUDA reported available but no CUDA device is visible.")
    device = th.device("cuda:0")
    th.cuda.set_device(device)
    actual_name = th.cuda.get_device_name(device)
    if actual_name != EXPECTED_GPU_NAME:
        raise RuntimeError(
            f"Expected {EXPECTED_GPU_NAME!r} at cuda:0, found {actual_name!r}."
        )
    try:
        probe = th.ones(1, device=device)
        if probe.device.type != "cuda":
            raise RuntimeError(f"CUDA probe was created on {probe.device}.")
    except Exception as exc:
        raise RuntimeError(f"CUDA allocation on {device} failed: {exc}") from exc
    _validated_device = device
    return device


def setup_dist():
    """
    Setup a distributed process group.
    """
    device = require_target_cuda()
    if dist.is_initialized():
        return
    backend = (
        "nccl"
        if th.cuda.is_available()
        and dist.is_nccl_available()
        and platform.system() != "Windows"
        else "gloo"
    )

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
    if dev() != device:
        raise RuntimeError("Distributed setup changed the validated CUDA device.")


def dev():
    """
    Get the device to use for torch.distributed.
    """
    return require_target_cuda()


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
