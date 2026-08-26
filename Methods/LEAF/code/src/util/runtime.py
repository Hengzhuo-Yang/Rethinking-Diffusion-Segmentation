from __future__ import annotations

from pathlib import Path

import torch


EXPECTED_GPU_NAME = "NVIDIA GeForce RTX 5090"


def resolve_from_code_root(value: str | Path, code_root: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = code_root / path
    return str(path.resolve())


def require_rtx5090(device: torch.device | str = "cuda:0") -> torch.device:
    requested = torch.device(device)
    if requested.type != "cuda":
        raise RuntimeError(
            f"LEAF training and inference require CUDA; received device={requested}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU fallback is not supported.")
    if torch.cuda.device_count() < 1:
        raise RuntimeError("No CUDA device is visible; CPU fallback is not supported.")
    index = requested.index if requested.index is not None else torch.cuda.current_device()
    actual_name = torch.cuda.get_device_name(index)
    if actual_name != EXPECTED_GPU_NAME:
        raise RuntimeError(
            f"Expected {EXPECTED_GPU_NAME!r} at {requested}, found {actual_name!r}."
        )
    torch.cuda.set_device(index)
    return torch.device("cuda", index)


def normalize_audit_mode(value: object) -> str:
    audit_mode = str(value if value is not None else "none").strip().lower()
    if audit_mode in {"", "null", "false", "full_diffusion", "original", "default"}:
        audit_mode = "none"
    allowed_modes = {"none", "train_shuffle_yt", "train_random_yt", "core_no_diff"}
    if audit_mode not in allowed_modes:
        raise ValueError(
            f"Unsupported audit_mode={audit_mode!r}. Supported modes: {sorted(allowed_modes)}"
        )
    return audit_mode
