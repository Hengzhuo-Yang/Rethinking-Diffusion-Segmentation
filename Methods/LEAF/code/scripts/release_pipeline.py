#!/usr/bin/env python3
"""Run one fixed LEAF dataset/condition through train-selection-test."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf


CODE_ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = CODE_ROOT.parent
GPU_PREFLIGHT = RELEASE_ROOT / "tests" / "gpu_preflight.py"
SPLIT_VALIDATOR = RELEASE_ROOT / "tests" / "validate_splits.py"

CONFIGS = {
    "btcv": {
        "full": "config-btcv.yaml",
        "random-yt": "config-btcv-train-random-yt.yaml",
        "shuffle-yt": "config-btcv-train-shuffle-yt.yaml",
        "core-no-diff": "config-btcv-core-no-diff.yaml",
    },
    "acdc": {
        "full": "config-acdc.yaml",
        "random-yt": "config-acdc-train-random-yt.yaml",
        "shuffle-yt": "config-acdc-train-shuffle-yt.yaml",
        "core-no-diff": "config-acdc-core-no-diff.yaml",
    },
    "isic2018": {
        "full": "config-isic2018.yaml",
        "random-yt": "config-isic2018-train-random-yt.yaml",
        "shuffle-yt": "config-isic2018-train-shuffle-yt.yaml",
        "core-no-diff": "config-isic2018-core-no-diff.yaml",
    },
}

AUDIT_MODES = {
    "full": "none",
    "random-yt": "train_random_yt",
    "shuffle-yt": "train_shuffle_yt",
    "core-no-diff": "core_no_diff",
}

EVALUATION = {
    "btcv": ("evaluate_btcv_leaf.py", "test", "per_slice_metrics.csv"),
    "acdc": ("evaluate_acdc_leaf.py", "testing", "per_slice_metrics.csv"),
    "isic2018": ("evaluate_isic2018_leaf.py", "testing", "per_image_metrics.csv"),
}
VALIDATION_SPLITS = {
    "btcv": "val",
    "acdc": "validation",
    "isic2018": "validation",
}


def config_path(args: argparse.Namespace) -> Path:
    return CODE_ROOT / "config" / CONFIGS[args.dataset][args.condition]


def load_config(args: argparse.Namespace):
    cfg = OmegaConf.load(config_path(args))
    actual = str(cfg.get("audit_mode", "none")).strip().lower()
    if actual in {"full_diffusion", "original", "default"}:
        actual = "none"
    expected = AUDIT_MODES[args.condition]
    if actual != expected:
        raise RuntimeError(
            f"Config audit-mode mismatch: condition={args.condition}, "
            f"expected={expected}, config={actual}"
        )
    return cfg


def run_dir(args: argparse.Namespace, cfg) -> Path:
    return (args.runs_root / str(cfg.job_name)).resolve()


def environment(args: argparse.Namespace) -> dict[str, str]:
    return {
        "CUDA_VISIBLE_DEVICES": str(args.cuda_device),
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def execute(
    stage: str,
    command: list[str],
    env_updates: dict[str, str],
    dry_run: bool,
) -> None:
    print(json.dumps({"stage": stage, "cwd": str(CODE_ROOT), "command": command}, indent=2))
    if dry_run:
        return
    merged = os.environ.copy()
    merged.update(env_updates)
    subprocess.run(command, cwd=CODE_ROOT, env=merged, check=True)


def validate_fixed_cache(args: argparse.Namespace) -> None:
    execute(
        "fixed_split_validation",
        [
            sys.executable,
            str(SPLIT_VALIDATOR),
            "--data-root",
            f"{args.dataset}={args.data_root.resolve()}",
        ],
        environment(args),
        args.dry_run,
    )


def preflight(args: argparse.Namespace) -> None:
    execute(
        "gpu_preflight",
        [sys.executable, str(GPU_PREFLIGHT), "--device", "cuda:0"],
        environment(args),
        args.dry_run,
    )


def require_inputs(args: argparse.Namespace, cfg) -> None:
    dataset_dir = args.data_root.resolve() / str(cfg.dataset_name)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Missing preprocessed dataset directory: {dataset_dir}")
    for subfolder in ("vae", "unet"):
        if not (args.assets_root.resolve() / subfolder).is_dir():
            raise FileNotFoundError(
                f"Missing locally prepared pretrained component: {args.assets_root / subfolder}"
            )


def train(args: argparse.Namespace, cfg) -> None:
    target = run_dir(args, cfg)
    if not args.dry_run:
        require_inputs(args, cfg)
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite existing run directory: {target}")
    command = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--num_processes",
        "1",
        "--num_machines",
        "1",
        "--mixed_precision",
        "bf16",
        "--dynamo_backend",
        "no",
        "train.py",
        "--config",
        str(config_path(args)),
        "--data-root",
        str(args.data_root.resolve()),
        "--output-dir",
        str(args.runs_root.resolve()),
        "--pretrained-path",
        str(args.assets_root.resolve()),
    ]
    execute("train_with_validation_selection", command, environment(args), args.dry_run)


def selected_checkpoint(args: argparse.Namespace, cfg) -> tuple[Path, dict]:
    target = run_dir(args, cfg)
    metadata_path = target / "best_checkpoint.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Final test requires validation-selection metadata: {metadata_path}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("selection_partition") != "validation":
        raise RuntimeError("The selected checkpoint was not chosen on validation data.")
    if metadata.get("selection_metric") != "mean_validation_dice":
        raise RuntimeError("The selected checkpoint does not use mean validation Dice.")
    if metadata.get("dataset") != str(cfg.dataset_name):
        raise RuntimeError("Checkpoint dataset does not match the requested config.")
    if metadata.get("validation_split") != VALIDATION_SPLITS[args.dataset]:
        raise RuntimeError(
            "Checkpoint validation split does not match the fixed release partition."
        )
    if metadata.get("audit_mode") != AUDIT_MODES[args.condition]:
        raise RuntimeError("Checkpoint audit condition does not match the requested config.")
    checkpoint = (target / str(metadata.get("checkpoint", ""))).resolve()
    if target not in checkpoint.parents or not checkpoint.is_dir():
        raise RuntimeError(f"Invalid or missing selected checkpoint directory: {checkpoint}")
    return checkpoint, metadata


def final_test(args: argparse.Namespace, cfg) -> None:
    target = run_dir(args, cfg)
    final_dir = target / "final_test"
    script_name, split, per_case_name = EVALUATION[args.dataset]
    if args.dry_run:
        checkpoint = target / "checkpoint" / "VALIDATION_SELECTED_STEP"
        metadata = {"dry_run": True}
    else:
        require_inputs(args, cfg)
        checkpoint, metadata = selected_checkpoint(args, cfg)
        if final_dir.exists() and any(final_dir.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite or reselect on final-test output: {final_dir}"
            )
        (final_dir / "samples").mkdir(parents=True, exist_ok=False)
        request = {
            "status": "started",
            "dataset": args.dataset,
            "condition": args.condition,
            "test_partition": "test",
            "checkpoint_selection": metadata,
        }
        (final_dir / "final_test_request.json").write_text(
            json.dumps(request, indent=2, sort_keys=True), encoding="utf-8"
        )

    per_case_flag = "--per_image_csv" if args.dataset == "isic2018" else "--per_slice_csv"
    command = [
        sys.executable,
        str(CODE_ROOT / "scripts" / script_name),
        "--config",
        str(config_path(args)),
        "--checkpoint_dir",
        str(checkpoint),
        "--split",
        split,
        "--sample_dir",
        str(final_dir / "samples"),
        "--summary_csv",
        str(final_dir / "metrics_summary.csv"),
        "--summary_json",
        str(final_dir / "metrics_summary.json"),
        per_case_flag,
        str(final_dir / per_case_name),
        "--batch_size",
        str(args.test_batch_size),
        "--num_workers",
        str(args.num_workers),
        "--num_inference_steps",
        "0" if args.condition == "core-no-diff" else "1",
        "--device",
        "cuda:0",
        "--prefer_ema",
        "--data-root",
        str(args.data_root.resolve()),
        "--pretrained-path",
        str(args.assets_root.resolve()),
    ]
    execute("validation_selected_checkpoint_on_final_test", command, environment(args), args.dry_run)
    if not args.dry_run:
        request_path = final_dir / "final_test_request.json"
        request = json.loads(request_path.read_text(encoding="utf-8"))
        request["status"] = "completed"
        request_path.write_text(
            json.dumps(request, indent=2, sort_keys=True), encoding="utf-8"
        )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Run one of the 12 fixed LEAF combinations. Training selects on validation; "
            "final test only loads the recorded best checkpoint."
        )
    )
    result.add_argument("action", choices=("train", "test", "run"))
    result.add_argument("--dataset", required=True, choices=tuple(CONFIGS))
    result.add_argument("--condition", required=True, choices=tuple(AUDIT_MODES))
    result.add_argument("--data-root", required=True, type=Path)
    result.add_argument("--assets-root", required=True, type=Path)
    result.add_argument("--runs-root", type=Path, default=RELEASE_ROOT / "runs")
    result.add_argument("--cuda-device", type=int, default=0)
    result.add_argument("--test-batch-size", type=int, default=32)
    result.add_argument("--num-workers", type=int, default=0)
    result.add_argument("--dry-run", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.cuda_device < 0 or args.test_batch_size < 1 or args.num_workers < 0:
        raise ValueError("CUDA index, batch size, and worker count must be non-negative/positive.")
    cfg = load_config(args)
    validate_fixed_cache(args)
    preflight(args)
    if args.action in {"train", "run"}:
        train(args, cfg)
    if args.action in {"test", "run"}:
        final_test(args, cfg)


if __name__ == "__main__":
    main()
