#!/usr/bin/env python3
"""Unified, manifest-locked train/validate/test entry point for all 12 runs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = CODE_ROOT.parent
SCRIPT_ROOT = CODE_ROOT / "scripts"
MANIFEST_ROOT = CODE_ROOT / "manifests"
GPU_PREFLIGHT = RELEASE_ROOT / "tests" / "gpu_preflight.py"
MANIFEST_NAMES = {
    "btcv": {
        "train": "train_cases.txt",
        "val": "val_cases.txt",
        "test": "test_cases.txt",
    },
    "acdc": {
        "train": "train_patients.txt",
        "val": "val_patients.txt",
        "test": "test_patients.txt",
    },
    "isic2018": {
        "train": "training.txt",
        "val": "validation.txt",
        "test": "testing.txt",
    },
}

CONDITIONS = {
    "full": "none",
    "random-yt": "train_random_yt",
    "shuffle-yt": "train_shuffle_yt",
    "core-no-diff": "core_no_diff",
}

DATASETS = {
    "btcv": {
        "image_channels": 1,
        "mask_channels": 1,
        "num_seg_classes": 2,
        "full_steps": 40000,
        "val_batch": {
            "full": 8,
            "random-yt": 8,
            "shuffle-yt": 64,
            "core-no-diff": 64,
        },
        "test_batch": 8,
        "evaluator": "evaluate_btcv_samples.py",
    },
    "acdc": {
        "image_channels": 1,
        "mask_channels": 4,
        "num_seg_classes": 4,
        "full_steps": 60000,
        "val_batch": {
            "full": 64,
            "random-yt": 32,
            "shuffle-yt": 32,
            "core-no-diff": 32,
        },
        "test_batch": 32,
        "evaluator": "evaluate_acdc_samples.py",
    },
    "isic2018": {
        "image_channels": 3,
        "mask_channels": 1,
        "num_seg_classes": 2,
        "full_steps": 60000,
        "val_batch": {
            "full": 32,
            "random-yt": 32,
            "shuffle-yt": 32,
            "core-no-diff": 32,
        },
        "test_batch": 16,
        "evaluator": "evaluate_isic2018_samples.py",
    },
}


def _manifest(dataset: str, split: str) -> Path:
    return (MANIFEST_ROOT / dataset / MANIFEST_NAMES[dataset][split]).resolve()


def _run_dir(args: argparse.Namespace) -> Path:
    return (args.runs_root / args.dataset / args.condition).resolve()


def _gpu_environment(args: argparse.Namespace) -> dict[str, str]:
    return {
        "CUDA_VISIBLE_DEVICES": str(args.cuda_device),
        "KMP_DUPLICATE_LIB_OK": "TRUE",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def _gpu_preflight_command() -> list[str]:
    return [sys.executable, str(GPU_PREFLIGHT)]


def _model_args(config: dict) -> list[str]:
    return [
        "--image_size", "256",
        "--image_channels", str(config["image_channels"]),
        "--mask_channels", str(config["mask_channels"]),
        "--num_seg_classes", str(config["num_seg_classes"]),
        "--num_channels", "128",
        "--num_res_blocks", "2",
        "--num_heads", "1",
        "--num_head_channels", "-1",
        "--num_heads_upsample", "-1",
        "--attention_resolutions", "16",
        "--channel_mult", "1,1,2,2,4,4",
        "--dropout", "0",
        "--class_cond", "False",
        "--use_checkpoint", "False",
        "--resblock_updown", "False",
        "--use_new_attention_order", "False",
        "--learn_sigma", "True",
        "--diffusion_steps", "1000",
        "--noise_schedule", "linear",
        "--use_kl", "False",
        "--predict_xstart", "False",
        "--use_scale_shift_norm", "False",
        "--rescale_timesteps", "False",
        "--rescale_learned_sigmas", "False",
        "--use_fp16", "False",
    ]


def build_train_command(args: argparse.Namespace) -> list[str]:
    config = DATASETS[args.dataset]
    steps = config["full_steps"] if args.condition == "full" else 60000
    run_dir = _run_dir(args)
    command = [
        sys.executable,
        str(SCRIPT_ROOT / "segmentation_train.py"),
        "--dataset", args.dataset,
        "--data_dir", str((args.data_root / "training").resolve()),
        "--train_manifest", str(_manifest(args.dataset, "train")),
        "--val_data_dir", str((args.data_root / "validation").resolve()),
        "--val_manifest", str(_manifest(args.dataset, "val")),
        "--audit_mode", CONDITIONS[args.condition],
        "--seed", str(args.seed),
        "--val_seed", str(args.validation_seed),
        "--batch_size", "8",
        "--microbatch", "-1",
        "--num_workers", str(args.num_workers),
        "--val_num_workers", str(args.validation_workers),
        "--val_batch_size", str(config["val_batch"][args.condition]),
        "--val_num_ensemble", "1",
        "--val_timestep_respacing", "100",
        "--val_use_ddim", "False",
        "--val_output_csv", str(run_dir / "validation_metrics.csv"),
        "--schedule_sampler", "uniform",
        "--lr", "0.0001",
        "--weight_decay", "0",
        "--lr_anneal_steps", str(steps),
        "--ema_rate", "0.9999",
        "--log_interval", "100",
        "--save_interval", "5000",
        "--val_interval", "5000",
    ]
    if args.resume_checkpoint:
        command.extend(["--resume_checkpoint", str(args.resume_checkpoint.resolve())])
    command.extend(_model_args(config))
    return command


def build_test_commands(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    config = DATASETS[args.dataset]
    run_dir = _run_dir(args)
    sample_dir = run_dir / "final_test" / "samples"
    ensemble = 1 if args.condition == "core-no-diff" else 5
    sample = [
        sys.executable,
        str(SCRIPT_ROOT / "segmentation_sample.py"),
        "--dataset", args.dataset,
        "--data_dir", str((args.data_root / "testing").resolve()),
        "--manifest", str(_manifest(args.dataset, "test")),
        "--model_path", str(run_dir / "best_model.pt"),
        "--output_dir", str(sample_dir),
        "--audit_mode", CONDITIONS[args.condition],
        "--seed", str(args.seed),
        "--batch_size", str(config["test_batch"]),
        "--num_workers", str(args.validation_workers),
        "--num_samples", "0",
        "--num_ensemble", str(ensemble),
        "--timestep_respacing", "1000",
        "--use_ddim", "False",
        "--clip_denoised", "True",
    ]
    sample.extend(_model_args(config))

    evaluate = [
        sys.executable,
        str(SCRIPT_ROOT / config["evaluator"]),
        "--data_dir", str((args.data_root / "testing").resolve()),
        "--sample_dir", str(sample_dir),
        "--num_ensemble", str(ensemble),
    ]
    if args.dataset == "acdc":
        evaluate.extend([
            "--num_classes", str(config["num_seg_classes"]),
            "--image_channels", str(config["image_channels"]),
        ])
    elif args.dataset == "isic2018":
        evaluate.extend(["--image_channels", str(config["image_channels"])])
    return sample, evaluate


def _emit(stage: str, command: list[str], env_updates: dict[str, str]) -> None:
    print(json.dumps(
        {
            "stage": stage,
            "cwd": str(CODE_ROOT),
            "environment": env_updates,
            "command": command,
        },
        indent=2,
    ))


def _execute(
    stage: str,
    command: list[str],
    env_updates: dict[str, str],
    dry_run: bool,
) -> None:
    _emit(stage, command, env_updates)
    if dry_run:
        return
    environment = os.environ.copy()
    environment.update(env_updates)
    subprocess.run(command, cwd=CODE_ROOT, env=environment, check=True)


def _require_inputs(args: argparse.Namespace, splits: tuple[str, ...]) -> None:
    manifest_names = {
        "training": "train",
        "validation": "val",
        "testing": "test",
    }
    for split in splits:
        manifest = _manifest(args.dataset, manifest_names[split])
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing fixed {split} manifest: {manifest}")
        data_dir = (args.data_root / split).resolve()
        if not data_dir.is_dir():
            raise FileNotFoundError(
                f"Missing preprocessed {split} directory: {data_dir}"
            )


def _load_best_metadata(args: argparse.Namespace) -> dict:
    run_dir = _run_dir(args)
    metadata_path = run_dir / "best_checkpoint.json"
    checkpoint_path = run_dir / "best_model.pt"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            "Final testing requires validation selection metadata: "
            f"{metadata_path}"
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Final testing requires the selected checkpoint: {checkpoint_path}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("selection_partition") != "validation":
        raise RuntimeError("best_checkpoint.json was not selected on validation data.")
    if metadata.get("checkpoint") != "best_model.pt":
        raise RuntimeError("best_checkpoint.json must explicitly select best_model.pt.")
    expected_mode = CONDITIONS[args.condition]
    if metadata.get("audit_mode") != expected_mode:
        raise RuntimeError(
            "Checkpoint condition mismatch: "
            f"metadata={metadata.get('audit_mode')!r}, expected={expected_mode!r}"
        )
    if metadata.get("training_seed") != args.seed:
        raise RuntimeError(
            "Checkpoint training-seed mismatch: "
            f"metadata={metadata.get('training_seed')!r}, expected={args.seed}"
        )
    if metadata.get("validation_seed") != args.validation_seed:
        raise RuntimeError(
            "Checkpoint validation-seed mismatch: "
            f"metadata={metadata.get('validation_seed')!r}, "
            f"expected={args.validation_seed}"
        )
    recorded_manifest = metadata.get("validation_manifest", "")
    if not recorded_manifest:
        raise RuntimeError("best_checkpoint.json does not record a validation manifest.")
    if Path(recorded_manifest).expanduser().resolve() != _manifest(args.dataset, "val"):
        raise RuntimeError(
            "Checkpoint was not selected with this release's fixed validation manifest."
        )
    return metadata


def run_train(args: argparse.Namespace) -> None:
    run_dir = _run_dir(args)
    if not args.dry_run:
        _require_inputs(args, ("training", "validation"))
        if run_dir.exists() and any(run_dir.iterdir()) and not args.resume_checkpoint:
            raise FileExistsError(
                f"Refusing to mix a new run into non-empty output: {run_dir}"
            )
        run_dir.mkdir(parents=True, exist_ok=True)
    environment = _gpu_environment(args)
    environment["OPENAI_LOGDIR"] = str(run_dir)
    _execute(
        "train_with_validation_selection",
        build_train_command(args),
        environment,
        args.dry_run,
    )


def run_test(args: argparse.Namespace) -> None:
    run_dir = _run_dir(args)
    final_dir = run_dir / "final_test"
    sample_dir = final_dir / "samples"
    metadata = None
    if not args.dry_run:
        _require_inputs(args, ("testing",))
        metadata = _load_best_metadata(args)
        if final_dir.exists() and any(final_dir.iterdir()):
            raise FileExistsError(
                "The one-time final-test directory is not empty; use a new runs root "
                f"instead of selecting on or overwriting test outputs: {final_dir}"
            )
        sample_dir.mkdir(parents=True, exist_ok=True)
        request = {
            "status": "started",
            "dataset": args.dataset,
            "condition": args.condition,
            "audit_mode": CONDITIONS[args.condition],
            "checkpoint": str(run_dir / "best_model.pt"),
            "checkpoint_selection": metadata,
            "test_manifest": str(_manifest(args.dataset, "test")),
            "test_partition": "testing",
        }
        (final_dir / "final_test_request.json").write_text(
            json.dumps(request, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    sample, evaluate = build_test_commands(args)
    env = _gpu_environment(args)
    env["OPENAI_LOGDIR"] = str(final_dir / "logs")
    _execute("sample_selected_checkpoint_on_final_test", sample, env, args.dry_run)
    _execute("evaluate_final_test_once", evaluate, env, args.dry_run)
    if not args.dry_run:
        request_path = final_dir / "final_test_request.json"
        request = json.loads(request_path.read_text(encoding="utf-8"))
        request["status"] = "completed"
        request_path.write_text(
            json.dumps(request, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one of the 12 fixed EnsemDiff experiments. Training selects only "
            "on validation; test loads best_model.pt and never selects a checkpoint."
        )
    )
    parser.add_argument("action", choices=("train", "test", "run"))
    parser.add_argument("--dataset", required=True, choices=tuple(DATASETS))
    parser.add_argument("--condition", required=True, choices=tuple(CONDITIONS))
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--runs-root", type=Path, default=CODE_ROOT.parent / "runs")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--validation-seed", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--validation-workers", type=int, default=0)
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=0,
        help=(
            "Physical CUDA device exposed as logical cuda:0. It must be an exact "
            "NVIDIA GeForce RTX 5090 and pass tests/gpu_preflight.py."
        ),
    )
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print exact commands without checking data, loading models, or writing files.",
    )
    return parser


def main() -> None:
    args = create_parser().parse_args()
    if args.seed < 0 or args.validation_seed < 0:
        raise ValueError("Training, validation, and test seeds must be non-negative.")
    if args.cuda_device < 0:
        raise ValueError("--cuda-device must be non-negative.")
    _execute(
        "gpu_preflight",
        _gpu_preflight_command(),
        _gpu_environment(args),
        args.dry_run,
    )
    if args.action in ("train", "run"):
        run_train(args)
    if args.action in ("test", "run"):
        run_test(args)


if __name__ == "__main__":
    main()
