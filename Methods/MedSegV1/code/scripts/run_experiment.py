"""Run one fixed MedSegDiff V1 train/validation/best/test experiment."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch


RELEASE_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = RELEASE_ROOT / "code"
CONFIG_PATH = CODE_ROOT / "configs" / "experiments.json"


def flag(value):
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def require_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU fallback is not permitted")
    if torch.cuda.device_count() < 1:
        raise RuntimeError("No CUDA device was detected")
    torch.cuda.set_device(0)
    name = torch.cuda.get_device_name(0)
    if name != "NVIDIA GeForce RTX 5090":
        raise RuntimeError(f"Expected NVIDIA GeForce RTX 5090 at cuda:0, found {name!r}")


def load_config(dataset, condition):
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return config, config["datasets"][dataset], config["conditions"][condition]


def existing(root, candidates):
    for candidate in candidates:
        path = root / candidate
        if path.is_dir():
            return path.resolve()
    raise FileNotFoundError(f"No expected dataset directory under {root}: {candidates}")


def data_paths(dataset, dataset_config, root):
    root = root.expanduser().resolve()
    if dataset == "btcv":
        train = existing(root, (Path(dataset_config["train_subdir"]), Path("BTCV/training"), Path("train")))
        combined_test = existing(root, (Path("BTCV/test"), Path("BTCV/testing"), Path("test"), Path("testing")))
        val_candidate = root / dataset_config["val_subdir"]
        val = val_candidate.resolve() if val_candidate.is_dir() else combined_test
        test = existing(root, (Path(dataset_config["test_subdir"]), Path("BTCV/testing"), Path("test")))
        return {"train": train, "val": val, "test": test}
    if dataset == "acdc":
        return {"train": root, "val": root, "test": root}
    if dataset == "isic2018":
        return {
            "train": existing(root, (Path(dataset_config["train_subdir"]), Path("training"))),
            "val": existing(root, (Path(dataset_config["val_subdir"]), Path("validation"))),
            "test": existing(root, (Path(dataset_config["test_subdir"]), Path("testing"))),
        }
    raise ValueError(dataset)


def common_model_args(common, dataset_config):
    return [
        "--gpu_dev", "0",
        "--version", common["version"],
        "--image_size", flag(common["image_size"]),
        "--num_channels", flag(common["num_channels"]),
        "--num_res_blocks", flag(common["num_res_blocks"]),
        "--num_heads", flag(common["num_heads"]),
        "--learn_sigma", flag(common["learn_sigma"]),
        "--class_cond", flag(common["class_cond"]),
        "--use_scale_shift_norm", flag(common["use_scale_shift_norm"]),
        "--attention_resolutions", common["attention_resolutions"],
        "--diffusion_steps", flag(common["diffusion_steps"]),
        "--noise_schedule", common["noise_schedule"],
        "--rescale_learned_sigmas", flag(common["rescale_learned_sigmas"]),
        "--rescale_timesteps", flag(common["rescale_timesteps"]),
        "--num_seg_classes", flag(dataset_config["num_seg_classes"]),
        "--num_mask_channels", flag(dataset_config["num_mask_channels"]),
    ]


def manifest_path(relative_path):
    path = (RELEASE_ROOT / relative_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def train_command(python, dataset, dataset_config, condition_config, common, paths, run_dir):
    audit_mode = condition_config["audit_mode"]
    command = [
        python,
        str(CODE_ROOT / "scripts" / "segmentation_train.py"),
        "--data_name", dataset_config["data_name"],
        "--data_dir", str(paths["train"]),
        "--data_manifest", str(manifest_path(dataset_config["train_manifest"])),
        "--out_dir", str(run_dir),
        "--seed", flag(common["seed"]),
        "--lr", flag(common["learning_rate"]),
        "--weight_decay", flag(common["weight_decay"]),
        "--batch_size", flag(common["training_batch_size"]),
        "--lr_anneal_steps", flag(common["training_steps"]),
        "--save_interval", flag(common["save_interval"]),
        "--log_interval", flag(common["log_interval"]),
        "--num_workers", flag(common["num_workers"]),
        "--pin_memory", flag(common["pin_memory"]),
        "--persistent_workers", flag(common["persistent_workers"]),
        "--prefetch_factor", flag(common["prefetch_factor"]),
        "--dpm_solver", "False",
        "--validation_runner", "True",
        "--val_data_dir", str(paths["val"]),
        "--val_manifest", str(manifest_path(dataset_config["val_manifest"])),
        "--val_out_csv", str(run_dir / "validation_metrics.csv"),
        "--val_dpm_solver", "True",
        "--val_diffusion_steps", flag(dataset_config["validation_sampling_steps"]),
        "--val_num_ensemble", "1",
        "--val_batch_size", flag(dataset_config["validation_batch_size"]),
        "--val_pred_threshold", "0.5",
        "--save_best_only", "True",
        "--best_metric_name", common["best_metric"],
        "--audit_mode", audit_mode,
    ]
    if dataset == "acdc":
        command.extend(["--acdc_split", "training", "--val_acdc_split", "validation"])
    command.extend(common_model_args(common, dataset_config))
    return command


def selected_checkpoint(run_dir):
    metadata_path = run_dir / "best_checkpoint_meta.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Missing {metadata_path}; final test requires validation-selected checkpoint metadata"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("best_metric_name") != "dice_mean":
        raise RuntimeError(f"Unexpected best metric metadata: {metadata}")
    step = int(metadata["best_step"])
    checkpoint = run_dir / f"savedmodel{step:06d}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint, metadata


def test_commands(python, dataset, dataset_config, condition_config, common, paths, run_dir):
    checkpoint, metadata = selected_checkpoint(run_dir)
    test_dir = run_dir / "final_test"
    test_dir.mkdir(parents=True, exist_ok=True)
    core = condition_config["audit_mode"] == "core_no_diff"
    steps = 0 if core else dataset_config["final_sampling_steps"]
    ensemble = 1 if core else dataset_config["final_ensemble"]
    summary_csv = test_dir / "metrics_summary.csv"
    summary_json = test_dir / "metrics_summary.json"
    per_slice_csv = test_dir / "metrics_per_slice.csv"
    sample = [
        python,
        str(CODE_ROOT / "scripts" / "segmentation_sample.py"),
        "--data_name", dataset_config["data_name"],
        "--data_dir", str(paths["test"]),
        "--data_manifest", str(manifest_path(dataset_config["test_manifest"])),
        "--model_path", str(checkpoint),
        "--out_dir", str(test_dir),
        "--num_samples", "-1",
        "--batch_size", flag(dataset_config["final_batch_size"]),
        "--num_workers", "0",
        "--num_ensemble", flag(ensemble),
        "--sample_steps", flag(steps),
        "--dpm_solver", "False",
        "--use_ddim", "False",
        "--compute_metrics", flag(dataset != "acdc"),
        "--pred_threshold", "0.5",
        "--metrics_summary_csv", str(summary_csv),
        "--metrics_summary_json", str(summary_json),
        "--metrics_per_slice_csv", str(per_slice_csv),
        "--audit_mode", condition_config["audit_mode"],
        "--binary_output", "sample",
    ]
    if dataset == "acdc":
        sample.extend(["--acdc_split", "testing", "--acdc_threshold", "0.5", "--acdc_output", "sample"])
    sample.extend(common_model_args(common, dataset_config))
    commands = [sample]
    if dataset == "acdc":
        commands.append([
            python,
            str(CODE_ROOT / "scripts" / "evaluate_acdc_multiclass.py"),
            "--pred_dir", str(test_dir),
            "--threshold", "0.5",
            "--num_classes", "4",
            "--out_csv", str(summary_csv),
            "--out_json", str(summary_json),
        ])
    final_metadata = {
        "partition": "test",
        "checkpoint_source": "validation",
        "best_step": metadata["best_step"],
        "best_metric_name": metadata["best_metric_name"],
        "best_metric": metadata["best_metric"],
        "checkpoint": checkpoint.name,
        "audit_mode": condition_config["audit_mode"],
        "sampling_steps": steps,
        "num_ensemble": ensemble,
        "test_manifest": str(Path(dataset_config["test_manifest"])),
    }
    return commands, test_dir / "final_test_metadata.json", final_metadata


def render(command):
    return subprocess.list2cmdline([str(part) for part in command])


def run(command, dry_run):
    print(render(command), flush=True)
    if not dry_run:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(CODE_ROOT)
        subprocess.run(command, cwd=CODE_ROOT, env=env, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("btcv", "acdc", "isic2018"), required=True)
    parser.add_argument("--condition", choices=("full", "train_random_yt", "train_shuffle_yt", "core_no_diff"), required=True)
    parser.add_argument("--stage", choices=("train", "test", "all"), default="all")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.dry_run:
        require_gpu()
    config, dataset_config, condition_config = load_config(args.dataset, args.condition)
    common = config["common"]
    paths = data_paths(args.dataset, dataset_config, args.data_root)
    run_dir = (args.output_root / args.dataset / args.condition / f"seed_{common['seed']}").resolve()
    if not args.dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
        resolved = {
            "dataset": args.dataset,
            "condition": args.condition,
            "common": common,
            "dataset_config": dataset_config,
            "condition_config": condition_config,
            "data_paths": {key: str(value) for key, value in paths.items()},
        }
        (run_dir / "resolved_config.json").write_text(
            json.dumps(resolved, indent=2) + "\n", encoding="utf-8"
        )

    if args.stage in {"train", "all"}:
        run(train_command(args.python, args.dataset, dataset_config, condition_config, common, paths, run_dir), args.dry_run)
    if args.stage in {"test", "all"}:
        commands, metadata_path, metadata = test_commands(
            args.python, args.dataset, dataset_config, condition_config, common, paths, run_dir
        )
        for command in commands:
            run(command, args.dry_run)
        if not args.dry_run:
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
