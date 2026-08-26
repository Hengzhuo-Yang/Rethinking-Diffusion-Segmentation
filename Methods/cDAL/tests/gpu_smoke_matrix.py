import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT / "code"
EXPECTED_GPU = "NVIDIA GeForce RTX 5090"

CONDITIONS = {
    "full": "none",
    "random-yt": "train_random_yt",
    "shuffle-yt": "train_shuffle_yt",
    "core-no-diff": "core_no_diff",
}

DATASETS = {
    "btcv": {
        "parameters": "parameters_btcv.json",
        "train_split": "train",
        "val_split": "validation",
        "test_split": "test",
        "train_manifest": "manifests/btcv/train_cases.txt",
        "val_manifest": "manifests/btcv/validation_cases.txt",
        "test_manifest": "manifests/btcv/test_cases.txt",
        "evaluate": "evaluate_btcv_cdal.py",
        "per_item_flag": "--per_slice_csv",
    },
    "acdc": {
        "parameters": "parameters_acdc.json",
        "train_split": "training",
        "val_split": "validation",
        "test_split": "testing",
        "train_manifest": "manifests/acdc/training_subjects.txt",
        "val_manifest": "manifests/acdc/validation_subjects.txt",
        "test_manifest": "manifests/acdc/test_subjects.txt",
        "evaluate": "evaluate_acdc_cdal.py",
        "per_item_flag": "--per_slice_csv",
    },
    "isic2018": {
        "parameters": "parameters_isic2018.json",
        "train_split": "train",
        "val_split": "validation",
        "test_split": "testing",
        "train_manifest": "manifests/isic2018/training_images.txt",
        "val_manifest": "manifests/isic2018/validation_images.txt",
        "test_manifest": "manifests/isic2018/test_images.txt",
        "evaluate": "evaluate_isic2018_cdal.py",
        "per_item_flag": "--per_image_csv",
    },
}


def run_command(command, timeout):
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        tail = "\n".join((result.stdout + "\n" + result.stderr).splitlines()[-40:])
        raise RuntimeError(tail)


def check_cuda_device(value, label):
    if not str(value).startswith("cuda:"):
        raise RuntimeError(f"{label} is not CUDA: {value!r}")


def execute_combination(dataset, condition, data_root, work_root, timeout):
    spec = DATASETS[dataset]
    audit_mode = CONDITIONS[condition]
    key = f"{dataset}__{condition}"
    combo_root = work_root / key
    combo_root.mkdir(parents=True, exist_ok=False)
    output_dir = combo_root / "outputs"
    evidence_path = combo_root / "training_evidence.json"
    exp_name = f"gpu_smoke_{key.replace('-', '_')}"
    smoke_batch_size = 2 if condition == "shuffle-yt" else 1

    train_command = [
        sys.executable,
        str(CODE_ROOT / "train_cDal_monu_and_lung.py"),
        "--dataset", dataset,
        "--data_dir", str(data_root),
        "--train_split", spec["train_split"],
        "--val_split", spec["val_split"],
        "--train_manifest", str(REPO_ROOT / spec["train_manifest"]),
        "--val_manifest", str(REPO_ROOT / spec["val_manifest"]),
        "--output_dir", str(output_dir),
        "--exp", exp_name,
        "--audit_mode", audit_mode,
        "--batch_size", str(smoke_batch_size),
        "--max_train_steps", "1",
        "--val_interval_steps", "1",
        "--lr_decay_steps", "1",
        "--major_vote_number", "1",
        "--eval_batch_size", "1",
        "--eval_max_items", "1",
        "--num_workers", "0",
        "--local_rank", "0",
        "--save_visuals", "false",
        "--save_content", "false",
        "--log_step", "1",
        "--gpu_evidence_json", str(evidence_path),
    ]
    run_command(train_command, timeout)

    experiment_root = output_dir / dataset / exp_name
    checkpoint = experiment_root / "best_checkpoint.pt"
    metadata = experiment_root / "best_checkpoint_meta.json"
    if not checkpoint.is_file() or not metadata.is_file():
        raise RuntimeError("Validation did not save the temporary best checkpoint and metadata")

    test_root = combo_root / "test"
    test_summary = combo_root / "test_summary.json"
    test_command = [
        sys.executable,
        str(CODE_ROOT / spec["evaluate"]),
        "--parameters", str(CODE_ROOT / spec["parameters"]),
        "--model_path", str(checkpoint),
        "--checkpoint_metadata", str(metadata),
        "--data_dir", str(data_root),
        "--manifest", str(REPO_ROOT / spec["test_manifest"]),
        "--split", spec["test_split"],
        "--output_folder", str(test_root),
        "--summary_csv", str(combo_root / "test_summary.csv"),
        "--summary_json", str(test_summary),
        spec["per_item_flag"], str(combo_root / "test_per_item.csv"),
        "--audit_mode", audit_mode,
        "--major_vote_number", "1",
        "--eval_batch_size", "1",
        "--eval_max_items", "1",
        "--local_rank", "0",
    ]
    run_command(test_command, timeout)

    training = json.loads(evidence_path.read_text(encoding="utf-8"))
    testing = json.loads(test_summary.read_text(encoding="utf-8"))
    required_training = (
        "forward_completed",
        "backward_completed",
        "optimizer_step_completed",
        "validation_forward_completed",
        "validation_metric_computed",
        "best_checkpoint_saved",
    )
    for field in required_training:
        if training.get(field) is not True:
            raise RuntimeError(f"Training evidence field {field} did not pass")
    if training.get("gpu_name") != EXPECTED_GPU or testing.get("gpu_name") != EXPECTED_GPU:
        raise RuntimeError("Smoke test did not execute on the required RTX 5090")
    for field in (
        "model_parameter_device",
        "training_input_device",
        "training_output_device",
        "loss_device",
        "validation_model_device",
        "validation_input_device",
        "validation_output_device",
    ):
        check_cuda_device(training.get(field), field)
    for field in ("model_parameter_device", "input_device", "output_device"):
        check_cuda_device(testing.get(field), f"test_{field}")
    if training.get("audit_branch_executed") != audit_mode:
        raise RuntimeError("Requested audit branch was not recorded as executed")
    if training.get("validation_split") != "validation":
        raise RuntimeError("Best checkpoint was not selected on validation")
    if testing.get("split") != "test" or testing.get("checkpoint_loaded") is not True:
        raise RuntimeError("Final smoke inference did not load the validation-selected checkpoint on test")

    return {
        "dataset": dataset,
        "condition": condition,
        "audit_mode": audit_mode,
        "status": "PASS",
        "formal_batch_size": 4,
        "smoke_training_batch_size": smoke_batch_size,
        "smoke_validation_batch_size": 1,
        "smoke_test_batch_size": 1,
        "train_manifest": spec["train_manifest"],
        "validation_manifest": spec["val_manifest"],
        "test_manifest": spec["test_manifest"],
        "gpu_name": training["gpu_name"],
        "cuda_device": training["cuda_device"],
        "model_parameter_device": training["model_parameter_device"],
        "training_input_device": training["training_input_device"],
        "training_output_device": training["training_output_device"],
        "loss_device": training["loss_device"],
        "forward": True,
        "backward": True,
        "optimizer_step": True,
        "validation_forward": True,
        "validation_metric": True,
        "best_checkpoint_save": True,
        "best_checkpoint_reload": True,
        "test_inference": True,
        "test_output_device": testing["output_device"],
        "audit_branch_executed": True,
        "peak_cuda_allocated_bytes": max(
            int(training.get("peak_cuda_allocated_bytes", 0)),
            int(float(testing.get("cuda_max_memory_allocated_mib", 0)) * 1024 * 1024),
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--btcv-root", required=True)
    parser.add_argument("--acdc-root", required=True)
    parser.add_argument("--isic-root", required=True)
    parser.add_argument("--work-dir", default=".smoke_work")
    parser.add_argument("--report", default="gpu_smoke_report.json")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--keep-work", action="store_true")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--conditions", nargs="+", choices=list(CONDITIONS), default=list(CONDITIONS))
    args = parser.parse_args()

    roots = {
        "btcv": Path(args.btcv_root),
        "acdc": Path(args.acdc_root),
        "isic2018": Path(args.isic_root),
    }
    work_root = Path(args.work_dir)
    if work_root.exists():
        raise FileExistsError(f"Smoke work directory must not already exist: {work_root}")
    work_root.mkdir(parents=True)

    results = []
    try:
        for dataset in args.datasets:
            for condition in args.conditions:
                key = f"{dataset} x {condition}"
                print(f"START {key}", flush=True)
                try:
                    result = execute_combination(
                        dataset,
                        condition,
                        roots[dataset],
                        work_root,
                        args.timeout,
                    )
                    print(f"PASS  {key}", flush=True)
                except Exception as error:
                    result = {
                        "dataset": dataset,
                        "condition": condition,
                        "audit_mode": CONDITIONS[condition],
                        "status": "FAIL",
                        "failure_stage": "GPU end-to-end smoke",
                        "error": str(error),
                    }
                    print(f"FAIL  {key}: {error}", flush=True)
                results.append(result)
    finally:
        report = {
            "gpu_required": EXPECTED_GPU,
            "all_passed": (
                all(item["status"] == "PASS" for item in results)
                and len(results) == len(args.datasets) * len(args.conditions)
            ),
            "combinations": results,
        }
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
        if not args.keep_work and work_root.is_dir():
            shutil.rmtree(work_root)

    if not report["all_passed"]:
        raise SystemExit(1)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
