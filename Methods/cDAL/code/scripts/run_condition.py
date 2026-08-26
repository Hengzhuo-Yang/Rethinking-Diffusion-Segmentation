"""Run one formal train -> validation selection -> final-test cDAL condition."""

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"

CONDITIONS = {
    "full": "none",
    "random-yt": "train_random_yt",
    "shuffle-yt": "train_shuffle_yt",
    "core-no-diff": "core_no_diff",
}

DATASETS = {
    "btcv": {
        "train_split": "train",
        "val_split": "validation",
        "test_split": "test",
        "train_manifest": "manifests/btcv/train_cases.txt",
        "val_manifest": "manifests/btcv/validation_cases.txt",
        "test_manifest": "manifests/btcv/test_cases.txt",
        "evaluate": "evaluate_btcv_cdal.py",
        "parameters": "parameters_btcv.json",
        "per_item_flag": "--per_slice_csv",
    },
    "acdc": {
        "train_split": "training",
        "val_split": "validation",
        "test_split": "testing",
        "train_manifest": "manifests/acdc/training_subjects.txt",
        "val_manifest": "manifests/acdc/validation_subjects.txt",
        "test_manifest": "manifests/acdc/test_subjects.txt",
        "evaluate": "evaluate_acdc_cdal.py",
        "parameters": "parameters_acdc.json",
        "per_item_flag": "--per_slice_csv",
    },
    "isic2018": {
        "train_split": "train",
        "val_split": "validation",
        "test_split": "testing",
        "train_manifest": "manifests/isic2018/training_images.txt",
        "val_manifest": "manifests/isic2018/validation_images.txt",
        "test_manifest": "manifests/isic2018/test_images.txt",
        "evaluate": "evaluate_isic2018_cdal.py",
        "parameters": "parameters_isic2018.json",
        "per_item_flag": "--per_image_csv",
    },
}


def run(command):
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASETS))
    parser.add_argument("--condition", required=True, choices=list(CONDITIONS))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    spec = DATASETS[args.dataset]
    audit_mode = CONDITIONS[args.condition]
    experiment = f"{args.dataset}_{args.condition.replace('-', '_')}"
    experiment_root = Path(args.output_root) / args.dataset / experiment
    if experiment_root.exists() and any(experiment_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty experiment: {experiment_root}")

    train = [
        sys.executable,
        str(CODE_ROOT / "train_cDal_monu_and_lung.py"),
        "--dataset", args.dataset,
        "--audit_mode", audit_mode,
        "--data_dir", args.data_root,
        "--train_split", spec["train_split"],
        "--val_split", spec["val_split"],
        "--train_manifest", spec["train_manifest"],
        "--val_manifest", spec["val_manifest"],
        "--output_dir", args.output_root,
        "--exp", experiment,
        "--local_rank", str(args.device),
    ]
    run(train)

    checkpoint = experiment_root / "best_checkpoint.pt"
    metadata = experiment_root / "best_checkpoint_meta.json"
    final_root = experiment_root / "final_test"
    evaluate = [
        sys.executable,
        str(CODE_ROOT / spec["evaluate"]),
        "--parameters", str(CODE_ROOT / spec["parameters"]),
        "--model_path", str(checkpoint),
        "--checkpoint_metadata", str(metadata),
        "--data_dir", args.data_root,
        "--manifest", spec["test_manifest"],
        "--split", spec["test_split"],
        "--output_folder", str(final_root / "artifacts"),
        "--summary_csv", str(final_root / "summary.csv"),
        "--summary_json", str(final_root / "summary.json"),
        spec["per_item_flag"], str(final_root / "per_item.csv"),
        "--audit_mode", audit_mode,
        "--major_vote_number", "5",
        "--local_rank", str(args.device),
    ]
    run(evaluate)


if __name__ == "__main__":
    main()
