#!/usr/bin/env python3
"""Run the guarded SDSeg train -> validation selection -> one-time test flow.

This is the only supported formal orchestration entry point in the release.
Training never runs the test partition.  The final test command refuses to run
unless ``best_checkpoint.json`` proves that the requested checkpoint was
selected on ``metric_validation`` by maximum ``val_avg_dice``.

Examples (run from the release ``code/`` directory)::

    python scripts/release_pipeline.py plan --dataset btcv --condition full \
        --data-root DATA --pretrained-root PRETRAINED --runs-root RUNS
    python scripts/release_pipeline.py run --dataset acdc --condition random-yt \
        --data-root DATA --pretrained-root PRETRAINED --runs-root RUNS
    python scripts/release_pipeline.py test --dataset isic2018 --condition core-no-diff \
        --data-root DATA --pretrained-root PRETRAINED --run-dir RUN_DIRECTORY
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from omegaconf import OmegaConf


CODE_ROOT = Path(__file__).resolve().parents[1]
TESTS_ROOT = CODE_ROOT / "tests"
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from validate_splits import validate_release_splits  # noqa: E402


EXPECTED_GPU = "NVIDIA GeForce RTX 5090"
SEED = 23
FORMAL_BATCH_SIZE = 4
FORMAL_NUM_WORKERS = 8
FORMAL_MAX_STEPS = 100_000
FORMAL_LEARNING_RATE = 1e-5


@dataclass(frozen=True)
class ReleaseExperiment:
    dataset: str
    condition: str
    audit_mode: str
    config: Path
    final_dataset: str


AUDIT_MODES = {
    "full": "full_diffusion",
    "random-yt": "train_random_yt",
    "shuffle-yt": "train_shuffle_yt",
    "core-no-diff": "core_no_diff",
}
FINAL_DATASETS = {
    "btcv": "btcv-b",
    "acdc": "acdc",
    "isic2018": "isic2018",
}
DATASET_CLASSES = {
    "btcv": 2,
    "acdc": 4,
    "isic2018": 2,
}
METRIC_DATASET_CLASSES = {
    "btcv": "ldm.data.btcv.BTCVValidationEval",
    "acdc": "ldm.data.acdc.ACDCFullLabelSliceEval",
    "isic2018": "ldm.data.isic2018.ISIC2018ValidationEval",
}


def experiment(dataset: str, condition: str) -> ReleaseExperiment:
    config = CODE_ROOT / "configs" / "experiments" / dataset / f"{condition}.yaml"
    return ReleaseExperiment(
        dataset=dataset,
        condition=condition,
        audit_mode=AUDIT_MODES[condition],
        config=config,
        final_dataset=FINAL_DATASETS[dataset],
    )


def _resolved_config(spec: ReleaseExperiment, pretrained_root: Path | None = None):
    if not spec.config.is_file():
        raise FileNotFoundError(f"Missing experiment config: {spec.config}")
    old_value = os.environ.get("SDSEG_PRETRAINED_ROOT")
    if pretrained_root is not None:
        os.environ["SDSEG_PRETRAINED_ROOT"] = str(pretrained_root.expanduser().resolve())
    try:
        config = OmegaConf.load(spec.config)
        # Resolve only when a root was supplied.  Static planning can validate
        # structure without forcing local pretrained files to exist.
        if pretrained_root is not None:
            config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    finally:
        if old_value is None:
            os.environ.pop("SDSEG_PRETRAINED_ROOT", None)
        else:
            os.environ["SDSEG_PRETRAINED_ROOT"] = old_value
    return config


def validate_formal_config(spec: ReleaseExperiment, pretrained_root: Path | None = None) -> dict:
    """Fail closed if a formal config drifts from the registered experiment."""

    config = _resolved_config(spec, pretrained_root)
    errors: list[str] = []

    def expect(label: str, actual, expected) -> None:
        if actual != expected:
            errors.append(f"{label}: {actual!r} != {expected!r}")

    expect("audit_mode", str(config.model.params.audit_mode), spec.audit_mode)
    expect("random_seed", int(config.model.params.random_seed), SEED)
    expect("num_classes", int(config.model.params.num_classes), DATASET_CLASSES[spec.dataset])
    expect("base_learning_rate", float(config.model.base_learning_rate), FORMAL_LEARNING_RATE)
    expect("batch_size", int(config.data.params.batch_size), FORMAL_BATCH_SIZE)
    expect("num_workers", int(config.data.params.num_workers), FORMAL_NUM_WORKERS)
    expect("max_steps", int(config.lightning.trainer.max_steps), FORMAL_MAX_STEPS)
    expect("monitor", str(config.model.params.monitor), "val_avg_dice")
    if "metric_validation" not in config.data.params:
        errors.append("data.params.metric_validation is missing")
    if "test" not in config.data.params:
        errors.append("data.params.test is missing")
    forbidden_sampling_audit = "sampling" + "_step_audit"
    if forbidden_sampling_audit in config:
        errors.append(f"{forbidden_sampling_audit} is forbidden in the release")

    validation_target = str(config.data.params.metric_validation.target)
    test_target = str(config.data.params.test.target)
    if spec.dataset == "btcv":
        expect("metric_validation.target", validation_target, "ldm.data.btcv.BTCVValidationEval")
        expect("metric_validation.split", str(config.data.params.metric_validation.params.split), "validation")
        expect("test.target", test_target, "ldm.data.btcv.BTCVValidationEval")
        expect("test.split", str(config.data.params.test.params.split), "test")
    elif spec.dataset == "acdc":
        expect("metric_validation.target", validation_target, "ldm.data.acdc.ACDCFullLabelSliceEval")
        expect("metric_validation.split", str(config.data.params.metric_validation.params.split), "validation")
        expect("test.target", test_target, "ldm.data.acdc.ACDCFullLabelSliceEval")
        expect("test.split", str(config.data.params.test.params.split), "test")
    else:
        expect("metric_validation.target", validation_target, "ldm.data.isic2018.ISIC2018ValidationEval")
        expect("test.target", test_target, "ldm.data.isic2018.ISIC2018Test")

    if errors:
        raise ValueError("Formal config validation failed:\n- " + "\n- ".join(errors))
    return {
        "dataset": spec.dataset,
        "condition": spec.condition,
        "audit_mode": spec.audit_mode,
        "config": spec.config.relative_to(CODE_ROOT).as_posix(),
        "seed": SEED,
        "batch_size": FORMAL_BATCH_SIZE,
        "num_workers": FORMAL_NUM_WORKERS,
        "max_steps": FORMAL_MAX_STEPS,
        "learning_rate": FORMAL_LEARNING_RATE,
        "scale_lr": False,
        "selection_partition": "validation",
        "metric_dataset_key": "metric_validation",
        "monitor": "val_avg_dice",
        "mode": "max",
        "final_partition": "test",
    }


def validate_selection_metadata(path: Path) -> tuple[dict, Path]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing selection metadata: {path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "selection_partition": "validation",
        "metric_dataset_key": "metric_validation",
        "monitor": "val_avg_dice",
        "mode": "max",
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"Selection metadata {key} must be {expected!r}, got {metadata.get(key)!r}"
            )
    if metadata.get("best_model_score") is None:
        raise ValueError("Selection metadata has no best_model_score.")
    raw_checkpoint = metadata.get("best_model_path")
    if not raw_checkpoint:
        raise ValueError("Selection metadata has no best_model_path.")
    checkpoint = Path(raw_checkpoint).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = (path.parent / checkpoint).resolve()
    else:
        checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Validation-selected checkpoint does not exist: {checkpoint}")
    return metadata, checkpoint


def validate_selection_binding(
    spec: ReleaseExperiment,
    metadata_path: Path,
    metadata: dict,
    checkpoint: Path,
    run_dir: Path | None = None,
) -> Path:
    """Bind selection evidence to the requested dataset and audit condition."""

    expected = {
        "audit_mode": spec.audit_mode,
        "metric_validation_dataset_class": METRIC_DATASET_CLASSES[spec.dataset],
        "num_classes": DATASET_CLASSES[spec.dataset],
        "random_seed": SEED,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"Selection metadata {key} must be {value!r} for "
                f"{spec.dataset}/{spec.condition}, got {metadata.get(key)!r}."
            )

    selected_run = (run_dir or metadata_path.parent).expanduser().resolve()
    if not selected_run.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {selected_run}")
    try:
        checkpoint.relative_to(selected_run)
    except ValueError as exc:
        raise ValueError(
            f"Selected checkpoint is outside its bound run directory: {checkpoint}"
        ) from exc

    project_configs = sorted((selected_run / "configs").glob("*-project.yaml"))
    if len(project_configs) != 1:
        raise ValueError(
            f"Expected exactly one saved project config under {selected_run / 'configs'}, "
            f"found {len(project_configs)}."
        )
    saved = OmegaConf.load(project_configs[0])
    if str(saved.model.params.audit_mode) != spec.audit_mode:
        raise ValueError("Saved project config audit_mode does not match the requested condition.")
    if int(saved.model.params.num_classes) != DATASET_CLASSES[spec.dataset]:
        raise ValueError("Saved project config num_classes does not match the requested dataset.")
    saved_metric_class = str(saved.data.params.metric_validation.target)
    if saved_metric_class != METRIC_DATASET_CLASSES[spec.dataset]:
        raise ValueError(
            f"Saved metric-validation dataset {saved_metric_class!r} does not match "
            f"{METRIC_DATASET_CLASSES[spec.dataset]!r}."
        )
    return selected_run


def _display_command(command: list[str]) -> str:
    try:
        return shlex.join(command)
    except AttributeError:  # pragma: no cover - Python 3.8 compatibility
        return subprocess.list2cmdline(command)


def _run(command: list[str], *, env: dict[str, str], dry_run: bool) -> None:
    print(f"$ {_display_command(command)}", flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=CODE_ROOT, env=env, check=True)


def _environment(data_root: Path, pretrained_root: Path | None) -> dict[str, str]:
    env = os.environ.copy()
    env["SDSEG_DATA_ROOT"] = str(data_root.expanduser().resolve())
    if pretrained_root is not None:
        env["SDSEG_PRETRAINED_ROOT"] = str(pretrained_root.expanduser().resolve())
    return env


def _preflight_command(spec: ReleaseExperiment, args: argparse.Namespace) -> list[str]:
    if args.pretrained_root is None:
        raise ValueError("--pretrained-root is required before training/preflight.")
    return [
        sys.executable,
        str(TESTS_ROOT / "gpu_preflight.py"),
        "--config", str(spec.config),
        "--pretrained-root", str(args.pretrained_root.expanduser().resolve()),
        "--cuda-device", "0",
        "--expected-gpu-name", EXPECTED_GPU,
    ]


def _train_command(spec: ReleaseExperiment, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(CODE_ROOT / "main.py"),
        "--base", str(spec.config),
        "--train", "true",
        "--no-test", "true",
        "--seed", str(SEED),
        "--name", f"{spec.dataset}-{spec.condition}",
        "--logdir", str(args.runs_root.expanduser().resolve()),
        "--data-root", str(args.data_root.expanduser().resolve()),
        "--pretrained-root", str(args.pretrained_root.expanduser().resolve()),
        "--expected-gpu-name", EXPECTED_GPU,
        "--scale_lr", "false",
        "--gpus", "0,",
        "--max_steps", str(FORMAL_MAX_STEPS),
    ]


def _find_new_run(runs_root: Path, before: set[Path], spec: ReleaseExperiment) -> Path:
    candidates = [path for path in runs_root.iterdir() if path.is_dir() and path not in before]
    suffix = f"_{spec.dataset}-{spec.condition}"
    matching = [path for path in candidates if path.name.endswith(suffix)]
    if len(matching) != 1:
        raise RuntimeError(
            f"Expected exactly one new run directory ending in {suffix!r}; found {matching}"
        )
    return matching[0].resolve()


def _find_selection_metadata(run_dir: Path) -> Path:
    candidates = sorted(run_dir.rglob("best_checkpoint.json"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one best_checkpoint.json under {run_dir}; found {len(candidates)}"
        )
    return candidates[0]


def run_training(spec: ReleaseExperiment, args: argparse.Namespace) -> Path | None:
    if args.pretrained_root is None:
        raise ValueError("--pretrained-root is required for training.")
    validate_formal_config(spec, args.pretrained_root)
    validate_release_splits(args.data_root)
    env = _environment(args.data_root, args.pretrained_root)
    _run(_preflight_command(spec, args), env=env, dry_run=args.dry_run)

    runs_root = args.runs_root.expanduser().resolve()
    before = set()
    if runs_root.is_dir():
        before = {path.resolve() for path in runs_root.iterdir() if path.is_dir()}
    if not args.dry_run:
        runs_root.mkdir(parents=True, exist_ok=True)
    _run(_train_command(spec, args), env=env, dry_run=args.dry_run)
    if args.dry_run:
        return None

    run_dir = _find_new_run(runs_root, before, spec)
    metadata_path = _find_selection_metadata(run_dir)
    metadata, checkpoint = validate_selection_metadata(metadata_path)
    validate_selection_binding(spec, metadata_path, metadata, checkpoint, run_dir=run_dir)
    print(f"Validation-selected run: {run_dir}")
    return run_dir


def _test_command(
    spec: ReleaseExperiment,
    args: argparse.Namespace,
    metadata_path: Path,
    checkpoint: Path,
    outdir: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(CODE_ROOT / "scripts" / "slice2seg.py"),
        "--dataset", spec.final_dataset,
        "--config", str(spec.config),
        "--ckpt", str(checkpoint),
        "--selection-metadata", str(metadata_path),
        "--outdir", str(outdir),
        "--data-root", str(args.data_root.expanduser().resolve()),
        "--seed", str(SEED),
        "--sampler", "direct",
        "--ddim_steps", "1",
        "--ddim_eta", "0.0",
        "--times", "1",
        "--device", "cuda",
        "--expected-gpu-name", EXPECTED_GPU,
    ]
    if args.save_results:
        command.append("--save_results")
    return command


def run_final_test(
    spec: ReleaseExperiment,
    args: argparse.Namespace,
    run_dir: Path | None = None,
) -> Path | None:
    validate_formal_config(spec, args.pretrained_root)
    validate_release_splits(args.data_root)

    selected_run = run_dir or args.run_dir
    if args.selection_metadata is not None:
        metadata_path = args.selection_metadata.expanduser().resolve()
    else:
        if selected_run is None:
            raise ValueError("Final test requires --run-dir or --selection-metadata.")
        selected_run = selected_run.expanduser().resolve()
        metadata_path = _find_selection_metadata(selected_run)
    metadata, checkpoint = validate_selection_metadata(metadata_path)
    selected_run = validate_selection_binding(
        spec,
        metadata_path,
        metadata,
        checkpoint,
        run_dir=selected_run,
    )

    if args.outdir is not None:
        outdir = args.outdir.expanduser().resolve()
    else:
        if selected_run is None:
            raise ValueError("--outdir is required when no run binding can be resolved.")
        outdir = selected_run.expanduser().resolve() / "final_test"

    # Existence, not merely non-emptiness, is the one-time boundary.  A failed
    # attempt therefore cannot silently be reclassified as the final result.
    if outdir.exists():
        raise FileExistsError(f"Final-test output path already exists; refusing a second test: {outdir}")

    env = _environment(args.data_root, args.pretrained_root)
    command = _test_command(spec, args, metadata_path, checkpoint, outdir)
    _run(command, env=env, dry_run=args.dry_run)
    if args.dry_run:
        return None

    summary = outdir / str(SEED) / "metrics_summary.json"
    if not summary.is_file():
        raise RuntimeError(f"Final test completed without its required summary: {summary}")
    result = json.loads(summary.read_text(encoding="utf-8"))
    if result.get("evaluation_partition") != "test":
        raise RuntimeError("Final evaluation did not declare evaluation_partition=test.")
    if result.get("selection_partition") != "validation":
        raise RuntimeError("Final evaluation did not declare selection_partition=validation.")
    if result.get("is_smoke_test"):
        raise RuntimeError("Formal final evaluation was unexpectedly marked as a smoke test.")

    receipt = {
        "status": "pass",
        "dataset": spec.dataset,
        "condition": spec.condition,
        "audit_mode": spec.audit_mode,
        "selection_partition": "validation",
        "final_partition": "test",
        "seed": SEED,
        "sampler": "direct",
        "ddim_steps": 1,
        "times": 1,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    (outdir / "final_test_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"One-time final test completed: {outdir}")
    return outdir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "validate", "train", "test", "run"])
    parser.add_argument("--dataset", required=True, choices=sorted(FINAL_DATASETS))
    parser.add_argument("--condition", required=True, choices=sorted(AUDIT_MODES))
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--pretrained-root", required=True, type=Path)
    parser.add_argument("--runs-root", type=Path, default=CODE_ROOT / "runs")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--selection-metadata", type=Path)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--save-results", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = experiment(args.dataset, args.condition)
    formal = validate_formal_config(spec, args.pretrained_root)
    print(json.dumps(formal, indent=2, sort_keys=True))

    if args.action in {"plan", "validate", "train", "run"}:
        validate_release_splits(args.data_root)
    env = _environment(args.data_root, args.pretrained_root)

    if args.action == "plan":
        if args.pretrained_root is None:
            raise ValueError("--pretrained-root is required to plan the executable train command.")
        print(_display_command(_preflight_command(spec, args)))
        print(_display_command(_train_command(spec, args)))
        return 0
    if args.action == "validate":
        _run(_preflight_command(spec, args), env=env, dry_run=args.dry_run)
        return 0
    if args.action == "train":
        run_training(spec, args)
        return 0
    if args.action == "test":
        run_final_test(spec, args)
        return 0
    if args.action == "run":
        run_dir = run_training(spec, args)
        if args.dry_run:
            print("Dry run stops before resolving a validation-selected checkpoint.")
            return 0
        run_final_test(spec, args, run_dir=run_dir)
        return 0
    raise AssertionError(f"Unhandled action: {args.action}")


if __name__ == "__main__":
    raise SystemExit(main())
