#!/usr/bin/env python3
"""Static release checks that require only the Python standard library."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


sys.dont_write_bytecode = True
TEST_ROOT = Path(__file__).resolve().parent
RELEASE_ROOT = TEST_ROOT.parent
CODE_ROOT = RELEASE_ROOT / "code"
SCRIPT_ROOT = CODE_ROOT / "scripts"
sys.path.insert(0, str(TEST_ROOT))

import validate_splits  # noqa: E402
sys.path.insert(0, str(SCRIPT_ROOT))
import release_pipeline  # noqa: E402


DATASETS = ("btcv", "acdc", "isic2018")
CONDITIONS = {
    "full": "none",
    "random-yt": "train_random_yt",
    "shuffle-yt": "train_shuffle_yt",
    "core-no-diff": "core_no_diff",
}
EVALUATORS = {
    "btcv": "evaluate_btcv_samples.py",
    "acdc": "evaluate_acdc_samples.py",
    "isic2018": "evaluate_isic2018_samples.py",
}
MANIFEST_FILES = {
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

TEXT_SUFFIXES = {
    ".py", ".ps1", ".sh", ".bat", ".cmd", ".md", ".txt", ".yml", ".yaml",
    ".json", ".toml", ".ini", ".cfg", ".csv",
}
WINDOWS_ABSOLUTE = re.compile(r"(?i)(?<![A-Za-z0-9])[A-Z]:[\\/]")
PRIVATE_POSIX = re.compile(
    r"(?i)(?<![A-Za-z0-9:])/(?:home|root|workspace|Users)(?:/|\\)"
)
BANNED_DIRECTORIES = {
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "data", "datasets", "weights", "checkpoints", "runs", "results", "logs",
}
BANNED_SUFFIXES = {
    ".pt", ".pth", ".ckpt", ".onnx", ".h5", ".hdf5", ".npy", ".npz",
    ".pkl", ".pickle", ".pyc", ".pyo", ".log", ".zip", ".7z", ".tar",
    ".tgz", ".new", ".bak", ".orig", ".tmp",
}
BANNED_COMPLETE_SUFFIXES = (".nii.gz", ".tar.gz")
ENTRY_SUFFIXES = {".py", ".ps1", ".sh", ".bat", ".cmd"}
LEGACY_ENTRY_TOKENS = ("quick", "10pct", "sampling_step", "sampling-step")


def _decode_json_stream(text: str) -> list[dict]:
    decoder = json.JSONDecoder()
    index = 0
    documents: list[dict] = []
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        document, index = decoder.raw_decode(text, index)
        if not isinstance(document, dict):
            raise AssertionError(f"Dry-run emitted a non-object JSON value: {document!r}")
        documents.append(document)
    return documents


def _flag(command: list[str], name: str) -> str:
    try:
        index = command.index(name)
    except ValueError as error:
        raise AssertionError(f"Missing {name} in command: {command}") from error
    if index + 1 >= len(command):
        raise AssertionError(f"Missing value after {name} in command: {command}")
    return command[index + 1]


class FixedManifestTests(unittest.TestCase):
    def test_exact_release_manifests(self):
        manifests = validate_splits.validate_release_manifests()
        self.assertEqual(
            {dataset: {split: len(ids) for split, ids in splits.items()}
             for dataset, splits in manifests.items()},
            {
                "btcv": {"training": 18, "validation": 2, "testing": 10},
                "acdc": {"training": 70, "validation": 10, "testing": 20},
                "isic2018": {
                    "training": 2594,
                    "validation": 100,
                    "testing": 1000,
                },
            },
        )


class ReleasePipelineDryRunTests(unittest.TestCase):
    maxDiff = None

    def test_all_twelve_pipelines(self):
        pipeline = SCRIPT_ROOT / "release_pipeline.py"
        self.assertTrue(pipeline.is_file(), pipeline)

        with tempfile.TemporaryDirectory(prefix="ensemdiff-static-") as temporary:
            temporary_root = Path(temporary)
            for dataset in DATASETS:
                for condition, audit_mode in CONDITIONS.items():
                    with self.subTest(dataset=dataset, condition=condition):
                        data_root = temporary_root / "data" / dataset
                        runs_root = temporary_root / "runs"
                        command = [
                            sys.executable,
                            str(pipeline),
                            "run",
                            "--dataset",
                            dataset,
                            "--condition",
                            condition,
                            "--data-root",
                            str(data_root),
                            "--runs-root",
                            str(runs_root),
                            "--dry-run",
                        ]
                        environment = os.environ.copy()
                        environment["PYTHONDONTWRITEBYTECODE"] = "1"
                        completed = subprocess.run(
                            command,
                            cwd=CODE_ROOT,
                            env=environment,
                            text=True,
                            encoding="utf-8",
                            capture_output=True,
                            check=False,
                        )
                        self.assertEqual(
                            completed.returncode,
                            0,
                            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
                        )
                        documents = _decode_json_stream(completed.stdout)
                        self.assertEqual(
                            [document.get("stage") for document in documents],
                            [
                                "gpu_preflight",
                                "train_with_validation_selection",
                                "sample_selected_checkpoint_on_final_test",
                                "evaluate_final_test_once",
                            ],
                        )
                        self.assertEqual(len(documents), 4)
                        preflight, train, sample, evaluate = documents
                        self.assertEqual(
                            Path(preflight["command"][1]).name, "gpu_preflight.py"
                        )
                        train_command = train["command"]
                        sample_command = sample["command"]
                        evaluate_command = evaluate["command"]

                        self.assertEqual(
                            Path(train_command[1]).name, "segmentation_train.py"
                        )
                        self.assertEqual(
                            Path(sample_command[1]).name, "segmentation_sample.py"
                        )
                        self.assertEqual(
                            Path(evaluate_command[1]).name, EVALUATORS[dataset]
                        )
                        self.assertEqual(_flag(train_command, "--dataset"), dataset)
                        self.assertEqual(_flag(sample_command, "--dataset"), dataset)
                        self.assertEqual(
                            _flag(train_command, "--audit_mode"), audit_mode
                        )
                        self.assertEqual(
                            _flag(sample_command, "--audit_mode"), audit_mode
                        )

                        manifest_root = CODE_ROOT / "manifests" / dataset
                        expected_manifests = {
                            split: (manifest_root / filename).resolve()
                            for split, filename in MANIFEST_FILES[dataset].items()
                        }
                        self.assertEqual(
                            Path(_flag(train_command, "--train_manifest")).resolve(),
                            expected_manifests["train"],
                        )
                        self.assertEqual(
                            Path(_flag(train_command, "--val_manifest")).resolve(),
                            expected_manifests["val"],
                        )
                        self.assertEqual(
                            Path(_flag(sample_command, "--manifest")).resolve(),
                            expected_manifests["test"],
                        )

                        self.assertEqual(
                            Path(_flag(train_command, "--data_dir")).resolve(),
                            (data_root / "training").resolve(),
                        )
                        self.assertEqual(
                            Path(_flag(train_command, "--val_data_dir")).resolve(),
                            (data_root / "validation").resolve(),
                        )
                        self.assertEqual(
                            Path(_flag(sample_command, "--data_dir")).resolve(),
                            (data_root / "testing").resolve(),
                        )
                        self.assertEqual(
                            Path(_flag(evaluate_command, "--data_dir")).resolve(),
                            (data_root / "testing").resolve(),
                        )

                        run_dir = (runs_root / dataset / condition).resolve()
                        best_model = run_dir / "best_model.pt"
                        sample_dir = run_dir / "final_test" / "samples"
                        self.assertEqual(
                            Path(_flag(sample_command, "--model_path")).resolve(),
                            best_model,
                        )
                        self.assertEqual(
                            Path(_flag(sample_command, "--output_dir")).resolve(),
                            sample_dir,
                        )
                        self.assertEqual(
                            Path(_flag(evaluate_command, "--sample_dir")).resolve(),
                            sample_dir,
                        )
                        ensemble = "1" if condition == "core-no-diff" else "5"
                        self.assertEqual(
                            _flag(sample_command, "--num_ensemble"), ensemble
                        )
                        self.assertEqual(
                            _flag(evaluate_command, "--num_ensemble"), ensemble
                        )
                        self.assertEqual(
                            _flag(train_command, "--val_num_ensemble"), "1"
                        )
                        self.assertEqual(
                            train["environment"]["OPENAI_LOGDIR"], str(run_dir)
                        )
                        expected_final_log = str(run_dir / "final_test" / "logs")
                        self.assertEqual(
                            sample["environment"]["OPENAI_LOGDIR"], expected_final_log
                        )
                        self.assertEqual(
                            evaluate["environment"]["OPENAI_LOGDIR"],
                            expected_final_log,
                        )
                        for document in documents:
                            self.assertEqual(
                                document["environment"]["CUDA_VISIBLE_DEVICES"],
                                "0",
                            )
                            self.assertEqual(
                                document["environment"]["PYTHONDONTWRITEBYTECODE"],
                                "1",
                            )
                        self.assertFalse(
                            runs_root.exists(),
                            "A dry-run must not create its runs root",
                        )


class FinalTestGateTests(unittest.TestCase):
    def test_requires_validation_selected_matching_best_checkpoint(self):
        with tempfile.TemporaryDirectory(prefix="ensemdiff-gate-") as temporary:
            temporary_root = Path(temporary)
            args = SimpleNamespace(
                runs_root=temporary_root,
                dataset="btcv",
                condition="full",
                seed=10,
                validation_seed=10,
            )
            run_dir = temporary_root / "btcv" / "full"
            run_dir.mkdir(parents=True)
            (run_dir / "best_model.pt").touch()
            valid = {
                "checkpoint": "best_model.pt",
                "selection_partition": "validation",
                "audit_mode": "none",
                "training_seed": 10,
                "validation_seed": 10,
                "validation_manifest": str(
                    release_pipeline._manifest("btcv", "val")
                ),
            }
            metadata_path = run_dir / "best_checkpoint.json"
            metadata_path.write_text(json.dumps(valid), encoding="utf-8")
            self.assertEqual(
                release_pipeline._load_best_metadata(args),
                valid,
            )

            invalid_values = {
                "selection_partition": "testing",
                "checkpoint": "savedmodel040000.pt",
                "audit_mode": "train_random_yt",
                "training_seed": 11,
                "validation_seed": 11,
                "validation_manifest": str(
                    CODE_ROOT / "manifests" / "btcv" / "test_cases.txt"
                ),
            }
            for field, value in invalid_values.items():
                with self.subTest(field=field):
                    invalid = dict(valid)
                    invalid[field] = value
                    metadata_path.write_text(
                        json.dumps(invalid),
                        encoding="utf-8",
                    )
                    with self.assertRaises(RuntimeError):
                        release_pipeline._load_best_metadata(args)


class ReleaseHygieneTests(unittest.TestCase):
    def test_required_public_contracts_are_present_and_root_environment_is_unique(self):
        required = (
            "RELEASE_COMPLIANCE_REPORT.md",
            "environment.yml",
            "requirements.txt",
            "docs/AUDIT_IMPLEMENTATION_MAP.md",
            "docs/CODE_STRUCTURE.md",
            "docs/DATA_SPLITS.md",
            "docs/SPLIT_USAGE_MAP.md",
            "docs/EXPERIMENT_MATRIX.md",
            "docs/GPU_ENVIRONMENT.md",
            "docs/GPU_SMOKE_TEST.md",
            "docs/REPRODUCTION.md",
            "tests/gpu_preflight.py",
            "tests/gpu_smoke_12.py",
        )
        for relative in required:
            with self.subTest(relative=relative):
                self.assertTrue((RELEASE_ROOT / relative).is_file(), relative)
        self.assertFalse((CODE_ROOT / "environment.yml").exists())
        self.assertFalse((CODE_ROOT / "requirements.txt").exists())
        self.assertFalse((CODE_ROOT / "requirements-cu128.txt").exists())

    def test_all_python_sources_parse(self):
        failures: list[str] = []
        for path in RELEASE_ROOT.rglob("*.py"):
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (OSError, SyntaxError, UnicodeError) as error:
                failures.append(f"{path.relative_to(RELEASE_ROOT)}: {error}")
        self.assertEqual(failures, [], "\n".join(failures))

    def test_preserved_upstream_readme_fingerprint(self):
        readme = CODE_ROOT / "README.md"
        digest = hashlib.sha256(readme.read_bytes()).hexdigest()
        self.assertEqual(
            digest,
            "e48150e18729695c3d58e92263b8ec373dea585c43f27736121a642f5907a3f0",
        )

    def test_no_absolute_private_paths(self):
        violations: list[str] = []
        for path in RELEASE_ROOT.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in TEXT_SUFFIXES and path.name != "LICENSE":
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for line_number, line in enumerate(text.splitlines(), start=1):
                if WINDOWS_ABSOLUTE.search(line) or PRIVATE_POSIX.search(line):
                    violations.append(
                        f"{path.relative_to(RELEASE_ROOT)}:{line_number}: {line.strip()}"
                    )
        self.assertEqual(violations, [], "\n".join(violations))

    def test_no_assistant_or_generation_workflow_language(self):
        banned = (
            "co" + "dex",
            "chat" + "gpt",
            "pro" + "mpt",
            "ai" + "-assisted",
            "generated" + " by ai",
            "written" + " by ai",
            "refactored" + " by ai",
        )
        violations: list[str] = []
        for path in RELEASE_ROOT.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in TEXT_SUFFIXES and path.name != "LICENSE":
                continue
            lowered = path.read_text(encoding="utf-8", errors="replace").lower()
            for term in banned:
                if term in lowered:
                    violations.append(f"{path.relative_to(RELEASE_ROOT)}: {term}")
        self.assertEqual(violations, [], "\n".join(violations))

    def test_no_data_weight_cache_or_staging_artifacts(self):
        violations: list[str] = []
        for path in RELEASE_ROOT.rglob("*"):
            relative = path.relative_to(RELEASE_ROOT)
            lowered_parts = {part.lower() for part in relative.parts}
            if path.is_dir() and path.name.lower() in BANNED_DIRECTORIES:
                violations.append(str(relative))
                continue
            if not path.is_file():
                continue
            lower_name = path.name.lower()
            if (
                path.suffix.lower() in BANNED_SUFFIXES
                or lower_name.endswith(BANNED_COMPLETE_SUFFIXES)
                or lowered_parts.intersection(BANNED_DIRECTORIES)
            ):
                violations.append(str(relative))
        self.assertEqual(sorted(set(violations)), [], "\n".join(sorted(set(violations))))

    def test_no_legacy_quick_or_sampling_step_entrypoints(self):
        violations: list[str] = []
        for path in RELEASE_ROOT.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in ENTRY_SUFFIXES:
                continue
            relative = path.relative_to(RELEASE_ROOT)
            normalized_name = path.name.lower().replace(" ", "_")
            if any(token in normalized_name for token in LEGACY_ENTRY_TOKENS):
                violations.append(str(relative))

        pipeline_text = (SCRIPT_ROOT / "release_pipeline.py").read_text(
            encoding="utf-8"
        ).lower()
        for token in LEGACY_ENTRY_TOKENS:
            if token in pipeline_text:
                violations.append(f"code/scripts/release_pipeline.py contains {token!r}")
        self.assertEqual(violations, [], "\n".join(violations))


if __name__ == "__main__":
    unittest.main(verbosity=2)
