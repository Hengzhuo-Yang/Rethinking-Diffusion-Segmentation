#!/usr/bin/env python3
"""Static release gates for the SDSeg public source package."""

from __future__ import annotations

import os
import re
import unittest
from pathlib import Path

from omegaconf import OmegaConf


CODE_ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = CODE_ROOT.parent
CONFIG_ROOT = CODE_ROOT / "configs" / "experiments"
DATASETS = ("btcv", "acdc", "isic2018")
MODES = {
    "full": "full_diffusion",
    "random-yt": "train_random_yt",
    "shuffle-yt": "train_shuffle_yt",
    "core-no-diff": "core_no_diff",
}


class ReleaseStaticTests(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("SDSEG_PRETRAINED_ROOT", "PRETRAINED_ROOT")

    def _config(self, dataset: str, condition: str):
        path = CONFIG_ROOT / dataset / f"{condition}.yaml"
        self.assertTrue(path.is_file(), path)
        return OmegaConf.load(path)

    def test_exactly_twelve_registered_experiments(self):
        actual = sorted(path.relative_to(CONFIG_ROOT).as_posix() for path in CONFIG_ROOT.rglob("*.yaml"))
        expected = sorted(f"{dataset}/{condition}.yaml" for dataset in DATASETS for condition in MODES)
        self.assertEqual(actual, expected)

    def test_audit_mode_matrix_uses_only_existing_branches(self):
        model_source = (CODE_ROOT / "ldm/models/diffusion/SDSeg.py").read_text(encoding="utf-8")
        for dataset in DATASETS:
            for condition, audit_mode in MODES.items():
                config = self._config(dataset, condition)
                self.assertEqual(str(config.model.params.audit_mode), audit_mode)
                self.assertIn(f'"{audit_mode}"', model_source)
        self.assertIn("_SUPPORTED_AUDIT_MODES", model_source)

    def test_formal_hyperparameters_are_fixed(self):
        expected_classes = {"btcv": 2, "acdc": 4, "isic2018": 2}
        for dataset in DATASETS:
            for condition in MODES:
                config = self._config(dataset, condition)
                self.assertEqual(int(config.model.params.random_seed), 23)
                self.assertEqual(int(config.model.params.num_classes), expected_classes[dataset])
                self.assertEqual(float(config.model.base_learning_rate), 1e-5)
                self.assertEqual(int(config.data.params.batch_size), 4)
                self.assertEqual(int(config.data.params.num_workers), 8)
                self.assertEqual(int(config.lightning.trainer.max_steps), 100000)
                self.assertEqual(str(config.model.params.monitor), "val_avg_dice")
                self.assertNotIn("sampling_step_audit", config)

    def test_validation_and_test_datasets_are_distinct(self):
        expected = {
            "btcv": (
                "ldm.data.btcv.BTCVValidationEval", "validation",
                "ldm.data.btcv.BTCVValidationEval", "test",
            ),
            "acdc": (
                "ldm.data.acdc.ACDCFullLabelSliceEval", "validation",
                "ldm.data.acdc.ACDCFullLabelSliceEval", "test",
            ),
            "isic2018": (
                "ldm.data.isic2018.ISIC2018ValidationEval", None,
                "ldm.data.isic2018.ISIC2018Test", None,
            ),
        }
        for dataset in DATASETS:
            for condition in MODES:
                config = self._config(dataset, condition)
                validation = config.data.params.metric_validation
                test = config.data.params.test
                val_target, val_split, test_target, test_split = expected[dataset]
                self.assertEqual(str(validation.target), val_target)
                self.assertEqual(str(test.target), test_target)
                if val_split is not None:
                    self.assertEqual(str(validation.params.split), val_split)
                if test_split is not None:
                    self.assertEqual(str(test.params.split), test_split)

    def test_checkpoint_selection_is_validation_only_and_bound(self):
        main_source = (CODE_ROOT / "main.py").read_text(encoding="utf-8")
        model_source = (CODE_ROOT / "ldm/models/diffusion/SDSeg.py").read_text(encoding="utf-8")
        pipeline = (CODE_ROOT / "scripts/release_pipeline.py").read_text(encoding="utf-8")
        required_main = (
            '"selection_partition": "validation"',
            '"metric_dataset_key": "metric_validation"',
            'monitor != "val_avg_dice"',
            '"mode"] = "max"',
            'summary["audit_mode"]',
            'summary["metric_validation_dataset_class"]',
            "Automatic post-training test is disabled",
        )
        for token in required_main:
            self.assertIn(token, main_source)
        self.assertIn('datasets["metric_validation"]', model_source)
        self.assertIn("validate_selection_binding", pipeline)
        self.assertIn("*-project.yaml", pipeline)

    def test_final_test_is_guarded_and_one_time(self):
        inference = (CODE_ROOT / "scripts/slice2seg.py").read_text(encoding="utf-8")
        pipeline = (CODE_ROOT / "scripts/release_pipeline.py").read_text(encoding="utf-8")
        for token in (
            "validate_selection_metadata",
            "selection_partition=validation",
            "metric_dataset_key=metric_validation",
            'choices=["cuda"]',
            "--times must equal 1",
            'partition = "test"',
        ):
            self.assertIn(token, inference)
        for token in (
            '"--no-test", "true"',
            '"--scale_lr", "false"',
            '"--gpus", "0,"',
            '"--sampler", "direct"',
            '"--ddim_steps", "1"',
            '"--times", "1"',
            "Final-test output path already exists",
            'parser.add_argument("--pretrained-root", required=True',
        ):
            self.assertIn(token, pipeline)

    def test_cuda_entrypoints_have_no_cpu_fallback(self):
        main_source = (CODE_ROOT / "main.py").read_text(encoding="utf-8")
        inference = (CODE_ROOT / "scripts/slice2seg.py").read_text(encoding="utf-8")
        pipeline = (CODE_ROOT / "scripts/release_pipeline.py").read_text(encoding="utf-8")
        preflight = (CODE_ROOT / "tests/gpu_preflight.py").read_text(encoding="utf-8")
        self.assertIn("CPU fallback is disabled", main_source)
        self.assertIn("CPU fallback is disabled", inference)
        self.assertIn("NVIDIA GeForce RTX 5090", pipeline)
        self.assertIn("NVIDIA GeForce RTX 5090", preflight)
        forbidden = re.compile(r"cuda[^\n]{0,80}else[^\n]{0,30}cpu", re.IGNORECASE)
        self.assertIsNone(forbidden.search(main_source))
        self.assertIsNone(forbidden.search(inference))

    def test_sampling_step_comparison_audit_is_absent(self):
        needle = "sampling" + "_step_audit"
        for path in CODE_ROOT.rglob("*"):
            if not path.is_file() or path == Path(__file__).resolve():
                continue
            if path.suffix.lower() not in {".py", ".yaml", ".yml", ".md", ".txt"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            self.assertNotIn(needle, text, path)

    def test_no_private_absolute_paths_in_release_text(self):
        windows_drive = re.compile("[A-Za-z]" + re.escape(":\\") + r"(?:Users|Research)\\")
        workspace_root = "/" + "workspace" + "/"
        home_root = "/" + "home" + "/"
        for path in CODE_ROOT.rglob("*"):
            if not path.is_file() or path == Path(__file__).resolve():
                continue
            if path.suffix.lower() not in {".py", ".yaml", ".yml", ".md", ".txt", ".csv"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            self.assertIsNone(windows_drive.search(text), path)
            self.assertNotIn(workspace_root, text, path)
            self.assertNotIn(home_root, text, path)

    def test_no_binary_data_weights_results_or_caches(self):
        forbidden_suffixes = {
            ".ckpt", ".pt", ".pth", ".safetensors", ".h5", ".hdf5", ".npy", ".npz",
            ".nii", ".gz", ".png", ".jpg", ".jpeg", ".pdf", ".zip", ".tar", ".pyc",
        }
        forbidden_top_level = {"data", "checkpoints", "logs", "results", "outputs", "runs"}
        offenders = []
        for path in CODE_ROOT.rglob("*"):
            relative = path.relative_to(CODE_ROOT)
            lower_parts = {part.lower() for part in relative.parts}
            if "__pycache__" in lower_parts or ".pytest_cache" in lower_parts:
                offenders.append(relative.as_posix())
            elif relative.parts and relative.parts[0].lower() in forbidden_top_level:
                offenders.append(relative.as_posix())
            elif path.is_file() and path.suffix.lower() in forbidden_suffixes:
                offenders.append(relative.as_posix())
        self.assertEqual(sorted(set(offenders)), [])

    def test_license_and_provenance_documents_exist(self):
        required = (
            "LICENSE",
            "UPSTREAM.md",
            "MODIFICATIONS.md",
            "THIRD_PARTY_NOTICES.md",
            "RELEASE_COMPLIANCE_REPORT.md",
        )
        for name in required:
            self.assertTrue((RELEASE_ROOT / name).is_file(), name)
        license_dir = RELEASE_ROOT / "LICENSES"
        self.assertTrue(license_dir.is_dir())
        self.assertGreaterEqual(len(list(license_dir.iterdir())), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
