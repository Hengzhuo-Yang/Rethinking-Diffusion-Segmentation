#!/usr/bin/env python3
"""Stdlib-only contracts for GPU acceptance programs; never executes CUDA."""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path


sys.dont_write_bytecode = True
TEST_ROOT = Path(__file__).resolve().parent
RELEASE_ROOT = TEST_ROOT.parent
CODE_ROOT = RELEASE_ROOT / "code"
sys.path.insert(0, str(TEST_ROOT))

import gpu_preflight  # noqa: E402
import gpu_smoke_12  # noqa: E402


EXPECTED_CONFIG = {
    "image_size": 256,
    "class_cond": False,
    "learn_sigma": True,
    "num_channels": 128,
    "num_res_blocks": 2,
    "channel_mult": "1,1,2,2,4,4",
    "num_heads": 1,
    "num_head_channels": -1,
    "num_heads_upsample": -1,
    "attention_resolutions": "16",
    "dropout": 0.0,
    "diffusion_steps": 1000,
    "noise_schedule": "linear",
    "use_kl": False,
    "predict_xstart": False,
    "rescale_timesteps": False,
    "rescale_learned_sigmas": False,
    "use_checkpoint": False,
    "use_scale_shift_norm": False,
    "resblock_updown": False,
    "use_fp16": False,
    "use_new_attention_order": False,
}


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _tree(path: Path) -> ast.Module:
    return ast.parse(_source(path), str(path))


class ImportSafetyTests(unittest.TestCase):
    def test_top_level_imports_are_stdlib_only(self):
        for filename in ("gpu_preflight.py", "gpu_smoke_12.py"):
            with self.subTest(filename=filename):
                tree = _tree(TEST_ROOT / filename)
                top_level_modules = set()
                for node in tree.body:
                    if isinstance(node, ast.Import):
                        top_level_modules.update(
                            alias.name.split(".", 1)[0] for alias in node.names
                        )
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        top_level_modules.add(node.module.split(".", 1)[0])
                self.assertTrue(
                    top_level_modules.isdisjoint(
                        {"torch", "numpy", "guided_diffusion"}
                    ),
                    top_level_modules,
                )

    def test_help_does_not_require_or_probe_gpu(self):
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        for filename in ("gpu_preflight.py", "gpu_smoke_12.py"):
            with self.subTest(filename=filename):
                completed = subprocess.run(
                    [sys.executable, "-B", str(TEST_ROOT / filename), "--help"],
                    cwd=RELEASE_ROOT,
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
                self.assertIn("usage:", completed.stdout.lower())


class PreflightContractTests(unittest.TestCase):
    def test_exact_target_and_formal_configuration(self):
        self.assertEqual(
            gpu_preflight.EXPECTED_GPU_NAME, "NVIDIA GeForce RTX 5090"
        )
        self.assertEqual(gpu_preflight.EXPECTED_CAPABILITY, (12, 0))
        self.assertEqual(gpu_preflight.EXPECTED_ARCH, "sm_120")
        self.assertEqual(gpu_preflight.CUDA_DEVICE, "cuda:0")
        self.assertEqual(gpu_preflight.FORMAL_MODEL_CONFIG, EXPECTED_CONFIG)

    def test_preflight_contains_all_fail_closed_checks(self):
        source = _source(TEST_ROOT / "gpu_preflight.py")
        required_tokens = {
            "torch.cuda.is_available",
            "torch.cuda.device_count",
            "torch.cuda.get_device_name",
            "torch.cuda.get_device_capability",
            "torch.cuda.get_arch_list",
            "cuda_matmul",
            "cuda_backward",
            "formal_model_parameters_cuda0",
            "formal_model_forward_cuda0",
            "cuda_memory_nonzero",
        }
        for token in required_tokens:
            with self.subTest(token=token):
                self.assertIn(token, source)

    def test_no_cpu_execution_fallback_branch(self):
        tree = _tree(TEST_ROOT / "gpu_preflight.py")
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            function = node.func
            if not (
                isinstance(function, ast.Attribute)
                and function.attr in {"device", "to"}
            ):
                continue
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                self.assertNotEqual(first.value.lower(), "cpu")

    def test_public_error_json_does_not_copy_exception_messages(self):
        source = _source(TEST_ROOT / "gpu_preflight.py")
        self.assertNotIn('"message": str(error)', source)
        self.assertNotIn("traceback", source.lower())


class TwelveCombinationContractTests(unittest.TestCase):
    def test_exact_three_by_four_mapping(self):
        self.assertEqual(
            gpu_smoke_12.CONDITIONS,
            {
                "full": "none",
                "random-yt": "train_random_yt",
                "shuffle-yt": "train_shuffle_yt",
                "core-no-diff": "core_no_diff",
            },
        )
        self.assertEqual(
            tuple(gpu_smoke_12.DATASETS), ("btcv", "acdc", "isic2018")
        )
        self.assertEqual(len(gpu_smoke_12.COMBINATIONS), 12)
        self.assertEqual(len(set(gpu_smoke_12.COMBINATIONS)), 12)
        self.assertEqual(
            set(gpu_smoke_12.COMBINATIONS),
            {
                (dataset, condition)
                for dataset in gpu_smoke_12.DATASETS
                for condition in gpu_smoke_12.CONDITIONS
            },
        )

    def test_shuffle_is_the_only_batch_two_condition(self):
        self.assertEqual(
            gpu_smoke_12.SMOKE_BATCH_SIZE,
            {
                "full": 1,
                "random-yt": 1,
                "shuffle-yt": 2,
                "core-no-diff": 1,
            },
        )

    def test_formal_ddpm_and_ensemble_values_are_not_shortened(self):
        self.assertEqual(gpu_smoke_12.VALIDATION_RESPACING, "100")
        self.assertEqual(gpu_smoke_12.TEST_RESPACING, "1000")
        self.assertEqual(gpu_smoke_12.VALIDATION_ENSEMBLE, 1)
        self.assertEqual(gpu_smoke_12.NON_CORE_TEST_ENSEMBLE, 5)
        self.assertEqual(gpu_smoke_12.TRAIN_LR, 0.0001)
        self.assertEqual(gpu_smoke_12.WEIGHT_DECAY, 0.0)
        self.assertEqual(gpu_smoke_12.SEED, 10)
        self.assertEqual(gpu_smoke_12.VALIDATION_SEED, 10)

    def test_fixed_partition_slice_and_image_totals(self):
        self.assertEqual(
            {
                dataset: config["partition_totals"]
                for dataset, config in gpu_smoke_12.DATASETS.items()
            },
            {
                "btcv": {
                    "training": 2211,
                    "validation": 295,
                    "testing": 1273,
                },
                "acdc": {
                    "training": 1304,
                    "validation": 182,
                    "testing": 416,
                },
                "isic2018": {
                    "training": 2594,
                    "validation": 100,
                    "testing": 1000,
                },
            },
        )

    def test_formal_loss_contains_all_three_audit_branches(self):
        self.assertEqual(
            gpu_smoke_12._audit_branch_contract(),
            {
                "core_no_diff",
                "train_random_yt",
                "train_shuffle_yt",
            },
        )
        self.assertEqual(
            set(gpu_smoke_12.EXPECTED_AUDIT_EFFECTS),
            {
                "none",
                "core_no_diff",
                "train_random_yt",
                "train_shuffle_yt",
            },
        )

    def test_blocked_path_still_emits_twelve_sanitized_records(self):
        preflight = gpu_preflight._base_result()
        records = gpu_smoke_12._blocked_records(
            preflight,
            status=gpu_smoke_12.BLOCKED,
            stage="gpu_preflight",
            code="cuda_available",
        )
        self.assertEqual(len(records), 12)
        for record in records:
            self.assertEqual(record["status"], "BLOCKED")
            self.assertIn("device", record)
            self.assertEqual(
                set(record["manifests"]),
                {"training", "validation", "testing"},
            )
            self.assertEqual(len(record["stages"]), 1)


class BestCheckpointContractTests(unittest.TestCase):
    def test_helper_schema_matches_product_trainloop_ast(self):
        self.assertEqual(
            gpu_smoke_12._trainloop_best_schema(),
            gpu_smoke_12.BEST_METADATA_KEYS,
        )

    def test_smoke_invokes_formal_loss_validation_save_reload_and_test(self):
        source = _source(TEST_ROOT / "gpu_smoke_12.py")
        for token in (
            "training_losses_segmentation",
            "loss.backward()",
            "optimizer.step()",
            "validate_segmentation",
            "_save_best_like_trainloop",
            "load_state_dict",
            "final_test_inference_and_metric",
            "TemporaryDirectory",
            "register_forward_pre_hook",
            "register_forward_hook",
            "model_parameter_device",
            "input_device",
            "output_device",
            "loss_device",
            "forward_completed",
            "backward_completed",
            "optimizer_step_completed",
            "audit_branch_executed",
            "partition_totals_expected",
            "partition_totals_observed",
        ):
            with self.subTest(token=token):
                self.assertIn(token, source)


class PublicTextSafetyTests(unittest.TestCase):
    def test_acceptance_sources_have_no_private_path_or_workflow_terms(self):
        windows_absolute = re.compile(r"(?i)(?<![A-Za-z0-9])[A-Z]:[\\/]")
        private_posix = re.compile(
            r"(?i)(?<![A-Za-z0-9:])/(?:home|root|workspace|Users)(?:/|\\)"
        )
        banned_terms = ("co" + "dex", "chat" + "gpt", "pro" + "mpt")
        for filename in (
            "gpu_preflight.py",
            "gpu_smoke_12.py",
            "test_gpu_acceptance_static.py",
        ):
            with self.subTest(filename=filename):
                source = _source(TEST_ROOT / filename)
                self.assertIsNone(windows_absolute.search(source))
                self.assertIsNone(private_posix.search(source))
                lowered = source.lower()
                for term in banned_terms:
                    self.assertNotIn(term, lowered)


if __name__ == "__main__":
    unittest.main()
