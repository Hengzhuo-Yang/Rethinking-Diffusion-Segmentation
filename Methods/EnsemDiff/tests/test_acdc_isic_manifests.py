from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = RELEASE_ROOT / "code"
SCRIPTS_DIR = CODE_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from fixed_split_manifests import (  # noqa: E402
    assert_exact_id_set,
    load_fixed_split_manifests,
    read_id_manifest,
)


class FixedSplitManifestTests(unittest.TestCase):
    def test_release_manifests_have_expected_counts_and_no_leakage(self) -> None:
        acdc = load_fixed_split_manifests(
            CODE_ROOT / "manifests" / "acdc",
            manifest_filenames={
                "training": "train_patients.txt",
                "validation": "val_patients.txt",
                "testing": "test_patients.txt",
            },
            expected_counts={"training": 70, "validation": 10, "testing": 20},
            id_pattern=r"patient\d{3}",
            dataset_name="ACDC",
        )
        self.assertEqual(
            {key: len(value) for key, value in acdc.items()},
            {"training": 70, "validation": 10, "testing": 20},
        )

        isic = load_fixed_split_manifests(
            CODE_ROOT / "manifests" / "isic2018",
            manifest_filenames={
                "training": "training.txt",
                "validation": "validation.txt",
                "testing": "testing.txt",
            },
            expected_counts={"training": 2594, "validation": 100, "testing": 1000},
            id_pattern=r"ISIC_\d{7}",
            dataset_name="ISIC2018",
        )
        self.assertEqual(
            {key: len(value) for key, value in isic.items()},
            {"training": 2594, "validation": 100, "testing": 1000},
        )

    def test_duplicate_within_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.txt"
            path.write_text("item001\nitem001\n", encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                read_id_manifest(
                    path,
                    expected_count=2,
                    id_pattern=r"item\d{3}",
                    dataset_name="example",
                    split_name="training",
                )
            self.assertIn("duplicate", str(caught.exception).lower())

    def test_cross_split_leakage_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "training.txt").write_text("item001\n", encoding="utf-8")
            (root / "validation.txt").write_text("item001\n", encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                load_fixed_split_manifests(
                    root,
                    manifest_filenames={
                        "training": "training.txt",
                        "validation": "validation.txt",
                    },
                    expected_counts={"training": 1, "validation": 1},
                    id_pattern=r"item\d{3}",
                    dataset_name="example",
                )
            self.assertIn("leakage", str(caught.exception).lower())

    def test_source_set_mismatch_is_rejected(self) -> None:
        with self.assertRaises(ValueError) as caught:
            assert_exact_id_set(
                {"item001", "item999"},
                ["item001", "item002"],
                dataset_name="example",
                split_name="testing",
                source_name="archive.zip",
            )
        message = str(caught.exception).lower()
        self.assertIn("missing 1", message)
        self.assertIn("unexpected 1", message)


if __name__ == "__main__":
    unittest.main()
