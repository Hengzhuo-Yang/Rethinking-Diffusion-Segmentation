from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = RELEASE_ROOT / "code" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from fixed_split_manifests import prepare_empty_output_directory  # noqa: E402


class PreprocessOutputSafetyTests(unittest.TestCase):
    def test_missing_destination_is_created(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "new-output"
            prepared = prepare_empty_output_directory(
                output_root,
                dataset_name="example",
            )
            self.assertEqual(prepared, output_root.resolve())
            self.assertTrue(prepared.is_dir())
            self.assertEqual(list(prepared.iterdir()), [])

    def test_existing_empty_destination_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "empty-output"
            output_root.mkdir()
            prepared = prepare_empty_output_directory(
                output_root,
                dataset_name="example",
            )
            self.assertEqual(prepared, output_root.resolve())

    def test_prior_split_or_metadata_output_is_rejected_without_deletion(self) -> None:
        prior_entries = (
            "training",
            "validation",
            "testing",
            "manifest.csv",
            "summary.json",
            "training.txt",
            "validation.txt",
            "testing.txt",
        )
        for entry_name in prior_entries:
            with self.subTest(entry_name=entry_name):
                with tempfile.TemporaryDirectory() as directory:
                    output_root = Path(directory) / "prior-output"
                    output_root.mkdir()
                    entry = output_root / entry_name
                    if "." in entry_name:
                        entry.write_text("prior output\n", encoding="utf-8")
                    else:
                        entry.mkdir()

                    with self.assertRaises(FileExistsError):
                        prepare_empty_output_directory(
                            output_root,
                            dataset_name="example",
                        )
                    self.assertTrue(entry.exists())

    def test_file_destination_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "not-a-directory"
            output_path.write_text("prior output\n", encoding="utf-8")
            with self.assertRaises(NotADirectoryError):
                prepare_empty_output_directory(
                    output_path,
                    dataset_name="example",
                )
            self.assertTrue(output_path.is_file())


if __name__ == "__main__":
    unittest.main()
