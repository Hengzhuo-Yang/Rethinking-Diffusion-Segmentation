import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "code" / "manifests" / "btcv"
EXPECTED = {
    "train_cases.txt": {
        "case0005",
        "case0006",
        "case0007",
        "case0009",
        "case0010",
        "case0021",
        "case0023",
        "case0024",
        "case0026",
        "case0027",
        "case0028",
        "case0030",
        "case0031",
        "case0033",
        "case0034",
        "case0037",
        "case0039",
        "case0040",
    },
    "val_cases.txt": {"case0001", "case0008"},
    "test_cases.txt": {
        "case0002",
        "case0003",
        "case0004",
        "case0022",
        "case0025",
        "case0029",
        "case0032",
        "case0035",
        "case0036",
        "case0038",
    },
}


def read_cases(filename):
    return [
        line.strip()
        for line in (MANIFEST_DIR / filename).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


class BTCVManifestTest(unittest.TestCase):
    def test_fixed_case_partitions(self):
        observed_sets = []
        for filename, expected in EXPECTED.items():
            cases = read_cases(filename)
            self.assertEqual(len(cases), len(set(cases)), filename)
            self.assertTrue(all(re.fullmatch(r"case\d{4}", case) for case in cases))
            self.assertEqual(set(cases), expected, filename)
            observed_sets.append(set(cases))

        self.assertEqual(sum(map(len, observed_sets)), 30)
        self.assertEqual(len(set.union(*observed_sets)), 30)

    def test_no_legacy_subset_names(self):
        all_text = "\n".join(
            (MANIFEST_DIR / filename).read_text(encoding="utf-8").lower()
            for filename in EXPECTED
        )
        self.assertNotIn("10pct", all_text)
        self.assertNotIn("quick", all_text)


if __name__ == "__main__":
    unittest.main()
