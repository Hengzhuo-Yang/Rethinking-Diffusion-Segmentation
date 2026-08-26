"""Run the release's strict RTX 5090/CUDA preflight."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parent / "code"
sys.path.insert(0, str(CODE_ROOT))

from ldm.runtime import require_rtx5090


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = require_rtx5090(args.device)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print("RTX 5090 CUDA preflight passed")
        for key, value in result.items():
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
