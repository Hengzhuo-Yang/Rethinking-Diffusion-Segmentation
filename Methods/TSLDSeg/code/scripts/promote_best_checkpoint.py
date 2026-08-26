import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path


def read_best_row(val_csv):
    path = Path(val_csv)
    if not path.is_file():
        return None
    best = None
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                dice = float(row.get("dice_mean", "nan"))
            except ValueError:
                continue
            if dice != dice:
                continue
            if best is None or dice > best[0]:
                best = (dice, row)
    return best[1] if best else None


def find_top_checkpoint(logdir):
    ckpt_dir = Path(logdir) / "checkpoints"
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Missing checkpoint directory: {ckpt_dir}")
    candidates = [
        path for path in ckpt_dir.glob("*.ckpt")
        if path.name != "last.ckpt" and not path.name.startswith("last")
    ]
    if not candidates:
        candidates = list(ckpt_dir.glob("*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint files found under {ckpt_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--stable_name", default="best_checkpoint.ckpt")
    parser.add_argument("--metadata_json", default=None)
    parser.add_argument("--delete_source", action="store_true")
    args = parser.parse_args()

    logdir = Path(args.logdir)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    source = find_top_checkpoint(logdir)
    target = run_dir / args.stable_name
    shutil.copy2(source, target)
    source_deleted = False
    if args.delete_source and source.resolve() != target.resolve():
        source.unlink()
        source_deleted = True

    best_row = read_best_row(args.val_csv)
    metadata = {
        "source_checkpoint": str(source),
        "stable_checkpoint": str(target),
        "source_checkpoint_deleted": source_deleted,
        "logdir": str(logdir),
        "validation_csv": str(Path(args.val_csv)),
        "best_validation_row": best_row,
        "promoted_at": datetime.now().isoformat(timespec="seconds"),
    }
    metadata_path = Path(args.metadata_json) if args.metadata_json else run_dir / "best_checkpoint_meta.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
