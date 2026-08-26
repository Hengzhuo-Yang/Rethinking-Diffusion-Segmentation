import hashlib
import json
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_best_checkpoint(metadata_path, model_path, dataset, audit_mode):
    path = Path(metadata_path)
    if not path.is_file():
        raise FileNotFoundError(f"Best-checkpoint metadata does not exist: {path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("dataset") != dataset:
        raise ValueError("Checkpoint metadata dataset does not match final-test dataset")
    if metadata.get("audit_mode", "none") != audit_mode:
        raise ValueError("Checkpoint metadata audit mode does not match final-test condition")
    if str(metadata.get("validation_split", "")).lower() not in {"val", "validation"}:
        raise ValueError("Checkpoint was not selected from a validation split")
    if not metadata.get("validation_manifest_sha256"):
        raise ValueError("Checkpoint metadata lacks validation-manifest evidence")
    if Path(str(metadata.get("checkpoint", ""))).name != Path(model_path).name:
        raise ValueError("Checkpoint filename does not match its selection metadata")
    return metadata
