"""Fail-closed public-release tree and sensitive-path audit."""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    "README.md",
    "LICENSE",
    "UPSTREAM.md",
    "MODIFICATIONS.md",
    "THIRD_PARTY_NOTICES.md",
    "RELEASE_COMPLIANCE_REPORT.md",
    "requirements.txt",
    "environment.yml",
    "gpu_preflight.py",
    "code/README.md",
    "code/LICENSE",
    "docs/AUDIT_IMPLEMENTATION_MAP.md",
    "docs/CODE_STRUCTURE.md",
    "docs/DATA_SPLITS.md",
    "docs/EXPERIMENT_MATRIX.md",
    "docs/GPU_ENVIRONMENT.md",
    "docs/GPU_SMOKE_TEST.md",
    "docs/PAPER_IMPLEMENTATION_NOTES.md",
    "docs/REPRODUCTION.md",
    "docs/SOURCE_PROVENANCE.csv",
    "docs/SPLIT_USAGE_MAP.md",
    "tests/validate_splits.py",
    "tests/gpu_smoke_12.py",
}
BANNED_ANYWHERE_DIRS = {
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".tmp_pytest",
}
BANNED_TOP_LEVEL_DIRS = {
    "data", "datasets", "data_preprocessed", "models", "checkpoints",
    "logs", "outputs", "runs", "runs_smoke",
}
BANNED_SUFFIXES = {
    ".ckpt", ".pt", ".pth", ".safetensors", ".nii", ".h5", ".hdf5",
    ".pdf", ".pyc", ".pyo", ".log", ".zip", ".tar", ".7z", ".rar",
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm", ".npy", ".npz",
}
BANNED_TEMP_PREFIXES = (".gpu_smoke", ".tmp_smoke")
TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".yaml", ".yml", ".json", ".csv", ".sh", ".gitignore",
}
UPSTREAM_HASHES = {
    "code/README.md": "4534006aa401a6bdc6e41d0f3cdbad6781a155fa500165a7e4a3c82897dfa7f8",
    "code/LICENSE": "1a69e98812b776ccf4bf2253c439b5bc488417b0d60f40dab3190ba8dc859e75",
}


def _relative(path: Path) -> str:
    return path.relative_to(RELEASE_ROOT).as_posix()


def audit_release() -> list[str]:
    errors = []
    observed = {
        _relative(path) for path in RELEASE_ROOT.rglob("*") if path.is_file()
    }
    missing = sorted(REQUIRED - observed)
    if missing:
        errors.append(f"missing required files: {missing}")

    forbidden_local_text = [
        "D:" + "\\Research",
        "C:" + "\\Users",
        "/root" + "/",
        "ahe" + "rn",
    ]
    legacy_split_token = "test_" + "10pct"
    forbidden_public_wording = [
        "AI-" + "assisted",
        "generated" + " by " + "AI",
        "written" + " by " + "AI",
        "refactored" + " by " + "AI",
        "Co" + "dex",
        "Chat" + "GPT",
        "Clau" + "de",
        "Cop" + "ilot",
        "提" + "示词",
        "自动" + "生成声明",
    ]
    secret_patterns = [
        re.compile(r"AKIA[0-9A-Z]{16}"),
        re.compile(r"sk-[A-Za-z0-9]{20,}"),
        re.compile(r"-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----"),
    ]

    for path in RELEASE_ROOT.rglob("*"):
        relative = _relative(path)
        relative_parts = path.relative_to(RELEASE_ROOT).parts
        attributes = getattr(path.stat(), "st_file_attributes", 0)
        if path.is_symlink() or bool(attributes & 0x400):
            errors.append(f"links/reparse points are forbidden: {relative}")
        if any(part in BANNED_ANYWHERE_DIRS for part in relative_parts):
            errors.append(f"banned directory in release: {relative}")
        if relative_parts and relative_parts[0] in BANNED_TOP_LEVEL_DIRS:
            errors.append(f"banned top-level artifact directory: {relative}")
        if not path.is_file():
            continue
        lowered_name = path.name.lower()
        if lowered_name.startswith(BANNED_TEMP_PREFIXES):
            errors.append(f"temporary smoke artifact in release: {relative}")
        compound_banned = lowered_name.endswith((".nii.gz", ".tar.gz"))
        if path.suffix.lower() in BANNED_SUFFIXES or compound_banned:
            errors.append(f"banned artifact type: {relative}")
        if path.stat().st_size > 5 * 1024 * 1024:
            errors.append(f"unexpected file larger than 5 MiB: {relative}")
        if path.suffix.lower() in TEXT_SUFFIXES or path.name == ".gitignore":
            text = path.read_text(encoding="utf-8-sig", errors="replace")
            for token in forbidden_local_text:
                if token.lower() in text.lower():
                    errors.append(f"forbidden local/path token in {relative}: {token!r}")
            for token in forbidden_public_wording:
                if token.lower() in text.lower():
                    errors.append(f"forbidden public wording in {relative}: {token!r}")
            if relative_parts and relative_parts[0] == "code":
                if legacy_split_token.lower() in text.lower():
                    errors.append(
                        f"legacy split token in runnable release code: {relative}"
                    )
            for pattern in secret_patterns:
                if pattern.search(text):
                    errors.append(f"possible secret in {relative}: {pattern.pattern}")

    nested = [
        relative for relative in observed
        if "GitHub_Audit_Release" in Path(relative).parts
    ]
    if nested:
        errors.append(f"nested release directories found: {nested[:5]}")
    for relative, expected in UPSTREAM_HASHES.items():
        actual = hashlib.sha256((RELEASE_ROOT / relative).read_bytes()).hexdigest()
        if actual != expected:
            errors.append(f"unchanged upstream hash mismatch: {relative}")
    return errors


def main() -> int:
    errors = audit_release()
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("release audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
