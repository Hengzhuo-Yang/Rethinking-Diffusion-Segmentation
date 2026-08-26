"""Generate deterministic file-level source and responsibility inventories."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = RELEASE_ROOT / "code"
CSV_OUTPUT = RELEASE_ROOT / "docs" / "SOURCE_PROVENANCE.csv"
MARKDOWN_OUTPUT = RELEASE_ROOT / "docs" / "CODE_STRUCTURE.md"
UPSTREAM_COMMIT = "381827e1dc64a99132bbc4adf68e3c73d295c15a"

USER_MODIFIED_UPSTREAM = {
    "code/configs/latent-diffusion/isic-ldm-kl-8.yaml",
    "code/ldm/data/__init__.py",
    "code/ldm/data/isic.py",
    "code/ldm/models/autoencoder.py",
    "code/ldm/models/diffusion/classifier.py",
    "code/ldm/models/diffusion/ddpm.py",
    "code/ldm/modules/diffusionmodules/model.py",
    "code/ldm/util.py",
    "code/main.py",
    "code/scripts/slice2seg.py",
}
USER_ADDED = {
    "code/configs/latent-diffusion/acdc-cls4-ldm-kl-8.yaml",
    "code/configs/latent-diffusion/btcv-cls2-ldm-kl-8.yaml",
    "code/ldm/audit.py",
    "code/ldm/data/acdc.py",
    "code/ldm/data/synapse.py",
    "code/scripts/evaluate_acdc_multiclass_logits.py",
    "code/scripts/evaluate_btcv_binary_logits.py",
    "code/scripts/evaluate_isic2018_binary_logits.py",
    "code/scripts/preprocess_isic2018_to_tsldseg.py",
    "code/scripts/promote_best_checkpoint.py",
}
RELEASE_CREATED = {
    "gpu_preflight.py",
    "tests/build_source_inventory.py",
    "tests/gpu_smoke_12.py",
    "tests/release_audit.py",
    "tests/test_acdc_multiclass.py",
    "tests/test_audit_contracts.py",
    "tests/test_isic2018_binary.py",
    "tests/test_release_configs.py",
    "tests/test_split_manifests.py",
    "tests/test_train_random_yt_audit.py",
    "tests/validate_splits.py",
    "code/ldm/data/manifest.py",
    "code/ldm/runtime.py",
    "code/scripts/preprocess_acdc_to_tsldseg.py",
    "code/scripts/preprocess_btcv_synapse_to_tsldseg.py",
}
RELEASE_DERIVED_LOCAL = {
    "code/scripts/preprocess_acdc_to_tsldseg.py",
    "code/scripts/preprocess_btcv_synapse_to_tsldseg.py",
}
RELEASE_MODIFIED = {
    "code/configs/latent-diffusion/acdc-cls4-ldm-kl-8.yaml",
    "code/configs/latent-diffusion/btcv-cls2-ldm-kl-8.yaml",
    "code/configs/latent-diffusion/isic-ldm-kl-8.yaml",
    "code/ldm/data/__init__.py",
    "code/ldm/data/acdc.py",
    "code/ldm/data/isic.py",
    "code/ldm/data/synapse.py",
    "code/ldm/models/autoencoder.py",
    "code/ldm/models/diffusion/ddpm.py",
    "code/ldm/modules/encoders/modules.py",
    "code/main.py",
    "code/scripts/preprocess_isic2018_to_tsldseg.py",
    "code/scripts/slice2seg.py",
}
PYTORCH_COMPATIBILITY = {
    "code/ldm/models/autoencoder.py",
    "code/ldm/models/diffusion/classifier.py",
    "code/ldm/modules/diffusionmodules/model.py",
    "code/ldm/util.py",
}
CUDA_RUNTIME_FILES = {
    "gpu_preflight.py",
    "tests/gpu_smoke_12.py",
    "code/ldm/runtime.py",
    "code/main.py",
    "code/scripts/slice2seg.py",
    "code/ldm/modules/encoders/modules.py",
}

PURPOSES = {
    "code/README.md": "Unchanged official TSLDSeg usage and citation document",
    "code/LICENSE": "Unchanged official TSLDSeg MIT license text",
    "code/setup.py": "Python package metadata and install requirements",
    "code/main.py": "Training entry point, data module, logging, validation selection, and checkpoint orchestration",
    "code/ldm/audit.py": "Validated audit-mode normalization, random Y_t replacement, and no-self-match shuffled Y_t construction",
    "code/ldm/runtime.py": "Fail-fast RTX 5090 CUDA runtime gate",
    "code/ldm/util.py": "Configuration instantiation, checkpoint loading, and shared utilities",
    "code/ldm/lr_scheduler.py": "Warm-up and learning-rate schedule implementations",
    "code/ldm/data/base.py": "Legacy shared dataset base utilities retained by model lineage",
    "code/ldm/data/manifest.py": "Strict ID-only manifest parsing and image-path filtering",
    "code/ldm/data/synapse.py": "BTCV/Synapse train, validation-loss, and evaluation-label dataset adapters",
    "code/ldm/data/acdc.py": "ACDC train, validation-loss, and evaluation-label dataset adapters",
    "code/ldm/data/isic.py": "ISIC2018 train, validation-loss, and evaluation-label dataset adapters",
    "code/ldm/models/autoencoder.py": "First-stage segmentation VAE and checkpoint loading",
    "code/ldm/models/diffusion/ddpm.py": "TSLDSeg objective, all audit branches, EMA validation, DDIM/direct inference, and metrics",
    "code/ldm/models/diffusion/ddim.py": "DDIM reverse-diffusion sampler",
    "code/ldm/models/diffusion/plms.py": "Retained PLMS sampler implementation",
    "code/ldm/models/diffusion/classifier.py": "Retained diffusion-classifier support with trusted checkpoint loading",
    "code/ldm/models/diffusion/dpm_solver/dpm_solver.py": "Retained DPM-Solver numerical implementation",
    "code/ldm/models/diffusion/dpm_solver/sampler.py": "Retained DPM-Solver adapter",
    "code/ldm/modules/attention.py": "Attention and transformer blocks",
    "code/ldm/modules/diffusionmodules/model.py": "Autoencoder encoder/decoder blocks and conditioning checkpoint loading",
    "code/ldm/modules/diffusionmodules/openaimodel.py": "Diffusion UNet, timestep embedding, and residual/attention blocks",
    "code/ldm/modules/diffusionmodules/util.py": "Beta schedules, timestep extraction, checkpointing, and tensor helpers",
    "code/ldm/modules/distributions/distributions.py": "Diagonal Gaussian latent distribution",
    "code/ldm/modules/ema.py": "Exponential moving average parameter tracking",
    "code/ldm/modules/encoders/modules.py": "Image-conditioning encoder wiring for TAM, HSEM, and MCF",
    "code/ldm/modules/encoders/mcf.py": "MCF feature-fusion module",
    "code/ldm/modules/encoders/trial.py": "TAM and HSEM auxiliary modules",
    "code/ldm/modules/x_transformer.py": "Retained transformer building blocks",
    "code/ldm/modules/losses/contperceptual.py": "Retained continuous-autoencoder loss support",
    "code/ldm/modules/losses/vqperceptual.py": "Retained VQ/perceptual loss support",
    "code/ldm/modules/image_degradation/bsrgan.py": "Retained BSRGAN degradation utilities from model lineage",
    "code/ldm/modules/image_degradation/bsrgan_light.py": "Retained lightweight BSRGAN degradation utilities",
    "code/ldm/modules/image_degradation/utils_image.py": "Retained image conversion and degradation helpers",
    "code/scripts/slice2seg.py": "Explicit final-test inference entry point for the three released datasets",
    "code/scripts/promote_best_checkpoint.py": "Promote the validation-selected checkpoint to a stable filename",
    "code/scripts/preprocess_btcv_synapse_to_tsldseg.py": "Build the fixed BTCV 18/2/10 PNG split cache from NIfTI",
    "code/scripts/preprocess_acdc_to_tsldseg.py": "Build the fixed ACDC 70/10/20 PNG split cache from NIfTI",
    "code/scripts/preprocess_isic2018_to_tsldseg.py": "Build official ISIC2018 Task 1 PNG split caches from archives",
    "code/scripts/evaluate_btcv_binary_logits.py": "Compute BTCV binary Dice/IoU result tables",
    "code/scripts/evaluate_acdc_multiclass_logits.py": "Compute ACDC class-wise and aggregate Dice/IoU result tables",
    "code/scripts/evaluate_isic2018_binary_logits.py": "Compute ISIC2018 binary Dice/IoU result tables",
    "code/scripts/download_first_stages_f8.sh": "Official first-stage checkpoint download helper",
    "code/scripts/download_models_lsun_churches.sh": "Official latent-diffusion checkpoint download helper",
    "gpu_preflight.py": "Public RTX 5090 CUDA matrix-multiply/backward preflight",
    "tests/gpu_smoke_12.py": "Twelve-cell CUDA train/validation/checkpoint/test smoke harness",
    "tests/validate_splits.py": "Manifest disjointness, ownership, pairing, leakage, and real-cache count validator",
    "tests/release_audit.py": "Fail-closed public-tree artifact, path, secret, attribution, and wording audit",
    "tests/test_acdc_multiclass.py": "ACDC class mapping, label-map, and metric-path tests",
    "tests/test_audit_contracts.py": "Static and tensor-level invariants for all retained audit modes",
    "tests/test_isic2018_binary.py": "ISIC2018 binary label and split contract tests",
    "tests/test_train_random_yt_audit.py": "Focused random-Y_t target/input contract tests",
    "tests/test_release_configs.py": "Config, split role, inference entry, and runtime contract tests",
    "tests/test_split_manifests.py": "Fixed-manifest disjointness and count tests",
    "tests/build_source_inventory.py": "Deterministic provenance CSV and code-structure table generator",
    "requirements.txt": "Pinned direct runtime dependencies from the validated environment",
    "environment.yml": "Minimal Conda environment declaration for the validated runtime",
}


def inventory_paths() -> list[Path]:
    code_paths = [
        path for path in CODE_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in {".py", ".yaml", ".yml", ".sh", ".txt", ".md"}
    ]
    release_paths = [RELEASE_ROOT / "gpu_preflight.py", RELEASE_ROOT / "requirements.txt", RELEASE_ROOT / "environment.yml"]
    release_paths.extend(sorted((RELEASE_ROOT / "tests").glob("*.py")))
    return sorted(code_paths + release_paths, key=lambda path: path.relative_to(RELEASE_ROOT).as_posix())


def dataset_for(relative: str) -> str:
    lowered = relative.lower()
    if "btcv" in lowered or "synapse" in lowered:
        return "BTCV/Synapse"
    if "acdc" in lowered:
        return "ACDC"
    if "isic" in lowered:
        return "ISIC2018"
    if any(token in lowered for token in ("main.py", "ddpm.py", "audit.py", "slice2seg.py", "gpu_smoke_12.py", "validate_splits.py", "test_release_configs.py")):
        return "BTCV; ACDC; ISIC2018"
    return "shared/not dataset-specific"


def audit_for(relative: str) -> str:
    lowered = relative.lower()
    if any(token in lowered for token in ("audit.py", "ddpm.py", "main.py", "latent-diffusion/", "gpu_smoke_12.py", "test_audit", "train_random", "slice2seg.py")):
        return "Full; Random-Yt; Shuffle-Yt; Core-No-Diff"
    if relative.startswith("code/ldm/"):
        return "shared by all four conditions"
    return "not condition-specific"


def split_role_for(relative: str) -> str:
    lowered = relative.lower()
    if "/manifests/" in lowered:
        return Path(relative).stem
    if "/data/" in lowered and relative.endswith(".py"):
        return "train/validation/test data infrastructure"
    if "preprocess_" in lowered or "validate_splits.py" in lowered:
        return "split construction/validation"
    if "evaluate_" in lowered or "slice2seg.py" in lowered:
        return "final test only"
    if "promote_best_checkpoint.py" in lowered:
        return "validation-selected checkpoint handoff"
    if "latent-diffusion/" in lowered or lowered.endswith("main.py") or lowered.endswith("ddpm.py"):
        return "train; validation selection; final test"
    if lowered.endswith("gpu_smoke_12.py"):
        return "train; validation selection; checkpoint reload; test"
    if relative.startswith("code/ldm/"):
        return "shared model/runtime infrastructure"
    return "not split-specific"


def purpose_for(relative: str) -> str:
    if relative in PURPOSES:
        return PURPOSES[relative]
    if "code/configs/latent-diffusion/" in relative:
        return f"Formal {dataset_for(relative)} model, data, optimizer-schedule, validation, and checkpoint configuration"
    if "/manifests/" in relative:
        dataset = dataset_for(relative)
        return f"Fixed ID-only {dataset} {Path(relative).stem} manifest"
    if relative.endswith("/__init__.py"):
        return "Python package marker/export surface"
    return "Retained supporting implementation from the validated TSLDSeg tree"


def source_fields(relative: str) -> dict[str, str]:
    if relative in RELEASE_DERIVED_LOCAL:
        return {
            "source_category": "User-derived local preprocessing, rewritten for release",
            "upstream_present": "no",
            "user_modified_or_added": "yes; validated local preprocessing lineage",
            "upstream_counterpart": "none at upstream commit",
            "reference": "validated local preprocessing lineage",
            "release_status": "release-created",
        }
    if relative in RELEASE_CREATED or "/manifests/" in relative:
        category = "Generated manifest/config" if "/manifests/" in relative else "Release-created infrastructure"
        return {
            "source_category": category,
            "upstream_present": "no",
            "user_modified_or_added": "no; release-created",
            "upstream_counterpart": "none at upstream commit",
            "reference": "release assembly",
            "release_status": "release-created",
        }
    if relative in USER_ADDED:
        return {
            "source_category": "User-added validated local work",
            "upstream_present": "no",
            "user_modified_or_added": "yes; added before release assembly",
            "upstream_counterpart": "none at upstream commit",
            "reference": "validated local working tree",
            "release_status": "release-modified" if relative in RELEASE_MODIFIED else "preserved",
        }
    if relative in USER_MODIFIED_UPSTREAM:
        category = "Upstream modified for audit" if relative in {"code/main.py", "code/ldm/models/diffusion/ddpm.py"} else "Upstream modified by user"
        if relative in PYTORCH_COMPATIBILITY:
            category = "Upstream modified for PyTorch/CUDA-environment compatibility"
        return {
            "source_category": category,
            "upstream_present": "yes",
            "user_modified_or_added": "yes; modified before release assembly",
            "upstream_counterpart": relative.removeprefix("code/"),
            "reference": f"upstream {UPSTREAM_COMMIT} + validated local working tree",
            "release_status": "release-modified" if relative in RELEASE_MODIFIED else "preserved",
        }
    if relative == "code/ldm/modules/encoders/modules.py":
        return {
            "source_category": "Upstream modified for CUDA-only release contract",
            "upstream_present": "yes",
            "user_modified_or_added": "no; release-only modification",
            "upstream_counterpart": relative.removeprefix("code/"),
            "reference": f"upstream {UPSTREAM_COMMIT}",
            "release_status": "release-modified",
        }
    if relative in {"requirements.txt", "environment.yml"}:
        return {
            "source_category": "Generated environment specification",
            "upstream_present": "no",
            "user_modified_or_added": "no; release-created from validated environment",
            "upstream_counterpart": "none; upstream legacy environment not used",
            "reference": "validated local sdseg Conda environment",
            "release_status": "release-created",
        }
    return {
        "source_category": "Upstream unchanged",
        "upstream_present": "yes",
        "user_modified_or_added": "no",
        "upstream_counterpart": relative.removeprefix("code/"),
        "reference": UPSTREAM_COMMIT,
        "release_status": "copied unchanged",
    }


def license_for(relative: str) -> str:
    if "/manifests/" in relative:
        return "Repository MIT; ID facts only; dataset content not included"
    if relative in RELEASE_CREATED or relative in USER_ADDED or relative in {"requirements.txt", "environment.yml"}:
        return "Repository MIT; layered upstream notices remain applicable where derived"
    if relative in {"code/LICENSE", "code/README.md", "code/setup.py"}:
        return "TSLDSeg MIT"
    return "TSLDSeg MIT plus applicable lineage notices; see THIRD_PARTY_NOTICES.md"


def rtx_field(relative: str) -> str:
    if relative in PYTORCH_COMPATIBILITY:
        return "yes; PyTorch 2.11 checkpoint-loading compatibility"
    if relative in CUDA_RUNTIME_FILES:
        return "yes; strict CUDA/RTX 5090 runtime contract"
    if relative in {"requirements.txt", "environment.yml"}:
        return "yes; validated cu128 environment"
    if "latent-diffusion/" in relative:
        return "runtime settings only (cuda:0, benchmark)"
    return "no"


def make_row(path: Path) -> dict[str, str]:
    relative = path.relative_to(RELEASE_ROOT).as_posix()
    source = source_fields(relative)
    return {
        "path": relative,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "purpose": purpose_for(relative),
        **source,
        "rtx5090_compatibility": rtx_field(relative),
        "applicable_license": license_for(relative),
        "dataset": dataset_for(relative),
        "audit_condition": audit_for(relative),
        "split_role": split_role_for(relative),
        "shared_infrastructure": "yes" if dataset_for(relative) in {"shared/not dataset-specific", "BTCV; ACDC; ISIC2018"} else "no",
        "included": "yes",
    }


def markdown_escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def write_markdown(rows: list[dict[str, str]]) -> None:
    columns = [
        ("path", "Relative path"),
        ("purpose", "Function"),
        ("source_category", "Source category"),
        ("upstream_present", "Upstream?"),
        ("user_modified_or_added", "User change?"),
        ("rtx5090_compatibility", "RTX 5090/CUDA change?"),
        ("upstream_counterpart", "Upstream counterpart"),
        ("applicable_license", "Applicable license"),
        ("dataset", "Dataset"),
        ("audit_condition", "Condition"),
        ("split_role", "Train/val/test role"),
        ("shared_infrastructure", "Shared?"),
        ("included", "Included?"),
    ]
    lines = [
        "# Code structure and file responsibility inventory",
        "",
        "This is the complete public-source inventory for the release. It covers every",
        "source file, configuration, script, fixed manifest, environment declaration,",
        "and executable test. `docs/SOURCE_PROVENANCE.csv` contains the same records plus",
        "SHA-256, source reference, and release-status fields for machine-readable review.",
        "",
        "The upstream baseline is TSLDSeg commit",
        f"`{UPSTREAM_COMMIT}`. `Unknown` would be used where provenance cannot be",
        "established; no row below relies on file timestamps as provenance evidence.",
        "",
        "No data, weights, experiment outputs, paper files, or local environment export is",
        "part of this inventory. All listed files are included in the public release.",
        "",
        "| " + " | ".join(title for _, title in columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(markdown_escape(row[key]) for key, _ in columns) + " |")
    lines.extend([
        "",
        "## License reading rule",
        "",
        "The per-file license column is deliberately layered. The official TSLDSeg MIT",
        "notice is retained, and files inherited or derived through SDSeg,",
        "latent-diffusion, CLIP, taming-transformers, or DeepHypergraph remain subject to",
        "the applicable notices summarized in `THIRD_PARTY_NOTICES.md` and preserved under",
        "`LICENSES/`. The inventory does not relicense third-party work.",
        "",
    ])
    MARKDOWN_OUTPUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    rows = [make_row(path) for path in inventory_paths()]
    with CSV_OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(rows)
    print(f"wrote {len(rows)} rows to {CSV_OUTPUT.relative_to(RELEASE_ROOT)} and {MARKDOWN_OUTPUT.relative_to(RELEASE_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
