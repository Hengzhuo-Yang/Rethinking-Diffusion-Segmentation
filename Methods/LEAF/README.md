# LEAF Fixed-Split Mechanism-Audit Release

This repository packages the source needed to reproduce the LEAF reference
condition and three controlled mechanism audits on BTCV/Synapse, ACDC, and
ISIC2018 Task 1. It is based on LEAF commit
`adb6ae37e641124107dc606c3642e406da8c0559` and the locally exercised changes
described in [UPSTREAM.md](UPSTREAM.md) and [MODIFICATIONS.md](MODIFICATIONS.md).

The upstream paper is
[LEAF: Latent Diffusion with Efficient Encoder Distillation for Aligned Features in Medical Image Segmentation](https://arxiv.org/abs/2507.18214).
The official upstream README is preserved at [code/README.md](code/README.md);
release commands and corrected split semantics are documented here.

## Public experiment scope

| Condition | Configuration value | Controlled interpretation |
| --- | --- | --- |
| Full | `audit_mode: none` | Reference LEAF diffusion path |
| Random-Yt | `audit_mode: train_random_yt` | Replaces only the training-time noisy-mask-latent input with independent Gaussian noise |
| Shuffle-Yt | `audit_mode: train_shuffle_yt` | Uses another batch member's clean mask latent, re-noised with the current sample's timestep and noise, only for the training-time input |
| Core-No-Diff | `audit_mode: core_no_diff` | Uses the image-only latent segmentation path without the noisy-mask input or reverse process |

Full is the reproduced reference condition; the other three are
falsification-oriented mechanism audits. All conditions use identical
dataset-specific train, validation, and test IDs.

This source package contains no datasets, medical images, weights,
checkpoints, predictions, logs, papers, or caches.

## Evaluation contract

The only supported formal flow is:

    fixed train IDs -> optimization
                    -> fixed validation IDs -> mean validation Dice
                    -> recorded best checkpoint
                    -> fixed test IDs -> one final evaluation

Validation does not update model parameters. Test metrics cannot select an
epoch, checkpoint, seed, hyperparameter, or condition. The release pipeline
first validates the physical cache against the fixed manifests, then requires
`best_checkpoint.json` proving validation-based selection on the configured
validation split. Final evaluation requires the corresponding EMA weights and
refuses to overwrite an existing final-test directory.

| Dataset | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| BTCV/Synapse | 18 cases / 2,211 slices | 2 cases / 295 slices | 10 cases / 1,273 slices |
| ACDC | 70 subjects / 1,304 slices | 10 subjects / 182 slices | 20 subjects / 416 slices |
| ISIC2018 Task 1 | 2,594 images | 100 images | 1,000 images |

See [docs/DATA_SPLITS.md](docs/DATA_SPLITS.md) and
[docs/SPLIT_USAGE_MAP.md](docs/SPLIT_USAGE_MAP.md).

## Supported runtime

Formal model work requires Windows, the `leaf-rtx5090` Conda environment,
CUDA-enabled PyTorch 2.7.0 from the CUDA 12.8 wheel index, and an NVIDIA
GeForce RTX 5090 visible as `cuda:0`. Formal entry points fail if CUDA is
unavailable or the selected GPU is wrong. Preprocessing and manifest checking
are CPU-side utilities.

```powershell
conda env create -f environment.yml
conda activate leaf-rtx5090
python tests\validate_splits.py
python tests\gpu_preflight.py --device cuda:0
```

Runtime evidence is not inferred from package versions. Strict preflight passed
on Python 3.11.11, PyTorch 2.7.0+cu128, CUDA 12.8, cuDNN 90701, driver 591.86,
and an NVIDIA GeForce RTX 5090. The physical-cache GPU smoke run completed all
twelve dataset-condition cells: 12 PASS, 0 FAIL, 0 BLOCKED in 63.427 seconds.

## Quick start

Preprocess a dataset into a new local output root:

```powershell
python code\scripts\preprocess_btcv_synapse_to_leaf.py --raw_root <BTCV_RAW_ROOT> --output_root <BTCV_CACHE_ROOT>
```

The ACDC and ISIC2018 preprocessors follow the same pattern. They load
repository-relative fixed manifests by default and verify expected IDs and
counts.

Prepare the upstream-compatible VAE and U-Net under a local asset root, then
inspect the formal lifecycle without executing it:

```powershell
python code\scripts\release_pipeline.py run --dataset btcv --condition full --data-root <BTCV_CACHE_ROOT> --assets-root <LEAF_ASSETS_ROOT> --runs-root <RUNS_ROOT> --dry-run
```

Remove `--dry-run` only after split validation and GPU preflight pass. Valid
datasets are `btcv`, `acdc`, and `isic2018`. Valid conditions are `full`,
`random-yt`, `shuffle-yt`, and `core-no-diff`.

## Documentation

- [docs/REPRODUCTION.md](docs/REPRODUCTION.md): end-to-end commands and artifacts
- [docs/EXPERIMENT_MATRIX.md](docs/EXPERIMENT_MATRIX.md): all twelve formal combinations
- [docs/AUDIT_IMPLEMENTATION_MAP.md](docs/AUDIT_IMPLEMENTATION_MAP.md): condition definitions and code locations
- [docs/CODE_STRUCTURE.md](docs/CODE_STRUCTURE.md): file roles and provenance classes
- [docs/GPU_ENVIRONMENT.md](docs/GPU_ENVIRONMENT.md): runtime policy and validated evidence
- [RELEASE_COMPLIANCE_REPORT.md](RELEASE_COMPLIANCE_REPORT.md): release boundary and licensing conclusion

## Licensing

The LEAF base is MIT licensed. Identified third-party-derived files retain
their own MIT or Apache-2.0 lineage and notices. Keep `LICENSE`, `LICENSES/`,
`THIRD_PARTY_NOTICES.md`, and file headers when redistributing this source.
Dataset and model-asset terms are separate and are not granted here.
