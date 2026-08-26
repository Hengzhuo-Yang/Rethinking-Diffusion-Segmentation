# Code Structure

This release keeps the executable experiment code, fixed split evidence, tests, and provenance documents together while leaving datasets, pretrained parameters, and run artifacts outside version control.

## Repository layout

```text
.
|-- code/
|   |-- config/                 # Twelve fixed dataset/condition configurations
|   |-- leaf/                   # VAE, U-Net, diffusion pipeline, and Core-No-Diff model
|   |-- manifests/              # ID-only train/validation/test manifests
|   |   |-- btcv/
|   |   |-- acdc/
|   |   `-- isic2018/
|   |-- scripts/
|   |   |-- release_pipeline.py # Strict train -> validation selection -> final test entry point
|   |   |-- preprocess_*.py     # Dataset-specific preprocessing
|   |   `-- evaluate_*.py       # Dataset-specific final evaluation
|   |-- src/
|   |   |-- data/               # BTCV, ACDC, and ISIC2018 dataset loaders
|   |   `-- util/               # Losses, metrics, seeding, and runtime checks
|   |-- train.py                # Shared training and validation implementation
|   `-- extract_weights.py      # Upstream checkpoint conversion helper
|-- docs/                       # Release, split, environment, and reproduction documentation
|-- LICENSES/                   # Full texts for retained upstream components
|-- tests/                      # Static, metric, split, GPU preflight, and GPU smoke checks
|-- environment.yml             # Reconstructed Conda environment
|-- requirements.txt            # Equivalent pinned pip dependencies
|-- LICENSE                     # Project license
|-- MODIFICATIONS.md            # Release modifications
|-- THIRD_PARTY_NOTICES.md      # Third-party attribution
`-- UPSTREAM.md                 # Upstream identity and revision
```

## Execution ownership

`code/scripts/release_pipeline.py` is the public entry point for formal experiments. It maps exactly three datasets and four conditions to the twelve YAML files in `code/config/`. Before either training or testing, it validates the selected physical cache against the fixed split contract and runs the strict GPU preflight. The `run` action executes both stages in order.

`code/train.py` owns optimization and validation. It loads the selected YAML, applies command-line path overrides, constructs the matching model condition, trains on the fixed training partition, evaluates only the validation partition for checkpoint selection, and writes `best_checkpoint.json`. That metadata records the dataset, condition, selected step, validation Dice, and relative checkpoint location.

The three `code/scripts/evaluate_*_leaf.py` programs own final evaluation. The release entry point calls the evaluator for the selected dataset only after checking `best_checkpoint.json`, including the configured validation split. Requested EMA weights are mandatory and never silently replaced by non-EMA weights. Final-test data never participates in checkpoint selection.

The preprocessing programs consume the ID-only manifests under `code/manifests/`. They create the physical image/mask cache expected by the dataset loaders. `tests/validate_splits.py` independently checks manifest fingerprints, partition disjointness, expected subject or image counts, and, when cache roots are supplied, generated sample counts and ownership.

## Condition-specific model code

The Full, Random-Yt, and Shuffle-Yt conditions share the diffusion model, target, and objective in `code/train.py`. Their only experimental difference is the training-time `Y_t` input construction selected by `audit_mode`.

Core-No-Diff uses `code/leaf/core_no_diff.py`. It retains the image encoder, segmentation path, clean mask-latent target, L1 reconstruction term, and aligned-feature term, while bypassing the diffusion-corruption input and timestep conditioning. See [AUDIT_IMPLEMENTATION_MAP.md](AUDIT_IMPLEMENTATION_MAP.md) for the exact implementation map.

## Data and artifact boundaries

The repository does not contain raw datasets, preprocessed images, pretrained model parameters, DINOv2 parameters, checkpoints, samples, or result logs. A local experiment layout may use:

```text
data_preprocessed/
|-- btcv_synapse_leaf_binary_png/BTCV/
|-- acdc_mt_unet_cascade_leaf_png/ACDC/
`-- isic2018_task1_leaf_png/ISIC18/

assets/
|-- vae/
`-- unet/

runs/
```

These directories are runtime inputs or outputs and are excluded by `.gitignore`. Paths are supplied at execution time; the public code and configuration do not depend on a workstation-specific absolute path.

## Related documents

- [DATA_SPLITS.md](DATA_SPLITS.md): fixed partition definitions and counts
- [SPLIT_USAGE_MAP.md](SPLIT_USAGE_MAP.md): logical-to-physical split routing
- [EXPERIMENT_MATRIX.md](EXPERIMENT_MATRIX.md): all twelve formal commands
- [REPRODUCTION.md](REPRODUCTION.md): environment and end-to-end procedure
- [GPU_ENVIRONMENT.md](GPU_ENVIRONMENT.md): verified hardware/software policy
- [GPU_SMOKE_TEST.md](GPU_SMOKE_TEST.md): one-step GPU coverage
