# SDSeg reproducibility release

This repository packages the source needed to reproduce the fixed SDSeg audit
matrix across BTCV, ACDC, and ISIC2018. It preserves the existing audited model
branches and separates validation-based checkpoint selection from the one-time
final test evaluation.

The release contains source, experiment configurations, fixed split manifests,
preprocessing utilities, validation checks, and documentation. It intentionally
contains **no datasets, medical images, masks, pretrained weights, trained
checkpoints, run logs, predictions, or reported experiment results**.

## Fixed experiment protocol

All 12 formal experiments use:

- seed `23`;
- batch size `4`;
- `8` data-loader workers;
- `100000` maximum training steps;
- base learning rate `1e-5` with learning-rate scaling disabled;
- CUDA only, with the formal hardware check requiring an exact
  `NVIDIA GeForce RTX 5090`;
- direct, fixed one-step segmentation evaluation;
- checkpoint selection by maximum validation Dice only;
- one-time final evaluation of the selected best checkpoint on the test split.

The datasets and fixed partitions are:

| Dataset | Train | Validation | Test |
|---|---:|---:|---:|
| BTCV | 18 cases / 2211 slices | 2 cases / 295 slices | 10 cases / 1273 slices |
| ACDC | 70 patients / 1304 slices | 10 patients / 182 slices | 20 patients / 416 slices |
| ISIC2018 | 2594 images | 100 images | 1000 images |

Each dataset has four existing conditions: `Full`, `Random-Yt`, `Shuffle-Yt`,
and `Core-No-Diff`. See [docs/AUDIT_IMPLEMENTATION_MAP.md](docs/AUDIT_IMPLEMENTATION_MAP.md)
for their exact code semantics. This release does not introduce new audit
algorithms.

## Repository map

```text
.
|-- code/
|   |-- configs/experiments/       # 3 datasets x 4 conditions
|   |-- ldm/                       # model and dataset implementation
|   |-- manifests/                 # fixed public split identifiers
|   |-- scripts/                   # preprocessing, training/eval orchestration
|   `-- main.py                    # training entry point
|-- docs/                          # protocol and reproducibility documentation
|-- code/tests/                    # static, split, and GPU preflight checks
|-- environment.yml
|-- requirements.txt
|-- LICENSE
`-- LICENSES/
```

More detail is in [docs/CODE_STRUCTURE.md](docs/CODE_STRUCTURE.md).

## Environment

The proven local environment is named `sdseg` and must be treated as a
read-only source of truth. Do not update it in place. Its key versions are
Python `3.10.20`, PyTorch `2.11.0+cu128`, torchvision `0.26.0+cu128`, and
PyTorch Lightning `1.9.5`.

For a new, separate environment:

```bash
conda env create -f environment.yml
conda activate sdseg-release
```

`requirements.txt` is a focused runtime snapshot, not a byte-for-byte export of
every unrelated package in the proven environment. GPU and backend details are
recorded in [docs/GPU_ENVIRONMENT.md](docs/GPU_ENVIRONMENT.md).

## Required local assets

Obtain each dataset from its official provider and accept its terms. Preprocess
into a parent data directory with this layout:

```text
<DATA_ROOT>/
|-- btcv/{train,validation,test}/{images,masks}/
|-- acdc/{train,validation,test}/{images,masks}/
`-- isic2018/{training,validation,testing}/{images,masks}/
```

Pretrained assets are not distributed. If you are separately authorized to use
them, place them under:

```text
<PRETRAINED_ROOT>/
|-- ldm/lsun_churches256/model.ckpt
`-- first_stage_models/kl-f8/model.ckpt
```

Distribution of checkpoints is outside this source release. Users are
responsible for the terms that govern any separately acquired datasets and
weights.

## Preprocess and verify data

Run from `code/` and use an empty output directory unless replacement is
intentional:

```bash
python scripts/preprocess_btcv_synapse_to_sdseg.py --raw-root <BTCV_RAW> --output-root <DATA_ROOT>/btcv
python scripts/preprocess_acdc_to_sdseg.py --raw-root <ACDC_RAW> --output-root <DATA_ROOT>/acdc
python scripts/preprocess_isic2018_to_sdseg.py --raw-root <ISIC2018_ARCHIVES> --output-root <DATA_ROOT>/isic2018
python tests/validate_splits.py --data-root <DATA_ROOT>
```

The preprocessing scripts refuse to replace a non-empty destination unless
`--overwrite` is supplied. The limit flags are smoke-only and must remain zero
for formal preprocessing.

## Run an experiment

The release pipeline is the preferred entry point. Inspect its final interface
before launching a long run:

```bash
python scripts/release_pipeline.py --help
```

Run the complete guarded workflow for one registered experiment:

```bash
python scripts/release_pipeline.py run \
  --dataset btcv --condition full \
  --data-root <DATA_ROOT> \
  --pretrained-root <PRETRAINED_ROOT> \
  --runs-root <RUNS_ROOT>
```

The pipeline validates the formal configuration and splits, runs the CUDA
preflight, trains without test access, verifies the validation-selected best
checkpoint, and then performs the one-time final test.

The complete action-by-action workflow is in [docs/REPRODUCTION.md](docs/REPRODUCTION.md).

The underlying training command for one condition is:

```bash
python main.py \
  --base configs/experiments/btcv/full.yaml \
  --train true --no-test true --gpus 0, \
  --seed 23 --scale_lr false \
  --data-root <DATA_ROOT> \
  --pretrained-root <PRETRAINED_ROOT> \
  --logdir <RUN_ROOT> --name btcv-full
```

Training writes `best_checkpoint.json`. That record must say
`selection_partition=validation`, `metric_dataset_key=metric_validation`,
`monitor=val_avg_dice`, and `mode=max`. Final test evaluation refuses a
checkpoint that does not match that record:

```bash
python scripts/slice2seg.py \
  --dataset btcv-b \
  --config configs/experiments/btcv/full.yaml \
  --ckpt <BEST_CHECKPOINT> \
  --selection-metadata <RUN_DIR>/best_checkpoint.json \
  --data-root <DATA_ROOT> \
  --outdir <EMPTY_FINAL_TEST_OUTPUT> \
  --seed 23 --sampler direct --ddim_steps 1 --ddim_eta 0 \
  --times 1 --device cuda
```

Replace the dataset and configuration names according to
[docs/EXPERIMENT_MATRIX.md](docs/EXPERIMENT_MATRIX.md). Do not use test metrics
to choose a checkpoint, condition, threshold, seed, epoch, or hyperparameter.

## Verification status

Static validation, split validation, GPU preflight, and the 12-condition smoke
matrix are distinct checks; passing one does not imply that the others passed.
The current release code completed the authoritative full smoke matrix on
2026-07-20 using an NVIDIA GeForce RTX 5090: 12 of 12 combinations passed and
0 failed. The generated report records `is_smoke_test=true` and
`cpu_fallback=false`.

The required evidence, actual report schema, and matrix result are documented in
[docs/GPU_SMOKE_TEST.md](docs/GPU_SMOKE_TEST.md). The original JSON report is a
locally generated validation artifact and is not included in the source-code
release package. For any future code change, all 12 rows again require explicit
passing evidence; a missing or stale report is not a pass.

## License and attribution

The SDSeg source is distributed under the included CreativeML Open RAIL-M
license, including its conditions for Complementary Material. The full license,
upstream attribution, third-party notices, and modification record must remain
with redistributed copies. This source release does not redistribute model
weights or checkpoints. See `LICENSE`, `LICENSES/`, `UPSTREAM.md`,
`MODIFICATIONS.md`, `THIRD_PARTY_NOTICES.md`, and
`RELEASE_COMPLIANCE_REPORT.md`.
