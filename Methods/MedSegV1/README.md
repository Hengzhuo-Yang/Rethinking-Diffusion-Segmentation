# MedSegDiff V1 audit release

This repository is a reproducibility-oriented release of the locally used
MedSegDiff V1 implementation for BTCV/Synapse, ACDC and ISIC2018 Task 1. It
contains the Full diffusion baseline and three controlled audit conditions,
fixed train/validation/test manifests, full-validation checkpoint selection,
and Windows/Conda/RTX 5090 verification tooling.

The original MedSegDiff project is associated with *MedSegDiff: Medical Image
Segmentation with Diffusion Probabilistic Model* (MIDL 2023, arXiv:2211.00611).
The unchanged upstream README is retained at `code/README.md`; use this root
README for the release commands and supported environment.

No medical data, masks, model weights, predictions, logs or paper assets are
distributed. The code does not download those materials automatically.

## Conditions

| Public condition | Internal mode | Training input and objective | Inference |
|---|---|---|---|
| Full | `none` | Image, case-matched noisy mask `Y_t`, timestep; epsilon MSE plus learned-variance VB | Reverse diffusion |
| Random-Yt | `train_random_yt` | Only model-input `Y_t` is replaced by independent Gaussian noise; target and loss are unchanged | Same as Full |
| Shuffle-Yt | `train_shuffle_yt` | Only model-input `Y_t` comes from another batch member's mask, using the current sample's timestep and noise; batch size > 1 | Same as Full |
| Core-No-Diff | `core_no_diff` | Image-only direct segmentation; binary BCE-with-logits + soft Dice or multiclass CE + foreground soft Dice | One image-only forward |

The local Full implementation differs from a clean upstream checkout: the
upstream calibration MSE term is disabled globally. The V1 highway condition
encoder remains active, while its legacy localization decoder is bypassed during
normal V1 training and sampling; formal V1 metrics use the diffusion sample.
The decoder is invoked only by the explicit Core-No-Diff counterfactual. See
`MODIFICATIONS.md` and `docs/AUDIT_IMPLEMENTATION_MAP.md`.

## Fixed data roles

| Dataset | Train | Validation (checkpoint selection only) | Test (one final run only) |
|---|---:|---:|---:|
| BTCV/Synapse | 18 cases / 2,211 slices | 2 cases / 295 slices | 10 cases / 1,273 slices |
| ACDC | 70 subjects / 1,304 slices | 10 subjects / 182 slices | 20 subjects / 416 slices |
| ISIC2018 Task 1 | 2,594 images | 100 images | 1,000 images |

Every run uses the ID-only files under `manifests/`. Test data are not used for
checkpoint selection, early stopping or hyperparameter decisions. Details are
in `docs/DATA_SPLITS.md` and `docs/SPLIT_USAGE_MAP.md`.

## Supported environment

This release is validated only on Windows with Conda and an NVIDIA GeForce RTX
5090 at `cuda:0`. CPU fallback and other GPU targets are intentionally rejected
by the public entry points.

```powershell
conda env create -f environment.yml
conda activate medsegdiff-audit-rtx5090
python -m pip install -r requirements-torch-cu128.txt
python tests\gpu_preflight.py
```

The verified stack is Python 3.10.20, PyTorch 2.7.0+cu128, torchvision 0.22.0,
torchaudio 2.7.0, CUDA runtime 12.8 and cuDNN 9.7.1. See
`docs/GPU_ENVIRONMENT.md` for the complete record.

## Prepare data

Acquire each dataset under its own terms. The following commands write a new
processed root; choose paths outside this repository if desired.

```powershell
python code\scripts\preprocess_btcv_synapse_to_medsegv1.py `
  --raw_root <btcv-training-root> `
  --output_root <processed-btcv-root>

python code\scripts\prepare_acdc.py `
  --raw_root <acdc-database-root> `
  --split_root manifests\acdc `
  --out_dir <processed-acdc-root>

python code\scripts\preprocess_isic2018_to_medsegv1.py `
  --raw_root <isic2018-archive-root> `
  --output_root <processed-isic2018-root>
```

The packaged ACDC `*_subjects.txt` files are preferred. The preprocessor also
accepts a custom legacy split root containing `train_patients.txt`,
`val_patients.txt`, and `test_patients.txt`.

Validate the prepared trees before training:

```powershell
python tests\validate_splits.py `
  --btcv-root <processed-btcv-root> `
  --acdc-root <processed-acdc-root> `
  --isic2018-root <processed-isic2018-root>
```

## Run the twelve formal experiments

The runner resolves immutable method parameters from
`code/configs/experiments.json`. Each `--stage all` invocation trains one
dataset/condition pair, evaluates every scheduled checkpoint on the complete
fixed validation set, retains the best mean-validation-Dice checkpoint, and
then performs one final test run from that checkpoint.

```powershell
$dataRoots = @{
  btcv = '<processed-btcv-root>'
  acdc = '<processed-acdc-root>'
  isic2018 = '<processed-isic2018-root>'
}
$conditions = @('full', 'train_random_yt', 'train_shuffle_yt', 'core_no_diff')

foreach ($dataset in @('btcv', 'acdc', 'isic2018')) {
  foreach ($condition in $conditions) {
    python code\scripts\run_experiment.py `
      --dataset $dataset `
      --condition $condition `
      --stage all `
      --data-root $dataRoots[$dataset] `
      --output-root <output-root>
  }
}
```

This loop expands to exactly 12 independent runs. Use `--stage train` or
`--stage test` to resume the two phases separately. The test phase requires
`best_checkpoint_meta.json` written by validation and does not accept a test
metric as a checkpoint-selection source.

Formal diffusion inference is 1,000-step DDPM with ensemble size 5. Validation
uses 20-step DPM-Solver++ with ensemble size 1 for BTCV, ACDC, and ISIC2018.
Training uses batch size 8, AdamW, learning
rate `5e-5`, seed 23 and 100,000 steps. Core-No-Diff has no reverse-sampling
steps. Full details are in `docs/EXPERIMENT_MATRIX.md` and
`docs/REPRODUCTION.md`.

## Verification

```powershell
python tests\test_train_random_yt_audit.py
python tests\test_isic2018_binary_loader_and_metrics.py
python tests\gpu_preflight.py
```

The release preparation ran `tests/gpu_smoke_matrix.py` across all twelve
dataset/condition paths with real prepared samples on 2026-07-19. After the
2026-07-20 condition-only/sample-only routing change, targeted compilation,
gradient, formal-forward and one-step sampling checks passed, but the complete
data-dependent twelve-path matrix has not yet been rerun. See
`docs/GPU_SMOKE_TEST.md` for the precise verification scope.

## Provenance, license and citation

The aggregate source is released under the retained MIT license. Review
`UPSTREAM.md`, `MODIFICATIONS.md`, `THIRD_PARTY_NOTICES.md`, `LICENSES/` and
`RELEASE_COMPLIANCE_REPORT.md` before redistribution. Dataset licenses and
access terms are not granted by this repository.

When using this code, cite the original MedSegDiff paper and any datasets used.
Upstream project: <https://github.com/ImprintLab/MedSegDiff>. Paper record:
<https://2023.midl.io/papers/p185>.
