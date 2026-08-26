# Code structure

## Release root

| Path | Purpose |
|---|---|
| `README.md` | Entry point, fixed protocol, and safe usage |
| `environment.yml` | New-environment Conda recipe |
| `requirements.txt` | Focused proven runtime snapshot |
| `LICENSE` | CreativeML Open RAIL-M license retained from upstream |
| `LICENSES/` | Relevant third-party license texts |
| `UPSTREAM.md` | Upstream repository and revision provenance |
| `MODIFICATIONS.md` | Release modification record |
| `THIRD_PARTY_NOTICES.md` | Third-party attribution and notices |
| `RELEASE_COMPLIANCE_REPORT.md` | Release-compliance audit |
| `docs/` | Protocol, split, environment, and reproduction documentation |
| `code/tests/` | Static release checks, split validation, GPU preflight, and smoke tooling |

## Training and model code

| Path | Responsibility |
|---|---|
| `code/main.py` | Training CLI, data-module construction, validation Dice callback, best-checkpoint metadata, CUDA fail-closed checks |
| `code/ldm/models/diffusion/SDSeg.py` | Baseline SDSeg model and the four existing audit branches |
| `code/ldm/models/diffusion/ddim.py` | Normal DDIM implementation retained as a model dependency |
| `code/ldm/models/diffusion/plms.py` | Normal PLMS implementation retained as a model dependency |
| `code/ldm/models/autoencoder.py` | First-stage latent autoencoder |
| `code/ldm/modules/` | U-Net, attention, distributions, encoders, and utilities |
| `code/ldm/util.py` | Configuration-based object instantiation and utilities |

The presence of DDIM/PLMS support does not define the formal release protocol.
Formal segmentation evaluation is fixed to direct one-step evaluation; no
sampling-step audit or sweep is included.

## Dataset code

| Path | Dataset and split responsibility |
|---|---|
| `code/ldm/data/btcv.py` | Public BTCV aliases |
| `code/ldm/data/synapse.py` | BTCV slice loading and fixed case filters |
| `code/ldm/data/acdc.py` | ACDC train/validation/test and full-label evaluation loaders |
| `code/ldm/data/isic2018.py` | Official ISIC2018 train/validation/testing loaders |
| `code/manifests/btcv/` | Fixed BTCV case identifiers |
| `code/manifests/acdc/` | Fixed ACDC patient identifiers |
| `code/manifests/isic2018/` | Official ISIC2018 image identifiers |

Data roots are supplied at runtime. `SDSEG_DATA_ROOT` is set by the public CLI
from `--data-root`; no private absolute path is required.

## Experiment configurations

`code/configs/experiments/<dataset>/<condition>.yaml` is the public formal
matrix. It contains 12 configurations:

- datasets: `btcv`, `acdc`, `isic2018`;
- conditions: `full`, `random-yt`, `shuffle-yt`, `core-no-diff`.

The older files in `code/configs/SDSeg/` are retained base/reference
configurations. Formal commands should name a file under `configs/experiments/`
so the audit mode and split mapping are explicit.

Pretrained references use `SDSEG_PRETRAINED_ROOT`, set by `--pretrained-root`.
Pretrained files themselves are excluded.

## Scripts

| Path | Purpose |
|---|---|
| `code/scripts/preprocess_btcv_synapse_to_sdseg.py` | BTCV NIfTI to fixed 18/2/10 case-level PNG cache |
| `code/scripts/preprocess_acdc_to_sdseg.py` | ACDC labeled frames to fixed 70/10/20 patient-level PNG cache |
| `code/scripts/preprocess_isic2018_to_sdseg.py` | Official ISIC2018 archives to official train/validation/testing cache |
| `code/scripts/release_pipeline.py` | Release orchestration and fail-closed protocol checks |
| `code/scripts/slice2seg.py` | Validation/final-test evaluation with best-checkpoint proof |

All preprocessing manifests store paths relative to their output root. Smoke
limit flags are not formal-run settings.

## Files deliberately absent

The repository must not contain data, images, masks, HDF5/NumPy dataset caches,
pretrained weights, trained checkpoints, logs, predictions, metrics from private
runs, environment archives, or paper PDFs/figures. Such artifacts are local
inputs or outputs and must remain outside source control.
