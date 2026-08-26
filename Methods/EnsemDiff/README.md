# EnsemDiff: RTX 5090 Audit Release

This is a source-only, publication-oriented release of EnsemDiff, derived from
[JuliaWolleb/Diffusion-based-Segmentation](https://github.com/JuliaWolleb/Diffusion-based-Segmentation)
at commit `676f214035e90edd0357f51feab45841e4aefcfb` and associated with
[Diffusion Models for Implicit Image Segmentation Ensembles](https://arxiv.org/abs/2112.03145).
It preserves the Full reproduction baseline and adds three controlled,
falsification-oriented mechanism analyses behind one manifest-locked pipeline:

| Dataset | Full | Random-Yt | Shuffle-Yt | Core-No-Diff |
| --- | --- | --- | --- | --- |
| BTCV / Synapse | yes | yes | yes | yes |
| ACDC | yes | yes | yes | yes |
| ISIC2018 Task 1 | yes | yes | yes | yes |

The only validated execution target is Windows, Conda, and an NVIDIA GeForce
RTX 5090. Formal training, validation, and inference fail closed if CUDA is
unavailable or logical `cuda:0` is not an RTX 5090; there is no CPU fallback.
The release contains no medical data, masks, checkpoints, private metrics,
logs, archives, or paper PDFs, and it makes no corrected final-score claim.

## Protocol guarantee

The canonical workflow is:

    train on fixed training IDs
      -> score checkpoints on fixed validation IDs
      -> write best_model.pt and best_checkpoint.json
      -> explicitly load best_model.pt
      -> evaluate once on fixed final-test IDs

The final test cannot select a checkpoint. The public wrapper verifies the
selection partition, condition, seeds, validation manifest, and selected
filename before testing, and refuses to overwrite a non-empty final-test
directory.

See [Data splits](docs/DATA_SPLITS.md),
[split usage](docs/SPLIT_USAGE_MAP.md), and the
[12-run matrix](docs/EXPERIMENT_MATRIX.md).

## RTX 5090 quick start

Recreate the cleaned contract from the locally successful `ensemdiff`
environment, then activate it:

    conda env create -f environment.yml
    conda activate ensemdiff

The root `requirements.txt` uses the official PyTorch CUDA 12.8 wheel index and
pins the versions verified on this machine. Do not replace it with the legacy
upstream environment or a CPU-only PyTorch build. Full evidence and runtime
flags are in [GPU environment](docs/GPU_ENVIRONMENT.md).

For an already activated clean environment, the exact CUDA wheel selection is:

    python -m pip install --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0

The recommended `conda env create` command above also installs these wheels and
all remaining pinned dependencies through the root requirements file.

Before preprocessing, training, validation, or inference, run the mandatory
acceptance check:

    python tests/gpu_preflight.py

It verifies the exact GPU name and capability, `sm_120` support, CUDA tensor
arithmetic, backward, a formal UNet forward, CUDA placement, and nonzero device
memory. Every non-dry-run pipeline action invokes the same check automatically.

Validate all published ID manifests without downloading any dataset:

    python tests/validate_splits.py

Inspect a fully resolved command graph without writing files or probing CUDA:

    python code/scripts/release_pipeline.py run --dataset btcv --condition full --data-root prepared/btcv --runs-root runs --cuda-device 0 --dry-run

For preprocessing, all dataset names, resume behavior, output artifacts, and
the final-test gate, follow the [reproduction guide](docs/REPRODUCTION.md).

## Data and fixed partitions

Obtain BTCV/Synapse, ACDC, and ISIC2018 only through their authorized channels;
their access and redistribution terms are separate from this source license.
Run the corresponding fixed-split preprocessor:

    python code/scripts/preprocess_btcv_synapse_to_npy.py --raw_root RAW_BTCV --output_root prepared/btcv
    python code/scripts/preprocess_acdc_to_npy.py --raw_root RAW_ACDC --output_root prepared/acdc
    python code/scripts/preprocess_isic2018_to_npy.py --raw_root RAW_ISIC2018 --output_root prepared/isic2018

The immutable roles are:

| Dataset | Training | Validation selection | Final test |
| --- | ---: | ---: | ---: |
| BTCV | 18 cases / 2,211 slices | `case0008`, `case0001` / 295 slices | 10 cases / 1,273 slices |
| ACDC | 70 subjects / 1,304 slices | 10 subjects / 182 slices | 20 subjects / 416 slices |
| ISIC2018 | 2,594 images | 100 images | 1,000 images |

Validate manifests and prepared trees before a run; see
[data splits](docs/DATA_SPLITS.md) for the optional `--data-root` checks.

    python tests/validate_splits.py

The selected upstream commit does not advertise an official pretrained
checkpoint for this release's three-dataset, four-condition matrix. Therefore
no weight URL is invented and no local or third-party weights are redistributed;
these runs train from initialization. If upstream later publishes applicable
weights, obtain them only from the official repository and keep them outside
this source tree. The present `--resume-checkpoint` option is for a matching
run's `savedmodelNNNNNN.pt`, not a generic pretrained-weight import.

## Twelve formal GPU runs

Each command below uses the verified environment, selects physical GPU 0 as
logical `cuda:0`, runs preflight, trains on `training/`, selects
`best_model.pt` using only `validation/`, reloads it, and evaluates `testing/`
once. Use separate `train` and `test` actions when a human review boundary is
needed before final testing.

    python code/scripts/release_pipeline.py run --dataset btcv --condition full --data-root prepared/btcv --runs-root runs --cuda-device 0
    python code/scripts/release_pipeline.py run --dataset btcv --condition random-yt --data-root prepared/btcv --runs-root runs --cuda-device 0
    python code/scripts/release_pipeline.py run --dataset btcv --condition shuffle-yt --data-root prepared/btcv --runs-root runs --cuda-device 0
    python code/scripts/release_pipeline.py run --dataset btcv --condition core-no-diff --data-root prepared/btcv --runs-root runs --cuda-device 0
    python code/scripts/release_pipeline.py run --dataset acdc --condition full --data-root prepared/acdc --runs-root runs --cuda-device 0
    python code/scripts/release_pipeline.py run --dataset acdc --condition random-yt --data-root prepared/acdc --runs-root runs --cuda-device 0
    python code/scripts/release_pipeline.py run --dataset acdc --condition shuffle-yt --data-root prepared/acdc --runs-root runs --cuda-device 0
    python code/scripts/release_pipeline.py run --dataset acdc --condition core-no-diff --data-root prepared/acdc --runs-root runs --cuda-device 0
    python code/scripts/release_pipeline.py run --dataset isic2018 --condition full --data-root prepared/isic2018 --runs-root runs --cuda-device 0
    python code/scripts/release_pipeline.py run --dataset isic2018 --condition random-yt --data-root prepared/isic2018 --runs-root runs --cuda-device 0
    python code/scripts/release_pipeline.py run --dataset isic2018 --condition shuffle-yt --data-root prepared/isic2018 --runs-root runs --cuda-device 0
    python code/scripts/release_pipeline.py run --dataset isic2018 --condition core-no-diff --data-root prepared/isic2018 --runs-root runs --cuda-device 0

Outputs live below `runs/DATASET/CONDITION/`. Training writes restart
checkpoints, `validation_metrics.csv`, `best_model.pt`, and
`best_checkpoint.json`; the guarded final pass writes only below `final_test/`
and refuses to overwrite a non-empty final-test directory.

## Conditions

- Full is the standard diffusion segmentation reproduction baseline; it is not
  a fourth audit condition.
- Random-Yt replaces only the training-time Yt model input with independent
  standard Gaussian noise while retaining the original diffusion target and
  loss.
- Shuffle-Yt constructs training-time Yt from a deranged batch mask while
  retaining the current timestep, original noise target, and original loss.
- Core-No-Diff is an image-only direct segmentation counterfactual. It removes
  Yt, timestep conditioning, q_sample, diffusion loss, and reverse sampling
  from the main path, so it is not objective-preserving.

The exact implementation locations are recorded in
[the audit implementation map](docs/AUDIT_IMPLEMENTATION_MAP.md).

## Repository guide

- code/scripts/release_pipeline.py: canonical train/test orchestration.
- code/manifests/: fixed, ID-only split definitions.
- code/guided_diffusion/: model, diffusion, training, validation, and loaders.
- code/scripts/preprocess_*: fixed-split preprocessing.
- code/scripts/evaluate_*: dataset-specific final evaluation.
- tests/: GPU preflight, 12-way GPU acceptance, split, matrix, hygiene, and
  static publication checks.
- docs/: protocol, structure, provenance, compliance, and reproduction.

See [code structure](docs/CODE_STRUCTURE.md) for the full map.

The official upstream README is preserved unchanged at
[code/README.md](code/README.md). This release is based on
[JuliaWolleb/Diffusion-based-Segmentation](https://github.com/JuliaWolleb/Diffusion-based-Segmentation)
commit 676f214035e90edd0357f51feab45841e4aefcfb and the associated
[paper](https://arxiv.org/abs/2112.03145).

## Licensing and provenance

The primary upstream code is MIT-licensed, with additional preserved MIT
notices for OpenAI-derived components and an Apache-2.0 notice for
TransUNet-derived BTCV list artifacts. The exact source versions and file
classes are documented in [UPSTREAM.md](UPSTREAM.md),
[MODIFICATIONS.md](MODIFICATIONS.md), and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Publication is classified as conditionally permitted: before publishing, the
project owner must confirm authorship or redistribution rights for release-local
additions. The public decision is summarized in
[RELEASE_COMPLIANCE_REPORT.md](RELEASE_COMPLIANCE_REPORT.md); full license
analysis and the residual Ho DDPM provenance risk are in
[LICENSING_COMPLIANCE.md](docs/LICENSING_COMPLIANCE.md). Dataset access,
privacy, and redistribution rights are separate and are not granted here.

## Release checks

Run:

    python -m unittest discover -s tests -p "test_*.py" -v

Then, with all three complete prepared datasets:

    python tests/gpu_smoke_12.py --data-root btcv=prepared/btcv --data-root acdc=prepared/acdc --data-root isic2018=prepared/isic2018

The tests cover exact manifest fingerprints and disjointness, all 12 dry-run
pipelines, explicit best-checkpoint loading, ensemble policy, private absolute
paths, excluded legacy entry points, and forbidden data/weight/cache artifacts.
The dated real-GPU result is [12 PASS, 0 FAIL, 0 BLOCKED](docs/GPU_SMOKE_TEST.md);
it is an execution acceptance record, not a performance claim.
