# Code Structure and Release Inventory

This repository is a source-only release for a fixed, auditable experimental
workflow. The inventory below describes the files that are published, the
runtime artifacts they operate on, and the provenance class assigned by
`UPSTREAM.md`, `MODIFICATIONS.md`, and `THIRD_PARTY_NOTICES.md`.

"Release-local" means that a file is absent from the selected JuliaWolleb
baseline. It is a provenance classification, not an independent assertion of
authorship. Publication remains subject to the rights confirmation described
in `docs/LICENSING_COMPLIANCE.md`.

The provenance column also determines the upstream counterpart and applicable
license for every row: “unchanged JuliaWolleb” and “modified JuliaWolleb” map to
the same relative path at commit `676f214` under MIT; “OpenAI-derived” maps to
the corresponding improved/guided-diffusion framework path under OpenAI MIT;
the Baselines and CLIP rows use their named MIT sources; “TransUNet derivative”
maps to the Synapse list files at commit `02ef001` under Apache-2.0; and
“release-local” has no JuliaWolleb counterpart and remains conditional on the
owner's authorship/redistribution-rights confirmation. No release-local label
is used to guess authorship or override embedded third-party terms.

The four condition names used below are **Full**, **Random-Yt**,
**Shuffle-Yt**, and **Core-No-Diff**. The fixed public matrix covers BTCV,
ACDC, and ISIC2018. BraTS support is retained as a low-level upstream path but
is not one of the twelve manifest-locked matrix entries.

## Environment contract

The release has exactly two environment contract files, both at repository
root:

- `environment.yml` fixes the Conda environment name, Python version, and pip
  installation through the root requirements file.
- `requirements.txt` is the only pip requirements contract and pins the
  CUDA 12.8 PyTorch wheels and the remaining runtime dependencies.

There is no `environment.yml` or requirements file under `code/`, and no
second dependency contract should be inferred from the preserved upstream
README. Device acceptance is a separate runtime gate implemented by
`tests/gpu_preflight.py` and `code/guided_diffusion/dist_util.py`.

## Root contracts

| Path or file family | Purpose or entry point | Provenance class | RTX 5090 relevance and device behavior | Dataset(s) | Audit condition(s) | Partition(s) | Release status |
|---|---|---|---|---|---|---|---|
| `README.md` | Public landing page: protocol guarantee, quick start, condition definitions, repository guide, and release checks | Release-local documentation | Describes the GPU-bound workflow; performs no device work | BTCV, ACDC, ISIC2018 | All four | Training, validation, final test | Public contract |
| `environment.yml` | Sole Conda contract; installs Python 3.10 and delegates pip packages to root `requirements.txt` | Release-local configuration | Establishes the host environment but does not select a device | All supported datasets | All four | All runtime partitions | Public environment contract |
| `requirements.txt` | Sole pip contract; pins PyTorch 2.7.0 CUDA 12.8 wheels and runtime libraries | Release-local configuration | Supplies the CUDA-enabled build checked for `sm_120`; no CPU execution policy is declared here | All supported datasets | All four | All runtime partitions | Public environment contract |
| `.gitignore` | Prevents caches, environments, medical data, weights, runs, logs, archives, and papers from entering the release | Release-local packaging contract | No device execution | All | All | All | Public contract; listed runtime classes are excluded |
| `.gitattributes` | Normalizes source text and line endings | Release-local packaging contract | No device execution | All | All | All | Public contract |
| `UPSTREAM.md` | Authoritative upstream source, commit, license-history, and excluded-material record | Release-local provenance record | No device execution | All | All | All | Public contract |
| `MODIFICATIONS.md` | Records substantive upstream changes, release-local additions, TransUNet list transformations, and exclusions | Release-local provenance record | No device execution | All | All | All | Public contract |
| `THIRD_PARTY_NOTICES.md` | Maps redistributed or derived material to JuliaWolleb, OpenAI, OpenAI Baselines, CLIP, TransUNet, and the disclosed DDPM ancestor | Release-local notice record | No device execution | All | All | All | Public contract |
| `RELEASE_COMPLIANCE_REPORT.md` | Root publication decision, license obligations, RTX 5090 acceptance, split corrections, exclusions, and residual risk | Release-local compliance record | Summarizes measured GPU evidence; performs no device work | BTCV, ACDC, ISIC2018 | All four | Training, validation, final test | Public contract; category A is conditional on rights confirmation |

## License files

| Path or file family | Purpose or entry point | Provenance class | RTX 5090 relevance and device behavior | Dataset(s) | Audit condition(s) | Partition(s) | Release status |
|---|---|---|---|---|---|---|---|
| `LICENSE` | Repository-level copy of the license governing the selected JuliaWolleb baseline | Unchanged JuliaWolleb MIT license text | No device execution | All | All | All | Public license |
| `LICENSES/JuliaWolleb-Diffusion-based-Segmentation-MIT.txt` | Preserved duplicate of the JuliaWolleb MIT text for explicit source mapping | Unchanged JuliaWolleb MIT license text | No device execution | All | All | All | Public license |
| `LICENSES/OpenAI-improved-guided-CLIP-MIT.txt` | MIT text applicable to the identified OpenAI diffusion-framework and CLIP-derived material | Unchanged OpenAI MIT license text | No device execution | All | All | All | Public license |
| `LICENSES/OpenAI-Baselines-MIT.txt` | MIT text applicable to `logger.py` and its Baselines utility lineage | Unchanged OpenAI Baselines MIT license text | No device execution | All | All | All | Public license |
| `LICENSES/TransUNet-Apache-2.0.txt` | Apache-2.0 text applicable to the TransUNet-derived BTCV manifests | Unchanged TransUNet Apache-2.0 license text | No device execution | BTCV | All four | Training, validation, final test identifiers | Public license |

## Preserved upstream project material

| Path or file family | Purpose or entry point | Provenance class | RTX 5090 relevance and device behavior | Dataset(s) | Audit condition(s) | Partition(s) | Release status |
|---|---|---|---|---|---|---|---|
| `code/README.md` | Official README from JuliaWolleb/Diffusion-based-Segmentation; historical upstream usage and citation | Unchanged JuliaWolleb upstream | Its historical commands are not the release device contract | Original BraTS-oriented project | Upstream baseline | Upstream training and sampling | Public and immutable; a release test guards its fingerprint |

`code/README.md` must remain unchanged. It is preserved evidence, not the
current reproduction guide. Use root `README.md` and `docs/REPRODUCTION.md` for
the release workflow.

## `code/guided_diffusion` modules

| Path or file family | Purpose or entry point | Provenance class | RTX 5090 relevance and device behavior | Dataset(s) | Audit condition(s) | Partition(s) | Release status |
|---|---|---|---|---|---|---|---|
| `code/guided_diffusion/__init__.py` | Package marker | Unchanged JuliaWolleb upstream | No device behavior | All | All | All | Public source |
| `code/guided_diffusion/acdcloader.py` | Manifest-filtered ACDC 2D image and four-class target loader | Release-local | Loads CPU tensors; callers transfer batches to the accepted GPU | ACDC | All four | Training, validation, final test | Public source; medical payloads excluded |
| `code/guided_diffusion/bratsloader.py` | Retained BraTS modality/label loader with 224-pixel crop and NumPy support | Modified JuliaWolleb upstream | Loads CPU tensors; GPU transfer occurs in train/sample/validation code | BraTS low-level path | Low-level conditions | Low-level training and sampling | Public source; BraTS payloads excluded |
| `code/guided_diffusion/btcvloader.py` | Manifest-filtered BTCV/Synapse CT slice and binary-label loader | Release-local | Loads CPU tensors; callers transfer batches to the accepted GPU | BTCV | All four | Training, validation, final test | Public source; medical payloads excluded |
| `code/guided_diffusion/dist_util.py` | Fail-closed device selection, single-process distributed setup, checkpoint loading, and parameter broadcast | Modified JuliaWolleb/OpenAI-derived upstream | Requires logical `cuda:0` to be exactly `NVIDIA GeForce RTX 5090`; CPU fallback is disabled; Windows uses gloo | All model paths | All four | Training, validation, final test | Public source; process state is runtime only |
| `code/guided_diffusion/fp16_util.py` | Mixed-precision/master-parameter conversion, scaling, optimization, and checkpoint state conversion | Unchanged JuliaWolleb/OpenAI-derived upstream | Operates on the model device; formal matrix configuration uses FP32, while the low-level flag remains available | All | All four | Training and checkpoint I/O | Public source |
| `code/guided_diffusion/gaussian_diffusion.py` | Diffusion schedules, forward/reverse processes, segmentation loss, sampling, direct segmentation terms, and Random-Yt/Shuffle-Yt/Core-No-Diff branches | Modified JuliaWolleb/OpenAI-derived upstream; DDPM ancestor disclosed | Tensor operations execute on the model device selected by `dist_util`; reverse sampling is bypassed for Core-No-Diff | BTCV, ACDC, ISIC2018; BraTS low-level | All four | Training, validation, final-test sampling | Public source; generated samples excluded |
| `code/guided_diffusion/isicloader.py` | Manifest-filtered ISIC2018 RGB image and binary-mask loader | Release-local | Loads CPU tensors; callers transfer batches to the accepted GPU | ISIC2018 | All four | Training, validation, final test | Public source; medical payloads excluded |
| `code/guided_diffusion/logger.py` | Key/value, JSON, CSV, TensorBoard, and human-readable logging utilities | Unchanged JuliaWolleb upstream; OpenAI Baselines-derived | No device selection; records values produced by GPU runs | All | All four | Training, validation, final test | Public source; emitted logs are runtime/excluded |
| `code/guided_diffusion/losses.py` | KL and discretized Gaussian likelihood primitives | Unchanged JuliaWolleb/OpenAI-derived upstream; DDPM ancestor disclosed | Runs on the input tensor device | All | Full, Random-Yt, Shuffle-Yt | Training | Public source |
| `code/guided_diffusion/nn.py` | Convolution/pooling helpers, normalization, EMA updates, timestep embeddings, and gradient checkpointing | Unchanged JuliaWolleb/OpenAI-derived upstream | Runs on the model tensor device | All | All four | Training, validation, final test | Public source |
| `code/guided_diffusion/resample.py` | Uniform and loss-aware timestep schedule samplers | Unchanged JuliaWolleb/OpenAI-derived upstream | Produces timestep tensors and weights on the requested GPU device | All | Full, Random-Yt, Shuffle-Yt; Core-No-Diff supplies zero timesteps in `TrainLoop` | Training | Public source |
| `code/guided_diffusion/respace.py` | Timestep respacing and model timestep mapping for reduced-step diffusion | Unchanged JuliaWolleb/OpenAI-derived upstream | Runs on the same device as timestep tensors and the wrapped model | All | Full, Random-Yt, Shuffle-Yt | Validation and final-test sampling; configurable training | Public source |
| `code/guided_diffusion/script_util.py` | Model/diffusion defaults, UNet and diffusion factories, classifier/super-resolution legacy factories, and argument helpers | Modified JuliaWolleb/OpenAI-derived upstream | Constructs models before callers move them to the accepted GPU; Core-No-Diff changes input/output heads and removes timestep embedding use | BTCV, ACDC, ISIC2018; BraTS low-level | All four | Training, validation, final test | Public source |
| `code/guided_diffusion/train_util.py` | `TrainLoop`, optimization, EMA, periodic saves, validation callbacks, and validation-selected `best_model.pt`/`best_checkpoint.json` | Modified JuliaWolleb/OpenAI-derived upstream | Training tensors and parameters use `dist_util.dev()`; distributed setup must already be initialized | All training loaders | All four | Training and validation-based checkpoint selection | Public source; checkpoints, optimizer state, and logs excluded |
| `code/guided_diffusion/unet.py` | UNet, residual/attention blocks, timestep conditioning, direct Core-No-Diff path, and retained classifier/super-resolution models | Modified JuliaWolleb/OpenAI-derived upstream; CLIP component and DDPM ancestor disclosed | Model parameters and activations run on the accepted GPU; canonical forward shape is checked by GPU preflight | All | All four | Training, validation, final test | Public source; model weights excluded |
| `code/guided_diffusion/validation_util.py` | Validation inference, label conversion, per-class Dice/IoU aggregation, CSV rows, and checkpoint-selection Dice return value | Release-local | Moves image/target batches through `dist_util.dev()`; uses reverse diffusion except in Core-No-Diff | BTCV, ACDC, ISIC2018; BraTS low-level | All four | Validation only | Public source; validation CSV is runtime/excluded |
| `code/guided_diffusion/visdom_util.py` | Explicit opt-in Visdom adapter with a no-op default | Release-local | Does not select a device; visualization is disabled unless explicitly enabled | All | All four | Training and sampling diagnostics | Public source; visualization sessions are runtime/excluded |

## `code/scripts` entry points and helpers

| Path or file family | Purpose or entry point | Provenance class | RTX 5090 relevance and device behavior | Dataset(s) | Audit condition(s) | Partition(s) | Release status |
|---|---|---|---|---|---|---|---|
| `code/scripts/release_pipeline.py` | Canonical public orchestrator for the 3-dataset by 4-condition matrix; builds commands, locks manifests, gates final testing on validation metadata, and refuses output mixing | Release-local | Exposes one physical device as logical `cuda:0`, runs GPU preflight before non-dry execution, and propagates the accepted device to train/sample stages | BTCV, ACDC, ISIC2018 | All four | Training, validation selection, final test | Public entry point; run directories are runtime/excluded |
| `code/scripts/segmentation_train.py` | Low-level training entry point; dataset/model construction, metadata logging, validation callback, and `TrainLoop` launch | Modified JuliaWolleb/OpenAI-derived upstream | Calls fail-closed distributed setup and runs model/loss/validation work on logical `cuda:0` | BTCV, ACDC, ISIC2018; BraTS low-level | All four | Training and validation | Public entry point; checkpoints/logs/metrics excluded |
| `code/scripts/segmentation_sample.py` | Low-level selected-checkpoint inference and per-slice ensemble artifact writer | Modified JuliaWolleb/OpenAI-derived upstream | Loads the checkpoint on CPU, then moves the exact architecture to the accepted GPU; Core-No-Diff uses one direct forward pass | BTCV, ACDC, ISIC2018; BraTS low-level | All four | Final test in the canonical workflow | Public entry point; sample tensors and metadata are runtime/excluded |
| `code/scripts/validate_checkpoint.py` | Standalone execution of the same validation path for an explicitly named checkpoint | Release-local | Loads state on CPU, then validates on the fail-closed accepted GPU | BTCV, ACDC, ISIC2018; BraTS low-level | All four | Validation only | Public diagnostic entry point; CSV output excluded |
| `code/scripts/fixed_split_manifests.py` | Shared manifest parsing, format/count/leakage checks, exact source-ID checks, and empty-output safety gate | Release-local | CPU/filesystem only | BTCV, ACDC, ISIC2018 | Condition-independent | Training, validation, final test identifiers | Public helper |
| `code/scripts/preprocess_btcv_synapse_to_npy.py` | Validates the raw BTCV case universe, assigns fixed partitions, converts CT/labels to 2D NumPy slices, and writes preprocessing metadata | Release-local | CPU preprocessing; does not satisfy or invoke the GPU gate | BTCV | Condition-independent | Creates training, validation, testing payload trees | Public entry point; raw and preprocessed payloads excluded |
| `code/scripts/preprocess_acdc_to_npy.py` | Validates ACDC patients/frames, assigns fixed partitions, normalizes/resizes images, and emits one-hot 2D targets | Release-local | CPU preprocessing; does not satisfy or invoke the GPU gate | ACDC | Condition-independent | Creates training, validation, testing payload trees | Public entry point; raw and preprocessed payloads excluded |
| `code/scripts/preprocess_isic2018_to_npy.py` | Validates image/mask archive membership against fixed IDs and creates normalized RGB/binary-mask NumPy payloads | Release-local | CPU preprocessing; does not satisfy or invoke the GPU gate | ISIC2018 | Condition-independent | Creates training, validation, testing payload trees | Public entry point; archives and preprocessed payloads excluded |
| `code/scripts/evaluate_btcv_samples.py` | Aggregates final-test BTCV ensemble masks and reports binary Dice/IoU | Release-local | CPU-only artifact loading and metric calculation | BTCV | All four | Final test only | Public entry point; generated CSV/results excluded |
| `code/scripts/evaluate_acdc_samples.py` | Reconstructs ACDC case/slice predictions and reports multiclass Dice/IoU summaries | Release-local | CPU-only artifact loading and metric calculation | ACDC | All four | Final test only | Public entry point; generated CSV/results excluded |
| `code/scripts/evaluate_isic2018_samples.py` | Aggregates ISIC2018 ensemble scores and reports lesion Dice/IoU summaries | Release-local | CPU-only artifact loading and metric calculation | ISIC2018 | All four | Final test only | Public entry point; generated CSV/results excluded |

## Fixed manifest families

The manifest files contain identifiers only. They are public protocol metadata;
the corresponding images and labels are not included.

| Path or file family | Purpose or entry point | Provenance class | RTX 5090 relevance and device behavior | Dataset(s) | Audit condition(s) | Partition(s) | Release status |
|---|---|---|---|---|---|---|---|
| `code/manifests/btcv/{train_cases,val_cases,test_cases}.txt` | Fixed 18/2/10 case partitions used by preprocessing, loaders, validation selection, and final-test gating | TransUNet derivative (Apache-2.0), transformed and repartitioned as recorded in `MODIFICATIONS.md` | No device execution; controls which payloads later reach GPU stages | BTCV | Shared by all four | Training, validation, final test | Public identifier metadata; medical payloads excluded |
| `code/manifests/acdc/{train_patients,val_patients,test_patients}.txt` | Fixed 70/10/20 patient partitions | Release-local | No device execution; controls which payloads later reach GPU stages | ACDC | Shared by all four | Training, validation, final test | Public identifier metadata; medical payloads excluded |
| `code/manifests/isic2018/{training,validation,testing}.txt` | Fixed 2594/100/1000 image-ID partitions | Release-local | No device execution; controls which payloads later reach GPU stages | ISIC2018 | Shared by all four | Training, validation, final test | Public identifier metadata; images, masks, and archives excluded |

## Tests and release checks

| Path or file family | Purpose or entry point | Provenance class | RTX 5090 relevance and device behavior | Dataset(s) | Audit condition(s) | Partition(s) | Release status |
|---|---|---|---|---|---|---|---|
| `tests/gpu_preflight.py` | Fail-closed acceptance program: exact device name/capability/compiled architecture, CUDA arithmetic/backward, canonical formal-UNet forward, and nonzero memory evidence | Release-local | Directly requires logical `cuda:0`, exact RTX 5090, capability 12.0, `sm_120`, FP32 CUDA work, and the formal release model | Uses BTCV-shaped 1+1 channel probe; architecture check is shared | Formal Full-path model probe | Pre-run gate for training, validation, and final test | Public executable check; emitted JSON is runtime evidence |
| `tests/gpu_smoke_12.py` | Real 12-combination acceptance: manifest-locked train step, validation metric, temporary validation-best save, strict reload, and final-test inference | Release-local | Uses the formal UNet/224 input, asserts all model inputs/outputs/loss/backward on `cuda:0`, retains DDPM 100/1,000 and ensemble policy, and cleans temporary artifacts | BTCV, ACDC, ISIC2018 | All four; exact branch contract recorded | Training, validation selection, final test | Public executable check; optional JSON report is runtime/excluded |
| `tests/test_gpu_acceptance_static.py` | Standard-library-only contracts for preflight/smoke import safety, exact formal config, all 12 mappings, branch AST, batch/sampling policy, and public text hygiene | Release-local | Does not probe CUDA; verifies that the executable checks require the exact RTX 5090 policy | BTCV, ACDC, ISIC2018 | All four | Training, validation, final test contracts | Public test |
| `tests/validate_splits.py` | Standalone exact manifest fingerprint/count/format/disjointness validator; optionally reconciles local preprocessed members and slice totals | Release-local | CPU/filesystem only | BTCV, ACDC, ISIC2018 | Condition-independent | Training, validation, final test | Public executable check; local payloads remain excluded |
| `tests/test_release_static.py` | Static suite for manifests, all twelve dry-run command graphs, final-test gate, syntax, upstream README fingerprint, path hygiene, and excluded artifact/legacy-entry checks | Release-local | Dry runs do not load a GPU; checks that the public pipeline wires the GPU preflight and device environment correctly | BTCV, ACDC, ISIC2018 | All four | Training, validation, final test | Public test |
| `tests/test_btcv_manifests.py` | Exact BTCV case membership, uniqueness, disjointness, and obsolete-subset-name checks | Release-local | CPU/filesystem only | BTCV | Condition-independent | Training, validation, final test | Public test |
| `tests/test_acdc_isic_manifests.py` | ACDC/ISIC counts and leakage checks plus negative tests for duplicate IDs and source-set mismatches | Release-local | CPU/filesystem only | ACDC, ISIC2018 | Condition-independent | Training, validation, final test | Public test |
| `tests/test_preprocess_output_safety.py` | Verifies that preprocessors accept only missing/empty destinations and never delete prior split or metadata output | Release-local | CPU/filesystem only | Shared preprocessing helper | Condition-independent | Preprocessing before all partitions | Public test; temporary test data is runtime only |

## Documentation

| Path or file family | Purpose or entry point | Provenance class | RTX 5090 relevance and device behavior | Dataset(s) | Audit condition(s) | Partition(s) | Release status |
|---|---|---|---|---|---|---|---|
| `docs/CODE_STRUCTURE.md` | Current source/runtime inventory and responsibility map | Release-local documentation | Documents device boundaries; performs no device work | All | All | All | Public contract |
| `docs/AUDIT_IMPLEMENTATION_MAP.md` | Final-line-number map of Full and three controlled conditions, targets, interventions, timestep/denoising state, and CUDA placement | Release-local documentation | Maps every model-compute responsibility to the fail-closed CUDA path | BTCV, ACDC, ISIC2018 | All four | Training, validation, final test | Public contract |
| `docs/DATA_SPLITS.md` | Split sources, fixed counts, identifiers, expected preprocessed layout, and validation commands | Release-local documentation | No device execution | BTCV, ACDC, ISIC2018 | Shared by all four | Training, validation, final test | Public contract |
| `docs/EXPERIMENT_MATRIX.md` | Dataset interfaces, condition semantics, twelve formal entries, and evaluation policy | Release-local documentation | Records the formal configuration executed after GPU acceptance | BTCV, ACDC, ISIC2018 | All four | Training, validation, final test | Public contract |
| `docs/GPU_ENVIRONMENT.md` | Supported RTX 5090 platform, environment creation, verified versions, numerical policy, implementation boundary, and evidence scope | Release-local documentation | Authoritative human-readable device policy; execution is delegated to preflight and `dist_util` | All | All four | All runtime partitions | Public contract |
| `docs/GPU_SMOKE_TEST.md` | Sanitized dated record of the 12 real GPU combinations, stages, device evidence, peak memory, status, and cleanup | Release-local documentation | Records 12 PASS / 0 FAIL / 0 BLOCKED on the exact RTX 5090 | BTCV, ACDC, ISIC2018 | All four | Training, validation selection, final test | Public evidence; contains no generated weight or local report |
| `docs/LICENSING_COMPLIANCE.md` | Conditional publication decision, license matrix, obligations, authorship confirmation, and residual risk | Release-local documentation | No device execution | All | All | All | Public contract |
| `docs/REPRODUCTION.md` | End-to-end setup, preprocessing, validation, dry-run inspection, training, selected-checkpoint testing, and release checks | Release-local documentation | Directs users through the root environment contract and mandatory GPU gate | BTCV, ACDC, ISIC2018 | All four | Training, validation, final test | Public contract |
| `docs/SPLIT_USAGE_MAP.md` | File-to-partition map, validation-selection flow, final-test gate, and prohibited leakage paths | Release-local documentation | No device execution | BTCV, ACDC, ISIC2018 | Shared by all four | Training, validation, final test | Public contract |

## Runtime layout and deliberate exclusions

Public source files define the workflow; they do not include the private or
generated payloads consumed and produced by that workflow. A typical local
execution creates structures like these outside the source inventory:

```text
<data-root>/
  training/       preprocessed images and labels
  validation/     preprocessed images and labels
  testing/        preprocessed images and labels

<runs-root>/<dataset>/<condition>/
  savedmodel*.pt
  emasavedmodel_*.pt
  optsavedmodel*.pt
  best_model.pt
  best_checkpoint.json
  resolved_config.json
  validation_metrics.csv
  final_test/
    final_test_request.json
    logs/
    samples/
    evaluation CSV files
```

All of the following are deliberately excluded from publication:

- medical images, labels, masks, NIfTI/HDF5/NumPy payloads, and dataset
  archives;
- pretrained weights, training checkpoints, EMA weights, and optimizer
  states;
- run directories, generated samples, logs, metrics, TensorBoard/Visdom
  state, and local GPU evidence files;
- Python caches, virtual environments, editor state, temporary staging
  directories, archives, and paper PDFs;
- the removed BraTS example payloads, legacy BTCV quick/ten-percent utilities,
  and the dedicated sampling-step experiment described in
  `MODIFICATIONS.md`.

The published manifests disclose only fixed identifiers. The licenses in this
repository do not grant dataset access, medical-data redistribution rights, or
rights to separately obtained model weights.
