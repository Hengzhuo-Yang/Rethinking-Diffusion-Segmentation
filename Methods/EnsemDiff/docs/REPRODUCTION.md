# Reproduction Guide

## 1. Confirm publication and dataset rights

Read LICENSE, THIRD_PARTY_NOTICES.md, and docs/LICENSING_COMPLIANCE.md. This
repository does not distribute medical data or grant dataset access. Obtain
BTCV/Synapse, ACDC, and ISIC2018 through their authorized channels and comply
with their terms, privacy requirements, and intended-use restrictions.

Release-local source publication remains conditional on the project owner
confirming authorship or redistribution rights as described in the compliance
report.

## 2. Recreate the verified RTX 5090 environment

The supported target is Windows, Conda, and NVIDIA GeForce RTX 5090. The
repository-root contracts were distilled from the successful local
`ensemdiff` environment; no nested or legacy upstream environment is canonical.

    conda env create -f environment.yml
    conda activate ensemdiff

`requirements.txt` installs PyTorch 2.7.0, torchvision 0.22.0, and torchaudio
2.7.0 from the official CUDA 12.8 wheel index. Do not substitute a CPU build or
the upstream PyTorch 1.9-era environment. Run the acceptance check immediately:

    python tests/gpu_preflight.py

The command returns nonzero unless logical `cuda:0` is an exact NVIDIA GeForce
RTX 5090 and CUDA arithmetic, backward, the formal model forward, placement,
and memory checks all pass. Every real pipeline action repeats preflight. Use
`--cuda-device N` to expose physical GPU N as logical `cuda:0`; no training,
validation, or inference entry point silently falls back to CPU. See
`docs/GPU_ENVIRONMENT.md` for the complete verified version and runtime record.

## 3. Preprocess into fixed partitions

Choose a separate output root for each dataset. The preprocessors read their
default fixed manifests relative to the repository, validate the full source
ID set, and create training, validation, and testing directories.

BTCV / Synapse:

    python code/scripts/preprocess_btcv_synapse_to_npy.py --raw_root RAW_BTCV --output_root prepared/btcv

ACDC:

    python code/scripts/preprocess_acdc_to_npy.py --raw_root RAW_ACDC --output_root prepared/acdc

ISIC2018 Task 1:

    python code/scripts/preprocess_isic2018_to_npy.py --raw_root RAW_ISIC2018 --output_root prepared/isic2018

BTCV requires an empty destination and rejects legacy quick/10-percent split
directories. ACDC and ISIC2018 validate the complete fixed source split before
an optional smoke limit is applied. Smoke-limited outputs are development
checks, not benchmark datasets.

### Official pretrained weights

The preserved upstream README at the selected commit provides train and sample
commands but no official pretrained checkpoint download applicable to this
three-dataset, four-condition release. The release therefore trains from
initialization, invents no download URL, and redistributes no local, upstream,
or third-party weight file. If an applicable checkpoint is later published,
obtain it only from the official upstream repository, keep it outside this
source tree, and supply it through the explicit checkpoint argument.

## 4. Validate manifests and local data

Manifest-only validation:

    python tests/validate_splits.py

Validate all three local preprocessed trees:

    python tests/validate_splits.py --data-root btcv=prepared/btcv --data-root acdc=prepared/acdc --data-root isic2018=prepared/isic2018

This checks exact manifest fingerprints, format, counts, uniqueness,
cross-partition disjointness, expected slice/image totals, directory ownership,
and required array filenames.

## 5. Inspect a resolved experiment

Dry-run emits the exact preflight, train, sample, and evaluator commands as JSON
and does not check data, import the model, create directories, or probe a GPU:

    python code/scripts/release_pipeline.py run --dataset btcv --condition full --data-root prepared/btcv --runs-root runs --cuda-device 0 --dry-run

Valid datasets are btcv, acdc, and isic2018. Valid conditions are full,
random-yt, shuffle-yt, and core-no-diff. Their Cartesian product is the complete
12-entry matrix.

## 6. Train and select on validation

Use action train:

    python code/scripts/release_pipeline.py train --dataset btcv --condition full --data-root prepared/btcv --runs-root runs --cuda-device 0

The output is `runs/btcv/full`. Periodic checkpoints and validation
metrics are written there. Only fixed validation Dice can update:

- best_model.pt
- best_checkpoint.json

Before final testing, inspect best_checkpoint.json and resolved_config.json.
The former must say selection_partition=validation and identify the fixed
validation manifest, seed, condition, metric, and step.

To resume an interrupted run, pass --resume-checkpoint explicitly. A new run
otherwise refuses to mix files into a non-empty output directory.

## 7. Run final test once

After accepting the validation-selected checkpoint:

    python code/scripts/release_pipeline.py test --dataset btcv --condition full --data-root prepared/btcv --runs-root runs --cuda-device 0

The gate verifies best checkpoint metadata before loading best_model.pt. It
uses only testing/ and the fixed test manifest. Sampling artifacts, metrics, and
final_test_request.json are written below the run's final_test directory. A
non-empty final_test directory is never overwritten; use a new run root for an
independent repetition.

Action run performs steps 6 and 7 sequentially. Separate train and test actions
are preferable when a human review boundary is required.

## 8. Run release checks

The publication checks require no dataset or GPU:

    python -m unittest discover -s tests -p "test_*.py" -v

They validate manifests, all 12 dry-run command graphs, fixed condition
mappings, explicit best-model loading, ensemble policy, absence of private
absolute paths, and absence of data, weights, caches, staging files, and legacy
quick/sampling-step entry points.

## 9. Run the mandatory 12-combination GPU acceptance

Use complete prepared roots, not smoke-limited preprocessing output:

    python tests/gpu_smoke_12.py --data-root btcv=prepared/btcv --data-root acdc=prepared/acdc --data-root isic2018=prepared/isic2018

The program first runs GPU preflight, then executes every dataset-condition
pair with the formal UNet and 224×224 tensors. Each pair reads the correct
training manifest, performs one CUDA optimizer step, runs one validation sample
with the formal 100-step DDPM policy (or the direct Core path), saves a temporary
validation-selected checkpoint, reloads it strictly on CUDA, and runs one final
test sample with the formal 1,000-step/five-member diffusion policy (or one
direct Core prediction). Shuffle-Yt alone uses smoke batch 2; the other
conditions use 1. These acceptance batches and one-step lengths do not alter
the formal batch 8 or formal training budgets.

The program emits sanitized JSON records and returns nonzero if any combination
is `FAIL` or `BLOCKED`. Its checkpoint, CSV, and logger artifacts live only in
managed temporary directories and are removed before exit. See
`docs/GPU_SMOKE_TEST.md` for the dated acceptance record.

## Reproducibility limits

The wrapper makes seeds and arguments explicit, but the verified flags do not
request bitwise-deterministic CUDA kernels. This release is validated only on
the documented Windows/Conda/RTX 5090 stack; CPU-only execution, other GPU
models, other operating systems, and the original paper's older environment
are outside scope. The source release does not ship private checkpoints or
claim that historic runs selected under obsolete protocols are results of this
corrected workflow.
