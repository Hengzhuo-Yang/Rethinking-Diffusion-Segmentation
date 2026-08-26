# Release compliance report

## Decision

**Outcome A — public code release is permitted with retained notices and
conditions.**

Basis:

1. The official TSLDSeg repository contains an MIT license from its initial
   public history and permits modification and redistribution with its notice.
2. TSLDSeg explicitly declares SDSeg and latent-diffusion lineage. Their
   applicable license texts and attribution are retained rather than treating
   TSLDSeg's MIT file as the only layer.
3. No dataset, pretrained/trained weight, PDF, experiment result, secret, or
   local environment export is distributed.
4. Release-modified upstream files are identified in `MODIFICATIONS.md`; the
   unmodified official README remains available in `code/README.md`.

This is a technical provenance review, not legal advice. Any later inclusion
of model weights requires a fresh review of the CreativeML Open RAIL-M model
terms and the relevant checkpoint source.

## Included

- TSLDSeg source needed by BTCV, ACDC, and ISIC2018
- full diffusion plus the three already implemented audit modes
- fixed ID-only manifests and validators
- explicit preprocessing/evaluation scripts
- portable configs, environment specifications, and documentation
- CPU/static tests and an RTX 5090 smoke-matrix script

## Excluded

- all medical images, labels, NIfTI/PNG/HDF5 caches, and dataset archives
- all `.ckpt`, `.pt`, `.pth`, and `.safetensors` files
- training logs, TensorBoard events, predictions, CSV result tables, and runs
- paper PDFs and local reading notes
- `__pycache__`, `.pytest_cache`, temporary smoke directories, and IDE files
- cloud/VAST/task-scheduler wrappers and machine-specific command notes
- historical BTCV `test_10pct_seed23` material
- the separate sampling-step audit

## Data-flow compliance

- Fixed manifests are mutually disjoint at case/patient/image level.
- Internal checkpoint evaluation reads the separate evaluation-label view
  `datasets["validation_metrics"]` and logs `val_avg_dice`; both validation
  views use the same fixed val manifest.
- Configured `test` datasets point to the real final test partitions.
- Final inference requires an explicit best checkpoint and test manifest.
- No formal command reports validation performance as a test result.

## Verification snapshot

- Unit/static checks: 25 passed.
- Real-cache split validation: passed for all three datasets and all expected
  sample counts.
- RTX 5090 preflight: passed.
- GPU smoke matrix: 12/12 complete train → validation → checkpoint reload →
  test conditions passed.
- Release tree path/secret/artifact scan: see final verification section in
  `docs/GPU_SMOKE_TEST.md` and the release handoff.
