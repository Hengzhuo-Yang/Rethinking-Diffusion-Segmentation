# Rethinking Diffusion Segmentation

**Paper:** [Rethinking Diffusion Segmentation: When Does It Rely on Its Noisy State, and Does Diffusion Matter?](https://arxiv.org/abs/2609.23967)

This repository is a code-only collection of six independently organized
diffusion-based medical image segmentation method releases. Each method keeps
its own environment, reproduction guide, upstream record, modification log,
and license notices.

The methods are intentionally isolated. Do not assume that one shared Python
environment or one repository-wide license applies to all six directories.

## Methods

| Method | Installation and overview | Reproduction commands | License and provenance |
|---|---|---|---|
| TSLDSeg | [README](Methods/TSLDSeg/README.md) | [Reproduction guide](Methods/TSLDSeg/docs/REPRODUCTION.md) | [License](Methods/TSLDSeg/LICENSE), [upstream record](Methods/TSLDSeg/UPSTREAM.md) |
| SDSeg | [README](Methods/SDSeg/README.md) | [Reproduction guide](Methods/SDSeg/docs/REPRODUCTION.md) | [License](Methods/SDSeg/LICENSE), [upstream record](Methods/SDSeg/UPSTREAM.md) |
| MedSegV1 | [README](Methods/MedSegV1/README.md) | [Reproduction guide](Methods/MedSegV1/docs/REPRODUCTION.md) | [License](Methods/MedSegV1/LICENSE), [upstream record](Methods/MedSegV1/UPSTREAM.md) |
| LEAF | [README](Methods/LEAF/README.md) | [Reproduction guide](Methods/LEAF/docs/REPRODUCTION.md) | [License](Methods/LEAF/LICENSE), [upstream record](Methods/LEAF/UPSTREAM.md) |
| EnsemDiff | [README](Methods/EnsemDiff/README.md) | [Reproduction guide](Methods/EnsemDiff/docs/REPRODUCTION.md) | [License](Methods/EnsemDiff/LICENSE), [upstream record](Methods/EnsemDiff/UPSTREAM.md) |
| cDAL | [README](Methods/cDAL/README.md) | [Reproduction guide](Methods/cDAL/docs/REPRODUCTION.md) | [License](Methods/cDAL/LICENSE), [upstream record](Methods/cDAL/UPSTREAM.md) |

For a method-specific setup, enter that method directory and follow its
`README.md`. The corresponding `docs/REPRODUCTION.md` contains the full data,
training, checkpoint-selection, inference, and evaluation flow. Validated GPU
and package details are recorded separately in each method's
`docs/GPU_ENVIRONMENT.md`.

## Repository layout

```text
Methods/<method>/
|-- README.md
|-- environment.yml
|-- requirements.txt
|-- code/
|-- docs/
|   `-- REPRODUCTION.md
|-- UPSTREAM.md
|-- MODIFICATIONS.md
|-- THIRD_PARTY_NOTICES.md
|-- RELEASE_COMPLIANCE_REPORT.md
|-- LICENSE
`-- LICENSES/
```

## Distribution scope

This repository includes source code, configurations, documentation, tests,
and small identifier-only split manifests. It does not distribute:

- medical images, annotations, or preprocessed dataset caches;
- pretrained weights, trained checkpoints, or prediction artifacts;
- experiment logs, result tables, TensorBoard/W&B output, or run directories;
- paper PDFs or third-party dataset licenses.

Datasets and model assets must be obtained from their authorized providers and
remain subject to the providers' terms. Paths in the reproduction guides are
placeholders for assets acquired separately by the user.

## Licensing

There is **no single license for the entire repository**. Read the top-level
[license index](LICENSE) and the license files inside the method you use.

Important scope notes:

- SDSeg is distributed under the included CreativeML Open RAIL-M terms.
- TSLDSeg includes an MIT-licensed upstream layer and identified SDSeg-derived
  material for which the retained CreativeML Open RAIL-M terms must also be
  considered.
- cDAL is governed primarily by the NVIDIA Source Code License-NC and is
  restricted to non-commercial research or evaluation. It is not an
  OSI-approved open-source component.
- Other methods contain separately identified MIT, Apache-2.0, BSD, or other
  third-party portions documented in their `LICENSES/` and
  `THIRD_PARTY_NOTICES.md` files.

The repository is therefore best described as a **mixed-license public source
release**, not as a repository wholly licensed under MIT or another single
open-source license. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for a
navigation summary.

## Provenance and modifications

For every method:

- `UPSTREAM.md` identifies the official project and pinned baseline;
- `MODIFICATIONS.md` records local and release-related changes;
- `THIRD_PARTY_NOTICES.md` identifies retained third-party material;
- `RELEASE_COMPLIANCE_REPORT.md` records the release audit and its limits.

Retain these files, all license texts, copyright notices, and modification
notices when redistributing any part of a method.

## Research-use notice

The code is provided for research and reproducibility. It is not a medical
device, is not validated for clinical deployment, and must not be used as a
substitute for professional medical judgment. Inclusion of an upstream project
does not imply endorsement by its authors.
