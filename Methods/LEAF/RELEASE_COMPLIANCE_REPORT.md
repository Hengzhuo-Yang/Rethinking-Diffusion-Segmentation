# Release compliance report

## Result

**Licensing classification: A — a modified source release is permitted subject to the applicable license conditions.**

The identified code licenses are permissive: LEAF is MIT, the mapped CompVis and OpenAI derivatives are MIT, and the mapped Marigold utility is Apache-2.0. Publication remains conditional on retaining all applicable copyright notices, file headers, full license texts, and this file-level source mapping. This is a release-engineering assessment, not legal advice or an unconditional legal guarantee.

## Audited base and scope

- Upstream LEAF base: `adb6ae37e641124107dc606c3642e406da8c0559`
- Upstream remote recorded locally: `https://github.com/Pearisli/LEAF.git`
- Functional source: a dirty `official_code` worktree based on that commit
- Release directory: this isolated `GitHub_Audit_Release` tree
- Official upstream README: retained without release edits at `code/README.md`

Because the functional source was dirty, the commit identifies the base, not every byte in the release. Local modifications are disclosed in `MODIFICATIONS.md`; third-party lineage is disclosed in `THIRD_PARTY_NOTICES.md`.

## License payload

| Scope | License file |
|---|---|
| LEAF base and covered LEAF modifications | `LICENSE` and `LICENSES/LEAF-MIT.txt` |
| CompVis latent-diffusion-derived autoencoder code | `LICENSES/CompVis-latent-diffusion-MIT.txt` |
| OpenAI guided-diffusion-derived U-Net code | `LICENSES/OpenAI-guided-diffusion-MIT.txt` |
| Marigold-derived seeding utility | `LICENSES/Marigold-Apache-2.0.txt` |

The Apache-2.0 file preserves the Bingxin Ke/ETH Zurich attribution, and the original header remains in `code/src/util/seeding.py`.

## Release boundary and exclusions

Only source, configuration templates, split definitions/IDs, tests, and documentation intended for reproduction are in scope. The release intentionally excludes:

- all raw or processed datasets, medical images, labels, patient/case content, and generated data caches;
- all pretrained or trained weights, checkpoints, optimizer states, extracted U-Net/VAE assets, and model caches (`*.ckpt`, `*.pt`, `*.pth`, `*.bin`, `*.safetensors`, and equivalents);
- predictions, sample images, metrics exports, TensorBoard/W&B state, run directories, console logs, temporary smoke-test products, and caches;
- paper PDFs, supplementary files, paper figures, and other publication copies whose redistribution rights were not established;
- archives, cloud/RunPod/Vast packaging copies, backups, and nested `GitHub_Audit_Release*` directories;
- credentials, tokens, machine-specific environment state, usernames, and private absolute paths;
- the historical `noise_coupled` experiment and its configs, scripts, outputs, and tests;
- the different-sampling-steps audit, including `sampling_step_audit` commands/configuration, sampling-audit scripts, results, and logs;
- the historical BTCV 10-percent-test checkpoint-selection mechanism and ACDC/ISIC quick-final-test shortcuts that conflated validation with final testing.

Exclusion from this source package is not a statement about the legal status of an external artifact. It only means the artifact is not redistributed here.

## Conditions for public distribution

Before publishing a final archive or Git commit:

1. keep `LICENSE`, every file in `LICENSES/`, `UPSTREAM.md`, and `THIRD_PARTY_NOTICES.md`;
2. keep third-party file headers and provenance comments;
3. do not present local audit or compatibility changes as work of the upstream LEAF authors;
4. do not add data, weights, papers, logs, caches, secrets, or private machine paths;
5. rerun the repository's release checks against the exact staged file set;
6. separately review the terms for any dataset, model weight, or package obtained after cloning.

Within that boundary and subject to those conditions, the source tree is suitable for a public GitHub release under conclusion A. No conclusion is made here about patent, privacy, medical-data, trademark, export-control, or third-party artifact rights beyond the identified source licenses.
