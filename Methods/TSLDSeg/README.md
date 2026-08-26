# TSLDSeg diffusion-input audit release

This is a code-only, reproducibility-oriented release of the locally validated
TSLDSeg audit implementation. It contains the original method plus three
existing audit conditions on BTCV, ACDC, and ISIC2018. It does **not** contain
datasets, pretrained weights, trained checkpoints, paper PDFs, experiment
outputs, or the separate sampling-step audit.

The upstream TSLDSeg README is preserved unchanged at
[`code/README.md`](code/README.md). Read [`UPSTREAM.md`](UPSTREAM.md),
[`MODIFICATIONS.md`](MODIFICATIONS.md), and
[`RELEASE_COMPLIANCE_REPORT.md`](RELEASE_COMPLIANCE_REPORT.md) before reuse.
The final line-level audit map is in
[`docs/AUDIT_IMPLEMENTATION_MAP.md`](docs/AUDIT_IMPLEMENTATION_MAP.md), the
12-cell parameter/command record is in
[`docs/EXPERIMENT_MATRIX.md`](docs/EXPERIMENT_MATRIX.md), and the complete
file-level responsibility table is in
[`docs/CODE_STRUCTURE.md`](docs/CODE_STRUCTURE.md).

## What is included

| Condition | `model.params.audit_mode` | Training objective | Inference |
|---|---|---|---|
| Full diffusion | `none` | Original epsilon-prediction plus latent segmentation term | DDIM-10 |
| Random Y_t | `train_random_yt` | Same target and loss; training Y_t input is independent Gaussian noise | Same as full |
| Shuffled Y_t | `train_shuffle_yt` | Same target and loss; training Y_t uses another batch case's Y_0 with current t/noise | Same as full |
| Core no diffusion | `core_no_diff` | Existing structural image-only clean-latent regression audit | Direct single pass |

The input-side audit modes do not remove TAM, HSEM, MCF, the conditioning
encoder, EMA, or other TSLDSeg components. See
[`docs/AUDIT_IMPLEMENTATION_MAP.md`](docs/AUDIT_IMPLEMENTATION_MAP.md) for the
exact code path and scientific interpretation.

## Supported runtime

The validated target is intentionally narrow:

- Windows and Conda
- Python 3.10.20
- PyTorch 2.11.0+cu128 / torchvision 0.26.0+cu128
- NVIDIA GeForce RTX 5090, compute capability 12.0
- one CUDA device; no CPU fallback

Create the environment from the release root:

```powershell
conda env create -f environment.yml
conda activate tsldseg-audit-rtx5090
python gpu_preflight.py
```

The preflight fails clearly if CUDA, the exact GPU, or a CUDA backward pass is
unavailable. Driver/runtime details from the validated machine are in
[`docs/GPU_ENVIRONMENT.md`](docs/GPU_ENVIRONMENT.md).

## External assets

Obtain the upstream KL-f8 and LSUN-churches latent-diffusion checkpoints. The
upstream download scripts are retained under `code/scripts/`, but no weights
are distributed here. Set these environment variables before training:

```powershell
$env:TSLDSEG_VAE_CKPT = 'X:\path\to\kl-f8\model.ckpt'
$env:TSLDSEG_LDM_CKPT = 'X:\path\to\lsun_churches256\model.ckpt'
$env:TSLDSEG_BTCV_PREPROCESSED_ROOT = 'X:\data\btcv_cache'
$env:TSLDSEG_ACDC_PREPROCESSED_ROOT = 'X:\data\acdc_cache'
$env:TSLDSEG_ISIC2018_PREPROCESSED_ROOT = 'X:\data\isic2018_cache'
```

Data acquisition remains subject to each dataset's own terms. Preprocessing
scripts accept explicit input/output paths and write no data into this release.
Obtain data only from the official providers: the
[BTCV/Synapse challenge](https://www.synapse.org/Synapse%3Asyn3193805/challenge/),
the [ACDC challenge](https://www.creatis.insa-lyon.fr/Challenge/acdc/), and the
[ISIC challenge data page](https://challenge.isic-archive.com/data/). Review and
accept the provider's current access, citation, and use conditions before
download; this repository grants no dataset rights.

## Fixed split contract

The ID-only manifests under `code/manifests/` are authoritative:

| Dataset | Train | Validation | Final test |
|---|---:|---:|---:|
| BTCV | 18 cases / 2,211 slices | 2 cases / 295 slices | 10 cases / 1,273 slices |
| ACDC | 70 patients / 1,304 slices | 10 patients / 182 slices | 20 patients / 416 slices |
| ISIC2018 | 2,594 images | 100 images | 1,000 images |

BTCV validation is exactly `case0008` and `case0001`; its final test set is the
other ten held-out cases. The historical 10% slice subset is excluded.
Validation logs use `val_avg_dice`, checkpointing retains only the best
validation-selected model, and final test metrics are produced only by the
explicit inference/evaluation commands. See
[`docs/DATA_SPLITS.md`](docs/DATA_SPLITS.md) and
[`docs/SPLIT_USAGE_MAP.md`](docs/SPLIT_USAGE_MAP.md).

Validate manifests and local caches before training:

```powershell
python tests\validate_splits.py `
  --data-root btcv=$env:TSLDSEG_BTCV_PREPROCESSED_ROOT `
  --data-root acdc=$env:TSLDSEG_ACDC_PREPROCESSED_ROOT `
  --data-root isic2018=$env:TSLDSEG_ISIC2018_PREPROCESSED_ROOT
```

## Train and select the best checkpoint

Run from `code/`. The config defaults to full diffusion; set only the audit
override for an audit run. Keep `--no-test True` so training never reports a
final test result.

```powershell
Set-Location code
python main.py --base configs\latent-diffusion\btcv-cls2-ldm-kl-8.yaml `
  --name btcv_train_shuffle_yt_s23 --logdir ..\runs --seed 23 `
  --scale_lr False --no-test True -t `
  model.params.audit_mode=train_shuffle_yt
```

Replace the config and data environment variable for ACDC or ISIC2018. Use one
of `none`, `train_random_yt`, `train_shuffle_yt`, or `core_no_diff`. Shuffled
Y_t requires training batch size greater than one; formal configs use 12.

The following PowerShell matrix issues the 12 formal GPU training commands
without changing any config defaults:

```powershell
$datasets = @(
  @{Name='btcv'; Config='configs\latent-diffusion\btcv-cls2-ldm-kl-8.yaml'},
  @{Name='acdc'; Config='configs\latent-diffusion\acdc-cls4-ldm-kl-8.yaml'},
  @{Name='isic2018'; Config='configs\latent-diffusion\isic-ldm-kl-8.yaml'}
)
$conditions = @(
  @{Name='full'; Mode='none'},
  @{Name='random_yt'; Mode='train_random_yt'},
  @{Name='shuffle_yt'; Mode='train_shuffle_yt'},
  @{Name='core_no_diff'; Mode='core_no_diff'}
)
foreach ($dataset in $datasets) {
  foreach ($condition in $conditions) {
    $runName = "$($dataset.Name)_$($condition.Name)_s23"
    python main.py --base $dataset.Config --name $runName --logdir ..\runs `
      --seed 23 --scale_lr False --no-test True -t `
      "model.params.audit_mode=$($condition.Mode)"
  }
}
```

This loop is intentionally a full experiment launcher, not a smoke test. The
expanded per-cell commands and all dataset-specific settings are recorded in
[`docs/EXPERIMENT_MATRIX.md`](docs/EXPERIMENT_MATRIX.md).

The run saves resolved project/Lightning configs and audit metadata. The
checkpoint callback monitors `val_avg_dice`, uses `mode=max`, `save_top_k=1`,
and does not save `last.ckpt`.

## Final test

Always pass the best validation-selected checkpoint and the fixed **test**
manifest. Full/random/shuffle checkpoints use DDIM-10:

```powershell
python scripts\slice2seg.py --dataset btcv `
  --data_dir $env:TSLDSEG_BTCV_PREPROCESSED_ROOT\BTCV\test `
  --manifest manifests\btcv\test.txt `
  --config configs\latent-diffusion\btcv-cls2-ldm-kl-8.yaml `
  --ckpt X:\runs\best-validation.ckpt --outdir X:\results\btcv `
  --sampler ddim --ddim_steps 10 --num_classes 2 --seed 23 --save_results
```

For `core_no_diff`, add `--audit_mode core_no_diff --sampler direct`. Use the
matching evaluation script in `code/scripts/` to summarize saved logits. Full
commands for all datasets are in [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md).

## Verification

The release was checked with 25 unit/static tests, exact real-cache validation,
and all 12 RTX 5090 end-to-end smoke conditions. Every condition includes an
optimizer step, validation metric, temporary best-checkpoint save/reload, and
test inference. To rerun the GPU matrix:

```powershell
python tests\gpu_smoke_12.py `
  --data-root btcv=$env:TSLDSEG_BTCV_PREPROCESSED_ROOT `
  --data-root acdc=$env:TSLDSEG_ACDC_PREPROCESSED_ROOT `
  --data-root isic2018=$env:TSLDSEG_ISIC2018_PREPROCESSED_ROOT `
  --ldm-ckpt $env:TSLDSEG_LDM_CKPT --vae-ckpt $env:TSLDSEG_VAE_CKPT
```

The smoke test retains no model or prediction artifacts: its temporary
validation-selected checkpoints are strictly reloaded and then deleted. Results
and limits are reported in
[`docs/GPU_SMOKE_TEST.md`](docs/GPU_SMOKE_TEST.md).

## Scope of conclusions

These audits test specific dependencies of the training interface. They do not
show that auxiliary modules are useless, that every diffusion checkpoint
ignores Y_t, or that a reverse trajectory is unnecessary. Interpretation must
respect the objective class and should combine the three audits with separate
sampling-step evidence.

## License and attribution

The primary upstream TSLDSeg code is distributed under the included MIT
license. The official TSLDSeg project also declares SDSeg lineage, so this
release retains the SDSeg CreativeML Open RAIL-M text at
[`LICENSES/SDSeg-CreativeML-Open-RAIL-M.txt`](LICENSES/SDSeg-CreativeML-Open-RAIL-M.txt)
for identified inherited material where those terms apply. The MIT file must
not be read as erasing that separate layer.

Retain `LICENSE`, `LICENSES/`, [`UPSTREAM.md`](UPSTREAM.md),
[`MODIFICATIONS.md`](MODIFICATIONS.md), and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) with redistributed copies.
See [`RELEASE_COMPLIANCE_REPORT.md`](RELEASE_COMPLIANCE_REPORT.md) for the audit
basis and its limits.
