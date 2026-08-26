# Reproduction commands

All commands assume the release Conda environment is active. Use paths that you
control; examples use `X:` placeholders and do not encode the build machine.

## 1. Environment and assets

```powershell
conda env create -f environment.yml
conda activate tsldseg-audit-rtx5090
$env:TSLDSEG_VAE_CKPT = 'X:\assets\kl-f8\model.ckpt'
$env:TSLDSEG_LDM_CKPT = 'X:\assets\lsun_churches256\model.ckpt'
$env:TSLDSEG_BTCV_PREPROCESSED_ROOT = 'X:\data\btcv'
$env:TSLDSEG_ACDC_PREPROCESSED_ROOT = 'X:\data\acdc'
$env:TSLDSEG_ISIC2018_PREPROCESSED_ROOT = 'X:\data\isic2018'
python gpu_preflight.py
```

## 2. Optional preprocessing

Run from `code/`. Each output directory must be absent or empty.

```powershell
python scripts\preprocess_btcv_synapse_to_tsldseg.py `
  --raw_root X:\raw\BTCV --output_root $env:TSLDSEG_BTCV_PREPROCESSED_ROOT

python scripts\preprocess_acdc_to_tsldseg.py `
  --raw_root X:\raw\ACDC --output_root $env:TSLDSEG_ACDC_PREPROCESSED_ROOT

python scripts\preprocess_isic2018_to_tsldseg.py `
  --raw_root X:\raw\ISIC2018 --output_root $env:TSLDSEG_ISIC2018_PREPROCESSED_ROOT
```

Validate before any experiment:

```powershell
Set-Location ..
python tests\validate_splits.py `
  --data-root btcv=$env:TSLDSEG_BTCV_PREPROCESSED_ROOT `
  --data-root acdc=$env:TSLDSEG_ACDC_PREPROCESSED_ROOT `
  --data-root isic2018=$env:TSLDSEG_ISIC2018_PREPROCESSED_ROOT
Set-Location code
```

## 3. Training matrix

Choose one dataset config:

```text
BTCV:     configs\latent-diffusion\btcv-cls2-ldm-kl-8.yaml
ACDC:     configs\latent-diffusion\acdc-cls4-ldm-kl-8.yaml
ISIC2018: configs\latent-diffusion\isic-ldm-kl-8.yaml
```

Choose one mode:

```text
full diffusion:   none
random Y_t:       train_random_yt
shuffled Y_t:     train_shuffle_yt
core no diffusion: core_no_diff
```

Template:

```powershell
$config = 'configs\latent-diffusion\btcv-cls2-ldm-kl-8.yaml'
$mode = 'none'
$runName = 'btcv_full_s23'
python main.py --base $config --name $runName --logdir ..\runs `
  --seed 23 --scale_lr False --no-test True -t `
  model.params.audit_mode=$mode
```

Repeat for 12 independent cells. Do not resume one condition from another audit
checkpoint. The run directory contains resolved configs, validation history,
audit metadata where applicable, and the top-1 validation checkpoint.

## 4. Final test inference

Dataset-specific test bindings:

| Dataset | Data directory suffix | Manifest | Classes | Evaluator |
|---|---|---|---:|---|
| BTCV | `BTCV\test` | `manifests\btcv\test.txt` | 2 | `evaluate_btcv_binary_logits.py` |
| ACDC | `ACDC\testing` | `manifests\acdc\test.txt` | 4 | `evaluate_acdc_multiclass_logits.py` |
| ISIC2018 | `ISIC18\testing` | `manifests\isic2018\test.txt` | 2 | `evaluate_isic2018_binary_logits.py` |

Diffusion-condition template:

```powershell
python scripts\slice2seg.py --dataset acdc `
  --data_dir $env:TSLDSEG_ACDC_PREPROCESSED_ROOT\ACDC\testing `
  --manifest manifests\acdc\test.txt `
  --config configs\latent-diffusion\acdc-cls4-ldm-kl-8.yaml `
  --ckpt X:\runs\acdc_full_s23\best-validation.ckpt `
  --outdir X:\results\acdc_full_s23 --sampler ddim --ddim_steps 10 `
  --num_classes 4 --seed 23 --save_results
```

Core-no-diff template uses `--sampler direct --audit_mode core_no_diff` and no
reverse-diffusion step count. Pass the saved-logit directory and real test data
directory to the matching evaluator; each evaluator exposes `--help` with its
CSV/JSON output arguments.

## 5. Verification

```powershell
python gpu_preflight.py
python -m pytest -p no:cacheprovider tests
python tests\gpu_smoke_12.py `
  --data-root btcv=$env:TSLDSEG_BTCV_PREPROCESSED_ROOT `
  --data-root acdc=$env:TSLDSEG_ACDC_PREPROCESSED_ROOT `
  --data-root isic2018=$env:TSLDSEG_ISIC2018_PREPROCESSED_ROOT `
  --ldm-ckpt $env:TSLDSEG_LDM_CKPT --vae-ckpt $env:TSLDSEG_VAE_CKPT
```

