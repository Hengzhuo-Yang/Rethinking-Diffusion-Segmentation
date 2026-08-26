# Paper-to-code implementation notes

The paper PDF was inspected to distinguish published method settings from the
locally validated executable code. The release does not silently rewrite the
validated implementation to make the two descriptions appear identical.

| Topic | Paper description | Local validated release |
|---|---|---|
| Steps / batch / LR | 50k / 12 / 1e-5 | Same |
| Optimizer | AdamW | AdamW |
| LR schedule | 10k warmup, linearly decaying | 5k warmup, then effectively constant (`f_min=1`) |
| Diffusion schedule | cosine, T=1000, s=0.008 | linear beta 0.0015→0.0155, T=1000 |
| Sampling | DDIM-10 | DDIM-10 for diffusion conditions |
| Resolution/augmentation | 256, horizontal and vertical flips | Same |
| Noise loss | L2 epsilon prediction | L1 epsilon prediction |
| Multi-objective weighting | paper describes MGDA | fixed `L_noise + L_seg`; no MGDA implementation identified |
| Precision | not used as a release assumption | training FP32 default; validation/inference autocast |

The paper's general 8:1:1 experimental split statement is not imposed on the
three audit datasets. This release uses the explicitly validated BTCV,
ACDC 70/10/20, and official ISIC2018 partitions documented in `DATA_SPLITS.md`.

These are implementation/protocol distinctions, not claims about the scientific
value of TSLDSeg or its auxiliary modules.

