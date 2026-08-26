# Third-party notices

The root NVIDIA Source Code License-NC governs the combined release. More permissive third-party terms continue to apply to the particular source portions identified below; their notices are not replaced by the root license.

| Component or file | Origin | Applicable text | Treatment |
|---|---|---|---|
| cDAL backbone and Denoising Diffusion GAN-derived training/model code | Hejrati/cDAL and NVlabs/denoising-diffusion-gan | `LICENSE`, `LICENSES/NVIDIA-Source-Code-License-NC.txt` | Retained and modified for non-commercial research/evaluation only. |
| `EMA.py` | NVlabs/denoising-diffusion-gan; file identifies adaptation from NVlabs/LSGM `util/ema.py` | `LICENSE`, `LICENSES/NVIDIA-Source-Code-License-LSGM.txt` | File header and source attribution retained. Conservatively preserve the LSGM non-commercial research/evaluation and NVIDIA-processor conditions for the identified adapted portion. |
| Score SDE-derived model utilities and layers | Google Research / Yang Song Score SDE | `LICENSES/Apache-2.0-Score-SDE.txt` | Existing attribution headers retained; modified files documented. |
| `score_sde/models/dense_layer.py` | Chin-Wei Huang, `sdeflow-light` | `code/score_sde/models/LICENSE_MIT`, `LICENSES/MIT-CW-Huang.txt` | MIT notice retained. |
| `score_sde/op/*` StyleGAN2 operations | rosinality/stylegan2-pytorch | `code/score_sde/op/LICENSE_MIT`, `LICENSES/MIT-StyleGAN2-PyTorch.txt` | MIT notice retained; optional-extension fallback documented. |
| `logger.py` | OpenAI Baselines, referenced commit `ea25b9e8...` | `LICENSES/MIT-OpenAI-Baselines.txt` | Source attribution comments retained. |
| `score_sde/models/nn.py` | MedSegDiff-derived utility code | `LICENSES/MIT-MedSegDiff.txt` | MIT text included. |
| Hippocampus loader/metrics in the source project | VinAI 3D-UCaps-derived code | `LICENSES/BSD-3-Clause-3D-UCaps.txt` | Those files are excluded from this three-dataset release; license retained as provenance evidence. |

PyTorch, torchvision, NumPy, SciPy, scikit-learn, Pillow, matplotlib, tqdm, nibabel, and Ninja are external dependencies installed from their official distribution channels. Their source is not vendored here and each remains subject to its own package license.

No dataset license is granted by this repository. Users must obtain BTCV/Synapse, ACDC, and ISIC2018 from their official sources and comply with the applicable access, use, and privacy terms.
