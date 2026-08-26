# Audit Implementation Map

Line numbers in this document refer to the final public files in this release.
Full is the reproduction baseline; Random-Yt, Shuffle-Yt, and Core-No-Diff are
the three controlled mechanism analyses.

## Shared public path and final line anchors

| Responsibility | Final public location |
| --- | --- |
| Dataset/condition registry and exact formal commands | `code/scripts/release_pipeline.py:37`, `:44`, `:140`, `:179` |
| Fail-closed exact RTX 5090 selection | `code/guided_diffusion/dist_util.py:20`, `:24`, `:53`, `:87` |
| Training entry and validation callback | `code/scripts/segmentation_train.py:79`, `:420`, `:486` |
| CUDA microbatch, timestep sampling, formal loss, backward | `code/guided_diffusion/train_util.py:365`, `:369`, `:380`, `:383`, `:411` |
| Formal condition loss implementation | `code/guided_diffusion/gaussian_diffusion.py:1053` |
| Validation CUDA transfer and Full/audit inference | `code/guided_diffusion/validation_util.py:92`, `:129`, `:131`, `:146` |
| Validation-best comparison and save metadata | `code/guided_diffusion/train_util.py:251`, `:316`, `:328`, `:338` |
| Final selected-checkpoint inference entry | `code/scripts/segmentation_sample.py:140`, `:151`, `:188`, `:234`, `:278` |
| UNet construction and forward | `code/guided_diffusion/script_util.py:78`, `:142`; `code/guided_diffusion/unet.py:400`, `:648` |

All formal model parameters, microbatches, model inputs/outputs, loss tensors,
backward operations, validation forwards, and final-test forwards use the
device returned by `dist_util.dev()`. That function accepts only logical
`cuda:0` whose exact name is `NVIDIA GeForce RTX 5090`. Preprocessing,
dataloader workers, serialization, and file-based metric aggregation may use
CPU; they are not model-compute fallbacks.

## Dataset targets and loaders

| Dataset | Target | Loader class and final line | Model interface |
| --- | --- | --- | --- |
| BTCV/Synapse | One binary foreground channel after the preserved label remapping/merge; two segmentation classes | `BTCVDataset`, `code/guided_diffusion/btcvloader.py:32` | image 1, mask 1, classes 2 |
| ACDC | Four one-hot channels: background, right ventricle, myocardium, left ventricle | `ACDCDataset`, `code/guided_diffusion/acdcloader.py:49` | image 1, mask 4, classes 4 |
| ISIC2018 | One binary lesion channel; two segmentation classes | `ISIC2018Dataset`, `code/guided_diffusion/isicloader.py:45` | image 3, mask 1, classes 2 |

The public orchestrator supplies each loader with a separate fixed training,
validation, or test manifest. Conditions never change the manifest or target.

## Exact condition semantics

### Full baseline

`training_losses_segmentation` creates the reference noisy mask with
`q_sample(mask, t, noise)` at `gaussian_diffusion.py:1098` and passes it to the
model unchanged at `:1099`. Uniformly sampled `t` is passed to the timestep
embedding. The prediction target remains epsilon/noise at `:1158`; learned
variance contributes VB and the loss is MSE + VB at `:1141` and `:1160-1164`.
Validation and final inference use DDPM reverse sampling, not DDIM.

### Random-Yt

Only while `model.training` is true, the model's Yt input is replaced by an
independent `randn_like(res_t_ref)` tensor at
`gaussian_diffusion.py:1112-1113`. The reference noisy mask used by the loss,
the current timestep, epsilon target, learned-variance term, model, reverse
sampler, and evaluation remain Full. Validation and test therefore use the
same Full inference path; this is a training-input intervention.

### Shuffle-Yt

Only while `model.training` is true, a strict within-batch derangement is built
by `_batch_derangement_indices` at `gaussian_diffusion.py:139`. The shuffled
mask is q-sampled with the current sample's `t` and the same sampled noise at
`:1101-1103`, then used only as model input. The Full reference noisy mask,
epsilon target, learned-variance term, loss, model, reverse sampler, and
evaluation remain unchanged. Effective training microbatch must exceed one;
formal batch is 8 and smoke batch is 2.

### Core-No-Diff

The branch begins at `gaussian_diffusion.py:1079`: it splits image/mask, calls
the model with image only, and returns direct logits/loss at `:1080-1088`.
`direct_segmentation_terms` at `:70` implements cross-entropy plus foreground
Dice and returns `ce`, `dice`, and their sum at `:128`. Model construction uses
image-only input, segmentation-class output, and disables the timestep
embedding at `script_util.py:184`, `:188`, and `:204`; `UNetModel.forward` at
`unet.py:648` rejects a supplied timestep at `:668`. Validation performs one
direct prediction at `validation_util.py:131-133`; final sampling likewise uses
one image-only forward at `segmentation_sample.py:234-235`. There is no Yt,
q_sample, timestep conditioning, diffusion objective, or reverse process in
the main Core path. This is not objective-preserving.

## Twelve dataset-condition entries

Each entry uses `release_pipeline.py` with the shown public CLI identifiers,
seed 10, the dataset target above, and the fixed train/validation/test flow.

| Dataset | Public condition / internal mode | Target and training branch | Timestep / validation / final denoising | Exact difference from Full | CUDA evidence required and observed |
| --- | --- | --- | --- | --- | --- |
| BTCV | `full` / `none` | Binary mask; reference q-sampled Yt; epsilon MSE+VB | Uniform train t; DDPM 100 / DDPM 1,000×5 | None | params/input/output/loss/backward/val/test on `cuda:0` |
| BTCV | `random-yt` / `train_random_yt` | Same target/loss; independent Gaussian Yt model input | Unchanged from Full | Training model input only | same CUDA chain; branch executed |
| BTCV | `shuffle-yt` / `train_shuffle_yt` | Same target/loss; batch-deranged-mask Yt model input | Unchanged from Full | Training model input only | same CUDA chain; branch executed with batch >1 |
| BTCV | `core-no-diff` / `core_no_diff` | Binary mask; direct 2-class CE + foreground Dice | No timestep; direct val/test forward; 0 reverse steps | Removes Yt/timestep/diffusion objective/reverse process | image/logits/loss/backward/direct val/test on `cuda:0` |
| ACDC | `full` / `none` | Four-class mask; reference q-sampled Yt; epsilon MSE+VB | Uniform train t; DDPM 100 / DDPM 1,000×5 | None | params/input/output/loss/backward/val/test on `cuda:0` |
| ACDC | `random-yt` / `train_random_yt` | Same target/loss; independent Gaussian Yt model input | Unchanged from Full | Training model input only | same CUDA chain; branch executed |
| ACDC | `shuffle-yt` / `train_shuffle_yt` | Same target/loss; batch-deranged-mask Yt model input | Unchanged from Full | Training model input only | same CUDA chain; branch executed with batch >1 |
| ACDC | `core-no-diff` / `core_no_diff` | Four-class mask; direct 4-class CE + foreground Dice | No timestep; direct val/test forward; 0 reverse steps | Removes Yt/timestep/diffusion objective/reverse process | image/logits/loss/backward/direct val/test on `cuda:0` |
| ISIC2018 | `full` / `none` | Binary lesion; reference q-sampled Yt; epsilon MSE+VB | Uniform train t; DDPM 100 / DDPM 1,000×5 | None | params/input/output/loss/backward/val/test on `cuda:0` |
| ISIC2018 | `random-yt` / `train_random_yt` | Same target/loss; independent Gaussian Yt model input | Unchanged from Full | Training model input only | same CUDA chain; branch executed |
| ISIC2018 | `shuffle-yt` / `train_shuffle_yt` | Same target/loss; batch-deranged-mask Yt model input | Unchanged from Full | Training model input only | same CUDA chain; branch executed with batch >1 |
| ISIC2018 | `core-no-diff` / `core_no_diff` | Binary lesion; direct 2-class CE + foreground Dice | No timestep; direct val/test forward; 0 reverse steps | Removes Yt/timestep/diffusion objective/reverse process | image/logits/loss/backward/direct val/test on `cuda:0` |

The real acceptance program `tests/gpu_smoke_12.py` verified every row,
including the exact branch contract and CUDA forward hooks: 12 PASS, 0 FAIL,
0 BLOCKED. See `docs/GPU_SMOKE_TEST.md`; its metrics are execution checks, not
performance results.
