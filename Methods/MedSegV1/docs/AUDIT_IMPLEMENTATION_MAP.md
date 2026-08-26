# Audit implementation map

## Authority and interpretation rule

The **formal local working tree and recorded run metadata are the implementation
source of truth**. External narrative descriptions were used only for semantic
cross-checking. They are not executable specifications, are not evidence that a
branch was implemented, and do not override the current release code. If a
narrative description, historical comment, upstream README, or condition name
differs from the code paths mapped below, the current release code governs.

The formal condition registry is
[`code/configs/experiments.json:112-138`](../code/configs/experiments.json#L112-L138).
`full` maps to `audit_mode="none"`; it is the local Full baseline, not a separate
training branch. The three audit values are normalized and validated at the
training entry point in
[`segmentation_train.py:28-39`](../code/scripts/segmentation_train.py#L28-L39),
and the sampling and validation entry points use the same accepted set in
[`segmentation_sample.py:45-56`](../code/scripts/segmentation_sample.py#L45-L56)
and
[`validation_runner.py:55-66`](../code/scripts/validation_runner.py#L55-L66).

## Shared notation and data contract

For sample `i`:

- `I_i` is the three-channel image.
- `Y0_i` is the clean segmentation target: one binary channel for BTCV or
  ISIC2018, and three mutually exclusive foreground channels for ACDC.
- `t_i` is the sampled diffusion timestep.
- `epsilon_i` is standard Gaussian noise with the same shape as `Y0_i`.
- `q(Y0_i, t_i, epsilon_i)` is the forward-diffused mask `Yt_i`, implemented by
  [`gaussian_diffusion.py:308-324`](../code/guided_diffusion/gaussian_diffusion.py#L308-L324).

The dataset target contract is implemented at these locations:

| Dataset | Image input | Clean target | Source |
|---|---|---|---|
| BTCV | RGB float tensor in `[0,1]` | One thresholded foreground channel | [`btcvloader.py:200-217`](../code/guided_diffusion/btcvloader.py#L200-L217) |
| ISIC2018 | RGB float tensor in `[0,1]` | One thresholded lesion channel | [`isicloader.py:197-214`](../code/guided_diffusion/isicloader.py#L197-L214) |
| ACDC | RGB float tensor in `[0,1]` | Three channels: RV, myocardium, LV; overlaps rejected | [`acdcloader.py:289-323`](../code/guided_diffusion/acdcloader.py#L289-L323) |

The training loop concatenates the loader outputs as `x_start = [I, Y0]` in
[`train_util.py:385-389`](../code/guided_diffusion/train_util.py#L385-L389).
The diffusion object then treats the last `num_mask_channels` as the mask and all
earlier channels as the image in
[`gaussian_diffusion.py:266-275`](../code/guided_diffusion/gaussian_diffusion.py#L266-L275).

## Shared formal diffusion definition

The formal configuration explicitly selects MedSegDiff V1, 1,000 linear
diffusion steps, `learn_sigma=true`, and float32 in
[`experiments.json:13-26`](../code/configs/experiments.json#L13-L26). Model
construction maps `version != "new"` to `UNetModel_v1preview`, sizes the
diffusion output as two values per mask channel when variance is learned, and
sizes the direct core output as one binary logit or four ACDC class logits in
[`script_util.py:186-229`](../code/guided_diffusion/script_util.py#L186-L229).

`predict_xstart` defaults to false in
[`script_util.py:11-24`](../code/guided_diffusion/script_util.py#L11-L24), so the
diffusion factory selects `ModelMeanType.EPSILON`. With `learn_sigma=true` it
selects learned-range variance; the exact factory mapping is
[`script_util.py:430-471`](../code/guided_diffusion/script_util.py#L430-L471).
Consequently, Full, Random-Yt, and Shuffle-Yt all predict `epsilon_i` plus a
learned variance component.

## Four-condition summary

| Condition | Model input during training | Supervision target | Training loss | Validation and test inference |
|---|---|---|---|---|
| **Full** (`audit_mode=none`) | `[I_i, q(Y0_i,t_i,epsilon_i)]`, conditioned on `t_i` | Current sample's `epsilon_i`; learned variance uses current `Y0_i` and its reference `Yt_i` | Epsilon MSE + learned-variance VB | Normal image-conditioned reverse diffusion; formal final test uses 1,000 steps and ensemble 5 |
| **train_random_yt** | `[I_i, R_i]`, where `R_i` is a fresh independent `randn_like(Yt_i)` tensor; `t_i` is unchanged | Exactly the same current-sample epsilon/VB targets as Full | Exactly the same epsilon MSE + learned-variance VB construction as Full | Exactly the same reverse-diffusion branch and output policy as Full |
| **train_shuffle_yt** | `[I_i, q(Y0_pi(i),t_i,epsilon_i)]`, with a no-self-match batch permutation `pi(i) != i` | Current sample's original `epsilon_i`; VB still uses current `Y0_i` and current reference `Yt_i` | Exactly the same epsilon MSE + learned-variance VB construction as Full | Exactly the same reverse-diffusion branch and output policy as Full |
| **core_no_diff** | `I_i` only, through the existing highway segmentation network; no timestep | Clean segmentation `Y0_i` | Binary: BCE-with-logits + soft Dice. ACDC: four-class CE + foreground soft Dice | Generic batch setup may allocate an auxiliary mask-noise tensor, but Core consumes only `I_i`; it makes one direct logits call with no timestep loop, reverse diffusion, calibration fusion, or ensemble sampling |

## Full (`audit_mode=none`)

### Training input

The training loop samples timesteps for every non-core condition and forwards the
condition name to the diffusion loss in
[`train_util.py:397-422`](../code/guided_diffusion/train_util.py#L397-L422).
The diffusion loss binarizes the clean mask channels, constructs the case-matched
reference `Yt_i = q(Y0_i,t_i,epsilon_i)`, and replaces only the mask portion of
the concatenated tensor in
[`gaussian_diffusion.py:1129-1136`](../code/guided_diffusion/gaussian_diffusion.py#L1129-L1136).
With mode `none`, neither audit replacement branch runs. The V1 U-Net receives
the image, matched noisy mask, and timestep; its normal forward separates the
image channels for the highway condition encoder while retaining the complete
tensor in the timestep-conditioned U-Net in
[`unet.py:773-809`](../code/guided_diffusion/unet.py#L773-L809).

### Target and loss

The V1 output is split into epsilon prediction and learned-range variance. The
VB term is calculated against the current clean mask and its case-matched
reference `Yt_i` in
[`gaussian_diffusion.py:1177-1199`](../code/guided_diffusion/gaussian_diffusion.py#L1177-L1199).
The target table selects `epsilon_i` for the formal `ModelMeanType.EPSILON`, and
the total is `MSE(epsilon_i, epsilon_hat_i) + VB` in
[`gaussian_diffusion.py:1201-1216`](../code/guided_diffusion/gaussian_diffusion.py#L1201-L1216).

Normal V1 forward returns `None` in the legacy calibration-output slot and does
not execute the Generic_UNet localization decoder. There is no active
`loss_cal` term in the local total. This paper-aligned routing is explicitly
recorded in
[`segmentation_train.py:54-93`](../code/scripts/segmentation_train.py#L54-L93).

### Inference

Normal inference initializes only the mask channels with Gaussian noise while
keeping the image channels fixed, then runs the requested reverse process in
[`gaussian_diffusion.py:611-708`](../code/guided_diffusion/gaussian_diffusion.py#L611-L708).
At each progressive DDPM step, the original image is concatenated back with the
current mask state before the timestep-conditioned model call in
[`gaussian_diffusion.py:710-769`](../code/guided_diffusion/gaussian_diffusion.py#L710-L769).

The formal final runner assigns non-core conditions the dataset's configured
1,000 steps and ensemble size 5 in
[`run_experiment.py:152-182`](../code/scripts/run_experiment.py#L152-L182).
The V1 sampling entry accepts only the diffusion `sample` for normal V1 and
fuses sample ensemble members in
[`segmentation_sample.py:499-544`](../code/scripts/segmentation_sample.py#L499-L544)
and
[`segmentation_sample.py:587-638`](../code/scripts/segmentation_sample.py#L587-L638).
The formal configuration uses `sample` with threshold 0.5 for all three datasets.

## `train_random_yt`

### Training input intervention

The current image and timestep are unchanged. After constructing the normal
case-matched reference `Yt_i`, the code creates an independent
`R_i = randn_like(Yt_i)`, verifies identical shape/dtype/device, and replaces
only the mask channels sent to the model in
[`gaussian_diffusion.py:1160-1171`](../code/guided_diffusion/gaussian_diffusion.py#L1160-L1171).

`R_i` is **not** `epsilon_i`, is not used to recompute a diffusion target, and
does not replace the current sample's target noise. The original reference
`Yt_i` and original `epsilon_i` remain live outside the model-input tensor.

### Target and loss

After the input replacement, execution rejoins the same model-output split,
current-sample VB calculation, epsilon target table, and total-loss code as Full
at
[`gaussian_diffusion.py:1177-1216`](../code/guided_diffusion/gaussian_diffusion.py#L1177-L1216).
Thus the intervention changes what the network observes, not the formal target
or loss formula.

### Inference

Neither final sampling nor validation has a Random-Yt inference branch. Both
entry points special-case only Core-No-Diff; every other mode enters the same
reverse-diffusion path in
[`segmentation_sample.py:490-505`](../code/scripts/segmentation_sample.py#L490-L505)
and
[`validation_runner.py:221-266`](../code/scripts/validation_runner.py#L221-L266).
Random-Yt is therefore a training-input audit only.

## `train_shuffle_yt`

### Training input intervention

The entry point rejects batch size or microbatch size one in
[`segmentation_train.py:139-147`](../code/scripts/segmentation_train.py#L139-L147).
Inside each actual microbatch, the loss builds a random cyclic permutation by
randomizing the order, rolling it once, and mapping it back. It explicitly
checks that no index maps to itself. The noisy input for position `i` is then
constructed from another member's clean mask using position `i`'s original
`t_i` and `epsilon_i` in
[`gaussian_diffusion.py:1137-1159`](../code/guided_diffusion/gaussian_diffusion.py#L1137-L1159).

The image is not shuffled. The permutation is local to the current microbatch,
not the whole dataset or epoch. This is why the formal batch and microbatch must
both permit at least two members.

### Target and loss

The shuffled mask is used only to build the model-input `Yt`. The reference
`res_t` created from current `Y0_i` remains unchanged, the VB term still uses
`x_start=res` and `x_t=res_t`, and the target remains current `epsilon_i` at
[`gaussian_diffusion.py:1182-1207`](../code/guided_diffusion/gaussian_diffusion.py#L1182-L1207).
The same epsilon MSE plus VB total is then assembled at
[`gaussian_diffusion.py:1212-1216`](../code/guided_diffusion/gaussian_diffusion.py#L1212-L1216).

### Inference

There is no shuffling at validation or test time. As with Random-Yt, the only
nonstandard inference branch is Core-No-Diff, so Shuffle-Yt uses the exact Full
reverse-diffusion and output path mapped above.

## `core_no_diff`

### Training input and target

The training loop does not sample a timestep and does not apply timestep
weights for this condition in
[`train_util.py:408-421`](../code/guided_diffusion/train_util.py#L408-L421).
The diffusion-loss function extracts the image and clean target channels and
returns through the direct branch before Gaussian noise, `q_sample`, or the
diffusion target are constructed in
[`gaussian_diffusion.py:1117-1134`](../code/guided_diffusion/gaussian_diffusion.py#L1117-L1134).

For the explicit Core-No-Diff counterfactual only,
`core_no_diff_logits(image)` invokes the otherwise dormant localization decoder
with `hs=None` and `return_logits=True` in
[`unet.py:764-770`](../code/guided_diffusion/unet.py#L764-L770). This bypasses
the timestep embedding and diffusion U-Net forward; it does not create a second
independently designed backbone.

Binary datasets use one output logit. ACDC uses four logits, including
background, because model construction derives direct-output channels from
`num_seg_classes` in
[`script_util.py:186-229`](../code/guided_diffusion/script_util.py#L186-L229).

### Loss

For one-logit binary targets, the direct loss is per-sample
`BCEWithLogits + soft Dice`. For multiclass output, the three foreground target
channels are converted to background/RV/myocardium/LV class indices and the
loss is `cross_entropy + mean foreground soft Dice`. Background is included in
cross-entropy but excluded from the Dice term. The authoritative implementation
is
[`gaussian_diffusion.py:49-108`](../code/guided_diffusion/gaussian_diffusion.py#L49-L108).

Core-No-Diff therefore changes the target representation, model output, and
objective relative to the diffusion conditions. It is a discriminative-capacity
counterfactual, not an objective-preserving `Yt` input intervention.

### Inference

Sampling converts the direct logits to sigmoid probabilities for a one-logit
binary output or softmax foreground probabilities for multiclass output in
[`segmentation_sample.py:110-137`](../code/scripts/segmentation_sample.py#L110-L137).
The shared batch setup allocates an auxiliary mask-noise tensor before condition
branching, but Core never consumes that tensor. Its branch moves only the image
to the device, makes one image-only call, and skips the reverse loop in
[`segmentation_sample.py:476-505`](../code/scripts/segmentation_sample.py#L476-L505).
Validation mirrors this direct behavior in
[`validation_runner.py:69-85`](../code/scripts/validation_runner.py#L69-L85)
and
[`validation_runner.py:221-266`](../code/scripts/validation_runner.py#L221-L266).

The formal runner records zero sampling steps and ensemble size one for Core in
[`run_experiment.py:152-182`](../code/scripts/run_experiment.py#L152-L182).
Thresholding and the dataset metric policy remain the same as the corresponding
Full condition; only the prediction path changes.

## Changed/unchanged contract

| Property | Full | Random-Yt | Shuffle-Yt | Core-No-Diff |
|---|---:|---:|---:|---:|
| Image paired with current target | Yes | Yes | Yes | Yes |
| Diffusion timestep sampled | Yes | Yes | Yes | No |
| Model-input noisy mask is case matched | Yes | No, independent Gaussian | No, another microbatch member | Not used |
| No-self-match batch permutation | No | No | Yes | No |
| Epsilon target is current sample's original noise | Yes | Yes | Yes | Not used |
| Learned-variance VB uses current clean/reference mask | Yes | Yes | Yes | Not used |
| Active calibration loss | No | No | No | No |
| Direct clean-mask segmentation objective | No | No | No | Yes |
| Reverse diffusion at validation/test | Yes | Yes | Yes | No |
| Dataset metric definition changed | No | No | No | No |

## Executable evidence and guardrails

- The focused semantic tests assert case-matched Full input and original target
  at
  [`test_train_random_yt_audit.py:51-69`](../tests/test_train_random_yt_audit.py#L51-L69),
  Random-Yt input-only replacement at
  [`:72-95`](../tests/test_train_random_yt_audit.py#L72-L95), Shuffle-Yt semantics
  at [`:97-121`](../tests/test_train_random_yt_audit.py#L97-L121), batch-size
  rejection at [`:124-140`](../tests/test_train_random_yt_audit.py#L124-L140),
  and the direct core objective at
  [`:143-163`](../tests/test_train_random_yt_audit.py#L143-L163).
  Their CPU fixture explicitly sets `learn_sigma=False` at
  [`:34-45`](../tests/test_train_random_yt_audit.py#L34-L45). These invariants
  establish the model-input replacement, epsilon-target, and direct-Core
  boundaries, but do not by themselves prove learned-variance VB equivalence
  for the formal `learn_sigma=true` configuration. That VB claim follows from
  the production branch and formal-configuration mapping documented above.
- The GPU matrix defines the complete three-dataset by four-condition mapping at
  [`gpu_smoke_matrix.py:27-34`](../tests/gpu_smoke_matrix.py#L27-L34) and calls
  the same production loss implementation at
  [`gpu_smoke_matrix.py:241-280`](../tests/gpu_smoke_matrix.py#L241-L280).
- Training writes condition-specific audit metadata, including whether target,
  loss, timestep, validation, and inference change, at
  [`segmentation_train.py:54-113`](../code/scripts/segmentation_train.py#L54-L113).

These tests and metadata are supporting evidence. The implementation itself is
the production path in `gaussian_diffusion.py`, `unet.py`, `train_util.py`,
`segmentation_sample.py`, and `validation_runner.py` mapped above.
