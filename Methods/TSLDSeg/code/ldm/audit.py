import torch


# Local audit implementation preserved in the release; see MODIFICATIONS.md.
AUDIT_MODE_NONE_ALIASES = {"", "none", "null", "false", "full_diffusion", "original", "default"}
SUPPORTED_AUDIT_MODES = {"none", "train_random_yt", "train_shuffle_yt", "core_no_diff"}


def normalize_audit_mode(audit_mode):
    mode = str(audit_mode or "none").strip().lower()
    if mode in AUDIT_MODE_NONE_ALIASES:
        return "none"
    if mode not in SUPPORTED_AUDIT_MODES:
        raise ValueError(f"Unsupported audit_mode={audit_mode!r}. Supported modes: {sorted(SUPPORTED_AUDIT_MODES)}")
    return mode


def _batch_derangement(batch_size, device):
    if batch_size <= 1:
        raise ValueError(
            "audit_mode=train_shuffle_yt requires batch_size > 1 for the "
            "batch-level no-self-match shuffle."
        )
    order = torch.randperm(batch_size, device=device)
    perm = torch.empty_like(order)
    perm[order] = order.roll(1)
    if torch.any(perm == torch.arange(batch_size, device=device)):
        raise RuntimeError("Failed to construct a no-self-match batch permutation for train_shuffle_yt.")
    return perm


def audited_y_t_input(
    y_t_ref,
    audit_mode,
    training,
    *,
    x_start=None,
    t=None,
    noise=None,
    q_sample_fn=None,
):
    mode = normalize_audit_mode(audit_mode)
    train_random_yt_active = mode == "train_random_yt" and bool(training)
    train_shuffle_yt_active = mode == "train_shuffle_yt" and bool(training)
    permutation = None

    if train_random_yt_active:
        y_t_input = torch.randn_like(y_t_ref)
        source = "independent_standard_gaussian"
    elif train_shuffle_yt_active:
        missing = [
            name for name, value in {
                "x_start": x_start,
                "t": t,
                "noise": noise,
                "q_sample_fn": q_sample_fn,
            }.items()
            if value is None
        ]
        if missing:
            raise ValueError(
                "audit_mode=train_shuffle_yt requires the original x_start, "
                f"timestep, noise, and q_sample function; missing: {missing}"
            )
        if x_start.shape[0] != y_t_ref.shape[0]:
            raise ValueError(
                "audit_mode=train_shuffle_yt requires x_start and y_t_ref to "
                f"have the same batch size, got {x_start.shape[0]} and {y_t_ref.shape[0]}"
            )
        permutation = _batch_derangement(x_start.shape[0], x_start.device)
        y_t_input = q_sample_fn(x_start=x_start[permutation], t=t, noise=noise)
        source = "q_sample_shuffled_x_start_current_t_and_noise"
    else:
        y_t_input = y_t_ref
        source = "original_q_sample"

    same_shape = y_t_input.shape == y_t_ref.shape
    same_dtype = y_t_input.dtype == y_t_ref.dtype
    same_device = y_t_input.device == y_t_ref.device
    if not (same_shape and same_dtype and same_device):
        raise RuntimeError(
            "audit Y_t input must preserve reference shape, dtype, and device: "
            f"ref={(tuple(y_t_ref.shape), y_t_ref.dtype, y_t_ref.device)}, "
            f"input={(tuple(y_t_input.shape), y_t_input.dtype, y_t_input.device)}"
        )

    trace = {
        "audit_mode": mode,
        "audit_active": train_random_yt_active or train_shuffle_yt_active,
        "train_random_yt_active": train_random_yt_active,
        "train_shuffle_yt_active": train_shuffle_yt_active,
        "core_no_diff_active": mode == "core_no_diff",
        "y_t_ref_shape": tuple(y_t_ref.shape),
        "y_t_input_shape": tuple(y_t_input.shape),
        "same_shape": same_shape,
        "same_dtype": same_dtype,
        "same_device": same_device,
        "y_t_input_source": source,
        "random_y_t_type": "independent_standard_gaussian" if train_random_yt_active else "not_applicable",
        "shuffle_strategy": "batch_level_derangement" if train_shuffle_yt_active else "not_applicable",
        "shuffle_permutation": permutation.detach().cpu().tolist() if permutation is not None else None,
        "no_self_match_enforced": bool(train_shuffle_yt_active),
        "y_t_input_reconstructed_from_shuffled_x_start": bool(train_shuffle_yt_active),
        "current_timestep_reused_for_y_t_input": bool(train_shuffle_yt_active),
        "current_noise_reused_for_y_t_input": bool(train_shuffle_yt_active),
        "target_changed": False,
        "loss_changed": False,
        "model_output_changed": False,
        "image_condition_changed": False,
        "timestep_changed": False,
        "epsilon_noise_target_replaced_by_y_t_input_noise": False,
    }
    return y_t_input, trace
