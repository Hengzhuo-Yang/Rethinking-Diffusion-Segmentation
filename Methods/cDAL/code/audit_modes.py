import torch


VALID_AUDIT_MODES = ("none", "train_random_yt", "train_shuffle_yt", "core_no_diff")


def normalize_audit_mode(args):
    audit_mode = getattr(args, "audit_mode", "none")
    if audit_mode is None:
        audit_mode = "none"
    audit_mode = str(audit_mode).strip().lower()
    if audit_mode in ("", "false", "off", "null"):
        audit_mode = "none"
    if audit_mode not in VALID_AUDIT_MODES:
        raise ValueError("Unknown audit_mode '{}'. Expected one of: {}".format(
            audit_mode, ", ".join(VALID_AUDIT_MODES)
        ))
    return audit_mode


def derange_batch_indices(batch_size, device):
    if batch_size <= 1:
        raise ValueError("train_shuffle_yt requires batch_size > 1 for batch-level shuffling")
    base = torch.arange(batch_size, device=device)
    for _ in range(16):
        perm = torch.randperm(batch_size, device=device)
        if torch.all(perm != base):
            return perm
    return torch.roll(base, shifts=1)


def shuffle_batch_tensor(tensor):
    perm = derange_batch_indices(tensor.shape[0], tensor.device)
    return tensor.index_select(0, perm), perm


def _check_same_tensor_contract(y_t_input, y_t_ref):
    if y_t_input.shape != y_t_ref.shape:
        raise RuntimeError(
            "audit Y_t shape changed: input={} reference={}".format(
                tuple(y_t_input.shape), tuple(y_t_ref.shape)
            )
        )
    if y_t_input.dtype != y_t_ref.dtype:
        raise RuntimeError(
            "audit Y_t dtype changed: input={} reference={}".format(y_t_input.dtype, y_t_ref.dtype)
        )
    if y_t_input.device != y_t_ref.device:
        raise RuntimeError(
            "audit Y_t device changed: input={} reference={}".format(y_t_input.device, y_t_ref.device)
        )


def construct_training_yt_from_y0(
    y_t_ref,
    y0_for_yt,
    coeff,
    t,
    q_sample_pairs_fn,
    noise_t,
    noise_tp1,
):
    _, y_t_input = q_sample_pairs_fn(
        coeff,
        y0_for_yt,
        t,
        noise_t=noise_t,
        noise_tp1=noise_tp1,
    )
    _check_same_tensor_contract(y_t_input, y_t_ref)
    return y_t_input


def maybe_replace_training_yt(y_t_ref, args):
    audit_mode = normalize_audit_mode(args)
    if audit_mode == "train_random_yt":
        y_t_input = torch.randn_like(y_t_ref)
        _check_same_tensor_contract(y_t_input, y_t_ref)
        return y_t_input
    if audit_mode == "train_shuffle_yt":
        raise ValueError(
            "train_shuffle_yt must construct Y_t from a shuffled Y_0 with the current timestep and original noise"
        )
    if audit_mode == "core_no_diff":
        raise ValueError("core_no_diff bypasses Y_t construction and must not call maybe_replace_training_yt")
    return y_t_ref


