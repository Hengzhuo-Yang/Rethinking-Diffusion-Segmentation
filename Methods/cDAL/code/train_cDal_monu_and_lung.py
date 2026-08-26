# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license,  01see the LICENSE file.
# ---------------------------------------------------------------

import argparse
import json
import csv
import hashlib
from pathlib import Path

import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist

from metrics import sampling_major_vote_func
from utils import *
from score_sde.models.discriminator import Discriminator_large
from score_sde.models.ncsnpp_generator_adagn import NCSNpp
from EMA import EMA

from preprocess_dataset.dataset import create_dataset
import logger
from audit_modes import (
    construct_training_yt_from_y0,
    maybe_replace_training_yt,
    normalize_audit_mode,
    shuffle_batch_tensor,
)


def count_parameters(module):
    return int(sum(parameter.numel() for parameter in unwrap_module(module).parameters()))

def forward_core_no_diff(generator, condition_image, latent_z):
    return unwrap_module(generator).forward_core_no_diff(condition_image, latent_z)


def write_training_metadata(exp_path, args, netG, netD):
    exp_path = Path(exp_path)
    audit_mode = normalize_audit_mode(args)
    metadata = {
        "audit_mode": audit_mode,
        "random_seed": int(args.seed),
        "parameter_count_generator": count_parameters(netG),
        "parameter_count_discriminator": count_parameters(netD),
        "random_yt_type": "independent_standard_gaussian" if audit_mode == "train_random_yt" else "not_applicable",
        "shuffle_strategy": "batch_level" if audit_mode == "train_shuffle_yt" else "not_applicable",
        "no_self_match_enforced": audit_mode == "train_shuffle_yt",
        "y_t_input_constructed_from_another_case_mask": audit_mode == "train_shuffle_yt",
        "original_output_type": "mask_x0",
        "original_target_type": "mask_y0",
        "original_loss_type": "mse_x0_prediction_to_ground_truth_mask",
        "target_changed": False,
        "loss_changed": False,
        "model_output_changed": False,
        "image_condition_changed": False,
        "timestep_changed": False,
        "timestep_sampling_changed": False,
        "inference_sampling_changed": False,
        "retained_auxiliary_modules": [
            "discriminator",
            "discriminator_feature_attention",
            "random_latent_z",
            "posterior_sampling",
        ],
        "modified_auxiliary_modules": [],
        "train_random_yt_scope": (
            "Only the training-time generator noisy-mask input is replaced by torch.randn_like(Y_t). "
            "The original target mask, MSE loss, timestep, image condition, posterior sampler, "
            "validation, and final evaluation remain unchanged."
        ) if audit_mode == "train_random_yt" else "",
        "train_shuffle_yt_scope": (
            "Only the training-time generator noisy-mask input is rebuilt from another batch sample's mask. "
            "The current sample's image, timestep, forward-diffusion noise, target mask, MSE loss, "
            "posterior sampler, validation, and final evaluation remain unchanged."
        ) if audit_mode == "train_shuffle_yt" else "",
    }
    if audit_mode == "core_no_diff":
        metadata.update({
            "counterfactual_type": "discriminative_capacity_diffusion_free_main_path",
            "objective_preserving": False,
            "target_changed": False,
            "loss_changed": False,
            "model_output_changed": False,
            "training_path_changed": True,
            "image_condition_changed": False,
            "timestep_changed": True,
            "timestep_sampling_changed": True,
            "inference_sampling_changed": True,
            "uses_Y_t_in_main_core": False,
            "uses_timestep_in_main_core": False,
            "uses_q_sample_for_main_core": False,
            "uses_reverse_sampler_for_main_core": False,
            "diffusion_loss_used_for_main_core": False,
            "uses_non_diffusion_latent": True,
            "latent_is_diffusion_noise": False,
            "retained_non_diffusion_modules": [
                "image_condition_encoder",
                "random_latent_z",
                "z_transform",
                "adaptive_group_norm",
                "clean_mask_x0_prediction_head",
            ],
            "removed_diffusion_modules": [
                "training_Y_t_or_x_tp1_input",
                "timestep_conditioning",
                "q_sample_main_path_input_construction",
                "discriminator_feature_attention_from_noisy_pair",
                "posterior_reverse_sampler_for_validation_and_test",
            ],
            "retained_auxiliary_modules": [
                "random_latent_z",
                "adaptive_group_norm",
            ],
            "modified_auxiliary_modules": [
                "generator_forward_path_uses_image_condition_features_as_core_input",
            ],
            "incompatible_auxiliary_modules": [
                "discriminator_feature_attention_requires_Y_t_timestep_and_noisy_pair",
                "posterior_sampling_requires_reverse_diffusion_state",
            ],
            "sampling_steps_main_core": 0,
            "main_core_input": "image_condition_plus_non_diffusion_latent_z",
            "core_no_diff_loss_formula": "mean_flat((Y0 - G_core(I, z)) ** 2)",
            "core_no_diff_scope": (
                "Structural counterfactual. Removes Y_t, timestep conditioning, q_sample main-path input, "
                "discriminator-derived noisy-pair attention, and reverse posterior sampling while retaining the "
                "paper latent embedding z and clean-mask MSE target."
            ),
        })
    (exp_path / "audit_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    config = dict(vars(args))
    config["audit_mode"] = audit_mode
    (exp_path / "resolved_config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")


# %% Diffusion coefficients
def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var


def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)


def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)

    return out


def get_time_schedule(args, device):
    n_timestep = args.num_timesteps
    eps_small = 1e-3
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small
    return t.to(device)


def get_sigma_schedule(args, device):
    n_timestep = args.num_timesteps
    beta_min = args.beta_min
    beta_max = args.beta_max
    eps_small = 1e-3

    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small

    if args.use_geometric:
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        var = var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]

    first = torch.tensor(1e-8)
    betas = torch.cat((first[None], betas)).to(device)
    betas = betas.type(torch.float32)
    sigmas = betas ** 0.5
    a_s = torch.sqrt(1 - betas)
    return sigmas, a_s, betas


class Diffusion_Coefficients():
    def __init__(self, args, device):
        self.sigmas, self.a_s, _ = get_sigma_schedule(args, device=device)
        self.a_s_cum = np.cumprod(self.a_s.cpu())
        self.sigmas_cum = np.sqrt(1 - self.a_s_cum ** 2)
        self.a_s_prev = self.a_s.clone()
        self.a_s_prev[-1] = 1

        self.a_s_cum = self.a_s_cum.to(device)
        self.sigmas_cum = self.sigmas_cum.to(device)
        self.a_s_prev = self.a_s_prev.to(device)


def q_sample(coeff, x_start, t, *, noise=None):
    """
    Diffuse the data (t == 0 means diffused for t step)
    """
    if noise is None:
        noise = torch.randn_like(x_start)

    x_t = extract(coeff.a_s_cum, t, x_start.shape) * x_start + \
          extract(coeff.sigmas_cum, t, x_start.shape) * noise

    return x_t


def q_sample_pairs(coeff, x_start, t, *, noise_t=None, noise_tp1=None, return_noise=False):
    """
    Generate a pair of disturbed images for training.
    Optional noises let audit modes rebuild x_{t+1} from another x_0 while
    preserving the reference timestep and forward-diffusion randomness.
    """
    if noise_tp1 is None:
        noise_tp1 = torch.randn_like(x_start)
    if noise_t is None:
        noise_t = torch.randn_like(x_start)
    x_t = q_sample(coeff, x_start, t, noise=noise_t)
    x_t_plus_one = extract(coeff.a_s, t + 1, x_start.shape) * x_t + \
                   extract(coeff.sigmas, t + 1, x_start.shape) * noise_tp1

    if return_noise:
        return x_t, x_t_plus_one, noise_t, noise_tp1
    return x_t, x_t_plus_one


# %% posterior sampling
class Posterior_Coefficients():
    def __init__(self, args, device):
        _, _, self.betas = get_sigma_schedule(args, device=device)

        # we don't need the zeros
        self.betas = self.betas.type(torch.float32)[1:]

        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
            (torch.tensor([1.], dtype=torch.float32, device=device), self.alphas_cumprod[:-1]), 0
        )
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        self.posterior_mean_coef2 = (
                (1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod))

        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))


def sample_posterior(coefficients, x_0, x_t, t):
    def q_posterior(x_0, x_t, t):
        mean = (
                extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
                + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped

    def p_sample(x_0, x_t, t):
        mean, _, log_var = q_posterior(x_0, x_t, t)

        noise = torch.randn_like(x_t)

        nonzero_mask = (1 - (t == 0).type(torch.float32))

        return mean + nonzero_mask[:, None, None, None] * torch.exp(0.5 * log_var) * noise

    sample_x_pos = p_sample(x_0, x_t, t)

    return sample_x_pos


def sample_from_model(coefficients, generator, n_time, x_init, y, opt):
    audit_mode = normalize_audit_mode(opt)
    if audit_mode == "core_no_diff":
        with torch.no_grad():
            latent_z = torch.randn(y.size(0), opt.nz, device=y.device)
            return forward_core_no_diff(generator, y, latent_z)

    x = x_init
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)

            t_time = t
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)
            x_0 = generator(x, t_time, y, latent_z)
            # x_0 = generator(x, t_time, y)
            x_new = sample_posterior(coefficients, x_0, x, t)
            x = x_new.detach()

    return x
# %%



def append_csv_row(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open('a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def update_gpu_evidence(args, **values):
    evidence_path = str(getattr(args, "gpu_evidence_json", "") or "").strip()
    if not evidence_path:
        return
    path = Path(evidence_path)
    payload = {}
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(values)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_best_checkpoint(netG, exp_path, train_step, mIoU, dice, args, metric_source):
    best_path = Path(exp_path) / args.best_checkpoint_name
    metadata_path = Path(exp_path) / args.best_metadata_name
    torch.save(unwrap_module(netG).state_dict(), best_path)
    metadata = {
        'train_step': int(train_step),
        'best_dice': float(dice),
        'best_miou': float(mIoU),
        'metric_source': Path(metric_source).name,
        'checkpoint': args.best_checkpoint_name,
        'dataset': args.dataset,
        'validation_split': args.val_split,
        'validation_manifest': str(args.val_manifest),
        'validation_manifest_sha256': sha256_file(args.val_manifest),
        'metric_policy': 'skip_empty_ground_truth_slices' if args.metric_skip_empty_gt else 'official_sklearn_all_slices',
        'audit_mode': normalize_audit_mode(args),
        'original_output_type': 'mask_x0',
        'original_target_type': 'mask_y0',
        'original_loss_type': 'mse_x0_prediction_to_ground_truth_mask',
        'target_changed': False,
        'loss_changed': False,
        'model_output_changed': False,
        'shuffle_strategy': 'batch_level' if normalize_audit_mode(args) == 'train_shuffle_yt' else 'not_applicable',
        'no_self_match_enforced': normalize_audit_mode(args) == 'train_shuffle_yt',
        'num_timesteps': int(args.num_timesteps),
        'major_vote_number': int(args.major_vote_number),
        'max_train_steps': int(args.max_train_steps),
        'val_interval_steps': int(args.val_interval_steps),
    }
    if normalize_audit_mode(args) == 'core_no_diff':
        metadata.update({
            'objective_preserving': False,
            'training_path_changed': True,
            'inference_sampling_changed': True,
            'uses_Y_t_in_main_core': False,
            'uses_timestep_in_main_core': False,
            'uses_q_sample_for_main_core': False,
            'uses_reverse_sampler_for_main_core': False,
            'uses_non_diffusion_latent': True,
            'latent_is_diffusion_noise': False,
            'sampling_steps_main_core': 0,
            'core_no_diff_loss_formula': 'mean_flat((Y0 - G_core(I, z)) ** 2)',
        })
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return best_path, metadata_path

def train(rank, gpu, args):

    set_random_seed_for_iterations(args.seed)
    device = dev(gpu)

    exp = args.exp
    parent_dir = Path(args.output_dir) / args.dataset
    exp_path = parent_dir / exp
    if rank == 0:
        if not exp_path.exists():
            exp_path.mkdir(parents=True)
            copy_source(__file__, exp_path)
            source_tree = Path(__file__).resolve().parent / 'score_sde'
            shutil.copytree(source_tree, exp_path / 'score_sde')

    logger.configure(dir=str(exp_path))
    logger.info('device: {}'.format(device))

    batch_size = args.batch_size
    if args.dataset == 'acdc':
        if int(args.num_channels) != 3:
            raise ValueError('ACDC is multi-class and must use --num_channels 3 for RV/myocardium/LV foreground mask channels')
        if int(args.num_channels_disc) != 6:
            raise ValueError('ACDC must use --num_channels_disc 6 because the discriminator receives concat(x_t, x_t+1) for 3 mask channels')

    if args.dataset in ('btcv', 'isic2018', 'isic18', 'isic'):
        if int(args.num_channels) != 1:
            raise ValueError(f'{args.dataset} is binary segmentation and must use --num_channels 1 for one foreground mask channel')
        if int(args.num_channels_disc) != 2:
            raise ValueError(f'{args.dataset} must use --num_channels_disc 2 because the discriminator receives concat(x_t, x_t+1) for one mask channel')
    nz = args.nz  # latent dimension

    train_data = create_dataset(
        data_dir=args.data_dir,
        mode=args.train_split,
        image_size=args.image_size,
        dataset_name=args.dataset,
        fold=args.fold,
        manifest_path=args.train_manifest,
    )
    val_data = create_dataset(
        data_dir=args.data_dir,
        mode=args.val_split,
        image_size=args.image_size,
        dataset_name=args.dataset,
        fold=args.fold,
        manifest_path=args.val_manifest,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=args.num_workers,
        shuffle=args.shuffle_train,
        drop_last=True,
    )

    netG = NCSNpp(args).to(device)
    netD = Discriminator_large(args.num_channels_disc, ngf=args.ngf,
                               t_emb_dim=args.t_emb_dim,
                               act=nn.LeakyReLU(0.2)).to(device)

    if args.distributed:
        broadcast_params(netG.parameters())
        broadcast_params(netD.parameters())

    optimizerD = optim.Adam(netD.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))

    optimizerG = optim.Adam(netG.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))

    if args.use_ema:
        optimizerG = EMA(optimizerG, ema_decay=args.ema_decay)

    lr_decay_steps = int(args.lr_decay_steps) if int(args.lr_decay_steps) > 0 else int(args.max_train_steps)
    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, lr_decay_steps, eta_min=1e-5)
    schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, lr_decay_steps, eta_min=1e-5)

    if args.distributed:
        netG = nn.parallel.DistributedDataParallel(netG, device_ids=[gpu])
        netD = CustomDDPWrapper(netD, device_ids=[gpu])

    netD.apply(weights_init_normal)

    audit_mode = normalize_audit_mode(args)
    if rank == 0:
        write_training_metadata(exp_path, args, netG, netD)
    logger.info('audit_mode: {}'.format(audit_mode))
    if audit_mode == "train_random_yt":
        logger.info('train_random_yt: generator noisy-mask inputs are independent torch.randn_like tensors; target mask and MSE loss are unchanged.')
    elif audit_mode == "train_shuffle_yt":
        if batch_size <= 1:
            raise ValueError('train_shuffle_yt requires batch_size > 1 unless dataset-level shuffle is implemented')
        logger.info('train_shuffle_yt: generator noisy-mask inputs are rebuilt from another batch mask with the current timestep and original forward noise; target mask and MSE loss are unchanged.')
    elif audit_mode == "core_no_diff":
        logger.info('core_no_diff: training uses image condition plus non-diffusion latent z, clean-mask MSE, and no Y_t, timestep, discriminator attention, or posterior sampler in the main path.')
    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)
    T = get_time_schedule(args, device)

    if args.resume:
        checkpoint_file = os.path.join(exp_path, 'content.pth')
        checkpoint = torch.load(checkpoint_file, map_location=device)
        netG.load_state_dict(checkpoint['netG_dict'])
        optimizerG.load_state_dict(checkpoint['optimizerG'])
        schedulerG.load_state_dict(checkpoint['schedulerG'])
        netD.load_state_dict(checkpoint['netD_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD'])
        schedulerD.load_state_dict(checkpoint['schedulerD'])
        global_step = int(checkpoint.get('global_step', checkpoint.get('train_step', 0)))
        print("=> loaded checkpoint (step {})".format(global_step))
    else:
        global_step = 0

    # Beginning of step-based training.
    terms = dict()
    best_mIoU, best_dice, errD_total, mse_total = 0, -1.0, 0, 0
    max_train_steps = int(args.max_train_steps)
    val_interval_steps = int(args.val_interval_steps)
    if max_train_steps <= 0:
        raise ValueError('max_train_steps must be positive')
    if val_interval_steps <= 0:
        raise ValueError('val_interval_steps must be positive')
    train_iter = iter(train_loader)

    while global_step < max_train_steps:
        try:
            pairs = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            pairs = next(train_iter)

        x = pairs[0]  # This is 'x' (label)
        cond_img = pairs[1]["conditioned_image"].to(device, non_blocking=True)  # This is 'I' (image)
        real_data = x.to(device, non_blocking=True)

        if audit_mode == "core_no_diff":
            for p in netD.parameters():
                p.requires_grad = False
            netG.zero_grad()
            latent_z = torch.randn(real_data.size(0), nz, device=device)
            x_0_predict = forward_core_no_diff(netG, cond_img, latent_z)
            mse = mean_flat((real_data - x_0_predict) ** 2)
            terms["mse"] = mse
            terms["errD_real"] = torch.zeros_like(mse)
            terms["errD_fake"] = torch.zeros_like(mse)
            errD = torch.tensor(0.0, device=device)
            mse.mean().backward()
            optimizerG.step()
            global_step += 1
        else:
            for p in netD.parameters():
                p.requires_grad = True

            netD.zero_grad()

            # sample from p(x_0)
            real_data = x.to(device, non_blocking=True)

            # sample t
            t = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)

            x_t, x_tp1, x_t_noise, x_tp1_noise = q_sample_pairs(coeff, real_data, t, return_noise=True)
            x_t.requires_grad = True

            # I- train Discriminator with real
            D_real = netD(x_t, t, x_tp1.detach()).view(-1)

            errD_real = F.softplus(-D_real)
            terms["errD_real"] = errD_real
            errD_real = errD_real.mean()

            errD_real.backward(retain_graph=True)

            if args.lazy_reg is None:
                grad_real = torch.autograd.grad(
                    outputs=D_real.sum(), inputs=x_t, create_graph=True
                )[0]
                grad_penalty = (
                        grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
                ).mean()

                grad_penalty = args.r1_gamma / 2 * grad_penalty
                grad_penalty.backward()
            else:
                if global_step % args.lazy_reg == 0:
                    grad_real = torch.autograd.grad(
                        outputs=D_real.sum(), inputs=x_t, create_graph=True
                    )[0]
                    grad_penalty = (
                            grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
                    ).mean()

                    grad_penalty = args.r1_gamma / 2 * grad_penalty
                    grad_penalty.backward()

            # II- train Discriminator with fake
            latent_z = torch.randn(batch_size, nz, device=device)

            if audit_mode == "train_shuffle_yt":
                shuffled_real_data, shuffle_perm = shuffle_batch_tensor(real_data)
                x_tp1_input = construct_training_yt_from_y0(
                    x_tp1,
                    shuffled_real_data,
                    coeff,
                    t,
                    q_sample_pairs,
                    x_t_noise,
                    x_tp1_noise,
                )
                if global_step == 0 and rank == 0:
                    logger.info("train_shuffle_yt D-fake generator input permutation: {}".format(
                        shuffle_perm.detach().cpu().tolist()
                    ))
            else:
                x_tp1_input = maybe_replace_training_yt(x_tp1, args)
            x_0_predict = netG(x_tp1_input.detach(), t, cond_img, latent_z)
            x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)

            output = netD(x_pos_sample, t, x_tp1.detach()).view(-1)

            errD_fake = F.softplus(output)
            terms["errD_fake"] = errD_fake
            errD_fake = errD_fake.mean()
            errD_fake.backward()

            errD = errD_real + errD_fake

            # Update D
            optimizerD.step()

            # III- train Generator
            for p in netD.parameters():
                p.requires_grad = False
            netG.zero_grad()

            t = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)

            x_t, x_tp1, x_t_noise, x_tp1_noise = q_sample_pairs(coeff, real_data, t, return_noise=True)

            latent_z = torch.randn(batch_size, nz, device=device)
            # get attention
            hs = netD.get_features(x_t, t, x_tp1.detach())
            attn = F.interpolate(hs[args.attn_scale].mean(dim=1).unsqueeze(1), real_data.size()[-1], mode='bilinear')
            att_real_data = attn * real_data

            _, x_tp1_attn, att_noise_t, att_noise_tp1 = q_sample_pairs(coeff, att_real_data, t, return_noise=True)
            if audit_mode == "train_shuffle_yt":
                shuffled_real_data, shuffle_perm = shuffle_batch_tensor(real_data)
                att_shuffled_real_data = attn * shuffled_real_data
                x_tp1_attn_input = construct_training_yt_from_y0(
                    x_tp1_attn,
                    att_shuffled_real_data,
                    coeff,
                    t,
                    q_sample_pairs,
                    att_noise_t,
                    att_noise_tp1,
                )
                if global_step == 0 and rank == 0:
                    logger.info("train_shuffle_yt G-step generator input permutation: {}".format(
                        shuffle_perm.detach().cpu().tolist()
                    ))
            else:
                x_tp1_attn_input = maybe_replace_training_yt(x_tp1_attn, args)
            x_0_predict = netG(x_tp1_attn_input.detach(), t, cond_img, latent_z)

            mse = mean_flat((real_data - x_0_predict) ** 2)

            terms["mse"] = mse

            mse.mean().backward()
            optimizerG.step()
            global_step += 1

        if global_step == 1:
            torch.cuda.synchronize(device)
            update_gpu_evidence(
                args,
                dataset=args.dataset,
                audit_mode=audit_mode,
                audit_branch_executed=audit_mode,
                gpu_name=torch.cuda.get_device_name(device),
                cuda_device=str(device),
                model_parameter_device=str(next(unwrap_module(netG).parameters()).device),
                training_input_device=str(cond_img.device),
                training_target_device=str(real_data.device),
                training_output_device=str(x_0_predict.device),
                loss_device=str(mse.device),
                forward_completed=True,
                backward_completed=True,
                optimizer_step_completed=True,
                training_manifest=str(args.train_manifest),
                peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
            )

        if not args.no_lr_decay:
            schedulerG.step()
            if audit_mode != "core_no_diff":
                schedulerD.step()

        mse_total += mse
        errD_total += errD.item()

        if global_step % args.log_step == 0:
            terms["total_errD"] = errD_total / global_step
            terms["total_mse"] = mse_total / global_step
            logger.log_loss_dict(terms)
            logger.log_loss_dict(global_step)
            logger.dumpkvs()
            logger.log('step: {}'.format(global_step))

        should_validate = global_step % val_interval_steps == 0 or global_step == max_train_steps
        if rank == 0 and should_validate:
            sample_path = os.path.join(exp_path, 'validation_samples_step_{:06d}'.format(global_step))
            if not os.path.exists(sample_path):
                os.makedirs(sample_path)
            val_csv = args.val_output_csv or os.path.join(exp_path, 'validation_per_slice.csv')
            val_summary_json = args.val_summary_json or os.path.join(exp_path, 'validation_latest_summary.json')
            val_history_csv = args.val_history_csv or os.path.join(exp_path, 'validation_history.csv')
            mIoU, f1 = sampling_major_vote_func(
                pos_coeff,
                sample_from_model,
                netG,
                sample_path,
                val_data,
                logger,
                global_step,
                args,
                device,
                metrics_csv=val_csv,
                summary_json=val_summary_json,
            )
            became_best = f1 > best_dice
            checkpoint_status = 'not_saved_nonbest'
            checkpoint_path = ''
            metadata_path = ''
            if became_best:
                best_mIoU = mIoU
                best_dice = f1

                if args.use_ema:
                    optimizerG.swap_parameters_with_ema(store_params_in_ema=True)

                if args.save_best_only:
                    best_path, meta_path = save_best_checkpoint(netG, exp_path, global_step, mIoU, f1, args, val_csv)
                    checkpoint_path = str(best_path)
                    metadata_path = str(meta_path)
                    checkpoint_status = 'promoted_to_best'
                else:
                    checkpoint_path = os.path.join(exp_path, f'netG_step{global_step:06d}_mIoU_{mIoU}_F1_{f1}.pth')
                    torch.save(unwrap_module(netG).state_dict(), checkpoint_path)
                    best_path, meta_path = save_best_checkpoint(netG, exp_path, global_step, mIoU, f1, args, val_csv)
                    metadata_path = str(meta_path)
                    checkpoint_status = 'saved_named_and_promoted_to_best'
            validation_summary = json.loads(Path(val_summary_json).read_text(encoding='utf-8'))
            update_gpu_evidence(
                args,
                validation_forward_completed=True,
                validation_metric_computed=True,
                validation_dice=float(f1),
                validation_split=str(args.val_split),
                validation_manifest=str(args.val_manifest),
                validation_model_device=validation_summary.get('model_parameter_device', ''),
                validation_input_device=validation_summary.get('input_device', ''),
                validation_output_device=validation_summary.get('output_device', ''),
                best_checkpoint_saved=bool(became_best),
                best_checkpoint_name=args.best_checkpoint_name,
                best_checkpoint_metadata_name=args.best_metadata_name,
                peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
            )
            append_csv_row(
                val_history_csv,
                {
                    'train_step': global_step,
                    'dice': f1,
                    'miou': mIoU,
                    'became_best': int(became_best),
                    'checkpoint_status': checkpoint_status,
                    'checkpoint_path': checkpoint_path,
                    'best_checkpoint_path': str(Path(exp_path) / args.best_checkpoint_name),
                    'metadata_path': metadata_path,
                    'metric_source_csv': val_csv,
                    'validation_split': args.val_split,
                },
            )

        if args.save_content and global_step % int(args.save_content_every_steps) == 0:
            print('Saving content...')
            content = {'global_step': global_step, 'args': args,
                       'netG_dict': unwrap_module(netG).state_dict(), 'optimizerG': optimizerG.state_dict(),
                       'schedulerG': schedulerG.state_dict(), 'netD_dict': unwrap_module(netD).state_dict(),
                       'optimizerD': optimizerD.state_dict(), 'schedulerD': schedulerD.state_dict()}
            torch.save(content, os.path.join(exp_path, 'content.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ddgan parameters')
    defaults = dict(
        seed=47,
        resume=False,
        image_size=256,
        num_channels=1,
        num_channels_disc=2,
        centered=True,
        use_geometric=False,
        beta_min=0.1,
        beta_max=20.,

        cond_enc_layers=3,
        cond_enc_num_res_blocks=2,

        num_channels_dae=32,
        n_mlp=3,
        ch_mult=(1, 1, 2, 2, 4, 4),
        num_res_blocks=1,
        attn_resolutions=(16,),
        dropout=0.,
        resamp_with_conv=True,  # False in ddpm, True in biggan
        conditional=True,
        fir=True,
        fir_kernel=[1, 3, 3, 1],
        skip_rescale=True,
        resblock_type='biggan',  # choice in biggan and ddpm
        progressive='none',
        progressive_input='residual',
        progressive_combine='sum',
        attn_scale=2,

        embedding_type='positional',
        fourier_scale=16.,
        not_use_tanh=False,

        # geenrator and training
        exp='cDAL',
        dataset='btcv',
        fold=0,
        nz=100,
        num_timesteps=4,

        z_emb_dim=256,
        t_emb_dim=256,
        batch_size=4,
        max_train_steps=10000,
        val_interval_steps=1000,
        lr_decay_steps=10000,
        num_epoch=12000,  # legacy, ignored by step-based training
        T_max=500,  # legacy, ignored by step-based training
        ngf=64,

        lr_g=2e-4,
        lr_d=1e-5,
        beta1=0.5,
        beta2=0.9,
        no_lr_decay=False,

        use_ema=False,
        ema_decay=0.9999,
        ###ddp

        r1_gamma=1.,
        lazy_reg=None,

        data_dir='data_preprocessed/btcv_synapse_cdal_binary_png',
        train_split='train',
        val_split='validation',
        train_manifest='manifests/btcv/train_cases.txt',
        val_manifest='manifests/btcv/validation_cases.txt',
        test_manifest='manifests/btcv/test_cases.txt',
        output_dir='outputs',
        num_workers=0,
        shuffle_train=False,
        distributed=False,
        dist_backend='gloo',
        save_best_only=True,
        best_checkpoint_name='best_checkpoint.pt',
        best_metadata_name='best_checkpoint_meta.json',
        val_output_csv='',
        val_summary_json='',
        val_history_csv='',
        major_vote_number=5,
        eval_batch_size=4,
        eval_max_items=0,
        metric_skip_empty_gt=True,
        audit_mode='none',
        save_visuals=False,
        gpu_evidence_json='',

        save_content=False,  # to save all models and data
        save_content_every_steps=4000,
        save_content_every=2,  # legacy, ignored by step-based training
        save_ckpt_every=2,  # legacy, ignored by step-based training
        log_step=10,
        num_proc_node=1,
        num_process_per_node=1,
        node_rank=0,
        local_rank=0,
        master_address='127.0.0.1',
        master_port="6021",

    )

    defaults.update(exp=f"experiment_attn{defaults.get('attn_scale')}_{defaults.get('dataset')}_{defaults.get('exp')}_fold{defaults.get('fold')}")
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--dataset', default=defaults['dataset'])
    known_args, _ = pre_parser.parse_known_args()
    parameters_path = Path(__file__).resolve().parent / f"parameters_{known_args.dataset}.json"
    if parameters_path.exists():
        with parameters_path.open('r', encoding='utf-8') as f:
            loaded_defaults = json.load(f)
        defaults.update(loaded_defaults)
    else:
        raise FileNotFoundError(f"Missing dataset parameters: {parameters_path}")

    logger.add_dict_to_argparser(parser, defaults)
    args = parser.parse_args()
    if isinstance(args.lazy_reg, str):
        args.lazy_reg = None if args.lazy_reg.lower() in ('none', 'null', '') else int(args.lazy_reg)

    gpu = args.local_rank
    dev(args.local_rank)
    if args.distributed:
        os.environ['MASTER_ADDR'] = args.master_address
        os.environ['MASTER_PORT'] = args.master_port
        dist.init_process_group(backend=args.dist_backend, init_method='env://', rank=0, world_size=args.num_process_per_node)
    train(0, gpu, args)
    if args.distributed:
        dist.barrier()
        dist.destroy_process_group()
















