import copy
import functools
import json
import math
import os
import random
import time

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
from .visdom_util import make_visdom
viz = make_visdom(port=8850)
loss_window = viz.line( Y=th.zeros((1)).cpu(), X=th.zeros((1)).cpu(), opts=dict(xlabel='epoch', ylabel='Loss', title='loss'))
grad_window = viz.line(Y=th.zeros((1)).cpu(), X=th.zeros((1)).cpu(),
                           opts=dict(xlabel='step', ylabel='amplitude', title='gradient'))


# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        classifier,
        diffusion,
        data,
        dataloader,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        validation_fn=None,
        validation_interval=0,
        audit_mode="none",
        seed=-1,
        validation_seed=10,
        validation_manifest="",
    ):
        self.model = model
        self.dataloader=dataloader
        self.classifier = classifier
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.validation_fn = validation_fn
        self.validation_interval = validation_interval
        self.audit_mode = audit_mode
        self.seed = seed
        self.validation_seed = validation_seed
        self.validation_manifest = validation_manifest
        self.best_validation_dice = float("-inf")
        self.best_validation_step = None

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available() and dist.get_world_size() > 1:
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

        self._load_best_validation_state()

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            print('resume model')
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        checkpoint_dir = bf.dirname(main_checkpoint)
        candidates = [
            bf.join(checkpoint_dir, f"opt{self.resume_step:06}.pt"),
            bf.join(checkpoint_dir, f"optsavedmodel{self.resume_step:06}.pt"),
        ]
        for opt_checkpoint in candidates:
            if bf.exists(opt_checkpoint):
                logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
                try:
                    state_dict = dist_util.load_state_dict(
                        opt_checkpoint,
                        map_location=dist_util.dev(),
                        weights_only=False,
                    )
                    self.opt.load_state_dict(state_dict)
                except Exception as exc:
                    logger.log(
                        "failed to load optimizer state; continuing with a fresh "
                        f"optimizer: {exc}"
                    )
                else:
                    logger.log(f"loaded optimizer state from checkpoint: {opt_checkpoint}")
                return
        logger.log(f"optimizer state not found for resume step {self.resume_step}")

    def run_loop(self):
        training_start_time = time.time()
        i = 0
        data_iter = iter(self.dataloader)
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):


            try:
                    batch, cond = next(data_iter)
            except StopIteration:
                    # StopIteration is thrown if dataset ends
                    # reinitialize data loader
                    data_iter = iter(self.dataloader)
                    batch, cond = next(data_iter)

            self.run_step(batch, cond)

           
            i += 1
          
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            current_step = self.step + self.resume_step
            skip_resume_boundary = self.resume_step > 0 and self.step == 0
            if current_step % self.save_interval == 0 and not skip_resume_boundary:
                self.save()
                self._maybe_validate()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    self._log_training_time(training_start_time)
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()
            self._maybe_validate()
        self._log_training_time(training_start_time)

    def _log_training_time(self, training_start_time):
        logger.log(
            f"training_time_sec = {round(time.time() - training_start_time, 3)}"
        )

    def _maybe_validate(self):
        current_step = self.step + self.resume_step
        if (
            self.validation_fn is None
            or not self.validation_interval
            or current_step == 0
            or current_step % self.validation_interval != 0
        ):
            return
        if dist.get_rank() == 0:
            rng_state = self._capture_rng_state()
            try:
                self._seed_validation_rng()
                row = self.validation_fn(current_step)
            finally:
                self._restore_rng_state(rng_state)
            self._save_best_checkpoint(row, current_step)
        if dist.is_initialized():
            dist.barrier()

    def _capture_rng_state(self):
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": th.random.get_rng_state(),
            "cuda": th.cuda.get_rng_state_all() if th.cuda.is_available() else None,
        }

    def _restore_rng_state(self, state):
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        th.random.set_rng_state(state["torch"])
        if state["cuda"] is not None:
            th.cuda.set_rng_state_all(state["cuda"])

    def _seed_validation_rng(self):
        if self.validation_seed < 0:
            return
        random.seed(self.validation_seed)
        np.random.seed(self.validation_seed)
        th.manual_seed(self.validation_seed)
        if th.cuda.is_available():
            th.cuda.manual_seed_all(self.validation_seed)

    def _best_metadata_path(self):
        return os.path.join(get_blob_logdir(), "best_checkpoint.json")

    def _load_best_validation_state(self):
        if dist.get_rank() != 0:
            return
        path = self._best_metadata_path()
        if not os.path.isfile(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            self.best_validation_dice = float(metadata["metric_value"])
            self.best_validation_step = int(metadata["step"])
            logger.log(
                "restored best validation state: "
                f"step={self.best_validation_step} dice={self.best_validation_dice:.6f}"
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.log(f"could not restore best validation metadata: {exc}")

    def _save_best_checkpoint(self, row, current_step):
        if not isinstance(row, dict) or "dice" not in row:
            raise ValueError("Validation callback must return a row containing 'dice'.")
        metric = float(row["dice"])
        if not math.isfinite(metric):
            raise ValueError(f"Validation Dice must be finite, got {metric!r}.")
        if metric <= self.best_validation_dice:
            return

        state_dict = self.mp_trainer.master_params_to_state_dict(
            self.mp_trainer.master_params
        )
        best_path = bf.join(get_blob_logdir(), "best_model.pt")
        with bf.BlobFile(best_path, "wb") as handle:
            th.save(state_dict, handle)

        metadata = {
            "checkpoint": "best_model.pt",
            "source_checkpoint": f"savedmodel{current_step:06d}.pt",
            "step": current_step,
            "metric_name": "dice",
            "metric_value": metric,
            "selection_partition": "validation",
            "validation_manifest": self.validation_manifest,
            "validation_seed": self.validation_seed,
            "training_seed": self.seed,
            "audit_mode": self.audit_mode,
        }
        with open(self._best_metadata_path(), "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
        self.best_validation_dice = metric
        self.best_validation_step = current_step
        logger.log(
            f"new best validation checkpoint: step={current_step} "
            f"dice={metric:.6f} path={best_path}"
        )

    def run_step(self, batch, cond):
        batch=th.cat((batch, cond), dim=1)

        cond={}
        sample = self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()
        return sample

    def forward_backward(self, batch, cond):

        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }

            last_batch = (i + self.microbatch) >= batch.shape[0]
            if self.audit_mode == "core_no_diff":
                t = th.zeros(micro.shape[0], device=dist_util.dev(), dtype=th.long)
                weights = th.ones(micro.shape[0], device=dist_util.dev())
            else:
                t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses_segmentation,
                self.ddp_model,
                self.classifier,
                micro,
                t,
                model_kwargs=micro_cond,
                audit_mode=self.audit_mode,
            )

            if last_batch or not self.use_ddp:
                losses1 = compute_losses()

            else:
                with self.ddp_model.no_sync():
                    losses1 = compute_losses()

            losses = losses1[0]
            sample = losses1[1]
            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()

            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)
        return sample

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"savedmodel{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"emasavedmodel_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"optsavedmodel{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    checkpoint_dir = bf.dirname(main_checkpoint)
    candidates = [
        bf.join(checkpoint_dir, f"ema_{rate}_{(step):06d}.pt"),
        bf.join(checkpoint_dir, f"emasavedmodel_{rate}_{(step):06d}.pt"),
    ]
    for path in candidates:
        if bf.exists(path):
            return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
