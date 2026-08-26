import copy
import functools
import json
import math
import os
from datetime import datetime

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
# from visdom import Visdom
# viz = Visdom(port=8850)
# loss_window = viz.line( Y=th.zeros((1)).cpu(), X=th.zeros((1)).cpu(), opts=dict(xlabel='epoch', ylabel='Loss', title='loss'))
# grad_window = viz.line(Y=th.zeros((1)).cpu(), X=th.zeros((1)).cpu(),
#                            opts=dict(xlabel='step', ylabel='amplitude', title='gradient'))


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
        validation_runner=None,
        save_best_only=False,
        best_metric_name="dice_mean",
        audit_mode="none",
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
        self.validation_runner = validation_runner
        self.save_best_only = save_best_only
        self.best_metric_name = best_metric_name
        self.audit_mode = str(audit_mode or "none").strip().lower()
        self.best_metric = None
        self.best_step = None
        if self.save_best_only and self.validation_runner is None:
            raise ValueError(
                "--save_best_only requires --validation_runner True and a "
                "validation Dice source"
            )

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
                find_unused_parameters=self.audit_mode == "core_no_diff",
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            print('resume model')
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_part_state_dict(
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
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"optsavedmodel{self.resume_step:06}.pt"
        )
        if not bf.exists(opt_checkpoint):
            opt_checkpoint = bf.join(
                bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
            )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        i = 0
        data_iter = iter(self.dataloader)
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):


            try:
                    batch, cond, name = next(data_iter)
            except StopIteration:
                    # StopIteration is thrown if dataset ends
                    # reinitialize data loader
                    data_iter = iter(self.dataloader)
                    batch, cond, name = next(data_iter)

            self.run_step(batch, cond)

           
            i += 1
          
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            validation_step = self.step + self.resume_step
            before_first_best_validation = (
                self.validation_runner is not None
                and self.save_best_only
                and validation_step < self.validation_runner.args.val_min_step
            )
            if self.step % self.save_interval == 0 and not before_first_best_validation:
                self.save()
                if self.validation_runner is not None:
                    checkpoint_name = f"savedmodel{validation_step:06d}.pt"
                    logger.log(
                        f"running full validation for {checkpoint_name}..."
                    )
                    if self.save_best_only:
                        checkpoint_path = bf.join(get_blob_logdir(), checkpoint_name)
                        row = self.validation_runner.evaluate_checkpoint_file(
                            self.model, self.diffusion, checkpoint_path
                        )
                    else:
                        row = self.validation_runner.evaluate_model(
                            self.model,
                            self.diffusion,
                            validation_step,
                            checkpoint_name,
                        )
                    if row is None:
                        logger.log(
                            f"validation skipped for {checkpoint_name}."
                        )
                    else:
                        logger.log(
                            "validation "
                            f"step={row['step']} "
                            f"dice_mean={row['dice_mean']:.4f} "
                            f"dice_nonempty_mean={row['dice_nonempty_mean']:.4f} "
                            f"iou_mean={row['iou_mean']:.4f}"
                        )
                        if self.save_best_only:
                            self._apply_best_checkpoint_policy(
                                row, validation_step
                            )
                    if row is None and self.save_best_only:
                        raise RuntimeError(
                            "save_best_only could not decide checkpoint "
                            "retention because validation was skipped"
                        )
                elif self.save_best_only:
                    raise RuntimeError(
                        "save_best_only could not decide checkpoint retention "
                        "because the validation runner is missing"
                    )
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            if self.save_best_only:
                logger.log("skipping unscheduled final checkpoint because save_best_only is enabled")
            else:
                self.save()

    def _metric_value(self, row):
        try:
            value = float(row[self.best_metric_name])
        except (KeyError, TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None
        return value

    def _checkpoint_paths_for_step(self, step, include_model=True, include_sidecars=True):
        logdir = get_blob_logdir()
        paths = []
        if include_model:
            paths.append(bf.join(logdir, f"savedmodel{step:06d}.pt"))
        if include_sidecars:
            for rate in self.ema_rate:
                paths.append(bf.join(logdir, f"emasavedmodel_{rate}_{step:06d}.pt"))
            paths.append(bf.join(logdir, f"optsavedmodel{step:06d}.pt"))
        return paths

    def _delete_checkpoint_files(self, step):
        for path in self._checkpoint_paths_for_step(step):
            if bf.exists(path):
                bf.remove(path)

    def _delete_checkpoint_sidecars(self, step):
        for path in self._checkpoint_paths_for_step(step, include_model=False, include_sidecars=True):
            if bf.exists(path):
                bf.remove(path)

    def _apply_best_checkpoint_policy(self, row, step):
        if dist.get_rank() != 0:
            return
        metric = self._metric_value(row)
        candidate = bf.join(get_blob_logdir(), f"savedmodel{step:06d}.pt")
        if metric is None:
            if self.best_metric is None:
                raise RuntimeError(
                    f"Invalid {self.best_metric_name} for first save_best_only checkpoint at step {step}; "
                    "refusing to delete the only checkpoint"
                )
            logger.log(f"checkpoint step {step} has invalid {self.best_metric_name}; deleting candidate")
            self._delete_checkpoint_files(step)
            self._mark_validation_status(step, False, "deleted_invalid_metric")
            return

        became_best = self.best_metric is None or metric > self.best_metric
        if became_best:
            if not bf.exists(candidate):
                raise RuntimeError(f"Missing candidate checkpoint for best promotion: {candidate}")
            previous_best_step = self.best_step
            self.best_metric = metric
            self.best_step = step
            if previous_best_step is not None and previous_best_step != step:
                self._delete_checkpoint_files(previous_best_step)
            self._delete_checkpoint_sidecars(step)
            metadata = {
                "best_step": step,
                "best_metric_name": self.best_metric_name,
                "best_metric": metric,
                "source_checkpoint": candidate,
                "best_checkpoint": candidate,
                "best_ema_checkpoints": {},
                "validation_row": row,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "retention_policy": "save_best_only_keep_scored_savedmodel_delete_older_best",
            }
            meta_path = bf.join(get_blob_logdir(), "best_checkpoint_meta.json")
            with open(meta_path, "w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2)
                handle.write("\n")
            logger.log(
                f"promoted step {step} to best checkpoint with "
                f"{self.best_metric_name}={metric:.6f}"
            )
            self._mark_validation_status(step, True, "promoted_to_best", candidate)
        else:
            logger.log(
                f"checkpoint step {step} {self.best_metric_name}={metric:.6f} "
                f"did not beat best step {self.best_step} value {self.best_metric:.6f}; deleting candidate"
            )
            self._delete_checkpoint_files(step)
            self._mark_validation_status(
                step,
                False,
                "deleted_not_best",
                bf.join(get_blob_logdir(), f"savedmodel{self.best_step:06d}.pt"),
            )

    def _mark_validation_status(
        self,
        step,
        became_best,
        checkpoint_status,
        stable_best_checkpoint=None,
    ):
        marker = getattr(
            self.validation_runner, "mark_checkpoint_status", None
        )
        if marker is not None:
            marker(
                step=step,
                became_best=became_best,
                checkpoint_status=checkpoint_status,
                stable_best_checkpoint=stable_best_checkpoint
                or bf.join(get_blob_logdir(), f"savedmodel{self.best_step:06d}.pt"),
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
            micro = batch[i : i + self.microbatch].to(dist_util.dev(), non_blocking=True)
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev(), non_blocking=True)
                for k, v in cond.items()
            }

            last_batch = (i + self.microbatch) >= batch.shape[0]
            if self.audit_mode == "core_no_diff":
                t = None
                weights = None
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

            if t is not None and isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses1[0]["loss"].detach()
                )
            losses = losses1[0]
            sample = losses1[1]

            if weights is None:
                loss = losses["loss"].mean()
                weighted_losses = losses
            else:
                loss = (losses["loss"] * weights).mean()
                weighted_losses = {k: v * weights for k, v in losses.items()}

            log_loss_dict(self.diffusion, t, weighted_losses)
            self.mp_trainer.backward(loss)
            if os.environ.get("MEDSEGDIFF_DEBUG_UNUSED_GRADS"):
                for name, param in self.ddp_model.named_parameters():
                    if param.grad is None:
                        print(name)
            return  sample

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
    for filename in (f"emasavedmodel_{rate}_{(step):06d}.pt", f"ema_{rate}_{(step):06d}.pt"):
        path = bf.join(bf.dirname(main_checkpoint), filename)
        if bf.exists(path):
            return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        if ts is None:
            continue
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
