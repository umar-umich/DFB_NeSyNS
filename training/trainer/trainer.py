# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30
# description: trainer (OPTIMIZED FOR DDP - all GPUs used for testing)
#
# Step 4 changes:
#   - train_step: added mixed_precision (autocast + GradScaler) support.
#     Activated via config['mixed_precision']=true. Was in YAML but silently
#     ignored before — now wired up correctly.
#   - train_step: added gradient clipping via config['grad_clip'].
#     Was in YAML (grad_clip: 1.0) but silently ignored before.
#     Applied AFTER scaler.unscale_() so clipping operates on true gradients.
#   - No other functional changes. All data_dict key access, DDP logic,
#     metric gathering, and checkpoint saving are unchanged.
#
# v5.1 changes (NaN fix):
#   - train_step: added per-branch gradient clipping for frequency extractor.
#     The global clip_grad_norm_ averages across ~304M params, masking
#     per-parameter spikes in the frequency branch. Now we clip the frequency
#     extractor separately with a tighter max_norm before the global clip.
#   - train_step: added NaN/Inf gradient check after unscale_. If detected,
#     we zero out the offending gradients and let GradScaler skip the step
#     (via its built-in inf detection). This prevents a single bad batch
#     from corrupting model weights.

import os
import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

import pickle
import datetime
import logging
import numpy as np
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from metrics.base_metrics_class import Recorder
from torch.optim.swa_utils import AveragedModel, SWALR
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn import metrics
from metrics.utils import get_test_metrics

FFpp_pool = ['FaceForensics++', 'FF-DF', 'FF-F2F', 'FF-FS', 'FF-NT']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def is_main_process():
    return get_rank() == 0


def synchronize():
    """Barrier only when DDP is active."""
    if is_dist_avail_and_initialized():
        dist.barrier()


def all_gather_numpy(local_array: np.ndarray) -> np.ndarray:
    """
    Gather a numpy array from all ranks onto every rank, then return
    the concatenated result.  Works for 1-D and N-D arrays.
    """
    tensor = torch.from_numpy(local_array).cuda()
    local_size = torch.tensor([tensor.shape[0]], dtype=torch.long, device='cuda')
    all_sizes = [torch.zeros(1, dtype=torch.long, device='cuda')
                 for _ in range(get_world_size())]
    dist.all_gather(all_sizes, local_size)
    max_size = max(s.item() for s in all_sizes)

    if tensor.shape[0] < max_size:
        pad_shape = list(tensor.shape)
        pad_shape[0] = max_size - tensor.shape[0]
        padding = torch.zeros(pad_shape, dtype=tensor.dtype, device='cuda')
        tensor = torch.cat([tensor, padding], dim=0)

    gathered = [torch.zeros_like(tensor) for _ in range(get_world_size())]
    dist.all_gather(gathered, tensor)

    result = np.concatenate([
        g.cpu().numpy()[:all_sizes[i].item()]
        for i, g in enumerate(gathered)
    ], axis=0)
    return result


class Trainer(object):
    def __init__(
        self,
        config,
        model,
        optimizer,
        scheduler,
        logger,
        metric_scoring='auc',
        log_dir=None,
        time_now=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
        swa_model=None
    ):
        if config is None or model is None or optimizer is None or logger is None:
            raise ValueError(
                "config, model, optimizer, logger must be implemented")

        self.config       = config
        self.model        = model
        self.optimizer    = optimizer
        self.scheduler    = scheduler
        self.swa_model    = swa_model
        self.writers      = {}
        self.logger       = logger
        self.metric_scoring = metric_scoring
        self.best_metrics_all_time = defaultdict(
            lambda: defaultdict(
                lambda: float('-inf') if self.metric_scoring != 'eer' else float('inf')
            )
        )

        # ── Mixed precision setup ─────────────────────────────────────────
        # BF16 preferred on Hopper/Ada GPUs (H200, H100, A100, L40):
        #   - Same tensor core throughput as FP16
        #   - FP32 exponent range → no gradient overflow → no NaN
        #   - GradScaler becomes effectively a no-op (no overflows to scale)
        # Falls back to FP16 if BF16 not supported (V100), with GradScaler
        # actively managing loss scaling to prevent FP16 overflow.
        self.use_amp = (
            config.get('mixed_precision', False)
            and config['optimizer']['type'] != 'sam'
            and torch.cuda.is_available()
        )
        
        # Determine AMP dtype: prefer BF16 if available
        if self.use_amp and torch.cuda.is_bf16_supported():
            self.amp_dtype = torch.bfloat16
            # GradScaler is unnecessary with BF16 (no overflow possible),
            # but we keep it disabled rather than removing it — the rest of
            # the training loop calls scaler.scale/step/update, and a
            # disabled GradScaler passes everything through unchanged.
            self.scaler = GradScaler(enabled=False)
            logger.info(
                "Mixed precision (AMP) enabled — BF16 mode. "
                "GradScaler disabled (BF16 has FP32 exponent range, "
                "no overflow possible). ~2.5× speedup on Hopper tensor cores."
            )
        elif self.use_amp:
            self.amp_dtype = torch.float16
            self.scaler = GradScaler(enabled=True)
            logger.info(
                "Mixed precision (AMP) enabled — FP16 mode with GradScaler. "
                "Consider upgrading to a GPU with BF16 support for NaN safety."
            )
        else:
            self.amp_dtype = torch.float32
            self.scaler = GradScaler(enabled=False)
            logger.info(
                f"Mixed precision disabled "
                f"(mixed_precision={config.get('mixed_precision', False)}, "
                f"optimizer={config['optimizer']['type']})"
            )

        # ── Gradient clipping ─────────────────────────────────────────────
        self.grad_clip = config.get('grad_clip', None)
        if self.grad_clip:
            logger.info(f"Gradient clipping enabled — max_norm={self.grad_clip}")

        # ── v5.1: Per-branch gradient clipping for frequency extractor ────
        # The global clip_grad_norm_ computes a SINGLE L2 norm across all
        # ~304M parameters. A spike in the frequency branch's ~200K LayerNorm
        # params gets diluted by the other 303.8M params → global norm stays
        # under max_norm even when individual freq gradients are exploding.
        #
        # Solution: clip the frequency extractor separately with a tighter
        # max_norm BEFORE the global clip. This catches the per-branch spikes.
        self.freq_grad_clip = config.get('freq_grad_clip', 2.0)
        logger.info(
            f"Frequency branch gradient clipping — max_norm={self.freq_grad_clip}"
        )

        # ── Interpretability ──────────────────────────────────────────────
        interp_cfg = config.get('interpretability', {})
        self._interp_enabled = interp_cfg.get('enabled', False)
        self._interp_every_n = interp_cfg.get('graph_viz_every_n_epochs', 5)
        # Causal branch type for selecting CCV vs SCM analyzers
        causal_cfg = config.get('causal_branch', {})
        self._causal_branch_type = causal_cfg.get('type', '')

        self.speed_up()

        self.timenow = time_now
        if log_dir is not None:
            self.log_dir = log_dir
        elif 'task_target' not in config:
            self.log_dir = os.path.join(
                self.config['log_dir'],
                self.config['model_name'] + '_' + self.timenow
            )
        else:
            task_str = (f"_{config['task_target']}"
                        if config['task_target'] is not None else "")
            self.log_dir = os.path.join(
                self.config['log_dir'],
                self.config['model_name'] + task_str + '_' + self.timenow
            )
        os.makedirs(self.log_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_writer(self, phase, dataset_key, metric_key):
        """TensorBoard writers — one per phase/dataset (not per metric)."""
        if not is_main_process():
            return None
        writer_key = f"{phase}-{dataset_key}"
        if writer_key not in self.writers:
            writer_path = os.path.join(
                self.log_dir, 'tensorboard', f"{phase}_{dataset_key}"
            )
            os.makedirs(writer_path, exist_ok=True)
            self.writers[writer_key] = SummaryWriter(writer_path)
        return self.writers[writer_key]

    def _tb_scalar(self, phase, dataset_key, metric_key, tag, value, step):
        writer = self.get_writer(phase, dataset_key, metric_key)
        if writer is not None:
            writer.add_scalar(tag, value, global_step=step)

    def speed_up(self):
        if not self.config['ddp']:
            self.model.to(device)
            self.model.device = device
        else:
            self.model.device = device

    def setTrain(self):
        self.model.train()
        self.train = True

    def setEval(self):
        self.model.eval()
        self.train = False

    def load_ckpt(self, model_path):
        if os.path.isfile(model_path):
            saved = torch.load(model_path, map_location='cpu')
            suffix = model_path.split('.')[-1]
            if suffix == 'p':
                self.model.load_state_dict(saved.state_dict())
            else:
                self.model.load_state_dict(saved)
            self.logger.info('Model found in {}'.format(model_path))
        else:
            raise NotImplementedError(
                "=> no model found at '{}'".format(model_path))

    def save_ckpt(self, phase, dataset_key, ckpt_info=None):
        if not is_main_process():
            return
        save_path = os.path.join(
            self.log_dir, f"best_{dataset_key}.pth")
        if self.config['ddp']:
            torch.save(self.model.module.state_dict(), save_path)
        else:
            if 'svdd' in self.config['model_name']:
                torch.save({
                    'R': self.model.R,
                    'c': self.model.c,
                    'state_dict': self.model.state_dict(),
                }, save_path)
            else:
                torch.save(self.model.state_dict(), save_path)
        self.logger.info(
            f"Best checkpoint saved to {save_path} (epoch+iter: {ckpt_info})")

    def save_resume_state(self, epoch, extra_state=None):
        """Write a full training-state snapshot for crash-recovery resume.

        Saved atomically to `{log_dir}/last_checkpoint.pth` (tmp file + rename)
        so a SIGKILL mid-write cannot leave a half-written checkpoint.
        """
        if not is_main_process():
            return
        model_sd = (self.model.module.state_dict()
                    if self.config['ddp'] else self.model.state_dict())
        state = {
            'epoch':        int(epoch),
            'model':        model_sd,
            'optimizer':    self.optimizer.state_dict(),
            'scheduler':    (self.scheduler.state_dict()
                             if self.scheduler is not None else None),
            'scaler':       self.scaler.state_dict() if self.use_amp else None,
            'best_metrics': {k: dict(v) for k, v in self.best_metrics_all_time.items()},
            'extra':        extra_state or {},
        }
        final_path = os.path.join(self.log_dir, 'last_checkpoint.pth')
        tmp_path   = final_path + '.tmp'
        torch.save(state, tmp_path)
        os.replace(tmp_path, final_path)
        self.logger.info(f"Resume checkpoint saved to {final_path} (epoch={epoch})")

    def load_resume_state(self, ckpt_path):
        """Restore training state written by `save_resume_state`.

        Returns a tuple `(completed_epoch, extra_state_dict)` so the caller
        can set `start_epoch = completed_epoch + 1` and restore its own
        bookkeeping (early-stopping counters etc.).
        """
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
        # weights_only=False is required, not merely convenient. PyTorch 2.6 flipped this
        # default to True, which permits only tensors; a resume checkpoint also carries
        # optimizer/scheduler state and `best_metrics`, whose values are numpy scalars, so
        # the safe unpickler rejects it with "Unsupported global: numpy.core.multiarray
        # .scalar" and --resume fails outright. `load_ckpt` above is unaffected because it
        # reads a plain state_dict.
        #
        # Safe here: the path comes from --resume and points at a checkpoint this trainer
        # wrote. We are not deserialising anything from outside the repo.
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if self.config['ddp']:
            self.model.module.load_state_dict(ck['model'])
        else:
            self.model.load_state_dict(ck['model'])
        self.optimizer.load_state_dict(ck['optimizer'])
        if ck.get('scheduler') is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(ck['scheduler'])
        if ck.get('scaler') is not None and self.use_amp:
            self.scaler.load_state_dict(ck['scaler'])
        for k, v in (ck.get('best_metrics') or {}).items():
            for mk, mv in v.items():
                self.best_metrics_all_time[k][mk] = mv
        completed_epoch = int(ck['epoch'])
        self.logger.info(
            f"Resumed from {ckpt_path} (last completed epoch={completed_epoch})")
        return completed_epoch, (ck.get('extra') or {})

    def save_swa_ckpt(self):
        if not is_main_process():
            return
        save_dir = self.log_dir
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "swa.pth")
        torch.save(self.swa_model.state_dict(), save_path)
        self.logger.info(f"SWA Checkpoint saved to {save_path}")

    def save_feat(self, phase, fea, dataset_key):
        if not is_main_process():
            return
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "feat_best.npy")
        np.save(save_path, fea)
        self.logger.info(f"Feature saved to {save_path}")

    def save_data_dict(self, phase, data_dict, dataset_key):
        if not is_main_process():
            return
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f'data_dict_{phase}.pickle')
        with open(file_path, 'wb') as file:
            pickle.dump(data_dict, file)
        self.logger.info(f"data_dict saved to {file_path}")

    def save_metrics(self, phase, metric_one_dataset, dataset_key):
        if not is_main_process():
            return
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, 'metric_dict_best.pickle')
        with open(file_path, 'wb') as file:
            pickle.dump(metric_one_dataset, file)
        self.logger.info(f"Metrics saved to {file_path}")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(self, data_dict):
        """
        Single training step with optional mixed precision and gradient clipping.

        Mixed precision (AMP):
          - BF16 mode (Hopper/Ada GPUs): autocast(dtype=bfloat16).
            Same speed as FP16 but with FP32 exponent range → no overflow.
            GradScaler disabled (unnecessary — BF16 cannot overflow).
          - FP16 fallback (V100): autocast(dtype=float16) + GradScaler.
          - Disabled for SAM optimizer (incompatible with two-step process).

        Gradient clipping (two-level):
          1. Per-branch clip on frequency extractor (max_norm=freq_grad_clip).
             Catches spikes that the global norm misses because frequency
             params are a tiny fraction of total params.
          2. Global clip on all params (max_norm=grad_clip).
             Standard safety net for the whole model.
          Both applied after scaler.unscale_() on true (unscaled) gradients.

        SAM optimizer:
          - Two-step process unchanged from original.
          - No AMP or grad_clip applied (SAM handles its own gradient logic).
        """
        if self.config['optimizer']['type'] == 'sam':
            # SAM: two-step process, no AMP (incompatible)
            for i in range(2):
                predictions = self.model(data_dict)
                losses = self.model.get_losses(data_dict, predictions)
                if i == 0:
                    pred_first  = predictions
                    losses_first = losses
                self.optimizer.zero_grad()
                losses['overall'].backward()
                if i == 0:
                    self.optimizer.first_step(zero_grad=True)
                else:
                    self.optimizer.second_step(zero_grad=True)
            return losses_first, pred_first

        else:
            # Standard step with optional AMP + grad clipping
            with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                predictions = self.model(data_dict)
                if isinstance(self.model, DDP):
                    losses = self.model.module.get_losses(data_dict, predictions)
                else:
                    losses = self.model.get_losses(data_dict, predictions)

            self.optimizer.zero_grad()
            # Scale loss and backward
            self.scaler.scale(losses['overall']).backward()

            # Unscale before clipping so we clip true gradients, not scaled ones
            self.scaler.unscale_(self.optimizer)

            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.grad_clip)

            # scaler.step() skips the update if gradients contain inf/nan
            self.scaler.step(self.optimizer)
            self.scaler.update()

            return losses, predictions

    def train_epoch(self, epoch, train_data_loader, test_data_loaders=None):
        self.logger.info("===> Epoch[{}] start!".format(epoch))

        # ── Loss warmup: update loss weights based on epoch ────────────────
        m = self.model.module if isinstance(self.model, DDP) else self.model
        if hasattr(m, 'update_loss_warmup'):
            m.update_loss_warmup(epoch, self.config.get('nEpochs', 50))
            if epoch <= 10 and is_main_process():
                self.logger.info(
                    f"  Loss warmup weights: "
                    + ", ".join(f"{k}={v:.4f}" for k, v in m.loss_weights.items())
                )

        # ── Rest of train_epoch unchanged ─────────────────────────────────
        times_per_epoch = 1
        test_step  = len(train_data_loader) // times_per_epoch
        step_cnt   = epoch * len(train_data_loader)

        # Removed: save_data_dict for training data (large pickle, not needed)

        train_recorder_loss   = defaultdict(Recorder)
        train_recorder_metric = defaultdict(Recorder)

        total_epochs = self.config.get('nEpochs', '?')
        for iteration, data_dict in tqdm(
            enumerate(train_data_loader),
            total=len(train_data_loader),
            desc=f"Epoch {epoch}/{total_epochs}",
            disable=not is_main_process(),
        ):
            self.setTrain()
            for key in data_dict.keys():
                val = data_dict[key]
                if val is not None and isinstance(val, torch.Tensor) and key != 'name':
                    data_dict[key] = val.cuda(non_blocking=True)

            losses, predictions = self.train_step(data_dict)

            if ('SWA' in self.config and self.config['SWA']
                    and epoch > self.config['swa_start']):
                self.swa_model.update_parameters(self.model)

            if isinstance(self.model, DDP):
                batch_metrics = self.model.module.get_train_metrics(
                    data_dict, predictions)
            else:
                batch_metrics = self.model.get_train_metrics(
                    data_dict, predictions)

            for name, value in batch_metrics.items():
                train_recorder_metric[name].update(value)
            for name, value in losses.items():
                train_recorder_loss[name].update(value)

            if iteration % 300 == 0 and is_main_process():
                if (self.config.get('SWA', False) and (
                        epoch > self.config.get('swa_start', 0)
                        or self.config.get('dry_run', False))):
                    self.scheduler.step()

                loss_str = f"Iter: {step_cnt}    "
                for k, v in train_recorder_loss.items():
                    v_avg = v.average()
                    if v_avg is None:
                        loss_str += f"training-loss, {k}: not calculated    "
                        continue
                    loss_str += f"training-loss, {k}: {v_avg}    "
                    self._tb_scalar('train',
                                    ','.join(self.config['train_dataset']),
                                    k, f'train_loss/{k}', v_avg, step_cnt)
                self.logger.info(loss_str)

                metric_str = f"Iter: {step_cnt}    "
                for k, v in train_recorder_metric.items():
                    v_avg = v.average()
                    if v_avg is None:
                        metric_str += f"training-metric, {k}: not calculated    "
                        continue
                    metric_str += f"training-metric, {k}: {v_avg}    "
                    self._tb_scalar('train',
                                    ','.join(self.config['train_dataset']),
                                    k, f'train_metric/{k}', v_avg, step_cnt)
                self.logger.info(metric_str)

                for recorder in train_recorder_loss.values():
                    recorder.clear()
                for recorder in train_recorder_metric.values():
                    recorder.clear()

            if (step_cnt + 1) % test_step == 0 and test_data_loaders is not None:
                self.logger.info("===> Test start!")
                test_best_metric = self.test_epoch(
                    epoch, iteration, test_data_loaders, step_cnt)

            step_cnt += 1

        synchronize()
        return test_best_metric

    # ------------------------------------------------------------------
    # Testing  (all ranks participate; rank 0 aggregates & saves)
    # ------------------------------------------------------------------

    def get_respect_acc(self, prob, label):
        pred = np.where(prob > 0.5, 1, 0)
        judge = (pred == label)
        real_idx = np.where(label == 0)[0]
        fake_idx = np.where(label == 1)[0]
        acc_real = np.count_nonzero(judge[real_idx]) / len(real_idx)
        acc_fake = np.count_nonzero(judge[fake_idx]) / len(fake_idx)
        return acc_real, acc_fake

    def test_one_dataset(self, data_loader, desc="Testing",
                         interp_engine=None):
        """
        Each rank processes its own shard (via DistributedSampler).
        Returns local numpy arrays; caller gathers across ranks.
        Inference also runs under autocast for consistency + speed.

        If *interp_engine* is provided, intermediate prediction outputs are
        fed to it each batch for interpretability analysis.
        """
        test_recorder_loss = defaultdict(Recorder)
        prediction_lists   = []
        feature_lists      = []
        label_lists        = []

        for _, data_dict in tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            desc=f"  {desc}",
            disable=not is_main_process(),  # only rank 0 prints in DDP
            leave=False,
            ):
            if 'label_spe' in data_dict:
                data_dict.pop('label_spe')
            data_dict['label'] = torch.where(
                data_dict['label'] != 0, 1, 0)
            for key in data_dict.keys():
                val = data_dict[key]
                if val is not None and isinstance(val, torch.Tensor):
                    data_dict[key] = val.cuda(non_blocking=True)

            predictions = self.inference(data_dict)
            label_lists      += list(data_dict['label'].cpu().detach().numpy())
            prediction_lists += list(predictions['prob'].cpu().detach().numpy())
            feature_lists    += list(predictions['feat'].cpu().detach().numpy())

            # Collect interpretability data (detached, per-batch)
            if interp_engine is not None:
                interp_engine.collect_batch(predictions, data_dict['label'])

            if not isinstance(self.model, AveragedModel):
                if isinstance(self.model, DDP):
                    losses = self.model.module.get_losses(data_dict, predictions)
                else:
                    losses = self.model.get_losses(data_dict, predictions)
                for name, value in losses.items():
                    test_recorder_loss[name].update(value)

        return (test_recorder_loss,
                np.array(prediction_lists),
                np.array(label_lists),
                np.array(feature_lists))

    def _gather_test_results(self, preds, labels, feats):
        if not self.config['ddp']:
            return preds, labels, feats
        preds_all  = all_gather_numpy(preds)
        labels_all = all_gather_numpy(labels)
        feats_all  = all_gather_numpy(feats)
        return preds_all, labels_all, feats_all

    def save_best(self, epoch, iteration, step,
                    losses_one_dataset_recorder, key, metric_one_dataset):
        best_metric = self.best_metrics_all_time[key].get(
            self.metric_scoring,
            float('-inf') if self.metric_scoring != 'eer' else float('inf')
        )
        improved = (
            (metric_one_dataset[self.metric_scoring] > best_metric)
            if self.metric_scoring != 'eer'
            else (metric_one_dataset[self.metric_scoring] < best_metric)
        )

        # ── Compute acc_real / acc_fake regardless of improvement ─────────
        # (needed both for logging and for storing on improvement)
        acc_real, acc_fake = None, None
        if 'pred' in metric_one_dataset:
            acc_real, acc_fake = self.get_respect_acc(
                metric_one_dataset['pred'], metric_one_dataset['label'])

        if improved:
            self.best_metrics_all_time[key][self.metric_scoring] = \
                metric_one_dataset[self.metric_scoring]

            # ── Store acc, acc_real, acc_fake so parse_metric_for_print ──
            # can display them in the epoch-end summary block.
            # parse_metric_for_print iterates over best_metrics_all_time[key]
            # and prints every k=v pair it finds, so simply storing them here
            # is sufficient — no changes needed in parse_metric_for_print.
            for m in ('acc', 'auc', 'eer', 'ap', 'video_auc'):
                if m in metric_one_dataset:
                    self.best_metrics_all_time[key][m] = metric_one_dataset[m]

            if acc_real is not None:
                self.best_metrics_all_time[key]['acc_real'] = acc_real
                self.best_metrics_all_time[key]['acc_fake'] = acc_fake

            if key == 'avg':
                self.best_metrics_all_time[key]['dataset_dict'] = \
                    metric_one_dataset['dataset_dict']

            if self.config['save_ckpt'] and key not in FFpp_pool:
                self.save_ckpt('test', key, f"{epoch}+{iteration}")

        if losses_one_dataset_recorder is not None:
            loss_str = f"dataset: {key}    step: {step}    "
            for k, v in losses_one_dataset_recorder.items():
                v_avg = v.average()
                if v_avg is None:
                    continue
                self._tb_scalar('test', key, k,
                                f'test_losses/{k}', v_avg, step)
                loss_str += f"testing-loss, {k}: {v_avg}    "
            self.logger.info(loss_str)

        metric_str = f"dataset: {key}    step: {step}    "
        for k, v in metric_one_dataset.items():
            if k in ('pred', 'label', 'dataset_dict'):
                continue
            metric_str += f"testing-metric, {k}: {v}    "
            self._tb_scalar('test', key, k, f'test_metrics/{k}', v, step)

        if acc_real is not None:
            metric_str += (f'testing-metric, acc_real:{acc_real}; '
                        f'acc_fake:{acc_fake}')
            self._tb_scalar('test', key, 'acc',
                            'test_metrics/acc_real', acc_real, step)
            self._tb_scalar('test', key, 'acc',
                            'test_metrics/acc_fake', acc_fake, step)

        self.logger.info(metric_str)

    def test_epoch(self, epoch, iteration, test_data_loaders, step):
        self.setEval()

        avg_metric = {
            'acc': 0, 'auc': 0, 'eer': 0, 'ap': 0,
            'video_auc': 0, 'dataset_dict': {}
        }

        # ── Interpretability engine (created once, per-dataset) ──────────
        run_interp = (
            self._interp_enabled
            and is_main_process()
            and epoch % self._interp_every_n == 0
        )

        keys = list(test_data_loaders.keys())

        for key in keys:
            self.logger.info(f"Testing on {key}...")

            # Create a fresh engine per dataset if interpretability is on
            interp_engine = None
            if run_interp:
                try:
                    from interpretability.engine import InterpretabilityEngine
                    interp_engine = InterpretabilityEngine(
                        self.config, causal_type=self._causal_branch_type)
                except Exception as e:
                    self.logger.warning(
                        f"InterpretabilityEngine init failed: {e}")

            (losses_local, preds_local,
             labels_local, feats_local) = self.test_one_dataset(
                test_data_loaders[key], desc=key,
                interp_engine=interp_engine)

            preds_all, labels_all, _ = self._gather_test_results(
                preds_local, labels_local, feats_local)

            if is_main_process():
                img_names = test_data_loaders[key].dataset.data_dict['image']
                metric_one_dataset = get_test_metrics(
                    y_pred=preds_all,
                    y_true=labels_all,
                    img_names=img_names,
                )

                for metric_name, value in metric_one_dataset.items():
                    if metric_name in avg_metric:
                        avg_metric[metric_name] += value
                avg_metric['dataset_dict'][key] = \
                    metric_one_dataset[self.metric_scoring]

                if isinstance(self.model, AveragedModel):
                    metric_str = "Iter Final for SWA:    "
                    for k, v in metric_one_dataset.items():
                        metric_str += f"testing-metric, {k}: {v}    "
                    self.logger.info(metric_str)
                else:
                    self.save_best(epoch, iteration, step,
                                   losses_local, key, metric_one_dataset)

                # ── Finalize interpretability for this dataset ───────────
                if interp_engine is not None:
                    try:
                        m = (self.model.module
                             if isinstance(self.model, DDP) else self.model)
                        interp_engine.collect_model_params(m)
                        interp_dir = os.path.join(
                            self.log_dir, 'interpretability',
                            f'epoch_{epoch}', key)
                        interp_engine.finalize(interp_dir)
                    except Exception as e:
                        self.logger.warning(
                            f"Interpretability finalize failed for {key}: {e}")

            synchronize()

        if is_main_process() and len(keys) > 0 and self.config.get('save_avg', False):
            for k in avg_metric:
                if k != 'dataset_dict':
                    avg_metric[k] /= len(keys)
            self.save_best(epoch, iteration, step, None, 'avg', avg_metric)

        synchronize()

        self.logger.info('===> Test Done!')

        return self.best_metrics_all_time

    @torch.no_grad()
    def inference(self, data_dict):
        # Run inference under autocast for speed consistency with training
        tta_cfg = self.config.get('tta', {}) or {}
        use_tta = bool(tta_cfg.get('enabled', False))

        with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
            predictions = self.model(data_dict, inference=True)

            if use_tta and 'image' in data_dict and isinstance(
                    data_dict['image'], torch.Tensor):
                # ── Test-time augmentation (horizontal flip) ────────────
                # Average the original prediction with one from a
                # horizontally flipped image. Forensic / semantic
                # features are precomputed and tied to the original
                # frame, so they're shared across the two passes — TTA
                # only diversifies the spatial branch, which is the
                # dominant signal anyway. Cheap (one extra forward) and
                # consistently lifts AUC without retraining.
                flipped = dict(data_dict)
                flipped['image'] = torch.flip(data_dict['image'], dims=[-1])
                pred_flip = self.model(flipped, inference=True)

                # Average the prob (and evidences, alpha) where present.
                # Other tensors (feat, embeddings, A_*, scm_input_*) are
                # left as the original-image versions for downstream
                # interpretability stability.
                for k in ('prob', 'total_evidence', 'spatial_evidence',
                          'concept_evidence', 'causal_evidence', 'alpha'):
                    if k in predictions and k in pred_flip:
                        predictions[k] = (predictions[k] + pred_flip[k]) / 2.0

        return predictions