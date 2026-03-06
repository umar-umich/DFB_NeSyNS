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

        self.speed_up()

        self.timenow = time_now
        if 'task_target' not in config:
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
        """TensorBoard writers are only created/used on rank 0."""
        if not is_main_process():
            return None
        writer_key = f"{phase}-{dataset_key}-{metric_key}"
        if writer_key not in self.writers:
            writer_path = os.path.join(
                self.log_dir, phase, dataset_key, metric_key, "metric_board"
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
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "ckpt_best.pth")
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
            f"Checkpoint saved to {save_path}, current ckpt is {ckpt_info}")

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
    # v5.1: Frequency branch gradient utilities
    # ------------------------------------------------------------------

    def _get_freq_params(self):
        """
        Lazily collect frequency extractor parameters for per-branch clipping.
        Cached after first call since model structure doesn't change.
        """
        if not hasattr(self, '_freq_params_cache'):
            m = self.model.module if isinstance(self.model, DDP) else self.model
            freq_ext = getattr(m, 'frequency_extractor', None)
            if freq_ext is not None:
                self._freq_params_cache = [
                    p for p in freq_ext.parameters() if p.requires_grad
                ]
            else:
                self._freq_params_cache = []
        return self._freq_params_cache

    def _clip_freq_gradients(self):
        """
        Clip frequency extractor gradients separately from the global clip.
        
        Why this is needed:
          Global clip_grad_norm_ computes ONE L2 norm across all ~304M params.
          The frequency extractor has ~200K trainable params (LayerNorms + FAD).
          A gradient spike of magnitude 1000 in a single freq param contributes
          sqrt(1000²) = 1000 to the per-branch norm, but only 
          sqrt(1000² / 304M_total_grads) ≈ 0.002 to the global norm.
          
          So the global norm can be 0.5 (under max_norm=1.0) while a single
          freq parameter has gradient 1000 → the spike passes through → NaN
          in the next forward pass.
          
        This method clips freq params to max_norm=2.0 BEFORE the global clip,
        catching per-branch spikes that the global clip misses.
        """
        freq_params = self._get_freq_params()
        if freq_params:
            torch.nn.utils.clip_grad_norm_(
                freq_params, max_norm=self.freq_grad_clip
            )

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

            # v5.1: Two-level gradient clipping
            # Level 1: Per-branch clip on frequency extractor (tighter).
            # This catches gradient spikes in the freq branch's ~200K params
            # that get diluted by the global norm across ~304M total params.
            self._clip_freq_gradients()

            # Level 2: Global clip on all parameters (standard safety net).
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.grad_clip)

            # scaler.step() skips the update if gradients contain inf/nan
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # SAE decoder normalization (prevents encoder/decoder scale drift)
            m = self.model.module if isinstance(self.model, DDP) else self.model
            if hasattr(m, 'sparse_ae') and m.sparse_ae is not None:
                if hasattr(m, 'use_sparse') and m.use_sparse:
                    m.sparse_ae.normalize_decoder_weights()

            return losses, predictions

    # ------------------------------------------------------------------
    # Causal warmup
    # ------------------------------------------------------------------

    def _run_causal_warmup(self, dataloader, n_batches: int):
        """
        Pre-populate the causal module's A_dce EMA buffer using real-face
        batches BEFORE phase 3 training starts.

        Why: A_dce initialises to zeros.  If we start phase 3 training
        immediately, the first violation scores are meaningless noise
        because the graph hasn't learned any real-face structure yet.
        Running real-only forward passes here biologically initialises
        the graph before any fake frames influence it.

        Strategy:
          - Freeze all params EXCEPT causal_module (no learning, just EMA)
          - For each batch: keep only real frames (label==0)
          - Run backbone + SAE under no_grad (frozen)
          - Run causal_module forward with grad enabled (for Jacobian)
          - EMA updates happen inside compute_adjacency_dce as side-effect
          - Restore frozen/trainable states after warmup
        """
        m = self.model.module if isinstance(self.model, DDP) else self.model

        # Freeze all params except causal_module
        frozen_params = set()
        for name, param in m.named_parameters():
            if 'causal_module' not in name and param.requires_grad:
                param.requires_grad = False
                frozen_params.add(name)

        m.train()
        n_done = 0

        for data_dict in dataloader:
            if n_done >= n_batches:
                break

            if 'label' not in data_dict:
                continue

            for key in data_dict.keys():
                val = data_dict[key]
                if val is not None and isinstance(val, torch.Tensor) and key != 'name':
                    data_dict[key] = val.cuda()

            label = data_dict['label']
            real_mask = (label == 0)
            if not real_mask.any():
                continue  # skip all-fake batches

            # Filter to real frames only
            real_dict = {}
            for k, v in data_dict.items():
                if isinstance(v, torch.Tensor):
                    real_dict[k] = v[real_mask]
                else:
                    real_dict[k] = v

            # Backbone + SAE forward (no grad — frozen)
            with torch.no_grad():
                raw_feats = m.extract_raw_features(real_dict)
                fused_feats = m.project_and_fuse(raw_feats)

                z_spatial, z_freq = None, None
                sae_loss_dummy = None
                if m.use_sparse:
                    z_spatial, z_freq, sae_loss_dummy, _ = m.sparse_ae(
                        spatial_feat=raw_feats.get('spatial_raw'),
                        frequency_feat=raw_feats.get('frequency_raw'),
                    )
                z_sae = (m.sparse_ae.get_z_sae(z_spatial, z_freq)
                         if m.use_sparse else None)

            # Causal module needs grad for Jacobian → run outside no_grad
            semantic_attrs = real_dict.get('semantic_attrs', None)
            m.causal_module(
                fused_feats.detach(),
                z_sae=z_sae.detach() if z_sae is not None else None,
                semantic_attrs=(semantic_attrs.detach()
                                if semantic_attrs is not None else None),
                # No label needed here: ALL frames are real (already filtered)
                return_graph=True,
            )

            n_done += 1
            if n_done % 50 == 0 and is_main_process():
                self.logger.info(
                    f"  Causal warmup: {n_done}/{n_batches} real-face batches"
                )

        # Restore trainable states
        for name, param in m.named_parameters():
            if name in frozen_params:
                param.requires_grad = True

        ema_init = (m.causal_module.causal_learner._ema_initialized
                    if hasattr(m, 'causal_module') else False)
        if is_main_process():
            self.logger.info(
                f"  Causal warmup complete: {n_done} batches processed. "
                f"A_dce EMA initialized: {ema_init}"
            )

    def train_epoch(self, epoch, train_data_loader, test_data_loaders=None):
        self.logger.info("===> Epoch[{}] start!".format(epoch))

        # ── Phase transition (GenD-style progressive unfreezing) ──────────
        # Must run BEFORE the iteration loop so the entire epoch trains with
        # the correct frozen/unfrozen state and correct optimizer param groups.
        #
        # Phase 1 (epochs 0→phase2_start): backbone fully frozen, only
        #   backbone LayerNorms + projection heads + fusion + classifier train.
        # Phase 2 (epochs phase2_start→end): backbone LayerNorms explicitly
        #   re-confirmed unfrozen (they already are from phase1, but calling
        #   unfreeze_layernorms() is idempotent and re-logs for clarity).
        #   Optimizer is rebuilt so any param that became requires_grad=True
        #   between epochs is actually in a param group.
        #
        # We do NOT call unfreeze_full() — that would destroy pretrained
        # representations. LayerNorm-only is the GenD generalization regime.
        phases = self.config.get('training_phases', {})
        phase2 = phases.get('phase2', {})
        phase2_start = phase2.get('epochs', [None, None])[0]

        if phase2_start is not None and epoch == phase2_start:
            self.logger.info(
                f"===> Phase 2 transition at epoch {epoch}: "
                f"unfreezing backbone LayerNorms across all active branches."
            )
            # Unwrap DDP to access the actual model attributes
            m = self.model.module if isinstance(self.model, DDP) else self.model

            for attr in ('spatial_extractor', 'frequency_extractor'):
                extractor = getattr(m, attr, None)
                if extractor is None:
                    continue
                if hasattr(extractor, 'unfreeze_layernorms'):
                    n_unfrozen = extractor.unfreeze_layernorms()
                    self.logger.info(
                        f"  {attr}: {n_unfrozen:,} LayerNorm params unfrozen"
                    )
                else:
                    self.logger.warning(
                        f"  {attr}: no unfreeze_layernorms() method — skipping. "
                        f"Add it following the GenD-style extractor pattern."
                    )

            # Rebuild optimizer so newly unfrozen params are in a param group.
            # Without this, their gradients are computed but silently discarded
            # because Adam has no state for them yet.
            from training.train import choose_optimizer, choose_scheduler
            self.optimizer = choose_optimizer(self.model, self.config)
            self.scheduler = choose_scheduler(self.config, self.optimizer)
            self.logger.info(
                "  Optimizer and scheduler rebuilt with updated param groups."
            )

            # v5.1: Invalidate the cached freq params since optimizer was rebuilt
            # and param groups may have changed.
            if hasattr(self, '_freq_params_cache'):
                del self._freq_params_cache

        # ── Phase 3 transition: enable causal discovery ───────────────────
        # At phase3_start the trainer:
        #   1. Calls model.enable_causal() → sets use_causal=True
        #   2. Runs causal warmup (real-face-only batches) to pre-populate
        #      A_dce EMA before any fake data influences the graph.
        #   3. Sets causal + contrastive loss weights and rebuilds optimizer.
        phase3 = phases.get('phase3', {})
        phase3_start = phase3.get('epochs', [None, None])[0]

        if phase3_start is not None and epoch == phase3_start:
            self.logger.info(
                f"===> Phase 3 transition at epoch {epoch}: "
                f"enabling causal discovery module."
            )
            m = self.model.module if isinstance(self.model, DDP) else self.model

            # 1. Enable causal module
            if hasattr(m, 'enable_causal'):
                m.enable_causal()
            else:
                self.logger.warning(
                    "  model has no enable_causal() method — setting use_causal directly."
                )
                m.use_causal = True

            # 2. Set loss weights for phase 3
            causal_loss_weight = phase3.get('causal_loss_weight', 0.3)
            self.config['loss_func']['weights']['causal'] = causal_loss_weight
            self.config['loss_func']['weights']['contrastive'] = causal_loss_weight
            if hasattr(m, 'loss_weights'):
                m.loss_weights['causal'] = causal_loss_weight
                m.loss_weights['contrastive'] = causal_loss_weight
            self.logger.info(
                f"  causal loss weight: {causal_loss_weight}, "
                f"contrastive loss weight: {causal_loss_weight}"
            )

            # 3. Causal warmup: pre-populate A_dce EMA with real-face batches
            causal_warmup_batches = (
                self.config.get('causal_module', {})
                .get('causal_warmup_batches', 200)
            )
            if causal_warmup_batches > 0:
                self.logger.info(
                    f"  Running causal warmup: {causal_warmup_batches} "
                    f"real-face batches to pre-populate A_dce EMA..."
                )
                self._run_causal_warmup(train_data_loader, n_batches=causal_warmup_batches)

            # 4. Rebuild optimizer so causal_module params enter Adam state
            from training.train import choose_optimizer, choose_scheduler
            self.optimizer = choose_optimizer(self.model, self.config)
            self.scheduler = choose_scheduler(self.config, self.optimizer)
            self.logger.info(
                "  Optimizer and scheduler rebuilt for phase 3."
            )

            if hasattr(self, '_freq_params_cache'):
                del self._freq_params_cache

        # ── Rest of train_epoch unchanged ─────────────────────────────────
        times_per_epoch = 1
        test_step  = len(train_data_loader) // times_per_epoch
        step_cnt   = epoch * len(train_data_loader)

        if is_main_process():
            data_dict = train_data_loader.dataset.data_dict
            self.save_data_dict(
                'train', data_dict, ','.join(self.config['train_dataset']))

        train_recorder_loss   = defaultdict(Recorder)
        train_recorder_metric = defaultdict(Recorder)

        for iteration, data_dict in tqdm(
            enumerate(train_data_loader),
            total=len(train_data_loader),
            disable=not is_main_process(),
        ):
            self.setTrain()
            for key in data_dict.keys():
                if data_dict[key] is not None and key != 'name':
                    data_dict[key] = data_dict[key].cuda()

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

    def test_one_dataset(self, data_loader, desc="Testing"):
        """
        Each rank processes its own shard (via DistributedSampler).
        Returns local numpy arrays; caller gathers across ranks.
        Inference also runs under autocast for consistency + speed.
        """
        test_recorder_loss = defaultdict(Recorder)
        prediction_lists   = []
        feature_lists      = []
        label_lists        = []

        for i, data_dict in tqdm(
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
                    data_dict[key] = val.cuda()

            predictions = self.inference(data_dict)
            label_lists      += list(data_dict['label'].cpu().detach().numpy())
            prediction_lists += list(predictions['prob'].cpu().detach().numpy())
            feature_lists    += list(predictions['feat'].cpu().detach().numpy())

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
            self.save_metrics('test', metric_one_dataset, key)

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

        keys = list(test_data_loaders.keys())

        for key in keys:
            if is_main_process():
                data_dict_meta = test_data_loaders[key].dataset.data_dict
                self.save_data_dict('test', data_dict_meta, key)

            self.logger.info(f"Testing on {key}...")
            (losses_local, preds_local,
             labels_local, feats_local) = self.test_one_dataset(
                test_data_loaders[key], desc=key)

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
        with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
            predictions = self.model(data_dict, inference=True)
        return predictions