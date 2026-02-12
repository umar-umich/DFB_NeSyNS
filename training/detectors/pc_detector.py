"""
Probabilistic Circuit Detector for DeepfakeBench
Wraps any existing DeepfakeBench detector backbone and adds a
class-conditional Sum-Product Network head for calibrated uncertainty
estimation alongside standard classification.

Reference architecture: NeSyDeFake Module 2
  Frozen backbone features -> Class-Conditional PC -> Fusion MLP
  -> logits + uncertainty score (predictive entropy / LLR)
"""

import os
import numpy as np
import logging
from typing import Union, Dict, Tuple, Optional
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn import metrics as sk_metrics

from metrics.base_metrics_class import calculate_metrics_for_train
from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Probabilistic Circuit components
# ─────────────────────────────────────────────────────────────────────────────

class GaussianLeaf(nn.Module):
    """
    Multivariate diagonal-Gaussian leaves.
    Models a partition of the feature space with K mixture components,
    assuming feature-wise independence within each component.
    """

    def __init__(self, num_features: int, num_components: int):
        super().__init__()
        self.num_features = num_features
        self.num_components = num_components

        # [K, D]  — one mean / log-std per component per feature
        self.means    = nn.Parameter(torch.randn(num_components, num_features) * 0.01)
        self.log_stds = nn.Parameter(torch.zeros(num_components, num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, D]
        Returns:
            log_probs : [B, K]  (sum of per-feature log-probs under each component)
        """
        x_exp  = x.unsqueeze(1)                         # [B, 1, D]
        means  = self.means.unsqueeze(0)                 # [1, K, D]
        stds   = torch.exp(self.log_stds).unsqueeze(0)   # [1, K, D]

        log_probs = -0.5 * (
            torch.log(2.0 * np.pi * stds ** 2) +
            ((x_exp - means) / stds) ** 2
        )                                                # [B, K, D]
        return log_probs.sum(dim=-1)                     # [B, K]


class SumNode(nn.Module):
    """
    Weighted mixture (log-space) over I inputs -> O outputs.
    Weights are normalised row-wise via log_softmax.
    """

    def __init__(self, num_inputs: int, num_outputs: int):
        super().__init__()
        self.log_weights = nn.Parameter(torch.randn(num_outputs, num_inputs))

    def forward(self, log_probs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            log_probs : [B, I]
        Returns:
            out       : [B, O]
        """
        log_w    = F.log_softmax(self.log_weights, dim=1)   # [O, I]
        weighted = log_probs.unsqueeze(1) + log_w.unsqueeze(0)  # [B, O, I]
        return torch.logsumexp(weighted, dim=2)              # [B, O]


class ProbabilisticCircuit(nn.Module):
    """
    Shallow Sum-Product Network (SPN) for density estimation.

    Architecture per class:
      Leaf (Gaussian, partitioned features)
        -> Product  (sum log-probs across partitions)
        -> Sum      (mixture layer, width nodes)
        -> Sum      (root, single output)

    Args:
        feature_dim        : dimensionality of input features
        width              : number of sum-node outputs in the hidden mixture layer
        num_leaf_components: number of Gaussian components per leaf partition
        feature_splits     : how many feature partitions to use
    """

    def __init__(
        self,
        feature_dim:         int,
        width:               int = 128,
        num_leaf_components: int = 16,
        feature_splits:      int = 4,
    ):
        super().__init__()
        self.feature_dim    = feature_dim
        self.feature_splits = feature_splits

        # Each partition covers (feature_dim // feature_splits) dimensions.
        # If feature_dim is not divisible, the last partition absorbs the remainder.
        self.split_dim = feature_dim // feature_splits
        self.last_split_dim = feature_dim - self.split_dim * (feature_splits - 1)

        # One Gaussian leaf per partition
        self.leaves = nn.ModuleList()
        for i in range(feature_splits):
            dim = self.last_split_dim if i == feature_splits - 1 else self.split_dim
            self.leaves.append(GaussianLeaf(dim, num_leaf_components))

        # Hidden mixture layer
        self.sum_hidden = SumNode(num_leaf_components, width)

        # Root: collapses to a single log-probability scalar
        self.sum_root = SumNode(width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x        : [B, feature_dim]
        Returns:
            log_prob : [B]
        """
        # Split features into partitions
        splits = []
        for i in range(self.feature_splits):
            start = i * self.split_dim
            end   = start + (self.last_split_dim if i == self.feature_splits - 1 else self.split_dim)
            splits.append(x[:, start:end])

        # Layer 1: evaluate leaves, then product across partitions
        leaf_outs = [leaf(split) for leaf, split in zip(self.leaves, splits)]  # [B, K] each
        prod_out  = sum(leaf_outs)   # [B, K]  (sum of log-probs = log of products)

        # Layer 2: hidden sum
        sum_out   = self.sum_hidden(prod_out)   # [B, width]

        # Layer 3: root sum
        log_prob  = self.sum_root(sum_out)      # [B, 1]
        return log_prob.squeeze(-1)             # [B]


class ClassConditionalPC(nn.Module):
    """
    Two SPNs — one per class (real / fake) — modelling
    p(z | real) and p(z | fake) independently.

    At inference time produces:
      - log_prob_real      : log p(z | real)
      - log_prob_fake      : log p(z | fake)
      - log_likelihood_ratio (LLR) : log p(z|fake) - log p(z|real)
      - entropy            : H(Y | z)  under a uniform class prior
    """

    def __init__(
        self,
        feature_dim:         int,
        width:               int = 128,
        num_leaf_components: int = 16,
        feature_splits:      int = 4,
    ):
        super().__init__()
        kwargs = dict(
            feature_dim=feature_dim,
            width=width,
            num_leaf_components=num_leaf_components,
            feature_splits=feature_splits,
        )
        self.pc_real = ProbabilisticCircuit(**kwargs)
        self.pc_fake = ProbabilisticCircuit(**kwargs)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        log_p_real = self.pc_real(x)   # [B]
        log_p_fake = self.pc_fake(x)   # [B]

        llr = log_p_fake - log_p_real  # [B]

        # Predictive entropy H(Y|z) under uniform prior
        log_probs = torch.stack([log_p_real, log_p_fake], dim=1)  # [B, 2]
        probs     = F.softmax(log_probs, dim=1)                   # [B, 2]
        entropy   = -(probs * torch.log(probs + 1e-10)).sum(dim=1)  # [B]

        return {
            'log_prob_real':         log_p_real,
            'log_prob_fake':         log_p_fake,
            'log_likelihood_ratio':  llr,
            'entropy':               entropy,
        }


class PCFusionHead(nn.Module):
    """
    MLP that fuses backbone features with PC statistics for final classification.

    Input  : [backbone_features  (D),  log_p_real (1),  log_p_fake (1),  entropy (1)]
    Output : logits [B, 2]
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim:  int   = 256,
        dropout:     float = 0.1,
    ):
        super().__init__()
        input_dim = feature_dim + 3   # features + 3 PC scalars

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(
        self,
        features:  torch.Tensor,
        pc_stats:  Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        pc_vec  = torch.stack([
            pc_stats['log_prob_real'],
            pc_stats['log_prob_fake'],
            pc_stats['entropy'],
        ], dim=1)                                          # [B, 3]
        combined = torch.cat([features, pc_vec], dim=1)   # [B, D+3]
        return self.net(combined)                          # [B, 2]


# ─────────────────────────────────────────────────────────────────────────────
# Feature standardisation (fitted on training data, used at inference)
# ─────────────────────────────────────────────────────────────────────────────

class FeatureNorm(nn.Module):
    """Running mean/std normalisation (no learnable parameters)."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.register_buffer('mean', torch.zeros(feature_dim))
        self.register_buffer('std',  torch.ones(feature_dim))
        self.fitted = False

    def fit(self, features: torch.Tensor):
        self.mean   = features.mean(0)
        self.std    = features.std(0).clamp(min=1e-6)
        self.fitted = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


# ─────────────────────────────────────────────────────────────────────────────
# DeepfakeBench detector wrapper
# ─────────────────────────────────────────────────────────────────────────────

@DETECTOR.register_module(module_name='pc_detector')
class PCDetector(AbstractDetector):
    """
    Probabilistic-Circuit Detector for DeepfakeBench.

    Uses any existing DeepfakeBench backbone (e.g. xception, efficientnet,
    clip_vit, etc.) as a frozen feature extractor and trains a PC head on top.

    Two-stage training
    ------------------
    Stage 1  (``train_mode='pc_only'``)
      The backbone is frozen.  Each class-conditional PC is fitted via
      maximum-likelihood on the training split.

    Stage 2  (``train_mode='fusion'``)
      Both the backbone and PCs are frozen.
      Only the fusion MLP is fine-tuned with cross-entropy loss.

    Full end-to-end  (``train_mode='full'``)
      All components are trainable simultaneously (use with care; the PC
      is typically better conditioned when pre-fitted in stage 1).

    Output dictionary
    -----------------
    ``pred_dict`` keys:
      - cls          : raw logits [B, 2]
      - prob         : fake probability [B]
      - feat         : backbone features [B, D]
      - uncertainty  : predictive entropy [B]   ← NEW
      - llr          : log-likelihood ratio [B] ← NEW
      - log_p_real   : log p(z | real) [B]
      - log_p_fake   : log p(z | fake) [B]
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        # ── 1. Backbone (any registered DeepfakeBench backbone) ──────────────
        backbone_class  = BACKBONE[config['backbone_name']]
        model_config    = config['backbone_config']
        self.backbone   = backbone_class(model_config)

        if config.get('pretrained'):
            state_dict = torch.load(config['pretrained'], map_location='cpu')
            # Some backbones prefix keys with 'module.' from DataParallel
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
            logger.info(f'Loaded pretrained weights | missing={len(missing)} unexpected={len(unexpected)}')

        # ── 2. Feature dimensionality ─────────────────────────────────────────
        # Try to infer from config; fall back to probing the backbone.
        feature_dim = config.get('pc_config', {}).get('feature_dim', None)
        if feature_dim is None:
            feature_dim = self._infer_feature_dim()
        self.feature_dim = feature_dim
        logger.info(f'PC feature dim: {feature_dim}')

        # ── 3. Feature normalisation ──────────────────────────────────────────
        self.feat_norm = FeatureNorm(feature_dim)

        # ── 4. PC head ────────────────────────────────────────────────────────
        pc_cfg = config.get('pc_config', {})
        self.pc_head = ClassConditionalPC(
            feature_dim         = feature_dim,
            width               = pc_cfg.get('width',               128),
            num_leaf_components = pc_cfg.get('num_leaf_components',   16),
            feature_splits      = pc_cfg.get('feature_splits',         4),
        )

        # ── 5. Fusion MLP ──────────────────────────────────────────────────────
        self.fusion_head = PCFusionHead(
            feature_dim = feature_dim,
            hidden_dim  = pc_cfg.get('fusion_hidden_dim', 256),
            dropout     = pc_cfg.get('fusion_dropout',    0.1),
        )

        # ── 6. Loss ───────────────────────────────────────────────────────────
        self.loss_func = self._build_loss(config)

        # ── 7. Training mode ──────────────────────────────────────────────────
        self.train_mode = config.get('train_mode', 'fusion')
        self._apply_freeze_policy()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _infer_feature_dim(self) -> int:
        """Run a dummy forward pass to discover backbone output dimension."""
        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.zeros(2, 3, 256, 256)
            try:
                out = self.backbone.features({'image': dummy})
            except Exception:
                try:
                    out = self.backbone(dummy)
                except Exception:
                    logger.warning('Cannot infer feature dim; defaulting to 512.')
                    return 512
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.dim() > 2:
            out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return out.shape[-1]

    def _build_loss(self, config: dict) -> dict:
        cls_loss_cls  = LOSSFUNC[config['loss_func']['cls_loss']]
        loss_func = {'cls': cls_loss_cls()}
        if 'mask_loss' in config.get('loss_func', {}):
            mask_loss_cls = LOSSFUNC[config['loss_func']['mask_loss']]
            loss_func['mask'] = mask_loss_cls()
        return loss_func

    def _apply_freeze_policy(self):
        """Freeze / unfreeze sub-modules according to train_mode."""
        mode = self.train_mode
        # Default: everything trainable
        for p in self.parameters():
            p.requires_grad_(True)

        if mode == 'pc_only':
            # Only the PC head is trained; backbone & fusion are frozen
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            for p in self.fusion_head.parameters():
                p.requires_grad_(False)
            logger.info('Train mode: PC only (backbone + fusion frozen)')

        elif mode == 'fusion':
            # Only the fusion MLP is trained; backbone & PCs are frozen
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            for p in self.pc_head.parameters():
                p.requires_grad_(False)
            logger.info('Train mode: Fusion only (backbone + PC frozen)')

        elif mode == 'full':
            logger.info('Train mode: Full end-to-end')

        else:
            logger.warning(f'Unknown train_mode "{mode}"; all parameters trainable.')

    # ── Feature extraction ────────────────────────────────────────────────────

    def features(self, data_dict: dict) -> torch.Tensor:
        """
        Extract a flat feature vector from the backbone.
        Handles different backbone API styles:
          (a) backbone.features(data_dict)   → tensor or (tensor, ...)
          (b) backbone(data_dict['image'])   → tensor or dict
        """
        x = data_dict['image']

        try:
            out = self.backbone.features(data_dict)
        except (AttributeError, TypeError):
            out = self.backbone(x)

        # Unwrap tuples / dicts
        if isinstance(out, (tuple, list)):
            out = out[0]
        elif isinstance(out, dict):
            out = out.get('feat', out.get('features', list(out.values())[0]))

        # Spatial -> vector
        if out.dim() == 4:
            out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        elif out.dim() == 3:
            out = out.mean(1)   # e.g. ViT sequence

        return out   # [B, D]

    def classifier(self, features: torch.Tensor) -> torch.Tensor:
        """Classifier that uses the fusion head (PC stats computed inside forward)."""
        raise NotImplementedError(
            'Use PCDetector.forward() — classifier() is not meaningful in isolation.'
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, data_dict: dict, inference: bool = False) -> dict:
        """
        Full forward pass.

        Returns
        -------
        pred_dict : dict with keys
            cls         – logits [B, 2]
            prob        – fake probability [B]
            feat        – backbone features [B, D]
            uncertainty – predictive entropy [B]
            llr         – log-likelihood ratio [B]
            log_p_real  – log p(z|real) [B]
            log_p_fake  – log p(z|fake) [B]
        """
        # 1. Backbone features
        feats = self.features(data_dict)      # [B, D]

        # 2. Normalise
        norm_feats = self.feat_norm(feats)    # [B, D]

        # 3. PC statistics
        pc_stats = self.pc_head(norm_feats)   # dict of [B] tensors

        # 4. Fusion MLP
        logits = self.fusion_head(norm_feats, pc_stats)   # [B, 2]

        prob = torch.softmax(logits, dim=1)[:, 1]          # [B]  fake probability

        pred_dict = {
            'cls':         logits,
            'prob':        prob,
            'feat':        feats,
            'uncertainty': pc_stats['entropy'],
            'llr':         pc_stats['log_likelihood_ratio'],
            'log_p_real':  pc_stats['log_prob_real'],
            'log_p_fake':  pc_stats['log_prob_fake'],
        }
        return pred_dict

    # ── Losses ────────────────────────────────────────────────────────────────

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label    = data_dict['label']
        pred_cls = pred_dict['cls']

        loss = self.loss_func['cls'](pred_cls, label)
        loss_dict = {'overall': loss, 'cls': loss}

        # Optional PC NLL terms (used during stage-1 / full training)
        if self.train_mode in ('pc_only', 'full'):
            real_mask = (label == 0)
            fake_mask = (label == 1)
            nll_real = nll_fake = torch.tensor(0.0, device=loss.device)

            if real_mask.any():
                nll_real = -pred_dict['log_p_real'][real_mask].mean()
            if fake_mask.any():
                nll_fake = -pred_dict['log_p_fake'][fake_mask].mean()

            pc_loss = nll_real + nll_fake
            loss    = loss + self.config.get('pc_loss_weight', 0.1) * pc_loss
            loss_dict.update({
                'overall':  loss,
                'nll_real': nll_real,
                'nll_fake': nll_fake,
            })

        return loss_dict

    # ── Metrics ───────────────────────────────────────────────────────────────

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred  = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}

    def get_test_metrics(self) -> dict:
        """
        Aggregate epoch-level metrics including calibration statistics.
        Called once per epoch by the DeepfakeBench trainer.
        """
        # Subclasses can override; the default implementation returns {}
        # so that the trainer uses its own aggregation logic.
        return {}

    # ── Two-stage fitting utilities ───────────────────────────────────────────

    @torch.no_grad()
    def collect_features(
        self,
        dataloader,
        device: torch.device,
        max_samples: int = 50_000,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run the backbone over a dataloader and collect normalised features.

        Returns
        -------
        features : [N, D]
        labels   : [N]
        """
        self.backbone.eval()
        all_feats, all_labels = [], []
        n = 0
        for batch in dataloader:
            if n >= max_samples:
                break
            imgs   = batch['image'].to(device)
            labels = batch['label'].to(device)
            feats  = self.features({'image': imgs})
            all_feats.append(feats.cpu())
            all_labels.append(labels.cpu())
            n += imgs.size(0)

        features = torch.cat(all_feats,  dim=0)[:max_samples]
        labels   = torch.cat(all_labels, dim=0)[:max_samples]
        return features, labels

    def fit_feature_norm(self, features: torch.Tensor):
        """Fit the running-mean/std normaliser on training features."""
        self.feat_norm.fit(features)
        logger.info('FeatureNorm fitted.')

    def fit_pc_stage1(
        self,
        features:   torch.Tensor,
        labels:     torch.Tensor,
        num_epochs: int = 100,
        lr:         float = 1e-3,
        batch_size: int  = 256,
        device:     Optional[torch.device] = None,
    ):
        """
        Stage 1: fit each class-conditional PC via maximum-likelihood.

        Args:
            features   : [N, D]  (already normalised)
            labels     : [N]     (0=real, 1=fake)
            num_epochs : training epochs per PC
            lr         : Adam learning rate
            batch_size : mini-batch size
            device     : compute device
        """
        if device is None:
            device = next(self.parameters()).device

        norm_feats = self.feat_norm(features.to(device))

        for cls_idx, (pc, cls_name) in enumerate(
            [(self.pc_head.pc_real, 'real'), (self.pc_head.pc_fake, 'fake')]
        ):
            mask  = (labels == cls_idx)
            data  = norm_feats[mask]
            if data.shape[0] == 0:
                logger.warning(f'No samples for class "{cls_name}" — skipping PC fit.')
                continue

            logger.info(f'Fitting PC for class "{cls_name}" on {data.shape[0]} samples …')
            pc.train()
            optimizer = optim.Adam(pc.parameters(), lr=lr)
            dataset   = torch.utils.data.TensorDataset(data)
            loader    = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True, drop_last=False
            )

            for epoch in range(num_epochs):
                total_nll = 0.0
                for (batch_x,) in loader:
                    optimizer.zero_grad()
                    nll = -pc(batch_x).mean()
                    nll.backward()
                    optimizer.step()
                    total_nll += nll.item()

                if (epoch + 1) % 20 == 0:
                    avg_nll = total_nll / max(len(loader), 1)
                    logger.info(
                        f'  [{cls_name}] epoch {epoch+1}/{num_epochs}  NLL: {avg_nll:.4f}'
                    )

            pc.eval()
            logger.info(f'PC for class "{cls_name}" fitted.')