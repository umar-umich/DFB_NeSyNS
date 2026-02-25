"""
detectors/nesy_defake_hybrid_detector.py
=========================================
REGULARIZATION FIXES APPLIED:

  Fix 1 — Class-weighted CrossEntropyLoss
           Addresses acc_real ≈ 0 across all three branches.
           FF++ is ~80% fake so unweighted CE has a degenerate local optimum
           of "predict fake always". Class weights break this early.
           Config: class_weights: [2.0, 1.0]  # [real, fake]

  Fix 2 — Dropout in projection heads
           Spatial was overfitting hard (train loss → 0.002 while test loss → 0.9).
           Per-branch dropout: spatial=0.2, temporal=0.1, frequency=0.1
           Config: projection_dropout: {spatial: 0.2, temporal: 0.1, frequency: 0.1}

  Fix 3 — Temporal consistency loss (DISABLED for baseline)
           Kept in codebase but temporal_consistency_weight defaults to 0.0.
           Re-enable after core branches validate on FF++ + Celeb-DF-v2.
           Config: temporal_consistency_weight: 0.0  # set >0 to enable

  Fix 4 — Class weight device move in get_losses()
           CrossEntropyLoss.weight is created on CPU. Without explicit
           .to(device), it crashes on first forward pass on GPU.

  Fix 5 — Uniform active_branches conditioning in features()
           All three branches are now symmetrically gated:
               if 'branch' in self.active_branches
           Previously only temporal had the guard. Spatial and frequency
           crashed with KeyError when running temporal-only ablations because
           data_dict['spatial_frames'] / data_dict['freq_frames'] may not
           be present if the dataset only prepares active-branch tensors.
           Safe key access via data_dict.get() added for all branches.

  Fix 6 — fused_dim recomputed from active branches only
           build_backbone() patches config['fusion']['fused_dim'] at runtime
           based on which branches are active. MultiModalFusion must be built
           AFTER build_backbone() — this order is already correct in __init__.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics.base_metrics_class import calculate_metrics_for_train
from .base_detector import AbstractDetector
from detectors import DETECTOR
from typing import Iterator, List, Tuple

from networks.nesy_defake.foundation_models import (
    TemporalFeatureExtractor,
    FrequencyFeatureExtractor,
    SpatialFeatureExtractor,
)
from networks.nesy_defake.fusion import MultiModalFusion
from networks.nesy_defake.causal import CausalDiscoveryModule
from networks.nesy_defake.classifiers import MultiTaskHead, SparseAutoencoder
from utils.semantic_grounding import SemanticGroundingModule
from utils.encoder_utils import build_encoder, ENCODER_REGISTRY

logger = logging.getLogger(__name__)
ALL_BRANCHES = ('temporal', 'spatial', 'frequency')

# Data dict keys for each branch — used for safe access in features()
_BRANCH_DATA_KEYS = {
    'temporal':  'temporal_clip',
    'spatial':   'spatial_frames',
    'frequency': 'freq_frames',
}


def _zero_grad_anchor(module: nn.Module, device: torch.device) -> torch.Tensor:
    anchor = torch.zeros(1, device=device, dtype=torch.float32)
    for p in module.parameters():
        if p.requires_grad:
            anchor = anchor + p.sum() * 0.0
    return anchor


def _make_projection(in_dim: int, out_dim: int, dropout: float = 0.0) -> nn.Sequential:
    """
    Learnable branch projection head: Linear → LayerNorm → GELU → Dropout.

    Dropout placed post-activation so it gates final projected features
    rather than disrupting LayerNorm statistics.

    Per-branch dropout rationale:
        spatial   : 0.2  (strongest branch — higher dropout prevents fusion dominance)
        frequency : 0.2  (weakest branch — higher dropout forces generalization)
        temporal  : 0.1  (moderate)
    """
    layers: list = [
        nn.Linear(in_dim, out_dim),
        nn.LayerNorm(out_dim),
        nn.GELU(),
    ]
    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


@DETECTOR.register_module(module_name='nesydefake_hybrid')
class NeSyDeFakeHybridDetector(AbstractDetector):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # build_backbone() must run BEFORE MultiModalFusion so that
        # config['fusion']['fused_dim'] is patched correctly for active branches.
        self.build_backbone(config)
        self.fusion = MultiModalFusion(config)

        self.use_semantic_grounding = config.get(
            'semantic_grounding', {}).get('enabled', False)
        if self.use_semantic_grounding:
            self.semantic_grounding = SemanticGroundingModule(
                config, feature_dim=config['fusion']['projection_dim'])
            self.grounded_feature_dim = self.semantic_grounding.output_dim
        else:
            self.grounded_feature_dim = config['fusion']['projection_dim']

        self.use_causal = config['causal_module']['enabled']
        if self.use_causal:
            self.causal_module = CausalDiscoveryModule(config)

        self.use_sparse = config['sparse_features']['enabled']
        if self.use_sparse:
            self.sparse_ae = SparseAutoencoder(config)

        self.multitaskhead = MultiTaskHead(config)
        self.loss_weights  = config['loss_func']['weights']
        self.build_loss(config)

        logger.info("NeSyDeFake Hybrid Detector initialised")
        logger.info(f"  Active branches : {sorted(self.active_branches)}")
        logger.info(f"  Fused dim       : {config['fusion']['fused_dim']}")
        logger.info(f"  Projection dim  : {config['fusion']['projection_dim']}")

    # ------------------------------------------------------------------ #
    #  Construction helpers                                                #
    # ------------------------------------------------------------------ #

    def _build_shared_encoder(self, config: dict):
        """
        Build ONE shared vision encoder for both spatial and frequency branches.

        Returns shared encoder module (already frozen per GenD regime),
        or None if sharing is disabled or paths differ.
        """
        fm           = config['foundation_models']
        spatial_path = fm['spatial']['model_path']
        freq_path    = fm['frequency'].get('model_path', '')
        do_share     = config.get('shared_encoder', True)

        if not do_share:
            logger.info("[Detector] Shared encoder disabled via config.")
            return None

        if spatial_path != freq_path:
            logger.info(
                f"[Detector] Spatial and frequency use different model paths — "
                f"cannot share encoder.\n"
                f"  spatial  : {spatial_path}\n"
                f"  frequency: {freq_path}"
            )
            return None

        logger.info(
            f"[Detector] Building shared encoder: {spatial_path}\n"
            f"  Shared between spatial + frequency branches."
        )
        shared_encoder, hidden_dim, required_size, api_type = \
            build_encoder(spatial_path)

        freeze    = fm['spatial'].get('freeze_backbone', True)
        train_lns = fm['spatial'].get('train_layernorms', True)

        if freeze:
            if train_lns:
                total, unfrozen = self._freeze_all_except_layernorms(shared_encoder)
                logger.info(
                    f"[Detector] Shared encoder GenD freeze: "
                    f"{unfrozen:,} / {total:,} params trainable (LayerNorms only)"
                )
            else:
                for p in shared_encoder.parameters():
                    p.requires_grad = False
                total = sum(p.numel() for p in shared_encoder.parameters())
                logger.info(
                    f"[Detector] Shared encoder hard frozen: {total:,} params."
                )
        else:
            logger.warning(
                "[Detector] freeze_backbone=False on shared encoder: "
                "full fine-tune. Hurts cross-dataset generalization."
            )

        return shared_encoder

    def build_backbone(self, config: dict) -> None:
        """
        Build all three branch extractors with optional shared encoder.

        All three extractors are always built regardless of active_branches
        so that ablation runs can re-enable branches without config changes.
        Inactive branch parameters are frozen after construction (Step 6).

        fused_dim is patched into config['fusion']['fused_dim'] here so
        MultiModalFusion receives the correct value when built in __init__.
        """
        # Step 1: shared encoder for spatial + frequency (or None)
        shared_encoder = self._build_shared_encoder(config)

        # Step 2: register shared encoder as detector submodule.
        # nn.Module deduplicates params — no double counting in optimizer.
        if shared_encoder is not None:
            self.shared_encoder = shared_encoder

        # Step 3: build all three extractors (always, for ablation compatibility)
        self.temporal_extractor  = TemporalFeatureExtractor(config)
        self.spatial_extractor   = SpatialFeatureExtractor(
            config, shared_encoder=shared_encoder)
        self.frequency_extractor = FrequencyFeatureExtractor(
            config, shared_encoder=shared_encoder)

        # Step 4: resolve active branches
        active = set(config.get('active_branches', list(ALL_BRANCHES)))
        self.active_branches = active

        # Step 5: projection heads (all three always built for ablation)
        proj_dim    = config['fusion']['projection_dim']
        dropout_cfg = config.get('projection_dropout', {})

        branch_dims = {
            'temporal':  self.temporal_extractor.output_dim,
            'spatial':   self.spatial_extractor.backbone_dim,    # see SpatialExtractor
            'frequency': self.frequency_extractor.output_dim,
        }

        self.temporal_proj  = _make_projection(
            branch_dims['temporal'],  proj_dim,
            dropout=dropout_cfg.get('temporal',  0.1))
        self.spatial_proj   = _make_projection(
            branch_dims['spatial'],   proj_dim,
            dropout=dropout_cfg.get('spatial',   0.2))
        self.frequency_proj = _make_projection(
            branch_dims['frequency'], proj_dim,
            dropout=dropout_cfg.get('frequency', 0.2))

        # Step 6: patch fused_dim AND projection_dim into config for MultiModalFusion.
        # fused_dim = proj_dim x n_active_branches (what fusion receives as input).
        # projection_dim is re-written explicitly so fusion cannot accidentally
        # read a stale YAML value or a raw backbone output_dim (e.g. 1408).
        # The 32x1024 -> 1408x1408 matmul error was caused by MultiModalFusion
        # reading fused_dim=1408 (temporal backbone dim) instead of 1024 (proj dim).
        n_active  = sum(1 for b in ALL_BRANCHES if b in active)
        fused_dim = proj_dim * n_active
        config['fusion']['fused_dim']      = fused_dim   # e.g. 1024 for temporal-only
        config['fusion']['projection_dim'] = proj_dim    # always 1024, never backbone dim

        # Step 7: freeze inactive branch extractors + their projection heads
        for name in ALL_BRANCHES:
            if name not in active:
                self._freeze_module(getattr(self, f'{name}_extractor'), name)
                self._freeze_module(getattr(self, f'{name}_proj'), f'{name}_proj')

        # Logging
        fm = config['foundation_models']
        logger.info(
            f"  Shared encoder  : "
            f"{'YES — ' + fm['spatial']['model_path'] if shared_encoder is not None else 'NO — separate encoders per branch'}"
        )
        logger.info(
            f"  Branch dims     : temporal={branch_dims['temporal']}, "
            f"spatial={branch_dims['spatial']}, frequency={branch_dims['frequency']}"
        )
        logger.info(f"  Projection dim  : {proj_dim} (all branches)")
        logger.info(
            f"  Proj dropout    : "
            f"temporal={dropout_cfg.get('temporal', 0.1)}, "
            f"spatial={dropout_cfg.get('spatial', 0.2)}, "
            f"frequency={dropout_cfg.get('frequency', 0.2)}"
        )
        logger.info(f"  Fused dim       : {fused_dim} ({n_active} active × {proj_dim})")

    def _freeze_all_except_layernorms(self, module: nn.Module) -> Tuple[int, int]:
        for p in module.parameters():
            p.requires_grad = False
        ln_types = (nn.LayerNorm, nn.GroupNorm)
        if hasattr(nn, 'RMSNorm'):
            ln_types = ln_types + (nn.RMSNorm,)
        unfrozen = 0
        for mod in module.modules():
            if isinstance(mod, ln_types):
                for p in mod.parameters(recurse=False):
                    p.requires_grad = True
                    unfrozen += p.numel()
        total = sum(p.numel() for p in module.parameters())
        return total, unfrozen

    def _freeze_module(self, module: nn.Module, name: str) -> None:
        for p in module.parameters():
            p.requires_grad = False
        logger.info(f"  Frozen (inactive): {name}")

    def build_loss(self, config: dict) -> None:
        cw = config.get('class_weights', None)
        if cw is not None:
            weight = torch.tensor(cw, dtype=torch.float32)
            self.cls_loss = nn.CrossEntropyLoss(weight=weight)
            logger.info(f"  Class weights   : real={cw[0]}, fake={cw[1]}")
        else:
            self.cls_loss = nn.CrossEntropyLoss()
            logger.info("  Class weights   : none (uniform)")

        self.reg_loss = nn.MSELoss()
        self.l1_loss  = nn.L1Loss()

    # ------------------------------------------------------------------ #
    #  Feature extraction                                                  #
    # ------------------------------------------------------------------ #

    def _pool_and_project(
        self,
        frames: torch.Tensor,
        extractor: nn.Module,
        proj_head: nn.Module,
    ) -> torch.Tensor:
        """
        Flatten (B, T, C, H, W) → run extractor frame-by-frame → pool → project.

        Used for spatial and frequency branches whose extractors operate on
        individual frames (B×T, C, H, W) rather than full video clips.
        """
        B, T, C, H, W = frames.shape
        flat   = frames.view(B * T, C, H, W)
        feats  = extractor(flat)            # (B*T, feat_dim)
        feats  = feats.view(B, T, -1)
        pooled = feats.mean(dim=1)          # (B, feat_dim) — temporal mean pool
        return proj_head(pooled)            # (B, proj_dim)

    def extract_temporal_features(
        self,
        video_clip: torch.Tensor,
        return_frame_features: bool = False,
    ):
        """
        Run temporal extractor on a full video clip (B, T, C, H, W).

        Args:
            video_clip           : (B, T, C, H, W)
            return_frame_features: if True, also return (B, T, D) frame features
                                   for the temporal consistency loss.

        Returns:
            projected                  : (B, proj_dim)
            frame_feats (if requested) : (B, T, D)
        """
        if return_frame_features:
            raw, frame_feats = self.temporal_extractor(
                video_clip, return_frame_features=True)
            projected = self.temporal_proj(raw)
            return projected, frame_feats
        else:
            raw       = self.temporal_extractor(video_clip)
            projected = self.temporal_proj(raw)
            return projected

    def extract_spatial_features(self, spatial_frames: torch.Tensor) -> torch.Tensor:
        return self._pool_and_project(
            spatial_frames, self.spatial_extractor, self.spatial_proj)

    def extract_frequency_features(self, freq_frames: torch.Tensor) -> torch.Tensor:
        return self._pool_and_project(
            freq_frames, self.frequency_extractor, self.frequency_proj)

    def features(self, data_dict: dict) -> tuple:
        """
        Extract features from all active branches and fuse them.

        Fix 5: All three branches are symmetrically gated by active_branches.
        data_dict keys are accessed via .get() with a clear error message so
        that missing keys in single-branch ablations produce actionable errors
        rather than silent None-pooling or AttributeErrors downstream.

        Returns:
            (fused_features, temporal_feat, spatial_feat, frequency_feat,
             temporal_frame_feats)
            Inactive branch features are None.
        """
        temporal_feat        = None
        temporal_frame_feats = None
        spatial_feat         = None
        frequency_feat       = None

        # ── Temporal branch ───────────────────────────────────────────────
        if 'temporal' in self.active_branches:
            clip = data_dict.get('temporal_clip')
            if clip is None:
                raise KeyError(
                    "active_branches includes 'temporal' but 'temporal_clip' "
                    "is missing from data_dict. Check your dataset's __getitem__."
                )
            tc_weight = self.config.get('temporal_consistency_weight', 0.0)
            if tc_weight > 0.0:
                temporal_feat, temporal_frame_feats = self.extract_temporal_features(
                    clip, return_frame_features=True)
            else:
                temporal_feat = self.extract_temporal_features(clip)

        # ── Spatial branch ────────────────────────────────────────────────
        if 'spatial' in self.active_branches:
            frames = data_dict.get('spatial_frames')
            if frames is None:
                raise KeyError(
                    "active_branches includes 'spatial' but 'spatial_frames' "
                    "is missing from data_dict. Check your dataset's __getitem__."
                )
            spatial_feat = self.extract_spatial_features(frames)

        # ── Frequency branch ──────────────────────────────────────────────
        if 'frequency' in self.active_branches:
            frames = data_dict.get('freq_frames')
            if frames is None:
                raise KeyError(
                    "active_branches includes 'frequency' but 'freq_frames' "
                    "is missing from data_dict. Check your dataset's __getitem__."
                )
            frequency_feat = self.extract_frequency_features(frames)

        fused_features = self.fusion(temporal_feat, spatial_feat, frequency_feat)
        return (fused_features, temporal_feat, spatial_feat, frequency_feat,
                temporal_frame_feats)

    def classifier(self, features: torch.Tensor) -> dict:
        return self.multitaskhead(features)

    # ------------------------------------------------------------------ #
    #  Temporal consistency loss                                           #
    #  NOTE: disabled for baseline (temporal_consistency_weight: 0.0)     #
    #  Re-enable after core branches validate.                             #
    # ------------------------------------------------------------------ #

    def _temporal_consistency_loss(
        self,
        frame_features: torch.Tensor,   # (B, T, D) — frame-level, already pooled
        labels: torch.Tensor,            # (B,)
    ) -> torch.Tensor:
        """
        Margin contrastive loss on pairwise cosine similarity of frame features.

        Real videos should have high temporal consistency (stable identity,
        smooth motion). Fake videos should have lower consistency (per-frame
        generation artifacts, temporal discontinuities, identity drift).

        Loss = max(0, margin − (mean_real_consistency − mean_fake_consistency))

        Zero when real consistency exceeds fake by at least `margin`.
        Never pushes fake consistency up or real consistency down.

        Args:
            frame_features: (B, T, D) — one D-dim vector per frame.
                Must be (B, T, D), not (B, N*T, D). TemporalFeatureExtractor
                returns the correctly pooled shape via _pool_to_frame_features().
            labels: (B,) — 0=real, 1=fake
        """
        device = frame_features.device
        B, T, D = frame_features.shape

        if T < 2:
            logger.debug(
                f"[TemporalConsistencyLoss] Skipping — only {T} frame(s). "
                f"Need T >= 2 for pairwise similarity."
            )
            return torch.zeros(1, device=device)

        # float32 for numerical stability — frame_features may be bf16/fp16 under AMP
        feats_f32  = frame_features.float()
        norm_feats = F.normalize(feats_f32, p=2, dim=-1)        # (B, T, D)
        sim_matrix = torch.bmm(
            norm_feats, norm_feats.transpose(1, 2))              # (B, T, T)

        # Off-diagonal mean (exclude self-similarity)
        off_diag = ~torch.eye(T, dtype=torch.bool, device=device).unsqueeze(0)
        consistency = (
            sim_matrix[off_diag.expand(B, T, T)]
            .view(B, T * (T - 1))
            .mean(dim=1)
        )   # (B,)

        real_mask = (labels == 0)
        fake_mask = (labels == 1)

        if real_mask.sum() == 0 or fake_mask.sum() == 0:
            # Batch is single-class — loss is undefined
            return torch.zeros(1, device=device)

        real_consistency = consistency[real_mask].mean()
        fake_consistency = consistency[fake_mask].mean()

        margin = self.config.get('temporal_consistency_margin', 0.1)
        return torch.clamp(
            margin - (real_consistency - fake_consistency), min=0.0)

    # ------------------------------------------------------------------ #
    #  Forward pass                                                        #
    # ------------------------------------------------------------------ #

    def forward(self, data_dict: dict, inference: bool = False) -> dict:
        device = data_dict['label'].device

        (fused_features, temporal_feat, spatial_feat,
         frequency_feat, temporal_frame_feats) = self.features(data_dict)

        # ── Semantic grounding (disabled for baseline) ────────────────────
        semantic_concepts = None
        grounded_features = fused_features
        if self.use_semantic_grounding:
            T   = data_dict['raw_frames'].shape[1]
            mid = T // 2
            raw_frame_anchor = data_dict['raw_frames'][:, mid]
            grounded_features, semantic_concepts = self.semantic_grounding(
                fused_features, raw_frame_anchor)

        # ── Causal module (disabled for baseline) ─────────────────────────
        violation_score = None
        causal_dag      = None
        causal_concepts = None
        if self.use_causal:
            if not inference:
                violation_score, causal_dag, causal_concepts = self.causal_module(
                    grounded_features,
                    explicit_concepts=semantic_concepts,
                    return_graph=True,
                )
            else:
                violation_score = self.causal_module(
                    grounded_features, explicit_concepts=semantic_concepts)

        # ── Sparse autoencoder (disabled for baseline) ────────────────────
        sparse_loss = None
        if self.use_sparse:
            sparse_features, sparse_loss = self.sparse_ae(grounded_features)
            classifier_input = sparse_features
        else:
            classifier_input = grounded_features

        task_outputs = self.classifier(classifier_input)
        cls_logits   = task_outputs['classification']
        prob         = torch.softmax(cls_logits, dim=1)[:, 1]

        pred_dict = {
            'cls':                     cls_logits,
            'prob':                    prob,
            'feat':                    fused_features,
            'grounded_feat':           grounded_features,
            'temporal_feat':           temporal_feat,
            'spatial_feat':            spatial_feat,
            'frequency_feat':          frequency_feat,
            'temporal_frame_features': temporal_frame_feats,
            'semantic_concepts':       semantic_concepts,
            'causal_concepts':         causal_concepts,
            'uncertainty':             task_outputs.get('uncertainty'),
            'violation_score':         task_outputs.get('violation_score'),
            'task_outputs':            task_outputs,
            'sparse_loss':             sparse_loss,
            'causal_dag':              causal_dag,
        }
        return pred_dict

    # ------------------------------------------------------------------ #
    #  Loss computation                                                    #
    # ------------------------------------------------------------------ #

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label  = data_dict['label']
        device = label.device

        # Fix 4: move class-weight tensor to GPU/correct dtype on first call
        if (hasattr(self.cls_loss, 'weight')
                and self.cls_loss.weight is not None):
            target_dtype = pred_dict['cls'].dtype
            if (self.cls_loss.weight.device != device
                    or self.cls_loss.weight.dtype != target_dtype):
                self.cls_loss.weight = self.cls_loss.weight.to(
                    device=device, dtype=target_dtype)

        cls_loss = self.cls_loss(pred_dict['cls'], label)

        # ── Uncertainty loss ──────────────────────────────────────────────
        uncertainty_loss = torch.zeros(1, device=device)
        if pred_dict.get('uncertainty') is not None:
            pred_label       = pred_dict['cls'].argmax(dim=1)
            is_correct       = (pred_label == label).float()
            uncertainty_loss = self.reg_loss(
                pred_dict['uncertainty'].squeeze(), 1 - is_correct)

        # ── Causal loss ───────────────────────────────────────────────────
        causal_loss = torch.zeros(1, device=device)
        if self.use_causal and pred_dict.get('violation_score') is not None:
            causal_loss = self.reg_loss(
                pred_dict['violation_score'], label.float())
            if pred_dict.get('causal_dag') is not None:
                dag_penalty = self.causal_module.causal_learner.compute_dag_penalty()
                causal_loss = causal_loss + (
                    self.config['causal_module']['dag_learning']['dag_penalty_weight']
                    * dag_penalty
                )

        # ── Sparse loss ───────────────────────────────────────────────────
        sparse_loss = pred_dict.get('sparse_loss')
        if sparse_loss is None or not isinstance(sparse_loss, torch.Tensor):
            sparse_loss = torch.zeros(1, device=device)

        # ── Temporal consistency loss (Fix 3 — off by default) ───────────
        # temporal_consistency_weight: 0.0 in baseline config.
        # Set >0 only after core branches validate on FF++ + Celeb-DF-v2.
        temporal_consistency_loss = torch.zeros(1, device=device)
        tc_weight = self.config.get('temporal_consistency_weight', 0.0)
        if (tc_weight > 0.0
                and 'temporal' in self.active_branches
                and pred_dict.get('temporal_frame_features') is not None):
            temporal_consistency_loss = self._temporal_consistency_loss(
                pred_dict['temporal_frame_features'], label)

        # ── DDP anchor — keeps inactive branch params in the compute graph ─
        # Required for DDP's find_unused_parameters=True. Without this,
        # inactive branch params have no gradient and DDP raises an error.
        ddp_anchor = torch.zeros(1, device=device)
        branch_pairs = {
            'temporal':  (self.temporal_extractor,  self.temporal_proj),
            'spatial':   (self.spatial_extractor,   self.spatial_proj),
            'frequency': (self.frequency_extractor, self.frequency_proj),
        }
        for name, (extractor, proj) in branch_pairs.items():
            if name not in self.active_branches:
                ddp_anchor = ddp_anchor + _zero_grad_anchor(extractor, device)
                ddp_anchor = ddp_anchor + _zero_grad_anchor(proj, device)

        for attr in ('causal_module', 'sparse_ae', 'semantic_grounding'):
            mod = getattr(self, attr, None)
            if mod is not None:
                ddp_anchor = ddp_anchor + _zero_grad_anchor(mod, device)

        # ── Total loss ────────────────────────────────────────────────────
        total_loss = (
            self.loss_weights['classification']                 * cls_loss
            + self.loss_weights.get('uncertainty',    0.0)     * uncertainty_loss
            + self.loss_weights.get('causal',         0.0)     * causal_loss
            + self.loss_weights.get('sparse',         0.0)     * sparse_loss
            + tc_weight                                         * temporal_consistency_loss
            + ddp_anchor
        )

        def _scalar(t):
            return t.squeeze() if isinstance(t, torch.Tensor) else t

        return {
            'overall':              _scalar(total_loss),
            'classification':       _scalar(cls_loss),
            'uncertainty':          _scalar(uncertainty_loss),
            'causal':               _scalar(causal_loss),
            'sparse':               _scalar(sparse_loss),
            'temporal_consistency': _scalar(temporal_consistency_loss),
        }

    # ------------------------------------------------------------------ #
    #  Metrics                                                             #
    # ------------------------------------------------------------------ #

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred  = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(
            label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        if pred_dict.get('uncertainty') is not None:
            uncertainty = pred_dict['uncertainty'].detach().cpu().numpy()
            metric_batch_dict['mean_uncertainty'] = float(uncertainty.mean())
        self.video_names = []
        return metric_batch_dict