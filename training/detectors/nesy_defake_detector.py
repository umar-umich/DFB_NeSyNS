"""
detectors/nesy_defake_hybrid_detector.py  — STEP 3 REVISION
============================================================
Changes from previous version:

  1. INPUT SHAPES (aligned with Step 1 dataset rewrite):
       temporal_clip  : (B, T=32, C, H, W)  — unchanged, VJEPA2 handles natively
       spatial_frames : (B, T=32, C, H, W)  — was (B, C, H, W) single frame
       freq_frames    : (B, T=32, C, H, W)  — was (B, C, H, W) single frame
       raw_frames     : (B, T=32, C, H, W)  — was (B, C, H, W) single frame

  2. DICT KEY RENAMES (matching dataset Step 1 output keys):
       data_dict['spatial_frame']   → data_dict['spatial_frames']
       data_dict['frequency_frame'] → data_dict['freq_frames']
       data_dict['raw_frame']       → data_dict['raw_frames']

  3. LEARNABLE PER-BRANCH PROJECTION HEADS (Step 3 core addition):
       Each branch gets its own nn.Sequential(Linear, LayerNorm, GELU) that
       maps raw_dim → projection_dim independently.
       These heads are always trainable, even when freeze_backbone=True,
       because they are new modules on top of the frozen foundation model.
       Branch dims: temporal=1024, spatial=1024, frequency=512 → all → 1024

  4. TEMPORAL POOLING for spatial + frequency branches:
       Spatial/frequency extractors process frames independently as image
       encoders, so they receive (B*T, C, H, W) and output (B*T, raw_dim).
       We reshape to (B, T, raw_dim), mean-pool over T → (B, raw_dim),
       then apply the learnable projection → (B, projection_dim).
       Temporal branch already outputs (B, temporal_dim) from its video
       encoder — just project directly.

  5. FUSION INPUT CHANGE:
       MultiModalFusion now receives projected features (all projection_dim)
       instead of raw heterogeneous dims.
       fused_dim = projection_dim × num_active_branches (patched at runtime).

  6. ABLATION SAFETY preserved:
       - All three extractors + projections always instantiated.
       - Inactive branches return None (skipped in forward, touched by DDP anchor).
       - _freeze_module() / DDP anchor / active_branches gate all intact.

  7. REMOVED dead attrs:
       self.num_clips (num_clips_per_video — no longer meaningful)
       self.clip_size (clip_size — now always == frame_num == 32)
"""

import logging
import torch
import torch.nn as nn

from metrics.base_metrics_class import calculate_metrics_for_train
from .base_detector import AbstractDetector
from detectors import DETECTOR

from networks.nesy_defake.foundation_models import (
    TemporalFeatureExtractor,
    FrequencyFeatureExtractor,
    SpatialFeatureExtractor,
)
from networks.nesy_defake.fusion import MultiModalFusion
from networks.nesy_defake.causal import CausalDiscoveryModule
from networks.nesy_defake.classifiers import MultiTaskHead, SparseAutoencoder
from utils.semantic_grounding import SemanticGroundingModule

logger = logging.getLogger(__name__)
ALL_BRANCHES = ('temporal', 'spatial', 'frequency')


def _zero_grad_anchor(module: nn.Module, device: torch.device) -> torch.Tensor:
    """
    Touch every parameter in an inactive module with a ×0 multiply so DDP
    sees a gradient path to all parameters and does not hang on barrier.
    """
    anchor = torch.zeros(1, device=device, dtype=torch.float32)
    for p in module.parameters():
        if p.requires_grad:
            anchor = anchor + p.sum() * 0.0
    return anchor


def _make_projection(in_dim: int, out_dim: int) -> nn.Sequential:
    """
    Learnable branch projection head: Linear → LayerNorm → GELU.
    Maps raw foundation-model output dim to the common projection_dim.
    Always trainable regardless of freeze_backbone setting.
    """
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.LayerNorm(out_dim),
        nn.GELU(),
    )


@DETECTOR.register_module(module_name='nesydefake_hybrid')
class NeSyDeFakeHybridDetector(AbstractDetector):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # ── Backbone + learnable projections ──────────────────────────────
        # Also patches config['fusion']['fused_dim'] for MultiModalFusion.
        self.build_backbone(config)

        # ── Fusion ────────────────────────────────────────────────────────
        self.fusion = MultiModalFusion(config)

        # ── Semantic Grounding ────────────────────────────────────────────
        self.use_semantic_grounding = config.get(
            'semantic_grounding', {}).get('enabled', False)
        if self.use_semantic_grounding:
            self.semantic_grounding = SemanticGroundingModule(
                config,
                feature_dim=config['fusion']['projection_dim'],
            )
            self.grounded_feature_dim = self.semantic_grounding.output_dim
        else:
            self.grounded_feature_dim = config['fusion']['projection_dim']

        # ── Causal Discovery ──────────────────────────────────────────────
        self.use_causal = config['causal_module']['enabled']
        if self.use_causal:
            self.causal_module = CausalDiscoveryModule(config)

        # ── Sparse Features ───────────────────────────────────────────────
        self.use_sparse = config['sparse_features']['enabled']
        if self.use_sparse:
            self.sparse_ae = SparseAutoencoder(config)

        # ── Multi-task Classifier ─────────────────────────────────────────
        self.multitaskhead = MultiTaskHead(config)

        # ── Loss ──────────────────────────────────────────────────────────
        self.loss_weights = config['loss_func']['weights']
        self.build_loss(config)

        logger.info("NeSyDeFake Hybrid Detector initialised (Step 3 — learnable projections)")
        logger.info(f"  Active branches  : {sorted(self.active_branches)}")
        logger.info(f"  Fused dim        : {config['fusion']['fused_dim']}")
        logger.info(f"  Projection dim   : {config['fusion']['projection_dim']}")

    # ------------------------------------------------------------------ #
    #  Construction helpers                                                #
    # ------------------------------------------------------------------ #

    def build_backbone(self, config: dict) -> None:
        """
        Instantiate all three foundation-model extractors plus a learnable
        per-branch projection head for each.

        Projection heads map each branch's raw output dimension to the shared
        projection_dim so all branches live in the same feature space before
        fusion. They are always trainable even when freeze_backbone=True.

        Also patches config['fusion']['fused_dim'] at runtime so
        MultiModalFusion always receives the correct concatenated dimension.

        Input/output shapes after this step:
            temporal  : (B, T, C, H, W) → extractor → (B, t_dim)
                        → temporal_proj  → (B, proj_dim)
            spatial   : (B, T, C, H, W) → flatten to (B*T, C, H, W)
                        → extractor      → (B*T, s_dim)
                        → mean-pool T    → (B, s_dim)
                        → spatial_proj   → (B, proj_dim)
            frequency : (B, T, C, H, W) → flatten to (B*T, C, H, W)
                        → extractor      → (B*T, f_dim)
                        → mean-pool T    → (B, f_dim)
                        → frequency_proj → (B, proj_dim)
        """
        # ── Foundation model extractors ───────────────────────────────────
        self.temporal_extractor  = TemporalFeatureExtractor(config)
        self.spatial_extractor   = SpatialFeatureExtractor(config)
        self.frequency_extractor = FrequencyFeatureExtractor(config)

        active = set(config.get('active_branches', list(ALL_BRANCHES)))
        self.active_branches = active

        # ── Per-branch raw output dims ─────────────────────────────────────
        fm = config['foundation_models']
        branch_dims = {
            'temporal':  fm['temporal']['output_dim'],    # e.g. 1024
            'spatial':   fm['spatial']['output_dim'],     # e.g. 1024
            'frequency': fm['frequency']['output_dim'],   # e.g.  512
        }
        proj_dim = config['fusion']['projection_dim']     # e.g. 1024

        # ── Learnable per-branch projection heads ─────────────────────────
        # Always created for all branches — inactive ones are touched by the
        # DDP anchor so distributed training never hangs.
        self.temporal_proj  = _make_projection(branch_dims['temporal'],  proj_dim)
        self.spatial_proj   = _make_projection(branch_dims['spatial'],   proj_dim)
        self.frequency_proj = _make_projection(branch_dims['frequency'], proj_dim)

        # ── Patch fused_dim: all active branches now output proj_dim ─────
        # fused_dim = proj_dim × number of active branches (all same dim now)
        fused_dim = proj_dim * sum(1 for b in ALL_BRANCHES if b in active)
        config['fusion']['fused_dim'] = fused_dim

        logger.info(
            f"  Branch raw dims  : temporal={branch_dims['temporal']}, "
            f"spatial={branch_dims['spatial']}, frequency={branch_dims['frequency']}"
        )
        logger.info(
            f"  Projection dim   : {proj_dim} (all branches → same dim)"
        )
        logger.info(
            f"  Computed fused_dim={fused_dim} "
            f"(active: {sorted(active)}, {len(active)} × {proj_dim})"
        )

        # ── Freeze inactive foundation-model extractors ───────────────────
        # Projection heads are intentionally NOT frozen — they must train.
        branch_extractors = {
            'temporal':  self.temporal_extractor,
            'spatial':   self.spatial_extractor,
            'frequency': self.frequency_extractor,
        }
        for name, module in branch_extractors.items():
            if name not in active:
                self._freeze_module(module, name)

    def _freeze_module(self, module: nn.Module, name: str) -> None:
        for p in module.parameters():
            p.requires_grad = False
        logger.info(f"  Frozen branch    : {name}")

    def build_loss(self, config: dict) -> None:
        self.cls_loss = nn.CrossEntropyLoss()
        self.reg_loss = nn.MSELoss()
        self.l1_loss  = nn.L1Loss()

    # ------------------------------------------------------------------ #
    #  Feature extraction                                                  #
    # ------------------------------------------------------------------ #

    def _pool_and_project(
        self,
        frames: torch.Tensor,         # (B, T, C, H, W)
        extractor: nn.Module,
        proj_head: nn.Module,
    ) -> torch.Tensor:
        """
        Run a frame-level extractor over all T frames then mean-pool and project.

        Steps:
            1. Flatten (B, T, C, H, W) → (B*T, C, H, W)
            2. Run extractor            → (B*T, raw_dim)
            3. Reshape                  → (B, T, raw_dim)
            4. Mean-pool over T         → (B, raw_dim)
            5. Learnable projection     → (B, proj_dim)

        The mean-pool is differentiable, so gradients flow back through
        all T frame representations into the projection head.
        """
        B, T, C, H, W = frames.shape
        flat   = frames.view(B * T, C, H, W)          # (B*T, C, H, W)
        feats  = extractor(flat)                        # (B*T, raw_dim)
        feats  = feats.view(B, T, -1)                  # (B, T, raw_dim)
        pooled = feats.mean(dim=1)                      # (B, raw_dim)
        return proj_head(pooled)                        # (B, proj_dim)

    def extract_temporal_features(
        self, video_clip: torch.Tensor               # (B, T, C, H, W)
    ) -> torch.Tensor:                               # (B, proj_dim)
        """
        Temporal extractor (VJEPA2 / VideoMAE) natively handles (B, T, C, H, W)
        and outputs (B, temporal_dim). We just apply the projection head.
        """
        raw = self.temporal_extractor(video_clip)    # (B, temporal_dim)
        return self.temporal_proj(raw)               # (B, proj_dim)

    def extract_spatial_features(
        self, spatial_frames: torch.Tensor           # (B, T, C, H, W)
    ) -> torch.Tensor:                               # (B, proj_dim)
        """
        Spatial extractor (GenD / DINOv2 / CLIP) is an image encoder.
        We process all T frames independently then mean-pool over T.
        """
        return self._pool_and_project(
            spatial_frames, self.spatial_extractor, self.spatial_proj)

    def extract_frequency_features(
        self, freq_frames: torch.Tensor              # (B, T, C, H, W)
    ) -> torch.Tensor:                               # (B, proj_dim)
        """
        Frequency extractor (SPSL / Xception) is also an image encoder.
        Same pool-and-project pattern as spatial.
        """
        return self._pool_and_project(
            freq_frames, self.frequency_extractor, self.frequency_proj)

    def features(self, data_dict: dict) -> tuple:
        """
        Run only active branches. Inactive branches return None so
        MultiModalFusion skips them cleanly.

        All active branches output (B, proj_dim) — same dimension.

        Returns
        -------
        fused_features : (B, fused_dim)  where fused_dim = proj_dim × active_count
        temporal_feat  : (B, proj_dim) or None
        spatial_feat   : (B, proj_dim) or None
        frequency_feat : (B, proj_dim) or None
        """
        temporal_feat = (
            self.extract_temporal_features(data_dict['temporal_clip'])
            if 'temporal' in self.active_branches else None
        )
        spatial_feat = (
            self.extract_spatial_features(data_dict['spatial_frames'])
            if 'spatial' in self.active_branches else None
        )
        frequency_feat = (
            self.extract_frequency_features(data_dict['freq_frames'])
            if 'frequency' in self.active_branches else None
        )

        fused_features = self.fusion(temporal_feat, spatial_feat, frequency_feat)
        return fused_features, temporal_feat, spatial_feat, frequency_feat

    def classifier(self, features: torch.Tensor) -> dict:
        return self.multitaskhead(features)

    # ------------------------------------------------------------------ #
    #  Forward pass                                                        #
    # ------------------------------------------------------------------ #

    def forward(self, data_dict: dict, inference: bool = False) -> dict:
        device = data_dict['label'].device

        fused_features, temporal_feat, spatial_feat, frequency_feat = \
            self.features(data_dict)

        # ── Semantic Grounding ────────────────────────────────────────────
        # raw_frames is (B, T, C, H, W) — SemanticGrounding should use
        # the middle frame or handle the temporal dim internally.
        semantic_concepts = None
        grounded_features = fused_features
        if self.use_semantic_grounding:
            # Pass middle frame: (B, C, H, W) for compatibility with current
            # SemanticGroundingModule which expects a single frame per sample.
            T = data_dict['raw_frames'].shape[1]
            mid = T // 2
            raw_frame_anchor = data_dict['raw_frames'][:, mid]   # (B, C, H, W)
            grounded_features, semantic_concepts = self.semantic_grounding(
                fused_features,
                raw_frame_anchor,
            )

        # ── Causal Discovery ──────────────────────────────────────────────
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
                    grounded_features,
                    explicit_concepts=semantic_concepts,
                )

        # ── Sparse Features ───────────────────────────────────────────────
        sparse_loss = None
        if self.use_sparse:
            sparse_features, sparse_loss = self.sparse_ae(grounded_features)
            classifier_input = sparse_features
        else:
            classifier_input = grounded_features

        # ── Classification ────────────────────────────────────────────────
        task_outputs = self.classifier(classifier_input)
        cls_logits   = task_outputs['classification']
        prob         = torch.softmax(cls_logits, dim=1)[:, 1]

        pred_dict = {
            'cls':               cls_logits,
            'prob':              prob,
            'feat':              fused_features,
            'grounded_feat':     grounded_features,
            'temporal_feat':     temporal_feat,
            'spatial_feat':      spatial_feat,
            'frequency_feat':    frequency_feat,
            'semantic_concepts': semantic_concepts,
            'causal_concepts':   causal_concepts,
            'uncertainty':       task_outputs.get('uncertainty'),
            'violation_score':   task_outputs.get('violation_score'),
            'task_outputs':      task_outputs,
            'sparse_loss':       sparse_loss,
            'causal_dag':        causal_dag,
        }
        return pred_dict

    # ------------------------------------------------------------------ #
    #  Loss computation                                                    #
    # ------------------------------------------------------------------ #

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label  = data_dict['label']
        device = label.device

        cls_loss = self.cls_loss(pred_dict['cls'], label)

        uncertainty_loss = torch.zeros(1, device=device)
        if pred_dict.get('uncertainty') is not None:
            pred_label = pred_dict['cls'].argmax(dim=1)
            is_correct = (pred_label == label).float()
            uncertainty_loss = self.reg_loss(
                pred_dict['uncertainty'].squeeze(), 1 - is_correct)

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

        sparse_loss = pred_dict.get('sparse_loss')
        if sparse_loss is None or not isinstance(sparse_loss, torch.Tensor):
            sparse_loss = torch.zeros(1, device=device)

        # ── DDP anchor ────────────────────────────────────────────────────
        # Touch ALL branch params (extractors + projection heads) regardless
        # of active_branches so DDP never complains about unused parameters.
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

        if self.use_causal:
            ddp_anchor = ddp_anchor + _zero_grad_anchor(self.causal_module, device)
        if self.use_sparse:
            ddp_anchor = ddp_anchor + _zero_grad_anchor(self.sparse_ae, device)
        if self.use_semantic_grounding:
            ddp_anchor = ddp_anchor + _zero_grad_anchor(self.semantic_grounding, device)
        ddp_anchor = ddp_anchor + _zero_grad_anchor(self.multitaskhead, device)

        total_loss = (
            self.loss_weights['classification']          * cls_loss
            + self.loss_weights.get('uncertainty', 0.0) * uncertainty_loss
            + self.loss_weights.get('causal',      0.0) * causal_loss
            + self.loss_weights.get('sparse',      0.0) * sparse_loss
            + ddp_anchor
        )

        return {
            'overall':        total_loss,
            'classification': cls_loss,
            'uncertainty':    uncertainty_loss,
            'causal':         causal_loss,
            'sparse':         sparse_loss,
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