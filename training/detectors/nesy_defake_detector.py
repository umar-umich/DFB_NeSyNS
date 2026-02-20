"""
detectors/nesy_defake_hybrid_detector.py
=========================================
REGULARIZATION FIXES APPLIED (this revision):

  Fix 1 — Class-weighted CrossEntropyLoss
           Addresses acc_real ≈ 0 across all three branches.
           FF++ is ~80% fake so unweighted CE has a degenerate local optimum
           of "predict fake always". Class weights break this early.
           Config: class_weights: [2.0, 1.0]  # [real, fake]

  Fix 2 — Dropout in projection heads
           Spatial was overfitting hard (train loss → 0.002 while test loss → 0.9).
           Per-branch dropout: spatial=0.2, temporal=0.1, frequency=0.1
           Config: projection_dropout: {spatial: 0.2, temporal: 0.1, frequency: 0.1}

  Fix 3 — Temporal consistency loss
           Pushes temporal branch to learn motion coherence, not just
           "does this look like a face". Real videos should have more
           consistent frame-level features than fake videos.
           Config: temporal_consistency_weight: 0.1
                   temporal_consistency_margin: 0.1

  Fix 4 — class weight device move in get_losses()
           CrossEntropyLoss.weight is created on CPU. Without explicit
           .to(device), it crashes on first forward pass on GPU.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    anchor = torch.zeros(1, device=device, dtype=torch.float32)
    for p in module.parameters():
        if p.requires_grad:
            anchor = anchor + p.sum() * 0.0
    return anchor


def _make_projection(in_dim: int, out_dim: int, dropout: float = 0.0) -> nn.Sequential:
    """
    Learnable branch projection head: Linear → LayerNorm → GELU → Dropout.

    Dropout is placed post-activation so it gates final projected features
    rather than disrupting LayerNorm statistics. 0.0 = no dropout (default).

    Per-branch recommendations based on observed overfitting:
        spatial   : 0.2  (overfits fastest — most iterations, highest-capacity backbone)
        temporal  : 0.1  (moderate)
        frequency : 0.1  (moderate)
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

    def build_backbone(self, config: dict) -> None:
        self.temporal_extractor  = TemporalFeatureExtractor(config)
        self.spatial_extractor   = SpatialFeatureExtractor(config)
        self.frequency_extractor = FrequencyFeatureExtractor(config)

        active = set(config.get('active_branches', list(ALL_BRANCHES)))
        self.active_branches = active

        fm       = config['foundation_models']
        proj_dim = config['fusion']['projection_dim']

        branch_dims = {
            'temporal':  self.temporal_extractor.output_dim,   # read from extractor
            'spatial':   fm['spatial']['output_dim'],          # CLIP-L: 1024
            'frequency': self.frequency_extractor.output_dim,  # read from extractor
        }

        # Per-branch dropout from config
        dropout_cfg = config.get('projection_dropout', {})
        self.temporal_proj  = _make_projection(
            branch_dims['temporal'],  proj_dim,
            dropout=dropout_cfg.get('temporal',  0.1))
        self.spatial_proj   = _make_projection(
            branch_dims['spatial'],   proj_dim,
            dropout=dropout_cfg.get('spatial',   0.2))
        self.frequency_proj = _make_projection(
            branch_dims['frequency'], proj_dim,
            dropout=dropout_cfg.get('frequency', 0.1))

        fused_dim = proj_dim * sum(1 for b in ALL_BRANCHES if b in active)
        config['fusion']['fused_dim'] = fused_dim

        logger.info(
            f"  Branch dims     : temporal={branch_dims['temporal']}, "
            f"spatial={branch_dims['spatial']}, frequency={branch_dims['frequency']}"
        )
        logger.info(f"  Projection dim  : {proj_dim} (all branches)")
        logger.info(
            f"  Proj dropout    : temporal={dropout_cfg.get('temporal', 0.1)}, "
            f"spatial={dropout_cfg.get('spatial', 0.2)}, "
            f"frequency={dropout_cfg.get('frequency', 0.1)}"
        )
        logger.info(
            f"  Fused dim       : {fused_dim} "
            f"(active: {sorted(active)}, {len(active)} × {proj_dim})"
        )

        for name in ALL_BRANCHES:
            if name not in active:
                self._freeze_module(
                    getattr(self, f'{name}_extractor'), name)

    def _freeze_module(self, module: nn.Module, name: str) -> None:
        for p in module.parameters():
            p.requires_grad = False
        logger.info(f"  Frozen branch   : {name}")

    def build_loss(self, config: dict) -> None:
        """
        CrossEntropyLoss with optional class weights (Fix 1).

        Config key: class_weights: [real_weight, fake_weight]
        Recommended starting point: [2.0, 1.0]
        Theoretical value for FF++ (~80% fake): [4.0, 1.0]

        Start with [2.0, 1.0]. If acc_real is still near 0 after 5 epochs,
        increase to [3.0, 1.0] or [4.0, 1.0].
        """
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
        B, T, C, H, W = frames.shape
        flat   = frames.view(B * T, C, H, W)
        feats  = extractor(flat)
        feats  = feats.view(B, T, -1)
        pooled = feats.mean(dim=1)
        return proj_head(pooled)

    def extract_temporal_features(
        self,
        video_clip: torch.Tensor,
        return_frame_features: bool = False,
    ) -> torch.Tensor:
        """
        Run temporal extractor. If return_frame_features=True, also returns
        per-frame/token features for the temporal consistency loss.
        """
        if return_frame_features:
            raw, frame_feats = self.temporal_extractor(
                video_clip, return_frame_features=True)
        else:
            raw = self.temporal_extractor(video_clip)
            frame_feats = None

        projected = self.temporal_proj(raw)
        if return_frame_features:
            return projected, frame_feats
        return projected

    def extract_spatial_features(self, spatial_frames: torch.Tensor) -> torch.Tensor:
        return self._pool_and_project(
            spatial_frames, self.spatial_extractor, self.spatial_proj)

    def extract_frequency_features(self, freq_frames: torch.Tensor) -> torch.Tensor:
        return self._pool_and_project(
            freq_frames, self.frequency_extractor, self.frequency_proj)

    def features(self, data_dict: dict) -> tuple:
        temporal_feat        = None
        temporal_frame_feats = None

        if 'temporal' in self.active_branches:
            use_consistency = (
                self.config.get('temporal_consistency_weight', 0.0) > 0
            )
            if use_consistency:
                temporal_feat, temporal_frame_feats = self.extract_temporal_features(
                    data_dict['temporal_clip'], return_frame_features=True)
            else:
                temporal_feat = self.extract_temporal_features(
                    data_dict['temporal_clip'])

        spatial_feat = (
            self.extract_spatial_features(data_dict['spatial_frames'])
            if 'spatial' in self.active_branches else None
        )
        frequency_feat = (
            self.extract_frequency_features(data_dict['freq_frames'])
            if 'frequency' in self.active_branches else None
        )

        fused_features = self.fusion(temporal_feat, spatial_feat, frequency_feat)
        return (fused_features, temporal_feat, spatial_feat, frequency_feat,
                temporal_frame_feats)

    def classifier(self, features: torch.Tensor) -> dict:
        return self.multitaskhead(features)

    # ------------------------------------------------------------------ #
    #  Temporal consistency loss (Fix 3)                                  #
    # ------------------------------------------------------------------ #

    def _temporal_consistency_loss(
        self,
        frame_features: torch.Tensor,   # (B, N, D)
        labels: torch.Tensor,            # (B,)
    ) -> torch.Tensor:
        """
        Margin contrastive loss on pairwise cosine similarity of frame features.

        Real videos should have high temporal consistency (features are similar
        across frames — stable identity, smooth motion).
        Fake videos should have lower consistency (per-frame generation artifacts,
        temporal discontinuities, identity drift between frames).

        Loss = max(0, margin − (mean_real_consistency − mean_fake_consistency))

        This is zero when real consistency exceeds fake consistency by at least
        `margin`, and positive otherwise. It never pushes fake consistency up
        or real consistency down — it only enforces the gap.
        """
        device = frame_features.device

        norm_feats = F.normalize(frame_features, p=2, dim=-1)   # (B, N, D)
        sim_matrix = torch.bmm(
            norm_feats, norm_feats.transpose(1, 2))             # (B, N, N)

        B, N, _ = sim_matrix.shape
        # Off-diagonal mask — exclude self-similarity
        off_diag = ~torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)
        consistency = (
            sim_matrix[off_diag.expand(B, N, N)]
            .view(B, N * (N - 1))
            .mean(dim=1)
        )   # (B,)

        real_mask = (labels == 0)
        fake_mask = (labels == 1)

        if real_mask.sum() == 0 or fake_mask.sum() == 0:
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

        semantic_concepts = None
        grounded_features = fused_features
        if self.use_semantic_grounding:
            T   = data_dict['raw_frames'].shape[1]
            mid = T // 2
            raw_frame_anchor = data_dict['raw_frames'][:, mid]
            grounded_features, semantic_concepts = self.semantic_grounding(
                fused_features, raw_frame_anchor)

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
            'cls':                    cls_logits,
            'prob':                   prob,
            'feat':                   fused_features,
            'grounded_feat':          grounded_features,
            'temporal_feat':          temporal_feat,
            'spatial_feat':           spatial_feat,
            'frequency_feat':         frequency_feat,
            'temporal_frame_features': temporal_frame_feats,
            'semantic_concepts':      semantic_concepts,
            'causal_concepts':        causal_concepts,
            'uncertainty':            task_outputs.get('uncertainty'),
            'violation_score':        task_outputs.get('violation_score'),
            'task_outputs':           task_outputs,
            'sparse_loss':            sparse_loss,
            'causal_dag':             causal_dag,
        }
        return pred_dict

    # ------------------------------------------------------------------ #
    #  Loss computation                                                    #
    # ------------------------------------------------------------------ #

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label  = data_dict['label']
        device = label.device

        # Fix 4: move class weights to GPU on first call (they are created on CPU)
        if (hasattr(self.cls_loss, 'weight')
            and self.cls_loss.weight is not None):
            target_dtype = pred_dict['cls'].dtype
            if (self.cls_loss.weight.device != device
                    or self.cls_loss.weight.dtype != target_dtype):
                self.cls_loss.weight = self.cls_loss.weight.to(
                    device=device, dtype=target_dtype)

        cls_loss = self.cls_loss(pred_dict['cls'], label)

        # Uncertainty loss
        uncertainty_loss = torch.zeros(1, device=device)
        if pred_dict.get('uncertainty') is not None:
            pred_label = pred_dict['cls'].argmax(dim=1)
            is_correct = (pred_label == label).float()
            uncertainty_loss = self.reg_loss(
                pred_dict['uncertainty'].squeeze(), 1 - is_correct)

        # Causal loss
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

        # Sparse loss
        sparse_loss = pred_dict.get('sparse_loss')
        if sparse_loss is None or not isinstance(sparse_loss, torch.Tensor):
            sparse_loss = torch.zeros(1, device=device)

        # Fix 3: Temporal consistency loss
        temporal_consistency_loss = torch.zeros(1, device=device)
        tc_weight = self.config.get('temporal_consistency_weight', 0.0)
        if (tc_weight > 0
                and 'temporal' in self.active_branches
                and pred_dict.get('temporal_frame_features') is not None):
            temporal_consistency_loss = self._temporal_consistency_loss(
                pred_dict['temporal_frame_features'], label)

        # DDP anchor
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

        for attr in ('causal_module', 'sparse_ae', 'semantic_grounding',
                     'multitaskhead'):
            mod = getattr(self, attr, None)
            if mod is not None:
                ddp_anchor = ddp_anchor + _zero_grad_anchor(mod, device)

        total_loss = (
            self.loss_weights['classification']                     * cls_loss
            + self.loss_weights.get('uncertainty',        0.0)     * uncertainty_loss
            + self.loss_weights.get('causal',             0.0)     * causal_loss
            + self.loss_weights.get('sparse',             0.0)     * sparse_loss
            + tc_weight                                             * temporal_consistency_loss
            + ddp_anchor
        )

        return {
            'overall':              total_loss,
            'classification':       cls_loss,
            'uncertainty':          uncertainty_loss,
            'causal':               causal_loss,
            'sparse':               sparse_loss,
            'temporal_consistency': temporal_consistency_loss,
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