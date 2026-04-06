"""
detectors/nesy_defake_hybrid_detector.py
=========================================
END-TO-END PER-BRANCH DUAL-GRAPH CAUSAL DISCOVERY (v4 — cleaned up)

Data flow:
  spatial_frames --> SpatialExtractor --> raw_spatial (1024-d) --+-> spatial_proj -> fused -> classifier
                                                                  +-> SAE.spatial -> z_spatial --+
  freq_frames ----> FreqExtractor ----> raw_freq (1024-d) ------+-> freq_proj -> fused -> classifier
                                                                  +-> SAE.freq -> z_freq --------+
  raw_frames -----> FacialSemanticExtractor (FaceBench precomputed)                              |
                      -> semantic_attrs (211-d) -------------------------------------------------+
                                                                                                  |
                            CausalModule (per-branch + semantic, from epoch 0)                    v
                              Spatial: SCM_real/fake -> residuals_s_real/fake
                              Freq:    SCM_real/fake -> residuals_f_real/fake
                              Semantic: SCM_real/fake -> residuals_sem_real/fake
                                                |
                         CausalViolationAttentionFusion (fused attends 4 residuals)
                                                |
                         classifier_input = fused + gate * causal_delta
                                                |
                         MultiTaskHead -> cls (2)

v4 changes (cleanup):
  - Removed ViolationScorer (dead: 1e-30 outputs, no training signal)
  - Removed ConceptPredictionHead (circular self-supervision)
  - Removed contrastive loss (saturated at margin)
  - Removed uncertainty head (weight=0, conflicts with classification)

v5 changes (causal graph quality + performance):
  - Feed raw backbone features to causal module when SAE disabled (was zeros!)
  - Causal gate init: -3.0 (sigmoid≈0.05, conservative start)
  - Causal loss warmup delayed to epoch 5, extended to epoch 25
  - Graph divergence loss re-added at small weight (pushes real≠fake)
  - Z-feature reconstruction weighted 3x vs semantic in SCM loss
  - Per-tier normalization (LayerNorm) before semantic concatenation
  - Single-phase training: all modules train from epoch 0 (causal gate suppresses early)
  - Class weights [0.8, 1.2] to reduce cross-dataset real bias

v6 changes (tractable causal + semantic projection):
  - z_dim: 1024→32 (SparseFeatureSelector compresses backbone features)
    → causal graph: 32+48=80 nodes (was 1072!) — within DAGMA validated range
  - SemanticProjection: direct 259-d gated semantic → classifier pathway
    → 259→128→1024, gated, provides explicit attribute signal for classification
  - Causal gate: -3.0→-1.5 (sigmoid=0.18) — causal signal usable earlier
    with properly-sized 80-node graphs
"""

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics.base_metrics_class import calculate_metrics_for_train
from .base_detector import AbstractDetector
from detectors import DETECTOR

from networks.nesy_defake.foundation_models import (
    FrequencyFeatureExtractor,
    SpatialFeatureExtractor,
)
from networks.nesy_defake.fusion import MultiModalFusion
from networks.nesy_defake.causal import CausalDiscoveryModule
from networks.nesy_defake.classifiers import MultiTaskHead
from networks.nesy_defake.classifiers.sparse_autoencoder import DualBranchSparseAutoencoder
from networks.nesy_defake.semantic import FacialSemanticExtractor
from networks.nesy_defake.semantic.consistency_rules import CrossAttributeConsistencyRules
from networks.nesy_defake.semantic.consistency_rules_v7 import CrossAttributeConsistencyRulesV7
from networks.nesy_defake.semantic.forensic_features import (
    FORENSIC_FEATURE_NAMES, get_forensic_feature_names,
)
from networks.nesy_defake.semantic.causal_intervention import CausalInterventionModule

logger = logging.getLogger(__name__)
ALL_BRANCHES = ('spatial', 'frequency')


# ---------------------------------------------------------------------------
# Uniformity-Alignment loss (Wang & Isola, 2020)
# ---------------------------------------------------------------------------

def uniformity_loss(x: torch.Tensor, t: float = 2.0) -> torch.Tensor:
    """
    Uniformity loss on the unit hypersphere.
    Encourages features to spread evenly, preventing representation collapse.
    """
    pdist = torch.pdist(x, p=2).pow(2)
    return pdist.mul(-t).exp().mean().clamp(min=1e-6).log()


def alignment_loss(x: torch.Tensor, labels: torch.Tensor,
                   alpha: float = 2.0) -> torch.Tensor:
    """Alignment loss: pull same-class features together on the hypersphere."""
    device = x.device
    total = torch.zeros(1, device=device)
    count = 0

    for c in labels.unique():
        mask = (labels == c)
        if mask.sum() < 2:
            continue
        x_c = x[mask]
        dists = torch.pdist(x_c, p=2).pow(alpha)
        total = total + dists.sum()
        count += len(dists)

    if count == 0:
        return torch.zeros(1, device=device)
    return total / count


def _zero_grad_anchor(module: nn.Module, device: torch.device) -> torch.Tensor:
    anchor = torch.zeros(1, device=device, dtype=torch.float32)
    for p in module.parameters():
        if p.requires_grad:
            anchor = anchor + p.sum() * 0.0
    return anchor


# ---------------------------------------------------------------------------
# NeSy Change 1: Causal Violation Attention Fusion
# ---------------------------------------------------------------------------

class CausalViolationAttentionFusion(nn.Module):
    """
    Attention-based fusion of per-branch causal violation residuals.

    The fused CLIP features (query) attend over the 4 causal violation
    residuals (keys/values), dynamically selecting which violations are
    most informative for each sample. This replaces both:
      - the original zero-init direct projections (4×d→proj_dim)
      - the 32-d bottleneck that was too lossy

    Benefits:
      - Full residual information preserved (values project to proj_dim)
      - Dynamic per-sample weighting — not a static compress
      - Interpretable: attention weights show which violation type matters
      - Cross-branch interaction: spatial_real can suppress freq_fake, etc.
      - Gated: if no violations exist, model learns to zero-out the delta

    Forward returns both the causal delta and the (B, 4) attention weights
    for interpretability logging.
    """

    RESIDUAL_KEYS = ('spatial_real', 'spatial_fake', 'freq_real', 'freq_fake')

    def __init__(self, fused_dim: int, d_spatial: int, d_freq: int,
                 attn_dim: int = 256):
        super().__init__()
        self.attn_dim = attn_dim
        self.scale = math.sqrt(attn_dim)

        d_map = {
            'spatial_real': d_spatial, 'spatial_fake': d_spatial,
            'freq_real':    d_freq,    'freq_fake':    d_freq,
        }

        # Query: fused features decide what to attend to
        self.q_proj = nn.Linear(fused_dim, attn_dim, bias=False)

        # Keys: each residual type contributes a key for attention scoring
        self.k_proj = nn.ModuleDict({
            k: nn.Linear(d, attn_dim, bias=False) for k, d in d_map.items()
        })

        # Values: each residual type is projected to full fused_dim
        self.v_proj = nn.ModuleDict({
            k: nn.Linear(d, fused_dim, bias=False) for k, d in d_map.items()
        })

        # LayerNorm on the weighted output for training stability
        self.out_norm = nn.LayerNorm(fused_dim)

        # Small init: causal delta starts near-zero, grows with training
        nn.init.normal_(self.q_proj.weight, std=0.01)
        for proj in list(self.k_proj.values()) + list(self.v_proj.values()):
            nn.init.normal_(proj.weight, std=0.01)

    def forward(self, fused_features: torch.Tensor,
                residuals: dict) -> tuple:
        """
        Args:
            fused_features: (B, fused_dim) main CLIP feature stream
            residuals: dict[str → (B, d)] causal residuals per violation type

        Returns:
            causal_delta: (B, fused_dim) to add to classifier_input
            attn_weights: (B, 4) attention weights (for interpretability)
        """
        q = self.q_proj(fused_features)                      # (B, attn_dim)

        keys = self.RESIDUAL_KEYS
        k_list = [self.k_proj[k](residuals[k]) for k in keys]   # 4×(B, attn_dim)
        v_list = [self.v_proj[k](residuals[k]) for k in keys]   # 4×(B, fused_dim)

        # Scaled dot-product attention over 4 violation types
        K = torch.stack(k_list, dim=1)                       # (B, 4, attn_dim)
        scores = (q.unsqueeze(1) * K).sum(-1) / self.scale   # (B, 4)
        attn_weights = torch.softmax(scores, dim=-1)          # (B, 4)

        # Weighted sum of values
        V = torch.stack(v_list, dim=1)                        # (B, 4, fused_dim)
        causal_delta = (attn_weights.unsqueeze(-1) * V).sum(1)  # (B, fused_dim)

        return self.out_norm(causal_delta), attn_weights


# ---------------------------------------------------------------------------
# NeSy Change 2: Semantic Projection (direct semantic → classifier pathway)
# ---------------------------------------------------------------------------

class SemanticProjection(nn.Module):
    """
    Projects gated semantic attributes directly into the classifier space.

    This gives the classifier explicit access to facial attribute signals:
      - The causal module captures anomaly patterns via SCM residuals
      - This pathway captures attribute presence/absence for classification

    Architecture: sem_attrs (259) → LayerNorm → Linear → GELU → Dropout
                  → Linear → LayerNorm → output (proj_dim)

    The bottleneck (hidden_dim=128) forces the model to learn a compact,
    discriminative representation of the semantic space rather than
    memorizing all 259 raw features.
    """

    def __init__(self, sem_dim: int, proj_dim: int, hidden_dim: int = 128,
                 dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(sem_dim),
            nn.Linear(sem_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )
        # Small init so the projection starts near-zero
        nn.init.normal_(self.net[1].weight, std=0.02)
        nn.init.zeros_(self.net[1].bias)
        nn.init.normal_(self.net[4].weight, std=0.01)
        nn.init.zeros_(self.net[4].bias)

    def forward(self, semantic_attrs: torch.Tensor) -> torch.Tensor:
        return self.net(semantic_attrs)


def _make_projection(in_dim: int, out_dim: int, dropout: float = 0.0) -> nn.Sequential:
    layers = [nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU()]
    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


@DETECTOR.register_module(module_name='nesydefake_hybrid')
class NeSyDeFakeHybridDetector(AbstractDetector):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ablation_spatial_only = config.get('ablation_spatial_only', False)
        self.ablation_mode = config.get('ablation_mode', 'spatial_ce')
        # ablation_mode values:
        #   'spatial_ce'   → Ablation 1 (GenD baseline, CE loss)
        #   'spatial_edl'  → Ablation 2 (spatial + EDL loss)
        #   'concept_edl'  → Ablation 3 (spatial + concept branch + EDL)
        #   'causal_edl'   → Ablation 4 (spatial + concept + causal + EDL)
        self._current_epoch = 0

        self.build_backbone(config)
        self.fusion = MultiModalFusion(config)

        # -- Module: Facial Semantic Attribute Extractor ----------------------
        sem_cfg = config.get('semantic_attributes', {})
        self.use_semantic_attrs = sem_cfg.get('enabled', False)
        self.use_refined_features = config.get('use_refined_features', False)

        if self.use_semantic_attrs:
            self.semantic_extractor = FacialSemanticExtractor(config)
            actual_dim = self.semantic_extractor.output_dim
            config['causal_module']['semantic_dim'] = actual_dim
            logger.info(f"  Semantic dim    : {actual_dim} "
                        f"(backend={sem_cfg.get('backend', 'farl')})")
        else:
            self.semantic_extractor = None
            actual_dim = config['causal_module'].get('semantic_dim', 211)

        # When using refined features, the dataset provides 122-d combined
        # [fast(58) || vlm(64)] instead of 211-d FaceBench.
        if self.use_refined_features:
            from networks.nesy_defake.semantic.refined_attributes import (
                NUM_COMBINED_FEATURES, COMBINED_FEATURE_NAMES)
            actual_dim = NUM_COMBINED_FEATURES  # 122
            self._combined_feature_names = list(COMBINED_FEATURE_NAMES)
            logger.info(f"  Refined features: {actual_dim}-d combined "
                        f"[fast(58) || vlm(64)]")

        # -- Tier 1: Cross-Attribute Consistency Rules -------------------------
        cr_cfg = config.get('consistency_rules', {})
        self.use_consistency_rules = (
            cr_cfg.get('enabled', False) and self.use_semantic_attrs)
        self._tier1_dim = 0
        if self.use_consistency_rules:
            if self.use_refined_features:
                self.consistency_rules = CrossAttributeConsistencyRulesV7()
                self._tier1_dim = cr_cfg.get('output_dim', 23)
                logger.info(f"  Tier 1 (consistency v7): {self._tier1_dim} features")
            else:
                self.consistency_rules = CrossAttributeConsistencyRules()
                self._tier1_dim = cr_cfg.get('output_dim', 20)
                logger.info(f"  Tier 1 (consistency): {self._tier1_dim} features")
            actual_dim += self._tier1_dim

        # -- Tier 2: Pixel-Level Forensic Features (precomputed) ---------------
        ff_cfg = config.get('forensic_features', {})
        self.use_forensic_features = ff_cfg.get('enabled', False)
        self._tier2_dim = 0
        if self.use_forensic_features:
            self._tier2_dim = ff_cfg.get('output_dim', 30)
            actual_dim += self._tier2_dim
            logger.info(f"  Tier 2 (forensic)  : {self._tier2_dim} features")

        # -- Per-tier normalization (prevents single LayerNorm from losing
        #    per-category signal structure across heterogeneous feature tiers) ---
        if self.use_refined_features:
            base_sem_dim = NUM_COMBINED_FEATURES if self.use_semantic_attrs else 0
        else:
            base_sem_dim = sem_cfg.get('precomputed_dim', 211) if self.use_semantic_attrs else 0
        if base_sem_dim > 0:
            self.tier0_norm = nn.LayerNorm(base_sem_dim)
        if self._tier1_dim > 0:
            self.tier1_norm = nn.LayerNorm(self._tier1_dim)
        if self._tier2_dim > 0:
            self.tier2_norm = nn.LayerNorm(self._tier2_dim)
        self._base_sem_dim = base_sem_dim
        logger.info(f"  Per-tier norms   : base={base_sem_dim}, "
                    f"tier1={self._tier1_dim}, tier2={self._tier2_dim}")

        # Update semantic_dim with augmented dimensions
        config['causal_module']['semantic_dim'] = actual_dim
        logger.info(f"  Augmented sem dim: {actual_dim} "
                    f"(base + {self._tier1_dim} tier1 + {self._tier2_dim} tier2)")

        # -- Module 4: Sparse Autoencoder ------------------------------------
        self.use_sparse = config['sparse_features']['enabled']
        self.sparse_ae = DualBranchSparseAutoencoder(config)
        logger.info(f"  SAE output dim  : {self.sparse_ae.output_dim}")

        # -- Module 3: Dual Sub-Graph Causal Discovery (v3) -------------------
        self.use_causal = config['causal_module']['enabled']
        # Pass training rule names so causal module can label identity graph nodes
        if self.use_consistency_rules:
            config['_training_rule_names'] = self.consistency_rules.get_feature_names()
        # Build combined semantic attribute names for interpretable graphs
        sem_attr_names = []
        if self.use_refined_features:
            sem_attr_names.extend(self._combined_feature_names)
        elif self.use_semantic_attrs and self.semantic_extractor:
            sem_attr_names.extend(self.semantic_extractor.get_attribute_names())
        if self.use_consistency_rules:
            sem_attr_names.extend(self.consistency_rules.get_feature_names())
        if self.use_forensic_features:
            sem_attr_names.extend(get_forensic_feature_names())
        self.causal_module = CausalDiscoveryModule(
            config, semantic_attr_names=sem_attr_names if sem_attr_names else None)

        # -- NeSy Change 1: Causal Violation Attention Fusion ----------------
        # The fused CLIP features (query) attend over 4 causal violation
        # residuals (keys/values) — each is now the concatenation of
        # identity + forensic sub-graph residuals per branch.
        d_spatial = self.causal_module.d_spatial  # 103+62 = 165
        d_freq = self.causal_module.d_freq        # 103+62 = 165
        proj_dim = config['fusion']['projection_dim']  # 1024

        attn_cfg = config.get('causal_attention', {})
        attn_dim = attn_cfg.get('attn_dim', 256)

        self.causal_attn_fusion = CausalViolationAttentionFusion(
            fused_dim=proj_dim,
            d_spatial=d_spatial,
            d_freq=d_freq,
            attn_dim=attn_dim,
        )
        n_attn_params = sum(p.numel() for p in self.causal_attn_fusion.parameters())
        logger.info(f"  Causal attn fusion: q({proj_dim}→{attn_dim}) × "
                    f"4×k/v({d_spatial}/{d_freq}→{attn_dim}/{proj_dim}) "
                    f"(params: {n_attn_params:,})")

        # -- (Removed) Concept Prediction Head — circular self-supervision -----
        # -- (Removed) Graph Divergence — adversarial minimax destabilizes -----
        self.use_concept_pred = False
        self.use_graph_div = False

        # -- Semantic Feature Gating (generalization) --------------------------
        # Trainable sigmoid gate on semantic attributes: learns which of the
        # 211 attributes carry universal (cross-dataset) signal vs
        # dataset-specific noise. Initialized to 0 (sigmoid(0) = 0.5 = neutral).
        sem_gate_cfg = config.get('semantic_gate', {})
        self.use_semantic_gate = sem_gate_cfg.get('enabled', True) and self.use_semantic_attrs
        if self.use_semantic_gate:
            gate_dim = config['causal_module']['semantic_dim']
            self.semantic_gate = nn.Parameter(torch.zeros(gate_dim))
            logger.info(f"  Semantic gate   : {gate_dim}-d (sigmoid, init=0.5)")

        # -- NeSy Change 2: Semantic Projection (direct sem → classifier) ------
        sp_cfg = config.get('semantic_projection', {})
        self.use_semantic_proj = (
            sp_cfg.get('enabled', False) and self.use_semantic_attrs)
        if self.use_semantic_proj:
            sem_proj_dim = config['causal_module']['semantic_dim']
            sp_hidden = sp_cfg.get('hidden_dim', 128)
            sp_dropout = sp_cfg.get('dropout', 0.2)
            self.semantic_proj = SemanticProjection(
                sem_dim=sem_proj_dim, proj_dim=proj_dim,
                hidden_dim=sp_hidden, dropout=sp_dropout)
            sp_gate_init = sp_cfg.get('gate_init', -1.0)
            self.semantic_proj_gate = nn.Parameter(
                torch.tensor(float(sp_gate_init)))
            n_sp_params = sum(p.numel() for p in self.semantic_proj.parameters())
            logger.info(
                f"  Semantic proj   : {sem_proj_dim}→{sp_hidden}→{proj_dim}, "
                f"gate_init={sp_gate_init} "
                f"(sigmoid={torch.sigmoid(torch.tensor(sp_gate_init)).item():.3f}), "
                f"params={n_sp_params:,}")

        # -- Learnable causal gate (controls causal contribution to classifier) -
        # sigmoid(-1.5) ≈ 0.18 → causal delta starts partially active.
        # With dual sub-graphs (identity=103, forensic=62 nodes), causal signal
        # captures both identity-constraint violations and pixel-level artifacts.
        self.causal_gate = nn.Parameter(torch.tensor(-1.5))
        logger.info(f"  Causal gate     : scalar (sigmoid, init={torch.sigmoid(torch.tensor(-1.5)).item():.3f})")

        # -- Tier 3: Causal Intervention (train + inference) --------------------
        ci_cfg = config.get('causal_intervention', {})
        self.use_causal_intervention = ci_cfg.get('enabled', False)
        if self.use_causal_intervention:
            self.causal_intervention = CausalInterventionModule(ci_cfg)
            ci_dim = ci_cfg.get('output_dim', 28)
            self.ci_norm = nn.LayerNorm(ci_dim)
            self.ci_projection = nn.Linear(ci_dim, proj_dim)
            ci_gate_init = ci_cfg.get('ci_gate_init', -2.0)
            self.ci_gate = nn.Parameter(torch.tensor(float(ci_gate_init)))
            # Init near zero — gate suppresses contribution early
            nn.init.normal_(self.ci_projection.weight, std=0.01)
            nn.init.zeros_(self.ci_projection.bias)
            logger.info(
                f"  Tier 3 (intervention): {ci_dim}→{proj_dim}, "
                f"gate_init={ci_gate_init} (sigmoid={torch.sigmoid(torch.tensor(ci_gate_init)).item():.3f})"
            )

        # -- L2 normalization + Uniformity-Alignment loss (GenD recipe) ------
        ua_cfg = config.get('uniformity_alignment', {})
        self.use_ua_loss = ua_cfg.get('enabled', True)
        self.ua_alpha = ua_cfg.get('alignment_weight', 0.1)
        self.ua_beta = ua_cfg.get('uniformity_weight', 0.5)
        self.use_l2_norm = ua_cfg.get('l2_normalize', True)
        if self.use_ua_loss:
            logger.info(f"  UA loss         : alpha={self.ua_alpha}, beta={self.ua_beta}")

        # -- Module 5: Classifier --------------------------------------------
        self.multitaskhead = MultiTaskHead(config)
        self.loss_weights = config['loss_func']['weights']
        self._base_loss_weights = dict(config['loss_func']['weights'])
        self.build_loss(config)

        # -- Loss warm-up schedule -------------------------------------------
        self.loss_warmup_cfg = config.get('loss_warmup', {})

        # -- Ablation 2/3/4: EDL loss + concept/causal branches ----------------
        self._use_edl = self.ablation_mode in ('spatial_edl', 'concept_edl', 'causal_edl')
        self._use_concept_branch = self.ablation_mode in ('concept_edl', 'causal_edl')
        self._use_causal_branch = self.ablation_mode == 'causal_edl'

        if self._use_edl:
            from networks.nesy_defake.losses.edl_loss import EvidentialLoss
            edl_cfg = config.get('edl', {})
            self.edl_loss = EvidentialLoss(
                num_classes=edl_cfg.get('num_classes', 2),
                annealing_epochs=edl_cfg.get('annealing_epochs', 10),
                kl_weight=edl_cfg.get('kl_weight', 0.1),
                avu_weight=edl_cfg.get('avu_weight', 0.0),
            )
            logger.info(f"  EDL loss        : annealing={edl_cfg.get('annealing_epochs', 10)} epochs")

        if self._use_concept_branch:
            from networks.nesy_defake.concept_branch import ConceptBranch
            cb_cfg = config.get('concept_branch', {})
            self.concept_branch = ConceptBranch(
                combined_dim=cb_cfg.get('combined_dim', 122),
                rules_dim=cb_cfg.get('rules_dim', 23),
                hidden_dim=cb_cfg.get('hidden_dim', 64),
                dropout=cb_cfg.get('dropout', 0.2),
            )
            gate_cfg = config.get('evidence_gate', {})
            self.concept_gate = nn.Parameter(
                torch.tensor(float(gate_cfg.get('concept_init', -3.0))))
            logger.info(f"  Concept branch  : gate_init={gate_cfg.get('concept_init', -3.0)}")

        if self._use_causal_branch:
            from networks.nesy_defake.concept_branch import SimplifiedCausalBranch
            sc_cfg = config.get('causal_branch', {})
            self.causal_branch = SimplifiedCausalBranch(
                backbone_dim=sc_cfg.get('backbone_dim', 1024),
                z_causal_dim=sc_cfg.get('z_causal_dim', 32),
                curated_dim=sc_cfg.get('curated_dim', 51),
                rules_dim=sc_cfg.get('rules_dim', 23),
                forensic_dim=sc_cfg.get('forensic_dim', 30),
                hidden_dim=sc_cfg.get('hidden_dim', 64),
                sparsity_penalty=sc_cfg.get('sparsity_penalty', 0.01),
            )
            gate_cfg = config.get('evidence_gate', {})
            self.causal_ev_gate = nn.Parameter(
                torch.tensor(float(gate_cfg.get('causal_init', -3.0))))
            self._dag_penalty_weight = sc_cfg.get('dag_penalty_weight', 0.05)
            logger.info(f"  Causal branch   : gate_init={gate_cfg.get('causal_init', -3.0)}")

        # -- torch.compile on frozen backbones (speed optimization) -----------
        # Frozen modules have static graphs — torch.compile fuses ops and
        # eliminates Python overhead. Only applied to inference-only modules.
        self._try_compile_frozen_modules()

        logger.info("NeSyDeFake Hybrid Detector initialised (v6 — tractable causal + semantic proj)")
        logger.info(f"  Active branches : {sorted(self.active_branches)}")
        logger.info(f"  Fused dim       : {config['fusion']['fused_dim']}")
        logger.info(f"  Projection dim  : {proj_dim}")
        logger.info(f"  Causal d_spatial: {d_spatial}, d_freq: {d_freq}")
        logger.info(f"  SAE enabled     : {self.use_sparse}")
        logger.info(f"  Causal enabled  : {self.use_causal}")
        logger.info(f"  Semantic attrs  : {self.use_semantic_attrs}")
        logger.info(f"  Causal attn fusion: attn_dim={attn_dim}")

        # All modules train from epoch 0. Causal gate init=-3.0 (sigmoid≈0.05)
        # naturally suppresses causal influence until the module learns.

    # ------------------------------------------------------------------ #
    #  Construction helpers                                                #
    # ------------------------------------------------------------------ #

    def build_backbone(self, config: dict) -> None:
        self.spatial_extractor = SpatialFeatureExtractor(config)
        self.frequency_extractor = FrequencyFeatureExtractor(config)

        active = set(config.get('active_branches', list(ALL_BRANCHES)))
        active = active & set(ALL_BRANCHES)
        self.active_branches = active

        # -- Share CLIP backbone between spatial and frequency branches --------
        # Both load the same openai/clip-vit-large-patch14 weights. Sharing
        # saves ~600MB VRAM and enables batched forward (2x backbone speed).
        # Each branch keeps its own pre/post processing (FAD front-end,
        # SafeLayerNorm, projection heads). Phase 2 shared LayerNorms adapt
        # to both input distributions simultaneously — beneficial for
        # generalization (see GenD: LN adaptation is distribution-agnostic).
        self._shared_backbone = False
        fm = config['foundation_models']
        spatial_path = fm.get('spatial', {}).get('model_path', '')
        freq_path = fm.get('frequency', {}).get('model_path', '')
        freq_name = fm.get('frequency', {}).get('name', '')
        if (spatial_path == freq_path
                and freq_name == 'fad_clip'
                and 'spatial' in active and 'frequency' in active):
            # Point freq extractor to spatial's backbone (shared weights)
            self.frequency_extractor.backbone = self.spatial_extractor.backbone
            self._shared_backbone = True
            n_saved = sum(p.numel() for p in self.spatial_extractor.backbone.parameters())
            logger.info(
                f"  Shared backbone : spatial & frequency share CLIP "
                f"({n_saved:,} params, ~{n_saved * 2 / 1e6:.0f}MB BF16 saved)")

        proj_dim = fm['spatial']['output_dim']
        proj_dim = config['fusion']['projection_dim']

        branch_dims = {
            'spatial': fm['spatial']['output_dim'],
            'frequency': self.frequency_extractor.output_dim,
        }

        dropout_cfg = config.get('projection_dropout', {})
        self.spatial_proj = _make_projection(
            branch_dims['spatial'], proj_dim,
            dropout=dropout_cfg.get('spatial', 0.2))
        self.frequency_proj = _make_projection(
            branch_dims['frequency'], proj_dim,
            dropout=dropout_cfg.get('frequency', 0.1))

        fused_dim = proj_dim * sum(1 for b in ALL_BRANCHES if b in active)
        config['fusion']['fused_dim'] = fused_dim

        for name in ALL_BRANCHES:
            if name not in active:
                self._freeze_module(getattr(self, f'{name}_extractor'), name)

    def _freeze_module(self, module: nn.Module, name: str) -> None:
        for p in module.parameters():
            p.requires_grad = False
        logger.info(f"  Frozen branch   : {name}")

    def _try_compile_frozen_modules(self) -> None:
        """
        Apply torch.compile to frozen backbone modules for faster inference.

        Uses default mode (inductor) — NOT reduce-overhead, which uses CUDA
        graphs that overwrite output tensors on replay, breaking downstream
        consumers outside the compiled region.

        Catches errors gracefully — compile is a pure speed optimization.
        """
        if not hasattr(torch, 'compile'):
            return

        compiled = []
        compiled_backbone_ids = set()
        # Compile frozen CLIP vision backbones
        for attr in ('spatial_extractor', 'frequency_extractor'):
            ext = getattr(self, attr, None)
            if ext is None:
                continue
            backbone = getattr(ext, 'backbone', None)
            if backbone is None or id(backbone) in compiled_backbone_ids:
                continue  # skip shared backbone (already compiled via other branch)
            if not any(p.requires_grad for p in backbone.parameters()):
                try:
                    ext.backbone = torch.compile(backbone, fullgraph=False)
                    compiled_backbone_ids.add(id(ext.backbone))
                    compiled.append(f'{attr}.backbone'
                                    + (' (shared)' if self._shared_backbone else ''))
                except Exception as e:
                    logger.warning(f"  torch.compile failed for {attr}: {e}")

        # Compile frozen semantic extractor components
        if self.use_semantic_attrs and self.semantic_extractor is not None:
            sem = self.semantic_extractor
            # Vision tower
            vt = getattr(sem, 'vision_tower', None)
            if vt is not None and not any(
                    p.requires_grad for p in vt.parameters()):
                try:
                    sem.vision_tower = torch.compile(vt, fullgraph=False)
                    compiled.append('semantic.vision_tower')
                except Exception as e:
                    logger.warning(
                        f"  torch.compile failed for semantic vision_tower: {e}")
            # LLM (13B frozen)
            llm = getattr(sem, 'llm', None)
            if llm is not None and not any(
                    p.requires_grad for p in llm.parameters()):
                try:
                    sem.llm = torch.compile(llm, fullgraph=False)
                    compiled.append('semantic.llm')
                except Exception as e:
                    logger.warning(
                        f"  torch.compile failed for semantic LLM: {e}")

        if compiled:
            logger.info(f"  torch.compile   : {', '.join(compiled)}")
        else:
            logger.info("  torch.compile   : no eligible frozen modules")

    def enable_causal(self) -> None:
        self.use_causal = True
        logger.info("Per-branch dual-graph causal module enabled")


    def enable_sparse(self) -> None:
        self.use_sparse = True
        logger.info("SAE enabled")

    def update_loss_warmup(self, epoch: int, total_epochs: int) -> None:
        self._current_epoch = epoch
        for loss_name, schedule in self.loss_warmup_cfg.items():
            start_epoch = schedule.get('start_epoch', 0)
            end_epoch = schedule.get('end_epoch', 10)
            start_w = schedule.get('start_weight', 0.01)
            end_w = schedule.get('end_weight', 0.3)

            if epoch < start_epoch:
                w = start_w
            elif epoch >= end_epoch:
                w = end_w
            else:
                progress = (epoch - start_epoch) / max(end_epoch - start_epoch, 1)
                w = start_w + progress * (end_w - start_w)

            self.loss_weights[loss_name] = w

    def build_loss(self, config: dict) -> None:
        cw = config.get('class_weights', None)
        if cw is not None:
            weight = torch.tensor(cw, dtype=torch.float32)
            self.cls_loss = nn.CrossEntropyLoss(weight=weight)
        else:
            self.cls_loss = nn.CrossEntropyLoss()
        self.reg_loss = nn.MSELoss()

    # ------------------------------------------------------------------ #
    #  Feature extraction                                                  #
    # ------------------------------------------------------------------ #

    def extract_raw_features(self, data_dict: dict) -> dict:
        raw = {}
        both_active = ('spatial' in self.active_branches
                       and 'frequency' in self.active_branches)

        if both_active and self._shared_backbone:
            # -- Batched forward through shared CLIP backbone ------------------
            # 1. Prepare spatial input (already CLIP-normalized by dataset)
            spatial_input = data_dict['spatial_frames']
            if self.spatial_extractor.needs_resize:
                spatial_input = F.interpolate(
                    spatial_input,
                    size=(self.spatial_extractor.required_size,
                          self.spatial_extractor.required_size),
                    mode='bilinear', align_corners=False)

            # 2. Prepare frequency input (FAD front-end in FP32, then
            #    CLIP-normalize — produces images CLIP can process)
            freq_ext = self.frequency_extractor
            with torch.cuda.amp.autocast(enabled=False):
                freq_input_raw = data_dict['freq_frames'].float()
                _, _, H, W = freq_input_raw.shape
                if H != freq_ext.required_size or W != freq_ext.required_size:
                    freq_input_raw = F.interpolate(
                        freq_input_raw,
                        size=(freq_ext.required_size, freq_ext.required_size),
                        mode='bilinear', align_corners=False)
                freq_input = freq_ext.fad_front_end(freq_input_raw)

            B = spatial_input.shape[0]

            # 3. Concatenate along batch dim and run single CLIP forward
            #    Both inputs are CLIP-ready: spatial is pre-normalized,
            #    freq is FAD-enhanced + CLIP-normalized by FADFrontEnd
            batched = torch.cat([spatial_input, freq_input], dim=0)

            use_bf16 = torch.cuda.is_bf16_supported()
            if use_bf16:
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                    outputs = self.spatial_extractor.backbone(pixel_values=batched)
                    pooled = outputs.pooler_output
            else:
                with torch.cuda.amp.autocast(enabled=False):
                    outputs = self.spatial_extractor.backbone(
                        pixel_values=batched.float())
                    pooled = outputs.pooler_output

            # 4. Split back into spatial and frequency features
            raw['spatial_raw'] = pooled[:B]
            with torch.cuda.amp.autocast(enabled=False):
                raw['frequency_raw'] = freq_ext.freq_norm(pooled[B:].float())
        else:
            # -- Standard separate forward (single branch or non-shared) ------
            if 'spatial' in self.active_branches:
                raw['spatial_raw'] = self.spatial_extractor(
                    data_dict['spatial_frames'])
            if 'frequency' in self.active_branches:
                raw['frequency_raw'] = self.frequency_extractor(
                    data_dict['freq_frames'])
        return raw

    def features(self):
        pass

    def project_and_fuse(self, raw_feats: dict) -> torch.Tensor:
        spatial_proj = (
            self.spatial_proj(raw_feats['spatial_raw'])
            if 'spatial_raw' in raw_feats else None
        )
        freq_proj = (
            self.frequency_proj(raw_feats['frequency_raw'])
            if 'frequency_raw' in raw_feats else None
        )
        return self.fusion(spatial_proj, freq_proj)

    def classifier(self, features: torch.Tensor) -> dict:
        return self.multitaskhead(features)

    # ------------------------------------------------------------------ #
    #  Forward pass                                                        #
    # ------------------------------------------------------------------ #

    def forward(self, data_dict: dict, inference: bool = False) -> dict:
        device = data_dict['label'].device

        # -- Ablation 1: Spatial-only shortcut (GenD-exact recipe) -----------
        # Skips ALL NeSy modules: no fusion, no causal, no SAE, no semantic.
        # Pipeline: CLIP → spatial_proj → linear classifier
        if self.ablation_spatial_only and not self._use_edl:
            raw_feats = self.extract_raw_features(data_dict)
            spatial_raw = raw_feats['spatial_raw']
            projected = self.spatial_proj(spatial_raw)
            l2_embeddings = F.normalize(projected, p=2, dim=1)
            task_outputs = self.classifier(projected)
            cls_logits = task_outputs['classification']
            prob = torch.softmax(cls_logits, dim=1)[:, 1]
            return {
                'cls':            cls_logits,
                'prob':           prob,
                'feat':           projected,
                'l2_embeddings':  l2_embeddings,
                'spatial_feat':   spatial_raw,
                'frequency_feat': None,
                'z_spatial':      None,
                'z_freq':         None,
                'causal_out':     None,
                'causal_primitives': None,
                'task_outputs':   task_outputs,
                'sae_loss':       torch.zeros(1, device=device),
                'sae_info':       {},
            }

        # -- Ablation 2/3/4: EDL-based evidence fusion ----------------------
        if self._use_edl:
            raw_feats = self.extract_raw_features(data_dict)
            spatial_raw = raw_feats['spatial_raw']
            projected = self.spatial_proj(spatial_raw)
            l2_embeddings = F.normalize(projected, p=2, dim=1)

            # Spatial evidence: linear head → softplus
            task_outputs = self.classifier(projected)
            spatial_logits = task_outputs['classification']  # (B, 2)
            spatial_evidence = F.softplus(spatial_logits.float())  # (B, 2)
            total_evidence = spatial_evidence

            # Concept evidence (Ablation 3+)
            concept_out = None
            if self._use_concept_branch:
                combined_features = data_dict.get('precomputed_attrs')
                if combined_features is not None:
                    combined_features = combined_features.to(device)
                    concept_out = self.concept_branch(combined_features)
                    concept_gate = torch.sigmoid(self.concept_gate)
                    total_evidence = total_evidence + concept_gate * concept_out['evidence']

            # Causal evidence (Ablation 4)
            causal_out = None
            if self._use_causal_branch and concept_out is not None:
                forensic_features = data_dict.get('forensic_features')
                if forensic_features is not None:
                    forensic_features = forensic_features.to(device)
                    causal_out = self.causal_branch(
                        spatial_raw=spatial_raw,
                        combined_features=combined_features,
                        violations=concept_out['violations'],
                        forensic_features=forensic_features,
                    )
                    causal_gate = torch.sigmoid(self.causal_ev_gate)
                    total_evidence = total_evidence + causal_gate * causal_out['evidence']

            # Dirichlet prediction from fused evidence
            alpha = total_evidence + 1.0
            S = alpha.sum(dim=1, keepdim=True)
            prob = (alpha / S)[:, 1]  # P(fake)
            uncertainty = 2.0 / S.squeeze(1)

            pred = {
                'cls':               spatial_logits,
                'prob':              prob,
                'total_evidence':    total_evidence,
                'spatial_evidence':  spatial_evidence,
                'alpha':             alpha,
                'uncertainty':       uncertainty,
                'feat':              projected,
                'l2_embeddings':     l2_embeddings,
                'spatial_feat':      spatial_raw,
                'frequency_feat':    None,
                'z_spatial':         None,
                'z_freq':            None,
                'causal_out':        None,
                'causal_primitives': None,
                'task_outputs':      task_outputs,
                'sae_loss':          torch.zeros(1, device=device),
                'sae_info':          {},
            }
            if concept_out is not None:
                pred['concept_evidence'] = concept_out['evidence']
                pred['concept_gate'] = torch.sigmoid(self.concept_gate).detach()
                pred['violations'] = concept_out['violations']
            if causal_out is not None:
                pred['causal_evidence'] = causal_out['evidence']
                pred['causal_gate'] = torch.sigmoid(self.causal_ev_gate).detach()
                pred['dag_penalty'] = causal_out['dag_penalty']
                pred['A_identity_real'] = causal_out['A_identity_real']
                pred['A_identity_fake'] = causal_out['A_identity_fake']
                pred['A_forensic_real'] = causal_out['A_forensic_real']
                pred['A_forensic_fake'] = causal_out['A_forensic_fake']
            return pred

        # -- Step 1: Extract raw branch features -----------------------------
        raw_feats = self.extract_raw_features(data_dict)

        # -- Step 2: Project and fuse for classifier -------------------------
        fused_features = self.project_and_fuse(raw_feats)

        # -- Step 3: SAE on raw features (parallel path) ---------------------
        z_spatial = None
        z_freq = None
        sae_loss = torch.zeros(1, device=device)
        sae_info = {}

        if self.use_sparse:
            z_spatial, z_freq, sae_loss, sae_info = self.sparse_ae(
                spatial_feat=raw_feats.get('spatial_raw'),
                frequency_feat=raw_feats.get('frequency_raw'),
            )
        else:
            # When SAE is disabled, feed raw backbone features to causal module
            # so it has actual visual features instead of zeros. The
            # SparseFeatureSelector (Linear 1024→128) will project them.
            z_spatial = raw_feats.get('spatial_raw')
            z_freq = raw_feats.get('frequency_raw')

        # -- Step 4: Per-branch dual-graph causal module ---------------------
        causal_out = None
        semantic_attrs = None   # initialise so Step 5a can safely reference it

        if self.use_causal:
            # Compute semantic attributes
            if self.use_refined_features:
                # Refined mode: dataset provides 122-d combined [fast||vlm]
                semantic_attrs = data_dict.get('precomputed_attrs')
                if semantic_attrs is not None:
                    semantic_attrs = semantic_attrs.to(device)
            elif self.use_semantic_attrs and self.semantic_extractor is not None:
                if self.semantic_extractor.is_precomputed:
                    # Use precomputed Face-LLaVA features from dataset
                    precomputed = data_dict.get('precomputed_attrs')
                    if precomputed is None:
                        precomputed = data_dict.get('semantic_attrs')
                    semantic_attrs = self.semantic_extractor(
                        precomputed_attrs=precomputed)
                else:
                    semantic_attrs = self.semantic_extractor(
                        raw_images=data_dict.get('raw_frames'))
            else:
                semantic_attrs = data_dict.get('semantic_attrs', None)

            # -- Augment semantic vector with Tier 1 + Tier 2 ------------------
            # Per-tier normalization: each category is LayerNorm'd independently
            # before concatenation. This prevents a single downstream LayerNorm
            # from losing per-category signal structure (e.g., consistency rules
            # are mostly near 0, forensic features have different variance).
            if semantic_attrs is not None:
                # Tier 0: base FaceBench attributes (B, 211)
                # Compute consistency rules BEFORE normalizing base attrs
                consistency_feats = None
                if self.use_consistency_rules:
                    consistency_feats = self.consistency_rules(semantic_attrs)

                # Now normalize each tier independently
                sem_parts = []
                if self._base_sem_dim > 0 and hasattr(self, 'tier0_norm'):
                    sem_parts.append(self.tier0_norm(semantic_attrs))
                else:
                    sem_parts.append(semantic_attrs)

                # Tier 1: cross-attribute consistency rules (differentiable)
                if self.use_consistency_rules and consistency_feats is not None:
                    sem_parts.append(self.tier1_norm(consistency_feats))

                # Tier 2: precomputed forensic features
                if self.use_forensic_features:
                    forensic_feats = data_dict.get('forensic_features')
                    if forensic_feats is not None:
                        sem_parts.append(
                            self.tier2_norm(forensic_feats.to(semantic_attrs.device)))
                    else:
                        # Zero-fill if not available (graceful fallback)
                        B = semantic_attrs.shape[0]
                        sem_parts.append(torch.zeros(
                            B, self._tier2_dim, device=semantic_attrs.device))

                # Concatenate: (B, 211 + tier1 + tier2) = (B, 259)
                if len(sem_parts) > 1:
                    semantic_attrs = torch.cat(sem_parts, dim=1)

            # Apply semantic feature gate on full augmented vector
            if (self.use_semantic_gate and semantic_attrs is not None
                    and hasattr(self, 'semantic_gate')):
                gate = torch.sigmoid(self.semantic_gate)  # (augmented_dim,)
                semantic_attrs = semantic_attrs * gate.unsqueeze(0)

            label = data_dict.get('label', None) if not inference else None

            # Detach SAE features: causal graphs learn on stable features,
            # preventing the moving-target problem where DAGMA-DCE tries to
            # discover structure in a non-stationary distribution.
            # Gradients still flow from causal_delta → attention fusion → classifier.
            causal_out = self.causal_module(
                z_spatial=z_spatial.detach() if z_spatial is not None else None,
                z_freq=z_freq.detach() if z_freq is not None else None,
                semantic_attrs=semantic_attrs,
                label=label,
                return_graph=True,
            )

        # -- Step 5: Causal attention fusion + Classification -------------------
        # Query = fused_features attends over 4 causal residuals (keys/values).
        # Returns causal_delta (B, proj_dim) and attention weights (B, 4).
        classifier_input = fused_features
        causal_primitives = None
        if self.use_causal and causal_out is not None:
            residuals = {
                'spatial_real': causal_out['residuals_spatial_real'],
                'spatial_fake': causal_out['residuals_spatial_fake'],
                'freq_real':    causal_out['residuals_freq_real'],
                'freq_fake':    causal_out['residuals_freq_fake'],
            }
            causal_delta, attn_weights = self.causal_attn_fusion(
                fused_features, residuals)
            causal_primitives = attn_weights   # (B, 4) for interpretability
            gate_value = torch.sigmoid(self.causal_gate)
            classifier_input = fused_features + gate_value * causal_delta

        # -- Tier 3: Causal Intervention (train + inference) --------------------
        # 28 features computed under no_grad (no gradient to SCMs).
        # ci_projection trains via classification loss backprop.
        if (self.use_causal_intervention and self.use_causal
                and semantic_attrs is not None and causal_out is not None):
            with torch.no_grad():
                ci_feats = self.causal_intervention(
                    causal_module=self.causal_module,
                    causal_out=causal_out,
                    augmented_semantic=semantic_attrs,
                    z_spatial=z_spatial.detach() if z_spatial is not None else None,
                    z_freq=z_freq.detach() if z_freq is not None else None,
                )  # (B, 28)
            ci_delta = self.ci_projection(self.ci_norm(ci_feats))
            ci_gate_value = torch.sigmoid(self.ci_gate)
            classifier_input = classifier_input + ci_gate_value * ci_delta

        # -- Step 5b: Semantic Projection (direct sem → classifier) -------------
        # Projects the gated 259-d semantic vector into classifier space.
        # Provides explicit attribute signal that the causal residual pathway
        # cannot capture (residuals = reconstruction error, not presence/absence).
        if self.use_semantic_proj and semantic_attrs is not None:
            sem_delta = self.semantic_proj(semantic_attrs)
            sem_gate_value = torch.sigmoid(self.semantic_proj_gate)
            classifier_input = classifier_input + sem_gate_value * sem_delta

        # L2-normalize for hyperspherical representation (GenD recipe)
        l2_embeddings = F.normalize(classifier_input, p=2, dim=1)

        # Classify on un-normalized features (GenD: normalize for UA loss only)
        task_outputs = self.classifier(classifier_input)
        cls_logits = task_outputs['classification']
        prob = torch.softmax(cls_logits, dim=1)[:, 1]

        pred_dict = {
            'cls':            cls_logits,
            'prob':           prob,
            'feat':           fused_features,
            'l2_embeddings':  l2_embeddings,
            'spatial_feat':   raw_feats.get('spatial_raw'),
            'frequency_feat': raw_feats.get('frequency_raw'),
            'z_spatial':      z_spatial,
            'z_freq':         z_freq,
            # Per-branch causal outputs
            'causal_out':     causal_out,
            'causal_primitives':  causal_primitives,    # (B, 4) attn weights
            # SAE
            'task_outputs':   task_outputs,
            'sae_loss':       sae_loss,
            'sae_info':       sae_info,
        }
        return pred_dict

    # ------------------------------------------------------------------ #
    #  Loss computation                                                    #
    # ------------------------------------------------------------------ #

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        device = label.device

        # Move class weights to correct device
        if hasattr(self.cls_loss, 'weight') and self.cls_loss.weight is not None:
            target_dtype = pred_dict['cls'].dtype
            if (self.cls_loss.weight.device != device
                    or self.cls_loss.weight.dtype != target_dtype):
                self.cls_loss.weight = self.cls_loss.weight.to(
                    device=device, dtype=target_dtype)

        cls_loss = self.cls_loss(pred_dict['cls'], label)

        # -- Ablation 2/3/4: EDL loss ----------------------------------------
        if self._use_edl:
            edl_out = self.edl_loss(
                evidence=pred_dict['total_evidence'],
                target=label,
                epoch=self._current_epoch,
            )
            total_loss = edl_out['loss']

            # Add DAG penalty for Ablation 4
            dag_penalty = torch.zeros(1, device=device).squeeze()
            if self._use_causal_branch and 'dag_penalty' in pred_dict:
                dag_penalty = pred_dict['dag_penalty']
                total_loss = total_loss + self._dag_penalty_weight * dag_penalty

            return {
                'overall':          total_loss,
                'classification':   edl_out['loss'].detach(),
                'edl_nll':          edl_out['loss_nll'],
                'edl_kl':           edl_out['loss_kl'],
                'dag_penalty':      dag_penalty.detach() if isinstance(dag_penalty, torch.Tensor) else dag_penalty,
                'causal_real':      torch.zeros(1, device=device).squeeze(),
                'causal_fake':      torch.zeros(1, device=device).squeeze(),
                'sparse':           torch.zeros(1, device=device).squeeze(),
                'ua_alignment':     torch.zeros(1, device=device).squeeze(),
                'ua_uniformity':    torch.zeros(1, device=device).squeeze(),
                'causal_semantic':  torch.zeros(1, device=device).squeeze(),
                'graph_divergence': torch.zeros(1, device=device).squeeze(),
            }

        # -- Ablation 1: pure CE loss only -----------------------------------
        if self.ablation_spatial_only:
            def _scalar(t):
                return t.squeeze() if isinstance(t, torch.Tensor) else t
            return {
                'overall':            _scalar(cls_loss),
                'classification':     _scalar(cls_loss),
                'causal_real':        torch.zeros(1, device=device).squeeze(),
                'causal_fake':        torch.zeros(1, device=device).squeeze(),
                'sparse':             torch.zeros(1, device=device).squeeze(),
                'ua_alignment':       torch.zeros(1, device=device).squeeze(),
                'ua_uniformity':      torch.zeros(1, device=device).squeeze(),
                'causal_semantic':    torch.zeros(1, device=device).squeeze(),
                'graph_divergence':   torch.zeros(1, device=device).squeeze(),
            }

        # -- Per-branch dual-graph causal structural losses --------------------
        causal_loss_real = torch.zeros(1, device=device)
        causal_loss_fake = torch.zeros(1, device=device)
        causal_semantic_loss = torch.zeros(1, device=device)
        graph_div_loss = torch.zeros(1, device=device)

        if self.use_causal:
            causal_out = pred_dict.get('causal_out')
            if causal_out is not None:
                real_mask = (label == 0)
                fake_mask = (label == 1)
                dag_w = self.config['causal_module']['dag_learning']['dag_penalty_weight']

                # Structural losses: SCM must reconstruct its target class.
                # z-feature portion weighted 3x to prevent semantic dominance.
                # v3: residuals are concatenated [identity(103), forensic(62)].
                # z_causal(32) is the first portion of each sub-graph's residual.
                z_c = self.causal_module.z_causal_dim    # 32
                d_id = self.causal_module.d_identity      # 103
                for branch in ('spatial', 'freq'):
                    r_real = causal_out.get(f'residuals_{branch}_real')
                    r_fake = causal_out.get(f'residuals_{branch}_fake')

                    if r_real is not None and real_mask.any():
                        r = r_real[real_mask]
                        # Identity sub-graph: z(0:32) + s(32:103)
                        z_loss = r[:, :z_c].pow(2).mean() * 3.0
                        s_loss = r[:, z_c:d_id].pow(2).mean()
                        # Forensic sub-graph: z(103:135) + s(135:165)
                        z_loss = z_loss + r[:, d_id:d_id+z_c].pow(2).mean() * 3.0
                        s_loss = s_loss + r[:, d_id+z_c:].pow(2).mean()
                        causal_loss_real = causal_loss_real + z_loss + s_loss
                    if r_fake is not None and fake_mask.any():
                        r = r_fake[fake_mask]
                        z_loss = r[:, :z_c].pow(2).mean() * 3.0
                        s_loss = r[:, z_c:d_id].pow(2).mean()
                        z_loss = z_loss + r[:, d_id:d_id+z_c].pow(2).mean() * 3.0
                        s_loss = s_loss + r[:, d_id+z_c:].pow(2).mean()
                        causal_loss_fake = causal_loss_fake + z_loss + s_loss

                # Semantic graph structural loss (Part A)
                if self.causal_module.use_semantic_graph:
                    r_sem_real = causal_out.get('residuals_sem_real')
                    r_sem_fake = causal_out.get('residuals_sem_fake')
                    if r_sem_real is not None and real_mask.any():
                        causal_semantic_loss = causal_semantic_loss + r_sem_real[real_mask].pow(2).mean()
                    if r_sem_fake is not None and fake_mask.any():
                        causal_semantic_loss = causal_semantic_loss + r_sem_fake[fake_mask].pow(2).mean()

                # Graph divergence: encourage real≠fake causal graphs.
                # Negative L1 distance → pushes graphs apart for specialization.
                # v3: 4 sub-graphs (2 branches × 2 sub-graph types)
                for branch in ('spatial', 'freq'):
                    for subgraph in ('identity', 'forensic'):
                        A_r = causal_out.get(f'A_{branch}_{subgraph}_real')
                        A_f = causal_out.get(f'A_{branch}_{subgraph}_fake')
                        if A_r is not None and A_f is not None:
                            graph_div_loss = graph_div_loss - torch.abs(A_r - A_f).mean()

                # DAG acyclicity penalties (8 graphs: 2 branches × 2 sub-graphs × 2 dist)
                cm = self.causal_module
                branch_pairs_for_dag = [
                    (cm.identity_spatial, 1.0),
                    (cm.identity_freq, 1.0),
                    (cm.forensic_spatial, 1.0),
                    (cm.forensic_freq, 1.0),
                ]
                for branch_pair, weight_mult in branch_pairs_for_dag:
                    causal_loss_real = causal_loss_real + (
                        dag_w * weight_mult
                        * branch_pair.causal_learner_real.compute_dag_penalty()
                    )
                    causal_loss_fake = causal_loss_fake + (
                        0.5 * dag_w * weight_mult
                        * branch_pair.causal_learner_fake.compute_dag_penalty()
                    )

        # -- Uniformity-Alignment loss (GenD recipe) --------------------------
        ua_align = torch.zeros(1, device=device)
        ua_uniform = torch.zeros(1, device=device)
        if self.use_ua_loss and pred_dict.get('l2_embeddings') is not None:
            l2_emb = pred_dict['l2_embeddings']
            ua_align = alignment_loss(l2_emb, label, alpha=2.0)
            ua_uniform = uniformity_loss(l2_emb, t=2.0)

        # SAE loss
        sae_loss = pred_dict.get('sae_loss', torch.zeros(1, device=device))
        if not isinstance(sae_loss, torch.Tensor):
            sae_loss = torch.zeros(1, device=device)

        # DDP anchor: ensures all params get a gradient even when not used
        ddp_anchor = torch.zeros(1, device=device)
        branch_pairs = {
            'spatial':   (self.spatial_extractor, self.spatial_proj),
            'frequency': (self.frequency_extractor, self.frequency_proj),
        }
        for name, (extractor, proj) in branch_pairs.items():
            if name not in self.active_branches:
                ddp_anchor = ddp_anchor + _zero_grad_anchor(extractor, device)
                ddp_anchor = ddp_anchor + _zero_grad_anchor(proj, device)

        # Tier 3 ci_projection/ci_gate/ci_norm now train via classification loss
        # (no DDP anchor needed — they get real gradients every step)

        for attr in ('causal_module', 'sparse_ae', 'multitaskhead',
                     'semantic_extractor', 'causal_attn_fusion',
                     'semantic_proj'):
            mod = getattr(self, attr, None)
            if mod is not None:
                ddp_anchor = ddp_anchor + _zero_grad_anchor(mod, device)
        for gate_name in ('semantic_gate', 'causal_gate', 'semantic_proj_gate'):
            gate = getattr(self, gate_name, None)
            if gate is not None and isinstance(gate, nn.Parameter):
                ddp_anchor = ddp_anchor + gate.sum() * 0.0

        # Loss weights
        w = self.loss_weights
        w_causal      = w.get('causal', 0.0)
        w_causal_fake = w.get('causal_fake', w_causal * 0.5)
        w_causal_sem  = w.get('causal_semantic', 0.0)
        w_graph_div   = w.get('graph_divergence', 0.0)

        total_loss = (
            w.get('classification', 1.0) * cls_loss
            + w_causal                   * causal_loss_real
            + w_causal_fake              * causal_loss_fake
            + w.get('sparse', 0.0)       * sae_loss
            + self.ua_alpha              * ua_align
            + self.ua_beta               * ua_uniform
            + w_causal_sem               * causal_semantic_loss
            + w_graph_div                * graph_div_loss
            + ddp_anchor
        )

        def _scalar(t):
            return t.squeeze() if isinstance(t, torch.Tensor) else t

        return {
            'overall':            _scalar(total_loss),
            'classification':     _scalar(cls_loss),
            'causal_real':        _scalar(causal_loss_real),
            'causal_fake':        _scalar(causal_loss_fake),
            'sparse':             _scalar(sae_loss),
            'ua_alignment':       _scalar(ua_align),
            'ua_uniformity':      _scalar(ua_uniform),
            'causal_semantic':    _scalar(causal_semantic_loss),
            'graph_divergence':   _scalar(graph_div_loss),
        }

    # ------------------------------------------------------------------ #
    #  Metrics                                                             #
    # ------------------------------------------------------------------ #

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(
            label.detach().float(), pred.detach().float())
        metrics = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}

        if self.ablation_spatial_only and not self._use_edl:
            self.video_names = []
            return metrics

        # EDL ablation metrics
        if self._use_edl:
            if 'uncertainty' in pred_dict:
                metrics['uncertainty_mean'] = float(
                    pred_dict['uncertainty'].detach().mean().item())
            if 'concept_gate' in pred_dict:
                metrics['concept_gate'] = float(pred_dict['concept_gate'].item())
            if 'causal_gate' in pred_dict:
                metrics['causal_gate'] = float(pred_dict['causal_gate'].item())
            self.video_names = []
            return metrics

        # SAE diagnostics
        sae_info = pred_dict.get('sae_info', {})
        for branch_name, branch_info in sae_info.items():
            if isinstance(branch_info, dict):
                metrics[f'sae_{branch_name}_l0']   = float(branch_info.get('l0', 0))
                metrics[f'sae_{branch_name}_fvu']  = float(branch_info.get('fvu', 0))
                metrics[f'sae_{branch_name}_dead'] = int(branch_info.get('n_dead', 0))

        # Causal attention weights (interpretability)
        causal_primitives = pred_dict.get('causal_primitives')
        if causal_primitives is not None:
            aw = causal_primitives.detach().float().mean(0)  # (4,)
            for i, name in enumerate(CausalViolationAttentionFusion.RESIDUAL_KEYS):
                metrics[f'attn_{name}'] = float(aw[i])

        # Gate values (monitor how much each pathway contributes)
        if hasattr(self, 'causal_gate'):
            metrics['causal_gate'] = float(torch.sigmoid(self.causal_gate).item())
        if hasattr(self, 'semantic_proj_gate') and self.semantic_proj_gate is not None:
            metrics['sem_proj_gate'] = float(
                torch.sigmoid(self.semantic_proj_gate).item())

        self.video_names = []
        return metrics
