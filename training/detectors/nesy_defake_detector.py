"""
detectors/nesy_defake_hybrid_detector.py
=========================================
END-TO-END PER-BRANCH DUAL-GRAPH CAUSAL DISCOVERY

Data flow:
  spatial_frames --> SpatialExtractor --> raw_spatial (1024-d) --+-> spatial_proj -> fused -> classifier
                                                                  +-> SAE.spatial -> z_spatial (4096) --+
  freq_frames ----> FreqExtractor ----> raw_freq (1024-d) ------+-> freq_proj -> fused -> classifier   |
                                                                  +-> SAE.freq -> z_freq (4096) --------+
  raw_frames -----> FacialSemanticExtractor (FaceBench/FaRL/precomputed)                                |
                      [frozen backbone] -> [trainable proj]                                             |
                      -> semantic_attrs (128-d) --------------------------------------------------------+
                                                                                                        |
                                    CausalModule (per-branch, all epochs)                               v
                                      Spatial branch:                             Frequency branch:
                                        SCM_spatial_real -> residuals_s_real        SCM_freq_real -> residuals_f_real
                                        SCM_spatial_fake -> residuals_s_fake        SCM_freq_fake -> residuals_f_fake
                                                   |                                           |
                      violation_proj_spatial_real(r_s_real) --+      violation_proj_freq_real(r_f_real) --+
                      violation_proj_spatial_fake(r_s_fake) --+      violation_proj_freq_fake(r_f_fake) --+
                                                              |                                          |
                                                              +-- additive fusion into classifier -------+
                                                                                    |
                                                       classifier_input = fused + delta_spatial + delta_freq
                                                                                    |
                                                       MultiTaskHead -> cls (2), uncertainty (1)

All modules train jointly from epoch 0 with loss weight warm-up.
Gradients flow end-to-end through causal module back to SAE and backbone.
"""

import logging
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

logger = logging.getLogger(__name__)
ALL_BRANCHES = ('spatial', 'frequency')


def _zero_grad_anchor(module: nn.Module, device: torch.device) -> torch.Tensor:
    anchor = torch.zeros(1, device=device, dtype=torch.float32)
    for p in module.parameters():
        if p.requires_grad:
            anchor = anchor + p.sum() * 0.0
    return anchor


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

        self.build_backbone(config)
        self.fusion = MultiModalFusion(config)

        # -- Module: Facial Semantic Attribute Extractor ----------------------
        sem_cfg = config.get('semantic_attributes', {})
        self.use_semantic_attrs = sem_cfg.get('enabled', False)
        if self.use_semantic_attrs:
            self.semantic_extractor = FacialSemanticExtractor(config)
            actual_dim = self.semantic_extractor.output_dim
            config['causal_module']['semantic_dim'] = actual_dim
            logger.info(f"  Semantic dim    : {actual_dim} "
                        f"(backend={sem_cfg.get('backend', 'farl')})")
        else:
            self.semantic_extractor = None

        # -- Module 4: Sparse Autoencoder ------------------------------------
        self.use_sparse = config['sparse_features']['enabled']
        self.sparse_ae = DualBranchSparseAutoencoder(config)
        logger.info(f"  SAE output dim  : {self.sparse_ae.output_dim}")

        # -- Module 3: Per-Branch Dual-Graph Causal Discovery ----------------
        self.use_causal = config['causal_module']['enabled']
        self.causal_module = CausalDiscoveryModule(config)

        # -- Per-branch violation projections -> classifier fusion -----------
        # 4 projections: spatial_real, spatial_fake, freq_real, freq_fake
        # All zero-initialized so causal signal starts neutral.
        causal_cfg = config['causal_module']
        s_dim = causal_cfg['semantic_dim']
        d_spatial = causal_cfg['latent_variables']['z_spatial_dim'] + s_dim
        d_freq = causal_cfg['latent_variables']['z_frequency_dim'] + s_dim
        proj_dim = config['fusion']['projection_dim']  # 1024

        self.violation_proj_spatial_real = nn.Linear(d_spatial, proj_dim, bias=False)
        self.violation_proj_spatial_fake = nn.Linear(d_spatial, proj_dim, bias=False)
        self.violation_proj_freq_real = nn.Linear(d_freq, proj_dim, bias=False)
        self.violation_proj_freq_fake = nn.Linear(d_freq, proj_dim, bias=False)
        nn.init.zeros_(self.violation_proj_spatial_real.weight)
        nn.init.zeros_(self.violation_proj_spatial_fake.weight)
        nn.init.zeros_(self.violation_proj_freq_real.weight)
        nn.init.zeros_(self.violation_proj_freq_fake.weight)

        # -- Module 5: Classifier --------------------------------------------
        self.multitaskhead = MultiTaskHead(config)
        self.loss_weights = config['loss_func']['weights']
        self._base_loss_weights = dict(config['loss_func']['weights'])
        self.build_loss(config)

        # -- Loss warm-up schedule -------------------------------------------
        self.loss_warmup_cfg = config.get('loss_warmup', {})

        logger.info("NeSyDeFake Hybrid Detector initialised (PER-BRANCH CAUSAL)")
        logger.info(f"  Active branches : {sorted(self.active_branches)}")
        logger.info(f"  Fused dim       : {config['fusion']['fused_dim']}")
        logger.info(f"  Projection dim  : {proj_dim}")
        logger.info(f"  Causal d_spatial: {d_spatial}, d_freq: {d_freq}")
        logger.info(f"  SAE enabled     : {self.use_sparse}")
        logger.info(f"  Causal enabled  : {self.use_causal}")
        logger.info(f"  Semantic attrs  : {self.use_semantic_attrs}")

    # ------------------------------------------------------------------ #
    #  Construction helpers                                                #
    # ------------------------------------------------------------------ #

    def build_backbone(self, config: dict) -> None:
        self.spatial_extractor = SpatialFeatureExtractor(config)
        self.frequency_extractor = FrequencyFeatureExtractor(config)

        active = set(config.get('active_branches', list(ALL_BRANCHES)))
        active = active & set(ALL_BRANCHES)
        self.active_branches = active

        fm = config['foundation_models']
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

    def enable_causal(self) -> None:
        self.use_causal = True
        logger.info("Per-branch dual-graph causal module enabled")

    def enable_sparse(self) -> None:
        self.use_sparse = True
        logger.info("SAE enabled")

    def update_loss_warmup(self, epoch: int, total_epochs: int) -> None:
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
        if 'spatial' in self.active_branches:
            raw['spatial_raw'] = self.spatial_extractor(data_dict['spatial_frames'])
        if 'frequency' in self.active_branches:
            raw['frequency_raw'] = self.frequency_extractor(data_dict['freq_frames'])
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

        # -- Step 4: Per-branch dual-graph causal module ---------------------
        causal_out = None

        if self.use_causal:
            # Compute semantic attributes from dedicated face analysis model
            if self.use_semantic_attrs and self.semantic_extractor is not None:
                if self.semantic_extractor.is_precomputed:
                    semantic_attrs = self.semantic_extractor(
                        precomputed_attrs=data_dict.get('semantic_attrs'))
                else:
                    semantic_attrs = self.semantic_extractor(
                        raw_images=data_dict.get('raw_frames'))
            else:
                semantic_attrs = data_dict.get('semantic_attrs', None)

            label = data_dict.get('label', None) if not inference else None

            causal_out = self.causal_module(
                z_spatial=z_spatial,
                z_freq=z_freq,
                semantic_attrs=semantic_attrs,
                label=label,
                return_graph=True,
            )

        # -- Step 5: Per-branch violation fusion + Classification ------------
        # 4 violation projections: one per (branch x distribution).
        # All zero-init so causal signal starts neutral and grows with training.
        classifier_input = fused_features
        if self.use_causal and causal_out is not None:
            classifier_input = (
                fused_features
                + self.violation_proj_spatial_real(causal_out['residuals_spatial_real'])
                + self.violation_proj_spatial_fake(causal_out['residuals_spatial_fake'])
                + self.violation_proj_freq_real(causal_out['residuals_freq_real'])
                + self.violation_proj_freq_fake(causal_out['residuals_freq_fake'])
            )

        task_outputs = self.classifier(classifier_input)
        cls_logits = task_outputs['classification']
        prob = torch.softmax(cls_logits, dim=1)[:, 1]

        # Aggregated scores for backward compat
        v_real = causal_out['v_real'] if causal_out else None
        v_fake = causal_out['v_fake'] if causal_out else None

        pred_dict = {
            'cls':            cls_logits,
            'prob':           prob,
            'feat':           fused_features,
            'spatial_feat':   raw_feats.get('spatial_raw'),
            'frequency_feat': raw_feats.get('frequency_raw'),
            'z_spatial':      z_spatial,
            'z_freq':         z_freq,
            'uncertainty':    task_outputs.get('uncertainty'),
            # Per-branch causal outputs
            'causal_out':     causal_out,
            # Aggregated (backward compat)
            'v_real':         v_real,
            'v_fake':         v_fake,
            'violation_score': v_real,
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

        # Uncertainty loss
        uncertainty_loss = torch.zeros(1, device=device)
        if pred_dict.get('uncertainty') is not None:
            pred_label = pred_dict['cls'].argmax(dim=1)
            is_correct = (pred_label == label).float()
            uncertainty_loss = self.reg_loss(
                pred_dict['uncertainty'].squeeze(), 1 - is_correct)

        # -- Per-branch dual-graph causal losses -----------------------------
        # Aggregated across branches: causal_real = spatial_real + freq_real
        causal_loss_real = torch.zeros(1, device=device)
        causal_loss_fake = torch.zeros(1, device=device)
        contrastive_loss_real = torch.zeros(1, device=device)
        contrastive_loss_fake = torch.zeros(1, device=device)

        if self.use_causal:
            causal_out = pred_dict.get('causal_out')
            if causal_out is not None:
                real_mask = (label == 0)
                fake_mask = (label == 1)
                dag_w = self.config['causal_module']['dag_learning']['dag_penalty_weight']

                # -- Structural losses: SCM must reconstruct its target class --
                for branch in ('spatial', 'freq'):
                    r_real = causal_out.get(f'residuals_{branch}_real')
                    r_fake = causal_out.get(f'residuals_{branch}_fake')

                    # L_structural_real: real SCM reconstructs reals well
                    if r_real is not None and real_mask.any():
                        causal_loss_real = causal_loss_real + r_real[real_mask].pow(2).mean()

                    # L_structural_fake: fake SCM reconstructs fakes well
                    if r_fake is not None and fake_mask.any():
                        causal_loss_fake = causal_loss_fake + r_fake[fake_mask].pow(2).mean()

                # -- DAG acyclicity penalties ----------------------------------
                cm = self.causal_module
                for branch_pair, weight_mult in [
                    (cm.causal_spatial, 1.0),
                    (cm.causal_freq, 1.0),
                ]:
                    causal_loss_real = causal_loss_real + (
                        dag_w * weight_mult
                        * branch_pair.causal_learner_real.compute_dag_penalty()
                    )
                    causal_loss_fake = causal_loss_fake + (
                        0.5 * dag_w * weight_mult
                        * branch_pair.causal_learner_fake.compute_dag_penalty()
                    )

                # -- Contrastive losses: v scores high for fakes, low for reals
                for branch in ('spatial', 'freq'):
                    v_r = causal_out.get(f'v_{branch}_real')
                    v_f = causal_out.get(f'v_{branch}_fake')
                    if v_r is not None:
                        contrastive_loss_real = contrastive_loss_real + (
                            self._contrastive_margin(v_r, real_mask, fake_mask, margin=1.0))
                    if v_f is not None:
                        contrastive_loss_fake = contrastive_loss_fake + (
                            self._contrastive_margin(v_f, real_mask, fake_mask, margin=1.0))

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

        for attr in ('causal_module', 'sparse_ae', 'multitaskhead',
                     'violation_proj_spatial_real', 'violation_proj_spatial_fake',
                     'violation_proj_freq_real', 'violation_proj_freq_fake',
                     'semantic_extractor'):
            mod = getattr(self, attr, None)
            if mod is not None:
                ddp_anchor = ddp_anchor + _zero_grad_anchor(mod, device)

        # Loss weights
        w = self.loss_weights
        w_causal      = w.get('causal', 0.0)
        w_causal_fake = w.get('causal_fake', w_causal * 0.5)
        w_contr       = w.get('contrastive', 0.0)
        w_contr_fake  = w.get('contrastive_fake', w_contr * 0.5)

        total_loss = (
            w.get('classification', 1.0) * cls_loss
            + w.get('uncertainty', 0.0)  * uncertainty_loss
            + w_causal                   * causal_loss_real
            + w_causal_fake              * causal_loss_fake
            + w_contr                    * contrastive_loss_real
            + w_contr_fake               * contrastive_loss_fake
            + w.get('sparse', 0.0)       * sae_loss
            + ddp_anchor
        )

        def _scalar(t):
            return t.squeeze() if isinstance(t, torch.Tensor) else t

        return {
            'overall':            _scalar(total_loss),
            'classification':     _scalar(cls_loss),
            'uncertainty':        _scalar(uncertainty_loss),
            'causal_real':        _scalar(causal_loss_real),
            'causal_fake':        _scalar(causal_loss_fake),
            'contrastive_real':   _scalar(contrastive_loss_real),
            'contrastive_fake':   _scalar(contrastive_loss_fake),
            'sparse':             _scalar(sae_loss),
            # Legacy keys
            'causal':             _scalar(causal_loss_real),
            'contrastive':        _scalar(contrastive_loss_real),
        }

    @staticmethod
    def _contrastive_margin(
        score: torch.Tensor,
        real_mask: torch.Tensor,
        fake_mask: torch.Tensor,
        margin: float = 1.0,
    ) -> torch.Tensor:
        """
        Margin contrastive loss: score[fake].mean() - score[real].mean() >= margin.
        """
        device = score.device
        if real_mask.any() and fake_mask.any():
            return F.relu(margin - (score[fake_mask].mean() - score[real_mask].mean()))
        elif real_mask.any():
            return score[real_mask].mean().clamp(min=0)
        elif fake_mask.any():
            return F.relu(margin - score[fake_mask].mean())
        return torch.zeros(1, device=device)

    # ------------------------------------------------------------------ #
    #  Metrics                                                             #
    # ------------------------------------------------------------------ #

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(
            label.detach().float(), pred.detach().float())
        metrics = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}

        if pred_dict.get('uncertainty') is not None:
            metrics['mean_uncertainty'] = float(
                pred_dict['uncertainty'].detach().float().cpu().numpy().mean())

        # SAE diagnostics
        sae_info = pred_dict.get('sae_info', {})
        for branch_name, branch_info in sae_info.items():
            if isinstance(branch_info, dict):
                metrics[f'sae_{branch_name}_l0']   = float(branch_info.get('l0', 0))
                metrics[f'sae_{branch_name}_fvu']  = float(branch_info.get('fvu', 0))
                metrics[f'sae_{branch_name}_dead'] = int(branch_info.get('n_dead', 0))

        # Per-branch causal diagnostics
        if self.use_causal:
            causal_out = pred_dict.get('causal_out')
            if causal_out is not None:
                real_mask = (label == 0)
                fake_mask = (label == 1)

                # Aggregated scores
                v_real = causal_out['v_real'].detach().float()
                v_fake = causal_out['v_fake'].detach().float()
                if real_mask.any():
                    metrics['v_real_reals'] = float(v_real[real_mask].mean())
                    metrics['v_fake_reals'] = float(v_fake[real_mask].mean())
                if fake_mask.any():
                    metrics['v_real_fakes'] = float(v_real[fake_mask].mean())
                    metrics['v_fake_fakes'] = float(v_fake[fake_mask].mean())
                if real_mask.any() and fake_mask.any():
                    metrics['v_real_separation'] = float(
                        v_real[fake_mask].mean() - v_real[real_mask].mean())
                    metrics['v_fake_separation'] = float(
                        v_fake[fake_mask].mean() - v_fake[real_mask].mean())

                # Per-branch scores
                for branch in ('spatial', 'freq'):
                    v_br = causal_out[f'v_{branch}_real'].detach().float()
                    v_bf = causal_out[f'v_{branch}_fake'].detach().float()
                    if real_mask.any() and fake_mask.any():
                        metrics[f'v_{branch}_real_sep'] = float(
                            v_br[fake_mask].mean() - v_br[real_mask].mean())
                        metrics[f'v_{branch}_fake_sep'] = float(
                            v_bf[fake_mask].mean() - v_bf[real_mask].mean())

        self.video_names = []
        return metrics
