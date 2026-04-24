"""
interpretability/engine.py
==========================
InterpretabilityEngine: orchestrates all analyzers, collects batch outputs
during eval, and produces aggregate reports + visualizations.
"""

import logging
import os
from typing import Dict, List, Optional

import numpy as np
import torch

from .analyzers import (
    BaseAnalyzer,
    BranchEvidenceAnalyzer,
    CaseStudyAnalyzer,
    CCVAnalyzer,
    ConsistencyRuleAnalyzer,
    DisagreementAnalyzer,
    EDLUncertaintyAnalyzer,
    GateAnalyzer,
    SCMAnalyzer,
    SelectivePredictionAnalyzer,
    TSNEEmbeddingAnalyzer,
)
from . import visualization as viz

logger = logging.getLogger(__name__)

# Default level flags
_DEFAULT_LEVELS = {
    'edl_uncertainty': True,
    'branch_evidence': True,
    'consistency_rules': True,
    'ccv_analysis': True,
    'scm_analysis': True,
    'gate_analysis': True,
    'disagreement': True,
    'selective_prediction': True,
    'case_study': True,
    # t-SNE is opt-in — disabled by default because sklearn.manifold.TSNE
    # is O(n^2) and adds a few seconds per test dataset.
    'tsne': False,
}


class InterpretabilityEngine:
    """
    Orchestrates interpretability analysis during model evaluation.

    Usage::

        engine = InterpretabilityEngine(config, causal_type='ccv')
        # During eval loop:
        for batch in data_loader:
            preds = model(batch)
            engine.collect_batch(preds, labels)
        # After eval:
        engine.collect_model_params(model)   # for SCM adjacency matrices
        results = engine.finalize(save_dir)
    """

    def __init__(self, config: dict, causal_type: Optional[str] = None):
        self.config = config
        self.causal_type = causal_type or ''

        interp_cfg = config.get('interpretability', {})
        levels = interp_cfg.get('levels', _DEFAULT_LEVELS)
        self.num_explain_samples = interp_cfg.get('num_explain_samples', 10)

        self.analyzers: Dict[str, BaseAnalyzer] = {}

        # Instantiate analyzers based on config and causal type
        if levels.get('edl_uncertainty', True):
            self.analyzers['edl_uncertainty'] = EDLUncertaintyAnalyzer()

        if levels.get('branch_evidence', True):
            self.analyzers['branch_evidence'] = BranchEvidenceAnalyzer()

        if levels.get('consistency_rules', True):
            self.analyzers['consistency_rules'] = ConsistencyRuleAnalyzer()

        if levels.get('ccv_analysis', True) and self.causal_type == 'ccv':
            self.analyzers['ccv_analysis'] = CCVAnalyzer()

        if levels.get('scm_analysis', True) and self.causal_type == 'improved_scm':
            self.analyzers['scm_analysis'] = SCMAnalyzer(
                top_k=interp_cfg.get('graph_viz_top_k', 20))

        if levels.get('gate_analysis', True):
            self.analyzers['gate_analysis'] = GateAnalyzer()

        if levels.get('disagreement', True):
            self.analyzers['disagreement'] = DisagreementAnalyzer()

        if levels.get('selective_prediction', True):
            self.analyzers['selective_prediction'] = SelectivePredictionAnalyzer()

        if levels.get('case_study', True):
            self.analyzers['case_study'] = CaseStudyAnalyzer(
                num_samples=interp_cfg.get('case_study_samples', 6))

        # ── t-SNE embedding plot (opt-in) ────────────────────────────────
        tsne_cfg = interp_cfg.get('tsne', {})
        tsne_enabled = (
            levels.get('tsne', False) or tsne_cfg.get('enabled', False))
        if tsne_enabled:
            self.analyzers['tsne'] = TSNEEmbeddingAnalyzer(
                mode=tsne_cfg.get('mode', 'worst'),
                top_k=tsne_cfg.get('top_k', 500),
                perplexity=tsne_cfg.get('perplexity', 30),
                n_iter=tsne_cfg.get('n_iter', 1000),
                feature_key=tsne_cfg.get('feature_key', 'feat'),
                seed=tsne_cfg.get('seed', 42),
            )

        # Build the set of keys we need from prediction dicts
        self._required_keys = set()
        for analyzer in self.analyzers.values():
            self._required_keys.update(analyzer.required_keys)
        # Always collect A_* keys for SCM
        if 'scm_analysis' in self.analyzers:
            self._collect_adjacency = True
        else:
            self._collect_adjacency = False

        logger.info(
            f"InterpretabilityEngine initialized: "
            f"causal_type={self.causal_type}, "
            f"analyzers={list(self.analyzers.keys())}, "
            f"required_keys={self._required_keys}")

    @property
    def required_keys(self) -> set:
        """Keys that should be collected from prediction dicts."""
        return self._required_keys

    def collect_batch(self, preds: dict, labels) -> None:
        """Fan out one batch of predictions to all active analyzers."""
        for analyzer in self.analyzers.values():
            try:
                analyzer.collect(preds, labels)
            except Exception as e:
                logger.warning(f"Analyzer {analyzer.name} collect failed: {e}")

    def collect_model_params(self, model) -> None:
        """One-time collection of model parameters (e.g., SCM adjacency matrices)."""
        scm = self.analyzers.get('scm_analysis')
        if scm is not None:
            try:
                scm.collect_adjacencies(model)
            except Exception as e:
                logger.warning(f"SCM adjacency collection failed: {e}")

    def set_image_paths(self, paths) -> None:
        """Attach an ordered list of per-sample image paths for the case
        study gallery. Must be called after inference (so len(paths) ==
        number of collected samples) and before finalize()."""
        cs = self.analyzers.get('case_study')
        if cs is not None:
            try:
                cs.set_image_paths(paths)
            except Exception as e:
                logger.warning(f"CaseStudy image-path injection failed: {e}")

    def set_method_labels(self, label_spe) -> None:
        """Attach per-sample specific-method labels (FF-DF, FF-F2F, ...).
        Enables per-method SCM fingerprint radars."""
        scm = self.analyzers.get('scm_analysis')
        if scm is not None and hasattr(scm, 'set_method_labels'):
            try:
                scm.set_method_labels(label_spe)
            except Exception as e:
                logger.warning(f"SCM method-label injection failed: {e}")

    def finalize(self, save_dir: str) -> dict:
        """
        Run analysis + visualization on all analyzers.
        Returns combined results dict and saves outputs to save_dir.
        """
        os.makedirs(save_dir, exist_ok=True)
        combined_results = {}
        report_lines = [
            '=' * 70,
            'NeSyDeFake Interpretability Report',
            f'Causal branch type: {self.causal_type}',
            '=' * 70,
            '',
        ]

        for name, analyzer in self.analyzers.items():
            try:
                results = analyzer.analyze()
                combined_results[name] = results

                # Log key metrics
                if results:
                    report_lines.append(f'--- {name} ---')
                    for k, v in results.items():
                        if isinstance(v, (int, float, str)):
                            report_lines.append(f'  {k}: {v}')
                        elif isinstance(v, dict) and len(v) <= 10:
                            for kk, vv in v.items():
                                if isinstance(vv, (int, float, str)):
                                    report_lines.append(f'  {k}.{kk}: {vv}')
                    report_lines.append('')
            except Exception as e:
                logger.warning(f"Analyzer {name} analyze failed: {e}")
                combined_results[name] = {'error': str(e)}

        # Visualization
        for name, analyzer in self.analyzers.items():
            try:
                analyzer.visualize(save_dir)
            except Exception as e:
                logger.warning(f"Analyzer {name} visualize failed: {e}")

        # Per-sample explanations (top uncertain + top confident)
        explanations = self._generate_explanations()
        if explanations:
            report_lines.append('--- Sample Explanations ---')
            report_lines.extend(explanations)
            combined_results['sample_explanations'] = explanations

        # Save outputs
        viz.save_json_summary(combined_results, os.path.join(save_dir, 'summary.json'))
        viz.save_text_report(report_lines, os.path.join(save_dir, 'report.txt'))

        logger.info(f"Interpretability report saved to {save_dir}")
        return combined_results

    def _generate_explanations(self) -> List[str]:
        """Generate multi-level explanations for selected samples."""
        # Find sample indices: top uncertain + top confident
        edl = self.analyzers.get('edl_uncertainty')
        if edl is None or not hasattr(edl, '_uncertainty'):
            return []

        uncertainty = edl._uncertainty
        n = len(uncertainty)
        k = min(self.num_explain_samples // 2, n)
        if k == 0:
            return []

        # Top uncertain
        top_uncertain = np.argsort(-uncertainty)[:k]
        # Top confident (lowest uncertainty)
        top_confident = np.argsort(uncertainty)[:k]
        indices = list(top_uncertain) + list(top_confident)

        lines = []
        for idx in indices:
            sample_lines = [f'\n  Sample {idx}:']
            for analyzer in self.analyzers.values():
                try:
                    explanation = analyzer.explain_sample(idx)
                    if explanation:
                        sample_lines.append(f'    {explanation}')
                except Exception:
                    pass
            lines.extend(sample_lines)

        return lines

    def explain_samples(self, indices: List[int]) -> List[str]:
        """Generate multi-level explanations for specific sample indices."""
        results = []
        for idx in indices:
            parts = []
            for analyzer in self.analyzers.values():
                try:
                    explanation = analyzer.explain_sample(idx)
                    if explanation:
                        parts.append(explanation)
                except Exception:
                    pass
            results.append(' | '.join(parts))
        return results
