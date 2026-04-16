"""
interpretability/analyzers/ccv_analysis.py
==========================================
Level 4-CCV: Causal Constraint Verification analysis — learned constraint
activations, per-group forensic anomaly radar, counterfactual mismatch.
"""

import os

import numpy as np

from .base import BaseAnalyzer
from .. import visualization as viz


class CCVAnalyzer(BaseAnalyzer):
    name = 'ccv_analysis'
    required_keys = ('violation_scores', 'anomaly_scores', 'counterfactual_residual')

    def __init__(self):
        super().__init__()
        try:
            from networks.nesy_defake.ccv_branch import FORENSIC_GROUPS
            self.forensic_group_names = [g[0] for g in FORENSIC_GROUPS]
        except ImportError:
            self.forensic_group_names = [
                'boundary_texture', 'symmetry_color', 'patch_noise',
                'srm_noise', 'fft_spectral',
            ]

    def analyze(self) -> dict:
        vscores = self._stack('violation_scores')
        anomaly = self._stack('anomaly_scores')
        cf_res = self._stack('counterfactual_residual')
        labels = self._all_labels
        if vscores is None and anomaly is None and cf_res is None:
            return {}

        results = {}
        real_mask = labels == 0
        fake_mask = labels == 1

        # Learned constraint analysis
        if vscores is not None:
            real_m = vscores[real_mask].mean(axis=0) if real_mask.any() else np.zeros(vscores.shape[1])
            fake_m = vscores[fake_mask].mean(axis=0) if fake_mask.any() else np.zeros(vscores.shape[1])
            results['learned_constraint_mean_real'] = real_m.tolist()
            results['learned_constraint_mean_fake'] = fake_m.tolist()
            gaps = np.abs(fake_m - real_m)
            top_idx = np.argsort(-gaps)[:5]
            results['top_discriminative_constraints'] = [
                {'idx': int(i), 'gap': float(gaps[i])} for i in top_idx
            ]

        # Forensic anomaly analysis
        if anomaly is not None:
            per_group = {}
            for gi, gname in enumerate(self.forensic_group_names):
                if gi < anomaly.shape[1]:
                    per_group[gname] = {
                        'mean_real': float(anomaly[real_mask, gi].mean()) if real_mask.any() else 0.0,
                        'mean_fake': float(anomaly[fake_mask, gi].mean()) if fake_mask.any() else 0.0,
                    }
            results['forensic_anomaly_per_group'] = per_group

        # Counterfactual mismatch
        if cf_res is not None:
            cf = cf_res.squeeze(-1) if cf_res.ndim > 1 else cf_res
            results['cf_mismatch_mean_real'] = float(cf[real_mask].mean()) if real_mask.any() else 0.0
            results['cf_mismatch_mean_fake'] = float(cf[fake_mask].mean()) if fake_mask.any() else 0.0

        self._vscores = vscores
        self._anomaly = anomaly
        self._cf_res = cf_res
        self._cached_labels = labels
        return results

    def visualize(self, save_dir: str) -> None:
        if not hasattr(self, '_cached_labels'):
            return
        labels = self._cached_labels
        real_mask = labels == 0
        fake_mask = labels == 1

        if self._anomaly is not None:
            n_groups = min(self._anomaly.shape[1], len(self.forensic_group_names))
            real_means = self._anomaly[real_mask, :n_groups].mean(axis=0) if real_mask.any() else np.zeros(n_groups)
            fake_means = self._anomaly[fake_mask, :n_groups].mean(axis=0) if fake_mask.any() else np.zeros(n_groups)
            viz.plot_radar(
                {'Real': real_means.tolist(), 'Fake': fake_means.tolist()},
                self.forensic_group_names[:n_groups],
                title='Forensic Anomaly by Group (CCV)',
                save_path=os.path.join(save_dir, 'ccv_forensic_radar.png'),
            )

        if self._vscores is not None:
            viz.plot_grouped_bar(
                {'Real': self._vscores[real_mask].mean(axis=0).tolist() if real_mask.any() else [],
                 'Fake': self._vscores[fake_mask].mean(axis=0).tolist() if fake_mask.any() else []},
                [f'C{i}' for i in range(self._vscores.shape[1])],
                title='Learned Constraint Activations by Class',
                save_path=os.path.join(save_dir, 'ccv_learned_constraints.png'),
                ylabel='Mean Activation',
            )

        if self._cf_res is not None:
            cf = self._cf_res.squeeze(-1) if self._cf_res.ndim > 1 else self._cf_res
            viz.plot_histogram(
                {'Real': cf[real_mask], 'Fake': cf[fake_mask]},
                title='Counterfactual Mismatch Distribution',
                xlabel='Mismatch Score',
                save_path=os.path.join(save_dir, 'ccv_counterfactual_hist.png'),
            )

    def explain_sample(self, idx: int) -> str:
        parts = []
        if self._anomaly is not None:
            scores = self._anomaly[idx]
            top_gi = int(np.argmax(scores))
            gname = self.forensic_group_names[top_gi] if top_gi < len(self.forensic_group_names) else f'group_{top_gi}'
            parts.append(f"top anomaly: {gname}={scores[top_gi]:.3f}")
        if self._vscores is not None:
            top_ci = int(np.argmax(self._vscores[idx]))
            parts.append(f"top learned constraint: C{top_ci}={self._vscores[idx, top_ci]:.3f}")
        if self._cf_res is not None:
            cf = self._cf_res[idx]
            cf_val = cf.item() if cf.ndim == 0 else cf[0]
            parts.append(f"counterfactual mismatch={cf_val:.3f}")
        return f"[CCV] {', '.join(parts)}" if parts else ''
