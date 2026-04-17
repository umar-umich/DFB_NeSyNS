"""
interpretability/analyzers/tsne_analyzer.py
===========================================
t-SNE visualization of fused feature embeddings.

Supports three sample-selection modes on top of the full test set:
    * ``best``   — top-K *correctly* classified samples with the highest
                   confidence in the true class (clearest, most confident cases).
    * ``worst``  — top-K *mis*-classified samples with the highest confidence
                   in the wrong class (most damaging failures).
    * ``random`` — K uniformly-random samples (sanity / overview scatter).

Config (under ``interpretability.tsne``)::

    interpretability:
      tsne:
        enabled: true
        mode: 'worst'         # 'best' | 'worst' | 'random' | 'all'
        top_k: 500            # sample budget for t-SNE
        perplexity: 30
        n_iter: 1000
        feature_key: 'feat'   # key in predictions dict ('feat' or 'l2_embeddings')
        seed: 42

If ``mode`` is ``'all'`` the analyzer produces one plot per mode.
"""

import os
from typing import List, Optional

import numpy as np

from .base import BaseAnalyzer
from .. import visualization as viz


class TSNEEmbeddingAnalyzer(BaseAnalyzer):
    name = 'tsne'

    # Collected per-batch; feature_key is overridable via config.
    # Default required_keys covers the most useful signals; we override in
    # __init__ to honour the configured feature_key.
    required_keys = ('feat', 'prob')

    _VALID_MODES = ('best', 'worst', 'random', 'all')

    def __init__(
        self,
        mode: str = 'worst',
        top_k: int = 500,
        perplexity: int = 30,
        n_iter: int = 1000,
        feature_key: str = 'feat',
        seed: int = 42,
    ):
        mode = str(mode).lower()
        if mode not in self._VALID_MODES:
            raise ValueError(
                f"tsne mode must be one of {self._VALID_MODES}, got {mode}")

        # Honour user-specified feature key by redefining required_keys
        # *before* calling super().__init__ (which allocates the buffers).
        self.feature_key = feature_key
        self.required_keys = (feature_key, 'prob')

        super().__init__()

        self.mode = mode
        self.top_k = int(top_k)
        self.perplexity = int(perplexity)
        self.n_iter = int(n_iter)
        self.seed = int(seed)

    # ── sample selection ──────────────────────────────────────────────────

    @staticmethod
    def _confidence_in_true_class(probs: np.ndarray,
                                  labels: np.ndarray) -> np.ndarray:
        """p(true class) — higher is more confidently correct."""
        return np.where(labels == 1, probs, 1.0 - probs)

    def _select(
        self,
        mode: str,
        feats: np.ndarray,
        labels: np.ndarray,
        probs: np.ndarray,
        preds: np.ndarray,
        correct: np.ndarray,
    ):
        """Return (indices, title_suffix) for the chosen selection mode."""
        n = len(feats)
        k = min(self.top_k, n)
        if k <= 0:
            return np.array([], dtype=int), mode

        rng = np.random.default_rng(self.seed)

        if mode == 'random':
            idx = rng.choice(n, size=k, replace=False)
            return np.sort(idx), 'random'

        if mode == 'best':
            # Highest confidence among correctly classified samples.
            conf_true = self._confidence_in_true_class(probs, labels)
            pool = np.where(correct)[0]
            if pool.size == 0:
                return np.array([], dtype=int), 'best_empty'
            scores = conf_true[pool]
            order = np.argsort(-scores)[:k]
            return pool[order], 'best'

        if mode == 'worst':
            # Highest confidence among misclassified samples
            # = lowest p(true class) among wrong predictions.
            conf_true = self._confidence_in_true_class(probs, labels)
            pool = np.where(~correct)[0]
            if pool.size == 0:
                return np.array([], dtype=int), 'worst_empty'
            scores = conf_true[pool]  # lower == more confidently wrong
            order = np.argsort(scores)[:k]
            return pool[order], 'worst'

        raise ValueError(f"Unknown mode {mode}")

    # ── analysis ──────────────────────────────────────────────────────────

    def analyze(self) -> dict:
        feats = self._stack(self.feature_key)
        probs = self._stack('prob')
        if feats is None or probs is None or len(self._labels) == 0:
            return {'skipped': 'no features or probs collected'}

        labels = self._all_labels.astype(int).reshape(-1)
        # Prob for positive class (fake=1) is expected to be 1-D.
        if probs.ndim > 1:
            probs = probs.reshape(-1)
        preds = (probs >= 0.5).astype(int)
        correct = preds == labels

        if feats.ndim > 2:
            feats = feats.reshape(feats.shape[0], -1)

        # Stash for visualize()
        self._feats = feats
        self._labels_flat = labels
        self._probs = probs
        self._preds = preds
        self._correct = correct

        modes = ('best', 'worst', 'random') if self.mode == 'all' else (self.mode,)
        summary = {
            'feature_key': self.feature_key,
            'feature_dim': int(feats.shape[1]),
            'n_total': int(feats.shape[0]),
            'n_correct': int(correct.sum()),
            'n_wrong': int((~correct).sum()),
            'top_k_requested': self.top_k,
            'modes': list(modes),
        }
        return summary

    # ── visualization ─────────────────────────────────────────────────────

    def visualize(self, save_dir: str) -> None:
        if not hasattr(self, '_feats'):
            return

        try:
            from sklearn.manifold import TSNE
        except ImportError:
            # Stay silent in the log; engine will catch & warn.
            raise RuntimeError(
                "sklearn is required for t-SNE plots. pip install scikit-learn")

        out_dir = os.path.join(save_dir, 'tsne')
        os.makedirs(out_dir, exist_ok=True)

        modes = ('best', 'worst', 'random') if self.mode == 'all' else (self.mode,)

        for mode in modes:
            idx, tag = self._select(
                mode,
                self._feats, self._labels_flat, self._probs,
                self._preds, self._correct,
            )
            if idx.size < 3:
                continue

            X = self._feats[idx]
            y = self._labels_flat[idx]
            correct_sub = self._correct[idx]

            # sklearn requires perplexity < n_samples.
            perp = max(2, min(self.perplexity, max(2, X.shape[0] - 1)))

            tsne = TSNE(
                n_components=2,
                perplexity=perp,
                max_iter=self.n_iter,
                init='pca',
                learning_rate='auto',
                random_state=self.seed,
            )
            Z = tsne.fit_transform(X)

            # Plot: colour by (label, correct) group to make failures visible.
            groups = np.where(
                y == 1,
                np.where(correct_sub, 0, 1),   # 0=fake-correct, 1=fake-wrong
                np.where(correct_sub, 2, 3),   # 2=real-correct, 3=real-wrong
            )
            group_labels = {
                0: 'fake (correct)',
                1: 'fake (wrong)',
                2: 'real (correct)',
                3: 'real (wrong)',
            }

            title = (f"t-SNE [{tag}] top-{len(idx)} of {len(self._feats)} "
                     f"({self.feature_key})")
            viz.plot_scatter(
                Z[:, 0], Z[:, 1],
                title=title,
                xlabel='t-SNE dim 1',
                ylabel='t-SNE dim 2',
                save_path=os.path.join(out_dir, f'tsne_{tag}.png'),
                hue=groups,
                hue_labels=group_labels,
                alpha=0.75,
                s=18,
            )

    # ── per-sample explanation (not really sample-aligned: summary line) ──

    def explain_sample(self, idx: int) -> str:
        if not hasattr(self, '_feats') or idx >= len(self._feats):
            return ''
        p = float(self._probs[idx])
        y = int(self._labels_flat[idx])
        ok = bool(self._correct[idx])
        return (f"tsne: label={y} prob_fake={p:.3f} "
                f"correct={ok}")
