"""
scripts/run_faithfulness.py
===========================
Predicate-substrate intervention faithfulness for the full DeFakeNet
checkpoint on CDFv2.

For every correctly-classified fake frame:
  1. Run a baseline forward (no mask) → record Ev_sym_base, prob_base,
     and the per-frame violations vector.
  2. For k ∈ {1, 3, 5}, run two interventions:
       • top-k:  zero the k retained predicates with the highest
                 violation values for this sample.
       • random: zero k retained predicates chosen uniformly without
                 replacement (seeded per-sample for reproducibility).
     Record Ev_sym_after, prob_after.

Aggregate metrics (per intervention type, per k):
  • drop_rate    : fraction of samples with Ev_sym dropping by > 50 %.
  • flip_rate    : fraction of samples whose final hard prediction
                   flips after intervention.
  • mean_drop_pct: mean relative Ev_sym drop, in %.

Output:  results/faithfulness/CDFv2.json (incremental write per k+type)

This runner reuses the existing detector / dataset stack:
  - DETECTOR registry from training.detectors
  - NeSyDeFakeDataset from training.dataset.nesy_defake_dataset
  - The new ``predicate_mask`` kwarg added to ConceptBranch.forward
    (the only additional API surface; back-compat default is None).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from copy import copy
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
TRAINING_DIR = REPO_ROOT / 'training'
sys.path.insert(0, str(TRAINING_DIR))   # for `dataset`, `detectors`, `networks`
sys.path.insert(0, str(SCRIPT_DIR))     # for `calibration_metrics`

from dataset.nesy_defake_dataset import NeSyDeFakeDataset  # noqa: E402
from detectors import DETECTOR  # noqa: E402

DEFAULT_DETECTOR_CFG = (
    REPO_ROOT / 'training' / 'config' / 'detector'
    / 'nesy_defake_ablation4_causal.yaml'
)
DEFAULT_CKPT = REPO_ROOT / 'checkpoints' / 'defakenet_full.pth'
DEFAULT_TARGET_DATASET = 'Celeb-DF-v2'
DEFAULT_OUT = REPO_ROOT / 'results' / 'faithfulness' / 'CDFv2.json'

K_VALUES = (1, 3, 5)


def _load_config(detector_cfg: Path) -> dict:
    """Mirror what training/test.py:main does (config + test_config merge)."""
    with open(detector_cfg) as f:
        config = yaml.safe_load(f)
    test_cfg_path = TRAINING_DIR / 'config' / 'test_config.yaml'
    if test_cfg_path.exists():
        with open(test_cfg_path) as f:
            test_cfg = yaml.safe_load(f) or {}
            for k, v in test_cfg.items():
                if k not in config or k == 'label_dict':
                    config[k] = v
    return config


def _build_model(config: dict, weights_path: Path,
                 device: torch.device) -> torch.nn.Module:
    cls = DETECTOR[config['model_name']]
    model = cls(config).to(device)
    state = torch.load(str(weights_path), map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f'[faithfulness] load_state_dict: '
              f'missing={len(missing)}, unexpected={len(unexpected)}',
              flush=True)
    model.eval()
    return model


def _get_concept_branch(model: torch.nn.Module):
    cb = getattr(model, 'concept_branch', None)
    if cb is None:
        raise RuntimeError(
            'Model has no concept_branch attribute — cannot run '
            'predicate-substrate intervention.')
    return cb


def _build_loader(config: dict, target_dataset: str,
                  batch_size: int) -> torch.utils.data.DataLoader:
    cfg = copy(config)
    cfg['test_dataset'] = [target_dataset]
    cfg['test_batchSize'] = batch_size
    cfg['mode'] = 'test'
    test_set = NeSyDeFakeDataset(cfg, mode='test')
    return torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.get('workers', 4),
        collate_fn=getattr(test_set, 'collate_fn', None),
        drop_last=False,
    )


def _to_device(data_dict: dict, device: torch.device) -> dict:
    out = {}
    for k, v in data_dict.items():
        out[k] = v.to(device, non_blocking=True) \
            if isinstance(v, torch.Tensor) else v
    return out


@torch.no_grad()
def _forward_with_mask(
    model: torch.nn.Module,
    data_dict: dict,
    mask: torch.Tensor | None,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (prob_fake (B,), ev_sym_total (B,)) where ev_sym_total
    sums concept_evidence across the 2 classes.
    """
    dd = _to_device(data_dict, device)
    if mask is not None:
        dd['predicate_mask'] = mask.to(device)
    pred = model(dd, inference=True)
    prob = pred['prob']
    ev_sym = pred.get('concept_evidence')
    if ev_sym is None:
        raise RuntimeError(
            'Model did not expose concept_evidence; the symbolic stream '
            'must be active for the faithfulness protocol.')
    return prob.detach().float().cpu(), ev_sym.detach().float().cpu().sum(dim=1)


def _build_topk_mask(viol: torch.Tensor, k: int) -> torch.Tensor:
    """For each row of viol (B, K_pred), return a (B, K_pred) {0,1}
    tensor whose top-k by value are zeroed and the rest are 1.
    """
    B, K = viol.shape
    mask = torch.ones((B, K), dtype=torch.float32)
    if k <= 0:
        return mask
    k = min(k, K)
    top_idx = torch.topk(viol, k=k, dim=1).indices  # (B, k)
    mask.scatter_(1, top_idx, 0.0)
    return mask


def _build_random_mask(num_predicates: int, batch_size: int,
                       k: int, seed_seed: int) -> torch.Tensor:
    """Independent-per-sample random k-zero mask. Fully deterministic
    given (seed_seed, batch_index)."""
    mask = torch.ones((batch_size, num_predicates), dtype=torch.float32)
    if k <= 0:
        return mask
    k = min(k, num_predicates)
    rng = np.random.default_rng(seed_seed)
    for b in range(batch_size):
        idx = rng.choice(num_predicates, size=k, replace=False)
        mask[b, idx] = 0.0
    return mask


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--detector-cfg', type=Path, default=DEFAULT_DETECTOR_CFG)
    p.add_argument('--checkpoint-path', type=Path, default=DEFAULT_CKPT)
    p.add_argument('--target-dataset', default=DEFAULT_TARGET_DATASET)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--out-path', type=Path, default=DEFAULT_OUT)
    p.add_argument('--seed', type=int, default=42,
                   help='Base seed for the random-k mask.')
    args = p.parse_args()

    args.out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = _load_config(args.detector_cfg)
    model = _build_model(config, args.checkpoint_path, device)
    cb = _get_concept_branch(model)
    rules = getattr(cb, 'consistency_rules', None)
    num_predicates = getattr(rules, 'k', None) or getattr(
        rules, 'NUM_TRAINING_RULES_V7', None) or 12
    print(f'[faithfulness] num_predicates={num_predicates}', flush=True)

    loader = _build_loader(config, args.target_dataset, args.batch_size)

    # Per-intervention running counters
    counters: Dict[Tuple[str, int], dict] = {
        (kind, k): {'n': 0, 'drops': 0, 'flips': 0,
                    'rel_drop_sum': 0.0}
        for kind in ('topk', 'random') for k in K_VALUES
    }
    total_correct_fakes = 0

    for batch_idx, data_dict in enumerate(loader):
        # Coerce labels to binary (mirrors test.py behaviour).
        if 'label' in data_dict:
            data_dict['label'] = torch.where(
                data_dict['label'] != 0, 1, 0)

        # ── Baseline forward (no mask) ─────────────────────────────────
        dd_dev = _to_device(data_dict, device)
        with torch.no_grad():
            pred_base = model(dd_dev, inference=True)
        prob_base = pred_base['prob'].detach().float().cpu()
        ev_base = pred_base['concept_evidence'].detach().float().cpu().sum(dim=1)
        viol = pred_base['violations'].detach().float().cpu()

        labels = data_dict['label'].cpu()
        # Correctly-classified fakes: label == 1 and (prob >= 0.5)
        sel = (labels == 1) & (prob_base >= 0.5)
        if sel.sum().item() == 0:
            continue
        idx_keep = torch.where(sel)[0]
        total_correct_fakes += int(idx_keep.numel())

        # Subset everything to the kept rows.
        sub_data = {k: (v[idx_keep] if isinstance(v, torch.Tensor) else
                        [v[i] for i in idx_keep.tolist()])
                    for k, v in data_dict.items()}
        sub_prob_base = prob_base[idx_keep]
        sub_ev_base = ev_base[idx_keep]
        sub_viol = viol[idx_keep]
        B_sub = int(idx_keep.numel())

        # ── Each intervention, each k ──────────────────────────────────
        for k in K_VALUES:
            # top-k
            mask_topk = _build_topk_mask(sub_viol, k)
            prob_t, ev_t = _forward_with_mask(model, sub_data, mask_topk, device)
            _accumulate(counters[('topk', k)],
                        sub_prob_base, sub_ev_base, prob_t, ev_t)

            # random-k (deterministic seed per (k, batch))
            mask_rand = _build_random_mask(
                num_predicates=num_predicates,
                batch_size=B_sub,
                k=k,
                seed_seed=args.seed * 1_000_003 + batch_idx * 13 + k,
            )
            prob_r, ev_r = _forward_with_mask(model, sub_data, mask_rand, device)
            _accumulate(counters[('random', k)],
                        sub_prob_base, sub_ev_base, prob_r, ev_r)

        # Incremental write so partial runs are recoverable.
        _write_summary(args.out_path, counters, total_correct_fakes,
                       num_predicates)
        if (batch_idx + 1) % 10 == 0:
            print(f'[faithfulness] batches={batch_idx + 1} '
                  f'kept_fakes={total_correct_fakes}', flush=True)

    _write_summary(args.out_path, counters, total_correct_fakes,
                   num_predicates)
    print(f'[faithfulness] DONE — wrote {args.out_path}', flush=True)


def _accumulate(counter: dict, prob_base, ev_base, prob_after, ev_after):
    n = int(prob_base.numel())
    counter['n'] += n
    # Ev_sym drop > 50 %. Guard tiny baselines.
    safe_base = ev_base.clamp_min(1e-6)
    rel_drop = (ev_base - ev_after) / safe_base   # 1.0 = full drop
    counter['drops'] += int((rel_drop > 0.5).sum().item())
    counter['rel_drop_sum'] += float(rel_drop.sum().item())
    # Hard-prediction flip: base ≥ 0.5 → after < 0.5 (or vice-versa).
    pred_base = (prob_base >= 0.5).int()
    pred_after = (prob_after >= 0.5).int()
    counter['flips'] += int((pred_base != pred_after).sum().item())


def _write_summary(out_path: Path, counters: dict, total: int,
                   num_predicates: int) -> None:
    summary = {
        'dataset': 'CDFv2',
        'num_predicates': int(num_predicates),
        'total_correct_fakes': int(total),
        'metrics': {
            f'{kind}_k{k}': {
                'n': c['n'],
                'drop_rate_pct': (100.0 * c['drops'] / c['n']) if c['n'] else 0.0,
                'flip_rate_pct': (100.0 * c['flips'] / c['n']) if c['n'] else 0.0,
                'mean_rel_drop_pct': (100.0 * c['rel_drop_sum'] / c['n'])
                    if c['n'] else 0.0,
            }
            for (kind, k), c in counters.items()
        },
    }
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
