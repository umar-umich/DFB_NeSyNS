"""
scripts/temporal_ladder.py  —  Phase-2 Task 9 (S3 data-prep): temporal ladder.

Builds per-video trajectory tensors from the per-frame CSVs (channels ordered by
frame_idx: p_fake, disagreement_d, the 18 rule violations, the 5 CCV anomaly
groups) and compares three video-scoring rungs of increasing opacity:

  Rung A  frame-mean p_fake                       (reference; no fitting)
  Rung B  transparent trajectory stats (mean, var, max, change-point strength,
          persistence per channel) + logistic head
  Rung C  1-layer GRU (hidden 32) over the sequence

The B and C heads are FIT ON FF++ ONLY (guarded) and evaluated frozen on OOD.
Reports video AUC per rung per dataset.

Usage (repo root):
    python scripts/temporal_ladder.py --run logs/test/p4b_full_ccv_reeval
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nesy_csv_utils as U
import risk_head as RH

SEED = 3407
T_LEN = 32


def channel_list(df):
    ch = ['p_fake', 'disagreement_d']
    ch += U.rule_columns(df)
    ch += sorted(c for c in df.columns if c.startswith('anomaly_'))
    return [c for c in ch if c in df.columns]


def build_sequences(df, channels, T=T_LEN):
    """Per-video (T, F) trajectories ordered by frame_idx; pad/truncate to T."""
    d = df.copy()
    if 'method' not in d.columns:
        d['method'] = ''
    d['method'] = d['method'].fillna('').astype(str)

    def fkey(s):
        import re
        try:
            return int(re.sub(r'\D', '', str(s)) or 0)
        except ValueError:
            return 0
    d['_fk'] = d['frame_idx'].map(fkey)
    X, y = [], []
    for (_, _), g in d.groupby(['method', 'video_id'], sort=False):
        g = g.sort_values('_fk')
        arr = g[channels].to_numpy(dtype=float)
        arr = np.nan_to_num(arr, nan=0.0)
        if len(arr) >= T:
            arr = arr[:T]
        else:
            pad = np.repeat(arr[-1:], T - len(arr), axis=0)   # repeat last frame
            arr = np.concatenate([arr, pad], axis=0)
        X.append(arr)
        y.append(int(g['label'].iloc[0]))
    return np.stack(X), np.array(y)


def transparent_stats(X):
    """(N,T,F) → (N, 5F) transparent per-channel stats.

    mean, var, max, change-point strength (max |Δ|), persistence (lag-1 autocorr).
    """
    mean = X.mean(1)
    var = X.var(1)
    mx = X.max(1)
    dif = np.abs(np.diff(X, axis=1))
    cps = dif.max(1) if dif.shape[1] > 0 else np.zeros_like(mean)
    # lag-1 autocorrelation per channel
    Xc = X - mean[:, None, :]
    num = (Xc[:, 1:, :] * Xc[:, :-1, :]).sum(1)
    den = (Xc ** 2).sum(1) + 1e-8
    persist = num / den
    return np.concatenate([mean, var, mx, cps, persist], axis=1)


class GRUHead(nn.Module):
    def __init__(self, f, hidden=32):
        super().__init__()
        self.gru = nn.GRU(f, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[-1]).squeeze(1)


def train_gru(Xtr, ytr, f, epochs=60, lr=1e-3):
    torch.manual_seed(SEED)
    model = GRUHead(f)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(model(xt), yt)
        loss.backward()
        opt.step()
    model.eval()
    return model


def auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='logs/test/p4b_full_ccv_reeval')
    ap.add_argument('--fit_dataset', default=RH.FF)
    ap.add_argument('--ood', nargs='+', default=RH.ALL_OOD)
    ap.add_argument('--out', default='results/temporal_ladder.md')
    args = ap.parse_args()

    if args.fit_dataset != RH.FF:
        raise SystemExit(f"[GUARD] temporal ladder B/C heads fit on {RH.FF} only.")

    ff = U.load_frames(args.run, RH.FF)
    if ff is None:
        raise SystemExit(f"missing FF++ CSV under {args.run}")
    channels = channel_list(ff)
    F = len(channels)

    Xff, yff = build_sequences(ff, channels)
    # Rung B: transparent stats + logistic (fit on FF++)
    Bff = transparent_stats(Xff)
    scB = StandardScaler().fit(Bff)
    clfB = LogisticRegression(max_iter=2000, class_weight='balanced')
    clfB.fit(scB.transform(Bff), yff)
    # Rung C: GRU (fit on FF++), standardize per-channel over the flattened frames
    flat = Xff.reshape(-1, F)
    scC = StandardScaler().fit(flat)
    XffN = scC.transform(flat).reshape(Xff.shape)
    gru = train_gru(XffN, yff, F)

    datasets = [RH.FF] + list(args.ood)
    rows = []
    for ds in datasets:
        df = U.load_frames(args.run, ds)
        if df is None:
            continue
        X, y = build_sequences(df, channels)
        a = auc(y, X[:, :, 0].mean(1))                       # Rung A: mean p_fake
        b = auc(y, clfB.predict_proba(scB.transform(transparent_stats(X)))[:, 1])
        Xn = scC.transform(X.reshape(-1, F)).reshape(X.shape)
        with torch.no_grad():
            c = auc(y, torch.sigmoid(
                gru(torch.tensor(Xn, dtype=torch.float32))).numpy())
        rows.append((ds, len(y), a, b, c))

    L = ['# Temporal ladder — video AUC per rung (Task 9 / S3)\n']
    L.append(f'- Run: `{args.run}`  |  channels (F={F}): p_fake, disagreement_d, '
             '18 rules, 5 anomaly groups  |  B/C fit on **FF++ only**, frozen on OOD')
    L.append('')
    L.append('| dataset | n | Rung A (mean) | Rung B (stats+LR) | Rung C (GRU) |')
    L.append('|---|---|---|---|---|')
    for ds, n, a, b, c in rows:
        tag = ' *(in-domain)*' if ds == RH.FF else ''
        L.append(f'| {ds}{tag} | {n} | {a:.4f} | {b:.4f} | {c:.4f} |')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    open(args.out, 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print(f'\nWritten {args.out}')


if __name__ == '__main__':
    main()
