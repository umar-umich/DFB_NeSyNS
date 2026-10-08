#!/usr/bin/env python3
"""Phase 3 (GATE OVERRIDE): vector evidence, additive versus low-rank interactions.

    <discern_ext python> upgrade/fusion/phase3_vectors.py

Run on UMAR's explicit instruction after the Phase 2 gate failed; `override_record.json` states what
was overridden. These numbers sit outside the pre-registered protocol and carry selection exposure
the main run does not.

  A   g_k = c_k + sum_v a_vk . s_v
  I   A plus three pairwise low-rank terms, s_i^T U_ij,k V_ij,k^T s_j with r = 2
  e_k = softplus(g_k), alpha = e + 1

These vectors are LEARNED SIGNED FEATURES. Their coordinates are not independent evidence counts and
are never summed as pseudo-counts. APP and OM share a trunk, so the three views are not independent
observations and nothing here assumes they are.

Controls the brief requires, all run:
  * I versus A
  * a parameter-matched additive head (A widened to I's budget)
  * zeroed interactions with identical remaining weights, which must recover A exactly
One interaction factor is initialised near zero and the other nonzero: initialising both to zero
leaves no gradient path through the product.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
sys.path[:0] = [str(ROOT / "upgrade"), str(ROOT / "upgrade/fusion"),
                str(ROOT / "analysis/tbiom/pcr6p"), str(ROOT / "analysis/tbiom/grouped"),
                str(ROOT / "analysis/tbiom"), str(ROOT / "analysis/tbiom/temporal"),
                "/data/umar/Repos/DISCERN_Ext"]
import phase2 as P2   # noqa: E402

D, RANK, SEEDS = 8, 2, P2.SEEDS


class VecHead(nn.Module):
    def __init__(self, dims, d=D, rank=RANK, interactions=False, extra=0):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(x, d) for x in dims])
        self.a = nn.Parameter(torch.zeros(len(dims), 2, d)); nn.init.normal_(self.a, std=0.05)
        self.c = nn.Parameter(torch.zeros(2))
        self.inter = interactions
        self.nv = len(dims)
        if interactions:
            pairs = [(i, j) for i in range(self.nv) for j in range(i + 1, self.nv)]
            self.pairs = pairs
            self.U = nn.ParameterList(); self.V = nn.ParameterList()
            for _ in pairs:
                u = nn.Parameter(torch.zeros(2, d, rank)); nn.init.normal_(u, std=0.05)
                v = nn.Parameter(torch.zeros(2, d, rank))        # ZERO: the other factor is nonzero,
                self.U.append(u); self.V.append(v)               # so the product still has a gradient
        # `extra` widens the additive head so A can be parameter-matched to I
        self.extra = nn.Linear(len(dims) * d, extra) if extra else None
        self.extra_out = nn.Linear(extra, 2) if extra else None
        self.score = nn.Linear(6, 2)

    def forward(self, s, fs):
        S = [torch.tanh(p(Fn.layer_norm(h, (h.shape[-1],)))) for p, h in zip(self.proj, fs)]
        g = self.c.unsqueeze(0) + sum(torch.einsum("bd,kd->bk", S[v], self.a[v]) for v in range(self.nv))
        if self.inter:
            for (i, j), U, V in zip(self.pairs, self.U, self.V):
                li = torch.einsum("bd,kdr->bkr", S[i], U)
                lj = torch.einsum("bd,kdr->bkr", S[j], V)
                g = g + (li * lj).sum(-1)
        if self.extra is not None:
            g = g + self.extra_out(Fn.gelu(self.extra(torch.cat(S, -1))))
        return Fn.softplus(g + self.score(s))


def run(kind, dims, ev, feats, y, dom, grp, folds, lr, epochs, seed, extra=0, zero_inter=False):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    p_all = np.zeros(len(y))
    for tri, tei in folds:
        torch.manual_seed(seed); np.random.seed(seed)
        m = VecHead(dims, interactions=(kind == "I"), extra=extra).to(dev)
        if zero_inter:
            with torch.no_grad():
                for u in m.U:
                    u.zero_()
        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=P2.WD)
        tr_s, te_s = P2.standardise(np.log1p(ev[tri]), np.log1p(ev[tei]))
        for ep in range(epochs):
            m.train()
            idx = np.random.default_rng(seed * 97 + ep).permutation(len(tri))
            for i in range(0, len(idx) - 1, P2.BATCH):
                j = idx[i:i + P2.BATCH]
                s = torch.from_numpy(tr_s[j]).to(dev)
                fs = [torch.from_numpy(feats[v][tri][j]).to(dev) for v in P2.VIEWS]
                e = m(s, fs)
                loss = P2.edl(e, torch.from_numpy(y[tri][j]).to(dev), ep, epochs)
                opt.zero_grad(set_to_none=True); loss.backward()
                if zero_inter:
                    for u in m.U:
                        u.grad = None
                opt.step()
        m.eval()
        with torch.no_grad():
            s = torch.from_numpy(te_s).to(dev)
            fs = [torch.from_numpy(feats[v][tei]).to(dev) for v in P2.VIEWS]
            a = m(s, fs) + 1.0
            p_all[tei] = (a / a.sum(1, keepdim=True))[:, 1].cpu().numpy()
    return p_all, sum(q.numel() for q in m.parameters() if q.requires_grad)


def main() -> int:
    run_id = (ROOT / "outputs/fusion_experiments/CURRENT_P3_RUN").read_text().strip()
    out = ROOT / "outputs/fusion_experiments" / run_id
    main_run = (ROOT / "outputs/fusion_experiments/CURRENT_RUN").read_text().strip()
    sel = json.load(open(ROOT / "outputs/fusion_experiments" / main_run / "phase2_search.json"))
    gate = json.load(open(ROOT / "outputs/fusion_experiments" / main_run / "gate_decision.json"))

    t, ev, feats, y, grp = P2.build()
    dom = t["source"].astype(str).to_numpy()
    dims = tuple(feats[v].shape[1] for v in P2.VIEWS)
    folds = P2.outer_folds(grp, y, 5)
    lr, epochs = sel["F"]["lr"], max(1, sel["F"]["epochs"] + 1)

    # parameter match: widen A until it reaches I's count
    nI = sum(p.numel() for p in VecHead(dims, interactions=True).parameters())
    nA = sum(p.numel() for p in VecHead(dims).parameters())
    extra = 0
    for e_ in range(0, 400, 2):
        if abs(sum(p.numel() for p in VecHead(dims, extra=e_).parameters()) - nI) < abs(nA - nI):
            nA, extra = sum(p.numel() for p in VecHead(dims, extra=e_).parameters()), e_
    print(f"I {nI} params; A {sum(p.numel() for p in VecHead(dims).parameters())}; "
          f"A-matched {nA} (extra width {extra})", flush=True)

    res, preds = {}, {}
    for name, kw in (("A", {}), ("A-matched", {"extra": extra}),
                     ("I", {}), ("I-zeroed", {"zero_inter": True})):
        kind = "I" if name.startswith("I") else "A"
        per = []
        for seed in SEEDS:
            p_, n_ = run(kind, dims, ev, feats, y, dom, grp, folds, lr, epochs, seed, **kw)
            sc = P2.macro_auroc(y, p_, dom); per.append(sc); preds[f"{name}_s{seed}"] = p_
            print(f"  {name:11s} seed {seed}: macro {sc:.4f}  ({n_} params)", flush=True)
        res[name] = {"per_seed": per, "mean": float(np.mean(per)), "sd": float(np.std(per))}

    mean_of = lambda k: np.mean([preds[f"{k}_s{s}"] for s in SEEDS], 0)
    contrasts = {}
    for nm, x, b in (("I - A", "I", "A"), ("I - A-matched", "I", "A-matched"),
                     ("I - I-zeroed", "I", "I-zeroed")):
        m_, lo, hi = P2.paired_bootstrap(y, dom, grp, mean_of(x), mean_of(b))
        contrasts[nm] = {"point": res[x]["mean"] - res[b]["mean"], "lo97.5": lo, "hi": hi}
    for nm, ref in (("A - S", gate["macro"]["S"]), ("I - S", gate["macro"]["S"]),
                    ("I - B1refit", gate["macro"]["B1_refit"])):
        k = nm.split(" ")[0]
        contrasts[nm] = {"point": res[k]["mean"] - ref, "interval": "reference from the main run"}
    np.savez(out / "phase3_preds.npz", y=y, dom=dom, grp=grp, **preds)
    json.dump({"run_id": run_id, "arms": res, "contrasts": contrasts,
               "reference": {"S": gate["macro"]["S"], "F": gate["macro"]["F"],
                             "B1_refit": gate["macro"]["B1_refit"]}},
              open(out / "phase3_results.json", "w"), indent=1)
    print("\n=== contrasts ===")
    for k, v in contrasts.items():
        lo = v.get("lo97.5")
        print(f"  {k:16s} {v['point']:+.4f}" + (f"   97.5% lower {lo:+.4f}" if lo is not None else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
