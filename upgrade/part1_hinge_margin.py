#!/usr/bin/env python3
"""Part 1 residue: the distribution of the hinge MARGIN, kappa - u_f, over the samples it is active on.

    <discern_ext python> upgrade/part1_hinge_margin.py

G0b established that the hinge is active often enough to pass its gate -- 65% of training samples
under equal trust, 9.8-13.5% under the deployed triples -- but it reported percentiles of kappa, not
of the quantity the loss actually sees. `max(0, kappa - u_f)` is what carries gradient, so "active"
counts samples where the margin is merely positive, including ones where it is ~1e-4 and the term is
numerically irrelevant. This reports the margin's quantiles over active samples, which is what the
IBDC brief's Option B follow-up asked for.

Everything is reused from `gate_g0b.py` -- same cache, same heads, same weight sets, same ordering --
so the active masks here are the ones that produced the G0b verdict and the two reports cannot drift.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
sys.path[:0] = [str(ROOT / "training/joint_ibdc"), str(ROOT / "analysis/tbiom/pcr6p"),
                str(ROOT / "analysis/tbiom/grouped"), str(ROOT / "analysis/tbiom"),
                str(ROOT / "analysis/tbiom/temporal")]
import fusion_torch as FT   # noqa: E402
import draws as DR          # noqa: E402
import gate_g0b as G        # noqa: E402

QS = [5, 25, 50, 75, 95]
OUT = ROOT / "analysis/upgrade"


def main() -> int:
    draws = 3
    missing = [d for d in range(draws) if not (G.CACHE / f"FFpp_train_B_d{d}.npz").exists()]
    if missing:
        print(f"cache missing for draws {missing}"); return 2

    sd = torch.load(G.CKPT, map_location="cpu", weights_only=False)["state_dict"]
    WS = G.weight_sets()
    pooled = {k: [] for k in WS}

    for d in range(draws):
        zs = np.load(G.CACHE / f"FFpp_train_B_d{d}.npz", allow_pickle=True)
        zt = np.load(G.CACHE / f"FFpp_train_T_B_d{d}.npz", allow_pickle=True)
        e_app = G.spatial_evidence(sd, zs["semantic"], "model.semantic_norm", "model.semantic_head")
        e_om = G.spatial_evidence(sd, zs["artifact"], "model.artifact_norm", "model.artifact_head")
        e_t_v = G.temporal_evidence(zt["feature"])
        tv = {str(v): i for i, v in enumerate(zt["video"])}
        frame_vid = [DR.video_of(str(k)) for k in zs["key"]]
        keep = np.array([i for i, v in enumerate(frame_vid) if v in tv])
        idx = np.array([tv[frame_vid[i]] for i in keep])
        e = torch.stack([e_t_v[idx], e_app[keep], e_om[keep]], dim=1)

        for name, w in WS.items():
            m = FT.opinion_of_evidence(e * torch.tensor(w).view(1, -1, 1))
            k = FT.kappa(m).numpy()
            uf = FT.vacuity_of(FT.fuse(m, plus=True)).numpy()
            margin = k - uf
            pooled[name].append(margin[margin > 0])
        print(f"  draw {d}: {len(keep)} frames", flush=True)

    rows = {}
    for name in WS:
        v = np.concatenate(pooled[name])
        q = np.percentile(v, QS)
        rows[name] = {"n_active": int(v.size), "mean": float(v.mean()),
                      **{f"p{p}": float(x) for p, x in zip(QS, q)},
                      "frac_active_above_1e-3": float((v > 1e-3).mean()),
                      "frac_active_above_1e-2": float((v > 1e-2).mean())}

    res = {"study": "Part 1 residue -- hinge margin kappa - u_f over ACTIVE samples",
           "quantiles": QS, "weight_sets": rows,
           "note": ("`active` in G0b means margin > 0. These quantiles show how much of that is "
                    "numerically meaningful: the fraction above 1e-3 and 1e-2 is reported beside "
                    "the percentiles so a large active fraction of negligible margins cannot read "
                    "as a strong signal.")}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "part1_hinge_margin.json", "w"), indent=1)

    print(f"\n{'weights':18s} {'n active':>10s} " + " ".join(f"{'p'+str(p):>9s}" for p in QS) + f" {'>1e-3':>8s} {'>1e-2':>8s}")
    for n, r in rows.items():
        print(f"{n:18s} {r['n_active']:10d} " + " ".join(f"{r['p'+str(p)]:9.5f}" for p in QS)
              + f" {r['frac_active_above_1e-3']:8.3f} {r['frac_active_above_1e-2']:8.3f}")
    print(f"\nwrote {OUT/'part1_hinge_margin.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
