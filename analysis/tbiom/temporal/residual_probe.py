#!/usr/bin/env python3
"""Temporal residual branch on FROZEN temporal features (stage 1 of the temporal residual experiment).

    OMP_NUM_THREADS=8 python analysis/tbiom/temporal/residual_probe.py

For each frozen family (VideoMAE, V-JEPA 2, TimeSformer, X-CLIP; ordered and shuffled frames) and the plain-CLIP spatial
control, train the temporal residual head (residual_head.py: 64-d projection, beta-VAE, orthogonal residual, temporal +
temporal-residual evidential heads, DS fusion, DISCERN's losses) on FF++ TRAIN features only. Budget fixed before launch:
AdamW lr 1e-3 wd 1e-2, batch 128, class-balanced sampling, 40 epochs, seed 0; epoch chosen by the v3.1 development score
of the FUSED opinion (as DISCERN checkpoints are). Opinions of the selected epoch are written in the branch format
(logs/tbiom/temporal/residual/<family>_<kind>/opinions_{temporal,residual,fused}/) for development, processed development,
test and processed test sets, so panel_probe.py fuses them with the existing views on the audit's grouped folds.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
torch.set_num_threads(8)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import panel_probe as P  # noqa: E402
import exp1_fusion_audit as A  # noqa: E402
from residual_head import TemporalResidualHead, to_pu  # noqa: E402

OUT = Path("logs/tbiom/temporal/residual"); EPOCHS, SEED = 40, 0
TESTS = A.TEST
PROC = {"jpeg40": "jpeg40", "rs06": "resize06"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--operator", default="bvae", choices=["bvae", "bvae_real", "mrvae", "order"])
    op = ap.parse_args().operator
    devtab = A.dev_tables({n: P.S23 / f"warmgate_e19_{n}.csv" for n in P.DEV_NAMES}, [1.0])
    for fam in ["videomae", "vjepa2", "timesformer", "xclip", "spatial"]:
        for kind, col in (("ordered", 1), ("shuffled", 2)):
            if fam == "spatial" and (kind == "shuffled" or op == "order"):
                continue
            if op == "order" and kind == "shuffled":
                continue
            od = OUT / (f"{fam}_{kind}" + ("" if op == "bvae" else f"_{op}"))
            if (od / "DONE").is_file():
                continue
            tr = P.load(fam, "FFpp_train"); dv = [P.load(fam, n) for n in P.DEV_NAMES]
            ddf = pd.concat([x[0] for x in dv], ignore_index=True); o = ddf.set_index("v").index.get_indexer(devtab.v)
            assert (o >= 0).all()
            Xtr, ytr, Xd = torch.as_tensor(tr[col]), torch.as_tensor(tr[0].y.to_numpy()).long(), torch.as_tensor(np.concatenate([x[col] for x in dv])[o])
            torch.manual_seed(SEED)
            Str = torch.as_tensor(tr[2]); Sd = torch.as_tensor(np.concatenate([x[2] for x in dv])[o])
            m = TemporalResidualHead(Xtr.shape[1], op); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
            w = torch.where(ytr == 1, 1.0 / (ytr == 1).sum(), 1.0 / (ytr == 0).sum())
            best, hist = (-1.0, None, None), []
            for ep in range(EPOCHS):
                m.train()
                idx = torch.multinomial(w, len(ytr), replacement=True, generator=torch.Generator().manual_seed(SEED + ep))
                for b in range(0, len(idx), 128):
                    j = idx[b:b + 128]; loss = m.loss(m(Xtr[j], Str[j]), ytr[j], ep, EPOCHS)
                    opt.zero_grad(); loss.backward(); opt.step()
                m.eval()
                with torch.no_grad():
                    out = m(Xd, Sd)
                scores = {k: A.v31(to_pu(out[k])[0], A.cells(devtab))[0] for k in ("fused", "t", "r")}
                hist.append({"epoch": ep, **scores})
                if scores["fused"] > best[0] + 1e-6:
                    best = (scores["fused"], ep, {k: v.clone() for k, v in m.state_dict().items()})
            m.load_state_dict(best[2]); m.eval()
            print(f"{fam}:{kind} epoch {best[1]} dev v3.1 fused {best[0]:.4f} | temporal {hist[best[1]]['t']:.4f} residual {hist[best[1]]['r']:.4f}", flush=True)

            def dump(df, X, name, XS=None):
                with torch.no_grad():
                    out = m(torch.as_tensor(X), torch.as_tensor(XS) if XS is not None else None)
                for k, sub in (("t", "opinions_temporal"), ("r", "opinions_residual"), ("fused", "opinions_fused")):
                    (od / sub).mkdir(parents=True, exist_ok=True)
                    p, u = to_pu(out[k]); pd.DataFrame({"v": df.v, "y": df.y, "p": p, "u": u}).to_csv(od / sub / f"{name}.csv", index=False)
            for n in P.DEV_NAMES:
                x = P.load(fam, n); dump(x[0], x[col], n, x[2])
                for c, spec in PROC.items():
                    x = P.load(fam, n, spec)
                    if x is not None:
                        dump(x[0], x[col], f"{n}_{c}", x[2])
            for t in TESTS:
                x = P.load(fam, t); dump(x[0], x[col], t, x[2])
            for t in ("CDFv2", "DFDCP"):
                for c, spec in PROC.items():
                    x = P.load(fam, t, spec); dump(x[0], x[col], f"{t}_{c}", x[2])
            rec = {"family": fam, "kind": kind, "operator": op, "best_epoch": best[1], "best_dev_v31": best[0], "history": hist,
                   "budget": {"epochs": EPOCHS, "lr": 1e-3, "batch": 128, "seed": SEED}}
            for sub in ("opinions_temporal", "opinions_residual", "opinions_fused"):
                sc = {"opinions_temporal": "t", "opinions_residual": "r", "opinions_fused": "fused"}[sub]
                json.dump({**rec, "best_dev_v31": hist[best[1]][sc], "view": sub}, open(od / f"train_record_{sub}.json", "w"), indent=1)
            (od / "DONE").touch()


if __name__ == "__main__":
    main()
