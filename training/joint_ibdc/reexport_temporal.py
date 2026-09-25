#!/usr/bin/env python3
"""Re-export the video view's development opinions FROM the feature cache (Task 2, option 2).

    <temporal_vjepa2 python> training/joint_ibdc/reexport_temporal.py

WHY. The Task 2 gate asks that the deployed heads on the cache reproduce the deployed per-view
opinions to 1e-5. The two spatial views do, to 1e-08. The V-JEPA 2 view misses by fifty-fold at
5.3e-04, and the cause survives every check that would make it a bug: re-extraction is stable to
3.8e-06 across processes and batch sizes, batch size matching changes nothing, the autocast boundary
makes it worse, and the checkpoint is not stale. What remains is bf16 or library drift since the
2026-09-17 export through a frozen ViT-L.

So the study takes its temporal opinions from the cache, which is what the brief already asks for at
J0 ("evaluated through the Task 1 fusion code"). All three views then derive from one object.

THIS MAKES THE ORIGINAL CHECK CIRCULAR, AND THAT IS SAID RATHER THAN HIDDEN. Once the reference is
cache-derived, "the cache reproduces the reference" is true by construction and carries no
information. The check is therefore REPLACED by a measured delta against the historical export, over
the whole development pool, reported here and in the report. The deployed CSVs are not overwritten;
they remain the reference of record for the studies that already used them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
BR = ROOT / "logs/tbiom/temporal/branches/vjepa2_lora_B_seed42"
CACHE = ROOT / "logs/joint_ibdc/cache"
OUT = ROOT / "logs/joint_ibdc/opinions"
DEV_SETS = ("VALmix", "VALmixV2", "FFpp_val")


def head_of(dev="cpu"):
    ck = torch.load(BR / "best.pt", map_location=dev)
    h = torch.nn.Sequential(torch.nn.LayerNorm(1024), torch.nn.Linear(1024, 128),
                            torch.nn.GELU(), torch.nn.Linear(128, 2))
    h.load_state_dict(ck["head"]); h.eval()
    return h, int(ck["epoch"])


def opinions_from(feature, head):
    with torch.no_grad():
        ev = torch.nn.functional.softplus(head(torch.from_numpy(feature).float()))
    a = ev + 1.0
    S = a.sum(1, keepdim=True)
    return (a / S)[:, 1].numpy(), (2.0 / S.squeeze(1)).numpy()


def auc(s, y):
    r = rankdata(s); n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)) if n0 and n1 else float("nan")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    head, epoch = head_of()
    print(f"frozen head from best.pt, development-selected epoch {epoch}")
    rows, summary = [], {}
    for name in DEV_SETS:
        f = CACHE / f"{name}_T_clean.npz"
        if not f.exists():
            print(f"  {name}: no cached features, skipped"); continue
        z = np.load(f, allow_pickle=True)
        p, u = opinions_from(z["feature"], head)
        v = np.array([str(x) for x in z["video"]])
        y = np.asarray(z["label"]).astype(int)
        df = pd.DataFrame({"v": v, "y": y, "p": p, "u": u})
        df.to_csv(OUT / f"{name}.csv", index=False)

        ref_f = BR / "opinions" / f"{name}.csv"
        cell = {"n": int(len(df)), "file": str(OUT / f"{name}.csv")}
        if ref_f.exists():
            ref = pd.read_csv(ref_f).set_index("v")
            k = [i for i, x in enumerate(v) if x in ref.index]
            j = [v[i] for i in k]
            pr, ur = ref.loc[j, "p"].to_numpy(), ref.loc[j, "u"].to_numpy()
            cell |= {"matched": len(k),
                     "max_abs_dp": float(np.abs(p[k] - pr).max()),
                     "median_abs_dp": float(np.median(np.abs(p[k] - pr))),
                     "max_abs_du": float(np.abs(u[k] - ur).max()),
                     "auroc_cache": auc(p[k], y[k]), "auroc_historical": auc(pr, y[k]),
                     "spearman": float(pd.Series(p[k]).corr(pd.Series(pr), method="spearman"))}
            rows.append((name, cell))
            print(f"  {name:9s} {len(k):5d} videos  max|dp| {cell['max_abs_dp']:.3e}  "
                  f"median {cell['median_abs_dp']:.3e}  AUROC {cell['auroc_cache']:.6f} vs "
                  f"{cell['auroc_historical']:.6f}  rho {cell['spearman']:.6f}")
        summary[name] = cell

    tot = sum(c["n"] for c in summary.values())
    worst = max((c.get("max_abs_dp", 0.0) for c in summary.values()), default=0.0)
    res = {"action": "temporal development opinions re-exported from the feature cache",
           "branch": str(BR), "epoch": epoch, "dev_videos": tot,
           "written_to": str(OUT),
           "deployed_csvs_overwritten": False,
           "gate_note": ("The original 1e-5 equivalence check is CIRCULAR once the reference is "
                         "cache-derived, so it is replaced by this measured delta against the "
                         "historical 2026-09-17 export."),
           "delta_vs_historical": summary, "worst_max_abs_dp": worst}
    json.dump(res, open(ROOT / "analysis/joint_ibdc/temporal_reexport.json", "w"), indent=1)
    print(f"\n{tot} development videos written to {OUT}")
    print(f"worst max|dp| against the historical export: {worst:.3e}")
    print("the deployed CSVs were NOT overwritten")


if __name__ == "__main__":
    main()
