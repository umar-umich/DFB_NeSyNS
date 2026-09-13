#!/usr/bin/env python3
"""Stage 2/3 — choose each arm's checkpoint on VALmix macro AUROC, with the FF++ guardrail logged.

    python analysis/tbiom/stage23_select.py --arm stage3

WHY NOT val AUROC. Every candidate epoch of every arm sits above 0.995 on FF++ val, so the
ranking among them is noise -- that is exactly how P0-DS came to be selected at epoch 1 on a
saturated number that said nothing about real-side behaviour. Step 2 replaced that with VALmix
macro AUROC (averaged over its three domains so no domain dominates) plus a real-side reading.

FIREWALL. Selection sees FF++ val and VALmix only. No CDFv2/CDFv3/DFD/DFDC/DFDCP/DFEval24
export is opened here; choosing an epoch on those would spend the very sets the Stage 6
comparison needs.

The FF++ val AUROC is printed as a GUARDRAIL, not a criterion: an arm whose in-domain number has
fallen away has broken rather than generalised, and that should be visible at selection time.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from health import auroc, dashboard, eer  # noqa: E402
from video_id import video_id  # noqa: E402

OUT = Path("logs/tbiom/stage23")
DOMAINS = ("CDFv2val", "DFDCPval", "DFEval24val")


def vid(path: Path, col: str) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    d = pd.read_csv(path)
    if col not in d:
        return None
    g = (pd.DataFrame({"v": video_id(d["key"]), "p": d[col], "y": d["label"],
                       "raw": d["key"].astype(str)})
         .groupby("v", as_index=False).agg(p=("p", "mean"), y=("y", "max"), raw=("raw", "first")))
    dom = pd.Series("other", index=g.index)
    for name in DOMAINS:
        dom = dom.mask(g["raw"].str.contains(name, regex=False), name)
    g["domain"] = dom
    return g


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", required=True, choices=["stage2", "stage3", "stage5"])
    ap.add_argument("--col", default="p_fused", help="readout to select on")
    args = ap.parse_args()

    tags = sorted({p.name.split("_FFpp_val.csv")[0] for p in OUT.glob(f"{args.arm}_*_FFpp_val.csv")})
    if not tags:
        print(f"no selection exports for {args.arm} — run stage23_export.sh {args.arm} select")
        return 1

    rows = {}
    for tag in tags:
        ff = vid(OUT / f"{tag}_FFpp_val.csv", args.col)
        vm = vid(OUT / f"{tag}_VALmix.csv", args.col)
        if ff is None or vm is None:
            print(f"  {tag}: incomplete, skipped")
            continue
        _, tau = eer(ff["y"].to_numpy(), ff["p"].to_numpy())
        per, macro, fprs = {}, [], []
        for dom in DOMAINS:
            s = vm[vm["domain"] == dom]
            if len(s) < 20 or s["y"].nunique() < 2:
                continue
            d = dashboard(s["y"].to_numpy(), s["p"].to_numpy(), tau)
            per[dom] = d
            macro.append(d["auroc"]); fprs.append(d["fpr_real_at_tau"])
        overall = dashboard(vm["y"].to_numpy(), vm["p"].to_numpy(), tau)
        rows[tag] = {
            "tau": tau,
            "ffpp_val_auroc": auroc(ff["y"].to_numpy(), ff["p"].to_numpy()),   # guardrail
            "macro_auroc": float(np.mean(macro)) if macro else float("nan"),
            "macro_fpr_real": float(np.mean(fprs)) if fprs else float("nan"),
            "valmix_auroc": overall["auroc"], "valmix_fpr_real": overall["fpr_real_at_tau"],
            "valmix_eer": overall["eer"], "per_domain": per,
        }

    print(f"\n{'checkpoint':18s} {'tau':>7s} {'FF++val(guard)':>15s} {'macroAUROC':>11s} "
          f"{'macroFPR_real':>14s} {'VALmix EER':>11s}")
    for tag, r in sorted(rows.items()):
        print(f"{tag:18s} {r['tau']:7.4f} {r['ffpp_val_auroc']:15.4f} {r['macro_auroc']:11.4f} "
              f"{r['macro_fpr_real']:14.3f} {r['valmix_eer']:11.4f}")

    if rows:
        best = max(rows.items(), key=lambda kv: kv[1]["macro_auroc"])
        g = best[1]["ffpp_val_auroc"]
        print(f"\nSELECTED: {best[0]}  macro AUROC {best[1]['macro_auroc']:.4f}, "
              f"macro FPR_real {best[1]['macro_fpr_real']:.3f}")
        print(f"FF++ val guardrail: {g:.4f} "
              f"{'— OK (in-domain intact)' if g > 0.95 else '— WARNING: in-domain has fallen away'}")
        (OUT / f"{args.arm}_selection.json").write_text(json.dumps(
            {"selected": best[0], "col": args.col, "rows": rows}, indent=2, default=float))
        print(f"wrote {OUT}/{args.arm}_selection.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
