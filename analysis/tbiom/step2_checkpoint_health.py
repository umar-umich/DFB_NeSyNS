#!/usr/bin/env python3
"""Step 2 — re-select the P0-DS checkpoint on the health dashboard, not on val AUROC.

    python analysis/tbiom/step2_checkpoint_health.py --out tbiom/STEP2_CHECKPOINT_HEALTH.md

P0-DS was selected at epoch 1 on the highest `val_auroc_video` (0.9960). That number is
saturated — every candidate epoch is above 0.995, so the ranking among them is noise — and it
says nothing about real-side behaviour. Step 1 then measured that checkpoint calling 31.7% of OOD
reals fake, worse than our own weaker CLIP port at 17.9%. This re-selects among every P0-DS
checkpoint on disk using the metrics that would have caught it.

SELECTION DATA ONLY, and this is the firewall rather than a preference. The operating threshold
is frozen on FF++ val; the selection metrics come from VALmix. No final OOD test set appears
here. Choosing an epoch on CDFv2/DFDC/DFD/DFDCP would be selecting on test and would spend the
sets the eventual comparison needs.

The four numbers the brief names, per checkpoint:

    macro video AUROC   averaged over VALmix's three domains, so no domain dominates
    EER                 threshold-independent operating quality
    OOD-real FPR@tau    reals called fake at the frozen point — the health gate that OVERRIDES
                        AUROC, per the brief
    d_RF                probability separation

The released checkpoint is scored alongside as a reference point, NOT as a candidate: it is not
retrainable, which is the property Step 2 exists to preserve.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from health import auroc, dashboard, eer  # noqa: E402
from video_id import video_id  # noqa: E402

STEP2 = Path("logs/tbiom/step2")
CANDIDATES = ["p0ds_e01", "p0ds_e02", "p0ds_e05", "p0ds_last"]
REFERENCE = ["released"]
LABEL = {"p0ds_e01": "P0-DS epoch 1 (the original pick)", "p0ds_e02": "P0-DS epoch 2",
         "p0ds_e05": "P0-DS epoch 5", "p0ds_last": "P0-DS last epoch",
         "released": "DiCoME released (reference, not retrainable)"}

# VALmix's three source domains, recovered from the split-file directory in each key.
DOMAINS = {"CDFv2val": "CDFv2val", "DFDCPval": "DFDCPval", "DFEval24val": "DFEval24val"}


def load(arm: str, ds: str, col: str) -> pd.DataFrame | None:
    p = STEP2 / f"{arm}_{ds}.csv"
    if not p.is_file():
        return None
    df = pd.read_csv(p)
    if col not in df:
        return None
    out = pd.DataFrame({"v": video_id(df["key"]), "p": df[col], "y": df["label"],
                        "raw": df["key"].astype(str)})
    g = out.groupby("v", as_index=False).agg(p=("p", "mean"), y=("y", "max"),
                                             raw=("raw", "first"))
    dom = pd.Series("other", index=g.index)
    for needle, name in DOMAINS.items():
        dom = dom.mask(g["raw"].str.contains(needle, regex=False), name)
    g["domain"] = dom
    return g


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--col", default="p_fused",
                    help="readout to select on; p_fused is DiCoME's deployed head")
    ap.add_argument("--out", type=Path, default=Path("tbiom/STEP2_CHECKPOINT_HEALTH.md"))
    args = ap.parse_args()

    rows = {}
    for arm in CANDIDATES + REFERENCE:
        ff = load(arm, "FFpp_val", args.col)
        vm = load(arm, "VALmix", args.col)
        if ff is None or vm is None:
            continue
        # tau frozen on FF++ val — the permitted development source, per arm because the
        # checkpoints are differently calibrated
        _, tau = eer(ff["y"].to_numpy(), ff["p"].to_numpy())
        overall = dashboard(vm["y"].to_numpy(), vm["p"].to_numpy(), tau)

        per_domain, macro, fprs = {}, [], []
        for dom in DOMAINS:
            sub = vm[vm["domain"] == dom]
            if len(sub) < 20 or sub["y"].nunique() < 2:
                continue
            d = dashboard(sub["y"].to_numpy(), sub["p"].to_numpy(), tau)
            per_domain[dom] = d
            macro.append(d["auroc"])
            fprs.append(d["fpr_real_at_tau"])
        rows[arm] = {
            "tau": tau,
            "ffpp_val_auroc": auroc(ff["y"].to_numpy(), ff["p"].to_numpy()),
            "macro_auroc": float(np.mean(macro)) if macro else float("nan"),
            "macro_fpr_real": float(np.mean(fprs)) if fprs else float("nan"),
            **{k: overall[k] for k in ("auroc", "eer", "fpr_real_at_tau", "delta_rf",
                                       "mean_p_on_real", "n_videos")},
            "per_domain": per_domain,
        }
        r = rows[arm]
        print(f"  {arm:12s} tau {tau:.4f}  macroAUROC {r['macro_auroc']:.4f}  "
              f"EER {r['eer']:.4f}  macroFPR_real {r['macro_fpr_real']:.3f}  "
              f"d_RF {r['delta_rf']:+.3f}")

    lines = ["# Step 2 — P0-DS checkpoint re-selection on the health dashboard", "",
             f"Readout: `{args.col}`. Threshold frozen per checkpoint on **FF++ val**; selection "
             f"metrics from **VALmix** only. No final OOD test set is touched — choosing an epoch "
             f"on CDFv2/DFDC/DFD/DFDCP would be selecting on test.", "",
             "`macro` averages VALmix's three domains (CDFv2val, DFDCPval, DFEval24val) so no "
             "domain dominates. **OOD-real FPR overrides AUROC as the health gate**, per the "
             "brief.", "",
             "| checkpoint | tau | FF++ val AUROC | macro AUROC | EER | **macro FPR_real** | d_RF | mean p on real |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in CANDIDATES + REFERENCE:
        r = rows.get(arm)
        if not r:
            lines.append(f"| {LABEL[arm]} | TODO(run) | | | | | | |")
            continue
        lines.append(
            f"| {LABEL[arm]} | {r['tau']:.4f} | {r['ffpp_val_auroc']:.4f} | "
            f"{r['macro_auroc']:.4f} | {r['eer']:.4f} | **{r['macro_fpr_real']:.3f}** | "
            f"{r['delta_rf']:+.3f} | {r['mean_p_on_real']:.3f} |")

    lines += ["", "## Per VALmix domain", "",
              "| checkpoint | " + " | ".join(f"{d} AUROC / FPR_real" for d in DOMAINS) + " |",
              "|---" * (len(DOMAINS) + 1) + "|"]
    for arm in CANDIDATES + REFERENCE:
        r = rows.get(arm)
        if not r:
            continue
        cells = []
        for dom in DOMAINS:
            d = r["per_domain"].get(dom)
            cells.append("—" if not d else f"{d['auroc']:.4f} / {d['fpr_real_at_tau']:.3f}")
        lines.append(f"| {LABEL[arm]} | " + " | ".join(cells) + " |")

    # --- the decision --------------------------------------------------------------------------
    cand = {a: r for a, r in rows.items() if a in CANDIDATES}
    lines += ["", "## Decision", ""]
    if cand:
        orig = cand.get("p0ds_e01")
        best_health = min(cand.items(), key=lambda kv: kv[1]["macro_fpr_real"])
        best_auroc = max(cand.items(), key=lambda kv: kv[1]["macro_auroc"])
        lines += [
            f"Best real-side health: **{LABEL[best_health[0]]}** at macro FPR_real "
            f"{best_health[1]['macro_fpr_real']:.3f} (macro AUROC "
            f"{best_health[1]['macro_auroc']:.4f}).",
            f"Best macro AUROC: **{LABEL[best_auroc[0]]}** at {best_auroc[1]['macro_auroc']:.4f} "
            f"(macro FPR_real {best_auroc[1]['macro_fpr_real']:.3f}).", ""]
        if orig:
            d_fpr = best_health[1]["macro_fpr_real"] - orig["macro_fpr_real"]
            d_auc = best_health[1]["macro_auroc"] - orig["macro_auroc"]
            lines.append(
                f"Against the original epoch-1 pick, the healthiest checkpoint moves FPR_real by "
                f"**{d_fpr:+.3f}** and macro AUROC by **{d_auc:+.4f}**.")
            lines.append("")
            if d_fpr < -0.05 and d_auc > -0.01:
                lines.append("**A different checkpoint substantially fixes real-side health "
                             "without giving up AUROC. The retrainable DiCoME chassis stays "
                             "viable** — carry the selected checkpoint into the recipe question "
                             "of why its semantic branch is stronger.")
            elif min(v["macro_fpr_real"] for v in cand.values()) > 0.25:
                lines.append("**Every P0-DS checkpoint has poor real-side health.** The problem "
                             "is not checkpoint timing, it is the retraining recipe itself, so "
                             "the VAE/alignment ablation becomes the next experiment rather than "
                             "a follow-up.")
            else:
                lines.append("**Mixed.** Real-side health improves but not decisively, or it "
                             "improves at a real AUROC cost. Record both and treat the chassis "
                             "choice as still open pending the recipe ablation.")
    else:
        lines.append("TODO(run) — exports incomplete.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    args.out.with_suffix(".json").write_text(json.dumps(rows, indent=2, default=float))
    print("\n".join(lines[-8:]))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
