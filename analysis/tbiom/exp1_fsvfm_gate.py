#!/usr/bin/env python3
"""Experiment 1 — does an FS-VFM view clear the Branch-3 gate?

    python analysis/tbiom/exp1_fsvfm_gate.py --out tbiom/EXP1_FSVFM_GATE.md

Two candidates, per the revised plan:

    (a) frozen FS-VFM + an FF++-fit probe      -- the representation on its own
    (b) the PRESERVATION student (LoRA-adapted FS-VFM, logs/fpad/studentA_preserve_seed42
        epoch 9)                                -- the DEFAULT candidate, because it is the one
                                                   carrying the existing Step-5/7b evidence

Promotion rule, as agreed: near-peer AUROC *and* a positive rescue-minus-harm margin against
CLIP. `error_overlap` is DIAGNOSTIC, not the gate -- a view can duplicate many of CLIP's errors
and still be worth fusing if it rescues more than it harms. FS-VFM is NOT rejected if only the
frozen probe fails while preservation stays near-peer and complementary.

Every number here comes from parquet already on disk. No GPU, no training.

Thresholds. Each arm's tau is frozen at its OWN EER on FF++ val and applied unchanged
everywhere. A shared threshold would compare arms at different operating points -- the error
that inflated an earlier expert table.

Rescue / harm / overlap are defined at VIDEO level against the CLIP anchor:

    rescue        P(expert correct | CLIP wrong)
    harm          P(expert wrong   | CLIP correct)
    error_overlap P(expert wrong   | CLIP wrong)      1.0 = no conditional information
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from health import auroc, dashboard, eer  # noqa: E402
from video_id import video_id  # noqa: E402

XD = Path("logs/tbiom/crossdataset")
SC = Path("logs/tbiom/score")

# dataset dir suffix -> display name. Celeb-DF-v3 has no sweep output; recorded as a gap.
SETS = [("FaceForensics__", "FF++"), ("Celeb_DF_v2", "Celeb-DF-v2"),
        ("DeepFakeDetection", "DFD"), ("DFDC", "DFDC"), ("DFDCP", "DFDCP"),
        ("Deepfake_Eval_2024", "Deepfake-Eval-2024"), ("Celeb_DF_v1", "Celeb-DF-v1"),
        ("UADFV", "UADFV")]

ARMS = {  # name -> (crossdataset prefix, FF++-val dir, probability column)
    "preserve": ("preserve", SC / "fsvfm_preserve_ffppval", "p_direct"),
    "ordinary": ("ordinary", SC / "fsvfm_ordinary_ffppval", "p_direct"),
    "clip":     ("clip",     SC / "clip_ffppval",           "prob_fused"),
}


def load(pattern: Path, col: str) -> pd.DataFrame | None:
    fs = sorted(glob.glob(str(pattern / "*.parquet")))
    if not fs:
        return None
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    if col not in d:
        return None
    d = d[np.isfinite(d[col])]
    out = pd.DataFrame({"v": video_id(d["key"]), "p": d[col], "y": d["label"]})
    return out.groupby("v", as_index=False).agg(p=("p", "mean"), y=("y", "max"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("tbiom/EXP1_FSVFM_GATE.md"))
    args = ap.parse_args()

    # --- tau per arm, frozen on FF++ val -------------------------------------------------------
    taus = {}
    for name, (_pref, ffval, col) in ARMS.items():
        g = load(ffval, col)
        if g is None:
            print(f"  !! {name}: no FF++ val export")
            continue
        _, taus[name] = eer(g["y"].to_numpy(), g["p"].to_numpy())
        print(f"  tau[{name}] = {taus[name]:.4f}  (FF++ val, n={len(g)})")

    rows, comp = {}, {}
    for suffix, nice in SETS:
        per = {}
        for name, (pref, _ff, col) in ARMS.items():
            g = load(XD / f"{pref}_{suffix}", col)
            if g is None or g["y"].nunique() < 2 or name not in taus:
                continue
            per[name] = g
            rows[(name, nice)] = dashboard(g["y"].to_numpy(), g["p"].to_numpy(), taus[name])
        if "clip" not in per:
            continue
        base = per["clip"]
        for name in ("preserve", "ordinary"):
            if name not in per:
                continue
            m = base.merge(per[name], on="v", suffixes=("_c", "_e"))
            if m.empty:
                continue
            c_ok = ((m.p_c >= taus["clip"]).astype(int) == m.y_c)
            e_ok = ((m.p_e >= taus[name]).astype(int) == m.y_e)
            n_cw, n_cr = int((~c_ok).sum()), int(c_ok.sum())
            comp[(name, nice)] = {
                "n": len(m),
                "rescue": float(e_ok[~c_ok].mean()) if n_cw else float("nan"),
                "harm": float((~e_ok[c_ok]).mean()) if n_cr else float("nan"),
                "overlap": float((~e_ok[~c_ok]).mean()) if n_cw else float("nan"),
                "n_clip_wrong": n_cw,
            }
            comp[(name, nice)]["margin"] = comp[(name, nice)]["rescue"] - comp[(name, nice)]["harm"]

    # --- report --------------------------------------------------------------------------------
    L = ["# Experiment 1 — does an FS-VFM view clear the Branch-3 gate?", "",
         "Candidate (b), the **preservation student** (LoRA-adapted FS-VFM, "
         "`logs/fpad/studentA_preserve_seed42` epoch 9), is the default candidate because it "
         "carries the existing Step-5 and Step-7b evidence. `ordinary` is its sibling. `clip` is "
         "the V1 anchor's fused output, scored on the same videos in the same sweep.", "",
         "`tau` frozen per arm at its own EER on **FF++ val**, applied unchanged everywhere. "
         "Every number is read from parquet already on disk — no GPU, no training.", "",
         "| arm | tau |", "|---|---:|"]
    for k, v in taus.items():
        L.append(f"| {k} | {v:.4f} |")

    L += ["", "## Zero-shot video AUROC and real-side FPR", "",
          "| dataset | n | preserve AUROC | clip AUROC | Δ | preserve FPR_real | clip FPR_real |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for _s, nice in SETS:
        p, c = rows.get(("preserve", nice)), rows.get(("clip", nice))
        if not p or not c:
            L.append(f"| {nice} | — | TODO(run) | | | | |")
            continue
        L.append(f"| {nice} | {p['n_videos']} | **{p['auroc']:.4f}** | {c['auroc']:.4f} | "
                 f"{p['auroc']-c['auroc']:+.4f} | {p['fpr_real_at_tau']:.3f} | "
                 f"{c['fpr_real_at_tau']:.3f} |")
    ok = [n for _s, n in SETS if ("preserve", n) in rows and ("clip", n) in rows]
    if ok:
        mp = np.mean([rows[("preserve", n)]["auroc"] for n in ok])
        mc = np.mean([rows[("clip", n)]["auroc"] for n in ok])
        mpf = np.mean([rows[("preserve", n)]["fpr_real_at_tau"] for n in ok])
        mcf = np.mean([rows[("clip", n)]["fpr_real_at_tau"] for n in ok])
        L += ["", f"Mean over {len(ok)} sets: preserve **{mp:.4f}** vs clip **{mc:.4f}** "
                  f"(**{mp-mc:+.4f}**); FPR_real {mpf:.3f} vs {mcf:.3f} (**{mpf-mcf:+.3f}**).", ""]

    L += ["## Complementarity against CLIP — the promotion criterion", "",
          "| dataset | n | CLIP wrong | rescue | harm | **margin** | error_overlap |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for _s, nice in SETS:
        c = comp.get(("preserve", nice))
        if not c:
            continue
        L.append(f"| {nice} | {c['n']} | {c['n_clip_wrong']} | {c['rescue']:.3f} | "
                 f"{c['harm']:.3f} | **{c['margin']:+.3f}** | {c['overlap']:.3f} |")
    margins = [comp[("preserve", n)]["margin"] for _s, n in SETS
               if ("preserve", n) in comp and np.isfinite(comp[("preserve", n)]["margin"])]
    if margins:
        pos = sum(1 for m in margins if m > 0)
        L += ["", f"Mean margin **{np.mean(margins):+.3f}**, positive on **{pos}/{len(margins)}** "
                  f"datasets.", ""]

    L += ["## Sibling arm — ordinary", "",
          "| dataset | ordinary AUROC | margin vs CLIP |", "|---|---:|---:|"]
    for _s, nice in SETS:
        o, c = rows.get(("ordinary", nice)), comp.get(("ordinary", nice))
        if not o:
            continue
        L.append(f"| {nice} | {o['auroc']:.4f} | "
                 + (f"{c['margin']:+.3f} |" if c else "— |"))

    L += ["", "## Verdict", ""]
    if ok and margins:
        near_peer = (mp - mc) > -0.06
        complementary = np.mean(margins) > 0
        L.append(f"- Near-peer AUROC: mean gap **{mp-mc:+.4f}** — "
                 f"{'**yes**' if near_peer else '**no**'}")
        L.append(f"- Positive rescue-harm margin: mean **{np.mean(margins):+.3f}**, positive on "
                 f"{pos}/{len(margins)} — {'**yes**' if complementary else '**no**'}")
        L.append("")
        if near_peer and complementary:
            L.append("**PROMOTE.** The preservation expert clears both gate conditions. It is the "
                     "version to wire as Branch 3 in Experiment 2, under fused-only supervision, "
                     "one variable against the Stage-5 baseline.")
        elif complementary:
            L.append("**MARGINAL.** Complementary but not near-peer. Wiring it risks the vacuity "
                     "attractor, which eliminates views materially weaker than the incumbent "
                     "pair. Report and decide rather than proceeding automatically.")
        else:
            L.append("**DO NOT PROMOTE on these numbers.** Record which condition failed; the "
                     "frozen-probe arm is still outstanding and may differ.")
    L += ["", "## Gaps", "",
          "- **Celeb-DF-v3 is absent** from this sweep, so the seven-set suite is covered by six "
          "of its members plus Celeb-DF-v1 and UADFV as extras. CDFv3 would need a scoring run.",
          "- Candidate (a), the frozen FS-VFM probe, needs an OOD feature-extraction pass and is "
          "reported separately.",
          "- `clip` here is the **V1 anchor's** fused output, not DISCERN-Ext's P0-DS. It is the "
          "correct comparator for these parquet files because both were produced in the same "
          "sweep on the same videos, but it is not the Stage-5 framework."]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L) + "\n")
    args.out.with_suffix(".json").write_text(json.dumps(
        {"taus": taus, "rows": {"|".join(k): v for k, v in rows.items()},
         "comp": {"|".join(k): v for k, v in comp.items()}}, indent=2, default=float))
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
