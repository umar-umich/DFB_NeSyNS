#!/usr/bin/env python3
"""DiCoME Table 1's second half, measured for our framework: DF40 per-generator cross-manipulation.

    <discern_ext python> upgrade/branchfusion/df40_crossmanip.py
    <discern_ext python> upgrade/branchfusion/df40_crossmanip.py --gens all --out DF40_CROSSMANIP_ALL

DiCoME reports eight face-swap generators -- UniFace, BleFace, MobSwap, e4s, FaceDan, FSGAN,
InSwap, SimSwap -- as ONE number each. DF40 ships every one of them twice, once built from
Celeb-DF sources and once from FaceForensics++ sources, and the paper does not say which it
used. Both are therefore reported, side by side, with the pooled column; the published row is
printed beside them as PUBLISHED, NOT RE-MEASURED, because its video census is unknown to us and
a difference of a few thousandths against a row whose denominator we cannot see is not a result.

What IS a like-for-like comparison is the `DiCoME (our training)` column: our own trained DiCoME
checkpoint, replayed through its own two-view DS fusion on the identical videos, which the
2026-09-22 campaign already computed. It is read from that campaign's JSON rather than
recomputed, so this module cannot score it differently than the report that owns it.

Weights are fitted ONCE on the development pool under F8 and applied unchanged to every
generator -- the locked protocol, and the same call `proposal_tables.py` makes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                               # noqa: E402
import branches as B                                             # noqa: E402
import opinions as O                                             # noqa: E402

C.thread_caps()

#: The eight in DiCoME's published cross-manipulation table, in ITS column order.
DICOME_EIGHT = ["uniface", "blendface", "mobileswap", "e4s", "facedancer", "fsgan",
                "inswap", "simswap"]
PRETTY = {"uniface": "UniFace", "blendface": "BleFace", "mobileswap": "MobSwap", "e4s": "e4s",
          "facedancer": "FaceDan", "fsgan": "FSGAN", "inswap": "InSwap", "simswap": "SimSwap",
          "faceswap": "FaceSwap", "deepfacelab": "DeepFaceLab", "danet": "DaNet",
          "facevid2vid": "FaceVid2Vid", "fomm": "FOMM", "hyperreenact": "HyperReenact",
          "lia": "LIA", "mcnet": "MCNet", "MRAA": "MRAA", "one_shot_free": "One-Shot Free-View",
          "pirender": "PIRender", "sadtalker": "SadTalker", "tpsm": "TPSM"}

#: DiCoME Table 1, cross-manipulation half, transcribed from the paper (page 6), in the column
#: order above. PUBLISHED NUMBERS -- not re-measured here, and on a census we cannot inspect.
PUBLISHED = {
    "F3Net (ECCV'20)":  [0.809, 0.808, 0.867, 0.494, 0.717, 0.845, 0.757, 0.674],
    "SPSL (CVPR'20)":   [0.747, 0.748, 0.885, 0.514, 0.666, 0.812, 0.643, 0.665],
    "SRM (CVPR'21)":    [0.749, 0.704, 0.779, 0.704, 0.659, 0.772, 0.793, 0.694],
    "CORE (CVPR'21)":   [0.871, 0.843, 0.959, 0.679, 0.774, 0.958, 0.855, 0.724],
    "RECCE (CVPR'22)":  [0.898, 0.832, 0.925, 0.683, 0.848, 0.949, 0.848, 0.768],
    "SBI (CVPR'22)":    [0.724, 0.891, 0.952, 0.750, 0.594, 0.803, 0.712, 0.701],
    "UCF (ICCV'23)":    [0.831, 0.827, 0.950, 0.731, 0.862, 0.937, 0.809, 0.647],
    "IID (CVPR'23)":    [0.839, 0.789, 0.888, 0.766, 0.844, 0.927, 0.789, 0.644],
    "LSDA (CVPR'24)":   [0.872, 0.875, 0.930, 0.694, 0.721, 0.939, 0.855, 0.793],
    "ProDet (NIPS'24)": [0.908, 0.929, 0.975, 0.771, 0.747, 0.928, 0.837, 0.844],
    "Effort (ICML'25)": [0.962, 0.873, 0.953, 0.983, 0.926, 0.957, 0.936, 0.926],
    "GenD (WACV'26)":   [0.974, 0.935, 0.976, 0.983, 0.942, 0.974, 0.943, 0.938],
    "DiCoME (published)": [0.982, 0.963, 0.978, 0.991, 0.972, 0.989, 0.970, 0.965],
}

CAMPAIGN = C.ROOT / "tbiom/reports/EXP_DF40_CROSS_MANIPULATION.json"
FINAL = ("A", "V", "O", "S16ctl")
#: BOTH candidate framework sizes, because the choice between them is UMAR's and is to be made
#: from the numbers. The three-branch group is the best of the ten in Table A at that size
#: (macro 0.9476 over twelve sets, against 0.9509 for the four), so this is the comparison that
#: decides whether STT earns its place on generators none of the branches were trained on.
GROUPS = [FINAL, ("A", "V", "O")]


def gens_available(gens, halves=("cdf", "ff")):
    """Generator sets where EVERY branch of the framework resolves, and why one does not."""
    out, missing = [], {}
    for g in gens:
        for h in halves:
            ds = f"DF40_{g}_{h}"
            if not B.sel_csv(ds).exists():
                missing[ds] = "no stage-23 selection export"
                continue
            gone = [b for b in set().union(*GROUPS) if not B.available(b, ds)]
            if gone:
                missing[ds] = "branches " + ",".join(C.show(b) for b in gone)
                continue
            out.append(ds)
    return out, missing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", default="dicome8", help="dicome8 | all | a comma list of methods")
    ap.add_argument("--out", default="DF40_CROSSMANIP")
    a = ap.parse_args()

    meta = json.loads((C.ROOT / "logs/tbiom/df40/meta.json").read_text())
    every = sorted({v["method"] for v in meta.values()})
    gens = (DICOME_EIGHT if a.gens == "dicome8" else
            every if a.gens == "all" else [x.strip() for x in a.gens.split(",") if x.strip()])

    sets, missing = gens_available(gens)
    for ds, why in sorted(missing.items()):
        print(f"  skipped {ds}: {why}", flush=True)
    if not sets:
        print("no generator set has every branch; run lanes/df40_feats.sh first")
        return 1
    need = sorted(set().union(*GROUPS))
    print(f"{len(sets)} generator sets, branches {C.show_subset(need)}", flush=True)

    import ops_pcr6p as P
    dev = B.dev_table(need, require_all=True)
    W = {}
    for g in GROUPS:
        W[g], _ = O.fit_weights(dev, g, "F8", objective="v31")
        print(f"  weights for {C.show_subset(g)}: {[round(x, 2) for x in W[g]]}", flush=True)

    res = {}
    for ds in sets:
        t = B.table([ds], need)
        y = t.y.to_numpy()
        row = {"videos": int(len(t)), "real": int((y == 0).sum()), "fake": int((y == 1).sum())}
        for g in GROUPS:
            row["+".join(g)] = C.auroc(y, P.p_of(P.fuse(O.masses(t, g, list(W[g])), True)))
        row["ours"] = row["+".join(FINAL)]
        for b in need:
            row[f"branch_{b}"] = C.auroc(y, t[f"p_{b}"].to_numpy())
        res[ds] = row
        print(f"  {ds:28s} n={row['videos']:5d}  "
              + "  ".join(f"{C.show_subset(g)} {row['+'.join(g)]:.4f}" for g in GROUPS), flush=True)

    # The like-for-like DiCoME column, from the campaign that owns that computation.
    camp = json.loads(CAMPAIGN.read_text()) if CAMPAIGN.exists() else {}
    for ds, r in res.items():
        if ds in camp:
            r["dicome_our_training"] = camp[ds].get("dicome")
            r["old_three_view"] = camp[ds].get("ours_dsr")

    (C.OUT / f"{a.out}.json").write_text(json.dumps(res, indent=1, default=float))

    def cell(x):
        return f"{x:.4f}" if isinstance(x, float) else "—"

    def mean_of(vals):
        got = [v for v in vals if isinstance(v, float)]
        return float(np.mean(got)) if got else None

    L = [f"# DF40 cross-manipulation — {C.show_subset(FINAL)} fused with PCR6+-R", "",
         "The structure of DiCoME Table 1's second half, measured on our videos.", "",
         "**Both framework sizes are reported.** The four-branch set and the best three-branch set",
         f"(`{C.show_subset(GROUPS[1])}`) each carry their own weights, fitted on development and",
         "applied unchanged here, so the question of whether STT earns its place is answered on",
         "generators no branch was trained on rather than on the twelve test sets alone.", "",
         "**Both source halves are reported.** DF40 ships each generator twice — built from",
         "Celeb-DF sources (`cdf`) and from FaceForensics++ sources (`ff`) — and DiCoME's paper",
         "reports one number per generator without saying which. Picking the more flattering half",
         "would be a choice this table hides, so neither is picked.", "",
         "`DiCoME (our training)` is our own trained DiCoME checkpoint on the IDENTICAL videos and",
         "is the comparison to read. The published rows below are transcribed from the paper on a",
         "census we cannot inspect and are context, not a measured difference.", ""]

    halves = [("cdf", "Celeb-DF-sourced"), ("ff", "FaceForensics++-sourced")]
    for h, title in halves:
        rows = [(g, f"DF40_{g}_{h}") for g in gens if f"DF40_{g}_{h}" in res]
        if not rows:
            continue
        gcols = [C.show_subset(g) for g in GROUPS]
        L += [f"## {title} generators", "",
              "| generator | videos (real / fake) | **" + "** | ".join(gcols)
              + "** | DiCoME (our training) | Δ (four) | " + " | ".join(C.show(b) for b in need)
              + " |",
              "|---|---|" + "---:|" * (len(GROUPS) + 2) + "---:|" * len(need)]
        for g, ds in rows:
            r = res[ds]
            d = r.get("dicome_our_training")
            delta = f"{r['ours'] - d:+.4f}" if isinstance(d, float) else "—"
            L.append(f"| {PRETTY.get(g, g)} | {r['real']} / {r['fake']} | "
                     + " | ".join(f"**{r['+'.join(q)]:.4f}**" for q in GROUPS)
                     + f" | {cell(d)} | {delta} | "
                     + " | ".join(cell(r[f'branch_{b}']) for b in need) + " |")
        m_dic = mean_of([res[ds].get("dicome_our_training") for _g, ds in rows])
        m_ours = mean_of([res[ds]["ours"] for _g, ds in rows])
        delta = f"{m_ours - m_dic:+.4f}" if isinstance(m_dic, float) else "—"
        L.append(f"| **mean over {len(rows)}** | | "
                 + " | ".join("**" + cell(mean_of([res[ds]["+".join(q)] for _g, ds in rows]))
                              + "**" for q in GROUPS)
                 + f" | {cell(m_dic)} | {delta} | "
                 + " | ".join(cell(mean_of([res[ds][f'branch_{b}'] for _g, ds in rows]))
                              for b in need) + " |")
        L.append("")

    # The published table, with our pooled number beside it in the same column order.
    eight = [g for g in DICOME_EIGHT if any(f"DF40_{g}_{h}" in res for h in ("cdf", "ff"))]
    if eight:
        L += ["## Against the published cross-manipulation table", "",
              "Column order is DiCoME's. Our three rows are the two halves and their pool; every",
              "other row is transcribed from the paper and was not re-measured here.", "",
              "| method | " + " | ".join(PRETTY[g] for g in eight) + " | Avg. |",
              "|---|" + "---:|" * (len(eight) + 1)]
        for q in GROUPS:
            key = "+".join(q)
            for lbl, half in ((f"**{C.show_subset(q)}** (cdf)", "cdf"),
                              (f"**{C.show_subset(q)}** (ff)", "ff")):
                vals = [res.get(f"DF40_{g}_{half}", {}).get(key) for g in eight]
                L.append(f"| {lbl} | " + " | ".join(cell(v) for v in vals)
                         + f" | {cell(mean_of(vals))} |")
            pooled = []
            for g in eight:
                v = [res[f"DF40_{g}_{h}"][key] for h in ("cdf", "ff") if f"DF40_{g}_{h}" in res]
                pooled.append(float(np.mean(v)) if v else None)
            L.append(f"| **{C.show_subset(q)}** (both halves) | "
                     + " | ".join(cell(v) for v in pooled) + f" | {cell(mean_of(pooled))} |")
        for lbl, vals in PUBLISHED.items():
            v = [vals[DICOME_EIGHT.index(g)] for g in eight]
            L.append(f"| {lbl} *(published)* | " + " | ".join(f"{x:.3f}" for x in v)
                     + f" | {np.mean(v):.3f} |")
        L.append("")

    (C.REPORT / f"{a.out}.md").write_text("\n".join(L) + "\n")
    print(f"-> {C.REPORT / f'{a.out}.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
