#!/usr/bin/env python3
"""PROPOSAL_TABLES_3.md -- the paper tables only. Nothing a reader would not quote.

    <discern_ext python> upgrade/branchfusion/proposal_tables_3.py

UMAR 2026-10-15: the framework is TD+LFS+STT. The ablation is those three branches and every
combination of them -- seven rows, not thirty-one. App and STT-order are gone entirely: they are
not in the framework, so a row about them is a detail for `PROPOSAL_TABLES_2.md`, which keeps
the full grid. FPR is reported at TPR 90% only. DF40's two source halves are pooled, giving one
compact table per manipulation family.

Everything here is recomputed for THIS framework rather than copied from the four-branch tables:
the fusion weights are fitted on the development pool for each subset, and `tau` is this
framework's own EER on FF++-val.
"""
from __future__ import annotations

import json
import os
import sys
from itertools import combinations
from pathlib import Path

# THE PAPER TABLES ARE THE WIDENED CENSUS. Set BEFORE `common` is imported, because that module
# reads BF_SCOPE at import time to pick its output paths. Relying on the caller's environment
# produced a file whose header said 5,418 and 2,239 while its numbers were 5,118 and 1,395 --
# a mislabelled census is worse than a wrong one, because nothing in the file contradicts it.
os.environ.setdefault("BF_SCOPE", "nolipadv")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                               # noqa: E402
import branches as B                                             # noqa: E402
import opinions as O                                             # noqa: E402
import sota_table1 as T1                                         # noqa: E402
import table2_calibration as T2                                  # noqa: E402
import df40_pool as PL                                           # noqa: E402
import df40_baselines as DB                                      # noqa: E402

C.thread_caps()

FRAME = ("V", "O", "S16ctl")                     # VD + LFS + STT, in display order
#: The framework's name in every comparative table. A row that lists its branches invites the
#: reader to compare branch counts instead of methods.
OURS_NAME = "**CROSS-EXAM (ours)**"
#: Full forms, for the ablation's single-branch rows. UMAR 2026-10-15: spell them out once where
#: a reader meets them, and keep the abbreviations in the combination rows.
FULL = {"V": "Video Dynamics (VD)", "O": "Latent Facial Structure (LFS)",
        "S16ctl": "Spatio-Temporal Texture (STT)"}
SHORT = {"V": "VD", "O": "LFS", "S16ctl": "STT"}


def show_subset(g):
    """Full form for a single branch, abbreviations for a combination."""
    return FULL[g[0]] if len(g) == 1 else "+".join(SHORT[b] for b in g)
# THE TOP LEVEL, not `C.REPORT`. This file IS the paper's table set and there is exactly one of
# it; `C.REPORT` moves into `scope_<name>/` for a non-default census, which would have left the
# stale v3.1 copy sitting at the path everyone reads while the corrected one hid in a
# subdirectory. The census it used is stated in its own header, counted from the data.
OUT = C.ROOT / "tbiom/reports/branch_fusion/PROPOSAL_TABLES_3.md"
#: `--all-metrics` renders every operator metric instead of the reported three.
ALL_METRICS = "--all-metrics" in sys.argv
#: UMAR's term for DF40's reenactment family.
FAMILY_NAME = {"face swap": "FaceSwap", "reenactment": "ExpressionSwap"}
#: The decision rules the framework could be fused with, on the SAME three branches. Restored
#: 2026-10-15: dropping it left the paper asserting ES-PCR6+ without the comparison that earns it.
#: Probability means, which produce no opinion -- their only uncertainty is the decision margin.
PROB_OPS = [("F1", "Mean (unweighted)"), ("F2", "Mean (fitted weights)")]
#: Belief-combination rules, all from ONE implementation (`src.fusion.rules`) reading the SAME
#: discounted masses, so the comparison isolates where conflicting mass goes and nothing else.
BELIEF_OPS = ["Dempster", "Yager", "WBF (Josang)", "Cautious (Denoeux)", "ES-PCR6+ (ours)"]
#: ES-PCR6+ = Evidence-Scaled PCR6+. Each source's evidence is scaled by a fixed reliability
#: coefficient BEFORE proportional conflict redistribution: e'_v = r_v e_v. Named for the
#: operation it performs, because it is NOT Shafer discounting -- that acts on belief masses,
#: this acts in the evidence domain, and the two are different functions. The coefficients are
#: fitted once on the development pool and held fixed at inference, so they are a property of
#: the representation rather than of the video. The coefficients are `w_full` below -- the same
#: vector every other table in this file fuses with, so there is one source of truth for them
#: and the operator comparison cannot silently use a different weighting than the headline rows.


def md(lines, header, rows, cols, fmt="{:.4f}"):
    lines += ["| " + header + " | " + " | ".join(cols) + " | mean |",
              "|---|" + "---:|" * (len(cols) + 1)]
    for name, cells in rows:
        got = [v for v in (cells.get(c) for c in cols) if isinstance(v, float)]
        m = fmt.format(np.mean(got)) if got else "—"
        lines.append(f"| {name} | " + " | ".join(
            fmt.format(cells[c]) if isinstance(cells.get(c), float) else "—" for c in cols)
            + f" | **{m}** |")
    lines.append("")


def main() -> int:
    sys.path[:0] = [str(C.ROOT / "analysis/tbiom/pcr6p")]
    import ops_pcr6p as P
    from health import eer

    dev = B.dev_table(list(FRAME), require_all=True)
    tabs = B.test_tables(list(FRAME))
    sets = [d for d in C.TEST_SETS if d in tabs]
    print(f"framework {C.show_subset(FRAME)} on {len(sets)} test sets", flush=True)

    # ------------------------------------------------------------------ the framework itself
    w_full, _ = O.fit_weights(dev, FRAME, "F8", objective="v31")
    fd = P.fuse(O.masses(dev, FRAME, list(w_full)), True)
    ffv = np.where(dev.source.to_numpy() == "FFpp_val")[0]
    tau = float(eer(dev.y.to_numpy()[ffv], P.p_of(fd)[ffv])[1])
    OURS = OURS_NAME

    def fused(t, g, w):
        fo = P.fuse(O.masses(t, g, list(w)), True)
        return P.p_of(fo), fo[:, 2], t.y.to_numpy()

    # ------------------------------------------------------------------ Table 1: ablation
    subsets = [s for n in (1, 2, 3) for s in combinations(FRAME, n)]
    ab_a, ab_f = [], []
    for g in subsets:
        w, _ = O.fit_weights(dev, g, "F8", objective="v31")
        a, f = {}, {}
        for ds in sets:
            p, _u, y = fused(tabs[ds], g, w)
            a[ds] = C.auroc(y, p)
            f[ds] = T1.fpr_at_tpr(y, p, 0.90)
        nm = show_subset(g) + (" **(CROSS-EXAM)**" if len(g) == len(FRAME) else "")
        ab_a.append((nm, a)); ab_f.append((nm, f))
        print(f"  ablation {'+'.join(SHORT[b] for b in g):14s} "
              f"{np.mean(list(a.values())):.4f}", flush=True)

    # ------------------------------------------------------------------ Tables 2 and 3
    det_a, det_f, cal = {}, {}, {}
    for ds in sets:
        p, u, y = fused(tabs[ds], FRAME, w_full)
        det_a[ds] = C.auroc(y, p)
        det_f[ds] = T1.fpr_at_tpr(y, p, 0.90)
        cal[ds] = {"ece": T2.ece(y, p), "brier": T2.brier(y, p),
                   "eaurc": T2.e_aurc(y, p, u, tau)}
    rows_a = [(OURS, det_a)]
    rows_f = [(OURS, det_f)]
    rows_c = {m: [(OURS, {d: cal[d][m] for d in sets})] for m in ("ece", "brier", "eaurc")}

    for label, get in (("DiCoME", T1.dicome_videos),) + tuple(
            (T1.PRETTY.get(m, m), (lambda mm: lambda ds: T1.baseline_videos(mm, ds))(m))
            for m in T1.MODELS):
        a, f = {}, {}
        cc = {m: {} for m in ("ece", "brier", "eaurc")}
        for ds in sets:
            r = get(ds)
            if r is None:
                continue
            y, s_ = r
            a[ds] = C.auroc(y, s_)
            f[ds] = T1.fpr_at_tpr(y, s_, 0.90)
            uu = 1.0 - np.maximum(s_, 1.0 - s_)
            cc["ece"][ds] = T2.ece(y, s_)
            cc["brier"][ds] = T2.brier(y, s_)
            cc["eaurc"][ds] = T2.e_aurc(y, s_, uu, 0.5)
        rows_a.append((label, a)); rows_f.append((label, f))
        for m in cc:
            rows_c[m].append((label, cc[m]))
        print(f"  baseline {label}", flush=True)

    # ------------------------------------------------------------------ DF40, halves pooled
    meta = json.loads((C.ROOT / "logs/tbiom/df40/meta.json").read_text())
    fam_raw = {k: v["family"] for k, v in meta.items()}
    import df40_crossmanip as D
    gens = sorted({v["method"] for v in meta.values()})
    d_sets, _ = D.gens_available(gens, )
    d_tabs = {ds: B.table([ds], list(FRAME)) for ds in d_sets}
    pooled = PL.pool_tables(d_tabs, d_sets)
    fam = {g: fam_raw[ms[0]] for g, ms in PL.group(d_sets).items()}
    jobs = PL.jobs_for(meta)

    df_rows = {OURS: {}}
    for g, t in pooled.items():
        p, _u, y = fused(t, FRAME, w_full)
        df_rows[OURS][g] = C.auroc(y, p)
    dic = PL.dicome_ours_pooled(jobs)
    if dic:
        df_rows["DiCoME (our training)"] = dic
    repo = C.ROOT / "logs/tbiom/df40/dicome_repo_df40_combined.json"
    if repo.exists():
        r = json.loads(repo.read_text())
        got = {g: r[g[len("DF40_"):]]["auc_video"] for g in pooled if g[len("DF40_"):] in r}
        if len(got) == len(pooled):
            df_rows["DiCoME (their repo)"] = got
        elif got:
            print(f"  DiCoME (their repo): {len(got)}/{len(pooled)} -- omitted, still running",
                  flush=True)
    for m in DB.MODELS:
        rr = PL.baseline_pooled(m, jobs)
        if rr:
            df_rows[DB.PRETTY[m]] = rr

    # ------------------------------------------------------------------ render
    L = ["# Proposal results — paper tables", "",
         f"Framework: **CROSS-EXAM** = {'+'.join(SHORT[b] for b in FRAME)}, fused with",
         "ES-PCR6+. Weights fitted once on the",
         "development pool and applied unchanged to every test set.", "",
         "Paper tables only. The full grids — all five branches, all 31 combinations, both KoDF",
         "censuses, FPR at 95%, the per-source DF40 halves — are in `PROPOSAL_TABLES_2.md`.", "",
         f"Twelve test sets, video level. Census `{C.SCOPE}` — {C.scope_note()}.",
         f"CDFv3 {len(tabs['CDFv3']):,} videos, KoDF {len(tabs['KoDF']):,}.", "",
         "---", "", "## Table 1 — Ablation", "",
         "Each branch alone and every combination of the three. Weights are refitted for each",
         "row, so a row is the best that subset can do rather than the full model with parts",
         "switched off.", "", "### AUROC (higher is better)", ""]
    md(L, "branches", ab_a, sets)
    L += ["### FPR at TPR = 90% (lower is better)", ""]
    md(L, "branches", ab_f, sets)

    # ------------------------------------------------------------------ Table 2: operators
    import torch
    sys.path.insert(0, "/data/umar/Repos/DISCERN_Ext")
    from src.fusion import rules as R
    from src.fusion import pcr6p as PP
    FNS = {"Dempster": R.dempster, "Yager": R.yager, "WBF (Josang)": R.wbf,
           "Cautious (Denoeux)": R.cautious, "ES-PCR6+ (ours)": R.pcr6_plus}

    METRICS = [("auroc", "AUROC", "higher"), ("fpr90", "FPR at TPR = 90%", "lower"),
               ("ece", "ECE-15", "lower"), ("brier", "Brier", "lower"),
               ("nll", "NLL", "lower"), ("eaurc", "E-AURC", "lower"),
               ("vac", "mean vacuity", "—")]
    #: UMAR 2026-10-15, reaffirmed: the operator comparison reports these three everywhere, in
    #: the paper file AND in the full-detail one. All seven are still COMPUTED -- regenerating
    #: them is `--all-metrics`, and they remain in git history -- so the restriction is a
    #: reporting decision that can be reversed without re-running anything.
    PAPER_METRICS = ("auroc", "fpr90", "vac")
    op = {m: [] for m, _t, _d in METRICS}

    def nll(y, p):
        """Binary negative log-likelihood, clipped so a saturated opinion cannot return inf."""
        q = np.clip(p, 1e-7, 1 - 1e-7)
        return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))

    def record(label, per):
        for m, _t, _d in METRICS:
            op[m].append((label, {d: per[d][m] for d in sets if m in per[d]}))

    for opc, label in PROB_OPS:                 # probability means: no opinion, margin only
        w = (1.0,) * len(FRAME) if opc == "F1" else \
            O.fit_weights(dev, FRAME, opc, objective="v31")[0]
        per = {}
        for ds in sets:
            pr = O.fuse(tabs[ds], FRAME, opc, list(w))
            y = tabs[ds].y.to_numpy()
            u = 1.0 - np.abs(2.0 * pr - 1.0)    # the only uncertainty a mean emits
            per[ds] = {"auroc": C.auroc(y, pr), "fpr90": T1.fpr_at_tpr(y, pr, 0.90),
                       "ece": T2.ece(y, pr), "brier": T2.brier(y, pr), "nll": nll(y, pr),
                       "eaurc": T2.e_aurc(y, pr, u, tau)}
        record(label + " †", per)
        print(f"  operator {label:22s} {np.mean([per[d]['auroc'] for d in sets]):.4f}", flush=True)

    for label in BELIEF_OPS:
        per = {}
        for ds in sets:
            t = tabs[ds]
            y = t.y.to_numpy()
            m = torch.tensor(O.masses(t, FRAME, list(w_full)), dtype=torch.float64)
            f = FNS[label](m)
            pr, u = PP.p_of(f).numpy(), f[:, 2].numpy()
            per[ds] = {"auroc": C.auroc(y, pr), "fpr90": T1.fpr_at_tpr(y, pr, 0.90),
                       "ece": T2.ece(y, pr), "brier": T2.brier(y, pr), "nll": nll(y, pr),
                       "eaurc": T2.e_aurc(y, pr, u, tau), "vac": float(u.mean())}
        record(label, per)
        print(f"  operator {label:22s} {np.mean([per[d]['auroc'] for d in sets]):.4f}", flush=True)

    L += ["---", "", "## Table 2 — Fusion operator", "",
          "The same three branches and the same reliability-scaled evidence under every rule,",
          "so the only thing that varies is where conflicting mass goes. The belief rules all",
          "come from one implementation; `Dempster` is the rule DiCoME uses and the one TMC's",
          "`DS_Combin` implements.", "",
          "**ES-PCR6+** is Evidence-Scaled PCR6+: each source's evidence is scaled by a fixed",
          "reliability coefficient before proportional conflict redistribution, `e'_v = r_v e_v`.",
          "The scaling acts in the EVIDENCE domain and is therefore not Shafer discounting, which",
          "acts on belief masses. Coefficients are fitted once on development and held fixed.", "",
          "† The two means produce a probability, not an opinion: they have no vacuity, and their",
          "E-AURC is ranked by the decision margin `1 - |2p - 1|` rather than by evidence. They",
          "are the comparator that shows what the evidential machinery buys.", "",
          "On a binary frame Yager coincides with Dubois-Prade, and PCR5 collapses onto PCR6, so",
          "several rules the literature separates are one row here.", "",
          "ES-PCR6+ is the adopted rule: it gives the lowest false-alarm rate at matched 90%",
          "sensitivity, which is the operating point an evidentiary use is read at, and it is",
          "the most committed rule by mean vacuity.", ""]
    def emit_operator_tables(dest, which):
        for m, title, direction in METRICS:
            if m not in which:
                continue
            # A probability mean has no vacuity at all -- an all-dash row is structurally empty,
            # not a missing measurement, so those rows leave this table rather than dashing.
            rows_m = [(n, c) for n, c in op[m] if c]
            if not rows_m:
                continue
            arrow = "" if direction == "—" else f" ({direction} is better)"
            dest += [f"### {title}{arrow}", ""]
            if m == "vac":
                dest += ["The two probability means are absent here: they emit no opinion, so "
                         "vacuity is undefined for them rather than zero.", ""]
            md(dest, "fusion rule", rows_m, sets)

    emit_operator_tables(L, tuple(m for m, _t, _d in METRICS) if ALL_METRICS else PAPER_METRICS)

    L += ["---", "", "## Table 3 — Comparison with the state of the art", "",
          "FPR is read at a MATCHED true-positive rate, so every method is compared at the same",
          "sensitivity: of the genuine videos, what fraction is flagged when the method is tuned",
          "to catch 90% of the fakes.", "", "### AUROC (higher is better)", ""]
    md(L, "method", rows_a, sets)
    L += ["### FPR at TPR = 90% (lower is better)", ""]
    md(L, "method", rows_f, sets)

    L += ["---", "", "## Table 4 — Probability quality and uncertainty", "",
          "ECE and Brier ask whether a reported probability means what it says; E-AURC asks",
          "whether the model knows which of its own answers to distrust. Ours ranks by the fused",
          "vacuity, the baselines by `1 - max p`, the only uncertainty a cross-entropy detector",
          "emits.", ""]
    for m, title in (("ece", "ECE-15"), ("brier", "Brier"), ("eaurc", "E-AURC")):
        L += [f"### {title} (lower is better)", ""]
        md(L, "method", rows_c[m], sets)

    for tno, f in enumerate(("face swap", "reenactment"), start=5):
        ks = sorted(g for g in pooled if fam[g] == f)
        if not ks:
            continue
        L += ["---", "", f"## Table {tno} — DF40 {FAMILY_NAME[f]}", "",
              f"{len(ks)} generators, each one test set: its Celeb-DF-sourced and",
              "FaceForensics++-sourced videos pooled, scored against both real pools. AUROC.", ""]
        order = sorted(df_rows, key=lambda k: -np.mean([df_rows[k][x] for x in ks
                                                        if x in df_rows[k]] or [0]))
        md(L, "method", [(k, {x: df_rows[k].get(x) for x in ks}) for k in order],
           ks)
        L[-1] = L[-1]
    # strip the DF40 column prefix for width
    text = "\n".join(L).replace("| DF40_", "| ")
    OUT.write_text(text + "\n")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
