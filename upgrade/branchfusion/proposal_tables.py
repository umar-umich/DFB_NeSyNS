#!/usr/bin/env python3
"""The three tables for the proposal draft. Cells that cannot be computed yet say IN PROGRESS.

    <discern_ext python> upgrade/branchfusion/proposal_tables.py

  Table A  BRANCH ABLATION      ten groups x every dataset. No baselines: this table answers
                                "what does each branch contribute", and a SOTA column in it
                                would invite the two questions to be read off one row.
  Table B  FUSION OPERATOR      the final five-branch combination under each decision rule --
                                mean-of-logits (the CE arm), Mean, DS-C, and PCR6+-R.
  Table C  COMPARATIVE          the final combination against every state-of-the-art method we
                                hold weights or scores for. Built by `sota_table1.py`.

IN PROGRESS, not blank and not omitted. A dataset with no branch scores yet, or an arm still
training, prints `*in progress*` in that cell. An empty cell reads as "not applicable" and a
missing row reads as "not tried"; neither is true here, and a draft table that quietly loses a
column is how a number goes into a thesis unchecked.

THE CE ROW IS NOT FUSED BY DS OR PCR6+. A cross-entropy branch has no Dirichlet, so its `u` is a
decision margin and not a vacuity; combining those as belief masses would invent evidence that
was never measured. It is fused by mean-of-logits, which is what DiCoME's row (D) does and the
only defensible choice.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C                                               # noqa: E402
import branches as B                                             # noqa: E402
import opinions as O                                             # noqa: E402
import subsets as S                                              # noqa: E402

#: The framework. `App + TD + LFS + STT` -- four branches, best group in the ablation at 0.9534.
#: The five-branch set is NOT used: its weight fit zeroes S16ctl, so it is literally identical to
#: App+TD+LFS+STT-order at 0.9518, and presenting a five-branch system in which one branch carries
#: zero weight would be a claim the fit does not support.
FINAL = ("A", "V", "O", "S16ctl")
FINAL_GROUP = ("A", "V", "O", "S16ctl")
CE_SET = ("AppCE", "VCE", "LFSCE", "S16ctlCE")
#: The ten ablation groups, per UMAR: each branch alone, the deployed trio, each new branch added
#: to it, the full five, and the drop-one check on the only branch the subset search found
#: irreplaceable.
GROUPS = [("A",), ("V",), ("O",), ("P16ctl",), ("S16ctl",),
          ("A", "V", "O"),
          ("A", "V", "O", "P16ctl"), ("A", "V", "O", "S16ctl"),
          ("A", "V", "O", "P16ctl", "S16ctl"),
          ("A", "O", "P16ctl", "S16ctl")]
OPS = [("F1", "Mean"), ("F7", "DS-C"), ("F8", "PCR6+-R (ours)")]
NEW_SETS = ["UADFV", "DeepSpeak-1.1", "DeepSpeak-2.0"]
TODO = "*in progress*"
OUT = C.ROOT / "reports/branch_fusion"


def fpr_at_tpr(y, s, target=0.90):
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return None
    return float((neg >= float(np.quantile(pos, 1.0 - target))).mean())


def table(lines, header, rows, sets):
    lines += ["| " + header + " | " + " | ".join(sets) + " | mean |",
              "|---|" + "---:|" * (len(sets) + 1)]
    for name, cells in rows:
        got = [v for v in cells.values() if isinstance(v, float)]
        m = f"{np.mean(got):.4f}" if got else TODO
        lines.append(f"| {name} | " +
                     " | ".join(f"{cells[d]:.4f}" if isinstance(cells.get(d), float)
                                else TODO for d in sets) + f" | {m} |")
    lines.append("")


def main() -> int:
    # Every branch any GROUP needs, not just the framework's four: the ablation still reports
    # STT-order and the singles, and loading only FINAL would make those rows unresolvable.
    need = sorted({b for g in GROUPS for b in g} | set(FINAL))

    # NEW_SETS are the corpora promoted into TEST_SETS once every branch covered them, so the
    # union is de-duplicated rather than concatenated -- otherwise each appears twice as a column.
    sets = list(dict.fromkeys(list(C.TEST_SETS) + NEW_SETS))
    # Compute on every set whose branches are ALL present. A newly packed corpus was otherwise
    # printed as a column and never filled, however complete its coverage -- the table said
    # "in progress" about data sitting on disk.
    avail = [d for d in sets if all(B.available(b, d) or (C.BRANCH_OUT / b / f"{d}.csv").exists()
                                    for b in need)]
    print(f"computing on {len(avail)} of {len(sets)} sets: {avail}", flush=True)

    dev = B.dev_table(need, require_all=True)
    tabs = B.test_tables(need, sets=avail)

    # ---------------------------------------------------------------- Table A
    # Weights are fitted ONCE per group on the development pool and then applied unchanged to
    # every test set -- which is also the locked protocol. Fitting inside the dataset loop, as
    # the first version did, refits the same 2,101-vector grid nine times per group and tunes
    # the weights on the set being reported.
    import ops_pcr6p as P
    auroc_rows, fpr_rows = [], []
    for g in GROUPS:
        name = C.show_subset(g) + (" *(deployed)*" if g == ("A", "V", "O") else
                                   " **(final)**" if g == FINAL_GROUP else "")
        w, _ = O.fit_weights(dev, g, "F8", objective="v31")
        a, f = {}, {}
        for ds in avail:
            t = tabs[ds]
            p_ = P.p_of(P.fuse(O.masses(t, g, list(w)), True))
            y = t.y.to_numpy()
            a[ds] = C.auroc(y, p_); f[ds] = fpr_at_tpr(y, p_)
        auroc_rows.append((name, a)); fpr_rows.append((name, f))
        print(f"  A: {name}  weights {[round(x, 2) for x in w]}", flush=True)

    L = ["# Table A — branch ablation", "",
         "Ten groups, every dataset, fused with **PCR6+-R**. No baseline rows: this table answers",
         "what each branch contributes, and a SOTA column here would invite two questions to be",
         "read off one row. The comparison against the field is Table C.", "",
         f"`{TODO}` means the cell is not computable yet, not that it is zero or not applicable.",
         f"The three newly packed sets ({', '.join(NEW_SETS)}) have no branch scores yet.", "",
         "## AUROC (higher is better)", ""]
    table(L, "branch group", auroc_rows, sets)
    L += ["## FPR at TPR = 90% (lower is better)", ""]
    table(L, "branch group", fpr_rows, sets)
    (OUT / "TABLE_A_ABLATION.md").write_text("\n".join(L) + "\n")

    # ---------------------------------------------------------------- Table B
    rows_a, rows_f = [], []
    for op, label in OPS:
        w, _ = O.fit_weights(dev, FINAL, op, objective="v31")
        a, f = {}, {}
        for ds in avail:
            t = tabs[ds]
            p_ = O.fuse(t, FINAL, op, list(w))
            y = t.y.to_numpy()
            a[ds] = C.auroc(y, p_); f[ds] = fpr_at_tpr(y, p_)
        rows_a.append((label, a)); rows_f.append((label, f))
        print(f"  B: {label}  weights {[round(x, 2) for x in w]}", flush=True)
    # CE arm: mean of the five CE branches' probabilities, the mean-of-logits analogue.
    ce_a, ce_f = {}, {}
    # The CE arm has its OWN coverage: its four counterparts were never scored on the three newly
    # packed corpora. Computing it over `avail` made the whole row fail on the first set it could
    # not resolve, which reported "in progress" for nine datasets whose CE numbers exist. The row
    # is built on the sets where every CE branch is present and marked in progress elsewhere.
    missing = [b for b in CE_SET if b not in B.SOURCES]
    if missing:
        print(f"  B: CE counterparts not registered: {missing}", flush=True)
    else:
        ce_sets = [d for d in avail
                   if all(B.available(b, d) or (C.BRANCH_OUT / b / f"{d}.csv").exists()
                          for b in CE_SET)]
        print(f"  B: CE arm on {len(ce_sets)} sets", flush=True)
        if ce_sets:
            ctabs = B.test_tables(list(CE_SET), sets=ce_sets)
            for ds in ce_sets:
                t = ctabs[ds]
                p_ = np.mean([t[f"p_{b}"].to_numpy() for b in CE_SET], axis=0)
                y = t.y.to_numpy()
                ce_a[ds] = C.auroc(y, p_); ce_f[ds] = fpr_at_tpr(y, p_)
    rows_a.insert(0, ("Cross-entropy (mean of logits)", ce_a))
    rows_f.insert(0, ("Cross-entropy (mean of logits)", ce_f))

    L = ["# Table B — fusion operator on the final combination (App + TD + LFS + STT)", "",
         "`App + TD + LFS + STT` under each decision rule. The cross-entropy row is fused",
         "by mean-of-logits and NOT by DS or PCR6+: a CE branch has no Dirichlet, so its",
         "uncertainty is a decision margin, and combining margins as belief masses would invent",
         "evidence that was never measured. Mean-of-logits is what DiCoME's row (D) does.", "",
         "## AUROC (higher is better)", ""]
    table(L, "fusion rule", rows_a, sets)
    L += ["## FPR at TPR = 90% (lower is better)", ""]
    table(L, "fusion rule", rows_f, sets)
    (OUT / "TABLE_B_FUSION.md").write_text("\n".join(L) + "\n")

    json.dump({"ablation_auroc": {n: c for n, c in auroc_rows},
               "ablation_fpr90": {n: c for n, c in fpr_rows},
               "fusion_auroc": {n: c for n, c in rows_a},
               "fusion_fpr90": {n: c for n, c in rows_f}},
              open(C.ROOT / "analysis/branchfusion/proposal_tables.json", "w"),
              indent=1, default=str)
    print(f"-> {OUT / 'TABLE_A_ABLATION.md'}\n-> {OUT / 'TABLE_B_FUSION.md'}")
    return 0


if __name__ == "__main__":
    sys.path[:0] = [str(C.ROOT / "analysis/tbiom/pcr6p")]
    raise SystemExit(main())
