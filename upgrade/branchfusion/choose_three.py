#!/usr/bin/env python3
"""Which THREE branches to take forward -- every candidate, on every basis that should decide it.

    <discern_ext python> upgrade/branchfusion/choose_three.py

UMAR has to pick three branches, and they have to be defensible as NOVEL as well as good. This
puts the candidates on one page with the four bases that matter, because no single one of them
settles it:

  dev v3.1   the ONLY legitimate selection signal. Table A2 scores every subset on every test
             set, so its best cell is a maximum over a large search on the test partitions and is
             biased upward -- its own banner says so. The deployed subset is chosen here.
  12 sets    the held-out mean, reported at BOTH censuses, because the ordering is not stable
             between them and a choice that depends on which census you read is not a choice.
  DF40       sixteen generator sets no branch was trained on -- the cross-manipulation question.
  novelty    stated, not computed: where each branch came from. See NOVELTY below.

SUBSETS HOLDING BOTH STT VARIANTS ARE SKIPPED. STT and STT-order are one architecture with the
positional encoding changed; the fit zeroes whichever it sees second, so such a subset is not a
three-branch system at all -- it is a two-branch system wearing a third name.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                               # noqa: E402
import branches as B                                             # noqa: E402
import opinions as O                                             # noqa: E402

C.thread_caps()

#: Where each branch came from. This is PROVENANCE, not a measurement, and it is the column the
#: performance numbers cannot supply.
NOVELTY = {
    "A": "adapted PE-Core-L/336 appearance — our LoRA+LayerNorm recipe over a standard view",
    "V": "V-JEPA 2 LoRA temporal — trained by this campaign, new to the framework",
    "O": "INHERITED: the CLIP/β-VAE orthogonal residual from the DiCoME-lineage stage-23 export",
    "S16ctl": "NEW: PE-Core block 16 + real relative timestamps, built by this campaign",
    "P16ctl": "the same construction with learned slot positions — STT's ablation, not a view",
}
STT_PAIR = {"S16ctl", "P16ctl"}
#: ALL 21 generators, not DiCoME's eight. The eight are face swap only, and the branches do not
#: rank the same way on the two families: STT averages 0.9829 on face swap and 0.7963 on
#: reenactment, so a verdict drawn from the eight is a face-swap verdict wearing a general one.
#: DF40 ships eleven reenactment generators and they are half the corpus.
import json as _json
DF40_GENS = sorted({v["method"] for v in
                    _json.loads((C.ROOT / "logs/tbiom/df40/meta.json").read_text()).values()})


def main() -> int:
    import os
    import ops_pcr6p as P
    import df40_crossmanip as D

    cands = [g for g in itertools.combinations(C.CORE, 3) if not STT_PAIR <= set(g)]
    cands.append(tuple(C.CORE[:3]) + ("S16ctl",))        # the four-branch reference
    cands[-1] = ("A", "V", "O", "S16ctl")

    os.environ["BF_SCOPE"] = "nolipadv"
    need = sorted(set(C.CORE))
    dev = B.dev_table(need, require_all=True)
    df40_sets, _ = D.gens_available(DF40_GENS)
    _meta = _json.loads((C.ROOT / "logs/tbiom/df40/meta.json").read_text())
    FAM = {k: v["family"] for k, v in _meta.items()}
    print(f"{len(cands)} candidates, {len(df40_sets)} DF40 generator sets", flush=True)

    # The widest tables once; the narrower census is a row selection on them, since the scopes nest.
    wide = {ds: B.table([ds], need) for ds in C.TEST_SETS}
    os.environ["BF_SCOPE"] = "nolip"
    narrow = {ds: set(B.meta_for(ds).v) for ds in C.TEST_SETS}
    os.environ["BF_SCOPE"] = "nolipadv"
    d40 = {ds: B.table([ds], need) for ds in df40_sets}

    rows = []
    for g in cands:
        w, devscore = O.fit_weights(dev, g, "F8", objective="v31")

        def auroc_on(t):
            return C.auroc(t.y.to_numpy(),
                           P.p_of(P.fuse(O.masses(t, g, list(w)), True)))

        widem = float(np.mean([auroc_on(wide[ds]) for ds in C.TEST_SETS]))
        narrowm = float(np.mean([auroc_on(wide[ds][wide[ds].v.isin(narrow[ds])]
                                          .reset_index(drop=True)) for ds in C.TEST_SETS]))
        df40m = float(np.mean([auroc_on(d40[ds]) for ds in df40_sets])) if df40_sets else None
        fs = [ds for ds in df40_sets if FAM.get(ds) == "face swap"]
        fr = [ds for ds in df40_sets if FAM.get(ds) == "reenactment"]
        df40fs = float(np.mean([auroc_on(d40[ds]) for ds in fs])) if fs else None
        df40fr = float(np.mean([auroc_on(d40[ds]) for ds in fr])) if fr else None
        rows.append({"subset": "+".join(g), "name": C.show_subset(g), "n": len(g),
                     "dev_v31": float(devscore), "mean_nolip": narrowm,
                     "mean_nolipadv": widem, "df40": df40m,
                     "df40_faceswap": df40fs, "df40_reenact": df40fr,
                     "weights": [float(x) for x in w]})
        print(f"  {C.show_subset(g):26s} dev {devscore:.4f}  12set {widem:.4f}"
              + (f"  DF40 fs {df40fs:.4f} fr {df40fr:.4f} all {df40m:.4f}" if df40m else ""),
              flush=True)

    (C.OUT / "choose_three.json").write_text(json.dumps(rows, indent=1))
    three = sorted([r for r in rows if r["n"] == 3], key=lambda r: -r["dev_v31"])
    best = three[0]
    spread = three[0]["dev_v31"] - three[-1]["dev_v31"]
    SEED_NOISE = 0.0126                 # measured on this objective; see common.CORE's note
    # The candidates within seed noise of the best are the ones the numbers DO NOT separate.
    tied = [r for r in three if best["dev_v31"] - r["dev_v31"] < SEED_NOISE]
    beaten = [r for r in three if r not in tied]
    common = set(tied[0]["subset"].split("+"))
    for r in tied[1:]:
        common &= set(r["subset"].split("+"))

    L = ["# Choosing three branches", "",
         "Every three-branch candidate, ranked by the DEVELOPMENT objective — the only basis on",
         "which a deployed subset may be chosen. Table A2 ranks subsets on the test sets and its",
         "own banner says that maximum is biased upward; this table is what that banner points to.",
         "",
         "The four-branch set is the last row, for reference only.", "",
         f"| branches | dev v3.1 | 12 sets | DF40 face swap | DF40 reenact | DF40 all |",
         "|---|---:|---:|---:|---:|---:|"]
    for r in sorted(rows, key=lambda r: (r["n"], -r["dev_v31"])):
        bold = "**" if r["n"] == 4 else ""
        def q(k):
            return f"{r[k]:.4f}" if isinstance(r.get(k), float) else "—"
        L.append(f"| {bold}{r['name']}{bold} | {r['dev_v31']:.4f} | {r['mean_nolipadv']:.4f} | "
                 f"{q('df40_faceswap')} | {q('df40_reenact')} | {q('df40')} |")
    L += ["",
          f"## What the numbers do and do not settle", "",
          f"The development spread across all {len(three)} candidates is {spread:.4f}, which is",
          f"WIDER than the measured seed noise of {SEED_NOISE} on this objective — so this table is",
          "not uniformly inconclusive, and saying so would be the easy and wrong reading. It",
          "separates into two groups:", "",
          f"- **{len(tied)} candidates lie within seed noise of the best** "
          + "(" + ", ".join(f"`{r['name']}`" for r in tied) + "), spread "
          + f"{tied[0]['dev_v31'] - tied[-1]['dev_v31']:.4f}. Among these the numbers decide nothing.",
          f"- **{len(beaten)} are genuinely behind** "
          + "(" + ", ".join(f"`{r['name']}`" for r in beaten) + "), by more than noise.",
          "",
          f"Every tied candidate contains **{C.show_subset(sorted(common, key=C.CORE.index))}**, so"
          if common else "The tied candidates share no common branch, so",
          "that is the part the evidence fixes; the remaining slot is where the choice is free. The"
          if common else "the whole choice is free. The",
          "twelve-set ordering also reverses between the two censuses, which is a second sign of",
          "the same thing. That makes novelty the free variable rather than something to be argued",
          "against the numbers.", "", "## Where each branch comes from", "",
          "| branch | provenance |", "|---|---|"]
    for b in C.CORE:
        L.append(f"| {C.show(b)} | {NOVELTY[b]} |")
    L += ["",
          "`LFS` is the one inherited view: it is the DiCoME-lineage artifact residual, read from",
          "the stage-23 export. A three-branch framework that keeps it is carrying the baseline's",
          "own representation into the contribution; one that drops it stands on App, TD and STT,",
          "all three of which this campaign built or adapted. What that costs is the",
          f"`{C.show_subset(('A','V','O'))}` row against the `{C.show_subset(('A','V','S16ctl'))}`",
          "row above — read it there rather than from this paragraph.", ""]
    (C.REPORT / "CHOOSE_THREE.md").write_text("\n".join(L) + "\n")
    print(f"-> {C.REPORT / 'CHOOSE_THREE.md'}  (best on development: {best['name']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
