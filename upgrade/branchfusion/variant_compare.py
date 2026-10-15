#!/usr/bin/env python3
"""Two candidate branch sets, head to head, on everything the choice should be made from.

    <discern_ext python> upgrade/branchfusion/variant_compare.py V+O+P16ctl V+O+S16ctl

UMAR's hypothesis on 2026-10-14: keep LFS and STT, drop App. The two candidates differ only in
STT's positional encoding, and the twelve test sets call that difference noise (0.0016 against
0.0126 seed) while DF40's reenactment half does not (+0.0213). So neither corpus decides it
alone, and this puts both in one place: per dataset on the twelve, per family on DF40's
forty-one generator sets, with detection AND probability quality, because a trio can win one and
lose the other.

Weights are fitted ONCE per candidate on the development pool and applied unchanged, and `tau`
is each candidate's own EER on FF++-val -- the same protocol as every other table here, so these
rows can be read beside them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                               # noqa: E402
import branches as B                                             # noqa: E402
import opinions as O                                             # noqa: E402
import table2_calibration as T2                                  # noqa: E402
import df40_crossmanip as D                                      # noqa: E402

C.thread_caps()
OUT_MD = "VARIANT_COMPARE"


def fpr_at_tpr(y, s, target):
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return None
    return float((neg >= float(np.quantile(pos, 1.0 - target))).mean())


def main() -> int:
    sys.path[:0] = [str(C.ROOT / "analysis/tbiom/pcr6p")]
    import ops_pcr6p as P
    from health import eer

    cands = [tuple(x.split("+")) for x in sys.argv[1:]] or [("V", "O", "P16ctl"),
                                                            ("V", "O", "S16ctl")]
    need = sorted(set().union(*cands))
    meta = json.loads((C.ROOT / "logs/tbiom/df40/meta.json").read_text())
    FAM = {k: v["family"] for k, v in meta.items()}
    gens = sorted({v["method"] for v in meta.values()})

    dev = B.dev_table(need, require_all=True)
    tabs = B.test_tables(need)
    sets = [d for d in C.TEST_SETS if d in tabs]
    df40_sets, _ = D.gens_available(gens)
    d40 = {ds: B.table([ds], need) for ds in df40_sets}
    print(f"{len(cands)} candidates, {len(sets)} test sets, {len(df40_sets)} DF40 sets",
          flush=True)

    res = {}
    for g in cands:
        w, devscore = O.fit_weights(dev, g, "F8", objective="v31")
        fd = P.fuse(O.masses(dev, g, list(w)), True)
        ffv = np.where(dev.source.to_numpy() == "FFpp_val")[0]
        tau = float(eer(dev.y.to_numpy()[ffv], P.p_of(fd)[ffv])[1])

        def score(t):
            fo = P.fuse(O.masses(t, g, list(w)), True)
            p, u, y = P.p_of(fo), fo[:, 2], t.y.to_numpy()
            return {"auroc": C.auroc(y, p), "fpr90": fpr_at_tpr(y, p, 0.90),
                    "fpr95": fpr_at_tpr(y, p, 0.95), "ece": T2.ece(y, p),
                    "brier": T2.brier(y, p), "eaurc": T2.e_aurc(y, p, u, tau), "n": int(len(t))}

        per = {ds: score(tabs[ds]) for ds in sets}
        d4 = {ds: score(d40[ds]) for ds in df40_sets}
        fam = {}
        for f in ("face swap", "reenactment"):
            ks = [k for k in df40_sets if FAM.get(k) == f]
            if ks:
                fam[f] = {m: float(np.mean([d4[k][m] for k in ks])) for m in ("auroc",)}
                fam[f]["sets"] = len(ks)
        res["+".join(g)] = {
            "name": C.show_subset(g), "weights": [float(x) for x in w],
            "dev_v31": float(devscore), "tau": tau, "per_dataset": per,
            "df40_family": fam,
            "df40_all": float(np.mean([d4[k]["auroc"] for k in df40_sets])),
            "means": {m: float(np.mean([per[d][m] for d in sets
                                        if isinstance(per[d][m], float)]))
                      for m in ("auroc", "fpr90", "fpr95", "ece", "brier", "eaurc")},
        }
        print(f"  {res['+'.join(g)]['name']:22s} dev {devscore:.4f}  "
              f"12set {res['+'.join(g)]['means']['auroc']:.4f}  "
              f"DF40 {res['+'.join(g)]['df40_all']:.4f}", flush=True)

    (C.OUT / "variant_compare.json").write_text(json.dumps(res, indent=1))

    names = [res[k]["name"] for k in res]
    L = [f"# {' vs '.join(names)}", "",
         "Both candidates keep LFS and STT and drop App, per UMAR 2026-10-14. They differ only in",
         "STT's positional encoding: real relative timestamps (`STT`) against learned slot",
         "positions (`STT-order`).", "",
         "Weights fitted once per candidate on development and applied unchanged; `tau` is each",
         "candidate's own EER on FF++-val. Same protocol as every other table here.", "",
         "| candidate | weights | dev v3.1 |", "|---|---|---:|"]
    for k, r in res.items():
        L.append(f"| {r['name']} | {[round(x, 2) for x in r['weights']]} | {r['dev_v31']:.4f} |")
    L.append("")

    for metric, title, better in (("auroc", "AUROC", "higher"), ("fpr90", "FPR at TPR = 90%", "lower"),
                                  ("fpr95", "FPR at TPR = 95%", "lower"), ("ece", "ECE-15", "lower"),
                                  ("brier", "Brier", "lower"), ("eaurc", "E-AURC", "lower")):
        L += [f"## {title} ({better} is better)", "",
              "| candidate | " + " | ".join(sets) + " | mean |",
              "|---|" + "---:|" * (len(sets) + 1)]
        for k, r in res.items():
            cells = [r["per_dataset"][d][metric] for d in sets]
            L.append(f"| {r['name']} | "
                     + " | ".join(f"{c:.4f}" if isinstance(c, float) else "—" for c in cells)
                     + f" | **{r['means'][metric]:.4f}** |")
        # the difference row, so the comparison does not have to be done by eye
        ks = list(res)
        if len(ks) == 2:
            a, b = res[ks[0]], res[ks[1]]
            diffs = [a["per_dataset"][d][metric] - b["per_dataset"][d][metric] for d in sets]
            L.append(f"| *{a['name']} − {b['name']}* | "
                     + " | ".join(f"{x:+.4f}" for x in diffs)
                     + f" | *{a['means'][metric] - b['means'][metric]:+.4f}* |")
        L.append("")

    L += ["## DF40 — all 41 generator sets, by manipulation family", "",
          "| candidate | face swap | reenactment | all 41 |", "|---|---:|---:|---:|"]
    for k, r in res.items():
        fs = r["df40_family"].get("face swap", {}).get("auroc")
        fr = r["df40_family"].get("reenactment", {}).get("auroc")
        L.append(f"| {r['name']} | {fs:.4f} | {fr:.4f} | **{r['df40_all']:.4f}** |")
    if len(res) == 2:
        a, b = [res[k] for k in res]
        L.append(f"| *{a['name']} − {b['name']}* | "
                 + f"{a['df40_family']['face swap']['auroc'] - b['df40_family']['face swap']['auroc']:+.4f} | "
                 + f"{a['df40_family']['reenactment']['auroc'] - b['df40_family']['reenactment']['auroc']:+.4f} | "
                 + f"*{a['df40_all'] - b['df40_all']:+.4f}* |")
    L.append("")
    (C.REPORT / f"{OUT_MD}.md").write_text("\n".join(L) + "\n")
    print(f"-> {C.REPORT / f'{OUT_MD}.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
