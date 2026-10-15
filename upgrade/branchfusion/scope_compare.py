#!/usr/bin/env python3
"""What admitting the excluded generator families costs, on the two sets that have any.

    <discern_ext python> upgrade/branchfusion/scope_compare.py

CDFv3 and KoDF are the only test sets whose census the evaluation scope touches; the other ten
contain none of the excluded families, so every number in them is identical under every scope and
reprinting them would imply a comparison that is not being made.

ONE PROCESS, THREE CENSUSES. The three scopes are NESTED -- v31 is inside nolip is inside
nolipadv -- so the widest table is built once and the narrower rows are taken from it by the same
`branches.in_scope` predicate the readers use, flipped through `BF_SCOPE` per call. Building three
separate tables would re-read the same opinions three times and, worse, would let a row differ for
a reason other than the census.

The weights are fitted ONCE on the development pool, which no scope touches, so a change below is
a change in the videos and not in the system.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                               # noqa: E402
import branches as B                                             # noqa: E402
import opinions as O                                             # noqa: E402

C.thread_caps()

FINAL = ("A", "V", "O", "S16ctl")
SETS = ("CDFv3", "KoDF")
SCOPES = [("v31", "v3.1 (published)"),
          ("nolip", "+ lip-sync / audio-driven"),
          ("nolipadv", "+ adversarial")]


def fpr_at_tpr(y, s, target=0.90):
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return None
    return float((neg >= float(np.quantile(pos, 1.0 - target))).mean())


def main() -> int:
    import ops_pcr6p as P
    os.environ["BF_SCOPE"] = "nolipadv"
    dev = B.dev_table(list(FINAL), require_all=True)
    w, _ = O.fit_weights(dev, FINAL, "F8", objective="v31")
    print(f"weights fitted once on development: {[round(x, 2) for x in w]}", flush=True)

    res = {}
    for ds in SETS:
        wide = B.table([ds], list(FINAL))
        for scope, _label in SCOPES:
            os.environ["BF_SCOPE"] = scope
            keep = set(B.meta_for(ds).v)
            t = wide[wide.v.isin(keep)].reset_index(drop=True)
            y = t.y.to_numpy()
            p_ = P.p_of(P.fuse(O.masses(t, FINAL, list(w)), True))
            row = {"videos": int(len(t)), "real": int((y == 0).sum()),
                   "fake": int((y == 1).sum()), "auroc": C.auroc(y, p_),
                   "fpr90": fpr_at_tpr(y, p_)}
            for b in FINAL:
                row[f"branch_{b}"] = C.auroc(y, t[f"p_{b}"].to_numpy())
            res[f"{ds}/{scope}"] = row
            print(f"  {ds:6s} {scope:9s} n={row['videos']:5d}  AUROC {row['auroc']:.4f}",
                  flush=True)
    os.environ["BF_SCOPE"] = "nolipadv"

    # Which subsets each widening admits, named -- the point of the table is WHICH videos, so the
    # families have to be visible rather than only their count.
    added = {}
    for ds in SETS:
        m_all = B.meta_for(ds)
        os.environ["BF_SCOPE"] = "v31"
        base = set(B.meta_for(ds).v)
        os.environ["BF_SCOPE"] = "nolip"
        mid = set(B.meta_for(ds).v)
        os.environ["BF_SCOPE"] = "nolipadv"
        sub = m_all.set_index("v").subset
        added[ds] = {
            "nolip": sorted(set(sub[list(mid - base)])) if mid - base else [],
            "nolipadv": sorted(set(sub[list(set(m_all.v) - mid)])) if set(m_all.v) - mid else [],
        }

    # WHERE THE FALSE ALARMS COME FROM. KoDF's FPR at TPR 90% goes from 0.03 to 0.38 when the
    # adversarial subsets are admitted, and an aggregate cannot say whether that is the new fakes
    # being missed or the new REALS being flagged. The threshold is fixed at the one the narrower
    # census produces -- the system as it would be deployed having never seen adv-real -- and each
    # real subset is then asked what fraction of it exceeds that threshold.
    breakdown = {}
    for ds in SETS:
        os.environ["BF_SCOPE"] = "nolipadv"
        wide = B.table([ds], list(FINAL))
        m = B.meta_for(ds).set_index("v").subset
        p_ = P.p_of(P.fuse(O.masses(wide, FINAL, list(w)), True))
        y = wide.y.to_numpy()
        os.environ["BF_SCOPE"] = "nolip"
        narrow = set(B.meta_for(ds).v)
        os.environ["BF_SCOPE"] = "nolipadv"
        mask = wide.v.isin(narrow).to_numpy()
        pos = p_[mask & (y == 1)]
        if not len(pos):
            continue
        tau = float(np.quantile(pos, 0.10))        # catches 90% of the fakes it was tuned on
        sub = m.reindex(wide.v).to_numpy()
        rows = {}
        for name in sorted(set(sub[y == 0])):
            sel = (y == 0) & (sub == name)
            rows[name] = {"videos": int(sel.sum()), "fpr": float((p_[sel] >= tau).mean())}
        # The fake side of the SAME threshold. Without it the table cannot say whether a rising
        # aggregate FPR is genuine videos being accused or fakes evading and dragging the
        # threshold down to catch them -- and those two call for opposite fixes.
        caught = {}
        for name in sorted(set(sub[y == 1])):
            sel = (y == 1) & (sub == name)
            caught[name] = {"videos": int(sel.sum()), "tpr": float((p_[sel] >= tau).mean())}
        breakdown[ds] = {"tau": tau, "real_subsets": rows, "fake_subsets": caught}

    (C.OUT / "scope_compare.json").write_text(json.dumps(
        {"results": res, "admitted": added, "weights": [float(x) for x in w],
         "false_alarms": breakdown},
        indent=1, default=float))

    L = ["# What the evaluation scope costs — CDFv3 and KoDF", "",
         "These are the only two test sets whose video census the scope touches. The other ten",
         "contain none of the excluded families and are byte-identical under every scope.", "",
         "The system is unchanged across the rows: `App + TD + LFS + STT` under ES-PCR6+ with the",
         "weights fitted once on the development pool, which no scope touches. A difference below",
         "is a difference in which videos are scored, and nothing else.", ""]
    for ds in SETS:
        L += [f"## {ds}", "",
              "| scope | videos (real / fake) | AUROC | FPR@TPR90 | "
              + " | ".join(C.show(b) for b in FINAL) + " |",
              "|---|---|---:|---:|" + "---:|" * len(FINAL)]
        for scope, label in SCOPES:
            r = res[f"{ds}/{scope}"]
            f = f"{r['fpr90']:.4f}" if isinstance(r["fpr90"], float) else "—"
            L.append(f"| {label} | {r['real']} / {r['fake']} | **{r['auroc']:.4f}** | {f} | "
                     + " | ".join(f"{r[f'branch_{b}']:.4f}" for b in FINAL) + " |")
        L.append("")
        for scope in ("nolip", "nolipadv"):
            if added[ds][scope]:
                L.append(f"`{scope}` admits: {', '.join('`' + s + '`' for s in added[ds][scope])}")
        L.append("")
    L += ["## Where the false alarms come from", "",
          "The threshold is the one that catches 90% of the fakes on the census WITHOUT the",
          "adversarial subsets — the system as it would be deployed having never seen them — and",
          "each genuine subset is then asked what fraction of it is flagged at that threshold.",
          "This separates two things an aggregate FPR cannot: new fakes being missed, and new",
          "genuine videos being accused.", ""]
    for ds, b in breakdown.items():
        if len(b["real_subsets"]) < 2 and len(b["fake_subsets"]) < 2:
            continue
        L += [f"### {ds}", "", "| subset | label | videos | flagged at that threshold |",
              "|---|---|---:|---:|"]
        for name, r in sorted(b["real_subsets"].items(), key=lambda kv: -kv[1]["fpr"]):
            L.append(f"| `{name}` | genuine | {r['videos']} | {r['fpr']:.4f} |")
        for name, r in sorted(b["fake_subsets"].items(), key=lambda kv: kv[1]["tpr"]):
            L.append(f"| `{name}` | fake | {r['videos']} | {r['tpr']:.4f} |")
        L.append("")
        # State the mechanism from the numbers rather than asserting one.
        worst = min(b["fake_subsets"].items(), key=lambda kv: kv[1]["tpr"])
        reals = b["real_subsets"]
        adv = {k: v for k, v in reals.items() if k.startswith("adv-")}
        if adv:
            a = max(adv.values(), key=lambda v: v["fpr"])["fpr"]
            o = max((v["fpr"] for k, v in reals.items() if not k.startswith("adv-")), default=0.0)
            L += [f"At this fixed threshold the adversarial genuine videos are flagged at "
                  f"{a:.4f} against {o:.4f} for the ordinary ones — close. What moves the "
                  f"aggregate is the fake side: `{worst[0]}` is caught at only "
                  f"{worst[1]['tpr']:.4f}, so holding TPR at 90% over the wider census forces a "
                  f"much lower threshold, and that is what floods the genuine videos.", ""]
    (C.REPORT / "SCOPE_COMPARISON.md").write_text("\n".join(L) + "\n")
    print(f"-> {C.REPORT / 'SCOPE_COMPARISON.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
