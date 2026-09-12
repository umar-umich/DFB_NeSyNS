#!/usr/bin/env python3
"""Stage 2 — per-cue reliability against processing severity.

    python analysis/tbiom/stage2_curves.py --out tbiom/reports/EVIDENCE_STAGE2.md

WHAT RELIABILITY MEANS HERE, stated before it is measured. Reliability is how much a cue can be
trusted under this sample's operating conditions, and it is NOT the same as the cue pointing to
fake. A cue can shift a long way under compression and still separate the classes perfectly, if
the shift is common-mode across both. So two quantities are reported per cue and severity and
they answer different questions:

  discrimination   the cue's AUROC at that severity, on DEV where both classes received the
                   IDENTICAL processing. Falling AUROC means the cue is losing its ability to
                   tell real from fake -- this is what reliability should track.
  value drift      standardised shift of the raw measurement against its own baseline,
                   pooled over both classes. Large drift with intact AUROC is a nuisance the
                   fusion can absorb; large drift WITH collapsing AUROC is the failure mode the
                   whole diagnostic is about.

MATCHED PROCESSING IS THE CONTROL. Both classes get the same severity distribution, so nothing
here can learn "compressed means fake". If processing were applied only to reals, every curve
below would be an artifact of that choice.

TWO LIMITS, RECORDED BECAUSE THEY BOUND THE CLAIM.

  Processing is ADDITIVE, not a clean intervention. Every frame already carries its dataset's
  own encoding history, so `jpeg:30` is thirty-quality applied on top of that. Severities are
  ordinal within a dataset and not absolute across datasets.

  Processing augmentation cannot supply genuine identities, cameras or scenes the real class
  does not already contain. If the real-class support is too narrow, no severity ladder fixes
  it, and that limit is stated rather than hidden.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from health import auroc  # noqa: E402
from video_id import video_id  # noqa: E402
from view_names import VIEW_NAME  # noqa: E402

S2 = Path("logs/tbiom/stage2")
BR = ["semantic", "artifact", "fsvfm"]
VALUE = {"semantic": "norm_semantic", "artifact": "recon_err", "fsvfm": None}
FIT_DS = ["FFpp_val", "VALmix"]
HELD_DS = ["CDFv2", "DFDCP"]
FIT_TAGS = [("base", "none"), ("jpeg70", "jpeg q70"), ("jpeg50", "jpeg q50"),
            ("jpeg30", "jpeg q30"), ("rs075", "resize 0.75"), ("rs05", "resize 0.50"),
            ("rs035", "resize 0.35")]
HELD_TAGS = [("base", "none"), ("jpeg40", "jpeg q40"), ("rs06", "resize 0.60")]


def load(ds: str, tag: str):
    f = S2 / f"p_{ds}_{tag}.csv"
    if not f.is_file():
        return None
    d = pd.read_csv(f)
    d["v"] = video_id(d["key"])
    cols = [c for c in d.columns if c.startswith(("p_", "u_", "recon", "resid", "kl_", "norm_"))]
    agg = {c: (c, "mean") for c in cols}
    agg["y"] = ("label", "max")
    return d.groupby("v", as_index=False).agg(**agg)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("tbiom/reports/EVIDENCE_STAGE2.md"))
    args = ap.parse_args()

    rows, missing = [], []
    for ds_list, tags, role in ((FIT_DS, FIT_TAGS, "fit"), (HELD_DS, HELD_TAGS, "held-out")):
        for ds in ds_list:
            base = load(ds, "base")
            for tag, nice in tags:
                g = load(ds, tag)
                if g is None or base is None:
                    missing.append(f"{ds}/{tag}"); continue
                y = g["y"].to_numpy()
                for b in BR:
                    if f"p_{b}" not in g:
                        continue
                    a = auroc(y, g[f"p_{b}"].to_numpy())
                    vc = VALUE[b]
                    if vc and vc in g and vc in base:
                        s = base[vc].std() + 1e-9
                        drift = float((g[vc].mean() - base[vc].mean()) / s)
                    else:
                        drift = float("nan")
                    rows.append({"role": role, "dataset": ds, "severity": nice, "tag": tag,
                                 "cue": b, "auroc": a, "drift": drift,
                                 "vacuity": float(g[f"u_{b}"].mean())})
                rows.append({"role": role, "dataset": ds, "severity": nice, "tag": tag,
                             "cue": "fused", "auroc": auroc(y, g["p_fused"].to_numpy()),
                             "drift": float("nan"), "vacuity": float(g["u_fused"].mean())})

    if not rows:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("# Stage 2\n\n**No processed exports yet.**\n")
        print("no data yet"); return 0
    R = pd.DataFrame(rows)

    L = ["# Stage 2 — reliability against processing", "",
         "Both classes receive identical processing, so nothing here can learn "
         "\"compressed means fake\". Severities are **ordinal within a dataset**: processing is "
         "additive on top of each dataset's existing encoding, not a clean intervention.", ""]

    for role in ("fit", "held-out"):
        S = R[R.role == role]
        if S.empty:
            continue
        L += [f"## {'Fit — dev, both classes' if role == 'fit' else 'Held-out severities'}", ""]
        for ds in sorted(S.dataset.unique()):
            D = S[S.dataset == ds]
            sev = [t for t, _ in (FIT_TAGS if role == "fit" else HELD_TAGS)
                   if t in set(D.tag)]
            names = {t: n for t, n in (FIT_TAGS if role == "fit" else HELD_TAGS)}
            L += [f"**{ds}** — AUROC by cue and severity", "",
                  "| cue | " + " | ".join(names[t] for t in sev) + " |",
                  "|---" * (len(sev) + 1) + "|"]
            for cue in BR + ["fused"]:
                c = D[D.cue == cue].set_index("tag")
                if c.empty:
                    continue
                lab = VIEW_NAME.get(cue, cue.title())
                cells = [f"{c.loc[t,'auroc']:.4f}" if t in c.index else "—" for t in sev]
                L.append(f"| {'**' + lab + '**' if cue == 'fused' else lab} | "
                         + " | ".join(cells) + " |")
            L.append("")
            # value drift, where a cue has a scalar
            drift_rows = [cue for cue in BR if VALUE[cue]]
            L += ["Value drift (standardised, vs this dataset's own baseline)", "",
                  "| cue | measurement | " + " | ".join(names[t] for t in sev if t != "base") + " |",
                  "|---" * (len(sev) + 1) + "|"]
            for cue in drift_rows:
                c = D[D.cue == cue].set_index("tag")
                cells = [f"{c.loc[t,'drift']:+.2f}" if t in c.index else "—"
                         for t in sev if t != "base"]
                L.append(f"| {VIEW_NAME[cue]} | `{VALUE[cue]}` | " + " | ".join(cells) + " |")
            L.append("")

    # ------------------------------------------------------------------ reading
    fit = R[(R.role == "fit")]
    if not fit.empty:
        base = fit[fit.tag == "base"].set_index(["dataset", "cue"])["auroc"]
        worst = []
        for cue in BR + ["fused"]:
            c = fit[(fit.cue == cue) & (fit.tag != "base")]
            if c.empty:
                continue
            drops = []
            for _, r in c.iterrows():
                b = base.get((r.dataset, cue))
                if b is not None:
                    drops.append((r.auroc - b, r.dataset, r.severity))
            if drops:
                d = min(drops)
                worst.append((cue, d[0], d[2]))
        L += ["## Reading the curves", "",
              "| cue | worst AUROC drop under processing | at |", "|---|---:|---|"]
        for cue, d, sev in worst:
            L.append(f"| {VIEW_NAME.get(cue, cue.title())} | **{d:+.4f}** | {sev} |")
        L += ["",
              "A cue whose AUROC survives a severity is reliable there whatever its value did; "
              "a cue whose AUROC collapses is not, and the fusion has no way to know that "
              "without this table. That is precisely the reliability field Stage 1 left null, "
              "and it is what Stage 3's M2 and M3 are given that M1 has to infer for itself."]

    if missing:
        L += ["", "## Not yet exported", ""] + [f"- `{m}`" for m in sorted(set(missing))[:20]]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L) + "\n")
    args.out.with_suffix(".json").write_text(R.to_json(orient="records", indent=2))
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
