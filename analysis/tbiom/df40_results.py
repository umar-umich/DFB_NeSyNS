#!/usr/bin/env python3
"""DF40 results in the reports' standard format, with FS and FR combined.

    python analysis/tbiom/df40_results.py --out tbiom/DF40_RESULTS.md

WHY FS AND FR ARE ONE DATASET. Our six existing OOD sets -- CDFv2, CDFv3, DFD, DFDC, DFDCP,
DFEval24 -- contain face swaps and face reenactments and nothing else. None of them contains
entire-face synthesis. So a DF40 subset built from FS + FR is the apples-to-apples SEVENTH
dataset: same kind of manipulation, 21 generators instead of the handful our current sets
cover. EFS is a different question -- synthesis from no source face -- and stays separate
rather than being averaged into a number it would change for reasons unrelated to
generalisation across manipulations.

HOW IT IS COMBINED, and the trap avoided. The FS and FR exports were run as separate datasets,
and BOTH included the same 178 DF40-cdf-real videos, because DF40 ships one real pool for all
its families. Concatenating the two CSVs naively would count every real video twice, which
would not change AUROC much but would silently halve the effective real:fake ratio and move
FPR_real. Reals are therefore deduplicated on video id.

That duplication is also a free integrity check: the same real video scored by the same frozen
checkpoint in two separate export runs must produce the same opinion. If it does not, the
export is non-deterministic and every number in every report is suspect. The check runs below
and is reported either way.

CONVENTION. `tau` is frozen at the model's EER on ITS OWN FF++ val and the probability is the
DS replay of the three branch opinions -- the same convention as Phase F, the seed summary and
the overlay, so these rows can be pasted beside the existing six datasets without mixing
bases.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from health import dashboard, eer  # noqa: E402
import symbolic_overlay as SO  # noqa: E402
from video_id import video_id  # noqa: E402

S23 = Path("logs/tbiom/stage23")
ARM = "run1auxedl"
BR = SO.BRANCHES
COMBINE = ["DF40_FS", "DF40_FR"]          # one dataset: face manipulation, 21 generators
SEPARATE = ["DF40_EFS", "DF40_FE"]


def load(name: str) -> pd.DataFrame | None:
    f = S23 / f"{ARM}_{name}.csv"
    if not f.is_file():
        return None
    d = pd.read_csv(f)
    need = [f"{k}_{v}" for v in BR for k in ("p", "u")] + ["p_fused", "u_fused", "label"]
    if any(c not in d for c in need):
        return None
    d["v"] = video_id(d["key"])
    agg = {c: (c, "mean") for c in need if c != "label"}
    agg["y"] = ("label", "max")
    return d.groupby("v", as_index=False).agg(**agg)


def combine(parts: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    """Concatenate the families, keeping each real video exactly once."""
    fakes = pd.concat([g[g.y == 1] for g in parts.values()], ignore_index=True)
    reals = pd.concat([g[g.y == 0] for g in parts.values()], ignore_index=True)
    # integrity: the same real video, same checkpoint, two export runs -> same opinion?
    chk = {"real_rows": int(len(reals)), "unique_reals": int(reals["v"].nunique())}
    dup = reals[reals.duplicated("v", keep=False)]
    if len(dup):
        spread = dup.groupby("v")["p_fused"].agg(lambda s: s.max() - s.min())
        chk["max_p_fused_disagreement"] = float(spread.max())
        chk["deterministic"] = bool(spread.max() < 1e-6)
    else:
        chk["deterministic"] = None
    reals = reals.drop_duplicates("v", keep="first")
    return pd.concat([reals, fakes], ignore_index=True), chk


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("tbiom/DF40_RESULTS.md"))
    args = ap.parse_args()

    ff = load("FFpp_val")
    if ff is None:
        print("no FF++ val export for the frozen arm"); return 1
    _, tau = eer(ff["y"].to_numpy(), SO.fuse(ff))

    have = {n: g for n in COMBINE if (g := load(n)) is not None}
    missing = [n for n in COMBINE + SEPARATE if load(n) is None]

    rows, chk = [], {}
    if len(have) == len(COMBINE):
        g, chk = combine(have)
        d = dashboard(g["y"].to_numpy(), SO.fuse(g), tau)
        rows.append(("DF40-FSFR (21 methods)", g, d))
    for n in SEPARATE:
        g = load(n)
        if g is not None:
            d = dashboard(g["y"].to_numpy(), SO.fuse(g), tau)
            rows.append((n.replace("DF40_", "DF40-"), g, d))

    L = ["# DF40 results — the frozen arm on 21 face-manipulation generators", "",
         "Frozen configuration (arm B, three-view, DS fusion, per-branch EDL, seed 42). `tau` "
         f"is its EER on its own FF++ val ({tau:.4f}); probabilities are the DS replay of the "
         "three branch opinions — the same convention as every other table in the writeup.", "",
         "**FS and FR are reported as one dataset.** Our six existing OOD sets contain face "
         "swaps and reenactments only, so FS + FR is the like-for-like seventh dataset, with 21 "
         "generators against the handful the current sets cover. EFS (synthesis from no source "
         "face) is a different question and stays separate.", ""]

    if not rows:
        L += ["**No DF40 exports yet.**", ""] + [f"- missing `{m}`" for m in missing]
        args.out.write_text("\n".join(L) + "\n"); print("\n".join(L)); return 0

    L += ["| dataset | videos | real / fake | AUROC | FPR_real @ τ |",
          "|---|---:|---|---:|---:|"]
    for nm, g, d in rows:
        nr, nf = int((g.y == 0).sum()), int((g.y == 1).sum())
        L.append(f"| {nm} | {len(g):,} | {nr:,} / {nf:,} | **{d['auroc']:.4f}** | "
                 f"{d['fpr_real_at_tau']:.3f} |")

    if chk:
        L += ["", "## Integrity check on the shared real pool", "",
              f"DF40 ships one real pool for all families, so the FS and FR exports both "
              f"contained it: {chk['real_rows']} real rows collapsing to "
              f"{chk['unique_reals']} unique videos. Reals are deduplicated on video id — "
              f"counting them twice would not move AUROC much but would halve the effective "
              f"real:fake ratio and shift FPR_real.", ""]
        if chk.get("deterministic") is not None:
            v = chk.get("max_p_fused_disagreement", 0.0)
            L.append(f"The duplication doubles as a determinism check: the same real video, "
                     f"same checkpoint, two separate export runs. Maximum disagreement in "
                     f"`p_fused` is **{v:.2e}** — "
                     + ("consistent, so the export is deterministic and the two runs agree."
                        if chk["deterministic"] else
                        "**NOT consistent. The export is non-deterministic and every exported "
                        "number in every report needs re-checking before use.**"))

    L += ["", "## Per-generator breakdown", "",
          "Within the combined set, so a family average cannot hide one generator the "
          "framework simply cannot see.", "",
          "| generator | videos | AUROC |", "|---|---:|---:|"]
    for n, gg in have.items():
        d0 = gg[gg.y == 0]
        for meth in sorted(gg[gg.y == 1]["v"].str.extract(r"DF40/([^/]+)/")[0].dropna().unique()):
            sub = gg[(gg.y == 1) & gg["v"].str.contains(f"DF40/{meth}/", regex=False)]
            j = pd.concat([d0, sub], ignore_index=True)
            if j["y"].nunique() < 2:
                continue
            dd = dashboard(j["y"].to_numpy(), SO.fuse(j), tau)
            L.append(f"| {meth} | {len(sub):,} | {dd['auroc']:.4f} |")

    if missing:
        L += ["", "## Not yet exported", ""] + [f"- `{m}`" for m in missing]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L) + "\n")
    args.out.with_suffix(".json").write_text(json.dumps(
        {"tau": float(tau), "check": chk,
         "rows": {nm: d for nm, _, d in rows}}, indent=2, default=float))
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
