#!/usr/bin/env python3
"""Step 5, second basis — complementarity against P0-DS on DF40-Dev.

    python analysis/tbiom/step5_df40_basis.py --out tbiom/STEP5_DF40.md

VALmix already returned the verdict at 100% join coverage. This is the confirmation on the
generator zoo, where the earlier CLIP-port gate was also run, so the two anchors are compared on
the same probe.

Joining two pipelines again: the anchor comes through DiCoME's h5 (`DF40/<subset>/<prefix>__<id>`)
and the experts through our PNG tree (`.../<method>/<subset>/frames/<id>`). Matched on
(method, id) after stripping the h5 prefix, with labels asserted to agree.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from health import eer  # noqa: E402
from stage1_membership import (MEANINGFUL_RESCUE_MARGIN, MIN_FAMILIES,  # noqa: E402
                               USEFUL_RECOVERY, gate_features, realizable_gate, rescue_harm)
from video_id import video_id  # noqa: E402

DF40 = Path("logs/tbiom/step5_df40")
SCORE = Path("logs/tbiom/score")
STEP2 = Path("logs/tbiom/step2")


def anchor_df40(col: str) -> pd.DataFrame:
    rows = []
    for f in sorted(DF40.glob("p0ds_*.csv")):
        method = f.stem.replace("p0ds_", "")
        d = pd.read_csv(f)
        if col not in d:
            continue
        v = video_id(d["key"])
        g = pd.DataFrame({"v": v, "p": d[col], "u": d.get(col.replace("p_", "u_"), np.nan),
                          "y": d["label"]}).groupby("v", as_index=False).agg(
            p=("p", "mean"), u=("u", "mean"), y=("y", "max"))
        g["method"] = method
        # The SIDE (authentic vs generated) must be part of the join key, taken from the SUBSET
        # name, never from the label. DF40 borrows its authentic halves, so within one method a
        # real `00011` and a generated `00011` both strip to `00011`; joining on the basename
        # alone merged them and produced 4,848 label mismatches. This is the same collision the
        # video_id rule was written for, re-introduced by stripping to a basename for a
        # cross-pipeline join.
        subset = g["v"].str.split("/").str[1]
        g["side"] = np.where(subset.str.contains("-real", case=False), "real", "fake")
        g["base"] = g["v"].str.rsplit("/", n=1).str[-1].str.replace(r"^[A-Za-z0-9\-]+__", "",
                                                                    regex=True)
        rows.append(g[["method", "side", "base", "p", "u", "y"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def expert_df40(path: Path, col: str) -> pd.DataFrame:
    d = pd.read_parquet(path)
    unc = col.replace("p_", "u_")
    v = video_id(d["key"])
    g = pd.DataFrame({"v": v, "method": d["dataset"], "p": d[col],
                      "u": d[unc] if unc in d else np.nan, "y": d["label"]}).groupby(
        ["method", "v"], as_index=False).agg(p=("p", "mean"), u=("u", "mean"), y=("y", "max"))
    # Side from the PATH, matching the anchor side: DF40's authentic halves live under
    # `df40/real/...`, its generated ones under `df40/test/<method>/...`.
    # `/real/` anywhere, not just `/df40/real/`. DF40 keeps the authentic half in two different
    # places: borrowed corpora under `df40/real/<corpus>/...`, but the flat whole-image methods
    # (stargan, starganv2) under `df40/test/<method>/real/...`. Keying on the first alone
    # mislabelled those methods' reals as generated and produced 1,296 label mismatches confined
    # entirely to them. `Celeb-real` does not match, since it has no surrounding slashes.
    g["side"] = np.where(g["v"].str.contains("/real/", regex=False), "real", "fake")
    g["base"] = g["v"].str.rsplit("/", n=1).str[-1]
    return g[["method", "side", "base", "p", "u", "y"]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--anchor-col", default="p_fused")
    ap.add_argument("--out", type=Path, default=Path("tbiom/STEP5_DF40.md"))
    args = ap.parse_args()

    av = anchor_df40(args.anchor_col)
    ff = pd.read_csv(STEP2 / "p0ds_e01_FFpp_val.csv")
    fv = pd.DataFrame({"v": video_id(ff["key"]), "p": ff[args.anchor_col],
                       "y": ff["label"]}).groupby("v", as_index=False).agg(
        p=("p", "mean"), y=("y", "max"))
    _, t_a = eer(fv["y"].to_numpy(), fv["p"].to_numpy())

    results = {}
    for name, path, col, val in [
        ("fsvfm_preserve", SCORE / "fsvfm_preserve_df40dev/profile_epoch_009.parquet", "p_direct",
         SCORE / "fsvfm_preserve_ffppval/profile_epoch_009.parquet"),
        ("fsvfm_ordinary", SCORE / "fsvfm_ordinary_df40dev/profile_epoch_009.parquet", "p_direct",
         SCORE / "fsvfm_ordinary_ffppval/profile_epoch_009.parquet"),
    ]:
        if not path.is_file():
            continue
        ev = expert_df40(path, col)
        m = av.merge(ev, on=["method", "side", "base"], suffixes=("_a", "_e"))
        if len(m) < 500:
            print(f"  {name}: join too thin ({len(m)}) — skipped"); continue
        if not (m["y_a"] == m["y_e"]).all():
            bad = int((m["y_a"] != m["y_e"]).sum())
            raise SystemExit(f"{name}: {bad} label mismatches after the join")
        ed = pd.read_parquet(val)
        edv = pd.DataFrame({"v": video_id(ed["key"]), "p": ed[col],
                            "y": ed["label"]}).groupby("v", as_index=False).agg(
            p=("p", "mean"), y=("y", "max"))
        _, t_e = eer(edv["y"].to_numpy(), edv["p"].to_numpy())

        y = m["y_a"].to_numpy(); pa, pe = m["p_a"].to_numpy(), m["p_e"].to_numpy()
        a_ok = (pa >= t_a).astype(int) == y
        e_ok = (pe >= t_e).astype(int) == y
        pooled = rescue_harm(a_ok, e_ok)
        per_fam = {}
        for fam, g in m.groupby("method"):
            i = m.index.get_indexer(g.index)
            if len(i) > 20:
                per_fam[str(fam)] = rescue_harm(a_ok[i], e_ok[i])
        feats = gate_features(pd.DataFrame({col: pe, "u": m["u_e"].to_numpy()}), col,
                              "u" if m["u_e"].notna().any() else None)
        gate = realizable_gate(pa, pe, feats, y, m["method"].to_numpy(), t_a, t_e)
        margin = (pooled["p_expert_right_given_anchor_wrong"]
                  - pooled["p_expert_wrong_given_anchor_right"])
        fams = [f for f, e in per_fam.items()
                if (e["p_expert_right_given_anchor_wrong"]
                    - e["p_expert_wrong_given_anchor_right"]) > MEANINGFUL_RESCUE_MARGIN]
        rec = gate.get("recovered_fraction")
        results[name] = {"n": int(len(m)), "methods": int(m["method"].nunique()),
                         "pooled": pooled, "gate": gate, "margin": margin,
                         "families_ok": len(fams),
                         "enters": bool(margin > MEANINGFUL_RESCUE_MARGIN
                                        and len(fams) >= MIN_FAMILIES
                                        and rec is not None and rec >= USEFUL_RECOVERY)}
        print(f"  {name:16s} {len(m):6d} videos / {m['method'].nunique()} methods  "
              f"rescue {pooled['p_expert_right_given_anchor_wrong']:.3f} "
              f"harm {pooled['p_expert_wrong_given_anchor_right']:.3f} "
              f"margin {margin:+.3f}  recovered {'n/a' if rec is None else f'{rec:.1%}'}  "
              f"AUROC {gate['anchor_video_auroc']:.4f}->{gate['fused_video_auroc']:.4f}")

    lines = ["# Step 5, second basis — DF40-Dev against P0-DS", "",
             "VALmix returned the verdict at 100% join coverage; this confirms it on the "
             "generator zoo, the same probe the earlier CLIP-port gate used.", "",
             "**34 of 36 methods.** `simswap` and `uniface` (DF40's bare 'unknown' arms) have no "
             "DiCoME config under those names — only `simswap_cdf/_ff` and `uniface_cdf/_ff` "
             "exist — so they are absent and recorded rather than silently dropped.", "",
             "| expert | videos | methods | rescue | harm | margin | ceiling | recovered | AUROC | enters? |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for name, r in results.items():
        p, g = r["pooled"], r["gate"]
        rec = g.get("recovered_fraction")
        lines.append(
            f"| `{name}` | {r['n']} | {r['methods']} | "
            f"{p['p_expert_right_given_anchor_wrong']:.3f} | "
            f"{p['p_expert_wrong_given_anchor_right']:.3f} | {r['margin']:+.3f} | "
            f"{g['ceiling_accuracy']:.3f} | {'n/a' if rec is None else f'{rec:.1%}'} | "
            f"{g['anchor_video_auroc']:.4f} → {g['fused_video_auroc']:.4f} | "
            f"{'**YES**' if r['enters'] else 'no'} |")

    lines += ["", "## Verdict", ""]
    if results and not any(r["enters"] for r in results.values()):
        lines.append(
            "**Confirmed — no expert enters on DF40-Dev either.** That makes FOUR bases in "
            "agreement: DF40-Dev and VALmix against the CLIP port, and VALmix and now DF40-Dev "
            "against the stronger P0-DS anchor. The membership negative is not an artifact of "
            "one probe or one anchor.")
    elif results:
        won = [k for k, r in results.items() if r["enters"]]
        lines.append(f"**{', '.join(won)} passes on DF40-Dev** but not on VALmix. A basis-"
                     f"dependent pass is weaker evidence than a consistent one and must be "
                     f"reconciled before anything is built on it.")
    else:
        lines.append("TODO(run) — no expert produced a result.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    args.out.with_suffix(".json").write_text(json.dumps(results, indent=2, default=float))
    print("\n".join(lines[-4:]))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
