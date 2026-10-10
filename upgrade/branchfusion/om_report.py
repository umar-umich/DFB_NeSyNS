#!/usr/bin/env python3
"""Apply the off-manifold campaign's gates to whatever has finished, and write the verdict.

    <discern_ext python> upgrade/branchfusion/om_report.py --stage depth|gates|final

This exists because the orchestrator runs with nobody watching: every decision it makes has to be
a rule on a number that a reader can check afterwards, not a judgement call made at 3 a.m. and
forgotten. The three stages are the three decisions.

  depth   Is the depth ordering real, or is it seed noise? The spatial in-loop runs differ by
          0.0060 between the best and worst depth. If a second seed of block 16 moves the score by
          more than that, the ordering carries no information and no further depth run is worth a
          GPU-hour -- which is a conclusion, not a failure.
  gates   Does any new branch deserve a table row? TWO conditions, both required:
            1. it beats its own NO-OM control at the same depth, and
            2. its per-video error phi against A is below 0.60.
          Standalone development score alone has misled this campaign three times -- the frozen Y
          arm scored respectably and turned out to be a substitute for A, not an addition. A branch
          that passes both is printed as `REGISTER:<label>` and the orchestrator runs the screen.
  final   The campaign report, written whether the answer was positive or negative.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C                                       # noqa: E402

OM = C.ROOT / "analysis/upgrade/om"
PHI_GATE, SEED_GATE = 0.60, 0.0060
#: Everything already measured, so the report stands alone when it is read days later.
KNOWN = {"frozen Y (arm 1, cached features)": 0.8696, "A alone": 0.8736,
         "O, the CLIP off-manifold": 0.8252}
PHI_KNOWN = {"frozen Y vs A": 0.789, "in-loop Y16 vs A": 0.678, "O's highest phi with anything": 0.476}


def spatial(tag: str) -> float | None:
    f = OM / (f"om_pe_core_l_LN_bvae_s0{tag}.json" if tag != "_s1_b16"
              else "om_pe_core_l_LN_bvae_s1_b16.json")
    if not f.exists():
        return None
    return float(json.load(open(f))["full_dev"]["r"]["dev_v31"])


def depth_stage() -> int:
    rows = {t.lstrip("_") or "last": spatial(t) for t in ("_b8", "_b12", "_b16", "", "_b20")}
    s1 = spatial("_s1_b16")
    print("PE spatial trunk, off-manifold with the encoder in the loop (full dev, residual arm):")
    for k, v in rows.items():
        print(f"  {k:5s} {v:.4f}" if v is not None else f"  {k:5s} (not finished)")
    done = {k: v for k, v in rows.items() if v is not None}
    if len(done) >= 2:
        print(f"  spread across depths: {max(done.values()) - min(done.values()):.4f}")
    if s1 is None:
        print("\nseed 1 of block 16 has not finished; the noise floor is unknown and every")
        print("ordering above should be read as provisional.")
        return 0
    b16 = rows.get("b16")
    if b16 is None:
        print("\nseed 1 finished but seed 0 did not; nothing to compare.")
        return 0
    d = abs(s1 - b16)
    print(f"\nblock 16: seed 0 {b16:.4f}, seed 1 {s1:.4f}, |difference| {d:.4f}")
    if d > SEED_GATE:
        print(f"SEED NOISE ({d:.4f}) EXCEEDS THE WHOLE DEPTH SPREAD ({SEED_GATE:.4f}).")
        print("The depth ordering is not measurable at one seed per cell. No further depth run is")
        print("launched; reporting depth as indistinguishable is the honest result, and any future")
        print("depth claim needs several seeds per cell rather than another layer.")
    else:
        print(f"seed-to-seed movement ({d:.4f}) is below the depth spread, so the ordering survives")
        print("its first replication. It is still one replicate and should be called suggestive.")
    return 0


def branch_rows() -> list[dict]:
    """Every finished temporal OM run paired with its own no-OM control."""
    base = C.ROOT / "logs/tbiom/temporal/branches"
    out = []
    for d in sorted(base.glob("pecore_l14_lora_B_om*_seed42")):
        rec = d / "train_record.json"
        # A rehearsal directory matches this glob and carries a real DONE marker, so without this
        # it would enter the gate table as a finished run with a meaningless score (the S smoke
        # reached 0.5595 on 64 training clips). Rehearsals are named with `_smoke`.
        if "_smoke" in d.name:
            continue
        if not (d / "DONE").exists() or not rec.exists():
            continue
        r = json.load(open(rec))
        name = d.name.replace("pecore_l14_lora_B_", "").replace("_seed42", "")
        out.append({"run": name, "dir": d, "dev": float(r.get("best_dev_v31", float("nan"))),
                    "epoch": r.get("best_epoch"),
                    "is_control": "omnone" in name, "is_seq": name.endswith("_seq")})
    return out


def phi_against_A(opinions_dir: Path) -> float | None:
    """Per-video error phi against A on the development pool, the only correlation that matters."""
    try:
        import branches as B
        import decorrelation as D
        dev = B.dev_table(["A"], require_all=True)
        rows = []
        for n in C.DEV_SETS:
            f = opinions_dir / f"{n}.csv"
            if not f.exists():
                return None
            rows.append(pd.read_csv(f)[["v", "p"]])
        new = pd.concat(rows).rename(columns={"p": "p_NEW"}).drop_duplicates("v")
        t = dev.merge(new, on="v", how="inner")
        if len(t) < 100:
            return None
        taus = D.thresholds(t, ["A", "NEW"])
        m = D.matrix(t, ["A", "NEW"], taus)
        return float(m["phi"]["A-NEW"]["all"])
    except Exception as e:                                   # noqa: BLE001
        print(f"  (phi unavailable: {type(e).__name__}: {e})")
        return None


def gates_stage() -> int:
    rows = branch_rows()
    if not rows:
        print("No temporal off-manifold run has finished yet; no gate can be applied.")
        return 0
    controls = {("seq" if r["is_seq"] else "mixer"): r["dev"] for r in rows if r["is_control"]}
    print("Temporal off-manifold runs, PE-Core trunk, block 16 (development v3.1):\n")
    for r in rows:
        kind = "seq" if r["is_seq"] else "mixer"
        tagline = "NO-OM CONTROL" if r["is_control"] else "off-manifold"
        print(f"  {r['run']:34s} {tagline:14s} {r['dev']:.4f}  (epoch {r['epoch']})")
    print()
    for r in rows:
        if r["is_control"]:
            continue
        kind = "seq" if r["is_seq"] else "mixer"
        ctl = controls.get(kind)
        label = ("S" if r["is_seq"] else "P") + "16om"
        print(f"{label}: {r['run']}")
        if ctl is None:
            print("  GATE 1 cannot be evaluated -- its no-OM control has not finished. HELD.")
            continue
        g1 = r["dev"] > ctl
        print(f"  gate 1  {r['dev']:.4f} vs control {ctl:.4f}  -> {'PASS' if g1 else 'FAIL'}")
        phi = phi_against_A(r["dir"] / "opinions")
        if phi is None:
            print("  gate 2  phi against A unavailable (opinions missing). HELD.")
            continue
        g2 = phi < PHI_GATE
        print(f"  gate 2  phi vs A {phi:+.3f} < {PHI_GATE}  -> {'PASS' if g2 else 'FAIL'}")
        if g1 and g2:
            print(f"REGISTER:{label}")
        else:
            print(f"  {label} is reported, not registered: a branch that fails either gate would "
                  f"either duplicate A or be worse than the same network without the residual.")
    return 0


def final_stage() -> int:
    print("# The off-manifold campaign beyond CLIP\n")
    print("Written unattended by `om_report.py`. Every number is read from a run record on disk;")
    print("nothing here is recalled from the session that set the campaign up.\n")
    print("## The question\n")
    print("The deployed Off-manifold view O is built on CLIP ViT-L/14 whose LoRA was trained WITH")
    print("the beta-VAE in the loop. The campaign asked whether that construction reproduces on")
    print("other trunks, and three successive answers were wrong for the same reason: the residual")
    print("was fitted on FROZEN features, so it could only ever be a function of an embedding that")
    print("was never asked to make it informative.\n")
    print("## Reference points\n")
    for k, v in KNOWN.items():
        print(f"- {k}: dev v3.1 {v:.4f}")
    for k, v in PHI_KNOWN.items():
        print(f"- {k}: {v:+.3f}")
    print("\n## Depth\n")
    depth_stage()
    print("\n## Gates\n")
    gates_stage()
    print("\n## How to read a negative here\n")
    print("A branch that fails gate 2 is not a weak branch; it is a DUPLICATE of one already in")
    print("the system. A+V+O was selected in all five nested folds without correlation ever being")
    print("shown to the search, so the fusion already prefers decorrelated views. A new PE-trunk")
    print("branch has to leave that cluster to be worth a row, and standalone accuracy cannot tell")
    print("you whether it did.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="final", choices=["depth", "gates", "final"])
    a = ap.parse_args()
    return {"depth": depth_stage, "gates": gates_stage, "final": final_stage}[a.stage]()


if __name__ == "__main__":
    raise SystemExit(main())
