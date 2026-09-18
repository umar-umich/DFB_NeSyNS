#!/usr/bin/env python3
"""Experiment 6: staged reference vs joint training, on the same folds and the same fusion rules (cached, CPU).

    OMP_NUM_THREADS=4 python analysis/tbiom/joint/exp6_analysis.py [--run joint_seed42]

Three arms, all fused by DS-R (primary) and logistic stacking on matched inputs (comparator), refitted per arm on the
same source-grouped fitting folds and scored on the same selection folds:

  staged            the reference: frozen warm+gate views (per-video mean over EVERY stored frame) + the B-augmented
                    V-JEPA 2 LoRA branch, combined afterwards.
  staged 4-frame    the same frozen views restricted to the four frames the joint model's spatial views see, so the
                    joint arm is not credited (or charged) for a different frame budget. Recomputed from the same
                    cached per-frame opinions.
  joint             `analysis/tbiom/joint/train_joint.py`: every branch adapter and head fine-tuned together from
                    those same staged checkpoints, with per-view EDL, the staged beta-VAE term and an EDL loss on the
                    equal-trust DS fusion.

Reported: development v3.1 (total and by forgery type), the equal-trust DS score (the objective joint training
actually optimised), per-branch scores and leave-one-out contributions for both systems, and the ten test sets.
The staged system remains the reference; this is an ablation and is reported whichever way it comes out.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent / "temporal"))
import exp1_fusion_audit as A  # noqa: E402
import dsr_system as D  # noqa: E402
import finish_line as F  # noqa: E402
from fusion_combos import attach  # noqa: E402

S23, BR, R = Path("logs/tbiom/stage23"), Path("logs/tbiom/temporal/branches"), Path("tbiom/reports")
JOINT = Path("logs/tbiom/joint")
DEV, VIEWS, T_BRANCH = ("VALmix", "VALmixV2", "FFpp_val"), ["T", "APP", "OM"], "vjepa2_lora_B_seed42"
PAIRS = {"T+APP": ["T", "APP"], "T+OM": ["T", "OM"], "APP+OM": ["APP", "OM"]}
T_FRAMES, S_PICK = 16, (0, 5, 10, 15)


def pick_frames(d: pd.DataFrame) -> pd.DataFrame:
    """The four stored frames the joint model's spatial views saw, reproduced from the cached per-frame rows.

    The clip is 16 frames at `linspace(0, n-1, 16)` over the video's sorted stored crops, and the spatial views take
    positions 0, 5, 10, 15 of those 16 -- the same rule `JointClips` applies.
    """
    d = d.sort_values(["v", "key"], kind="stable").reset_index(drop=True)
    keep = []
    for _v, g in d.groupby("v", sort=False):
        n = len(g)
        idx16 = np.linspace(0, n - 1, T_FRAMES).round().astype(int)
        keep.extend(g.index[np.unique(idx16[list(S_PICK)])].tolist())
    return d.loc[sorted(keep)].reset_index(drop=True)


def dev_table(tag, frame_filter=None):
    parts = []
    for name in DEV:
        d = A.load_frames(S23 / f"{tag}_{name}.csv")
        src = d.key.map(lambda k: A.dev_source(k, name))
        d = d[[A.in_scope(s, sub) for s, sub in zip(src, d.subset)]].reset_index(drop=True)
        if frame_filter is not None:
            d = frame_filter(d)
        t = A.video_table(d, [1.0])
        first = d.groupby("v", sort=False).key.first()
        t["source"] = [A.dev_source(first[v], name) for v in t.v]
        parts.append(t)
    return pd.concat(parts, ignore_index=True)


def test_table(tag, t, frame_filter=None):
    d = A.load_frames(S23 / f"{tag}_{t}.csv")
    keep = ~d.subset.str.contains("IP_LAP") & (d.subset != "fake-audio-driven") & ~d.subset.str.startswith("adv-")
    d = d[keep].reset_index(drop=True)
    if frame_filter is not None:
        d = frame_filter(d)
    return A.video_table(d, [1.0])


def staged(frame_filter=None):
    ob = BR / T_BRANCH / "opinions"
    dev = attach(dev_table("warmgate_e19", frame_filter), [ob / f"{n}.csv" for n in DEV])
    tests = {t: attach(test_table("warmgate_sel", t, frame_filter), [ob / f"{t}.csv"]) for t in A.TEST}
    return dev, tests


def joint(run):
    o = {p.stem: pd.read_csv(p) for p in (JOINT / run / "opinions").glob("*.csv")}
    base_dev, base_tests = staged()
    cols = [f"{q}_{v}" for v in VIEWS for q in "pu"]
    dev = pd.concat([o[n] for n in DEV], ignore_index=True).drop_duplicates("v").set_index("v")
    d = base_dev.drop(columns=cols).merge(dev[cols], left_on="v", right_index=True, how="left")
    assert d[cols].notna().all().all(), "development videos missing joint opinions"
    tests = {}
    for t in A.TEST:
        jt = o[t].drop_duplicates("v").set_index("v")
        x = base_tests[t].drop(columns=cols).merge(jt[cols], left_on="v", right_index=True, how="left")
        assert x[cols].notna().all().all(), f"{t}: videos missing joint opinions"
        tests[t] = x
    return d, tests


def ds_equal(t, views=VIEWS):
    return F.dsr(t, views, tuple(1.0 for _ in views))[0]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", default="joint_seed42"); a = ap.parse_args()
    arms = {"staged": staged(), "staged 4-frame": staged(pick_frames), "joint": joint(a.run)}
    base = arms["staged"][0]
    rows = []
    for k, Fd, Sd in D.splits(base):
        row = {"split": k, "arms": {}}
        for name, (dev, tests) in arms.items():
            fd = dev.set_index("v").loc[Fd.v].reset_index(); fd["source"] = Fd.source.to_numpy()
            sd = dev.set_index("v").loc[Sd.v].reset_index(); sd["source"] = Sd.source.to_numpy()
            cl = A.cells(sd)
            rules, params, w = F.rules_for(fd, VIEWS)
            r = {"w": params["w"], "dev": {}, "branches": {}, "loo": {}, "tests": {}}
            for rule, f in rules.items():
                tot, types = A.v31(f(sd), cl); r["dev"][rule] = {"v31": tot, **types}
            tot, types = A.v31(ds_equal(sd), cl); r["dev"]["DS equal"] = {"v31": tot, **types}
            for v in VIEWS:
                r["branches"][v] = A.v31(sd[f"p_{v}"].to_numpy(), cl)[0]
            for pname, views in PAIRS.items():
                pr, _, _ = F.rules_for(fd, views)
                r["loo"][pname] = {rule: A.v31(pr[rule](sd), cl)[0] for rule in ("DS-R", "LR")}
            for t, T in tests.items():
                y = T.y.to_numpy(); auc = lambda s: A.fast_auc(s[y == 0], s[y == 1])
                r["tests"][t] = {**{rule: auc(f(T)) for rule, f in rules.items()},
                                 "DS equal": auc(ds_equal(T)), **{f"branch {v}": auc(T[f"p_{v}"].to_numpy()) for v in VIEWS}}
            row["arms"][name] = r
        rows.append(row)
        print(f"split {k}: " + " | ".join(f"{n} DS-R {row['arms'][n]['dev']['DS-R']['v31']:.4f}" for n in arms), flush=True)
    json.dump(rows, open(R / "EXP_JOINT_ABLATION.json", "w"), indent=1, default=float)
    write(rows, list(arms), a.run)


def write(rows, names, run):
    rec = json.load(open(JOINT / run / "train_record.json"))
    m = lambda n, rule, field: float(np.mean([r["arms"][n]["dev"][rule][field] for r in rows]))
    L = ["# Experiment 6 — joint training vs the staged reference", "",
         "The deployed reference is staged: the Appearance and Off-manifold views are frozen (warm-started from our "
         "DiCoME run), the temporal branch was adapted separately, and DS-R combines them afterwards on development "
         "folds. This fine-tunes all of it together from those same checkpoints — temporal LoRA and head, spatial CLIP "
         "LoRA, β-VAE projector and view heads — with per-view EDL, the staged β-VAE term and an EDL loss on the "
         "**equal-trust** DS fusion (DS-R's weights are fitted on development, so they are kept out of the training "
         "objective). The face prior stays frozen and outside every loss. **The staged system remains the reference; "
         "this is the ablation.**", "",
         f"Budget fixed before launch: {rec['budget']['epochs']} epochs, batch {rec['budget']['batch']}, "
         f"{rec['budget']['spatial_frames_per_clip']} of the clip's 16 frames to the spatial views, adapters "
         f"{rec['budget']['lr_adapters']}, heads {rec['budget']['lr_heads']}, arm-B augmentation, "
         f"{rec['budget']['trainable_adapters'] / 1e6:.2f}M adapter + {rec['budget']['trainable_heads'] / 1e6:.2f}M "
         f"head parameters trainable. Development-only selection chose epoch {rec['best_epoch']} "
         f"(equal-trust v3.1 {rec['best_dev_v31']:.4f}).", "",
         "`staged 4-frame` restricts the frozen views to exactly the four frames the joint model's spatial views see, "
         "recomputed from the same cached per-frame opinions, so the comparison is not a frame-budget difference in "
         "disguise.", "", F.CITE, "",
         "## 1. Development (selection folds, mean of 5 splits)", "",
         "| arm | rule | v3.1 | swap | reenactment | in-the-wild |", "|---|---|---:|---:|---:|---:|"]
    for n in names:
        for rule in ("DS-R", "LR", "DS equal"):
            L.append(f"| {n} | {rule} | {m(n, rule, 'v31'):.4f} | {m(n, rule, 'swap'):.4f} | "
                     f"{m(n, rule, 'reenactment'):.4f} | {m(n, rule, 'in_the_wild'):.4f} |")
    L += ["", "## 2. Per-branch contributions", "",
          "Each view's own development v3.1, and what the triple loses when that view is removed (Δ = triple − pair, "
          "both refitted with the same procedure).", "",
          "| arm | branch v3.1 T | APP | OM | Δ drop T (DS-R) | Δ drop APP | Δ drop OM |", "|---|---:|---:|---:|---:|---:|---:|"]
    for n in names:
        b = {v: float(np.mean([r["arms"][n]["branches"][v] for r in rows])) for v in VIEWS}
        tri = m(n, "DS-R", "v31")
        loo = {p: float(np.mean([r["arms"][n]["loo"][p]["DS-R"] for r in rows])) for p in PAIRS}
        L.append(f"| {n} | {b['T']:.4f} | {b['APP']:.4f} | {b['OM']:.4f} | {tri - loo['APP+OM']:+.4f} | "
                 f"{tri - loo['T+OM']:+.4f} | {tri - loo['T+APP']:+.4f} |")
    L += ["", "## 3. Test sets — video AUROC (read after fitting; mean over the 5 fitted splits)", "",
          "| test | " + " | ".join(f"{n} {rule}" for n in names for rule in ("DS-R", "LR")) + " |",
          "|---|" + "---:|" * (2 * len(names))]
    for t in A.TEST:
        L.append(f"| {t} | " + " | ".join(f"{np.mean([r['arms'][n]['tests'][t][rule] for r in rows]):.4f}"
                                          for n in names for rule in ("DS-R", "LR")) + " |")
    L.append("| **10-set mean** | " + " | ".join(
        f"**{np.mean([np.mean([r['arms'][n]['tests'][t][rule] for r in rows]) for t in A.TEST]):.4f}**"
        for n in names for rule in ("DS-R", "LR")) + " |")
    L += ["", "| per-branch test AUROC (10-set mean) | " + " | ".join(f"{n} {v}" for n in names for v in VIEWS) + " |",
          "|---|" + "---:|" * (3 * len(names)),
          "| | " + " | ".join(f"{np.mean([np.mean([r['arms'][n]['tests'][t][f'branch {v}'] for r in rows]) for t in A.TEST]):.4f}"
                              for n in names for v in VIEWS) + " |", "",
          "Interpretability is not assumed to change: the per-branch columns above are the measurement. One seed per "
          "arm.", "", F.FOOT]
    (R / "EXP_JOINT_ABLATION.md").write_text("\n".join(L) + "\n"); print("wrote", R / "EXP_JOINT_ABLATION.md")


if __name__ == "__main__":
    main()
