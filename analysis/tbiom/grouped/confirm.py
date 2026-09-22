#!/usr/bin/env python3
"""Stage 3 — the confirmation gate for grouped-R.

    OMP_NUM_THREADS=8 python analysis/tbiom/grouped/confirm.py [--seeds 42,1337]

Three things the brief asks for, in order:
  1. a 95% interval on the DF40 reenactment gain of grouped-R over Dempster-R, from a paired source-group bootstrap
     with 2,000 draws at fixed seeds, resampling reused real videos as one group across generator sets;
  2. the same contrasts under a second detector seed;
  3. the falsification controls, grouped-TA-R and grouped-TO-R, which must lose to grouped-R under both seeds.

A fourth contrast is carried at the same status as the first, because the exposed-suite run showed the fitted
discount zeroes the appearance weight in most folds: grouped-R against the plain two-view T+OM fusion. If that
interval contains zero, the grouped operator is not doing the work, whatever the comparison against Dempster-R says.

Source grouping. A generator set's genuine videos are DF40's shared real pool, so the same real video appears in
every set of its source; grouping on the video name makes those one group across sets, which is what the brief
requires. For FF++-sourced sets the fake's first name token is its real source id, so real and fake share a group.
For Celeb-DF-sourced sets the real pool is numbered in a different namespace from the identity-pair naming of the
fakes, so the real-fake link is not recoverable from the manifest: reals group by video, fakes by identity pair, and
this is a documented limitation rather than an assumed linkage.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parent / "temporal")]
import ops, score as S  # noqa: E402
import exp1_fusion_audit as A  # noqa: E402
import dsr_system as D  # noqa: E402
import df40_analysis as X  # noqa: E402
from fusion_combos import attach  # noqa: E402

R, LOG, N_BOOT = Path("tbiom/reports"), Path("logs/tbiom/df40"), 2000
BR = Path("logs/tbiom/temporal/branches")
CONTRASTS = [("grouped-R", "Dempster-R"), ("grouped-R", "AVG-R"), ("grouped-R", "T+OM"),
             ("grouped-R", "grouped-TA-R"), ("grouped-R", "grouped-TO-R")]
SEEDS = {"42": ("ours", "vjepa2_lora_B_seed42", "vjepa2_lora_B_seed42"),
         "armB42": ("armB42", "vjepa2_lora_B_seed42", "vjepa2_lora_B_seed42"),
         "armB1337": ("armB1337", "vjepa2_lora_B_seed1337", "vjepa2_lora_B_seed1337")}


def groups_of(t):
    """One label per video, shared across generator sets wherever the naming allows it."""
    out = []
    for nm in t.vname:
        head, _, tail = str(nm).partition("__")
        if head.endswith("-real"):
            out.append(f"real:{head}:{tail}")
        elif head.endswith("-ff"):
            out.append(f"ff:{tail.split('_')[0]}")                      # FF++ source id, shared with its real
        else:
            out.append(f"cdfid:{'_'.join(tail.split('_')[:2])}")        # Celeb-DF identity pair
    return np.array(out)


def table(name, spatial_tag, branch):
    d = A.load_frames(X.S23 / f"{spatial_tag}_{name}.csv")
    return attach(A.video_table(d, [1.0]), [BR / branch / "opinions" / f"{name}.csv"])


def run_seed(tag, meta, arms):
    spatial, branch, devbranch = SEEDS[tag]
    dev = D.load(devbranch)[0]
    names = [n for n in meta if (X.S23 / f"{spatial}_{n}.csv").exists()
             and (BR / branch / "opinions" / f"{n}.csv").exists()]
    tabs = {n: table(n, spatial, branch) for n in names}
    ws = {}
    for _k, Fd, _Sd in D.splits(dev):
        clF = A.cells(Fd)
        for a in arms:
            ws.setdefault(a, []).append(S.fit_w(Fd, a, clF))
    scores = {a: {n: [ops.p_of(S.fused(t, a, w)) for w in ws[a]] for n, t in tabs.items()} for a in arms}
    return tabs, scores, ws, names


def auc_weighted(s, y, W):
    """Exact weighted AUROC for every bootstrap draw at once, ties counted at half.

    Integer resampling weights do not change the score ordering, so the sort is done once and each draw is a
    weighted walk over it: for every fake, the weight of the reals strictly below it, plus half the weight of the
    reals tied with it. W is (draws, n)."""
    o = np.argsort(s, kind="stable")
    so, yo, Wo = s[o], y[o], W[:, o]
    real, fake = (yo == 0), (yo == 1)
    wr, wf = Wo * real, Wo * fake
    below = np.cumsum(wr, 1) - wr                                   # real weight strictly before this position
    # ties: positions sharing a score value. group by the unique score blocks
    _, start = np.unique(so, return_index=True)
    blk = np.zeros(len(so), int)
    blk[start[1:]] = 1
    blk = np.cumsum(blk)
    tot_r = np.zeros((W.shape[0], blk[-1] + 1))
    np.add.at(tot_r.T, blk, wr.T)                                   # total real weight per tied block
    first = np.zeros(blk[-1] + 1, int); first[blk[::-1]] = np.arange(len(blk))[::-1]
    before_blk = below[:, first]                                    # real weight before each block
    num = ((before_blk[:, blk] + 0.5 * tot_r[:, blk]) * wf).sum(1)
    den = wr.sum(1) * wf.sum(1)
    return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def boot_cell(tabs, scores, names, a, b, seed):
    """Paired bootstrap of the mean per-set AUROC difference over `names`, one group draw applied to every set."""
    keys = {n: groups_of(tabs[n]) for n in names}
    ug = sorted({g for n in names for g in keys[n]})
    idx = {g: i for i, g in enumerate(ug)}
    pos = {n: np.array([idx[g] for g in keys[n]]) for n in names}
    rng = np.random.default_rng(seed)
    mult = np.stack([np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug)) for _ in range(N_BOOT)])
    acc, cnt = np.zeros(N_BOOT), 0
    for n in names:
        W = mult[:, pos[n]].astype(float)
        y = tabs[n].y.to_numpy()
        per = [auc_weighted(sa, y, W) - auc_weighted(sb, y, W)
               for sa, sb in zip(scores[a][n], scores[b][n])]
        acc += np.nanmean(np.stack(per), 0); cnt += 1
    d = acc / max(cnt, 1)
    return {"delta": float(np.nanmean(d)), "ci": [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="42")
    tags = ap.parse_args().seeds.split(",")
    t0 = time.time()
    meta = json.load(open(LOG / "meta.json"))
    arms = sorted({a for c in CONTRASTS for a in c})
    res = {}
    for tag in tags:
        tabs, scores, ws, names = run_seed(tag, meta, arms)
        ree = [n for n in names if meta[n]["family"] == "reenactment"]
        swp = [n for n in names if meta[n]["family"] == "face swap"]
        print(f"seed {tag}: {len(names)} sets ({len(ree)} reenactment) ({time.time() - t0:.0f}s)", flush=True)
        def point(a, ns):
            if not ns:
                return float("nan")
            per = []
            for n in ns:
                y = tabs[n].y.to_numpy()
                per.append(np.mean([A.fast_auc(sc[y == 0], sc[y == 1]) for sc in scores[a][n]]))
            return float(np.mean(per))
        r = {"n_sets": len(names), "n_reenactment": len(ree), "w": {a: [list(x) for x in ws[a]] for a in arms},
             "point": {a: {"reenactment": point(a, ree), "swap": point(a, swp), "all": point(a, names)} for a in arms},
             "note": "point estimates are mean per-set video AUROC over the five fitted folds",
             "boot": {}}
        for a, b in CONTRASTS:
            for fam, ns in (("reenactment", ree), ("swap", swp), ("all", names)):
                if ns:
                    r["boot"][f"{a} − {b}|{fam}"] = boot_cell(tabs, scores, ns, a, b, 20260922)
            print(f"  {a} − {b} done ({time.time() - t0:.0f}s)", flush=True)
        res[tag] = r
    old = json.load(open(R / "EXP_GROUPED_CONFIRM.json")) if (R / "EXP_GROUPED_CONFIRM.json").exists() else {}
    old.update(res)
    json.dump(old, open(R / "EXP_GROUPED_CONFIRM.json", "w"), indent=1, default=float)
    write(old, time.time() - t0)


def write(res, secs):
    L = ["# Stage 3 — the confirmation gate for grouped-R", "", __doc__.split("\n\n", 2)[2].strip(), "",
         f"Wall clock {secs / 60:.1f} minutes, CPU. {N_BOOT} draws per contrast, fixed seed.", ""]
    for tag, r in res.items():
        who = {"42": "deployed warm+gate spatial views, V-JEPA 2 seed 42",
               "armB42": "SBI arm B spatial seed 42, V-JEPA 2 seed 42",
               "armB1337": "SBI arm B spatial seed 1337, V-JEPA 2 seed 1337"}.get(tag, tag)
        L += [f"## Detector seed `{tag}` — {who}", "",
              f"{r['n_sets']} generator sets, {r['n_reenactment']} of them reenactment.", "",
              "| arm | reenactment | swap | all sets |", "|---|---:|---:|---:|"]
        for a, p in r["point"].items():
            L.append(f"| {a} | {p['reenactment']:.4f} | {p['swap']:.4f} | {p['all']:.4f} |")
        L += ["", "| contrast | family | Δ | 95% CI | excludes zero |", "|---|---|---:|---|---|"]
        for k, b in r["boot"].items():
            c, fam = k.split("|")
            ex = b["ci"][0] > 0 or b["ci"][1] < 0
            L.append(f"| {c} | {fam} | {b['delta']:+.4f} | [{b['ci'][0]:+.4f}, {b['ci'][1]:+.4f}] | "
                     f"{'**yes**' if ex else 'no'} |")
        L.append("")
    L += gate(res)
    (R / "EXP_GROUPED_CONFIRM.md").write_text("\n".join(L) + "\n")
    print("wrote EXP_GROUPED_CONFIRM.md")


def gate(res):
    seeds = [t for t in res if t != "42"] or list(res)
    get = lambda t, c, f: res[t]["boot"].get(f"{c}|{f}")
    L = ["## The gate", "", "| requirement | status |", "|---|---|"]
    rows = []
    for t in res:
        b = get(t, "grouped-R − Dempster-R", "reenactment")
        if b:
            rows.append((f"reenactment interval excludes zero, seed {t}", b["ci"][0] > 0, f"{b['delta']:+.4f} [{b['ci'][0]:+.4f}, {b['ci'][1]:+.4f}]"))
    for t in res:
        for c in ("grouped-TA-R", "grouped-TO-R"):
            b = get(t, f"grouped-R − {c}", "all")
            if b:
                rows.append((f"control {c} loses to grouped-R, seed {t}", b["delta"] > 0 and b["ci"][0] > 0,
                             f"{b['delta']:+.4f} [{b['ci'][0]:+.4f}, {b['ci'][1]:+.4f}]"))
    for t in res:
        b = get(t, "grouped-R − T+OM", "all")
        if b:
            rows.append((f"grouped-R does more than the two-view T+OM fusion, seed {t}", b["ci"][0] > 0,
                         f"{b['delta']:+.4f} [{b['ci'][0]:+.4f}, {b['ci'][1]:+.4f}]"))
    for n, ok, d in rows:
        L.append(f"| {n} | {'pass' if ok else '**fail**'} — {d} |")
    seeds_done = len(res) >= 2
    passed = all(ok for _n, ok, _d in rows) and seeds_done
    L += ["", f"**Verdict: {'grouped-R CONFIRMED' if passed else 'grouped-R NOT confirmed'}.**", ""]
    if not passed:
        L += ["Per the pre-registration the matrix in `EXP_GROUPED_DEVELOPMENT.md` and `EXP_GROUPED_EXPOSED.md` is "
              "therefore a diagnostic, not a comparison supporting a novelty claim, and grouped-R is reported as an "
              "exploratory trade-off. No fusion-operator novelty claim is made.", ""]
    if not seeds_done:
        L += ["The second detector seed is still exporting; this run covers the seeds listed above only.", ""]
    return L


if __name__ == "__main__":
    main()
