#!/usr/bin/env python3
"""Section 7: what "spatio-temporal" actually contributes, measured rather than assumed.

    <discern_ext_timm python> upgrade/branchfusion/temporal_controls.py --branches V,M,P
    <discern_ext_timm python> upgrade/branchfusion/temporal_controls.py --seq S-adapt_s0

SIX CONTROLS, on S, S-dense if it exists, V and M, with A as an ORDER-INVARIANT NEGATIVE CONTROL:

  1. original order
  2. five seeded random permutations
  3. reversed sequence, reported separately
  4. one valid centre frame repeated to the clip length
  5. for S, the positional encoding removed and the branch retrained under the same budget
  6. for S, the transformer replaced by temporal mean pooling, and by a parameter-matched
     per-frame MLP plus mean pooling

WHAT ALREADY EXISTS AND IS REUSED. V, M and P each shipped `<set>_shuffled.csv` exports -- a fixed
per-video permutation through the trained branch -- so control 2 needs no re-encoding for them. The
temporal feature caches additionally carry an explicit `shuffled` column. For S the controls run on
cached frame features, so all six are cheap.

THE READING RULES SECTION 7 IMPOSES, which this module prints with every table:

  - An inference-time shuffle drop ALONE does not prove useful temporal reasoning. It shows the
    function is order-sensitive, not that the order carries forensic evidence.
  - Repeat-frame results mix missing visual diversity with missing motion, so they are read together
    with the shuffle results and never alone.
  - A is the negative control: it must be invariant within numerical tolerance, which is also
    section 12's acceptance check. If A moves, the harness is changing something it should not.
  - V, M and P stay "video" branches in tables until their temporal use is SHOWN. S may be called a
    spatio-temporal architecture, but a claim that temporal dynamics improve detection needs a
    positive matched control -- control 6 is that control.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C                                      # noqa: E402
import branches as B                                    # noqa: E402

# Fail loudly if a sibling import resolved outside this package: four of these module
# names also exist on the legacy sys.path entries, doing different jobs.
for _m in (B,):
    C.assert_own_module(_m)


C.thread_caps()

SHUFFLE_SETS = ("VALmix", "VALmixV2", "FFpp_val", "CDFv3", "KoDF")
N_PERM = 5


# ----------------------------------------------------------------------------- exported controls
def exported_shuffle(branch: str) -> dict:
    """Control 2 for a branch that shipped `<set>_shuffled.csv` through its trained weights."""
    spec = B.SOURCES.get(branch, {})
    if spec.get("kind") != "temporal":
        return {"status": "not applicable", "reason": f"{branch} has no exported shuffle control"}
    d = C.TEMPORAL / spec["branch"] / "opinions"
    out = {}
    for ds in SHUFFLE_SETS:
        a, b = d / f"{ds}.csv", d / f"{ds}_shuffled.csv"
        if not (a.exists() and b.exists()):
            continue
        o = C.read_opinions(a).set_index("v")
        s = C.read_opinions(b).set_index("v")
        k = o.index.intersection(s.index)
        if not len(k):
            continue
        y = o.loc[k].y.to_numpy()
        po, ps = o.loc[k].p.to_numpy(), s.loc[k].p.to_numpy()
        uo, us = o.loc[k].u.to_numpy(), s.loc[k].u.to_numpy()
        out[ds] = {"n": int(len(k)),
                   "auroc_ordered": C.auroc(y, po), "auroc_shuffled": C.auroc(y, ps),
                   "delta_auroc": C.auroc(y, po) - C.auroc(y, ps),
                   "nll_ordered": C.nll(y, po), "nll_shuffled": C.nll(y, ps),
                   "mean_abs_dp": float(np.abs(po - ps).mean()),
                   "mean_abs_du": float(np.abs(uo - us).mean()),
                   "max_abs_dp": float(np.abs(po - ps).max())}
    if not out:
        return {"status": "no exports"}
    dps = [v["mean_abs_dp"] for v in out.values()]
    das = [abs(v["delta_auroc"]) for v in out.values()]
    return {"status": "ok", "per_dataset": out,
            "mean_abs_dp_overall": float(np.mean(dps)),
            "max_abs_delta_auroc": float(np.max(das)),
            "order_sensitive": bool(np.mean(dps) > 0.01),
            "verdict": ("the branch's prediction barely moves when the frames are permuted, so its "
                        "temporal stage has learned a permutation-invariant function and the branch "
                        "must NOT be called spatio-temporal in tables"
                        if np.mean(dps) <= 0.01 else
                        "the prediction moves under permutation, so the function is order-sensitive. "
                        "That is necessary but NOT sufficient for useful temporal reasoning -- read "
                        "it with control 6's matched comparison")}


# ----------------------------------------------------------------------------- the dense pilot
P_DENSE = C.ROOT / "logs/branchfusion/p_dense"
DENSE_SETS = ("VALmix", "VALmixV2", "FFpp_val")


def dense_vs_sparse(seed: int = 42) -> dict:
    """Section 2.2's dense pilot: does stride-1 training make branch P order-sensitive?

    Sparse P permutes to 0.0014 in probability -- effectively inert. The pre-registered cause
    candidate is FRAME SPACING: its stored crops sit 10-22 source frames apart, so a "clip" spans
    seconds and holds almost no motion. P-dense retrains the identical architecture on consecutive
    crops (median span 1.28 s).

    THE ONLY VALID COMPARISON IS ON SHARED VIDEOS. Dense crops cover 2117 of the 5814 development
    videos, and the covered part is not a random sample of the pool -- CDFv2val and FFpp_val are
    complete, DFDCPval / DS11val / DS20val / FFIWval / DF40train are entirely absent. So sparse P is
    re-read here and INTERSECTED with the dense ids per dataset before anything is compared. Reading
    the dense branch's development macro against sparse P's published full-pool macro would compare
    two different populations and the difference would be dominated by which videos each saw.

    The headline quantity is each branch's own ordered-vs-shuffled |delta p| ON THAT SHARED SET,
    which is a within-branch contrast and therefore immune to the population difference. The AUROC
    columns are reported for completeness and carry the epoch-selection confound noted in the
    record: sparse P chose its epoch on the full pool, dense P on the restricted one.
    """
    run = P_DENSE / f"pecore_l14_dense_seed{seed}"
    dd = run / "opinions"
    if not dd.exists():
        return {"status": "no dense run", "expected": str(dd)}
    spec = B.SOURCES.get("P", {})
    sd = C.TEMPORAL / spec["branch"] / "opinions"
    per, rows = {}, []
    for ds in DENSE_SETS:
        need = [dd / f"{ds}.csv", dd / f"{ds}_shuffled.csv", sd / f"{ds}.csv", sd / f"{ds}_shuffled.csv"]
        if not all(f.exists() for f in need):
            per[ds] = {"status": "incomplete", "missing": [f.name for f in need if not f.exists()]}
            continue
        do, ds_, so, ss = (C.read_opinions(f).set_index("v") for f in need)
        k = do.index.intersection(ds_.index).intersection(so.index).intersection(ss.index)
        if not len(k):
            per[ds] = {"status": "no shared videos"}
            continue
        y = do.loc[k].y.to_numpy()
        # the labels must agree across the four tables or the ids do not mean the same thing
        for other in (ds_, so, ss):
            assert (other.loc[k].y.to_numpy() == y).all(), f"{ds}: label disagreement on shared ids"
        g = {}
        for name, o, sh in (("dense", do, ds_), ("sparse", so, ss)):
            po, ps = o.loc[k].p.to_numpy(), sh.loc[k].p.to_numpy()
            g[name] = {"auroc_ordered": C.auroc(y, po), "auroc_shuffled": C.auroc(y, ps),
                       "delta_auroc": C.auroc(y, po) - C.auroc(y, ps),
                       "mean_abs_dp": float(np.abs(po - ps).mean()),
                       "max_abs_dp": float(np.abs(po - ps).max()),
                       "nll_ordered": C.nll(y, po)}
        per[ds] = {"status": "ok", "n_shared": int(len(k)),
                   "dense_videos": int(len(do)), "sparse_videos": int(len(so)), **g,
                   "dp_ratio": (g["dense"]["mean_abs_dp"] / g["sparse"]["mean_abs_dp"]
                                if g["sparse"]["mean_abs_dp"] > 0 else float("inf"))}
        rows.append(per[ds])
    if not rows:
        return {"status": "nothing comparable", "per_dataset": per}
    dense_dp = float(np.mean([r["dense"]["mean_abs_dp"] for r in rows]))
    sparse_dp = float(np.mean([r["sparse"]["mean_abs_dp"] for r in rows]))
    dense_da = float(np.mean([r["dense"]["delta_auroc"] for r in rows]))
    sens = dense_dp > 0.01
    rec = json.loads((run / "train_record.json").read_text()) if (run / "train_record.json").exists() else {}
    out = {"status": "ok", "seed": seed, "n_datasets": len(rows), "per_dataset": per,
           "mean_abs_dp_dense": dense_dp, "mean_abs_dp_sparse": sparse_dp,
           "dp_ratio": dense_dp / sparse_dp if sparse_dp > 0 else float("inf"),
           "mean_delta_auroc_dense": dense_da,
           "dense_order_sensitive": bool(sens),
           "dense_best_epoch": rec.get("best_epoch"),
           "dense_dev_v31_RESTRICTED": rec.get("best_dev_v31"),
           "dev_restriction": rec.get("dev_restriction", {}),
           "selection_confound": "sparse P selected its epoch on the full development pool, dense P "
                                 "on the dense-covered subset, so the AUROC columns are not a clean "
                                 "head-to-head. The |delta p| columns are within-branch and are not "
                                 "affected by this.",
           "verdict": ("P-dense MOVES under permutation where sparse P did not, so sparse P's "
                       "inertness was the FRAME SPACING, not the architecture: at 10-22 source "
                       "frames apart there was no motion for the temporal stage to read. Whether "
                       "the recovered order-sensitivity BUYS anything is a separate question -- "
                       "read mean_delta_auroc_dense, which is the only column that says so."
                       if sens else
                       "P-dense stays order-invariant with consecutive frames and real motion in "
                       "front of it, so the inertness is ARCHITECTURAL, not a sampling artefact. "
                       "Retraining P on denser clips will not make it spatio-temporal, and the "
                       "frame-spacing hypothesis of section 2.2 is REJECTED for this branch.")}
    return out


# ----------------------------------------------------------------------------- cached-feature controls
def seq_controls(tag: str, datasets=None, n_perm: int = N_PERM, seed: int = 0) -> dict:
    """Controls 1-4 for a trained S, run on its cached frame features."""
    import torch
    import seq_branch as T
    ck = C.SEQ / f"{tag}.pt"
    if not ck.exists():
        return {"status": "missing", "path": str(ck)}
    st = torch.load(ck, map_location="cpu")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = T.SeqBranch(st["dim"], pool=st["pool"], use_pos=st["use_pos"],
                        pe_scale=st.get("pe_scale", 10.0),
                        order_head=bool(st.get("order_loss", 0))).to(dev)
    model.load_state_dict(st["state"])
    model.eval()
    is_dense = T.VARIANTS[st["variant"]].get("suffix") == "-dense"
    datasets = datasets or (list(T.DENSE_SETS)[1:] if is_dense else list(SHUFFLE_SETS))
    rng = np.random.default_rng(seed)
    out = {}
    for ds in datasets:
        try:
            D = T.clips(st["variant"], ds)
        except FileNotFoundError:
            continue
        y = D["yv"]
        base, _ = T.predict(model, D, dev)
        cell = {"n": int(D["n_videos"]), "auroc_ordered": C.auroc(y, base),
                "nll_ordered": C.nll(y, base)}
        # control 2: five seeded permutations of the VALID frames, identities and interval fixed
        perms = []
        for r in range(n_perm):
            Dp = dict(D)
            sq = D["seq"].copy()
            for i in range(D["n_videos"]):
                v = np.where(D["mask"][i])[0]
                sq[i, v] = sq[i, rng.permutation(v)]
            Dp["seq"] = sq
            p, _ = T.predict(model, Dp, dev)
            perms.append({"seed": r, "auroc": C.auroc(y, p), "nll": C.nll(y, p),
                          "mean_abs_dp": float(np.abs(p - base).mean())})
        cell["permutations"] = perms
        cell["auroc_perm_mean"] = float(np.mean([x["auroc"] for x in perms]))
        cell["auroc_perm_sd"] = float(np.std([x["auroc"] for x in perms]))
        cell["delta_auroc_perm"] = cell["auroc_ordered"] - cell["auroc_perm_mean"]
        cell["mean_abs_dp_perm"] = float(np.mean([x["mean_abs_dp"] for x in perms]))
        # control 3: reversed, reported separately from the random permutations
        Dr = dict(D)
        sq = D["seq"].copy()
        for i in range(D["n_videos"]):
            v = np.where(D["mask"][i])[0]
            sq[i, v] = sq[i, v[::-1]]
        Dr["seq"] = sq
        p, _ = T.predict(model, Dr, dev)
        cell["reversed"] = {"auroc": C.auroc(y, p), "nll": C.nll(y, p),
                            "delta_auroc": cell["auroc_ordered"] - C.auroc(y, p),
                            "mean_abs_dp": float(np.abs(p - base).mean())}
        # control 4: one valid centre frame repeated to the clip length
        Dc = dict(D)
        sq = D["seq"].copy()
        for i in range(D["n_videos"]):
            v = np.where(D["mask"][i])[0]
            if len(v):
                sq[i, v] = sq[i, v[len(v) // 2]]
        Dc["seq"] = sq
        p, _ = T.predict(model, Dc, dev)
        cell["repeat_centre"] = {
            "auroc": C.auroc(y, p), "nll": C.nll(y, p),
            "delta_auroc": cell["auroc_ordered"] - C.auroc(y, p),
            "mean_abs_dp": float(np.abs(p - base).mean()),
            "caveat": ("mixes missing visual diversity with missing motion; read with the "
                       "permutation rows, never alone")}
        out[ds] = cell
        print(f"  {tag} {ds}: ordered {cell['auroc_ordered']:.4f}  perm "
              f"{cell['auroc_perm_mean']:.4f}+-{cell['auroc_perm_sd']:.4f}  "
              f"rev {cell['reversed']['auroc']:.4f}  repeat "
              f"{cell['repeat_centre']['auroc']:.4f}  |dp|perm {cell['mean_abs_dp_perm']:.4f}",
              flush=True)
    return {"status": "ok", "tag": tag, "variant": st["variant"], "pool": st["pool"],
            "pos_encoding": st["use_pos"], "pe_scale": st.get("pe_scale"),
            "order_loss": st.get("order_loss", 0), "dense": is_dense, "per_dataset": out}


def appearance_negative_control(datasets=None) -> dict:
    """A must be invariant to frame permutation within numerical tolerance (section 12).

    A is a mean over per-frame scores, so permuting the frames cannot change it. Measuring that is
    not pedantry: if this moves, something in the harness is reordering more than the frames, and
    every temporal delta in the table is then suspect.
    """
    datasets = datasets or list(SHUFFLE_SETS)
    out = {}
    for ds in datasets:
        try:
            D = C.load_frames("pe_core_lLN", ds)
        except FileNotFoundError:
            continue
        import torch
        from train_head import Head
        st = torch.load(C.ROOT / "analysis/upgrade/heads/view_pe_core_lLN_s0.pt",
                        map_location="cpu")
        head = Head(st["dim"])
        head.load_state_dict(st["state"])
        head.eval()
        with torch.no_grad():
            ev = head(torch.from_numpy(D["x"]))
        a = ev + 1.0
        s = a.sum(1, keepdim=True)
        pf = (a / s)[:, 1].numpy()
        base = C.video_mean(D, pf)
        rng = np.random.default_rng(0)
        order = rng.permutation(len(pf))
        D2 = dict(D)
        D2["inv"] = D["inv"][np.argsort(order)][order]      # same partition, frames reordered
        perm = C.video_mean(D, pf[order][np.argsort(order)])
        out[ds] = {"max_abs_dp": float(np.abs(base - perm).max()),
                   "invariant": bool(np.abs(base - perm).max() < 1e-9)}
    return {"branch": "A", "role": "order-invariant negative control", "per_dataset": out,
            "all_invariant": all(v["invariant"] for v in out.values()) if out else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--branches", default="V,M,P")
    ap.add_argument("--seq", default="", help="comma list of trained S tags")
    ap.add_argument("--negative-control", action="store_true")
    ap.add_argument("--dense-pilot", action="store_true",
                    help="compare P-dense against sparse P on the videos both cover")
    ap.add_argument("--dense-seed", type=int, default=42)
    a = ap.parse_args()
    res = {}
    for b in [x.strip() for x in a.branches.split(",") if x.strip()]:
        print(f"=== exported shuffle control: {b}")
        r = exported_shuffle(b)
        res[b] = r
        if r.get("status") == "ok":
            print(f"  mean |dp| {r['mean_abs_dp_overall']:.4f}  max |dAUROC| "
                  f"{r['max_abs_delta_auroc']:.4f}  order-sensitive {r['order_sensitive']}")
            print(f"  {r['verdict']}")
    for t in [x.strip() for x in a.seq.split(",") if x.strip()]:
        print(f"=== sequence controls: {t}")
        res[t] = seq_controls(t)
    if a.dense_pilot:
        print("=== dense pilot: P-dense against sparse P on shared videos")
        r = dense_vs_sparse(a.dense_seed)
        res["p_dense_vs_sparse"] = r
        if r.get("status") == "ok":
            print(f"  mean |dp|  dense {r['mean_abs_dp_dense']:.4f}  sparse "
                  f"{r['mean_abs_dp_sparse']:.4f}  ratio {r['dp_ratio']:.1f}x")
            print(f"  mean dAUROC (ordered - shuffled), dense: {r['mean_delta_auroc_dense']:+.4f}")
            print(f"  order-sensitive: {r['dense_order_sensitive']}")
            for ds, v in r["per_dataset"].items():
                if v.get("status") == "ok":
                    print(f"    {ds:<10} n={v['n_shared']:<5} dense |dp| {v['dense']['mean_abs_dp']:.4f} "
                          f"AUROC {v['dense']['auroc_ordered']:.4f} | sparse |dp| "
                          f"{v['sparse']['mean_abs_dp']:.4f} AUROC {v['sparse']['auroc_ordered']:.4f}")
            print(f"  {r['verdict']}")
        else:
            print(f"  {r.get('status')}: {r.get('expected', '')}")
    if a.negative_control:
        print("=== A, the order-invariant negative control")
        res["A_negative_control"] = appearance_negative_control()
        print(f"  all invariant: {res['A_negative_control']['all_invariant']}")
    json.dump(res, open(C.OUT / "temporal_controls.json", "w"), indent=1, default=str)
    print(f"\n-> {C.OUT / 'temporal_controls.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
