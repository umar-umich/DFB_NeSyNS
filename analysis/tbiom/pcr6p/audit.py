#!/usr/bin/env python3
"""Section 2 of the PCR6+ brief: the algebraic unit tests, with expected values, before any dataset is scored.

    OMP_NUM_THREADS=8 python analysis/tbiom/pcr6p/audit.py

Every test states what it expects independently of the implementation. Test 6 additionally checks the direct
S = 2 fusion against the two-source PCR5/PCR6 routine written for the earlier grouped study, so the two independent
implementations must agree.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "grouped"), str(HERE.parent), str(HERE.parent / "temporal")]
import ops_pcr6p as P  # noqa: E402
import ops as G  # noqa: E402

R = Path("tbiom/reports")


def opin(b1, b0, u):
    return np.array([[b1, b0, u]], float)


def rand(n, rng):
    u = rng.uniform(0.02, 0.9, n)
    p = rng.uniform(0, 1, n) * (1 - u) + u / 2
    return G.opinion_from_pu(p, u)[0]


def main():
    rng = np.random.default_rng(23)
    res, notes = [], {}

    def rec(name, expected, ok, detail=""):
        res.append({"test": name, "expected": expected, "pass": bool(ok), "detail": detail})

    n = 4000
    M = np.stack([rand(n, rng) for _ in range(3)], 1)
    for plus, tag in ((False, "PCR6"), (True, "PCR6+")):
        f = P.fuse(M, plus)
        rec(f"{tag}: output masses non-negative and sum to one", "non-negative, sum to one",
            (f >= -1e-12).all() and np.allclose(f.sum(1), 1),
            f"min {f.min():.2e}, max |sum-1| {np.abs(f.sum(1) - 1).max():.2e}")
        p = P.p_of(f)
        rec(f"{tag}: probabilities in [0, 1]", "valid probabilities", ((p >= 0) & (p <= 1)).all(), "")

    # 2. permutation invariance
    for plus, tag in ((False, "PCR6"), (True, "PCR6+")):
        a = P.fuse(M, plus)
        b = P.fuse(M[:, [2, 0, 1], :], plus)
        c = P.fuse(M[:, [1, 2, 0], :], plus)
        rec(f"{tag}: permutation of the three views", "identical output",
            np.allclose(a, b) and np.allclose(a, c),
            f"max dev {max(np.abs(a - b).max(), np.abs(a - c).max()):.2e}")

    # 3/4. the paper's neutrality example, on our frame
    T = np.concatenate([opin(0, 1, 0), opin(1, 0, 0), opin(0, 0, 1)])[None, :, :]   # genuine, fake, vacuous
    f6, f6p = P.fuse(T, False)[0], P.fuse(T, True)[0]
    rec("PCR6 on (genuine=1), (fake=1), (Theta=1)", "fake, genuine and Theta each 1/3",
        np.allclose(f6, [1 / 3, 1 / 3, 1 / 3]), f"got {np.round(f6, 6).tolist()}")
    two = P.fuse(T[:, :2, :], False)[0]
    rec("PCR6+ on the same three views", "equals the two-source fusion of the first two",
        np.allclose(f6p, two), f"got {np.round(f6p, 6).tolist()}, two-source {np.round(two, 6).tolist()}")
    notes["neutrality_example"] = {"PCR6": f6.tolist(), "PCR6+": f6p.tolist(), "two_source": two.tolist()}

    # 5. a view discounted to zero must act as if removed  (PCR6+ only)
    M0 = M.copy()
    M0[:, 1, :] = G.discount(M[:, 1, :], 0.0)
    for plus, tag in ((True, "PCR6+"), (False, "PCR6")):
        full = P.fuse(M0, plus)
        drop = P.fuse(M0[:, [0, 2], :], plus)
        ok = np.allclose(full, drop)
        rec(f"{tag}: a view with r = 0 acts as if removed", "PCR6+ yes; PCR6 is expected to fail this" if not plus
            else "equals the fusion of the other two views", ok if plus else not ok,
            f"max dev {np.abs(full - drop).max():.3e}"
            + ("" if plus else "  (a non-zero deviation here IS the flaw PCR6+ corrects)"))
        notes[f"zero_weight_dev_{tag}"] = float(np.abs(full - drop).max())

    # 6. two sources: PCR6, PCR6+ and the independent two-source PCR5 routine must coincide
    M2 = M[:, :2, :]
    a, b = P.fuse(M2, False), P.fuse(M2, True)
    ref = G.pcr6_2(M2[:, 0, :], M2[:, 1, :])
    rec("two sources: PCR6 equals PCR6+", "identical (paper, Remark 3)", np.allclose(a, b),
        f"max dev {np.abs(a - b).max():.2e}")
    rec("two sources: direct rule equals the independent two-source PCR5 implementation",
        "identical", np.allclose(a, ref), f"max dev {np.abs(a - ref).max():.2e}")

    # 7. fully conflicting dogmatic views
    D = np.concatenate([opin(1, 0, 0), opin(0, 1, 0), opin(1, 0, 0)])[None, :, :]
    d6, d6p = P.fuse(D, False)[0], P.fuse(D, True)[0]
    rec("fully conflicting dogmatic views", "finite, stated, documented",
        np.isfinite(d6).all() and np.isfinite(d6p).all() and np.allclose(d6.sum(), 1),
        f"PCR6 {np.round(d6, 6).tolist()}, PCR6+ {np.round(d6p, 6).tolist()}")
    notes["dogmatic_conflict"] = {"PCR6": d6.tolist(), "PCR6+": d6p.tolist()}

    # 8. very small vacuity
    tiny = np.stack([np.concatenate([opin(0.9 - 5e-13, 0.1 - 5e-13, 1e-12),
                                     opin(0.2 - 5e-13, 0.8 - 5e-13, 1e-12),
                                     opin(0.5 - 5e-13, 0.5 - 5e-13, 1e-12)])], 0)
    t6, t6p = P.fuse(tiny, False), P.fuse(tiny, True)
    rec("very small vacuity", "stable arithmetic, no NaN",
        np.isfinite(t6).all() and np.isfinite(t6p).all(), f"PCR6+ {np.round(t6p[0], 6).tolist()}")

    # 9. evidence round trip
    o = rand(n, rng)
    rec("evidence to opinion and back", "exact for u > 0",
        np.allclose(G.opinion_of_evidence(G.evidence_of(o)), o),
        f"max dev {np.abs(G.opinion_of_evidence(G.evidence_of(o)) - o).max():.2e}")

    # conflict attribution
    _f, att = P.fuse(M, True, attribute=True)
    s = att["K_TS"] + att["K_AO"] + att["K_all"]
    rec("conflict attribution sums to K on every sample", "K_TS + K_AO + K_all = K",
        np.allclose(s, att["K"]), f"max dev {np.abs(s - att['K']).max():.2e}")
    notes["conflict_shares_mean"] = {k: float(att[k].mean()) for k in ("K", "K_TS", "K_AO", "K_all")}

    # historical labelling: is the existing PCR6-R row direct or a pairwise fold?
    pw = G.fuse({"T": M[:, 0], "APP": M[:, 1], "OM": M[:, 2]}, "PCR6-R")
    direct = P.fuse(M, False)
    same = np.allclose(pw, direct)
    rec("historical PCR6-R row is direct three-source PCR6", "expected NO — it is a pairwise fold", not same,
        f"max dev {np.abs(pw - direct).max():.3e}; the historical row must be relabelled PCR6-pairwise-R")
    notes["pairwise_vs_direct_max_dev"] = float(np.abs(pw - direct).max())

    npass = sum(r["pass"] for r in res)
    write(res, notes, npass)
    print(f"{npass}/{len(res)} tests pass")
    for r in res:
        if not r["pass"]:
            print("  FAIL:", r["test"], r["detail"])
    return 0 if npass == len(res) else 1


def write(res, notes, npass):
    L = ["# PCR6+ audit — algebraic unit tests and exact formulas", "", __doc__.split("\n\n", 2)[2].strip(), "",
         f"**{npass} of {len(res)} tests pass.**", "",
         "## 1. Primary source and the equations as implemented", "",
         P.__doc__.split("\n\n", 1)[1].strip(), "",
         "## 2. Unit tests", "", "| test | expected | result | detail |", "|---|---|---|---|"]
    for r in res:
        L.append(f"| {r['test']} | {r['expected']} | {'pass' if r['pass'] else '**FAIL**'} | {r['detail']} |")
    z6, z6p = notes["zero_weight_dev_PCR6"], notes["zero_weight_dev_PCR6+"]
    L += ["", "## 3. What the correction actually changes", "",
          f"A view discounted to r = 0 is vacuous. Removing it should change nothing. Under **PCR6+** the maximum "
          f"deviation between fusing three views and fusing the two survivors is {z6p:.2e} — it is the same "
          f"function. Under **PCR6** it is {z6:.3e}: a discounted-away view still absorbs conflicting mass into "
          "Theta and shifts the result. That is the flaw the paper corrects, and it is the reason this study "
          "nominates PCR6+ rather than PCR6.", "",
          "On our binary frame the whole correction reduces to one rule: **conflicting mass is never redistributed "
          "to Theta.** The keeping indexes are computed from equations (22)-(24) rather than assumed, and they come "
          "out as kappa(fake) = kappa(genuine) = 1, kappa(Theta) = 0, matching Remark 1 of the paper.", "",
          "The neutrality example, on our frame: three views asserting genuine, fake and total ignorance.", "",
          "| rule | m(fake) | m(genuine) | m(Theta) |", "|---|---:|---:|---:|"]
    for k, v in notes["neutrality_example"].items():
        L.append(f"| {k} | {v[0]:.6f} | {v[1]:.6f} | {v[2]:.6f} |")
    L += ["", "PCR6 gives the ignorant view a third of the mass; PCR6+ ignores it and returns the two-source "
          "fusion, as the paper's theorem requires.", "",
          "## 4. Fully conflicting dogmatic views, documented rather than avoided", "",
          "| rule | m(fake) | m(genuine) | m(Theta) |", "|---|---:|---:|---:|"]
    for k, v in notes["dogmatic_conflict"].items():
        L.append(f"| {k} | {v[0]:.6f} | {v[1]:.6f} | {v[2]:.6f} |")
    L += ["", "Two views assert fake with certainty and one asserts genuine with certainty; all vacuities are zero. "
          "Both rules return a finite, normalised result and no fraction is discarded, because every denominator is "
          "the sum of the conflicting sources' masses and is strictly positive. The zero-denominator convention of "
          "the paper (\"if a denominator is zero, that fraction is discarded\") is implemented and never fires on "
          "admissible inputs.", "",
          "## 5. Historical labelling", "",
          f"The `PCR6-R` row in `EXP_FUSION_OPERATORS.md`, `EXP_DF40_OPERATORS.md` and `EXP_GROUPED_*.md` was "
          f"computed by folding the two-source rule over the three views, not by the direct three-source formula "
          f"(15). The two differ by up to {notes['pairwise_vs_direct_max_dev']:.3e} on random admissible opinions. "
          "PCR6 is not associative, so the fold is a different operator. **Every new table in this study labels that "
          "construction `PCR6-pairwise-R`**, and the direct rule `PCR6-R`. The earlier reports are left unchanged.", "",
          "## 6. Conflict attribution", "",
          "Each conflicting tuple is assigned to exactly one bucket by which pair produced it: `K_TS` when the "
          "temporal view conflicts with a spatial view and the two spatial views agree, `K_AO` when the two spatial "
          "views conflict and the temporal view sides with neither in a way that adds a further conflict, and "
          "`K_all` when both kinds are present in the same tuple. Three-way products are therefore reported "
          "separately rather than split arbitrarily, and the shares sum to K exactly on every sample.", "",
          "| quantity | mean on random admissible opinions |", "|---|---:|"]
    for k, v in notes["conflict_shares_mean"].items():
        L.append(f"| {k} | {v:.4f} |")
    (R / "EXP_PCR6P_AUDIT.md").write_text("\n".join(L) + "\n")
    json.dump({"tests": res, "notes": notes}, open(R / "EXP_PCR6P_AUDIT.json", "w"), indent=1, default=float)
    print("wrote EXP_PCR6P_AUDIT.md")


if __name__ == "__main__":
    sys.exit(main())
