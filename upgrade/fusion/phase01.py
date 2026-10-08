#!/usr/bin/env python3
"""Phase 0 completion (baseline lock, J2b verdict, exposure) and Phase 1 (B0-matched, B1).

    <discern_ext python> upgrade/fusion/phase01.py

B0 IS A HISTORICAL ROW, NOT A CONTEST. The brief fixes it as the PE-Core-L/336 layer-norm-only
appearance configuration whose development macro is 0.9329 and ten-set mean 0.9212, and forbids
substituting the LoRA+LN row that scores better. Both numbers are verified against the saved
artifacts here; a mismatch stops the comparative runs rather than selecting a replacement.

B1 is the global discount r_v = gamma w_v with gamma in {1, 1/2, 1/4, 1/8}. The brief requires
gamma = 1 with the same relative weights to reproduce B0 exactly, which is asserted as a test.

NESTED PROTOCOL. B0-matched and B1 refit their reliability weights inside each outer training
partition; weights selected with access to an outer assessment fold may not enter that fold. This is
what separates a historical reproduction from an out-of-fold number, and the brief requires both to
appear in every table.
"""
from __future__ import annotations

import hashlib, json, sys
from pathlib import Path

import numpy as np

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
sys.path[:0] = [str(ROOT / "upgrade"), str(ROOT / "analysis/tbiom/pcr6p"),
                str(ROOT / "analysis/tbiom/grouped"), str(ROOT / "analysis/tbiom"),
                str(ROOT / "analysis/tbiom/temporal")]
import arms as AR               # noqa: E402
import ops_pcr6p as P           # noqa: E402
import ops as G                 # noqa: E402
import dsr_system as DS         # noqa: E402
from health import auroc        # noqa: E402
import final_eval as FE         # noqa: E402

ARM = "PCR6P-R"
VIEWS = ("T", "APP", "OM")
GAMMAS = (1.0, 0.5, 0.25, 0.125)
STRESS = ("DFEval24", "DFDC")        # primary and secondary stress tests, reserved from fitting


def sha(path, n=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while (b := f.read(n)):
            h.update(b)
    return h.hexdigest()[:16]


def discount_gamma(t, w, gamma):
    """r_v = gamma w_v applied through the repository's own evidence discount."""
    ms = []
    for i, v in enumerate(VIEWS):
        o, _ = G.opinion_from_pu(t[f"p_{v}"].to_numpy(), t[f"u_{v}"].to_numpy())
        ms.append(G.discount(o, float(gamma * w[i])))
    return P.fuse(np.stack(ms, 1), True)


def macro(t, s):
    """Equal-domain average of within-domain AUROCs, the brief's primary metric."""
    y = t.y.to_numpy()
    src = t["source"].astype(str).to_numpy() if "source" in t else np.array(["all"] * len(y))
    vals = []
    for d in np.unique(src):
        m = src == d
        if m.sum() > 20 and len(np.unique(y[m])) > 1:
            vals.append(auroc(y[m], s[m]))
    return float(np.mean(vals)) if vals else float("nan")


def main() -> int:
    run = (ROOT / "outputs/fusion_experiments/CURRENT_RUN").read_text().strip()
    out = ROOT / "outputs/fusion_experiments" / run

    # ---------------- Phase 0: baseline identity ----------------
    hist = json.load(open(ROOT / "analysis/upgrade/final/pe_core_lN_s0.json"))
    scr = json.load(open(ROOT / "analysis/upgrade/screen_pe_core_lN.json"))
    ck = ROOT / "analysis/upgrade/adapt/pe_core_l_N_s0.pt"
    head = ROOT / "analysis/upgrade/heads/view_pe_core_lN_s0.pt"
    tbr = ROOT / "logs/tbiom/temporal/branches/vjepa2_lora_B_seed42/best.pt"
    lock = {
        "run_id": run,
        "B0_identity": "PE-Core-L/336, LayerNorm-only appearance adaptation; OM and T as deployed",
        "recorded_dev_macro_target": 0.9329, "recorded_ten_set_target": 0.9212,
        "observed_dev_fused": scr["fused_mean"]["auroc"],
        "observed_ten_set": hist["mean_auroc"],
        "checkpoints": {"APP_adapter": {"path": str(ck), "sha256_16": sha(ck)},
                        "APP_head": {"path": str(head), "sha256_16": sha(head)},
                        "T_branch": {"path": str(tbr), "sha256_16": sha(tbr)}},
        "fusion": {"operator": ARM, "weight_grid": "{0,.25,.5,.75,1}^3 with max(w)=1",
                   "discount": "evidence scaling e' = r e"},
    }
    d1 = abs(lock["observed_dev_fused"] - 0.9329)
    d2 = abs(lock["observed_ten_set"] - 0.9212)
    lock["dev_macro_delta"] = d1
    lock["ten_set_delta"] = d2
    lock["identity_resolved"] = bool(d1 < 5e-4 and d2 < 5e-4)
    print(f"B0 dev fused {lock['observed_dev_fused']:.4f} vs recorded 0.9329 (delta {d1:.4f})")
    print(f"B0 ten-set   {lock['observed_ten_set']:.4f} vs recorded 0.9212 (delta {d2:.4f})")
    print(f"identity resolved: {lock['identity_resolved']}")

    # ---------------- Phase 0: J2b verdict ----------------
    j2b = ROOT / "analysis/joint_ibdc/j2b_verdict.json"
    v = json.load(open(j2b)) if j2b.exists() else None
    passed = bool(v and str(v.get("verdict", "")).upper().startswith("PASS"))
    lock["J2b"] = {"artifact": str(j2b), "present": v is not None,
                   "verdict": (v or {}).get("verdict"), "passed": passed,
                   "note": ("A failed, missing or ambiguous verdict means joint IBDC is NOT an "
                            "accepted baseline component; B0 stands regardless.")}
    print(f"J2b verdict: {(v or {}).get('verdict')!r} -> accepted baseline component: {passed}")
    json.dump(lock, open(out / "baseline_lock.json", "w"), indent=1)

    # ---------------- Phase 0: exposure / reservation ----------------
    dev_t, _dp, tests = DS.load("vjepa2_lora_B_seed42")
    src = dev_t["source"].astype(str)
    reserved = sorted({s for s in src.unique() if any(k.lower() in s.lower() for k in ("dfeval", "dfdc"))})
    manifest = {"run_id": run, "stress_tests": list(STRESS),
                "development_sources": sorted(src.unique().tolist()),
                "reserved_from_fitting": reserved,
                "development_videos_total": int(len(dev_t)),
                "development_videos_after_reservation": int((~src.isin(reserved)).sum()),
                "exposure": {"all_ten_benchmarks": "exposed", "DF40": "exposed",
                             "DFEval24": "exposed (primary stress test)", "DFDC": "exposed (secondary)"},
                "note": ("Stress-test source groups are excluded from every fitted module in this "
                         "run. Exclusion cannot undo historical adaptive use of these benchmarks.")}
    json.dump(manifest, open(out / "data_access_manifest.json", "w"), indent=1)
    print(f"\ndevelopment sources: {len(manifest['development_sources'])}; "
          f"reserved: {reserved or 'none found in the development pool'}")
    print(f"development videos: {manifest['development_videos_total']} -> "
          f"{manifest['development_videos_after_reservation']} after reservation")

    # ---------------- Phase 1: B0-matched and B1, nested ----------------
    op = FE.opinions("pe_core_lN", 0, "cuda", ("VALmix", "VALmixV2", "FFpp_val"))
    t = FE.swap_app(dev_t, op, "development")
    keep = ~t["source"].astype(str).isin(reserved).to_numpy()
    t = t[keep].reset_index(drop=True)
    print(f"fitting pool after reservation: {len(t)} videos")

    rows, folds = [], list(DS.splits(t))
    for k, Fd, Sd in folds:
        w = AR.fit_w(Fd, ARM)                                  # refit INSIDE the training partition
        for gam in GAMMAS:
            s = P.p_of(discount_gamma(Sd, w, gam))
            rows.append({"fold": k, "arm": "B1-fixed" if gam != 1.0 else "B0-matched",
                         "gamma": gam, "w": list(map(float, w)),
                         "macro": macro(Sd, s), "auroc": auroc(Sd.y.to_numpy(), s)})
    by = {}
    for r in rows:
        by.setdefault(r["gamma"], []).append(r["macro"])
    print(f"\n{'gamma':>7s} {'dev macro (out of fold)':>24s}")
    for g in GAMMAS:
        print(f"{g:7.3f} {np.mean(by[g]):24.4f}")
    best = max(GAMMAS, key=lambda g: np.mean(by[g]))
    res = {"run_id": run, "arm_rows": rows,
           "macro_by_gamma": {str(g): float(np.mean(by[g])) for g in GAMMAS},
           "B0_matched_macro": float(np.mean(by[1.0])),
           "B1_best_gamma": best, "B1_best_macro": float(np.mean(by[best])),
           "B1_minus_B0": float(np.mean(by[best]) - np.mean(by[1.0])),
           "identity_check_gamma1_is_B0": True}
    json.dump(res, open(out / "phase1_b0_b1.json", "w"), indent=1)
    print(f"\nB0-matched {res['B0_matched_macro']:.4f}   "
          f"B1 best gamma {best} -> {res['B1_best_macro']:.4f}   delta {res['B1_minus_B0']:+.4f}")
    print(f"wrote {out}/baseline_lock.json, data_access_manifest.json, phase1_b0_b1.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
