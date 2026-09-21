#!/usr/bin/env python3
"""Stage 2 (executable brief): the 2x2 temporal comparison and the stopping rule, on cached opinions (CPU).

    OMP_NUM_THREADS=4 python analysis/tbiom/temporal/stage2_analysis.py

Rules frozen in tbiom/docs/stage2-temporal-preregistration.md. Each configuration's opinions replace the temporal
view of the reference triple (Appearance and Off-manifold unchanged); DS-R and the logistic comparator are refitted on
the FITTING part; the candidate is chosen on the SELECTION part; the stopping rule is assessed only on the
CONFIRMATION part, against the reference exported through the same code path.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import exp1_fusion_audit as A  # noqa: E402
import dsr_system as D  # noqa: E402
import finish_line as F  # noqa: E402
from health import eer  # noqa: E402
import stage2_train as S  # noqa: E402

R = Path("tbiom/reports"); VIEWS = ["T", "APP", "OM"]; PROC = ("jpeg40", "rs06"); N_BOOT = 1000
CONFIGS = {"sparse × pooling (reference)": "reference_sparse_pool", "sparse × GRU head": "sparse_gru_seed42",
           "dense × pooling": "dense_pool_seed42", "dense × GRU head": "dense_gru_seed42",
           "dense-subsampled × pooling (diagnostic)": "dsub_pool_seed42",
           "sparse × Mamba head (only if triggered)": "sparse_mamba_seed42",
           "dense × Mamba head (only if triggered)": "dense_mamba_seed42"}
TRIGGER = 0.005        # pre-registration amendment of 2026-09-21: GRU over pooling, same sampling, selection part
MARGIN, ROBUST, REEN, COMPUTE = 0.005, -0.005, -0.005, 2.0


def with_T(tab, op):
    o = op.drop_duplicates("v").set_index("v"); t = tab.copy()
    idx = o.index.get_indexer(t.v); assert (idx >= 0).all(), f"{(idx < 0).sum()} videos without temporal opinions"
    t["p_T"], t["u_T"] = o.p.to_numpy()[idx], o.u.to_numpy()[idx]
    return t


def main():
    part = S.partition()
    dev, dproc, _ = D.load("vjepa2_lora_B_seed42")
    keep = lambda t: t[t.v.isin(part)].reset_index(drop=True)
    dev = keep(dev); dproc = {c: keep(t) for c, t in dproc.items()}
    parts = {p: dev[dev.v.map(part) == p].reset_index(drop=True) for p in ("fit", "select", "confirm")}
    rng = np.random.default_rng(2026); res = {}
    for name, run in CONFIGS.items():
        d = S.OUT / run
        if not (d / "DONE").exists():
            res[name] = None; continue
        ops = {c: pd.read_csv(d / "opinions" / f"dev2_{c}.csv") for c in ("clean", *PROC)}
        tabs = {"clean": with_T(dev, ops["clean"]), **{c: with_T(dproc[c], ops[c]) for c in PROC}}
        fit = tabs["clean"][tabs["clean"].v.map(part) == "fit"].reset_index(drop=True)
        rules, params, w = F.rules_for(fit, VIEWS)
        ff = np.where(fit.source == "FFpp_val")[0]
        r = {"w": params["w"], "scores": {}, "rec": json.load(open(d / "train_record.json"))}
        for p in ("select", "confirm"):
            for c, t in tabs.items():
                sub = t[t.v.map(part) == p].reset_index(drop=True)
                for rule, f in rules.items():
                    s = f(sub); tot, ty = A.v31(s, A.cells(sub))
                    r[f"{p}|{c}|{rule}"] = {"v31": tot, **ty}
                    if c == "clean":
                        tau = F.tau_of(f(fit), fit); pred = (s >= tau).astype(int); y = sub.y.to_numpy()
                        r[f"{p}|{c}|{rule}"].update(fpr=float((pred[y == 0] == 1).mean()), tpr=float((pred[y == 1] == 1).mean()))
                    r["scores"][f"{p}|{c}|{rule}"] = s
            r[f"{p}|branch T"] = A.v31(tabs["clean"][tabs["clean"].v.map(part) == p].p_T.to_numpy(),
                                       A.cells(tabs["clean"][tabs["clean"].v.map(part) == p].reset_index(drop=True)))[0]
        diag = {}
        for order in ("ordered", "reversed", "local", "shuffled"):
            f = d / "opinions" / (f"dev2_clean_{order}.csv" if order != "ordered" else "dev2_clean.csv")
            if f.exists():
                o = pd.read_csv(f).drop_duplicates("v").set_index("v").loc[dev.v]
                diag[order] = (o.p.to_numpy(), o.y.to_numpy())
        r["diag"] = diag
        res[name] = r
    ref = res["sparse × pooling (reference)"]
    new = [n for n in CONFIGS if n != "sparse × pooling (reference)" and "diagnostic" not in n and res.get(n)]
    pairs = {"sparse": ("sparse × pooling (reference)", "sparse × GRU head", "sparse × Mamba head (only if triggered)"),
             "dense": ("dense × pooling", "dense × GRU head", "dense × Mamba head (only if triggered)")}
    trig = {}
    for smp, (pn, gn, mn) in pairs.items():
        if res.get(pn) and res.get(gn):
            d = res[gn]["select|clean|DS-R"]["v31"] - res[pn]["select|clean|DS-R"]["v31"]
            trig[smp] = {"delta": d, "fires": d >= TRIGGER, "mamba_run": bool(res.get(mn))}
    cand = max(new, key=lambda n: res[n]["select|clean|DS-R"]["v31"]) if new else None

    L = ["# Stage 2 — bounded temporal experiment", "",
         "Pre-registration: `tbiom/docs/stage2-temporal-preregistration.md`. Each configuration replaces the temporal "
         "view of the reference triple; Appearance and Off-manifold are the frozen reference views. Fusion refitted on the "
         "fitting part; candidate chosen on the selection part; the stopping rule assessed on the confirmation part only. "
         f"Partition sizes: " + ", ".join(f"{p} {len(t)}" for p, t in parts.items()) + " videos.", "",
         "## 1. The 2 × 2 (plus the same-window diagnostic), fused DS-R subset-v3.1", "",
         "| configuration | branch alone (select) | fused, select clean | select JPEG | select resize | fused, confirm clean | confirm reenactment | export ms/video |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for n in CONFIGS:
        r = res.get(n)
        if r is None:
            L.append(f"| {n} | TODO(run) | TODO(run) | TODO(run) | TODO(run) | TODO(run) | TODO(run) | TODO(run) |"); continue
        L.append(f"| {n}{' **(candidate)**' if n == cand else ''} | {r['select|branch T']:.4f} | {r['select|clean|DS-R']['v31']:.4f} | "
                 f"{r['select|jpeg40|DS-R']['v31']:.4f} | {r['select|rs06|DS-R']['v31']:.4f} | {r['confirm|clean|DS-R']['v31']:.4f} | "
                 f"{r['confirm|clean|DS-R']['reenactment']:.4f} | {r['rec'].get('export_ms_per_video', float('nan')):.1f} |")
    if cand:
        c, rf = res[cand], ref
        conf = parts["confirm"]; g = A.groups_for(conf); ug = np.unique(g); pos = {k: np.where(g == k)[0] for k in ug}; cl = A.cells(conf)
        sc, sr = c["scores"]["confirm|clean|DS-R"], rf["scores"]["confirm|clean|DS-R"]
        boot = []
        for _ in range(N_BOOT):
            wts = np.zeros(len(conf), dtype=int)
            for k in rng.choice(ug, len(ug), replace=True):
                wts[pos[k]] += 1
            boot.append(A.v31(sc, cl, wts)[0] - A.v31(sr, cl, wts)[0])
        d = c["confirm|clean|DS-R"]["v31"] - rf["confirm|clean|DS-R"]["v31"]; lo, hi = np.percentile(boot, [2.5, 97.5])
        rob = np.mean([c[f"confirm|{p}|DS-R"]["v31"] - rf[f"confirm|{p}|DS-R"]["v31"] for p in PROC])
        ren = c["confirm|clean|DS-R"]["reenactment"] - rf["confirm|clean|DS-R"]["reenactment"]
        comp = c["rec"].get("export_ms_per_video", np.nan) / rf["rec"].get("export_ms_per_video", np.nan)
        ok = d >= MARGIN and lo > 0 and rob >= ROBUST and ren >= REEN and comp <= COMPUTE
        L += ["", "## 2. The stopping rule, on the confirmation part", "",
              f"Candidate (best fused DS-R on the selection part): **{cand}**.", "",
              f"1. Margin: fused Δ **{d:+.4f}**, 95% CI [{lo:+.4f}, {hi:+.4f}] (need ≥ +{MARGIN} and CI excluding zero) → {'met' if d >= MARGIN and lo > 0 else 'NOT met'}.",
              f"2. Robustness: mean Δ over JPEG q40 and resize 0.60 **{rob:+.4f}** (need ≥ {ROBUST}) → {'met' if rob >= ROBUST else 'NOT met'}.",
              f"3. Reenactment: Δ **{ren:+.4f}** (need ≥ {REEN}) → {'met' if ren >= REEN else 'NOT met'}.",
              f"4. Compute: {comp:.2f}× the reference's export time per video (need ≤ {COMPUTE}×); the backbone sees 16 frames in "
              "every configuration, so model FLOPs are equal up to the small GRU; dense windows additionally need decoding "
              "and face cropping of 32 consecutive source frames, reported separately.", "",
              f"**Verdict: {'CONTINUE temporal-architecture development' if ok else 'STOP temporal-architecture development for this paper'}.** "
              + ("" if ok else "Reported as insufficient demonstrated benefit under the tested configurations within the "
                                "allocated budget — not as evidence that temporal information is absent."),
              f"FPR / TPR at the fitting part's τ, confirmation clean: candidate {c['confirm|clean|DS-R']['fpr']:.3f} / "
              f"{c['confirm|clean|DS-R']['tpr']:.3f}, reference {rf['confirm|clean|DS-R']['fpr']:.3f} / {rf['confirm|clean|DS-R']['tpr']:.3f}.", ""]
    L += ["## Mamba trigger (pre-registered 2026-09-21, before any fused number was computed)", "",
          "A small temporal head is useful if a GRU configuration beats pooling at the same sampling by ≥ +0.005 fused "
          "DS-R subset-v3.1 on the selection part; only then is a Mamba head trained, for that sampling.", ""]
    for smp, t in trig.items():
        L.append(f"- {smp}: GRU − pooling = {t['delta']:+.4f} → trigger **{'fires' if t['fires'] else 'does not fire'}**"
                 + (f"; Mamba head {'trained and in the candidate set' if t['mamba_run'] else 'not trained yet'}" if t["fires"] else "; no Mamba head trained") + ".")
    L += ["", "## 3. Order diagnostics (branch alone, whole subset) — diagnostics, never gates", "",
          "| configuration | ordered AUROC | reversed | locally permuted | shuffled | mean |Δp| reversed | mean |Δp| shuffled |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for n in CONFIGS:
        r = res.get(n)
        if not r or "ordered" not in r["diag"]:
            continue
        auc = {o: A.fast_auc(p[y == 0], p[y == 1]) for o, (p, y) in r["diag"].items()}
        dp = lambda o: float(np.mean(np.abs(r["diag"][o][0] - r["diag"]["ordered"][0]))) if o in r["diag"] else float("nan")
        L.append(f"| {n} | {auc['ordered']:.4f} | {auc.get('reversed', float('nan')):.4f} | {auc.get('local', float('nan')):.4f} | "
                 f"{auc.get('shuffled', float('nan')):.4f} | {dp('reversed'):.4f} | {dp('shuffled'):.4f} |")
    L += ["", "Frames for every configuration are cut by the same dlib / 81-landmark aligned-crop function; the dense "
          "window is 32 consecutive source frames at the video's midpoint. One seed per configuration."]
    (R / "EXP_TEMPORAL_SAMPLING.md").write_text("\n".join(L) + "\n"); print("wrote", R / "EXP_TEMPORAL_SAMPLING.md")
    json.dump({"_mamba_trigger": trig} | {n: (None if r is None else {k: v for k, v in r.items() if k not in ("scores", "diag")}) for n, r in res.items()},
              open(R / "EXP_TEMPORAL_SAMPLING.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
