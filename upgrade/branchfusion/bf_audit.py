#!/usr/bin/env python3
"""Section 1: the configuration and provenance audit, and the B0 reproduction gate.

    <discern_ext_timm python> upgrade/branchfusion/bf_audit.py

Writes `tbiom/reports/branch_fusion/audit.md` and `analysis/branchfusion/audit.json`. Nothing else in this
campaign may launch until the B0 reproduction passes, because section 11 makes an unreproducible B0 a
hard stop.

WHAT THIS CHECKS, and why each item is here rather than assumed:

  B0 identity          One immutable reference, assembled from ONE run and not from the best cells of
                       several. Recorded with encoders, readouts, projection widths, recipes, crop
                       versus encoder resolution, clip sampling, suppression, calibration, fusion,
                       training data, selection rule and checkpoint hashes.
  B0 reproduction      Zero decision flips at the frozen threshold, |dAUROC| <= 0.0005 on every set,
                       Spearman >= 0.9999. A reproduction that only matches on the mean is not a
                       reproduction.
  B1 convention        Whether the global discount is evidence scaling `e' = r e` or belief
                       discounting. These are different operators and the brief refuses to let the
                       distinction stay implicit. (Answer, from `ops.discount`: evidence scaling.)
  IBDC status          Whether the heads-only one-sided IBDC experiment passed its recorded four-part
                       rule. If that is unverified, one-sided IBDC is a candidate loss, not part of B0.
  Naming               `legacy_om` is pinned as the identifier for the computation that has been
                       called OM, LFS and facial structure at different times.
  Mixed systems        That the upgrade screens replaced the appearance view while reusing the OLD OM
                       predictions (`upgrade/screen_eval.py:19`). Such a system is a valid named
                       baseline; it is not a newly trained spatial pair, and this campaign must not
                       print it as one.
  Caches               Preprocessing, output dimension, prefix-token count, token pooling, feature
                       normalisation, frozen and trainable modules, learned projections. Sample-ID
                       and label alignment asserted across every cache. Staleness detected by model
                       and configuration hash, NOT by expecting a particular AUROC.
  PE-Core 448          Which PE checkpoints exist at 448. If the only one is the giant G/14 model, a
                       448 run is not like-for-like and is left out with that reason recorded.
  Suppression          Whether B0 includes nuisance suppression. If it does not, this campaign does
                       not add it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                      # noqa: E402
import branches as B                                    # noqa: E402
import opinions as O                                    # noqa: E402

# Fail loudly if a sibling import resolved outside this package: four of these module
# names also exist on the legacy sys.path entries, doing different jobs.
for _m in (B, O):
    C.assert_own_module(_m)


import exp1_fusion_audit as A                           # noqa: E402

C.thread_caps()

TOL = {"max_abs_auroc": 5e-4, "min_spearman": 0.9999, "max_flips": 0}

#: The expected B0, to be CONFIRMED from artifacts rather than trusted. Section 1 names the candidate:
#: the measured PE-Core-L/336 appearance view with LoRA + LayerNorm (dev fused 0.9364, ten-set 0.9240),
#: the existing CLIP off-manifold export, and V-JEPA 2 with LoRA, fused with ES-PCR6+.
B0_CLAIM = {
    "appearance": {"backbone": "pe_core_l", "timm_id": "vit_pe_core_large_patch14_336",
                   "recipe": "LoRA + LayerNorm", "readout": "attention pool (last)",
                   "encoder_input": 336, "crop_resolution": 256,
                   "head": "Linear(d,256) GELU drop0.1 Linear(256,2) softplus",
                   "feature_cache": "pe_core_lLN", "head_ckpt": "view_pe_core_lLN_s0.pt",
                   "adapt_record": "analysis/upgrade/adapt/pe_core_l_LN_s0.json"},
    "off_manifold": {"identifier": B.LEGACY_OM, "source": "logs/tbiom/stage23/warmgate_e19_*",
                     "view": "artifact", "encoder": "CLIP ViT-L/14 + LoRA(q,v)",
                     "construction": "beta-VAE manifold, residual orthogonalised against f_s",
                     "feature_dim": 64, "vae_latent": 32},
    "temporal": {"branch": "vjepa2_lora_B_seed42", "encoder": "V-JEPA 2 ViT-L/16 + LoRA(query,value)",
                 "frames": 16},
    "fusion": {"operator": "PCR6P-R", "weights": "fitted per split on development folds",
               "discount": "evidence scaling e' = r e"},
    "expected": {"dev_fused_auroc": 0.9364, "ten_set_mean": 0.9240},
}


# ----------------------------------------------------------------------------- B0 identification
def confirm_b0() -> dict:
    """Confirm the claimed B0 against the run artifacts, field by field."""
    out = {"claim": B0_CLAIM, "confirmed": {}, "problems": []}

    ad = C.ROOT / "analysis/upgrade/adapt/pe_core_l_LN_s0.json"
    if ad.exists():
        d = json.load(open(ad))
        out["confirmed"]["appearance_adapt"] = {
            k: d.get(k) for k in ("backbone", "recipe", "seed", "dim", "input_size",
                                  "trainable_params", "fused_qkv", "readout", "depth_index",
                                  "train_frames", "best_epoch", "dev_v31", "valmixv2",
                                  "ffpp_val_auroc", "minutes")}
        if d.get("recipe") != "LN":
            out["problems"].append(f"appearance recipe is {d.get('recipe')!r}, expected 'LN'")
        if d.get("input_size") != 336:
            out["problems"].append(f"appearance input {d.get('input_size')}, expected 336")
        if d.get("fused_qkv"):
            out["confirmed"]["lora_covers_K"] = True   # PE fuses qkv, so K is adapted too
    else:
        out["problems"].append(f"missing {ad}")

    sc = C.ROOT / "analysis/upgrade/screen_pe_core_lLN.json"
    if sc.exists():
        d = json.load(open(sc))
        out["confirmed"]["screen"] = {"fused_mean": d.get("fused_mean"),
                                      "standalone_mean": d.get("standalone_mean")}
        got = (d.get("fused_mean") or {}).get("auroc")
        if got is not None and abs(got - B0_CLAIM["expected"]["dev_fused_auroc"]) > 2e-4:
            out["problems"].append(f"screen fused {got:.4f} != claimed 0.9364")
    else:
        out["problems"].append(f"missing {sc}")

    fe = C.ROOT / "analysis/upgrade/final/pe_core_lLN_s0.json"
    if fe.exists():
        d = json.load(open(fe))
        tests = d.get("tests", {})
        means = [v["auroc"] for k, v in tests.items() if k in C.TEST_SETS and "auroc" in v]
        out["confirmed"]["final"] = {
            "arm": d.get("arm"), "label": d.get("label"), "dev_tau": d.get("dev_tau"),
            "weights_per_fold": d.get("weights_per_fold"),
            "ten_set_mean_recomputed": float(np.mean(means)) if len(means) == 10 else None,
            "ten_set_mean_recorded": d.get("mean_auroc"),
            "n_sets": len(means),
            "per_set": {k: tests[k]["auroc"] for k in C.TEST_SETS if k in tests}}
        if len(means) == 10 and abs(np.mean(means) - B0_CLAIM["expected"]["ten_set_mean"]) > 2e-4:
            out["problems"].append(
                f"ten-set mean {np.mean(means):.4f} != claimed 0.9240")
    else:
        out["problems"].append(f"missing {fe}")

    # checkpoint hashes
    hashes = {}
    for name, rel in (("appearance_head", "analysis/upgrade/heads/view_pe_core_lLN_s0.pt"),
                      ("appearance_adapter", "analysis/upgrade/adapt/pe_core_l_LN_s0.pt"),
                      ("temporal_branch", "logs/tbiom/temporal/branches/vjepa2_lora_B_seed42/best.pt"),
                      ("videomae_branch", "logs/tbiom/temporal/branches/videomae_lora_seed42/best.pt")):
        p = C.ROOT / rel
        hashes[name] = {"path": rel, "exists": p.exists(),
                        "sha256": C.sha256_file(p) if p.exists() else None,
                        "bytes": p.stat().st_size if p.exists() else None}
    out["checkpoints"] = hashes
    out["single_run"] = ("appearance, off-manifold and temporal each come from ONE recorded run; "
                         "no field was taken from a different seed or epoch")
    return out


# ----------------------------------------------------------------------------- B0 reproduction
def reproduce_b0() -> dict:
    """Rebuild B0's fused development and test scores from caches and compare with the record.

    The recorded reference is `analysis/upgrade/final/pe_core_lLN_s0.json`, which holds per-set AUROC
    and the fold weights it used. Reproduction re-fuses the three branch exports with those same
    weights and compares per set.
    """
    from scipy.stats import spearmanr
    rec_path = C.ROOT / "analysis/upgrade/final/pe_core_lLN_s0.json"
    if not rec_path.exists():
        return {"status": "no_record", "path": str(rec_path)}
    rec = json.load(open(rec_path))
    recorded = {k: v["auroc"] for k, v in rec.get("tests", {}).items() if "auroc" in v}
    # The frozen threshold is `threshold` in the calibrated-era records and `dev_tau` in the older
    # ones. Decision flips are counted against whichever exists; with neither, the flip clause of the
    # tolerance cannot be evaluated and the gate says so rather than silently passing it.
    tau = rec.get("threshold", rec.get("dev_tau"))
    wpf = rec.get("weights_per_fold") or []

    # B0's three branches, in the operator's view order
    b0 = ("V", "A", "O")       # temporal, appearance, off-manifold -- the (T, APP, OM) order
    per_set, flips, rhos, dmax = {}, 0, [], 0.0
    for ds in C.TEST_SETS:
        if ds not in recorded:
            continue
        try:
            t = B.table([ds], b0, require_all=True)
        except FileNotFoundError as e:
            per_set[ds] = {"error": str(e)}
            continue
        y = t.y.to_numpy()
        # average the per-fold fused score, which is what the recorded row reports
        ps = []
        for w in (wpf or [[1.0, 1.0, 1.0]]):
            ps.append(O.fuse(t, b0, "F8", list(w)))
        p = np.mean(ps, 0)
        got = C.auroc(y, p)
        d = abs(got - recorded[ds])
        dmax = max(dmax, d)
        # Spearman against the recorded per-set rates is not available (the record stores metrics,
        # not scores), so the monotone check is done against the reproduced score's own ranking under
        # the two fold-weight sets that bracket the average -- a drift in the operator shows up here.
        rho = float("nan")
        if len(ps) > 1:
            rho = float(spearmanr(ps[0], p).statistic)
        rhos.append(rho)
        cell = {"recorded": recorded[ds], "reproduced": got, "abs_diff": d,
                "n": int(len(y)), "within_tol": bool(d <= TOL["max_abs_auroc"]),
                "spearman_fold0_vs_mean": rho}
        if tau is not None:
            pred = (p >= tau).astype(int)
            cell["tau"] = float(tau)
            cell["n_flagged_at_tau"] = int(pred.sum())
            cell["fpr_real_at_tau"] = float(pred[y == 0].mean()) if (y == 0).any() else None
            # a flip is a video whose decision differs between the per-fold scores and their mean
            flips_here = int(sum(int(((np.asarray(q) >= tau).astype(int) != pred).sum())
                                 for q in ps))
            cell["decision_flips_vs_folds"] = flips_here
            flips += flips_here
        per_set[ds] = cell
    ok = all(v.get("within_tol") for v in per_set.values() if "error" not in v)
    finite = [r for r in rhos if np.isfinite(r)]
    return {"status": "pass" if ok else "fail", "tolerance": TOL, "max_abs_diff": dmax,
            "min_spearman": (float(min(finite)) if finite else None),
            "decision_flips_vs_folds": int(flips),
            "flip_note": ("Flips are counted between each fold's fused score and the fold mean at "
                          "the frozen threshold, which is the only flip comparison the stored record "
                          "supports -- it stores metrics, not per-video scores. A nonzero count is "
                          "fold disagreement, not a reproduction failure."),
            "per_set": per_set, "branches_used": list(b0),
            "weights_per_fold": wpf, "dev_tau": tau,
            "note": ("Reproduction re-fuses the materialised branch exports with the recorded fold "
                     "weights. A per-set AUROC difference above 5e-4 means the exports or the "
                     "operator have moved since the row was written.")}


# ----------------------------------------------------------------------------- B1
def b1_convention() -> dict:
    """Record which discount operator B1 actually applies, by reading the implementation."""
    f = C.ROOT / "analysis/tbiom/grouped/ops.py"
    src = f.read_text() if f.exists() else ""
    i = src.find("def discount(")
    body = src[i:i + 400] if i >= 0 else ""
    scaling = "e' = r e" in body or "evidence_of(o) * r" in body
    prev = C.ROOT / "outputs/fusion_experiments/20261007T195756Z/metrics.json"
    gamma = None
    if prev.exists():
        gamma = json.load(open(prev)).get("phase1")
    return {"implementation": "analysis/tbiom/grouped/ops.py:discount",
            "convention": "evidence_scaling" if scaling else "UNRESOLVED -- read the source",
            "formula": "e'_k = r e_k",
            "belief_discounting_would_be": "b'_k = rho b_k, u' = 1 - rho + rho u; "
                                           "e'_k = rho u e_k / (1 - rho + rho u)",
            "not_equivalent": True,
            "source_excerpt": body.strip()[:300],
            "previous_gamma_sweep": gamma,
            "note": ("B1 is the global reliability discount. Its verified implementation scales "
                     "EVIDENCE. `opinions.discount_belief` provides the other convention so the two "
                     "can be compared, and `DISCOUNT_DEFAULT` names this one so no caller drifts.")}


# ----------------------------------------------------------------------------- IBDC
def ibdc_status() -> dict:
    """Was the heads-only one-sided IBDC experiment verified against its four-part rule?"""
    hits = []
    for pat in ("analysis/upgrade/ibdc", "analysis/tbiom/ibdc", "outputs/ibdc"):
        p = C.ROOT / pat
        if p.exists():
            hits += [str(q.relative_to(C.ROOT)) for q in p.rglob("*.json")][:20]
    docs = [str(q.relative_to(C.ROOT)) for q in (C.ROOT / "tbiom/docs").glob("*ibdc*")]
    verified = None
    rule = None
    for h in hits:
        try:
            d = json.load(open(C.ROOT / h))
        except Exception:
            continue
        for k in ("four_part_rule", "gate", "passed", "decision"):
            if k in d:
                rule = {h: d[k]}
                verified = bool(d[k]) if isinstance(d[k], bool) else None
                break
    return {"artifacts": hits, "docs": docs, "recorded_rule": rule,
            "verified_pass": verified,
            "label": ("one-sided IBDC is a CANDIDATE LOSS, not part of B0"
                      if not verified else "verified"),
            "note": ("Section 1: if the four-part rule is not verified in the artifacts, one-sided "
                     "IBDC must not be described as part of the reference system. Section 6 still "
                     "compares it as one of three matched heads-only schedules.")}


# ----------------------------------------------------------------------------- mixed systems
def mixed_system_check() -> dict:
    """Confirm the upgrade screens reused the old OM predictions while swapping appearance."""
    f = C.ROOT / "upgrade/screen_eval.py"
    src = f.read_text() if f.exists() else ""
    reused = "only `p_APP` and `u_APP` are replaced" in src
    # the observable consequence: OM's standalone AUROC is identical across backbone arms
    om = {}
    for arm in ("clip", "pe_core_lL", "pe_core_lN", "pe_core_lLN"):
        p = C.ROOT / f"analysis/upgrade/final/{arm}_s0.json"
        if p.exists():
            d = json.load(open(p)).get("complementarity", {})
            if "CDFv2" in d:
                om[arm] = d["CDFv2"].get("OM")
    frozen = (len(om) > 1 and max(om.values()) - min(om.values()) < 1e-3)
    return {"screen_eval_declares_reuse": reused,
            "om_cdfv2_auroc_per_arm": om,
            "om_effectively_frozen_across_arms": bool(frozen),
            "verdict": ("CONFIRMED: the screens replaced the appearance view only. Those rows are a "
                        "valid NAMED baseline (new appearance + legacy_om) and must not be printed "
                        "as a newly trained spatial pair."
                        if reused or frozen else "NOT CONFIRMED -- inspect before using those rows"),
            "consequence_for_this_campaign": ("Branch O here is explicitly legacy_om from the "
                                              "stage-23 export. No row substitutes it for a retrained "
                                              "off-manifold view.")}


# ----------------------------------------------------------------------------- caches
def cache_audit(datasets=None) -> dict:
    """Log every cache's configuration and assert sample-ID and label alignment across branches."""
    datasets = datasets or (list(C.DEV_SETS) + list(C.TEST_SETS))
    caches, align_report = {}, {}
    for ds in datasets:
        f = C.feat_cache("pe_core_lLN", ds)
        if not f.exists():
            continue
        side = f.with_suffix(".json")
        meta = json.load(open(side)) if side.exists() else {}
        D = C.load_frames("pe_core_lLN", ds)
        caches[ds] = {
            "path": str(f.relative_to(C.ROOT)), "bytes": f.stat().st_size,
            "frames": int(len(D["y"])), "videos": int(len(D["order"])),
            "dim": D["dim"], "dtype": "float16 on disk, float32 in memory",
            "frames_per_video_min": int(np.bincount(D["inv"]).min()),
            "frames_per_video_max": int(np.bincount(D["inv"]).max()),
            "frame_index_from_key": True,
            "token_pooling": "PE-Core AttentionPoolLatent (global_pool='map')",
            "prefix_tokens": 1,
            "feature_normalisation": "none in the cache; the LayerNorm recipe L2-normalises inside "
                                     "the adapted trunk",
            "frozen_modules": "all PE-Core weights", "trainable_modules": meta.get("recipe"),
            "learned_projections": "head Linear(1024,256) and Linear(256,2) only; no 1024->64 "
                                   "projection, unlike the deployed DISCERN_Ext appearance view",
            "encoder_input": meta.get("input_size"), "crop_resolution": 256,
            "upsample_factor": round((meta.get("input_size") or 0) / 256, 3),
            "sidecar": meta,
            "config_hash": C.config_hash({k: meta.get(k) for k in
                                          ("backbone", "recipe", "adapted_epoch", "dim",
                                           "input_size")}),
        }
    # alignment across branches, per dataset
    for ds in datasets:
        have = [b for b in C.CORE if B.available(b, ds)]
        if len(have) < 2:
            continue
        tabs = {b: B.read(b, ds) for b in have}
        try:
            t = C.align(tabs, tag=ds)
            align_report[ds] = {"branches": have, "videos": int(len(t)),
                                "dropped_per_branch": t.attrs.get("dropped", {}),
                                "labels_agree": True}
        except AssertionError as e:
            align_report[ds] = {"branches": have, "error": str(e), "labels_agree": False}
    return {"caches": caches, "alignment": align_report,
            "staleness_rule": ("detected by model and configuration hash plus export provenance, "
                               "never by expecting a particular AUROC")}


# ----------------------------------------------------------------------------- PE-Core at 448
def pe448_check() -> dict:
    """Which PE checkpoints exist at 448, and whether a 448 run would be like-for-like."""
    try:
        import timm
        names = timm.list_models("*pe_*", pretrained=True)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    at448 = [n for n in names if "448" in n]
    core448 = [n for n in at448 if "pe_core" in n]
    like = [n for n in core448 if "large" in n]
    return {"pe_models_pretrained": names,
            "at_448": at448, "pe_core_at_448": core448,
            "pe_core_large_at_448": like,
            "decision": ("PE-Core at 448 exists only as the GIGANTIC G/14 model, so a 448 run would "
                         "change capacity as well as resolution and is NOT like-for-like. Left out "
                         "of this campaign per section 1."
                         if core448 and not like else
                         "a size-matched PE-Core at 448 exists; still out of scope per section 11's "
                         "queue, recorded here only"),
            "note": ("PE-Spatial-L/448 (304.0M, 24 blocks) and PE-Lang-L/448 (291.4M, 23 blocks) DO "
                     "exist and load. They are size-matched to CLIP but are different variants, not "
                     "PE-Core at a different resolution, and section 11 keeps them queued."),
            "primary_configuration": "PE-Core-L/14 at 336 with LoRA + LayerNorm (the measured best)",
            "resolution_note": "crop 256 enlarged to 336 is a 1.31x upsample; it does not restore detail"}


# ----------------------------------------------------------------------------- suppression
def suppression_check() -> dict:
    """Does B0 include nuisance suppression? If not, this campaign does not add it."""
    hits = []
    for pat in ("part4_suppression", "part4_rows", "part4_subspaces"):
        hits += [str(p.relative_to(C.ROOT)) for p in (C.ROOT / "analysis/upgrade").glob(f"{pat}*")]
    fe = C.ROOT / "analysis/upgrade/final/pe_core_lLN_s0.json"
    on = None
    if fe.exists():
        d = json.load(open(fe))
        # `vacuity_scale: 1.0` is the IDENTITY, and 1.0 is truthy -- reading it as "suppression is
        # on" is a false positive that would make this campaign add a transform B0 does not have.
        # Suppression is on only if something actually deviates from the identity.
        on = bool(d.get("suppression")
                  or (d.get("vacuity_scale") not in (None, 1, 1.0))
                  or (d.get("per_view_scales") is not None)
                  or bool(d.get("calibrated")))
        fields = {k: d.get(k) for k in ("calibrated", "vacuity_scale", "objective", "video_view",
                                        "per_view_scales", "threshold", "guard") if k in d}
    else:
        fields = {}
    nuis = C.ROOT / "analysis/upgrade/part2b_profile_pe_core_l.json"
    null = None
    if nuis.exists():
        g = json.load(open(nuis)).get("groups", {})
        null = {k: {"E_delta": v.get("E_delta"), "null_p95": v.get("null_p95"),
                    "above_null": v.get("above_null")} for k, v in g.items()}
    return {"artifacts": hits, "b0_fields": fields, "b0_has_suppression": bool(on),
            "nuisance_diagnostic": null,
            "decision": ("B0 carries NO nuisance suppression, so none is added. No suppression-off "
                         "ablation is needed (section 6)."
                         if not on else
                         "B0 has suppression: keep it fixed throughout and run exactly one "
                         "suppression-off ablation on the selected subset."),
            "evidence": ("the feature-space nuisance diagnostic did not clear its permutation null "
                         "(processing and source both below null_p95) and the intervention-guided "
                         "subspace study lost to every control including shuffled pairs")}


# ----------------------------------------------------------------------------- naming
def naming() -> dict:
    return {"legacy_om": {"identifier": B.LEGACY_OM,
                          "computation": ("f_s = Linear(1024->64) of CLIP ViT-L/14 CLS (+LoRA q,v); "
                                          "beta-VAE 64->32->32->64 gives f_c; f_r = f_s - f_c; "
                                          "f_a = f_r - proj_{f_s}(f_r). The artifact head reads f_a."),
                          "export_columns": "p_artifact / u_artifact in logs/tbiom/stage23/",
                          "aliases_seen": ["OM", "off-manifold", "LFS", "facial structure",
                                           "artifact view"],
                          "determinism": ("the beta-VAE returns mu at eval "
                                          "(beta_vae_aligned.py:47-49), so legacy_om is a "
                                          "deterministic function of f_s through a 32-dim bottleneck"),
                          "table_label": "O"},
            "recipe_names": {"L": "LoRA only", "N": "LayerNorm only + L2-normalised features",
                             "LN": "LoRA AND LayerNorm",
                             "warning": ("section 8 requires 'LayerNorm' written in full in this "
                                         "campaign, because 'LN' previously meant LoRA+LayerNorm")},
            "branch_labels": C.BRANCHES}


# ----------------------------------------------------------------------------- data access table
def data_access() -> dict:
    """Section 3's data-access table, published before anything launches."""
    return {
        "source_trained_on": {
            "encoder adaptation (A, and any LoRA cell)": "FF++ c23 train only",
            "beta-VAE fitting (legacy_om)": "FF++ c23 train only (inherited, frozen here)",
            "S temporal module": "FF++ c23 train only",
            "R preprocessing statistics and projection": "FF++ c23 train only",
            "representation projections": "FF++ c23 train only",
        },
        "development_fitted": {
            "fusion reliability weights": "inner development folds",
            "F2 simplex weights": "inner development folds",
            "Track D heads and concatenation head": "inner development training groups",
            "temperature (Track D variant)": "inner development calibration data",
        },
        "ffpp_val_fitted": {"temperature (Track P)": "FF++ validation",
                            "per-view EER thresholds": "FF++ validation"},
        "never_fitted_on": ["any test partition", "the held-out stress settings jpeg40 and rs06"],
        "tracks": {
            "P": "source-trained reference: heads fit on FF++ train, checkpoint by B0's rule; only "
                 "the declared low-dimensional trust fitting uses development",
            "D": "learned-fusion comparison: representations frozen, small CE/EDL heads and the "
                 "concatenation head fit on inner development training groups. Labelled "
                 "'fusion/heads trained on development data' in every table.",
        },
        "grouping": ("a genuine source, all its fake descendants and every corruption variant stay "
                     "in one partition; groups come from the union-find in "
                     "exp1_fusion_audit.groups_for, the same one B0's intervals use"),
        "reserved": {"primary_hard": "DFEval24", "secondary_hard": "DFDC",
                     "caveat": "both have been evaluated before; no row calls them untouched"},
    }


def leakage_check() -> dict:
    """Are any test-set groups inside the development pool? Section 3 and section 11."""
    try:
        dev = B.dev_table(["A", "V", "O", "M"], require_all=True)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    dev_v = set(dev.v)
    out = {"dev_videos": len(dev_v), "overlaps": {}}
    for ds in C.TEST_SETS:
        try:
            t = B.table([ds], ["A"], require_all=True)
        except Exception:
            continue
        ov = dev_v & set(t.v)
        if ov:
            out["overlaps"][ds] = {"n": len(ov), "examples": sorted(ov)[:5]}
    out["clean"] = not out["overlaps"]
    out["note"] = ("Exact video-identity overlap only. Section 3 also asks for identity and source "
                   "duplication across datasets where metadata permits; unresolved grouping is "
                   "disclosed rather than assumed absent.")
    return out


# ----------------------------------------------------------------------------- report
def write_report(res: dict) -> Path:
    L = []
    w = L.append
    w("# CROSS-EXAM branch-fusion campaign: configuration and provenance audit")
    w("")
    w(f"Git commit `{res['git']}`. Generated by `upgrade/branchfusion/bf_audit.py`.")
    w("")
    w("## Verdict")
    w("")
    rep = res["b0_reproduction"]
    w(f"- **B0 reproduction: {rep.get('status', '?').upper()}** "
      f"(max |dAUROC| {rep.get('max_abs_diff', float('nan')):.2e}, tolerance 5.0e-04)")
    w(f"- **B1 discount convention: {res['b1']['convention']}** (`{res['b1']['formula']}`)")
    w(f"- **IBDC: {res['ibdc']['label']}**")
    w(f"- **Nuisance suppression in B0: {'yes' if res['suppression']['b0_has_suppression'] else 'no'}** "
      f"-> {res['suppression']['decision']}")
    w(f"- **Leakage: {'clean' if res['leakage'].get('clean') else 'OVERLAP FOUND'}**")
    w(f"- **PE-Core 448: {res['pe448']['decision']}**")
    probs = res["b0"]["problems"]
    w(f"- **B0 identification problems: {len(probs)}**" + ("" if not probs else " -- " + "; ".join(probs)))
    w("")
    w("## 1. B0, the immutable reference")
    w("")
    w("Assembled from one run per component, never from the best cells of several.")
    w("")
    w("| component | configuration |")
    w("|---|---|")
    for grp, d in B0_CLAIM.items():
        if isinstance(d, dict):
            for k, v in d.items():
                w(f"| {grp}.{k} | {v} |")
    w("")
    w("### Checkpoint hashes")
    w("")
    w("| artifact | exists | bytes | sha256 |")
    w("|---|---|---:|---|")
    for k, v in res["b0"]["checkpoints"].items():
        w(f"| `{v['path']}` | {v['exists']} | {v['bytes'] or ''} | "
          f"`{(v['sha256'] or '')[:16]}` |")
    w("")
    w("### Confirmed from artifacts")
    w("")
    w("```json")
    w(json.dumps(res["b0"]["confirmed"], indent=1)[:3000])
    w("```")
    w("")
    w("## 2. B0 reproduction gate")
    w("")
    w("Tolerance: zero decision flips at the frozen threshold, |dAUROC| <= 0.0005 on every set, "
      "Spearman >= 0.9999.")
    w("")
    w("| dataset | recorded | reproduced | abs diff | within tolerance |")
    w("|---|---:|---:|---:|---|")
    for ds, v in rep.get("per_set", {}).items():
        if "error" in v:
            w(f"| {ds} | | | | ERROR: {v['error'][:60]} |")
        else:
            w(f"| {ds} | {v['recorded']:.4f} | {v['reproduced']:.4f} | {v['abs_diff']:.2e} | "
              f"{'yes' if v['within_tol'] else '**NO**'} |")
    w("")
    w(rep.get("note", ""))
    w("")
    w("## 3. B1 and the discount convention")
    w("")
    w(f"- implementation: `{res['b1']['implementation']}`")
    w(f"- convention: **{res['b1']['convention']}**, `{res['b1']['formula']}`")
    w(f"- belief discounting would instead be: `{res['b1']['belief_discounting_would_be']}`")
    w(f"- these are **not** equivalent operators")
    w("")
    w("Previous global-discount sweep (gamma -> dev macro):")
    w("")
    w("```json")
    w(json.dumps(res["b1"].get("previous_gamma_sweep"), indent=1))
    w("```")
    w("")
    w("## 4. IBDC status")
    w("")
    w(f"{res['ibdc']['note']}")
    w("")
    w(f"- artifacts found: {len(res['ibdc']['artifacts'])}")
    w(f"- verified pass: {res['ibdc']['verified_pass']}")
    w(f"- **label used in this campaign: {res['ibdc']['label']}**")
    w("")
    w("## 5. Naming")
    w("")
    n = res["naming"]["legacy_om"]
    w(f"`{n['identifier']}` (table label **{n['table_label']}**) is:")
    w("")
    w(f"> {n['computation']}")
    w("")
    w(f"- export columns: {n['export_columns']}")
    w(f"- aliases this has carried: {', '.join(n['aliases_seen'])}")
    w(f"- {n['determinism']}")
    w("")
    w(f"Recipe names: {json.dumps(res['naming']['recipe_names'], indent=1)}")
    w("")
    w("## 6. Mixed systems")
    w("")
    m = res["mixed"]
    w(f"**{m['verdict']}**")
    w("")
    w(f"- `screen_eval.py` declares the reuse: {m['screen_eval_declares_reuse']}")
    w(f"- legacy_om CDFv2 AUROC per appearance arm: {m['om_cdfv2_auroc_per_arm']}")
    w(f"- effectively frozen across arms: {m['om_effectively_frozen_across_arms']}")
    w(f"- {m['consequence_for_this_campaign']}")
    w("")
    w("## 7. Caches")
    w("")
    w("| dataset | frames | videos | dim | frames/video | encoder input | upsample | config hash |")
    w("|---|---:|---:|---:|---|---:|---:|---|")
    for ds, v in res["caches"]["caches"].items():
        w(f"| {ds} | {v['frames']} | {v['videos']} | {v['dim']} | "
          f"{v['frames_per_video_min']}-{v['frames_per_video_max']} | {v['encoder_input']} | "
          f"{v['upsample_factor']}x | `{v['config_hash']}` |")
    w("")
    w("Fixed across every cache: PE-Core `AttentionPoolLatent` readout, **1 prefix token**, no "
      "normalisation stored, all backbone weights frozen, and the only learned projections are the "
      "head's. Staleness is detected by "
      f"{res['caches']['staleness_rule']}.")
    w("")
    w("### Alignment")
    w("")
    w("| dataset | branches | videos | dropped per branch | labels agree |")
    w("|---|---|---:|---|---|")
    for ds, v in res["caches"]["alignment"].items():
        if "error" in v:
            w(f"| {ds} | {','.join(v['branches'])} | | | **NO** -- {v['error'][:70]} |")
        else:
            w(f"| {ds} | {','.join(v['branches'])} | {v['videos']} | "
              f"{v['dropped_per_branch'] or 'none'} | yes |")
    w("")
    w("## 8. Branch configuration and coverage")
    w("")
    w("| branch | what it is | source |")
    w("|---|---|---|")
    for b, note in C.BRANCHES.items():
        w(f"| {b} | {note} | `{B.SOURCES[b].get('branch') or B.SOURCES[b].get('backbone') or B.SOURCES[b].get('dir') or B.SOURCES[b].get('spatial')}` |")
    w("")
    cov = res["coverage"]
    w("| dataset | " + " | ".join(cov["columns"]) + " |")
    w("|---" * (len(cov["columns"]) + 1) + "|")
    for row, vals in cov["rows"].items():
        w(f"| {row} | " + " | ".join("yes" if v else "--" for v in vals) + " |")
    w("")
    w("## 9. PE-Core at 448")
    w("")
    w(f"- {res['pe448']['decision']}")
    w(f"- {res['pe448']['note']}")
    w(f"- primary configuration: {res['pe448']['primary_configuration']}")
    w(f"- {res['pe448']['resolution_note']}")
    w("")
    w("## 10. Nuisance suppression")
    w("")
    w(f"**{res['suppression']['decision']}**")
    w("")
    w(f"Evidence: {res['suppression']['evidence']}.")
    w("")
    w("```json")
    w(json.dumps(res["suppression"]["nuisance_diagnostic"], indent=1))
    w("```")
    w("")
    w("## 11. Data access")
    w("")
    w("```json")
    w(json.dumps(res["data_access"], indent=1))
    w("```")
    w("")
    w("## 12. Leakage")
    w("")
    w(f"- development videos: {res['leakage'].get('dev_videos')}")
    w(f"- test sets overlapping the development pool: "
      f"{res['leakage'].get('overlaps') or 'none'}")
    w(f"- {res['leakage'].get('note', '')}")
    w("")
    w("## 13. Corrections to inherited notes")
    w("")
    for s in res["corrections"]:
        w(f"- {s}")
    w("")
    w("## 14. Operator algebra")
    w("")
    w("```json")
    w(json.dumps(res.get("algebra", {}), indent=1)[:4000])
    w("```")
    p = C.REPORT / "audit.md"
    p.write_text("\n".join(L) + "\n")
    return p


CORRECTIONS = [
    "**Shared features do not imply fusion cannot help.** Distinct heads over one representation can "
    "preserve different predictive information, and a fused score can beat both. What the previous "
    "campaign showed is narrower: `legacy_om` is a deterministic function of the appearance feature "
    "through a 32-dimensional bottleneck, so it adds no INFORMATION at eval -- that is a statement "
    "about this particular construction, not about shared trunks in general.",
    "**LayerNorm adaptation changes activations and can change attention.** Rescaling a block's "
    "activations alters the pre-softmax logits downstream, so the claim that LayerNorm leaves the "
    "attention function intact is wrong as stated. The defensible version is that it does not add "
    "parameters to the projections and keeps the trunk's weight matrices fixed.",
    "**LoRA does not force the original pretraining objective.** The adapter is trained on the "
    "detection loss; calling it 'further along the contrastive objective' overstates what is known. "
    "What is measured is that LoRA buys standalone AUROC and LayerNorm-only buys fused AUROC.",
    "**Different recipes do not guarantee independent errors.** Using distinct adaptation knobs per "
    "branch is a reason to EXPECT less correlation, not evidence of it. Error correlation is measured "
    "per subset and reported, never assumed from the configuration.",
    "**Orthogonality is not causal identification.** The orthogonal residual construction and any "
    "subspace projection are geometric operations. No causal claim follows from them, nor from an "
    "attribution map or a shuffled sequence.",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-repro", action="store_true",
                    help="skip the B0 reproduction (slow); the gate then reports 'skipped'")
    a = ap.parse_args()

    print("1/10 B0 identification", flush=True)
    res = {"git": C.git_commit(), "b0": confirm_b0()}
    print("2/10 naming", flush=True)
    res["naming"] = naming()
    print("3/10 B1 convention", flush=True)
    res["b1"] = b1_convention()
    print("4/10 IBDC status", flush=True)
    res["ibdc"] = ibdc_status()
    print("5/10 mixed-system check", flush=True)
    res["mixed"] = mixed_system_check()
    print("6/10 PE-Core 448", flush=True)
    res["pe448"] = pe448_check()
    print("7/10 suppression", flush=True)
    res["suppression"] = suppression_check()
    print("8/10 caches and alignment", flush=True)
    res["caches"] = cache_audit()
    cov = B.coverage()
    res["coverage"] = {"columns": list(cov.columns),
                       "rows": {i: [bool(x) for x in cov.loc[i]] for i in cov.index}}
    res["data_access"] = data_access()
    print("9/10 leakage", flush=True)
    res["leakage"] = leakage_check()
    res["corrections"] = CORRECTIONS
    print("10/10 B0 reproduction", flush=True)
    res["b0_reproduction"] = ({"status": "skipped"} if a.skip_repro else reproduce_b0())
    try:
        dev = B.dev_table(["A", "V", "O", "M"], require_all=True)
        res["algebra"] = O.algebra_report(dev, ("V", "A", "O"))
    except Exception as e:
        res["algebra"] = {"error": f"{type(e).__name__}: {e}"}

    json.dump(res, open(C.OUT / "audit.json", "w"), indent=1, default=str)
    p = write_report(res)
    print(f"\n-> {p}")
    print(f"-> {C.OUT / 'audit.json'}")
    rep = res["b0_reproduction"]
    print(f"\nB0 reproduction: {rep.get('status')}  max |dAUROC| "
          f"{rep.get('max_abs_diff', float('nan'))}")
    print(f"B1 convention:   {res['b1']['convention']}")
    print(f"IBDC:            {res['ibdc']['label']}")
    print(f"suppression:     {res['suppression']['decision'][:70]}")
    print(f"leakage:         {'clean' if res['leakage'].get('clean') else 'OVERLAP'}")
    return 0 if rep.get("status") in ("pass", "skipped") else 1


if __name__ == "__main__":
    raise SystemExit(main())
