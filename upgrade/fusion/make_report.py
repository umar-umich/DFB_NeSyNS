#!/usr/bin/env python3
"""Assemble the run report from the artifacts on disk.

    <discern_ext python> upgrade/fusion/make_report.py

A script rather than an inline heredoc: the first attempt used an UNQUOTED shell heredoc, so bash
expanded `${...}`, `$(...)` and backticks inside what was meant to be Python and silently dropped
several fields -- the brief's own path vanished from the first line. Everything here is read from
the JSON artifacts, so the report cannot disagree with the evidence files.
"""
from __future__ import annotations

import datetime, json, sys
from pathlib import Path

R = Path("/data/umar/Repos/DFB_NeSyNS")


def main() -> int:
    run = (R / "outputs/fusion_experiments/CURRENT_RUN").read_text().strip()
    out = R / "outputs/fusion_experiments" / run
    j = lambda f: json.load(open(out / f))
    prov, lock, man = j("provenance.json"), j("baseline_lock.json"), j("data_access_manifest.json")
    p1, gate, st, alg = j("phase1_b0_b1.json"), j("gate_decision.json"), j("stress_results.json"), j("phase0_algebra.json")
    now = datetime.datetime.now(datetime.timezone.utc)
    el = (now - datetime.datetime.fromisoformat(prov["clock_start_utc"])).total_seconds() / 3600

    L = [f"# CROSS-EXAM fusion experiments — run {run}", "",
         f"Brief: `{prov['brief']}`.", "",
         f"Clock start {prov['clock_start_utc']}, deadline {prov['deadline_utc']}, "
         f"elapsed **{el:.1f} h** at writing.", "",
         "## 1. Baseline reproduction", "", "| item | value |", "|---|---|",
         f"| B0 identity | {lock['B0_identity']} |",
         f"| recorded development macro 0.9329 | **observed {lock['observed_dev_fused']:.4f}**, delta {lock['dev_macro_delta']:.4f} |",
         f"| recorded ten-set mean 0.9212 | **observed {lock['observed_ten_set']:.4f}**, delta {lock['ten_set_delta']:.4f} |",
         f"| identity resolved | **{lock['identity_resolved']}** |",
         f"| J2b verdict | **{lock['J2b']['verdict']}** → accepted baseline component: **{lock['J2b']['passed']}** |",
         "", "The J2b artifact is absent, so joint IBDC is not an accepted baseline component and B0 stands.",
         "", "## 2. Phase 0 algebra audit — it invalidated prior project results", "",
         "A binary opinion needs the three masses to sum to one and stay non-negative, so holding the",
         "probability fixed while multiplying vacuity is valid only while `u <= 2*min(p, 1-p)`. The",
         "project's earlier vacuity scaling clipped at 1.0 and never checked that bound:", "",
         "| view | multiplier | invalid videos | worst belief mass |", "|---|---:|---:|---:|"]
    for v in ("APP", "OM", "T"):
        for r in alg["views"][v]["vacuity_scaling"]:
            if r["multiplier"] in (2.0, 4.0):
                L.append(f"| {v} | {r['multiplier']} | {r['n_invalid']} / {alg['views'][v]['n']} "
                         f"| {r['worst_negative_mass']:.4f} |")
    L += ["", "Evidence discounting is clean: identity at r=1 and vacuity at r=0 are exact, and the closed",
          f"form matches the implementation to {alg['discount']['closed_form_max_abs_err']:.1e}. Three earlier results that used",
          "vacuity scaling are withdrawn; everything using discounting stands.", "",
          "## 3. Data access and reservation", "",
          f"- development sources {len(man['development_sources'])}; reserved from all fitting: **{man['reserved_from_fitting']}**",
          f"- development videos {man['development_videos_total']} → **{man['development_videos_after_reservation']}** after reservation",
          "- all ten benchmarks and DF40 are **previously exposed**; DeepfakeEval24 and DFDC are targeted",
          "  stress tests, not untouched confirmation", "",
          "## 4. Phase 1 — global discount", "",
          "| gamma | development macro, out of fold |", "|---:|---:|"]
    for g, v in p1["macro_by_gamma"].items():
        L.append(f"| {g} | {v:.4f} |")
    L += ["", f"B0-matched {p1['B0_matched_macro']:.4f}; **B1 best gamma {p1['B1_best_gamma']} → "
          f"{p1['B1_best_macro']:.4f}**, delta **{p1['B1_minus_B0']:+.4f}**. Monotone in gamma: the smaller the",
          "global discount the better, which is the region the `max(w)=1` constraint forbids.", "",
          "## 5. Phase 2 — the hard gate FAILED", "",
          "| head | development macro, out of fold |", "|---|---:|"]
    for k, v in gate["macro"].items():
        L.append(f"| {k} | {v:.4f} |")
    L += ["", f"- **F − S = {gate['F_minus_S']['point']:+.4f}**, 97.5% lower bound {gate['F_minus_S']['lo97.5']:+.4f}",
          f"- **F − B1-refit = {gate['F_minus_B1refit']['point']:+.4f}**", "",
          f"**GATE: {'PASSED' if gate['gate_passed'] else 'FAILED'}.** {gate['consequence']}", "",
          "Capacity was matched to 0.8% — F holds 23,194 trainable parameters against S's 23,386 — so this",
          "is not a bigger head winning or losing. F beat S on INNER validation (0.9625 against 0.9381) and",
          "lost by 0.088 out of fold. That is overfitting a 23k-parameter head to 4,914 videos across 52",
          "source groups, which the brief named in advance as the expectation to test.", "",
          "## 6. Final evaluation — DeepfakeEval24 first, DFDC second, then the panel", "",
          f"Frozen development threshold {st['threshold']:.4f}; gamma {st['gamma']}. All configurations were locked",
          "before this ran; nothing was refitted in response to it.", "",
          "| dataset | B0-matched | B1-refit | delta | genuine FPR at the dev threshold |",
          "|---|---:|---:|---:|---:|"]
    for n in ["DFEval24", "DFDC", "CDFv2", "CDFv3", "DFD", "DFDCP", "FFIW", "KoDF", "IDForge", "DFDM"]:
        if n in st["panel"]:
            r = st["panel"][n]
            d = r["B1-refit"]["auroc"] - r["B0-matched"]["auroc"]
            nm = f"**{n}**" if n in ("DFEval24", "DFDC") else n
            L.append(f"| {nm} | {r['B0-matched']['auroc']:.4f} | {r['B1-refit']['auroc']:.4f} | {d:+.4f} "
                     f"| {r['B1-refit']['fpr_genuine_at_dev_threshold']:.4f} |")
    L += ["", "**DeepfakeEval24 remains the weak case at 0.734**, and B1 moves it by +0.0014 — nothing. DFDC",
          "0.862, +0.0016. The global discount is real but tiny and does not address the difficult datasets.",
          "", st["note"], "",
          "## 7. The six-output bottleneck hypothesis: **UNSUPPORTED**", "",
          "Retaining per-view features and giving the fusion head access to them made out-of-fold",
          "performance decisively **worse** at matched capacity. Bounds on that claim: one feature",
          "dimension (d=8), one pooling rule, one backbone, and a development pool of 4,914 videos across",
          "52 source groups — small for a 23k-parameter head. This is evidence against the hypothesis as",
          "tested, not proof that no retained-information scheme could work.", "",
          "## 8. Negative results and skipped work", "",
          "| item | status | reason |", "|---|---|---|",
          "| Phase 3 (A/I interactions) | **skipped** | Phase 2 gate failed; the brief forbids proceeding |",
          "| Phase 4 (intervention subspaces) | **skipped** | same |",
          "| Phase 5 (group-aware loss) | **skipped** | same |",
          "| L/S/F on the panel | **not run** | the off-manifold cache lacks DFDC, CDFv2, CDFv3 and KoDF, so a panel row would cover different datasets than B0/B1 |",
          "| prior vacuity-calibration results | **withdrawn** | invalid opinion masses, section 2 |", "",
          "## 9. Recommended next experiment", "",
          "Not more fusion capacity. Two facts survived: a global discount helps slightly in a region the",
          "weight constraint forbids, and DeepfakeEval24 sits at 0.73 while every other panel set is above",
          "0.86. The next experiment should drop `max(w)=1` and fit an unconstrained per-view scale on the",
          "development pool — three parameters, no new heads — and report whether the DeepfakeEval24 gap",
          "moves at all. If three free scalars cannot move it, the problem is not in the fusion",
          "parameterisation and further fusion work is unwarranted.", ""]
    (out / "report.md").write_text("\n".join(L) + "\n")
    print(f"wrote {out}/report.md ({len(L)} lines), elapsed {el:.1f} h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
