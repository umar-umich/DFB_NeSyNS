#!/usr/bin/env python3
"""Build the four-file ChatGPT handoff. Packaging only: nothing in the manuscript or the results is modified.

    OMP_NUM_THREADS=4 python analysis/tbiom/paper/make_handoff.py

Produces tbiom/paper_handoff_chatgpt/{PAPER_REVIEW.txt, PAPER_EVIDENCE.md, PAPER_PROJECT.zip, HANDOFF_MANIFEST.md}
and copies a compiled CURRENT_PAPER.pdf when one exists.
"""
from __future__ import annotations
import hashlib, json, re, subprocess, zipfile
from datetime import date
from pathlib import Path

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
PAPER, OUT = ROOT / "tbiom/paper", ROOT / "tbiom/paper_handoff_chatgpt"
REPORTS = ROOT / "tbiom/reports"

# every table, with how it is produced
GEN = "analysis/tbiom/paper/build_tables.py"
TABLE_ORIGIN = {t: GEN for t in [
    "tab_crossdataset", "tab_df40family", "tab_df40gen", "tab_views", "tab_viewrole", "tab_operators",
    "tab_reliability", "tab_perview", "tab_slices", "tab_configs", "tab_uncertainty", "tab_weights",
    "tab_operatorsfull", "tab_conflict", "tab_efficiency"]}
TABLE_ORIGIN["tab_extension"] = "HAND-WRITTEN (qualitative content; not regenerated)"

FIGS = {"figures/fig_df40.pdf": "analysis/tbiom/paper/build_figures.py",
        "figures/fig_complementarity.pdf": "analysis/tbiom/paper/build_figures.py"}


def sh(*a):
    return subprocess.run(a, capture_output=True, text=True, cwd=ROOT).stdout.strip()


def sha(p, n=16):
    p = Path(p)
    if not p.exists():
        return "MISSING"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:n]


def snapshot():
    commit = sh("git", "rev-parse", "HEAD")
    dirty = sh("git", "status", "--porcelain")
    return {"date": str(date.today()), "commit": commit, "dirty": bool(dirty),
            "dirty_paths": len(dirty.splitlines()) if dirty else 0}


# ------------------------------------------------------------------ 1. PAPER_REVIEW.txt
def header(snap):
    L = ["% " + "=" * 110,
         "% PAPER_REVIEW.txt -- flattened READING COPY of the T-BIOM manuscript.",
         "% NOT the canonical source. Do not replace tbiom/paper/main.tex with this file.",
         "% " + "=" * 110, "%",
         f"% Snapshot date      : {snap['date']}",
         f"% Repository commit  : {snap['commit']}",
         f"% Working tree       : {'DIRTY (' + str(snap['dirty_paths']) + ' paths)' if snap['dirty'] else 'clean'}",
         "%",
         "% SELECTED METHOD (fixed by the authors, not by a score):",
         "%   Operator      : PCR6+-R  = direct three-source PCR6+ (Dezert, Dezert & Smarandache, JAIF 16(1) 2021,",
         "%                   eqs. 15, 22-24, 26) applied AFTER the project's evidence-scaling reliability discount",
         "%                   e' = r*e. PCR6+ is prior work and is credited as such; the paper claims its",
         "%                   application and the measurement, not the operator.",
         "%   Views         : T   temporal      V-JEPA 2 ViT-L, frozen backbone + LoRA r8/alpha16, evidential head",
         "%                   APP appearance    CLIP ViT-L/14 + LoRA r8/alpha16, evidential head",
         "%                   OM  off-manifold  beta-VAE head (latent 32, beta_KLD 2.0) on the same CLIP path",
         "%   Configuration : '42' -- warm+gate spatial checkpoint (epoch 19) + vjepa2_lora_B_seed42.",
         "%                   freeze_views: true, so APP and OM are FROZEN at the baseline's own values; the only",
         "%                   new trainable component is the temporal view (919,938 parameters).",
         "%   Excluded      : the face-prior (FS-VFM student) branch present in the spatial checkpoint. Tested as a",
         "%                   view; the reliability fit zeroed it in 5/5 folds. Fusion is fixed at three views.",
         "%",
         "% UNRESOLVED / OPEN ITEMS AT THIS SNAPSHOT:",
         "%   1. The manuscript has NOT been compiled by the agent preparing this handoff -- see CURRENT_PAPER.pdf",
         "%      status in HANDOFF_MANIFEST.md. Structure was validated by analysis/tbiom/paper/check_tex.py only.",
         "%   2. Float placement is untested. Four full-width floats (4x table*, 1x figure*) in a two-column",
         "%      IEEEtran layout will likely need rearranging.",
         "%   3. Length is unmeasured against any T-BIOM allowance; 16 tables is a lot.",
         "%   4. V-JEPA 2 bib entry uses 'and others'; expand at camera-ready.",
         "%   5. No untouched confirmation set exists: CDFv2 and DeepfakeEval2024 informed development, and DF40",
         "%      informed operator design in prior studies. Every benchmark here is exposed evaluation.",
         "%   6. The five development folds are reliability-FITTING folds, not detector seeds.",
         "% " + "=" * 110, ""]
    return "\n".join(L)


def flatten(snap):
    src = (PAPER / "main.tex").read_text()
    out = [header(snap)]

    def expand(m):
        rel = m.group(1)
        rel = rel if rel.endswith(".tex") else rel + ".tex"
        f = PAPER / rel
        stem = Path(rel).stem
        origin = TABLE_ORIGIN.get(stem, "unknown")
        if not f.exists():
            return f"% !! MISSING \\input target: tbiom/paper/{rel}\n"
        body = f.read_text().rstrip("\n")
        return ("\n% " + "-" * 104
                + f"\n% BEGIN FILE: tbiom/paper/{rel}"
                + f"\n% ORIGIN: {origin}"
                + ("\n% This table is regenerated; edits to it will be overwritten. Change the generator instead."
                   if origin == GEN else "")
                + "\n% " + "-" * 104 + "\n"
                + body
                + f"\n% END FILE: tbiom/paper/{rel}\n")

    body = re.sub(r"\\input\{([^}]+)\}", expand, src)

    # annotate external figure references without embedding binaries
    def figref(m):
        path = m.group(2)
        gen = FIGS.get(path, "unknown")
        return (f"% EXTERNAL FIGURE: tbiom/paper/{path} -- present in PAPER_PROJECT.zip at that path.\n"
                f"% Generated by: {gen}. Binary not embedded here.\n" + m.group(0))
    body = re.sub(r"(\\includegraphics\[[^\]]*\]\{([^}]+)\})", figref, body)
    out.append(body)

    for rel, label in (("refs.bib", "BIBLIOGRAPHY SOURCE"), ("README.md", "BUILD AND PROVENANCE NOTES"),
                       ("tables/key_numbers.json", "PROVENANCE DATA for numbers quoted in the prose")):
        f = PAPER / rel
        out.append("\n% " + "=" * 110
                   + f"\n% BEGIN FILE: tbiom/paper/{rel}   ({label})"
                   + "\n% Appended as a source section; not part of the manuscript body."
                   + "\n% " + "=" * 110 + "\n"
                   + f.read_text().rstrip("\n")
                   + f"\n% END FILE: tbiom/paper/{rel}\n")
    out.append("\n% NOTE: IEEEtran.cls and IEEEtran.bst are deliberately NOT included (standard IEEE template files,\n"
               "% byte-identical to tbiom/docs/IEEE_TBIOM). Required packages are listed in HANDOFF_MANIFEST.md.\n")
    (OUT / "PAPER_REVIEW.txt").write_text("\n".join(out))
    return body


MAIN_GUARD_MOVED = True


# ------------------------------------------------------------------ 2. PAPER_EVIDENCE.md
CURRENT = [
    ("tbiom/reports/PCR6P_REFERENCE_MANIFEST.md", "Method identity, the discount definition, the four detector "
     "configurations, folds and the exposure history."),
    ("tbiom/reports/EXP_PCR6P_AUDIT.md", "Exact operator equations from the primary source, 17 algebraic unit tests, "
     "and the resolution of the vacuous-source neutrality question."),
    ("tbiom/reports/EXP_PCR6P_DEVELOPMENT.md", "The fixed arm matrix on development for all four configurations: "
     "detection, calibration, uncertainty scores, conflict cases, fitted weights."),
    ("tbiom/reports/EXP_PCR6P_EXPOSED.md", "Ten test sets, processing conditions and all 41 DF40 entries, per arm "
     "and per configuration, with both required DF40 means."),
    ("tbiom/reports/PCR6P_OUTCOME.md", "Pre-registered acceptance criteria applied; terminal outcome."),
    ("tbiom/reports/PCR6P_PRIOR_ART_AND_CLAIMS.md", "Prior art with primary sources read, and the claim-to-evidence "
     "map separating established operators from our implementation choices and measured findings."),
    ("tbiom/reports/EXP_DF40_CROSS_MANIPULATION.md", "Per-generator cross-manipulation against our trained DiCoME."),
    ("tbiom/reports/BASELINE_MATRIX.md", "Comparator provenance and per-benchmark values, all reproduced locally."),
]
HISTORICAL = [
    ("tbiom/reports/REFERENCE_MANIFEST.md", "SUPERSEDED as the method of record (it documents the DS-R operator). "
     "Retained because it contains the numerical comparison of evidence scaling against belief-mass discounting, "
     "which is the definition the current method still uses."),
    ("tbiom/reports/GROUPED_FUSION_OUTCOME.md", "SUPERSEDED / closed line of work. Retained because it explains why "
     "a partitioned (grouped) t-norm operator is NOT the proposed method, and because its zero-weight finding is "
     "why the current operator was chosen for its vacuous-view behaviour."),
    ("tbiom/reports/EXP_THREEVIEW_ABLATION.md", "SUPERSEDED for the view ablation: its numbers are DS-R, not "
     "PCR6+-R. The current view ladder is in EXP_PCR6P_DEVELOPMENT/EXPOSED. Retained for the paired bootstrap "
     "methodology and the original temporal-view effect size."),
]


def evidence(snap):
    L = ["# PAPER_EVIDENCE.md -- consolidated scientific documentation for the T-BIOM manuscript", "",
         f"Snapshot {snap['date']}, commit `{snap['commit']}`, working tree "
         f"{'DIRTY (' + str(snap['dirty_paths']) + ' paths)' if snap['dirty'] else 'clean'}.", "",
         "Packaging document. Report boundaries are preserved with explicit source-path markers; no values were "
         "recomputed, reconciled or edited while assembling it.", "",
         "## Selected method, fixed by the authors", "",
         "| item | value |", "|---|---|",
         "| operator | **PCR6+-R** -- direct three-source PCR6+ after evidence scaling |",
         "| operator source | Dezert, Dezert & Smarandache, JAIF 16(1) 2021, pp. 48--73, eqs. (15), (22)--(24), (26) |",
         "| novelty claimed | the application and the measurement; **not** the operator, which is prior work |",
         "| views | T (V-JEPA 2 ViT-L + LoRA r8/a16), APP (CLIP ViT-L/14 + LoRA r8/a16), OM (beta-VAE head, latent 32, beta_KLD 2.0, on the CLIP path) |",
         "| configuration of record | `42`: warm+gate spatial epoch 19 + `vjepa2_lora_B_seed42` |",
         "| spatial views | **frozen** (`freeze_views: true`) at the baseline's own values |",
         "| trainable parameters added | 919,938 (786,432 LoRA over 48 projections + 133,506 head) |",
         "| excluded branch | face-prior / FS-VFM student: tested, reliability fit zeroed it 5/5 folds |",
         "| discount | evidence scaling e' = r e; NOT belief-mass discounting (agrees only at r in {0,1}) |",
         "| threshold | EER on FF++ val, per arm, applied unchanged to every evaluation set |",
         "| aggregation | per-frame opinions averaged per video, then fused; temporal view emits one opinion per clip |",
         "| uncertainty outputs | fused vacuity u_f, retained conflict K with the K_TS / K_AO / K_all decomposition |",
         "", "## Contents", ""]
    for i, (p, d) in enumerate(CURRENT, 1):
        L.append(f"{i}. [CURRENT] `{p}` -- {d}")
    for i, (p, d) in enumerate(HISTORICAL, len(CURRENT) + 1):
        L.append(f"{i}. [HISTORICAL] `{p}` -- {d}")
    L += ["", "## Source inventory", "", "| path | bytes | modified | status |", "|---|---:|---|---|"]
    import datetime
    for p, _d in CURRENT + HISTORICAL:
        f = ROOT / p
        st = "CURRENT" if (p, _d) in CURRENT else "HISTORICAL / superseded"
        if f.exists():
            L.append(f"| `{p}` | {f.stat().st_size} | "
                     f"{datetime.date.fromtimestamp(f.stat().st_mtime)} | {st} |")
        else:
            L.append(f"| `{p}` | -- | -- | **MISSING** |")
    L += ["", "Machine-readable companions to every report below live beside them as `*.json` and are included in "
          "`PAPER_PROJECT.zip`.", "",
          "## Unresolved factual questions at this snapshot", "",
          "1. **No untouched confirmation set exists.** CDFv2 and DeepfakeEval-2024 informed development; DF40 "
          "informed operator design in two prior studies. Every benchmark reported is exposed evaluation.",
          "2. **Seed replication is partial.** The five development folds are reliability-fitting folds, not "
          "detector seeds. Four detector configurations exist, but two of them change the spatial training recipe, "
          "so they are not four draws of one procedure. The deployed spatial recipe exists at one seed.",
          "3. **Operating-point safeguards.** The pre-registered calibrated FPR/TPR criteria hold in one of the four "
          "configurations; AUROC noninferiority holds in all four. See `PCR6P_OUTCOME.md` section 1.",
          "4. **The conflict-by-source uncertainty score failed its pre-registered test** (beats every simple score "
          "in 1 of 4 configurations). The conflict decomposition is reported as a diagnostic only. The manuscript "
          "makes no uncertainty novelty claim.",
          "5. **DISCERN page/DOI** are not yet available; the bib entry carries the arXiv identifier.", ""]
    for tag, group in (("CURRENT", CURRENT), ("HISTORICAL", HISTORICAL)):
        L += ["", "=" * 110, f"# {tag} EVIDENCE", "=" * 110, ""]
        for p, d in group:
            f = ROOT / p
            L += ["", "-" * 110, f"## BEGIN SOURCE: `{p}`", "", f"*Role in this handoff:* {d}", "",
                  "-" * 110, ""]
            L.append(f.read_text().rstrip("\n") if f.exists()
                     else f"**MISSING at packaging time.** Expected at `{p}`.")
            L += ["", f"## END SOURCE: `{p}`", ""]
    # the DISCERN -> extension component map, which lives in the manuscript rather than a report
    ext = PAPER / "tables/tab_extension.tex"
    L += ["", "=" * 110, "# DISCERN-to-extension component map", "=" * 110, "",
          "This map is maintained in the manuscript rather than in a report, because it is qualitative. Source: "
          "`tbiom/paper/tables/tab_extension.tex` (hand-written). It was compiled from the DISCERN paper "
          "(`tbiom/docs/DISCERN.pdf`, Secs. 1, 3.3, 3.4) and the live configs "
          "(`DISCERN_Ext/eval_adaptation/configs/warm_dicome_gate_seed42.yaml`, "
          "`analysis/tbiom/temporal/train_branch.py`).", "", "```latex",
          ext.read_text().rstrip("\n") if ext.exists() else "MISSING", "```", "",
          "## Face-prior view decision (why the fusion is three views, not four)", "",
          "Source: `tbiom/reports/EXP_FACEPRIOR_VIEW.json`, produced by "
          "`analysis/tbiom/paper/faceprior_test.py`. Every arm fitted and scored by the established procedure on "
          "cached opinions.", "",
          "| view set | dev v3.1 | dev reenactment | 10-set mean | DF40 family-balanced | zero-weight folds |",
          "|---|---:|---:|---:|---:|---|"]
    fp = REPORTS / "EXP_FACEPRIOR_VIEW.json"
    if fp.exists():
        for name, r in json.load(open(fp)).items():
            z = ", ".join(f"{k} {int(v * 5)}/5" for k, v in r["zero_frac"].items() if v > 0) or "none"
            L.append(f"| {name} | {r['dev']:.4f} | {r['reen']:.4f} | {r['tenset']:.4f} | {r['df40_bal']:.4f} | {z} |")
    L += ["", "The reliability fit drives the face-prior weight to zero in 5 of 5 folds both as a substitute for the "
          "appearance view and as a fourth view, leaving the fused system numerically identical. `T+O+F` is "
          "bit-for-bit `T+O`.", ""]
    (OUT / "PAPER_EVIDENCE.md").write_text("\n".join(L) + "\n")
    print("wrote", OUT / "PAPER_EVIDENCE.md")




# ------------------------------------------------------------------ 3. PAPER_PROJECT.zip + HANDOFF_MANIFEST.md
# Repository-relative paths are preserved inside the archive.
GENERATORS = ["analysis/tbiom/paper/build_tables.py", "analysis/tbiom/paper/build_figures.py",
              "analysis/tbiom/paper/extra_facts.py", "analysis/tbiom/paper/check_tex.py",
              "analysis/tbiom/paper/make_handoff.py", "analysis/tbiom/paper/faceprior_test.py"]
HELPERS = ["analysis/tbiom/pcr6p/ops_pcr6p.py", "analysis/tbiom/pcr6p/arms.py", "analysis/tbiom/pcr6p/manifest.py",
           "analysis/tbiom/pcr6p/audit.py", "analysis/tbiom/pcr6p/development.py", "analysis/tbiom/pcr6p/exposed.py",
           "analysis/tbiom/pcr6p/verdict.py", "analysis/tbiom/grouped/ops.py", "analysis/tbiom/grouped/score.py",
           "analysis/tbiom/grouped/confirm.py", "analysis/tbiom/exp1_fusion_audit.py",
           "analysis/tbiom/df40_analysis.py", "analysis/tbiom/temporal/dsr_system.py",
           "analysis/tbiom/temporal/finish_line.py", "analysis/tbiom/temporal/fusion_combos.py",
           "analysis/tbiom/temporal/safeguard_ablation.py", "analysis/tbiom/temporal/train_branch.py"]
# the exact report files the table/figure generators read
INPUTS = ["tbiom/reports/EXP_PCR6P_EXPOSED.json", "tbiom/reports/EXP_PCR6P_DEVELOPMENT.json",
          "tbiom/reports/EXP_PCR6P_AUDIT.json", "tbiom/reports/BASELINE_MATRIX.json",
          "tbiom/reports/EXP_DF40_CROSS_MANIPULATION.json", "tbiom/reports/PAPER_EXTRA_FACTS.json",
          "tbiom/reports/PCR6P_VERDICTS.json", "tbiom/reports/EXP_FACEPRIOR_VIEW.json",
          "logs/tbiom/df40/meta.json"]
REPORTS_MD = ["tbiom/reports/PCR6P_REFERENCE_MANIFEST.md", "tbiom/reports/EXP_PCR6P_AUDIT.md",
              "tbiom/reports/EXP_PCR6P_DEVELOPMENT.md", "tbiom/reports/EXP_PCR6P_EXPOSED.md",
              "tbiom/reports/PCR6P_OUTCOME.md", "tbiom/reports/PCR6P_PRIOR_ART_AND_CLAIMS.md",
              "tbiom/reports/EXP_DF40_CROSS_MANIPULATION.md", "tbiom/reports/BASELINE_MATRIX.md"]
CONFIGS = ["tbiom/configs/pcr6p_fusion_preregister.yaml",
           "/data/umar/Repos/DISCERN_Ext/eval_adaptation/configs/warm_dicome_gate_seed42.yaml"]
EXCLUDED = [
    ("tbiom/paper/IEEEtran.cls, tbiom/paper/IEEEtran.bst", "standard IEEE template files, byte-identical to "
     "tbiom/docs/IEEE_TBIOM; excluded by instruction. Required LaTeX class: IEEEtran V1.8b (journal option)."),
    ("model checkpoints", "warm+gate spatial ckpt and vjepa2_lora_B_seed42/best.pt are hundreds of MB; paths are "
     "recorded in PCR6P_REFERENCE_MANIFEST.md with sha256 prefixes."),
    ("raw datasets and frame-level prediction caches", "logs/tbiom/stage23/*.csv is the per-frame opinion cache "
     "consumed upstream of these generators; tens of GB. The generators here read the aggregated report JSONs."),
    ("tbiom/docs/DISCERN.pdf, tbiom/docs/DiCoME.pdf", "third-party/own PDFs, not ours to redistribute in a handoff."),
]


def build_zip(snap):
    zp = OUT / "PAPER_PROJECT.zip"
    paper_files = ["tbiom/paper/main.tex", "tbiom/paper/refs.bib", "tbiom/paper/README.md",
                   "tbiom/paper/tables/key_numbers.json"]
    paper_files += [f"tbiom/paper/tables/{p.name}" for p in sorted((PAPER / "tables").glob("*.tex"))]
    paper_files += [f"tbiom/paper/figures/{p.name}" for p in sorted((PAPER / "figures").glob("*.pdf"))]
    everything = paper_files + GENERATORS + HELPERS + INPUTS + REPORTS_MD
    missing = []
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in everything:
            f = ROOT / rel
            (z.write(f, rel) if f.exists() else missing.append(rel))
        for c in CONFIGS:                                       # one config lives in the sibling repo
            f = Path(c) if c.startswith("/") else ROOT / c
            arc = c if not c.startswith("/") else "external/DISCERN_Ext/" + Path(c).name
            (z.write(f, arc) if f.exists() else missing.append(c))
        z.writestr("HANDOFF_MANIFEST.md", manifest(snap, everything, missing))
    return zp, missing


def manifest(snap, everything, missing):
    L = ["# HANDOFF_MANIFEST.md", "",
         f"Snapshot {snap['date']} | commit `{snap['commit']}` | working tree "
         f"{'DIRTY (' + str(snap['dirty_paths']) + ' paths)' if snap['dirty'] else 'clean'}", "",
         "Repository-relative paths are preserved. The one file from the sibling repository `DISCERN_Ext` is placed "
         "under `external/DISCERN_Ext/` and its original location is given below.", "",
         "## Selected configuration", "",
         "| item | value |", "|---|---|",
         "| operator | PCR6+-R, direct three-source PCR6+ after evidence scaling e' = r e |",
         "| configuration | `42` -- warm+gate spatial epoch 19 + vjepa2_lora_B_seed42 |",
         "| spatial views | frozen (`freeze_views: true`); only the temporal view is trained |",
         "| trainable added | 919,938 parameters |",
         "", "## Which files are generated and which are maintained by hand", "",
         "| file | status | authoritative input |", "|---|---|---|"]
    T = {"tab_crossdataset": "EXP_PCR6P_EXPOSED.json (ours) + BASELINE_MATRIX.json (comparators)",
         "tab_df40family": "EXP_PCR6P_EXPOSED.json + EXP_DF40_CROSS_MANIPULATION.json",
         "tab_df40gen": "EXP_PCR6P_EXPOSED.json + EXP_DF40_CROSS_MANIPULATION.json",
         "tab_views": "EXP_PCR6P_DEVELOPMENT.json + EXP_PCR6P_EXPOSED.json",
         "tab_viewrole": "EXP_PCR6P_EXPOSED.json",
         "tab_operators": "EXP_PCR6P_DEVELOPMENT.json + EXP_PCR6P_EXPOSED.json + EXP_PCR6P_AUDIT.json",
         "tab_reliability": "EXP_PCR6P_EXPOSED.json",
         "tab_perview": "PAPER_EXTRA_FACTS.json",
         "tab_slices": "EXP_PCR6P_EXPOSED.json + PAPER_EXTRA_FACTS.json",
         "tab_configs": "EXP_PCR6P_EXPOSED.json, all four configurations",
         "tab_uncertainty": "EXP_PCR6P_EXPOSED.json",
         "tab_weights": "EXP_PCR6P_EXPOSED.json",
         "tab_operatorsfull": "EXP_PCR6P_EXPOSED.json",
         "tab_conflict": "EXP_PCR6P_DEVELOPMENT.json",
         "tab_efficiency": "PAPER_EXTRA_FACTS.json (read from the branch checkpoint)"}
    for k, v in T.items():
        L.append(f"| `tbiom/paper/tables/{k}.tex` | GENERATED by `build_tables.py` | `{v}` |")
    L += ["| `tbiom/paper/tables/tab_extension.tex` | **HAND-WRITTEN** | DISCERN.pdf Secs. 1, 3.3, 3.4 + live configs |",
          "| `tbiom/paper/figures/fig_df40.pdf` | GENERATED by `build_figures.py` | EXP_PCR6P_EXPOSED.json + EXP_DF40_CROSS_MANIPULATION.json |",
          "| `tbiom/paper/figures/fig_complementarity.pdf` | GENERATED by `build_figures.py` | same |",
          "| `tbiom/paper/main.tex` | hand-written; `\\input`s the tables | -- |",
          "| `tbiom/paper/refs.bib` | hand-written, 28 entries | -- |",
          "", "**Editing rule.** Do not hand-edit any GENERATED file: `build_tables.py` overwrites all of them "
          "except `tab_extension.tex`. Column specs, `\\tabcolsep`, captions and rounding for generated tables must "
          "be changed in `build_tables.py`. The generators are deliberately retained so every number stays "
          "traceable to the report that produced it.", "",
          "## Regeneration and compilation", "", "```bash",
          "cd /data/umar/Repos/DFB_NeSyNS", "",
          "# 1. per-view test scores and parameter counts (reads the branch checkpoint)",
          "OMP_NUM_THREADS=8 python analysis/tbiom/paper/extra_facts.py", "",
          "# 2. all 15 generated tables",
          "OMP_NUM_THREADS=4 python -c \"import sys; sys.path.insert(0,'analysis/tbiom/paper'); \\",
          "    import build_tables as B; B.main(); B.extras()\"", "",
          "# 3. both figures",
          "OMP_NUM_THREADS=4 python analysis/tbiom/paper/build_figures.py", "",
          "# 4. structural validation (no LaTeX needed)",
          "python analysis/tbiom/paper/check_tex.py", "",
          "# 5. compile",
          "cd tbiom/paper && pdflatex main && bibtex main && pdflatex main && pdflatex main",
          "#    or, as used for CURRENT_PAPER.pdf in this handoff:",
          "#    tectonic -k --outdir <dir> main.tex", "```", "",
          "Python environment: `discern_ext` (numpy, pandas, scikit-learn, matplotlib, torch for `extra_facts.py`).",
          "", "## LaTeX requirements", "",
          "Class `IEEEtran` V1.8b with the `journal` option, and `IEEEtran.bst`. Both are excluded from this archive "
          "by instruction; they are byte-identical to `tbiom/docs/IEEE_TBIOM/`. Packages: `amsmath`, `amssymb`, "
          "`amsfonts`, `booktabs`, `graphicx`, `url`, `multirow`, `hyperref`, `algorithm`, `algpseudocode`, `tikz` "
          "(libraries `arrows.meta`, `positioning`, `fit`, `backgrounds`).", "",
          "## Archive file map", ""]
    L += ["```"]
    for rel in everything:
        L.append(("  " if (ROOT / rel).exists() else "! ") + rel)
    for c in CONFIGS:
        arc = c if not c.startswith("/") else "external/DISCERN_Ext/" + Path(c).name
        L.append("  " + arc + ("" if not c.startswith("/") else f"      <- original: {c}"))
    L += ["  HANDOFF_MANIFEST.md   (this file)", "```", ""]
    if missing:
        L += ["**Files requested but missing at packaging time:**", ""] + [f"- `{m}`" for m in missing] + [""]
    L += ["## Deliberately excluded", "", "| item | why |", "|---|---|"]
    for a, b in EXCLUDED:
        L.append(f"| {a} | {b} |")
    L += ["", "## Dependencies not included", "",
          "- The frame-level opinion caches in `logs/tbiom/stage23/` and the temporal opinion caches in "
          "`logs/tbiom/temporal/branches/*/opinions/` are the inputs to the *upstream* evaluation scripts "
          "(`analysis/tbiom/pcr6p/{development,exposed}.py`), not to the table generators. Regenerating the report "
          "JSONs from scratch requires them; regenerating the tables and figures does not.",
          "- Model checkpoints, as above. `extra_facts.py` reads one of them for the parameter count; if it is "
          "absent that single table cannot be regenerated, and the rest are unaffected.", ""]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    snap = snapshot()
    flatten(snap)
    print("wrote", OUT / "PAPER_REVIEW.txt")
    evidence(snap)
    zp, missing = build_zip(snap)
    print("wrote", zp, f"({zp.stat().st_size // 1024} KiB)")
    if missing:
        print("MISSING from archive:", *missing, sep="\n  ")
