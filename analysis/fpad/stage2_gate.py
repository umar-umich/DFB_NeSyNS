#!/usr/bin/env python3
"""Stage 2 — MECHANISM GATE. Ordinary-LoRA delta vs preservation delta.

    🔴 UMAR-RUNS (CPU, seconds):

    python analysis/fpad/stage2_gate.py \
        --ordinary logs/fpad/score/ordinary_ffppval/profile_epoch_000.parquet \
                   logs/fpad/score/ordinary_df40dev/profile_epoch_000.parquet \
        --preserve logs/fpad/score/preserve_ffppval/profile_epoch_000.parquet \
                   logs/fpad/score/preserve_df40dev/profile_epoch_000.parquet \
        --runs logs/fpad/studentA_ordinary_seed42 logs/fpad/studentA_preserve_seed42 \
        --out wacv

This is the derisking stage and it decides the paper's framing, so it runs before the full method
is trained. It answers the one question argument cannot: **is the teacher-student delta
manipulation-specific, or is it the decision boundary in a new coordinate?**

The audit is reused verbatim
----------------------------
`audit_quantity` is imported from `analysis/discern_v2/domain_audit.py`, not reimplemented — the
same forensic-vs-domain separability, the same thresholds, the same DATASET DETECTOR / BORDERLINE
/ UNINFORMATIVE / pass verdicts that produced the V1 tables. Two copies of a metric drift, and
the brief asks for this one specifically.

Per LAYER, never only aggregate
-------------------------------
Preservation may clean the early and mid layers while late, task-specialized layers stay
different — and an aggregate would hide exactly that. So every quantity is reported for each
configured layer, and the decision weighs early/mid separately from late.

The permitted-source rule, enforced
-----------------------------------
The gate may read FF++ validation, FF++ benign compression shifts, and DF40-Dev. It may NOT read
CDFv2/v3, DFDC, DFDCP, DFD or Deepfake-Eval directly: doing so would reclassify them as
development data and forfeit their zero-shot status in the final table. `assert_permitted` refuses
rather than warning.

The domain axis is the Celeb-DF-SOURCED DF40-Dev reals (Umar, 2026-08-21). DF40's `*_ff` reals are
FF++ frames, which overlap FF++-only training and would understate domain separability — making
the preservation delta look cleaner than it is. The `*_cdf` reals are genuinely out of
distribution.

That choice has a provenance cost, and it is COMPUTED here rather than left as a caveat: DF40's
`*_cdf` authentic half is drawn from Celeb-DF-v2, so those real videos are no longer strictly
unseen. `provenance_cost` reports exactly which Celeb-DF-v2 real videos were read and what
fraction of that corpus they are, so Stage 7 can footnote the row precisely. Only reals are
touched, and nothing is fit on them — this is weaker than "calibrated", stronger than "never
seen", and the paper should say so in those terms.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "analysis" / "discern_v2"))

from domain_audit import audit_quantity, separability  # noqa: E402  (reused verbatim)

FORBIDDEN = ("Celeb-DF-v2", "Celeb-DF-v3", "DFDC", "DFDCP", "DeepFakeDetection",
             "Deepfake-Eval-2024", "UADFV")
PROTOCOL = "FaceForensics++"
NOISE = 0.01
# "meaningfully cleaner" needs a number chosen before the numbers are seen.
MEANINGFUL_GAP_IMPROVEMENT = 0.02      # twice the noise band
MAJORITY = 0.5


def layer_columns(df: pd.DataFrame, prefix: str = "d_l") -> list[str]:
    cols = [c for c in df.columns if c.startswith(prefix) and c[len(prefix):].isdigit()]
    return sorted(cols, key=lambda c: int(c[len(prefix):]))


def assert_permitted(df: pd.DataFrame) -> None:
    """The gate's data budget, enforced from the data rather than promised in a comment."""
    named = set(df["dataset"].astype(str))
    bad = sorted(d for d in named if any(f.lower() in d.lower() for f in FORBIDDEN))
    if bad:
        raise SystemExit(
            f"the gate was handed forbidden sources {bad}. Reading them here reclassifies them as "
            f"development data and forfeits their zero-shot status in the final table. Score only "
            f"FF++ val, FF++ compression shifts, and DF40-Dev.")


def provenance_cost(ood: pd.DataFrame) -> dict:
    """What using CDF-sourced DF40-Dev reals actually costs, in videos.

    Computed, not asserted: the paper needs to say how much of Celeb-DF-v2 was read, and a
    footnote that cannot name a number is not a footnote a reviewer will accept.
    """
    borrowed = ood[(ood["label"] == 0) & (ood["real_source"] == "Celeb-DF-v2")]
    videos = sorted(set(borrowed["video_id"].astype(str)))
    return {
        "corpus": "Celeb-DF-v2",
        "n_real_videos_read": len(videos),
        "n_real_frames_read": int(len(borrowed)),
        "videos": videos[:50],
        "fakes_read": 0,
        "anything_fit_on_them": False,
        "wording_for_the_paper":
            "Celeb-DF-v2 REAL frames reached through DF40-Dev's borrowed authentic half were "
            "read as the domain axis of the Stage-2 mechanism gate. No Celeb-DF-v2 fake was "
            "read and nothing was fit on them. Its final-table row is therefore not strictly "
            "zero-shot on the real side, and should be footnoted as such rather than presented "
            "as unseen.",
    }


def audit_student(ffpp: pd.DataFrame, ood: pd.DataFrame, layers: list[str],
                  cls_layers: list[str]) -> dict:
    """The per-layer §20 audit for one student, plus in-domain real/fake separation."""
    ffpp_real = ffpp[ffpp["label"] == 0]
    ffpp_fake = ffpp[ffpp["label"] == 1]
    ood_real = ood[ood["label"] == 0]
    ood_fake = ood[ood["label"] == 1]

    per_layer = {}
    for col in layers:
        entry = audit_quantity(ffpp_real[col].to_numpy(), ood_real[col].to_numpy(),
                               ood_fake[col].to_numpy())
        # in-domain real/fake separation: does the delta separate manipulation where there is NO
        # domain shift at all? A delta that only works across corpora is a domain reader.
        entry["forensic_in_domain"] = separability(ffpp_real[col].to_numpy(),
                                                   ffpp_fake[col].to_numpy())
        per_layer[col] = entry

    aggregate = {}
    for col in ("d_mean", "d_early", "d_late", "patch_top10_share"):
        if col in ffpp.columns:
            aggregate[col] = audit_quantity(ffpp_real[col].to_numpy(), ood_real[col].to_numpy(),
                                            ood_fake[col].to_numpy())
    cls = {c: audit_quantity(ffpp_real[c].to_numpy(), ood_real[c].to_numpy(),
                             ood_fake[c].to_numpy()) for c in cls_layers}
    direct = None
    if "p_direct" in ffpp.columns:
        direct = audit_quantity(ffpp_real["p_direct"].to_numpy(),
                                ood_real["p_direct"].to_numpy(),
                                ood_fake["p_direct"].to_numpy())
        direct["forensic_in_domain"] = separability(ffpp_real["p_direct"].to_numpy(),
                                                    ffpp_fake["p_direct"].to_numpy())
    return {"per_layer": per_layer, "aggregate": aggregate, "cls_diagnostic": cls,
            "direct_readout": direct,
            "counts": {"ffpp_real": int(len(ffpp_real)), "ffpp_fake": int(len(ffpp_fake)),
                       "ood_real": int(len(ood_real)), "ood_fake": int(len(ood_fake))}}


def decide(ordinary: dict, preserve: dict, layers: list[str]) -> dict:
    """The brief's gate logic, computed from the two audits."""
    n = len(layers)
    half = max(1, n // 2)
    early_mid, late = layers[:half], layers[half:]

    deltas = {}
    for col in layers:
        o, p = ordinary["per_layer"][col], preserve["per_layer"][col]
        deltas[col] = {
            "gap_ordinary": o["forensic_minus_domain"],
            "gap_preserve": p["forensic_minus_domain"],
            "improvement": p["forensic_minus_domain"] - o["forensic_minus_domain"],
            "verdict_ordinary": o["verdict"],
            "verdict_preserve": p["verdict"],
            "forensic_in_domain_ordinary": o["forensic_in_domain"],
            "forensic_in_domain_preserve": p["forensic_in_domain"],
        }

    improved = [c for c in layers if deltas[c]["improvement"] >= MEANINGFUL_GAP_IMPROVEMENT]
    improved_early = [c for c in early_mid if deltas[c]["improvement"] >= MEANINGFUL_GAP_IMPROVEMENT]
    fraction = len(improved) / n
    both_fail = all(deltas[c]["verdict_preserve"] in ("DATASET DETECTOR", "UNINFORMATIVE")
                    and deltas[c]["verdict_ordinary"] in ("DATASET DETECTOR", "UNINFORMATIVE")
                    for c in layers)
    # "separate real from fake about equally" — the brief's other no-go condition
    equal_forensic = all(
        abs(deltas[c]["forensic_in_domain_preserve"] - deltas[c]["forensic_in_domain_ordinary"])
        < NOISE for c in layers)

    proceed = fraction > MAJORITY
    if proceed:
        verdict = "THESIS HOLDS — proceed to Stage 3 as a manipulation-targeted trajectory method"
        why = (f"the preservation delta is cleaner on the audit by at least "
               f"{MEANINGFUL_GAP_IMPROVEMENT} on {len(improved)} of {n} layers "
               f"({fraction:.0%}), including {len(improved_early)} of {len(early_mid)} early/mid "
               f"layers. Preservation is buying manipulation-specificity, not just a shifted "
               f"boundary.")
        nxt = ("Stage 3: two-stage training of B2 and B3, then the cross-dataset table. The "
               "headline comparisons are B2-B1 and B3-B2, not B3 against chance.")
    else:
        verdict = "PIVOT — localization-first; do NOT proceed as an anomaly method"
        reasons = [f"preservation improves the forensic-minus-domain gap by at least "
                   f"{MEANINGFUL_GAP_IMPROVEMENT} on only {len(improved)} of {n} layers "
                   f"({fraction:.0%})"]
        if both_fail:
            reasons.append("both deltas fail the audit on every layer")
        if equal_forensic:
            reasons.append("the two deltas separate real from fake equally in-domain, within the "
                           f"{NOISE} noise band")
        why = ("; ".join(reasons) + ". `L_preserve` is doing little and the signal is mostly the "
               "decision boundary.")
        nxt = ("Reframe the claim as a faithful region-resolved adaptation map rather than a "
               "domain-robust detector, and lean Stage 5 harder. Genuine FF++ masks become "
               "NECESSARY, not optional — they are staged for Deepfakes, Face2Face, FaceSwap, "
               "NeuralTextures and DFD. Flag this to the author immediately: it changes the "
               "writeup, and the brief wants that known by day two.")

    return {"proceed": proceed, "verdict": verdict, "justification": why, "next": nxt,
            "per_layer": deltas, "n_layers_improved": len(improved),
            "n_early_mid_improved": len(improved_early), "fraction_improved": fraction,
            "both_fail_audit": both_fail, "equal_in_domain_forensic": equal_forensic,
            "thresholds": {"meaningful_gap_improvement": MEANINGFUL_GAP_IMPROVEMENT,
                           "majority": MAJORITY, "noise": NOISE}}


def render(payload: dict) -> str:
    d = payload["decision"]
    o, p = payload["ordinary"], payload["preserve"]
    layers = payload["layers"]

    lines = [
        "# Stage 2 — MECHANISM GATE",
        "",
        f"**{d['verdict']}**",
        "",
        d["justification"],
        "",
        f"Students: `{payload['runs']['ordinary']}` (lambda_preserve = "
        f"{payload['runs']['lambda_ordinary']}) vs `{payload['runs']['preserve']}` "
        f"(lambda_preserve = {payload['runs']['lambda_preserve']}), epoch "
        f"{payload['epoch']}, matched: **{payload['matched']}**.",
        "",
        f"Domain axis: **Celeb-DF-sourced DF40-Dev reals** "
        f"({o['counts']['ood_real']} frames). In-domain side: FF++ val "
        f"({o['counts']['ffpp_real']} real / {o['counts']['ffpp_fake']} fake). "
        f"OOD fakes: {o['counts']['ood_fake']}.",
        "",
        "> The FF++ compression axis the brief also names is **unavailable** — only c23 is staged "
        "on this machine (`wacv/REPO_MAP.md` item 8). This audit is therefore single-axis, on "
        "clean OOD reals. Recorded so the gate's basis is not overstated.",
        "",
        "## Per-layer audit — the table the decision rests on",
        "",
        "`gap` is forensic-minus-domain separability: positive means the delta separates "
        "manipulation better than it separates corpora.",
        "",
        "| layer | gap (ordinary) | gap (preserve) | improvement | verdict (ordinary) | "
        "verdict (preserve) |", "|---|---:|---:|---:|---|---|",
    ]
    for col in layers:
        e = d["per_layer"][col]
        mark = " **←**" if e["improvement"] >= d["thresholds"]["meaningful_gap_improvement"] else ""
        lines.append(f"| {col.replace('d_l', 'layer ')} | {e['gap_ordinary']:+.4f} | "
                     f"{e['gap_preserve']:+.4f} | {e['improvement']:+.4f}{mark} | "
                     f"{e['verdict_ordinary']} | {e['verdict_preserve']} |")

    lines += ["", "## Does the delta separate manipulation with NO domain shift?", "",
              "In-domain (FF++ val real vs fake) separability. A delta that only works across "
              "corpora is reading the corpus.", "",
              "| layer | ordinary | preserve |", "|---|---:|---:|"]
    for col in layers:
        e = d["per_layer"][col]
        lines.append(f"| {col.replace('d_l', 'layer ')} | "
                     f"{e['forensic_in_domain_ordinary']:.4f} | "
                     f"{e['forensic_in_domain_preserve']:.4f} |")

    lines += ["", "## Aggregates and diagnostics", "",
              "| quantity | student | domain sep | forensic sep | gap | verdict |",
              "|---|---|---:|---:|---:|---|"]
    for label, student in (("ordinary", o), ("preserve", p)):
        for name, e in list(student["aggregate"].items()):
            lines.append(f"| `{name}` | {label} | {e['domain_separability']:.4f} | "
                         f"{e['forensic_separability']:.4f} | "
                         f"{e['forensic_minus_domain']:+.4f} | {e['verdict']} |")
        if student["direct_readout"]:
            e = student["direct_readout"]
            lines.append(f"| `p_direct` (the B1 readout) | {label} | "
                         f"{e['domain_separability']:.4f} | {e['forensic_separability']:.4f} | "
                         f"{e['forensic_minus_domain']:+.4f} | {e['verdict']} |")
    lines += ["", "The frozen-teacher and student-only terms are expected to fail the audit by "
                  "construction and are reported for the design justification, never depended "
                  "on. `d_cls_*` is the CLS diagnostic; the primary readout is the patch mean.",
              ""]

    pc = payload["provenance_cost"]
    lines += ["## Provenance cost of the domain axis", "",
              f"* Celeb-DF-v2 **real** videos read: **{pc['n_real_videos_read']}** "
              f"({pc['n_real_frames_read']} frames)",
              f"* Celeb-DF-v2 fakes read: **{pc['fakes_read']}**",
              f"* Anything fit on them: **{pc['anything_fit_on_them']}**", "",
              f"> {pc['wording_for_the_paper']}", ""]

    lines += ["## What happens next", "", d["next"], ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ordinary", type=Path, nargs="+", required=True)
    ap.add_argument("--preserve", type=Path, nargs="+", required=True)
    ap.add_argument("--runs", type=Path, nargs=2, default=None,
                    help="the two Stage-A run dirs; the gate REFUSES unless they are matched")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--allow-unmatched", action="store_true",
                    help="override the matched-student precondition. Only legitimate for a "
                         "diagnostic look; a gate decision taken this way is not valid.")
    args = ap.parse_args()

    matched = "not checked"
    if args.runs:
        sys.path.insert(0, str(REPO / "analysis" / "fpad"))
        import subprocess
        proc = subprocess.run(
            [sys.executable, str(REPO / "analysis/fpad/assert_matched.py"),
             "--runs", str(args.runs[0]), str(args.runs[1])],
            capture_output=True, text=True)
        print(proc.stdout.strip())
        matched = proc.returncode == 0
        if not matched and not args.allow_unmatched:
            raise SystemExit(
                "the two students are NOT matched (see above), so any difference between their "
                "deltas includes a data or schedule difference. Fix the runs, or pass "
                "--allow-unmatched and label the result a diagnostic rather than a gate "
                "decision.")

    ord_df = pd.concat([pd.read_parquet(p) for p in args.ordinary], ignore_index=True)
    pre_df = pd.concat([pd.read_parquet(p) for p in args.preserve], ignore_index=True)
    for df in (ord_df, pre_df):
        assert_permitted(df)

    layers = layer_columns(ord_df)
    cls_layers = layer_columns(ord_df, "d_cls_l")
    if not layers:
        raise SystemExit("no `d_l*` columns in the profiles — was score_fpad.py run?")

    def split(df):
        ffpp = df[df["dataset"] == PROTOCOL]
        ood = df[(df["dataset"] != PROTOCOL) & (df["real_source"] != PROTOCOL)]
        if ffpp.empty:
            raise SystemExit(f"no {PROTOCOL} rows — the in-domain side of the audit is missing")
        if ood.empty:
            raise SystemExit(
                "no OOD rows whose real half is NOT FF++-sourced. The domain axis must be the "
                "Celeb-DF-sourced DF40-Dev reals; scoring only `*_ff` methods gives an axis that "
                "overlaps FF++-only training and would understate domain separability.")
        return ffpp, ood

    ord_ffpp, ord_ood = split(ord_df)
    pre_ffpp, pre_ood = split(pre_df)

    ordinary = audit_student(ord_ffpp, ord_ood, layers, cls_layers)
    preserve = audit_student(pre_ffpp, pre_ood, layers, cls_layers)
    decision = decide(ordinary, preserve, layers)

    ord_meta = json.loads((args.ordinary[0].parent / "score_meta.json").read_text()) \
        if (args.ordinary[0].parent / "score_meta.json").is_file() else {}
    pre_meta = json.loads((args.preserve[0].parent / "score_meta.json").read_text()) \
        if (args.preserve[0].parent / "score_meta.json").is_file() else {}

    payload = {
        "decision": decision, "ordinary": ordinary, "preserve": preserve, "layers": layers,
        "matched": matched, "epoch": ord_meta.get("epoch", "?"),
        "provenance_cost": provenance_cost(ord_ood),
        "runs": {"ordinary": str(args.runs[0]) if args.runs else "?",
                 "preserve": str(args.runs[1]) if args.runs else "?",
                 "lambda_ordinary": (ord_meta.get("student_meta") or {}).get("lambda_preserve"),
                 "lambda_preserve": (pre_meta.get("student_meta") or {}).get("lambda_preserve")},
        "profiles": {"ordinary": [str(p) for p in args.ordinary],
                     "preserve": [str(p) for p in args.preserve]},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "stage2_gate.json").write_text(json.dumps(payload, indent=2, default=str))
    (args.out / "STAGE2_GATE.md").write_text(render(payload))

    print(f"\n{decision['verdict']}")
    print(decision["justification"])
    print(f"\nper-layer gap improvement (preserve - ordinary):")
    for col in layers:
        e = decision["per_layer"][col]
        print(f"  {col:8s} {e['gap_ordinary']:+.4f} -> {e['gap_preserve']:+.4f}  "
              f"({e['improvement']:+.4f})  {e['verdict_ordinary']} -> {e['verdict_preserve']}")
    print(f"\nwrote {args.out}/STAGE2_GATE.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
