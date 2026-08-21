#!/usr/bin/env python3
"""Stage 3 — the HARD GATE. Which specialists, if any, survive into the architecture?

    🔴 UMAR-RUNS (CPU, seconds):

    python analysis/discern_v2/phase2/stage3_gate.py \
        --stage0 phase2/stage0/stage0_protocol.json \
        --stage0-ceiling phase2/stage0/stage0_insample.json \
        --stage2 phase2/stage2/stage2_rate_response.json \
        --audit analysis/discern_v2/V1_domain_audit_epoch007/domain_audit.json \
        --out phase2

Everything after this stage is conditional on a go, so the decision is computed from the stage
artifacts rather than argued in prose. The script reads the JSONs, applies the brief's rule, and
writes the verdict with the numbers that produced it. It does not re-derive any measurement.

The rule, as the brief states it
---------------------------------
* GO if at least one candidate shows meaningful oracle complementarity on DF40-Dev **and** a
  realizable gate recovers a useful portion of it. Proceed with the specialists that passed; drop
  any specialist that shows neither conditional information nor audit compliance.
* NO-GO if no candidate — the validated P1d branch included — has meaningful complementarity and
  no realizable gate recovers a useful portion. Then the multi-specialist build stops and the
  paper centres on the reliability decomposition and defer, which V1 already showed working.

"Meaningful" and "useful" are thresholds, and a threshold chosen after seeing the numbers is not a
threshold. They are fixed here, in code, with the reasoning attached:

    MEANINGFUL_HEADROOM  0.02   twice the brief's 0.01 noise band, so a "meaningful" headroom is
                                one a confirming seed could not plausibly erase
    USEFUL_RECOVERY      0.25   a gate that recovers a quarter of the available headroom is doing
                                real work; one recovering a tenth is a rounding error dressed as
                                a mechanism
    MIN_METHODS          2      a single method clearing both bars is an anecdote

Change them by editing this file and re-running, so that a moved threshold is a diff.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lib  # noqa: E402

MEANINGFUL_HEADROOM = 0.02
USEFUL_RECOVERY = 0.25
MIN_METHODS = 2
ORACLE = "oracle_method"     # the honest denominator; see complementarity.py


def summarise_specialist(stage0: dict, name: str) -> dict:
    """What this specialist did across DF40-Dev, from the Stage 0 export."""
    results = stage0["results"]
    rescues, harms, deltas, qs = [], [], [], []
    for method, r in results.items():
        rh = r["rescue_harm"].get(name)
        if rh is None:
            continue
        deltas.append(rh["delta_vs_anchor"])
        if rh["rescue"] is not None:
            rescues.append((method, rh["rescue"]))
        if rh["harm"] is not None:
            harms.append((method, rh["harm"]))
        if rh["mean_q"] is not None:
            qs.append(rh["mean_q"])
    return {
        "n_methods": len(deltas),
        "median_delta_vs_anchor": float(np.median(deltas)) if deltas else float("nan"),
        "rescues": sorted((m for m, v in rescues if v > lib.NOISE_FLOOR)),
        "harms": sorted((m for m, v in harms if v < -lib.NOISE_FLOOR)),
        "mean_q": float(np.mean(qs)) if qs else None,
    }


def system_evidence(stage0: dict) -> dict:
    """Headroom and recovery, pooled over DF40-Dev. These are shared by all specialists: the
    gated system is one system, so 'did the gate recover anything' is not a per-branch quantity."""
    results = stage0["results"]
    with_headroom, recoveries, gains = [], [], []
    for method, r in results.items():
        rec = r["recovery"][ORACLE]
        gains.append(r["system"]["gated_minus_anchor"])
        if rec["no_headroom"]:
            continue
        if rec["headroom"] >= MEANINGFUL_HEADROOM:
            with_headroom.append((method, rec["headroom"]))
            recoveries.append((method, rec["recovered_fraction"]))
    useful = [(m, v) for m, v in recoveries if v >= USEFUL_RECOVERY]
    return {
        "n_methods": len(results),
        "methods_with_meaningful_headroom": sorted(m for m, _ in with_headroom),
        "median_headroom": float(np.median([v for _, v in with_headroom]))
        if with_headroom else 0.0,
        "median_recovered_fraction": float(np.median([v for _, v in recoveries]))
        if recoveries else float("nan"),
        "methods_with_useful_recovery": sorted(m for m, _ in useful),
        "median_system_gain": float(np.median(gains)) if gains else float("nan"),
        "n_system_harmed": int(sum(1 for g in gains if g < -lib.NOISE_FLOOR)),
    }


def audit_status(audit: dict | None, name: str) -> dict:
    """Whether this specialist reads provenance, from the §20 domain audit."""
    if audit is None:
        return {"available": False,
                "note": "no domain audit supplied; audit compliance could not be checked"}
    failures, checked = [], 0
    for source, entry in (audit.get("per_source") or audit.get("sources") or {}).items():
        for quantity, verdict in (entry or {}).items():
            if not quantity.startswith(name) and f"_{name}" not in quantity:
                continue
            checked += 1
            flagged = (verdict.get("domain_detector") if isinstance(verdict, dict) else None)
            if flagged:
                failures.append(f"{source}:{quantity}")
    return {"available": True, "n_checked": checked, "failures": sorted(failures),
            "clears_audit": checked > 0 and not failures}


def render(decision: dict) -> str:
    sysev = decision["system"]
    lines = [
        "# Stage 3 — HARD GATE",
        "",
        f"**Decision: {decision['verdict']}**",
        "",
        decision["justification"],
        "",
        "## Thresholds (fixed in `stage3_gate.py` before the numbers were read)",
        "",
        f"| name | value | why |", "|---|---:|---|",
        f"| meaningful headroom | {MEANINGFUL_HEADROOM} | twice the {lib.NOISE_FLOOR} noise band, "
        f"so a confirming seed could not plausibly erase it |",
        f"| useful recovery | {USEFUL_RECOVERY} | a gate recovering a quarter of the headroom is "
        f"doing real work; a tenth is a rounding error dressed as a mechanism |",
        f"| minimum methods | {MIN_METHODS} | one method clearing both bars is an anecdote |",
        "",
        "## System-level evidence (shared: the gated system is one system)",
        "",
        f"* Methods evaluated: **{sysev['n_methods']}**",
        f"* With meaningful oracle headroom (>= {MEANINGFUL_HEADROOM} against `{ORACLE}`): "
        f"**{len(sysev['methods_with_meaningful_headroom'])}** "
        f"{sysev['methods_with_meaningful_headroom'] or ''}",
        f"* Median headroom where it exists: **{sysev['median_headroom']:+.4f}**",
        f"* Median recovered fraction: **{sysev['median_recovered_fraction']:+.3f}**",
        f"* With useful recovery (>= {USEFUL_RECOVERY}): "
        f"**{len(sysev['methods_with_useful_recovery'])}** "
        f"{sysev['methods_with_useful_recovery'] or ''}",
        f"* Median system gain over the anchor alone: **{sysev['median_system_gain']:+.4f}**",
        f"* Methods the system made **worse**: **{sysev['n_system_harmed']}**",
        "",
        "## Per specialist", "",
        "| specialist | median delta vs anchor | rescues | harms | mean q | audit | keep? |",
        "|---|---:|---|---|---:|---|---|",
    ]
    for name, s in decision["specialists"].items():
        audit = s["audit"]
        audit_cell = ("not checked" if not audit["available"] else
                      "clears" if audit["clears_audit"] else
                      f"FAILS on {len(audit['failures'])}")
        q = "—" if s["mean_q"] is None else f"{s['mean_q']:.3f}"
        lines.append(
            f"| `{name}` | {s['median_delta_vs_anchor']:+.4f} | {len(s['rescues'])} "
            f"{'(' + ', '.join(s['rescues'][:4]) + ')' if s['rescues'] else ''} | "
            f"{len(s['harms'])} {'(' + ', '.join(s['harms'][:4]) + ')' if s['harms'] else ''} | "
            f"{q} | {audit_cell} | {'**KEEP**' if s['keep'] else 'drop'} |")

    if decision["ceiling"]:
        c = decision["ceiling"]
        lines += ["", "## Ceiling vs deployable", "",
                  f"The `insample` gate — cross-fit on the evaluation pool itself, so it has seen "
                  f"sibling generators — recovers a median **{c['median_recovered_fraction']:+.3f}** "
                  f"against the deployable protocol gate's "
                  f"**{sysev['median_recovered_fraction']:+.3f}**.",
                  "",
                  "The gap between them is itself a result: it separates *applicability is not "
                  "inferable at all* from *applicability is inferable from sibling generators but "
                  "does not transfer from the training protocol*. Only the second leaves room for "
                  "a better gate; the first does not.", ""]

    if decision["stage2"]:
        s2 = decision["stage2"]
        lines += ["## Stage 2 — does the rate response carry structure?", "",
                  f"* real/fake, shape over level: **{s2['forgery_increment']:+.4f}**",
                  f"* anchor-error, shape over level: **{s2['error_increment']:+.4f}**",
                  f"* family Fisher ratio, full **{s2['fisher_full']:.3f}** vs "
                  f"shape **{s2['fisher_shape']:.3f}**",
                  "",
                  "⚠️ **P1d's candidacy is DF40-contaminated by construction.** It was kept for "
                  "the D4 arm on an A2a gate-recovery number computed on DF40 "
                  "(`training/config/discern_v2/_base.yaml`), so a DF40-Dev result confirming it "
                  "is a re-test of a hypothesis DF40 already selected — not independent evidence. "
                  "See `phase2/DF40_SPLIT.md`.", ""]

    lines += ["## What happens next", "", decision["next"], ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage0", type=Path, required=True,
                    help="the DEPLOYABLE arm: stage0_protocol.json")
    ap.add_argument("--stage0-ceiling", type=Path, default=None,
                    help="stage0_insample.json — reported beside it, never used to decide")
    ap.add_argument("--stage2", type=Path, default=None)
    ap.add_argument("--audit", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    stage0 = json.loads(args.stage0.read_text())
    if stage0["meta"]["protocol"] != "protocol":
        raise SystemExit(
            f"--stage0 is the `{stage0['meta']['protocol']}` arm. The gate must be decided on the "
            f"DEPLOYABLE gate (fit on FF++ only), not on the cross-fit ceiling; pass the ceiling "
            f"to --stage0-ceiling instead, where it is reported but does not decide.")
    audit = json.loads(args.audit.read_text()) if args.audit and args.audit.is_file() else None
    ceiling = None
    if args.stage0_ceiling and args.stage0_ceiling.is_file():
        ceiling = system_evidence(json.loads(args.stage0_ceiling.read_text()))

    sysev = system_evidence(stage0)
    specialists = {}
    for name in stage0["meta"]["specialists"]:
        s = summarise_specialist(stage0, name)
        s["audit"] = audit_status(audit, name)
        # Keep a specialist if it carries conditional information (it rescues somewhere) OR it
        # clears the audit and is at worst harmless. The brief drops only those showing NEITHER.
        carries_information = len(s["rescues"]) >= MIN_METHODS
        harmless = len(s["harms"]) == 0
        clears_audit = bool(s["audit"].get("clears_audit"))
        s["keep"] = bool(carries_information or (clears_audit and harmless))
        s["keep_because"] = ("rescues on "
                             f"{len(s['rescues'])} methods" if carries_information else
                             "clears the audit and harms nothing" if s["keep"] else
                             "neither conditional information nor audit compliance")
        specialists[name] = s

    stage2 = None
    if args.stage2 and args.stage2.is_file():
        blob = json.loads(args.stage2.read_text())
        stage2 = {
            "forgery_increment": blob["probes"]["forgery"]["shape_increment"],
            "error_increment": blob["probes"]["anchor_error"]["shape_increment"],
            "fisher_full": blob["family_separability"]["full"]["fisher_ratio"],
            "fisher_shape": blob["family_separability"]["shape"]["fisher_ratio"],
        }

    headroom_ok = len(sysev["methods_with_meaningful_headroom"]) >= MIN_METHODS
    recovery_ok = len(sysev["methods_with_useful_recovery"]) >= MIN_METHODS
    keepers = [n for n, s in specialists.items() if s["keep"]]
    go = bool(headroom_ok and recovery_ok and keepers)

    if go:
        verdict = "GO — proceed to Stage 4"
        justification = (
            f"{len(sysev['methods_with_meaningful_headroom'])} DF40-Dev methods show oracle "
            f"headroom of at least {MEANINGFUL_HEADROOM}, and the deployable gate recovers at "
            f"least {USEFUL_RECOVERY} of it on "
            f"{len(sysev['methods_with_useful_recovery'])} of them "
            f"(median {sysev['median_recovered_fraction']:+.3f}). Both bars are cleared, so "
            f"conditional specialist reasoning has something to exploit.")
        nxt = (f"Stage 4 trains Branch A and the evidence heads with specialists "
               f"{keepers}. Specialists dropped here are removed from the config, not disabled by "
               f"a flag — a branch that can be re-enabled by editing one line will be.")
    else:
        verdict = "NO-GO — stop the multi-specialist build"
        reasons = []
        if not headroom_ok:
            reasons.append(
                f"only {len(sysev['methods_with_meaningful_headroom'])} method(s) show oracle "
                f"headroom of {MEANINGFUL_HEADROOM} or more, so there is little to win even with "
                f"perfect routing")
        if not recovery_ok:
            reasons.append(
                f"the deployable gate reaches useful recovery on only "
                f"{len(sysev['methods_with_useful_recovery'])} method(s) "
                f"(median {sysev['median_recovered_fraction']:+.3f}) — applicability is not "
                f"inferable from the observable evidence here")
        if not keepers:
            reasons.append("no specialist shows either conditional information or audit "
                           "compliance")
        justification = ("; ".join(reasons).capitalize() + ".\n\nThis is a legitimate negative "
                         "result, not a failure to be worked around. The brief is explicit: "
                         "\"Do not spend a week building a reasoner over specialists that carry "
                         "no recoverable signal.\"")
        nxt = ("Centre the paper on the reliability decomposition and Real/Fake/Defer, which V1 "
               "already showed working (error-detection AUROC 0.8118, selective risk "
               "0.1524 -> 0.1232 at a 10% abstention budget). Stages 4-8 as written do not run; "
               "the reliability half of Stage 7 and the evaluation of Stage 8 still do, over the "
               "anchor alone.")

    decision = {"verdict": verdict, "go": go, "justification": justification, "next": nxt,
                "thresholds": {"meaningful_headroom": MEANINGFUL_HEADROOM,
                               "useful_recovery": USEFUL_RECOVERY, "min_methods": MIN_METHODS,
                               "oracle": ORACLE},
                "system": sysev, "specialists": specialists, "ceiling": ceiling,
                "stage2": stage2, "keep": keepers,
                "sources": {"stage0": str(args.stage0),
                            "stage0_ceiling": str(args.stage0_ceiling or ""),
                            "stage2": str(args.stage2 or ""), "audit": str(args.audit or "")}}

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "stage3_gate.json").write_text(json.dumps(decision, indent=2, default=str))
    (args.out / "STAGE3_GATE.md").write_text(render(decision))
    print(f"\n{verdict}")
    print(justification)
    print(f"\nkeep: {keepers or 'nothing'}")
    print(f"wrote {args.out}/STAGE3_GATE.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
