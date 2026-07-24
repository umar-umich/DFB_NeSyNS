"""T19 — Disclosure Policy (replaces "Assembly").

This module performs NO certification. It decides what may be SHOWN, given a
Certified Evidence Record the gate already produced. It is a display policy, not
a model.

Three-tier rule, in order:
  1. any REGION-certified claim   -> show region evidence (most specific).
  2. else any COMPOSITE-certified -> show the composite statement:
     "the inner face is a composited replacement; no individual region is
      independently necessary."
  3. else                         -> abstain with the contentful form.

Deployment modes (config flag):
  certified_mode   a reference/source image exists and the gate ran live.
                   Wording may say "certified evidence".
  screening_mode   no paired real; the gate did NOT run. Wording must say
                   "causally supervised evidence" (from the CEC-trained
                   proposer) and must never say "certified".

Disagreement handling: if the detector says REAL but the proposer asserts
manipulation claims -> suppress the claims and emit
`detector_explainer_disagreement`. Default OFF for the audit (so the FP-claim
metric measures RAW proposer behaviour) and ON for the demo. Metrics are always
computed pre-suppression.
"""
from __future__ import annotations

from typing import Dict

ABSTENTION = "rationale_ungrounded"
DISAGREEMENT = "detector_explainer_disagreement"

CERTIFIED_MODE = "certified_mode"
SCREENING_MODE = "screening_mode"

COMPOSITE_STATEMENT = ("the inner face is a composited replacement; "
                       "no individual region is independently necessary")

# artifact -> human phrasing (region filled in).
_TEMPLATES = {
    "boundary_artifact": "a manipulation boundary at the {loc}",
    "blend_seam_visible": "a visible blend seam at the {loc}",
    "frequency_anomaly": "an anomalous frequency signature ({loc})",
    "noise_inconsistency": "inconsistent noise at the {loc}",
    "geometry_inconsistency": "geometric inconsistency at the {loc}",
    "identity_drift": "identity drift at the {loc}",
    "texture_anomaly": "anomalous texture at the {loc}",
    "physiology_violation": "an implausible physiological detail at the {loc}",
    "lighting_inconsistency": "inconsistent lighting ({loc})",
    "OTHER": "an artifact at the {loc}",
}


def _verbalise(claim: Dict) -> str:
    loc = (claim.get("location") or "face").replace("_", " ")
    tmpl = _TEMPLATES.get(claim.get("artifact"), _TEMPLATES["OTHER"])
    nm = claim.get("NM")
    phrase = tmpl.format(loc=loc)
    return f"{phrase} (necessity {nm:.2f})" if nm is not None else phrase


class DisclosurePolicy:
    """Decides what a Certified Evidence Record is allowed to display."""

    def __init__(self, mode: str = CERTIFIED_MODE, suppress_on_disagreement: bool = False):
        if mode not in (CERTIFIED_MODE, SCREENING_MODE):
            raise ValueError(f"unknown mode '{mode}' (certified_mode|screening_mode)")
        self.mode = mode
        self.suppress_on_disagreement = suppress_on_disagreement

    @property
    def evidence_phrase(self) -> str:
        """The ONLY place the certified/supervised wording is decided."""
        return ("certified evidence" if self.mode == CERTIFIED_MODE
                else "causally supervised evidence")

    def apply(self, record: Dict) -> Dict:
        claims = record.get("claims", [])
        p_fake = record.get("p_fake")
        verdict = "FAKE" if (p_fake is not None and p_fake >= 0.5) else "REAL"
        untestable = sum(1 for c in claims if c.get("label") == "UNTESTABLE")

        certified = [c for c in claims if c.get("label") == "CERTIFIED"]
        region_cert = [c for c in certified if c.get("scope") == "region"]
        composite_cert = [c for c in certified if c.get("scope") == "composite"]

        out = {
            "image": record.get("image"),
            "verdict": verdict,                 # detector passthrough, never modified
            "detector": record.get("detector"),
            "p_fake": p_fake,
            "mode": self.mode,
            "evidence_phrase": self.evidence_phrase,
            "n_untestable": untestable,
            "disagreement": False,
        }

        # Disagreement: detector says REAL but the proposer asserted manipulations.
        asserted = record.get("n_manipulation_claims")
        if asserted is None:
            asserted = len([c for c in claims if c.get("reason") != "abstention"])
        if verdict == "REAL" and asserted > 0:
            out["disagreement"] = True
            if self.suppress_on_disagreement:
                out.update({"tier": "suppressed", "rationale": DISAGREEMENT,
                            "abstained": True, "n_certified": 0})
                return out

        # Tier 1: region evidence (most specific).
        if region_cert:
            out.update({"tier": "region", "abstained": False,
                        "n_certified": len(region_cert),
                        "rationale": [_verbalise(c) for c in region_cert]})
            return out
        # Tier 2: composite statement.
        if composite_cert:
            nm = max((c.get("NM") or 0) for c in composite_cert)
            out.update({"tier": "composite", "abstained": False,
                        "n_certified": len(composite_cert),
                        "rationale": [f"{COMPOSITE_STATEMENT} (necessity {nm:.2f})"]})
            return out
        # Tier 3: abstain.
        out.update({"tier": "abstention", "abstained": True, "n_certified": 0,
                    "rationale": ABSTENTION})
        return out

    def render(self, record: Dict) -> str:
        o = self.apply(record)
        head = f"VERDICT: {o['verdict']}  (detector {o['detector']}, p_fake={o['p_fake']})"
        if o["tier"] == "suppressed":
            return f"{head}\n{DISAGREEMENT}: detector says REAL; proposer claims suppressed."
        if o["abstained"]:
            return (f"{head}\nRATIONALE: {ABSTENTION} "
                    f"(no cited evidence survived the counterfactual gate)")
        lines = [head, f"{o['evidence_phrase'].upper()}:"]
        lines += [f"  - {r}" for r in o["rationale"]]
        if o["n_untestable"]:
            lines.append(f"({o['n_untestable']} claim(s) untestable, not shown)")
        return "\n".join(lines)


# Back-compat wrappers (older callers used assemble/render).
def assemble(record: Dict, mode: str = CERTIFIED_MODE) -> Dict:
    return DisclosurePolicy(mode).apply(record)


def render(record: Dict, mode: str = CERTIFIED_MODE) -> str:
    return DisclosurePolicy(mode).render(record)
