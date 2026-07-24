"""Output assembly — the hallucination-free guarantee.

~50 lines, no model. Takes a Certified Evidence Record and produces what the user
sees:
  verdict    = the detector's verdict (PASSTHROUGH — never modified)
  rationale  = ONLY the CERTIFIED claims, verbalised from templates
  abstain    = if no claim certifies -> `rationale_ungrounded` (glossary Step-6)
  untestable = surfaced as a count, never guessed

This is where the guarantee lives: the rationale can contain nothing the gate
did not causally certify. An empty certified set is an honest abstention, not a
fabricated explanation.
"""
from __future__ import annotations

from typing import Dict

# artifact -> human phrasing (region is filled in).
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

ABSTENTION = "rationale_ungrounded"


def _verbalise(claim: Dict) -> str:
    loc = (claim.get("location") or "face").replace("_", " ")
    tmpl = _TEMPLATES.get(claim.get("artifact"), _TEMPLATES["OTHER"])
    nm = claim.get("NM")
    phrase = tmpl.format(loc=loc)
    return f"{phrase} (necessity {nm:.2f})" if nm is not None else phrase


def assemble(record: Dict) -> Dict:
    """Certified Evidence Record -> user-facing output. Verdict is passthrough."""
    claims = record.get("claims", [])
    certified = [c for c in claims if c.get("label") == "CERTIFIED"]
    untestable = sum(1 for c in claims if c.get("label") == "UNTESTABLE")

    p_fake = record.get("p_fake")
    verdict = "FAKE" if (p_fake is not None and p_fake >= 0.5) else "REAL"

    if certified:
        rationale = [_verbalise(c) for c in certified]
    else:
        rationale = ABSTENTION  # nothing certified -> honest abstention

    return {
        "image": record.get("image"),
        "verdict": verdict,                    # detector passthrough, never modified
        "detector": record.get("detector"),
        "p_fake": p_fake,
        "rationale": rationale,
        "n_certified": len(certified),
        "n_untestable": untestable,
        "abstained": not certified,
    }


def render(record: Dict) -> str:
    """A human-readable block (glossary Step-6 shape)."""
    out = assemble(record)
    lines = [f"VERDICT: {out['verdict']}  (detector {out['detector']}, p_fake={out['p_fake']})"]
    if out["abstained"]:
        lines.append(f"CERTIFIED RATIONALE: {ABSTENTION} "
                     f"(no cited evidence survived the counterfactual gate)")
    else:
        lines.append("CERTIFIED RATIONALE:")
        for r in out["rationale"]:
            lines.append(f"  - {r}")
    if out["n_untestable"]:
        lines.append(f"({out['n_untestable']} claim(s) untestable, not shown)")
    return "\n".join(lines)
