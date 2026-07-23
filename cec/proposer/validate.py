"""Claim validator — the ~30-line gate that replaced the standalone parser.

We control the proposer prompt (strict JSON schema + embedded vocab), so parsing
reduces to: is the JSON well-formed, and does each claim's artifact/location fall
in the frozen codebook? Anything out of vocab is coerced, never guessed:
  artifact not in vocab -> OTHER (dropped downstream; recorded)
  location not a region -> None
  location == whole_face + frequency/noise artifact -> routes to spectral
  location == whole_face + non-spectral artifact -> untestable (no spatial mask)

Schema-validity (Task 6 gate, ≥95%): a response is schema-valid iff it parses as
the object schema and `decision` ∈ {real, fake}. Individual out-of-vocab claim
fields are coerced, not counted as invalidity — that is the validator's job.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional

from cec.registration import load_params
from cec.registration.vocab import ARTIFACTS, OTHER, REGIONS, WHOLE_FACE, routes_to_spectral

_K = load_params().k_claims


@dataclass
class Claim:
    id: str
    artifact: str
    location: Optional[str]
    description: str
    route: str  # 'spatial' | 'spectral' | 'untestable'


@dataclass
class ValidatedResponse:
    schema_valid: bool
    decision: Optional[str]           # 'real' | 'fake' | None
    claims: List[Claim] = field(default_factory=list)
    error: Optional[str] = None
    raw: str = ""


def _strip_fences(text: str) -> str:
    """Tolerate a stray ```json fence even though the prompt forbids it."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return m.group(0) if m else text


def _route(artifact: str, location: Optional[str]) -> str:
    if routes_to_spectral(artifact, location):
        return "spectral"
    if location is not None and location != WHOLE_FACE:
        return "spatial"          # a real landmark region -> spatial repair
    return "untestable"           # no locus, or whole_face non-spectral


def validate(text: str) -> ValidatedResponse:
    """Parse and coerce one proposer response against the frozen schema/vocab."""
    try:
        obj = json.loads(_strip_fences(text))
    except (json.JSONDecodeError, TypeError) as e:
        return ValidatedResponse(schema_valid=False, decision=None, error=f"json: {e}", raw=text)

    if not isinstance(obj, dict) or obj.get("decision") not in ("real", "fake"):
        return ValidatedResponse(schema_valid=False, decision=None,
                                 error="missing/invalid 'decision'", raw=text)

    claims: List[Claim] = []
    for i, c in enumerate(obj.get("claims", [])[:_K]):
        if not isinstance(c, dict):
            continue
        artifact = c.get("artifact") if c.get("artifact") in ARTIFACTS else OTHER
        loc = c.get("location")
        location = loc if loc in REGIONS else None
        claims.append(Claim(
            id=c.get("id") or f"c{i + 1}",
            artifact=artifact,
            location=location,
            description=(c.get("description") or "").strip(),
            route=_route(artifact, location),
        ))

    return ValidatedResponse(schema_valid=True, decision=obj["decision"], claims=claims, raw=text)
