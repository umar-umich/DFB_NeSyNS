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
from cec.registration.vocab import ARTIFACTS, OTHER, REGIONS, claim_scope

_K = load_params().k_claims


@dataclass
class Claim:
    id: str
    artifact: str
    location: Optional[str]
    description: str
    route: str  # 'region' | 'composite' | 'spectral' | 'untestable' | 'abstention'
    examined: Optional[List[str]] = None  # set only for abstention claims

    @property
    def is_abstention(self) -> bool:
        return self.route == "abstention"

    @property
    def is_manipulation_claim(self) -> bool:
        """A positive assertion of a manipulation (the thing that can be a false positive)."""
        return not self.is_abstention


@dataclass
class ValidatedResponse:
    schema_valid: bool
    decision: Optional[str]           # 'real' | 'fake' | None
    claims: List[Claim] = field(default_factory=list)
    error: Optional[str] = None
    raw: str = ""

    @property
    def abstained(self) -> bool:
        """True if the proposer asserted NO manipulation claim (abstention or empty)."""
        return not any(c.is_manipulation_claim for c in self.claims)

    @property
    def manipulation_claims(self) -> List[Claim]:
        return [c for c in self.claims if c.is_manipulation_claim]


def _strip_fences(text: str) -> str:
    """Tolerate a stray ```json fence even though the prompt forbids it."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return m.group(0) if m else text


def _route(artifact: str, location: Optional[str]) -> str:
    # T13 scope routing: region | composite | spectral | untestable.
    return claim_scope(artifact, location)


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
        cid = c.get("id") or f"c{i + 1}"
        # Abstention object (T15): "no_certified_evidence" + regions examined.
        if c.get("verdict_support") == "no_certified_evidence":
            examined = c.get("examined")
            claims.append(Claim(id=cid, artifact=None, location=None, description="",
                                route="abstention",
                                examined=examined if isinstance(examined, list) else []))
            continue
        artifact = c.get("artifact") if c.get("artifact") in ARTIFACTS else OTHER
        loc = c.get("location")
        location = loc if loc in REGIONS else None
        claims.append(Claim(
            id=cid,
            artifact=artifact,
            location=location,
            description=(c.get("description") or "").strip(),
            route=_route(artifact, location),
        ))

    return ValidatedResponse(schema_valid=True, decision=obj["decision"], claims=claims, raw=text)
