"""Certification gate — per (image, claim), certify counterfactually.

The heart of CEC. Each claim is routed to a SCOPE by its location (T13):
    region      a specific landmark region -> repair THAT region, judge against
                the region-calibrated `gate_region:` block.
    composite   whole_face + non-spectral -> repair the full inner-face region,
                judge against the frozen full-mask `gate:` block. Same operation
                the frozen block was calibrated on, so no bar is lowered.
    spectral    whole_face + frequency/noise -> UNTESTABLE (spectral dropped).
    untestable  no locus / unmappable.

For a testable claim: NM = p_fake(orig) - p_fake(repaired) on the instrument;
controls = matched corruption (LPIPS-tuned) · wrong region · real-repair offset.
    CERTIFIED  NM >= margin  AND  NM - max(blur,shift) >= gap
               AND |wrong| <= inert  AND real_offset <= offset_max     (all four)
    REJECTED   otherwise, with `fail_reasons` recording which checks failed.

Orig + every variant are scored in ONE batch so the ~1e-4 batch-composition
offset cancels in each NM (Task 2). Instrument: single spatial (fsfm primary).
"""
from __future__ import annotations

from typing import Dict, Optional

from cec.data.pairing import PairSample, align_paired_real
from cec.instruments.spatial import SpatialInstrument
from cec.masks.regions import composite_mask, region_mask
from cec.proposer.validate import Claim, ValidatedResponse
from cec.registration import load_params
from cec.repair import Lpips, make_control_region, match_corruption, poisson_paste

# scope -> which calibrated params.yaml block judges it.
_SCOPE_BLOCK = {"region": "gate_region", "composite": "gate"}


class CertificationGate:
    def __init__(self, instrument: str = "fsfm", device: str = "cuda:0",
                 lpips: Optional[Lpips] = None):
        self.instrument_name = instrument
        self.instrument = SpatialInstrument(instrument, device=device)
        self.params = load_params()
        # Thresholds per scope-block, resolved once (KeyError if a block is missing).
        self._thresholds = {
            scope: self.params.gate_thresholds(instrument, block)
            for scope, block in _SCOPE_BLOCK.items()
        }
        self._lp = lpips
        self._device = device

    @property
    def lp(self) -> Lpips:
        if self._lp is None:
            self._lp = Lpips(self._device)
        return self._lp

    # ------------------------------------------------------------------ per claim
    def _region_for(self, pair: PairSample, claim: Claim, scope: str):
        if scope == "composite":
            return composite_mask(pair.landmarks)
        return region_mask(claim.location, pair.landmarks)

    def measure_region(self, pair: PairSample, region) -> Optional[Dict]:
        """Raw NM + control NMs for repairing `region` (bool mask). No thresholding.

        The single source of the counterfactual measurement — both the gate and
        the region-gate calibration call this, so the two can never drift apart
        (the plan's key risk). Returns None for an empty region.
        """
        if region is None or region.sum() == 0:
            return None
        fake, real = pair.fake, pair.real
        repaired = poisson_paste(fake, real, region)

        # LPIPS-matched corruptions of the SAME region.
        target = self.lp.dist(fake, repaired)
        blur, _, _ = match_corruption(fake, region, target, self.lp, "blur")
        shift, _, _ = match_corruption(fake, region, target, self.lp, "shift")

        # Wrong-region: equal-area region off the cited cue.
        ctrl = make_control_region(region)
        wrong = poisson_paste(fake, real, ctrl) if ctrl is not None else fake

        # Real-repair offset: same op on the paired real (+/-1 frame).
        real_b = align_paired_real(pair.landmarks, pair.vid, pair.frame, offset=1)
        if real_b is None:
            real_b = align_paired_real(pair.landmarks, pair.vid, pair.frame, offset=-1)
        real_off_img = poisson_paste(real, real_b, region) if real_b is not None else None

        # Score everything in ONE batch (batch-composition determinism).
        batch = [fake, repaired, blur, shift, wrong, real]
        if real_off_img is not None:
            batch.append(real_off_img)
        p = self.instrument.p_fake_batch(batch)
        p_fake, p_rep, p_blur, p_shift, p_wrong, p_real = p[:6]
        return {
            "nm": float(p_fake - p_rep),
            "nm_blur": float(p_fake - p_blur),
            "nm_shift": float(p_fake - p_shift),
            "nm_wrong": float(p_fake - p_wrong),
            "real_offset": float(abs(p[6] - p_real)) if real_off_img is not None else 0.0,
            "p_fake": float(p_fake),
        }

    def _certify(self, pair: PairSample, claim: Claim, scope: str) -> Dict:
        """Repair the scope's region, measure, and apply the scope's thresholds."""
        region = self._region_for(pair, claim, scope)
        m = self.measure_region(pair, region)
        if m is None:
            return {"label": "UNTESTABLE", "reason": "empty region mask", "scope": scope}

        th = self._thresholds[scope]
        nm = m["nm"]
        checks = {
            "margin": nm >= th["margin"],
            "gap": (nm - max(m["nm_blur"], m["nm_shift"])) >= th["gap"],
            "wrong": abs(m["nm_wrong"]) <= th["wrong_inert"],
            "offset": m["real_offset"] <= th["offset_max"],
        }
        certified = all(checks.values())
        return {
            "label": "CERTIFIED" if certified else "REJECTED",
            "scope": scope,
            "NM": round(nm, 4),
            "controls": {"corr": round(max(m["nm_blur"], m["nm_shift"]), 4),
                         "wrong": round(m["nm_wrong"], 4), "offset": round(m["real_offset"], 4)},
            "checks": checks,
            "fail_reasons": [k for k, ok in checks.items() if not ok],
            "thresholds": {k: round(v, 4) for k, v in th.items()},
            "p_fake": round(m["p_fake"], 4),
            "instrument": self.instrument_name,
        }

    def certify_claim(self, pair: PairSample, claim: Claim) -> Dict:
        base = {"id": claim.id, "artifact": claim.artifact, "location": claim.location}
        scope = claim.route  # region | composite | spectral | untestable | abstention
        # An abstention claim is not a manipulation assertion — nothing to certify.
        if scope == "abstention":
            base.update({"label": "ABSTENTION", "reason": "abstention", "scope": scope,
                         "examined": claim.examined, "instrument": self.instrument_name})
            return base
        if scope in ("spectral", "untestable"):
            base.update({"label": "UNTESTABLE", "reason": scope, "scope": scope,
                         "instrument": self.instrument_name})
            return base
        base.update(self._certify(pair, claim, scope))
        return base

    # ------------------------------------------------------------------ per image
    def certify_image(self, pair: PairSample, validated: ValidatedResponse,
                      provenance: Dict) -> Dict:
        """One Certified Evidence Record for an image."""
        p_fake = float(self.instrument.p_fake(pair.fake))
        claims = [self.certify_claim(pair, c) for c in validated.claims]
        return {
            "image": f"{pair.method}/{pair.vid}/{pair.frame}",
            "detector": self.instrument_name,
            "p_fake": round(p_fake, 4),
            "decision_proposer": validated.decision,
            "claims": claims,
            "provenance": provenance,
        }
