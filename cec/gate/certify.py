"""Certification gate — per (image, claim), certify counterfactually.

The heart of CEC. For each proposed claim:
  route:
    spatial (a real landmark region)  -> repair the CITED region with paired-real
                                         pixels, re-query the instrument
    spectral / untestable             -> UNTESTABLE (spectral dropped after Pilot S;
                                         whole_face/no-locus has no spatial mask)
  measure: NM = p_fake(orig) - p_fake(repaired), on the instrument
  control: matched corruption (LPIPS-tuned) · wrong region (equal-area, off-cue) ·
           real-repair offset
  label:
    CERTIFIED  NM >= margin[det]  AND  NM - max(blur,shift) >= gap
               AND |wrong| <= inert  AND real_offset <= offset_max
    REJECTED   otherwise
    UNTESTABLE no testable locus

Thresholds are the FROZEN gate values (params.yaml). Orig + every variant are
scored in ONE batch so the ~1e-4 batch-composition offset cancels in each NM
(Task 2 finding). Output is a per-image Certified Evidence Record.

Instrument: single spatial instrument (fsfm primary; Pilot S dropped spectral).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from cec.data.pairing import PairSample, align_paired_real
from cec.instruments.spatial import SpatialInstrument
from cec.masks.regions import region_mask
from cec.proposer.validate import Claim, ValidatedResponse
from cec.registration import load_params
from cec.repair import Lpips, make_control_region, match_corruption, poisson_paste


class CertificationGate:
    def __init__(self, instrument: str = "fsfm", device: str = "cuda:0", lpips: Optional[Lpips] = None):
        self.instrument_name = instrument
        self.instrument = SpatialInstrument(instrument, device=device)
        self.params = load_params()
        self.margin = self.params.certify_margin(instrument)
        self.gap = self.params.matched_corruption_gap
        self.wrong_inert = self.params.wrong_region_inert_max
        self.offset_max = self.params.real_offset_max
        self._lp = lpips
        self._device = device

    @property
    def lp(self) -> Lpips:
        if self._lp is None:
            self._lp = Lpips(self._device)
        return self._lp

    # ------------------------------------------------------------------ per claim
    def _certify_spatial(self, pair: PairSample, claim: Claim) -> Dict:
        """Repair the cited region, score orig+repair+controls in one batch, label."""
        region = region_mask(claim.location, pair.landmarks)
        if region is None or region.sum() == 0:
            return {"label": "UNTESTABLE", "reason": "empty region mask"}

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

        nm = float(p_fake - p_rep)
        nm_blur = float(p_fake - p_blur)
        nm_shift = float(p_fake - p_shift)
        nm_wrong = float(p_fake - p_wrong)
        real_offset = float(abs(p[6] - p_real)) if real_off_img is not None else 0.0

        certified = (
            nm >= self.margin
            and (nm - max(nm_blur, nm_shift)) >= self.gap
            and abs(nm_wrong) <= self.wrong_inert
            and real_offset <= self.offset_max
        )
        return {
            "label": "CERTIFIED" if certified else "REJECTED",
            "NM": round(nm, 4),
            "controls": {"corr": round(max(nm_blur, nm_shift), 4),
                         "wrong": round(nm_wrong, 4), "offset": round(real_offset, 4)},
            "p_fake": round(float(p_fake), 4),
            "instrument": self.instrument_name,
        }

    def certify_claim(self, pair: PairSample, claim: Claim) -> Dict:
        base = {"id": claim.id, "artifact": claim.artifact, "location": claim.location}
        if claim.route != "spatial":
            base.update({"label": "UNTESTABLE", "reason": claim.route,
                         "instrument": self.instrument_name})
            return base
        base.update(self._certify_spatial(pair, claim))
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
