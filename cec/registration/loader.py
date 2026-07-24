"""Readers for the frozen registration files.

The YAML files in this directory are the contract; this module is only the
typed door onto them. It deliberately does no defaulting: a missing key is an
error, not a silently-invented number.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

REGISTRATION_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = REGISTRATION_DIR / "frozen_prompts"


def _read_yaml(name: str) -> Dict[str, Any]:
    path = REGISTRATION_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"registration file missing: {path}")
    with path.open() as fh:
        return yaml.safe_load(fh)


@dataclass(frozen=True)
class Params:
    """params.yaml, with the gate thresholds reachable by name."""

    raw: Dict[str, Any]

    # --- proposer -----------------------------------------------------------
    @property
    def k_claims(self) -> int:
        return self.raw["proposer"]["k_claims"]

    @property
    def seed(self) -> int:
        return self.raw["proposer"]["seed"]

    @property
    def temperature(self) -> float:
        return self.raw["proposer"]["temperature"]

    # --- gate ---------------------------------------------------------------
    # Two calibrated blocks (T13): `gate` (frozen, full-mask/composite repair) and
    # `gate_region` (region-sized repair). Accessors take a block name; missing
    # keys raise (no silent defaults) — a threshold must have a measured
    # distribution behind it.
    def _gate_block(self, block: str) -> Dict[str, Any]:
        if block not in self.raw:
            raise KeyError(
                f"no gate block '{block}' in params.yaml. Known gate blocks: "
                f"{[b for b in ('gate', 'gate_region') if b in self.raw]}."
            )
        return self.raw[block]

    def certify_margin(self, detector: str, block: str = "gate") -> float:
        margins = self._gate_block(block)["certify_margin"]
        if detector not in margins:
            raise KeyError(
                f"no certify margin for detector '{detector}' in '{block}'. "
                f"Known: {sorted(margins)}. Margins are pilot-derived — add one "
                f"only with a measured control distribution behind it."
            )
        return margins[detector]

    def wrong_region_inert_max(self, block: str = "gate") -> float:
        return self._gate_block(block)["wrong_region_inert_max"]

    def real_offset_max(self, block: str = "gate") -> float:
        return self._gate_block(block)["real_offset_max"]

    def matched_corruption_gap(self, block: str = "gate") -> float:
        return self._gate_block(block)["matched_corruption_gap"]

    def gate_thresholds(self, detector: str, block: str = "gate") -> Dict[str, float]:
        """All four thresholds for a (detector, block), as a dict."""
        return {
            "margin": self.certify_margin(detector, block),
            "gap": self.matched_corruption_gap(block),
            "wrong_inert": self.wrong_region_inert_max(block),
            "offset_max": self.real_offset_max(block),
        }

    # --- qc -----------------------------------------------------------------
    @property
    def qc(self) -> Dict[str, Any]:
        return self.raw["qc"]

    @property
    def controls(self) -> Dict[str, Any]:
        return self.raw["controls"]

    @property
    def data(self) -> Dict[str, Any]:
        return self.raw["data"]


@lru_cache(maxsize=1)
def load_params() -> Params:
    return Params(_read_yaml("params.yaml"))


@lru_cache(maxsize=1)
def load_detectors() -> Dict[str, Any]:
    return _read_yaml("detectors.yaml")


@lru_cache(maxsize=1)
def load_pins() -> Dict[str, Any]:
    return _read_yaml("pins.yaml")


def prompt_hash(text: str) -> str:
    """Stable short hash of a prompt, recorded in every record's provenance."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@lru_cache(maxsize=8)
def load_prompt(name: str) -> Tuple[str, str]:
    """Return (prompt_text, prompt_hash) for a frozen prompt, e.g. 'proposer_type_b'."""
    path = PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        available = sorted(p.stem for p in PROMPTS_DIR.glob("*.txt"))
        raise FileNotFoundError(f"no frozen prompt '{name}'. Available: {available}")
    text = path.read_text()
    return text, prompt_hash(text)
