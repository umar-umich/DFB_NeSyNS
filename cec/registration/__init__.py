"""CEC registration: the frozen run configuration.

Everything downstream (instruments, gate, records, assembly, DPO, eval) reads
its constants from here rather than hard-coding them, so that the Task 9 audit
freeze covers a single source of truth.

Typical use:

    from cec.registration import load_params, load_detectors, load_prompt

    params = load_params()
    margin = params.certify_margin("effort")          # 0.10
    prompt, prompt_hash = load_prompt("proposer_type_b")
"""
from .loader import (  # noqa: F401
    REGISTRATION_DIR,
    Params,
    load_params,
    load_detectors,
    load_pins,
    load_prompt,
    prompt_hash,
)

__all__ = [
    "REGISTRATION_DIR",
    "Params",
    "load_params",
    "load_detectors",
    "load_pins",
    "load_prompt",
    "prompt_hash",
]
