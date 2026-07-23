"""CEC — Causal Evidence Certification.

Pipeline and experiment code. Model code lives under `training/` in this repo's
DeepfakeBench house style (detectors in `training/detectors/`, modular guts in
`training/networks/cec/`); `cec/instruments/` holds thin adapters onto it.

Spine: detector decides · MLLM proposes · two instruments certify
counterfactually · assembly shows only certified evidence and abstains
otherwise · DPO teaches the proposer to cite load-bearing evidence · nothing
trained ever touches a verdict.

Invariants: freeze before you measure · training only after the audit freeze ·
every component earns a number · no invented numbers.

Decisions and gates are logged in docs/cec/LOG.md.
"""

__version__ = "0.1.0"
