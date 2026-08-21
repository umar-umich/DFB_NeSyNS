# Terminology to fix before the writeup

Running list of terms whose *code* names are fine but whose *paper* names are not. Nothing here is
renamed in code yet — a rename mid-run is churn, and file names like `stage2_gate.py` are already
referenced by launch scripts and commit messages. Apply at writeup, or alias late.

## "gate" — REPLACE (Umar, 2026-08-21)

The brief calls Stage 2 the "MECHANISM GATE" and the term propagated into
`analysis/fpad/stage2_gate.py`, `STAGE2_GATE.md` and `stage2_gate.json`. It reads as generic
LLM-generated scaffolding and should not appear in the paper.

There is also a substantive reason beyond style: **"gate" already means something else in this
codebase.** The V1/Phase-2 work uses "applicability gate" for the learned `q_b` that discounts a
specialist per sample (`networks/discern_v2/applicability_gate.py`). That is a *learned, per-sample
weighting*; Stage 2 is a *one-off go/no-go on the mechanism*. Two unrelated things under one word
is a real ambiguity in a paper that may cite both lines of work.

Candidates, best first:

| candidate | why |
|---|---|
| **mechanism validation** | says exactly what the stage does; "Stage 2: Mechanism Validation" reads as a normal paper section |
| **specificity test** | names the *question* — is the delta manipulation-specific or domain-driven — rather than the bureaucracy of deciding |
| **falsification test** | strongest scientifically; appropriate because the stage is designed to be able to kill the thesis |
| derisking experiment | accurate but sounds like project management, not science |
| go/no-go | plain, but still process-flavoured |

Recommendation: **"mechanism validation"** for the stage name, **"specificity test"** for the
specific ordinary-vs-preservation comparison inside it. Reserve "gate" exclusively for the learned
applicability `q_b`, so the word keeps one meaning.

Files that would need touching at rename time: `analysis/fpad/stage2_gate.py`,
`wacv/STAGE2_GATE.md`, `wacv/stage2_gate.json`, the `gate:` block in
`training/config/fpad/FPAD_CONFIG.yaml`, and the `--stage2` flags in `main_table.py`.

## Others to watch

* **"rung"** for B0-B4 — fine internally, but the paper should say "configuration" or just name
  them; "ladder"/"rung" is informal.
* **"arm"** for ordinary/preservation — acceptable (standard in trial design), but "variant" or
  "condition" is more usual in vision papers.
* **"derisking"** — appears in the brief; avoid in the paper.
