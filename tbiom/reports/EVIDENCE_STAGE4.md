# Stage 4 — missing-measurement and duplicate-evidence properties

Replays over the frozen arm's exported opinions. No GPU, no retraining. `tau` is the frozen FF++ val EER (0.3971).

## Test 1 — missing measurement

A vacuous opinion is the **identity element** of Dempster's orthogonal sum, so `fuse(A, B, vacuous)` must equal `fuse(A, B)` exactly. Any movement means the implementation fabricates something when a cue is declared absent. This has a right answer.

| dataset | masked branch | max \|Δp_fused\| vs dropping it | mean signed Δ (toward Fake is +) | decisions flipped |
|---|---|---:|---:|---:|
| CDFv2 | Appearance | 2.22e-16 | +6.56e-19 | 0 |
| CDFv2 | Off-manifold residual | 1.11e-16 | +4.99e-19 | 0 |
| CDFv2 | Bona-fide face prior | 1.11e-16 | +9.38e-20 | 0 |
| CDFv3 | Appearance | 2.22e-16 | +4.70e-19 | 0 |
| CDFv3 | Off-manifold residual | 2.22e-16 | +2.67e-19 | 0 |
| CDFv3 | Bona-fide face prior | 2.22e-16 | +1.49e-19 | 0 |
| DFD | Appearance | 2.22e-16 | -3.06e-19 | 0 |
| DFD | Off-manifold residual | 2.22e-16 | -2.08e-19 | 0 |
| DFD | Bona-fide face prior | 2.22e-16 | -2.33e-19 | 0 |
| DFDC | Appearance | 2.22e-16 | -2.76e-19 | 0 |
| DFDC | Off-manifold residual | 2.22e-16 | -2.50e-19 | 0 |
| DFDC | Bona-fide face prior | 2.22e-16 | +3.25e-20 | 0 |
| DFDCP | Appearance | 1.11e-16 | +4.01e-19 | 0 |
| DFDCP | Off-manifold residual | 1.11e-16 | -1.80e-19 | 0 |
| DFDCP | Bona-fide face prior | 1.11e-16 | +5.44e-19 | 0 |
| DFEval24 | Appearance | 1.11e-16 | -6.20e-19 | 0 |
| DFEval24 | Off-manifold residual | 2.22e-16 | +9.21e-19 | 0 |
| DFEval24 | Bona-fide face prior | 1.11e-16 | -8.52e-20 | 0 |

**Maximum deviation across all six sets and all three branches: 2.22e-16.** Vacuity is handled exactly as the identity element: masking a cue is indistinguishable from not having it, the decision never moves, and nothing is fabricated. Mean signed drift is +1.04e-19, so masking does not push toward Fake either — *unavailable* is not silently converted into *contradicted*.

Decisions flipped by masking-versus-omitting: **0**.

## Test 2 — duplicate evidence

The same branch opinion supplied twice. Confidence must not rise from double-counting.

| dataset | duplicated branch | mean fused certainty 1−u | duplicated | Δ | AUROC | AUROC dup |
|---|---|---:|---:|---:|---:|---:|
| CDFv2 | Appearance | 0.9845 | 0.9935 | **+0.0089** | 0.9474 | 0.9506 |
| CDFv2 | Off-manifold residual | 0.9845 | 0.9934 | **+0.0089** | 0.9474 | 0.9531 |
| CDFv2 | Bona-fide face prior | 0.9845 | 0.9955 | **+0.0110** | 0.9474 | 0.9355 |
| CDFv3 | Appearance | 0.9779 | 0.9892 | **+0.0113** | 0.8871 | 0.8829 |
| CDFv3 | Off-manifold residual | 0.9779 | 0.9894 | **+0.0115** | 0.8871 | 0.8872 |
| CDFv3 | Bona-fide face prior | 0.9779 | 0.9928 | **+0.0149** | 0.8871 | 0.8869 |
| DFD | Appearance | 0.9901 | 0.9963 | **+0.0062** | 0.9454 | 0.9467 |
| DFD | Off-manifold residual | 0.9901 | 0.9963 | **+0.0061** | 0.9454 | 0.9466 |
| DFD | Bona-fide face prior | 0.9901 | 0.9966 | **+0.0064** | 0.9454 | 0.9368 |
| DFDC | Appearance | 0.9862 | 0.9936 | **+0.0074** | 0.8906 | 0.8923 |
| DFDC | Off-manifold residual | 0.9862 | 0.9935 | **+0.0073** | 0.8906 | 0.8918 |
| DFDC | Bona-fide face prior | 0.9862 | 0.9962 | **+0.0101** | 0.8906 | 0.8850 |
| DFDCP | Appearance | 0.9798 | 0.9907 | **+0.0109** | 0.9038 | 0.9054 |
| DFDCP | Off-manifold residual | 0.9798 | 0.9900 | **+0.0102** | 0.9038 | 0.9065 |
| DFDCP | Bona-fide face prior | 0.9798 | 0.9930 | **+0.0132** | 0.9038 | 0.8936 |
| DFEval24 | Appearance | 0.9692 | 0.9869 | **+0.0176** | 0.6866 | 0.6859 |
| DFEval24 | Off-manifold residual | 0.9692 | 0.9848 | **+0.0156** | 0.6866 | 0.6853 |
| DFEval24 | Bona-fide face prior | 0.9692 | 0.9886 | **+0.0193** | 0.6866 | 0.6883 |

## Verdict

**Missing measurement: passes, exactly.** Deviation 2.22e-16 is floating-point noise. Vacuous is the identity element of the fusion, so a masked cue is arithmetically the same as an absent one. The system does not fabricate evidence and does not turn *unavailable* into *contradicted*. This is a property of Dempster's rule rather than something engineered for the occasion, and it holds by construction.

**Duplicate evidence: fails, as Dempster's rule must.** Supplying a branch twice raises mean fused certainty by **+0.0110** while mean AUROC moves -0.0012 — confidence rises without discrimination improving, which is double-counting in its purest form. The rule assumes independent sources and has no defence against the same source presented twice.

This is not a surprise and it is not a bug to fix in the operator. It is the mechanism behind a result already in the record: `semantic` and `artifact` both derive from the CLIP trunk — `artifact` is the orthogonal residual `f_s − f_c` — so the three-view fusion is *already* partly double-counting a shared source. That is why dropping `semantic` is worth +0.008 mean OOD AUROC, and why nine conflict-aware operators failed to recover it: the problem was never conflict, it was redundancy.

**What this means for the brief's Stage 1 `source` field.** The duplicate test is the reason that field matters, and it also shows what the field has to record. Not region or segment, which do not exist here — all three branches read the same aligned face crop — but *shared trunk*. A source tag that cannot express "these two cues came from one encoder" would not catch the one real instance of correlated evidence this model has.
