# Main-Track Paper — Neuro-Symbolic Attribution over Verified Predicates

**Target** TOMM special issue (Responsible and Explainable Multi-Modal Fusion) or WACV / AAAI / CVPR main track.
**One-line claim** verified forensic predicates, reasoned over by a symbolic program, give cross-generator forgery-family attribution and a checkable proof, where current methods sit near chance on unseen generators.
**Relationship to workshop** consumes the workshop's verified predicates directly. The workshop proves a single cue can be verified, the main track reasons over many verified cues toward a number with headroom.

---

## 1. The gap it fills

Real-versus-fake AUC is saturated, which is why DISCERN's machinery showed no gain there. Two numbers still have headroom: attribution on unseen generators (near chance today, Forensics-Bench) and faithful multi-step reasoning. This paper points the neuro-symbolic mechanism at attribution, a hard number, not at saturated AUC. That is the explicit fix for the DISCERN failure mode.

## 2. The two candidate targets, pick by pilot

| Target | Claim | Headroom | Risk |
|---|---|---|---|
| **A. Cross-generator attribution (preferred)** | predict forgery family / generator from the pattern of verified predicates, on generators not seen in training | high, baselines near chance | medium, depends on whether probe+predicate signatures separate generators |
| **B. Faithful reasoning engine (fallback)** | Scallop/DeepProbLog program emits a multi-step checkable proof, plus human rule-correction of confounds, plus temporal predicates for video | high on faithfulness | higher, more apparatus |

Do NOT build both. Run the attribution pilot first. If frozen CLIP features plus probe signatures separate 3-4 generators above chance, build A. If not, fall back to B.

## 3. Architecture (agent-free hybrid)

```
                 frozen contrastive encoder (GenD/ForAda/Effort backbone)
input frame  ->  neural path  ->  calibrated p(fake) + uncertainty   [owns the verdict]
                 |
                 +-> verified predicates (from workshop verifier)
                       -> symbolic program (Scallop/DeepProbLog)
                            -> forgery family + checkable proof        [owns attribution + why]
```

The neural path owns AUC, no regression risk. The symbolic overlay owns attribution and proof. In the safe version the overlay never overrides the verdict, it explains and attributes. A bounded calibrated correction is an option only if an ablation justifies it.

## 4. Where the predicate-supervised adapter lives (your VS Code idea, placed correctly)

Your captions-plus-adapter idea belongs HERE, not in the workshop, and aimed at generalization, not at the intervention test. Use verified predicates as supervision for a lightweight adapter on the frozen encoder, and measure cross-generator generalization or attribution accuracy. The claim is "training on verified cues improves generalization to unseen generators."

**Critical** faithfulness of a trained model must be checked with an INDEPENDENT test the model never saw, that is ground-truth mask overlap on held-out generators, NOT the intervention test on the trained model. Training toward the cues and then testing sensitivity to those cues is circular. Mask overlap is independent because the mask is ground truth.

## 5. Components and exact choices

| Component | Choice | Note |
|---|---|---|
| Neural backbone | frozen foundation encoder, GenD/ForAda/Effort-style | compare against these as baselines too |
| Adapter (optional, target A path) | lightweight, predicate-supervised | trained, backbone frozen, preserves generalization |
| Predicate source | the workshop's verified predicates | only confirmed cues enter the program |
| Reasoner | Scallop or DeepProbLog | deterministic program, this is the differentiation from MLLM judges |
| Attribution labels | exist in FF++, Celeb-DF, DF40 | little new data |
| Temporal predicates | target B / video only | deferred, since VLMs miss temporal cues (Beyond Static Artifacts) |

## 6. Differentiation, state precisely

- vs DeepfakeJudge (Pixels Don't Lie), a learned MLLM judge that needs reasoning labels and is itself fallible. Ours uses a deterministic program over independently verified predicates, no learned judge.
- vs Deepfake-Agent and VRAG-DFD, which keep reasoning in neural or text space, or verify by retrieval. Ours verifies by physical measurement and reasons by program.
- vs the TOMM 2023 emotion-based neurosymbolic paper, which is emotion-only over a hand-built psychological KB. Ours is general forensic predicates with a verified grounding step.
- avoid the word "verifiable" in title (VRAG-DFD owns it) and "FCG" (a DISCERN baseline).

## 7. The three contributions, each falsifiable

1. **Generalization** via frozen perception plus predicate-supervised adapter. Falsified if it loses to an end-to-end neural baseline cross-generator.
2. **Attribution and proof** from the symbolic program. Falsified if a free-text VLM matches its faithfulness.
3. **Correction** via editable symbolic rules (Stammer, Right for the Right Concept). Falsified if editing a rule does not change behavior as predicted.

## 8. Go/no-go before building

Attribution pilot. Take frozen CLIP/encoder features plus your probe outputs on 3-4 generators, check whether a simple classifier separates them above chance. Above chance, the headroom is real, build target A. At chance, fall back to target B or change direction. Do this before committing weeks.

## 9. Dissertation role

Builds the reasoning engine on the workshop's verified-grounding piece. Discharges survey RQ on reasoning. With the workshop, the thread is one sentence, move past saturated yes-or-no detection to explain honestly (workshop) and attribute reliably (main track). Both stages are committed work, which satisfies Prof Steve's no-optional-extensions rule.