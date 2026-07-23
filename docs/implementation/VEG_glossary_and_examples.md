# VEG — Glossary and Worked Examples
Companion to the deck and speaker notes. Part 1: every short form, spelled out. Part 2: one image walked through the whole pipeline, showing what each intermediate artifact actually looks like.

---

## Part 1 — Full forms

### Our framework
| Short | Full form | One-line meaning |
|---|---|---|
| VEG | Verified Evidence Graph | The graph whose nodes are only causally certified claims |
| CGT | Counterfactual Grounding Test | Repair the cited region, re-query, check whether the verdict flips |
| CGA | Causal Grounding Audit | Running the CGT over many MLLMs' claims to measure grounding rates |
| NM | Necessity Margin | Size of the p(fake) drop when the cited region is repaired |
| PN | Probability of Necessity | Pearl's formal name for the quantity NM estimates |
| NeSy | Neuro-Symbolic | Neural perception combined with symbolic (rule-based) reasoning |
| DBaGNet / DISCERN | (our prior papers) | Dissertation arc: prediction → calibration → verification (VEG) |

### Training methods
| Short | Full form | One-line meaning |
|---|---|---|
| DPO | Direct Preference Optimization | Fine-tuning from preference pairs (A is better than B), no reward model, no sampling loop. What we use: certified claim ≻ rejected claim |
| GRPO | Group Relative Policy Optimization | An online reinforcement-learning method: sample a group of responses, score each with a reward, push the model toward above-average ones. What Saliency-R1 uses; we avoid it because each reward call would cost a live intervention |
| RL / RLHF | Reinforcement Learning (from Human Feedback) | Learning from reward signals rather than labels |
| SFT | Supervised Fine-Tuning | Ordinary fine-tuning on input→target pairs |
| LoRA | Low-Rank Adaptation | Cheap fine-tuning that trains small added matrices, leaving the base model frozen |
| ILP | Inductive Logic Programming | Learning symbolic rules from examples (deferred; rules are hand-written first) |

### Models and detectors
| Short | Full form | Role for us |
|---|---|---|
| MLLM | Multimodal Large Language Model | The proposer (e.g. InternVL, Qwen-VL): sees the image, writes claims |
| VLM / LVM | Vision-Language Model / Large Vision Model | Umbrella terms; MLLM emphasizes the language side |
| CLIP | Contrastive Language-Image Pre-training | The pre-training behind most SOTA deepfake detector backbones |
| PE-Core | Perception Encoder (Core) | Meta's newer CLIP-class vision encoder; optional third backbone |
| ForAda | Forensic Adapter | CLIP-based detector, one of our four frozen instruments |
| FSFM | Face Security Foundation Model | Self-supervised face model; second frozen detector |
| Effort / GenD | (method names) | Third and fourth frozen detectors |
| NPR | Neighboring Pixel Relationships | Candidate frequency-domain detector for the spectral instrument |
| FreqNet | Frequency-aware Network | Alternative candidate frequency detector |
| DCT / FFT | Discrete Cosine / Fast Fourier Transform | The frequency representations spectral features live in |
| DINO | self-DIstillation with NO labels | Self-supervised vision backbone (appears in related work) |
| SAM3 | Segment Anything Model 3 | Promptable segmenter; gated pilot for grounding vague region phrases |

### Metrics
| Short | Full form | One-line meaning |
|---|---|---|
| AUC | Area Under the ROC Curve | Threshold-free detection quality; 0.5 = chance, 1.0 = perfect |
| ROC | Receiver Operating Characteristic | The true-positive vs false-positive trade-off curve |
| pAUC | partial AUC | AUC restricted to a low false-positive band (e.g. @10% FPR) |
| EER | Equal Error Rate | Where false-accept rate equals false-reject rate |
| ECE | Expected Calibration Error | How far confidence is from accuracy; needs temperature scaling first |
| CHAIR | Caption Hallucination Assessment with Image Relevance | Borrowed from captioning; in TriDF: fraction of mentioned artifacts not in the human annotation |
| Cover | Coverage | TriDF: fraction of annotated artifacts the model did mention |
| IoU | Intersection over Union | Mask overlap measure (SAM3 gate, localization scoring) |
| MSE / PSNR | Mean Squared Error / Peak Signal-to-Noise Ratio | Pixel-level difference measures |
| LPIPS | Learned Perceptual Image Patch Similarity | Perceptual difference; used to match corruption strength to repair strength |
| κ (kappa) | Cohen's kappa | Agreement corrected for chance (parser gate: κ ≥ 0.70) |
| macro-F1 | macro-averaged F1 score | Attribution metric, averaging F1 over classes equally |
| FPR / TPR | False / True Positive Rate | Standard error rates |

### Data
| Short | Full form | Role |
|---|---|---|
| FF++ | FaceForensics++ | Core paired dataset; fakes with their source videos |
| DF / F2F / FS / NT | Deepfakes / Face2Face / FaceSwap / NeuralTextures | The four FF++ manipulation types (NT is localized: mouth region) |
| CDF | Celeb-DF (v2) | Hard cross-dataset face-swap benchmark |
| DFDC / DFDCP | Deepfake Detection Challenge (/ Preview) | Large diverse cross-dataset benchmarks |
| FFIW | Face Forensics In the Wild | Multi-person in-the-wild benchmark (Case C, out of certification scope) |
| DF40 | (benchmark of 40 deepfake generation methods) | Cross-generator attribution and transfer protocol |
| DD-VQA | Deepfake Detection Visual Question Answering | FF++-based, human-annotated reasons per region; explanation baseline |
| DiffSwap / BlendFace / CSCS | (three face-swap generators from TriDF) | Regenerated paired set; CSCS = dual-surrogate identity-preserving swapping |
| GT | Ground Truth | The true label / true mask / true paired real |

### Prompting, parsing, process
| Short | Full form | Meaning |
|---|---|---|
| OEQ / Type-B | Open-Ended Question, format B | TriDF's elicitation prompt: verdict + titled artifact findings, without revealing the label |
| K = 4 | claim cap | At most 4 claims tested per image, taken in order of listing |
| T = 0 | temperature zero | Deterministic decoding for reproducibility |
| Poisson blend | (Poisson image editing) | Seamless compositing used for the repair operation |
| PCH | Pearl Causal Hierarchy | L1 seeing / L2 doing / L3 imagining (counterfactuals); the CGT realizes an L3 quantity |
| SCM | Structural Causal Model | The formal machinery behind the PCH |
| ASG | Ask-Solve-Generate | Self-consistency-reward training paper; contrast for our causal reward |
| STE | Solver Token Entropy | ASG's gate against reward degeneracy (we don't need it: labels are external) |
| IRB | Institutional Review Board | Human-subjects approval; reason the user study was cut |
| TIFS | IEEE Transactions on Information Forensics and Security | Primary target venue |
| WACV | Winter Conference on Applications of Computer Vision | Optional lean variant venue |
| TBIOM | IEEE Transactions on Biometrics, Behavior, and Identity Science | Earlier-considered extension venue |

---

## Part 2 — One image through the pipeline (worked example)

Frame: `FF++ / Deepfakes / 371_367 / frame 000` (the Effort strip on slide 5). Values marked * are from the real pilot; the MLLM text is an illustrative example of the format.

**Step 0 — Detector verdict (frozen).**
```
detector: forada     p_fake = 0.918*    calibrated_conf = high
```

**Step 1 — Proposer raw output (Type-B).** The MLLM sees only the image and the Type-B prompt:
```
Decision: Fake.
Artifact Findings:
1. Blending seam: a soft transition line is visible along the jaw and
   lower cheeks where skin tone changes abruptly.
2. Waxy texture: the skin across the nose and mouth area looks overly
   smooth, lacking pores.
3. Lifeless eyes: the eyes appear slightly asymmetric and glassy.
4. Lighting: overall illumination looks inconsistent with the studio
   background.
```

**Step 2 — Parsed claims (predicates).** The parser maps free text to the codebook:
```json
[
 {"id": "c1", "artifact": "Blending Artifacts",     "location": "Neck/Jaw Region"},
 {"id": "c2", "artifact": "Unnatural Texture",      "location": "Nasal/Oral Region"},
 {"id": "c3", "artifact": "Unnatural Gaze",         "location": "Eyes Region"},
 {"id": "c4", "artifact": "Lighting Inconsistency", "location": null}
]
```
c4 has no locus → cannot be repaired → **untestable** (goes to the graph as UNKNOWN, counted in the untestable fraction).

**Step 3 — Dual-instrument certification (per claim).**
```
c1 spatial → repair jaw region w/ paired-real pixels → p_fake 0.918 → 0.14   Δ = +0.78  → CERTIFIED
     controls: matched corruption Δ=+0.02 · wrong region Δ=+0.01 · real-repair offset 0.03  → all inert ✓
c2 spatial → repair nose+mouth → p_fake 0.918 → 0.31                Δ = +0.61  → CERTIFIED
c3 spatial → repair eyes region → p_fake 0.918 → 0.90               Δ = +0.02  → REJECTED
     (the eyes may genuinely look odd, but they are not what drives this verdict)
c4 → untestable
```
A spectral example from another image, for the second instrument:
```
c5 {"artifact": "Grid/Checkerboard spectrum", "location": "whole_face"}
   spectral → suppress checkerboard peaks → freq-detector p_fake 0.88 → 0.22  Δ = +0.66 → CERTIFIED
   (the CLIP detector would have moved ≈ 0 on this same intervention — why the second instrument exists)
```

**Step 4 — Graph nodes (what enters the VEG).**
```yaml
- node: c1
  predicate: BlendingArtifacts(NeckJaw)
  status: certified
  necessity_margin: 0.78        # estimate of the probability of necessity
  instrument: forada (CLIP-family, GT paired repair)
  provenance: {proposer: internvl-2.5, parser: pinned-open-llm, seed: 17}
- node: c2
  predicate: UnnaturalTexture(NasalOral)
  status: certified
  necessity_margin: 0.61
- node: c3
  predicate: UnnaturalGaze(Eyes)
  status: rejected              # kept for the audit tables, excluded from reasoning
- node: c4
  predicate: LightingInconsistency(?)
  status: untestable            # UNKNOWN; reported, never guessed
edges:
- co_verified: [c1, c2]         # both flip under the same class of intervention
```

**Step 5 — DPO preference pairs (training arm input).** Built automatically from Step 3:
```
image 371_367:  chosen = c1 ("blending seam along the jaw ...")
                rejected = c3 ("eyes appear asymmetric and glassy ...")
image 371_367:  chosen = c2   rejected = c3
```
The proposer is LoRA-tuned to prefer emitting claims like c1/c2 over claims like c3. Detectors, instruments, parser, and rules never change.

**Step 6 — Solver output (what the user sees).**
```
VERDICT: FAKE  (detector, calibrated confidence: high)
CERTIFIED RATIONALE:
  • Blending artifact at the jaw line — repairing this region alone
    collapses the fake score (necessity 0.78).
  • Unnatural skin texture around the nose and mouth (necessity 0.61).
NOT RELIED UPON: the cited eye anomaly did not affect the verdict.
UNVERIFIABLE: one lighting claim could not be localized.
```
And the abstention case, when every tested claim is rejected:
```
VERDICT: FAKE (detector)   EXPLANATION: ABSTAIN —
no proposed evidence survived causal testing; rationale withheld
rather than reported unverified. Flag: rationale_ungrounded.
```

That last block is the deployable promise in one screen: the verdict is never weakened, and every sentence of the explanation is backed by an intervention or explicitly declined.
