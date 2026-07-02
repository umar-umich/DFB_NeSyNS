# Kill-Test Spec — Forgery-Evidence Homogeneity (FEH) axis

**Status:** pre-registered falsification test. Run this *before* writing any method.
**Goal:** decide in ~1 week whether "localized vs diffuse forgery evidence"
separates **face-swap** from **fully-synthetic** faces in a **generator-agnostic**
way. If it does, it becomes the anchor of the paper. If it does not, we stop here.

This test needs **no training** and **no generative VLM**. Stage A is pure
forensics; Stage B adds PE-Core open-vocab grounding. Both reuse existing code.

---

## 0. The hypothesis, stated so it can fail

Define a per-frame **evidence field** `E ∈ R^{P×P}` over an P×P patch grid
(P=16 for a 224px face, matching ViT-L/14 patches). Each cell holds a
training-free forgery signal (high-frequency DCT energy + SRM noise). The
hypothesis is a claim about **two scalars** of that field:

| Class           | Magnitude `μ(E)` | Concentration `C(E)` |
|-----------------|------------------|----------------------|
| Real            | low              | low (n/a)            |
| Fully-synthetic | high             | **low** (diffuse — one global generator pass) |
| Face-swap       | high             | **high** (localized — composited region vs clean host) |

- `μ(E)`  = mean patch energy → separates real from fake (expected easy).
- `C(E)`  = **Gini coefficient** of the patch-energy map (or top-10%-mass) →
  separates **localized swap** from **diffuse synthetic**. THIS is the crux.

**The claim is generator-agnostic.** If `C(E)` only separates families
*within* a dataset, it is just generator fingerprinting → **hypothesis falsified.**

---

## 1. Datasets & protocol (cross-dataset is mandatory)

Frame crops already on disk under `/data/umar/Datasets/preprocessed/`.

| Role            | FIT (calibrate) source                          | TEST (held-out) source                |
|-----------------|--------------------------------------------------|---------------------------------------|
| Real            | FF++ `original_sequences/youtube/c23`           | Celeb-DF-v2 real                      |
| Face-swap       | FF++ `manipulated_sequences/{Deepfakes,FaceSwap,FaceShifter}/c23` | Celeb-DF-v2 fake          |
| Fully-synthetic | **DiFF** (download — primary)                    | **DiffusionFace** or DeepFakeFace     |

- **Swap source** differs between FIT and TEST (FF++ → Celeb-DF).
- **Synthetic source** differs between FIT and TEST (DiFF → DiffusionFace).
- This is what makes a pass meaningful: a generator-specific cue cannot survive
  swapping *both* the swap and the synthetic generators.
- Exclude Face2Face / NeuralTextures from the "swap" bucket (they are
  reenactment, not identity-swap) — keep as a separate diagnostic bucket.
- Sample ~300–500 frames/class/dataset, balanced. Use existing `test.json`
  indices in `/data/umar/Datasets/preprocessed/FaceForensics++/`.

**Synthetic dataset download:** DiFF (Diffusion Facial Forgery, CVPR'24) primary;
DiffusionFace as the cross-source held-out. Both are frame/image-level, matching
this pipeline. Avoid GAN-only sets as the headline.

---

## 2. Evidence-field extractor (reuse existing code, training-free)

### 2a. High-frequency channel — reuse `FADFrontEnd`
`training/networks/nesy_defake/foundation_models/frequency_extractor.py`
already builds the DCT machinery. Use it deterministically:

```python
from networks.nesy_defake.foundation_models.frequency_extractor import FADFrontEnd

fad = FADFrontEnd(img_size=224, mode='single', use_learnable=False).eval()
# x: (B,3,224,224) raw [0,1] pixels
x_freq = fad._dct_forward(x.float())          # DCT
high   = fad._idct_forward(fad.filter_high(x_freq))   # high-band residual (B,3,224,224)
energy = high.pow(2).mean(1)                   # (B,224,224) per-pixel HF energy
# pool to 16x16 patch grid:
E_hf = F.avg_pool2d(energy.unsqueeze(1), kernel_size=14).squeeze(1)  # (B,16,16)
```

`use_learnable=False` → pure F3Net bandpass, fully deterministic, no params.

### 2b. Noise channel — reuse SRM from `forensic_helpers.py`
`preprocessing/forensic_helpers.py:324 extract_srm` computes SRM residuals. For a
patch map, apply the SRM kernel as a conv and pool to 16×16 (mirror 2a). Take the
per-patch **std** of the residual as the noise-energy map `E_srm`.

### 2c. Combine
`E = normalize(E_hf) + normalize(E_srm)`, per-frame min-max to [0,1].
(Start with `E_hf` alone; add `E_srm` only if it helps separation.)

### 2d. The two scalars
```python
mu   = E.mean()                                  # magnitude
# concentration via Gini of flattened patch energies:
def gini(v):
    v = v.flatten().sort().values
    n = v.numel(); idx = torch.arange(1, n+1, device=v.device)
    return ((2*idx - n - 1) * v).sum() / (n * v.sum() + 1e-8)
C = gini(E)                                       # concentration
```
Also log `top10_mass` (fraction of total energy in the top 10% of patches) and
**Dirichlet energy** `Σ_(i~j) (E_i−E_j)^2` on the 4-neighbour patch graph as
alternative concentration measures — report whichever separates best.

---

## 3. Stage A — pure forensic separability (no semantics)

1. Compute `(μ, C)` for every sampled frame in FIT and TEST.
2. **Anchor figure:** 2D scatter / density of real / swap / synthetic in the
   `(μ, C)` plane, FIT and TEST side by side.
3. Fit a shallow classifier (logistic regression or depth-2 tree) on `(μ, C)`
   using **FIT only**; evaluate on **TEST**.
4. Report:
   - Real-vs-fake AUC (sanity; expect high).
   - **Swap-vs-synthetic AUC from `(μ,C)` only** ← the decision metric.
   - 3-way balanced accuracy.
   - **within-dataset minus cross-dataset gap** (rigor check).

---

## 4. Stage B — add PE-Core grounding (still no training, no generative VLM)

Only run if Stage A is borderline (see §5), or to quantify how much semantics adds.

1. Load **PE-Spatial-L14** (`facebook/PE-Spatial-L14-448`, already a config
   option [C] in `nesy_defake_ablation4_causal.yaml`). Get dense patch features.
2. Score a **small fixed semantic predicate bank** (hand-written for the
   kill-test only — the generative VLM replaces this later) against each patch via
   cosine similarity to text embeddings, e.g.:
   - swap-leaning: `"a blended seam along the face boundary"`,
     `"a region pasted from another face"`, `"mismatched skin tone at the jaw"`
   - synthetic-leaning: `"waxy over-smooth skin"`, `"unnaturally uniform texture"`,
     `"plastic AI-generated look"`
3. Build a **semantic-evidence map** `S ∈ R^{16×16}` = max predicate similarity
   per patch.
4. **Verification (the safe step):** `V = E * 1[S > τ]` — keep forensic evidence
   only where a semantic predicate also fires. Recompute `(μ, C)` on `V`.
5. Repeat §3 with `V`. Does grounding **lift cross-dataset swap-vs-synthetic AUC**?

---

## 5. Pre-registered decision (do not move the goalposts later)

Decision metric = **cross-dataset swap-vs-synthetic AUC**.

| Result (Stage A, then B)                       | Verdict        | Action                                              |
|------------------------------------------------|----------------|-----------------------------------------------------|
| A ≥ 0.70                                        | **PASS**       | Build the full system (`grounded_abductive_plan.md`)|
| A in 0.60–0.70, **B lifts to ≥ 0.70**          | **PASS (semantics earns it)** | Build; contribution 1 (grounding) is justified |
| A in 0.60–0.70, B does **not** lift            | **WEAK**       | Axis is real but soft — reconsider scope/claims     |
| A < 0.60 and B < 0.70                           | **FALSIFIED**  | **Stop.** Homogeneity axis is not generator-agnostic|

Plus a hard rigor gate: if (within-dataset AUC − cross-dataset AUC) > 0.20, the
signal is mostly fingerprinting → treat as **FALSIFIED** regardless of absolute AUC.

---

## 6. Known risks this test must surface (report them, don't hide them)

- **High-quality blended swaps** (FaceShifter, and BlendFace-style) minimize the
  seam → may drift toward the *diffuse* region and collapse the swap/synthetic
  split. Keep FaceShifter as its own bucket; if it lands with synthetics, say so.
- **Compression (c23/c40)** homogenizes statistics — exactly where Face X-Ray
  broke. Report c23; optionally probe c40 as a stress test.
- **Inpainting / region-synthesis** is *locally* synthetic → expect it near the
  swap region. Argue that placement is arguably correct, not an error.

---

## 7. Deliverables of the kill-test
- `results/feh_killtest/scatter_fit.png`, `scatter_test.png` (anchor figures)
- `results/feh_killtest/metrics.json` (all AUCs + within/cross gap)
- A one-paragraph verdict against §5. That paragraph decides the project.
