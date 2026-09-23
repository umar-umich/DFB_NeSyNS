# PCR6+ outcome

Section 9 of `tbiom/docs/pcr6plus-fusion-brief.md`. Pre-registration: `tbiom/configs/pcr6p_fusion_preregister.yaml`, written before any candidate was scored.

## Terminal outcome: 4. **no demonstrated benefit**

AUROC noninferiority holds in **4 of 4** configurations; the calibrated operating-point safeguards hold in **1 of 4**; the conflict-by-source score beats every simple score in **1 of 4**. A benefit appearing in one configuration of four is reported as what it is — the configuration that happened to favour it — and not as a demonstrated benefit.

Outcome 6, independent confirmation pending, **also applies and applies unconditionally**: no untouched set remains. CDFv2 and DeepfakeEval2024 informed development, and DF40 has informed operator design in three consecutive studies.

## 1. Detection — the declared trade

| configuration | Δ v3.1 clean vs Dempster-R (95% CI) | noninferior (lb > −0.005) | Δ JPEG q40 | Δ resize 0.60 | Δ reenactment | FPR rise | TPR loss |
|---|---|---|---:|---:|---:|---:|---:|
| `42` | -0.0014 [lb -0.0040] | **yes** | +0.0008 | -0.0008 | +0.0025 | -0.003 ok | +0.002 ok |
| `T1337` | +0.0002 [lb -0.0016] | **yes** | +0.0014 | -0.0000 | +0.0010 | +0.013 **breach** | -0.034 ok |
| `armB42` | +0.0004 [lb -0.0010] | **yes** | -0.0008 | +0.0002 | -0.0009 | -0.004 ok | +0.029 **breach** |
| `armB1337` | -0.0008 [lb -0.0025] | **yes** | -0.0008 | -0.0013 | +0.0014 | -0.004 ok | +0.015 **breach** |

## 2. Reliability — the requirement for any uncertainty claim

The conflict-by-source score had to beat **each** of fused vacuity, total conflict K, predictive entropy and margin on development error-detection AUROC, with the paired interval excluding zero.

| configuration | conflict-by-source | vacuity | K | entropy | margin | beats all four | E-AURC vs vacuity |
|---|---:|---:|---:|---:|---:|---|---|
| `42` | **0.7432** | 0.7343 | 0.6727 | 0.7508 | 0.7508 | **no** | 0.06662 vs 0.07190 (improves) |
| `T1337` | **0.8047** | 0.7502 | 0.7062 | 0.8087 | 0.8087 | **no** | 0.05134 vs 0.07807 (improves) |
| `armB42` | **0.7089** | 0.7190 | 0.7107 | 0.7794 | 0.7794 | **no** | 0.10463 vs 0.10091 (worse) |
| `armB1337` | **0.8409** | 0.7398 | 0.7535 | 0.8311 | 0.8311 | yes | 0.05283 vs 0.09959 (improves) |

Entropy and margin are numerically identical as error rankers here, and necessarily so: on a binary frame both are strictly monotone functions of |p − 0.5|, so they induce the same ordering. They are reported separately because the pre-registration named both.

**The requirement is not met.** The conflict-by-source model is beaten by entropy in three of the four configurations. It does improve selective prediction over fused vacuity (lower E-AURC in three of four), but the pre-registration required it to beat every simple score, not the weakest one, precisely so that this kind of partial result could not be presented as a benefit.

## 2b. Why it fails — the research question, answered

Section 1 of the brief asked whether conflict attributed to its source views predicts fused errors better than vacuity, entropy or margin. The conflict case analysis on development answers it directly, and the answer is no, for a reason that is visible rather than statistical.

| case (configuration `42`) | videos | K_TS | K_AO | K_all | PCR6+-R correct |
|---|---:|---:|---:|---:|---:|
| temporal right, both spatial wrong | 362 | 0.4194 | 0.0072 | 0.0142 | 0.6552 |
| temporal wrong, both spatial right | 520 | 0.3740 | 0.0076 | 0.0106 | 0.6472 |
| all three agree and right | 1512 | 0.0324 | 0.0038 | 0.0095 | 1.0000 |
| all three wrong | 214 | 0.0439 | 0.0092 | 0.0135 | 0.0018 |

Temporal-versus-spatial conflict is high in **both** disagreement cases — 0.4194 when the temporal view is the one that is right, and 0.3740 when it is the one that is wrong. The quantity marks that the views disagree; it carries almost no information about *which* of them to believe, and a score built on it inherits that. Predictive entropy already captures the part that does predict errors — how confidently the fusion committed — which is why entropy wins in three of the four configurations.

K_AO and K_all are an order of magnitude smaller than K_TS in every case, so the decomposition by source pair adds little beyond the total: on this system essentially all conflict is temporal-versus-spatial.

## 3. What the PCR6+ correction changes

| configuration | Δ v3.1 (PCR6+ − PCR6) | Δ DF40 family-balanced | Δ DF40 reenactment, Celeb-DF |
|---|---:|---:|---:|
| `42` | +0.0001 | +0.0020 | +0.0070 |
| `T1337` | -0.0000 | +0.0006 | +0.0019 |
| `armB42` | +0.0001 | +0.0005 | +0.0014 |
| `armB1337` | +0.0004 | +0.0004 | +0.0009 |

The correction is not cosmetic on this data. It is worth stating why it can matter here at all: the reliability discount routinely drives a view's weight to zero, and a zero-weighted view is exactly the vacuous source whose neutrality PCR6 violates and PCR6+ restores. The largest effect is on the reenactment half, where the appearance view is the one being discounted away.

## 4. Decisions

1. **Dempster-R remains a baseline row only.** It is the DS standard and the DiCoME-family reference. It is not the proposed method and is not reinstated as one by this study's outcome.

2. **No uncertainty novelty claim is made.** Conflict attribution by source pair failed its pre-registered test. It would in any case have been a narrow application claim: Discounted Belief Fusion (AISTATS 2025) already decomposes conflict per source pair, and uses it for more — to discount the fusion itself, not merely to rank errors (`PCR6P_PRIOR_ART_AND_CLAIMS.md`).

3. **No fusion-operator novelty claim is made.** PCR6+ is Dezert, Dezert and Smarandache's rule and is credited as theirs throughout. What this study contributes is the application and the measurement.

4. A further honest caution for the write-up: our `Averaging-R` baseline is essentially the conflictive opinion aggregation of Xu et al. (AAAI 2024), which is a published method rather than a naive control, and it is the strongest arm on several cells.

