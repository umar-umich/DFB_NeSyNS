# Stage 2 — MECHANISM GATE

**BORDERLINE — re-run before committing the paper's framing**

preservation improves the forensic-minus-domain gap on 5 of 6 layers (median +0.0276), and clears the 0.02 bar on 3 of 6 — which misses the "more than half" rule by 0 layer. A margin that small does not distinguish the two outcomes.

Students: `logs/fpad/studentA_ordinary_seed42` (lambda_preserve = 0.0) vs `logs/fpad/studentA_preserve_seed42` (lambda_preserve = 1.0), epoch 1, matched: **True**.

Domain axis: **Celeb-DF-sourced DF40-Dev reals** (2841 frames). In-domain side: FF++ val (515 real / 2045 fake). OOD fakes: 9959.

> The FF++ compression axis the brief also names is **unavailable** — only c23 is staged on this machine (`wacv/REPO_MAP.md` item 8). This audit is therefore single-axis, on clean OOD reals. Recorded so the gate's basis is not overstated.

## Per-layer audit — the table the decision rests on

`gap` is forensic-minus-domain separability: positive means the delta separates manipulation better than it separates corpora.

| layer | gap (ordinary) | gap (preserve) | improvement | verdict (ordinary) | verdict (preserve) |
|---|---:|---:|---:|---|---|
| layer 4 | -0.1370 | -0.0978 | +0.0393 **←** | DATASET DETECTOR | DATASET DETECTOR |
| layer 8 | +0.0297 | +0.0447 | +0.0150 | BORDERLINE | BORDERLINE |
| layer 12 | +0.1138 | +0.1297 | +0.0159 | pass | pass |
| layer 16 | +0.1081 | +0.1640 | +0.0559 **←** | pass | pass |
| layer 20 | +0.1408 | +0.1163 | -0.0245 | pass | pass |
| layer 24 | -0.0713 | -0.0180 | +0.0533 **←** | DATASET DETECTOR | DATASET DETECTOR |

## Does the delta separate manipulation with NO domain shift?

In-domain (FF++ val real vs fake) separability. A delta that only works across corpora is reading the corpus.

| layer | ordinary | preserve |
|---|---:|---:|
| layer 4 | 0.5588 | 0.5289 |
| layer 8 | 0.5003 | 0.5627 |
| layer 12 | 0.6648 | 0.5689 |
| layer 16 | 0.7707 | 0.8207 |
| layer 20 | 0.9673 | 0.9649 |
| layer 24 | 0.7846 | 0.9522 |

## Aggregates and diagnostics

| quantity | student | domain sep | forensic sep | gap | verdict |
|---|---|---:|---:|---:|---|
| `d_mean` | ordinary | 0.6280 | 0.5407 | -0.0873 | DATASET DETECTOR |
| `d_early` | ordinary | 0.5044 | 0.6436 | +0.1391 | pass |
| `d_late` | ordinary | 0.6272 | 0.5442 | -0.0831 | DATASET DETECTOR |
| `patch_top10_share` | ordinary | 0.6462 | 0.6264 | -0.0198 | DATASET DETECTOR |
| `p_direct` (the B1 readout) | ordinary | 0.6094 | 0.6395 | +0.0301 | BORDERLINE |
| `d_mean` | preserve | 0.5117 | 0.6513 | +0.1396 | pass |
| `d_early` | preserve | 0.6522 | 0.6447 | -0.0075 | DATASET DETECTOR |
| `d_late` | preserve | 0.5835 | 0.6132 | +0.0297 | BORDERLINE |
| `patch_top10_share` | preserve | 0.5083 | 0.6195 | +0.1112 | pass |
| `p_direct` (the B1 readout) | preserve | 0.6236 | 0.6601 | +0.0365 | BORDERLINE |

The frozen-teacher and student-only terms are expected to fail the audit by construction and are reported for the design justification, never depended on. `d_cls_*` is the CLS diagnostic; the primary readout is the patch mean.

## Provenance cost of the domain axis

* Celeb-DF-v2 **real** videos read: **178** (2841 frames)
* Celeb-DF-v2 fakes read: **0**
* Anything fit on them: **False**

> Celeb-DF-v2 REAL frames reached through DF40-Dev's borrowed authentic half were read as the domain axis of the Stage-2 mechanism gate. No Celeb-DF-v2 fake was read and nothing was fit on them. Its final-table row is therefore not strictly zero-shot on the real side, and should be footnoted as such rather than presented as unseen.

## What happens next

Do NOT commit the framing on this. Re-run at the final Stage-A epoch on the full scoring set: this read is one layer from flipping, and both the epoch count and the sample cap are things that move it. Report the per-layer table either way — it is informative regardless of which side the verdict lands on.
