# Step 3 — anchor health dashboard

Anchor: **P0-DS epoch 1** (the provisional retrainable chassis). All three readouts scored as genuine anchor candidates.

`tau` frozen once per readout on **FF++ VAL**, applied unchanged to every domain. Domains are labelled by what the anchor has seen, because "OOD" is not one thing.

## semantic (CLIP branch)   (tau = 0.6086)

| domain | provenance | videos | AUROC | EER | **FPR_real@tau** | d_RF | mean p on real |
|---|---|---:|---:|---:|---:|---:|---:|
| FF++ | in-domain (trained) | 700 | 0.9929 | 0.0277 | **0.029** | +0.659 | 0.244 |
| VALmix | development (selection only) | 1350 | 0.8818 | 0.1933 | **0.314** | +0.314 | 0.493 |
| Celeb-DF-v2 | zero-shot | 518 | 0.9564 | 0.1163 | **0.225** | +0.386 | 0.460 |
| Celeb-DF-v3 | zero-shot | 5418 | 0.8199 | 0.2422 | **0.225** | +0.268 | 0.460 |
| DFD | zero-shot | 3431 | 0.9399 | 0.1427 | **0.223** | +0.381 | 0.465 |
| DFDC | zero-shot | 4704 | 0.8812 | 0.1996 | **0.368** | +0.304 | 0.529 |
| DFDCP | zero-shot | 654 | 0.8547 | 0.2308 | **0.448** | +0.248 | 0.581 |
| Deepfake-Eval-2024 | zero-shot | 814 | 0.6909 | 0.3587 | **0.439** | +0.127 | 0.559 |

Zero-shot means: AUROC **0.8572**, FPR_real **0.321**, d_RF **+0.286**.

## artifact view   (tau = 0.3477)

| domain | provenance | videos | AUROC | EER | **FPR_real@tau** | d_RF | mean p on real |
|---|---|---:|---:|---:|---:|---:|---:|
| FF++ | in-domain (trained) | 700 | 0.9912 | 0.0214 | **0.014** | +0.658 | 0.191 |
| VALmix | development (selection only) | 1350 | 0.8789 | 0.2044 | **0.127** | +0.368 | 0.247 |
| Celeb-DF-v2 | zero-shot | 518 | 0.9685 | 0.0948 | **0.073** | +0.408 | 0.221 |
| Celeb-DF-v3 | zero-shot | 5418 | 0.8490 | 0.2075 | **0.073** | +0.292 | 0.221 |
| DFD | zero-shot | 3431 | 0.9487 | 0.1209 | **0.132** | +0.488 | 0.242 |
| DFDC | zero-shot | 4704 | 0.8764 | 0.2085 | **0.261** | +0.377 | 0.294 |
| DFDCP | zero-shot | 654 | 0.8570 | 0.2284 | **0.291** | +0.345 | 0.311 |
| Deepfake-Eval-2024 | zero-shot | 814 | 0.6712 | 0.3881 | **0.182** | +0.106 | 0.273 |

Zero-shot means: AUROC **0.8618**, FPR_real **0.169**, d_RF **+0.336**.

## fused (DS)   (tau = 0.4784)

| domain | provenance | videos | AUROC | EER | **FPR_real@tau** | d_RF | mean p on real |
|---|---|---:|---:|---:|---:|---:|---:|
| FF++ | in-domain (trained) | 700 | 0.9929 | 0.0313 | **0.007** | +0.812 | 0.126 |
| VALmix | development (selection only) | 1350 | 0.8852 | 0.1859 | **0.213** | +0.439 | 0.322 |
| Celeb-DF-v2 | zero-shot | 518 | 0.9646 | 0.0991 | **0.129** | +0.531 | 0.275 |
| Celeb-DF-v3 | zero-shot | 5418 | 0.8409 | 0.2182 | **0.129** | +0.381 | 0.275 |
| DFD | zero-shot | 3431 | 0.9421 | 0.1365 | **0.174** | +0.543 | 0.293 |
| DFDC | zero-shot | 4704 | 0.8828 | 0.2009 | **0.317** | +0.424 | 0.382 |
| DFDCP | zero-shot | 654 | 0.8573 | 0.2387 | **0.391** | +0.368 | 0.425 |
| Deepfake-Eval-2024 | zero-shot | 814 | 0.6922 | 0.3513 | **0.341** | +0.162 | 0.391 |

Zero-shot means: AUROC **0.8633**, FPR_real **0.247**, d_RF **+0.401**.

## Which readout should be the anchor

| readout | zero-shot AUROC | zero-shot FPR_real | zero-shot d_RF |
|---|---:|---:|---:|
| semantic (CLIP branch) | 0.8572 | 0.321 | +0.286 |
| artifact view | 0.8618 | 0.169 | +0.336 |
| fused (DS) | 0.8633 | 0.247 | +0.401 |

## Real-side verdict — does Step 9 trigger?

- **semantic (CLIP branch)**: FPR_real 0.029 in-domain -> 0.321 zero-shot (x11.2 if finite); d_RF +0.659 -> +0.286, retaining 43% of in-domain separation.
- **artifact view**: FPR_real 0.014 in-domain -> 0.169 zero-shot (x11.8 if finite); d_RF +0.658 -> +0.336, retaining 51% of in-domain separation.
- **fused (DS)**: FPR_real 0.007 in-domain -> 0.247 zero-shot (x34.6 if finite); d_RF +0.812 -> +0.401, retaining 49% of in-domain separation.

**Real-side health is intact.** The best readout (artifact view) holds FPR_real at 0.169 on domains whose reals it has never seen, against the collapse signature of 1.000, and retains d_RF +0.336 against the collapsed 0.000. Degradation is graded, not catastrophic. **The FF++-only corpus is fine and Step 9 is skipped**; real-support asymmetry was a property of the FF++ (+) DF40 manifest, not of FF++ training as such.
