# Two requested result tables

Built only from cached outputs by `analysis/tbiom/paper/two_tables.py`. Nothing was trained. Where a number was never computed it is marked **not computed**, never estimated.

---

## Table 1 — acquisition study

**Split.** The five source-grouped v3.1 **selection folds** (the development pool: VALmix, VALmix-v2 and FF++ val). Videos per fold: 2903, 2843, 2872, 2849, 2968 (mean 2887); 1364 source groups. Every row is scored on the same videos with the same extra-clip budget as every other row of its budget.

**Fusion and threshold.** The deployed DS-R fusion with weights fitted on each fold's fitting half, and $\tau$ = EER of the no-acquisition score on that fold's FF++ val videos. Acquired windows are aggregated within a branch (the `conf` rule) before fusion, so overlapping clips are never treated as independent sources.

| policy | extra clips / video | AUROC | $\Delta$ vs no extra clip | 95% CI on $\Delta$ | FPR @ $\tau$ |
|---|---:|---:|---:|---|---:|
| no extra clip | 0 | 0.9068 | +0.0000 | [+0.0000, +0.0000] | 0.107 |
| random extra clip | 1 | 0.8938 | -0.0130 | [-0.0172, -0.0088] | 0.161 |
| fixed single window $w_0$ | 1 | 0.8939 | -0.0128 | [-0.0175, -0.0083] | 0.152 |
| evenly spaced extra clips ($w_0{+}w_2$) | 2 | 0.8921 | -0.0147 | [-0.0197, -0.0097] | 0.172 |
| quality-based choice | 1 | 0.8973 | -0.0095 | [-0.0135, -0.0056] | 0.151 |
| learned one-step choice | 1 | 0.8952 | -0.0115 | [-0.0159, -0.0074] | 0.155 |
| learned two-step choice | 2 | 0.8919 | -0.0149 | [-0.0199, -0.0099] | 0.174 |
| oracle using the true label | 1 | 0.9382 | +0.0315 | [+0.0268, +0.0363] | 0.115 |
| oracle using the true label | 2 | 0.9302 | +0.0234 | [+0.0187, +0.0284] | 0.130 |
| oracle using scores shuffled from other videos | 1 | 0.9875 | +0.0806 | [+0.0670, +0.0956] | 0.105 |

Intervals are a paired source-group bootstrap, 2,000 draws per fold, resampling source groups with all of their linked videos, then averaged over the five folds. **These intervals did not exist in the original report** (`EXP_ACQUISITION_PILOT.md`, which gives point estimates only); they were computed here by reproducing the per-video policy scores from the cached window outputs.

Reproduction check against the published report: eight of the ten policies match to $0.00000$ AUROC. The two that differ are the only stochastic ones — the random-window policy (the published value averages several random draws, this one uses a single draw, difference $0.0008$) and the permuted-window null oracle (a fresh permutation, difference $0.0019$).

**Test sets: not computed.** The acquisition study was run on the development selection folds only. No acquisition policy was ever scored on CDFv2, CDFv3, DFD, DFDC, DFDCP, DFEval24, FFIW, KoDF, IDForge, DFDM or DF40, so there are no test-set rows to add.

**Reading.** Every deployable policy is *worse* than not acquiring anything. The only rows that gain are the two oracles, and the null oracle — which picks among candidate scores borrowed from other videos and therefore cannot know anything about the video it is scoring — gains more (+0.0807) than the true-label oracle (+0.0314). Most of the apparent headroom is selection on labels among four correlated scores, not information a policy could exploit.

---

## Table 2 — video view contribution

**Fusion rule.** PCR6+-R (direct three-source PCR6+ after evidence scaling), the manuscript's operator. **Folds.** the five source-grouped v3.1 fitting folds, with the reliability weights refitted independently for every row by the same grid and objective, and $\tau$ = EER on FF++ val computed per row. Only the view set, or the backbone supplying the video view, changes between rows.

**Sets.** The six of the configuration slide (CDFv2, CDFv3, DFD, DFDC, DFDCP, DFEval24) plus KoDF and the reenactment subset of CDFv3. If the six intended on that slide were a different set, the rows are unaffected -- only the columns shown would change.

**Why three FPR columns are identical.** CDFv2 and CDFv3 are scored against the *same 178 real videos* (the identifiers differ only by the dataset prefix; the per-video branch opinions match to $1.8$e$-7$), and the CDFv3 reenactment slice keeps all of CDFv3's reals. FPR is a property of the real videos alone, so it is necessarily identical across those three columns. Their AUROCs differ because the fake sets differ.

### 2a. Video AUROC

| row | video backbone | seeds | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | DFEval24 | KoDF | CDFv3-reen |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Appearance + Off-manifold only | -- | spatial 42 | 0.9732 | 0.8787 | 0.9341 | 0.8811 | 0.8733 | 0.6701 | 0.8543 | 0.8397 |
| \quad + face-prior view | -- | spatial 42 | 0.9705 | 0.8808 | 0.9365 | 0.8828 | 0.8780 | 0.6741 | 0.8519 | 0.8439 |
| \quad + V-JEPA 2 video view (**reference**) | V-JEPA 2 ViT-L, LoRA, B-aug | spatial 42, temporal 42 | 0.9583 | 0.9639 | 0.9307 | 0.8530 | 0.8597 | 0.7235 | 0.9600 | 0.9681 |
| V-JEPA 2 alone | V-JEPA 2 ViT-L, LoRA, B-aug | temporal 42 | 0.7383 | 0.9263 | 0.7967 | 0.7009 | 0.7408 | 0.6732 | 0.9656 | 0.9741 |
| three-view, X-CLIP as video view | X-CLIP, LoRA | spatial 42, temporal 42 | 0.9509 | 0.8868 | 0.9325 | 0.8609 | 0.8857 | 0.6666 | 0.8470 | 0.8588 |
| three-view, VideoMAE as video view | VideoMAE, LoRA | spatial 42, temporal 42 | 0.9393 | 0.9429 | 0.8758 | 0.8312 | 0.8370 | 0.7379 | 0.9414 | 0.9344 |
| three-view, TimeSformer as video view | TimeSformer, LoRA | spatial 42, temporal 42 | 0.9570 | 0.8995 | 0.9148 | 0.8595 | 0.8979 | 0.6967 | 0.8712 | 0.8795 |
| three-view, V-JEPA 2 *without* B-augmentation | V-JEPA 2 ViT-L, LoRA | spatial 42, temporal 42 | 0.9542 | 0.9596 | 0.9253 | 0.8497 | 0.8606 | 0.7033 | 0.9578 | 0.9642 |
| reference with **shuffled** video frames | V-JEPA 2 ViT-L, LoRA, B-aug | spatial 42, temporal 42 | n/c | 0.9634 | n/c | n/c | n/c | n/c | 0.9595 | 0.9664 |

### 2b. FPR at the frozen threshold

| row | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | DFEval24 | KoDF | CDFv3-reen |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Appearance + Off-manifold only | 0.139 | 0.139 | 0.140 | 0.291 | 0.353 | 0.253 | 0.036 | 0.139 |
| \quad + face-prior view | 0.134 | 0.134 | 0.131 | 0.279 | 0.347 | 0.255 | 0.033 | 0.134 |
| \quad + V-JEPA 2 video view (**reference**) | 0.040 | 0.040 | 0.123 | 0.191 | 0.179 | 0.191 | 0.026 | 0.040 |
| V-JEPA 2 alone | 0.019 | 0.019 | 0.095 | 0.150 | 0.075 | 0.148 | 0.034 | 0.019 |
| three-view, X-CLIP as video view | 0.055 | 0.055 | 0.125 | 0.181 | 0.156 | 0.203 | 0.023 | 0.055 |
| three-view, VideoMAE as video view | 0.084 | 0.084 | 0.175 | 0.228 | 0.177 | 0.138 | 0.045 | 0.084 |
| three-view, TimeSformer as video view | 0.110 | 0.110 | 0.137 | 0.262 | 0.354 | 0.214 | 0.068 | 0.110 |
| three-view, V-JEPA 2 *without* B-augmentation | 0.028 | 0.028 | 0.094 | 0.114 | 0.103 | 0.144 | 0.003 | 0.028 |
| reference with **shuffled** video frames | n/c | 0.033 | n/c | n/c | n/c | n/c | 0.034 | 0.033 |

`n/c` = **not computed**. The shuffled-frame control was exported for the reference branch on CDFv3, FFpp_val, KoDF, VALmix, VALmixV2 only; the remaining columns of that row were never produced and are not estimated here.

**Seeds.** Every row uses spatial seed 42 (the deployed warm+gate checkpoint) and, where a video view is present, temporal seed 42. No row averages over seeds; there is one seed behind each.

**A caveat on the backbone-swap rows.** The reference video view is trained with B-augmentation; the X-CLIP, VideoMAE and TimeSformer branches are not. Those three rows therefore differ from the reference in two things at once, not one. The row without B-augmentation is included as the matched anchor: comparing the swapped backbones against *it* isolates the backbone, and comparing the two V-JEPA 2 rows against each other isolates the augmentation.

**Shuffling the frames costs almost nothing, and that is a real qualification.** On the sets where the control exists the fused reference moves 0.9639 $\to$ 0.9634 on CDFv3 (-0.0005); 0.9600 $\to$ 0.9595 on KoDF (-0.0005); 0.9681 $\to$ 0.9664 on CDFv3-reen (-0.0018). Scored alone the video view moves $0.9263 \to 0.9271$ on CDFv3 and $0.9656 \to 0.9633$ on KoDF, i.e. within noise in both directions, even though the individual probabilities move a long way (mean $|\Delta p|$ of $0.127$ and $0.041$).

The reference video view samples 16 frames spread over the whole video, and at that sampling it is largely order-insensitive: it is reading multi-frame evidence, not frame order. The same pattern was measured independently at branch level in `EXP_TEMPORAL_SAMPLING.md` (sparse sampling: $0.8934$ ordered against $0.8873$ shuffled), where only the *dense* 32-consecutive-frame configurations showed a large order effect ($0.9045 \to 0.8606$ pooled, $0.9049 \to 0.6890$ with a GRU head). Any claim that this system exploits temporal *order* is therefore not supported by these numbers; what the evidence supports is that it exploits information spread across frames.

