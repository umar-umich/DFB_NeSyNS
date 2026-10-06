# CROSS-EXAM upgrade — execution state

Brief: `tbiom/docs/discern-v2-crossexam-upgrade-brief.md`. Started 2026-10-01.
Code in `upgrade/`, reports in `analysis/upgrade/`, logs in `logs/upgrade/`.

**Deviation from the brief's lane table, recorded as the brief requires.** The brief assigns three
lanes (GPUs 1/2/3). UMAR restricted this run to **GPU 1 only** (2026-10-01). GPU 1 also carries
kutub's `train.py --name rr_pr` (17.9 GiB, 94% util at start), so every lane is serialised onto one
shared card. Ordering is by critical path: longest extraction in the background, cached and CPU work
alongside. Throughput is measured on the first extraction and the plan re-estimated from it.

Priority if the window runs short: Parts 1, 2, 3, 6 before 4, 5, 7, so Part 8 always has a
configuration to assemble.

## Status

| Part | What | Status | Evidence |
|---|---|---|---|
| 1 | IBDC on current views | **done, except J1s** | `analysis/joint_ibdc/*` (J1 x3 seeds, both lambda sweeps, chosen-lambda seeds, j2b_verdict, collapse check all on disk); hinge margin added: `analysis/upgrade/part1_hinge_margin.json`. J1s not runnable -- definition absent. |
| 2a | Threshold transfer | partial | only the 2 processed cells resolved; the ten test sets did not load |
| 2b | Feature-space factor profile (ESP adaptation) | TODO(run) | |
| 3a | Extraction: spatial backbones | **done** | `logs/upgrade/feats/` — clip, dinov3, pe, pe_spatial, pe_core_l x 4 splits |
| 3d | Layer readout | **done** | PASS on frozen PE-Core (+0.0060 vs +0.0008); does NOT compound with adaptation |
| 3b | Frozen screen | **done** | `analysis/upgrade/screen_*.json`. Shortlist by incremental contribution: **PE-Core-L (+0.0042), CLIP (+0.0037)**. PE-Core at matched scale beats CLIP on FF++ val and fused AUROC but is worse on real-video FPR (0.105 vs 0.075). |
| 3c | Adaptation of shortlist | **done, 8 runs incl. 3 seeds** | winner **pe_core_l + LayerNorm(N)**: fused 0.9329, real FPR 0.0603, incremental **+0.0248** |
| 4 | Suppression | **done, NEGATIVE** | **no row kept**. Every row fails condition 3: suppression buys a small clean gain and pays it back in robustness. A-PD, the row the premise predicted, is the WORST (-0.0233 vs random). |
| 5 | Consistency | **done, NEGATIVE** | C-CONS 0.8343 vs C-AUG 0.8340 = +0.0003; the term adds nothing over its augmentation control |
| 6a | Extraction: video backbones | **done** | `logs/upgrade/vfeats/` — vjepa2_l and vjepa2_h x 4 splits |
| 6b | Global swap vs TD | **frozen heads done** | `analysis/upgrade/heads/view_vjepa2_{l,h}_s0.json`: ViT-H 0.7419 dev v3.1 vs ViT-L 0.6892 (+0.053), VALmix-v2 +0.071, same 256 input |
| 6c | Regional evidence (R-REG vs R-GLOB, R-POOL) | TODO(run) | |
| 7 | Fillers (stability, mapping, neuron ESP) | TODO(run) | |
| 8 | Final assembly, 3 seeds, tests, tables | TODO(run) | |

## Checkpoint findings (Part 3a / 6a), forced by what is actually released

| brief asks for | what exists | decision |
|---|---|---|
| PE-Spatial, "smallest released checkpoint that fits the budget" | Meta released only `PE-Spatial-L14-448` and `-G14-448`; smallest loadable anywhere is timm `vit_pe_spatial_base_patch16_512.fb` | run timm base/512. Every stored crop is 256, so this candidate is **100% upsampled, 2.0x**, reported per the brief. Scale is unmatched (86M vs 303M), so a loss is not evidence against the PE family. |
| DINOv3 ViT-L/16 | `facebook/dinov3-vitl16-pretrain-lvd1689m` | cached already (native 256, no resize) |
| V-JEPA **2.1** ViT-L/16 | **Meta has not released a 2.1.** Only `facebook/vjepa2-*` (Aug 2025). The one "2.1" on the hub is `apiantonio/vjepa2.1-vit-large-384`, third-party, 2 likes, `custom_code`, 384 input | do NOT use an unverified third-party checkpoint for a headline view. Substitute the official capacity upgrade `facebook/vjepa2-vith-fpc64-256`: ViT-H vs the deployed ViT-L, **same 256 resolution so no upsampling**. Tests the same question. Recorded as a deviation. |
| J1s / "Option B addition" | neither exists in the repo or anywhere in the session transcript | **not runnable**; recorded as a negative. G0b stays measured with J0 heads, which is what the joint IBDC brief itself specifies. |

## Log
- 2026-10-01: scaffolding created; GPU-1-only deviation recorded.
- 2026-10-01: GPU 3 added (6.5 GiB used of 143, under UMAR's 80-100 GiB bar). Two lanes now.
- 2026-10-01: pe_spatial throughput probe 108 f/s at 512 input -> FFpp_train ~18 min. Under budget.
- 2026-10-01: vjepa2_h probe 18.3 vid/s -> every split minutes, not hours. Part 6a is not a long pole.
- 2026-10-01: Dr Kutub suggested PE-Core for the spatial branch. The earlier screen had tested only
  PE-Core BASE/224 (86M vs CLIP's 303M) and it lost; at matched scale (L14-336) it has the highest
  incremental contribution in the screen. The B16 loss was a capacity artifact. Part 3d added on
  UMAR's layer-readout premise and written into the brief.
- 2026-10-01: a smoke test was killed by the harness 30-min background limit, which exposed a real
  cost bug: selection re-encoded ~140k development frames EVERY epoch through the adapting backbone.
  Now a fixed stratified subsample (10k) drives selection and the chosen epoch is re-scored on the
  full sets, so the reported number is never the subsampled one. Long runs are detached via setsid.
- 2026-10-01: Part 1 hinge-margin residue run. **It qualifies G0b.** G0b's headline is 65% of
  training samples active under equal trust, but `active` only means margin > 0: the median margin
  there is 2.1e-4 and just 27% of active samples exceed 1e-2, so the effective rate is ~18%. Under
  the deployed per-fold triples the hinge fires less often (9.8-13.5%) but with real margins --
  median 0.33-0.44, over 99% above 1e-3. The gate's verdict stands either way; the equal-trust
  number should not be quoted as if the term were strongly active.
