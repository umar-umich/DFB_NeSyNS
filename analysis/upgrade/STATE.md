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
| 2a | Threshold transfer vs ranking | TODO(run) | |
| 2b | Feature-space factor profile (ESP adaptation) | TODO(run) | |
| 3a | Extraction: DINOv3, PE-Spatial | DINOv3 cached; pe_spatial running (GPU 1) | `logs/upgrade/feats/`; 108 f/s at 512 |
| 3b | Frozen screen | TODO(run) | |
| 3c | Adaptation of shortlist (L and N recipes) | TODO(run) | |
| 4 | Processing suppression (MC/PD/LEACE/RND) | TODO(run) | |
| 5 | Label-preserving consistency (C-AUG, C-CONS) | TODO(run) | |
| 6a | Extraction: video backbones | running (GPU 3) | `logs/upgrade/vfeats/`; 18.3 vid/s probed |
| 6b | Global swap vs TD | TODO(run) | |
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
- 2026-10-01: Part 1 hinge-margin residue run. **It qualifies G0b.** G0b's headline is 65% of
  training samples active under equal trust, but `active` only means margin > 0: the median margin
  there is 2.1e-4 and just 27% of active samples exceed 1e-2, so the effective rate is ~18%. Under
  the deployed per-fold triples the hinge fires less often (9.8-13.5%) but with real margins --
  median 0.33-0.44, over 99% above 1e-3. The gate's verdict stands either way; the equal-trust
  number should not be quoted as if the term were strongly active.
