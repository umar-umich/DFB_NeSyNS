# PCR6+ reference manifest

Reuses the grouped-fusion manifest where it is still accurate and says which parts were reused; everything specific
to this study -- the four detector configurations and their own development exports, the operator sources, the
historical PCR6 labelling -- is recomputed here from the live tree.

Repository commit `e58c751f53d21c1d99a2bb7f318be4de4b2fe92a`; working tree DIRTY at audit time (6 paths).

## 1. Reused from the grouped-fusion manifest

`GROUPED_REFERENCE_MANIFEST.md` is reused unchanged for: the branch order and the shared CLIP path (§2), the opinion algebra verified on the caches — 0 rows outside u/2 <= p <= 1-u/2 on all three views (§3), the five source-grouped folds and the DF40 source-grouping rule (§4), and the exposure history (§5). Nothing in those sections changed between the two studies; the checkpoints and caches are the same files, and their hashes are re-verified below.

## 2. The discount, stated exactly

Reliability enters as **evidence scaling**, e'_k = r * e_k, applied in the evidence domain and converted back to an opinion. This is the discount the project deploys and it is kept throughout this study.

It is **not** classical belief-mass (Shafer) discounting, m'(A) = r m(A) for A != Theta and m'(Theta) = 1 - r(1 - m(Theta)). The two agree at the endpoints r = 1 (both the identity) and r = 0 (both produce the vacuous opinion), and differ at every intermediate r. A worked comparison at r = 0.25 is in `REFERENCE_MANIFEST.md` §2, where the two differ by up to 0.29 in p.

The shared r = 0 endpoint is what makes the PCR6+ neutrality property load-bearing here: a view discounted to zero is vacuous under either convention, and PCR6+ is the rule under which a vacuous view is exactly a removed view (`EXP_PCR6P_AUDIT.md` §3).

## 3. The four detector configurations and their own development exports

| tag | spatial views | temporal branch | development exports used to fit the discount |
|---|---|---|---|
| `42` | deployed warm+gate, epoch 19 | `vjepa2_lora_B_seed42` | `warmgate_e19_*` — present |
| `T1337` | deployed warm+gate, epoch 19 | `vjepa2_lora_B_seed1337` | `warmgate_e19_*` — present |
| `armB42` | SBI arm B seed 42, v3.1-selected epoch 8 | `vjepa2_lora_B_seed42` | `sbiarmB_e08_*` — present |
| `armB1337` | SBI arm B seed 1337, v3.1-selected epoch 3 | `vjepa2_lora_B_seed1337` | `sbiarmBs1337_e03_*` — present |

**Arm B is a different training recipe, not merely a different seed.** Within arm B the seed is the only difference between seed 42 and seed 1337, but arm B differs from the deployed warm+gate recipe in more than a seed, and the four configurations are described that way wherever they are compared. The deployed recipe exists at one seed and this task forbids training another.

A configuration's reliability weights are fitted on **its own** development exports. An earlier run of the grouped study fitted the arm-B weights on the deployed system's development exports, because `dsr_system.load` hardcodes the `warmgate_e19` spatial tag; that was corrected in commit `e58c751` and this study inherits the corrected loader.

## 4. Checkpoint and cache hashes

| artefact | sha256 (16) |
|---|---|
| T, V-JEPA 2 LoRA B seed 42 | `6f0d9c63a6c052de` |
| T, V-JEPA 2 LoRA B seed 1337 | `cdb205184f70ceab` |
| APP + OM, warm+gate epoch 19 | `fdc924a7cc9e1579` |
| spatial, armB42 (`discern_ext-best-epoch=08-val_auroc_video=0.9951.ckpt`) | `bc2ca7e4d0b73513` |
| spatial, armB1337 (`discern_ext-best-epoch=03-val_auroc_video=0.9946.ckpt`) | `1a64945f6677a350` |

## 5. Operator sources

| operator | source | status |
|---|---|---|
| PCR6, direct S-source, eq. (15) | Dezert, Dezert & Smarandache, *Improvement of Proportional Conflict Redistribution Rules of Combination of Basic Belief Assignments*, JAIF 16(1), June 2021, pp. 48-73 | **obtained and read** — PDF sha256 `d87ae493e7c409d1b194ed701b6fe746`, https://isif.org/files/isif/2024-01/last.pdf |
| PCR6+, direct S-source, eq. (26), with the binary keeping index of eqs. (22)-(24) | same paper | **obtained and read**; implemented from the equations, not from a summary |
| Dempster, Yager, evidence averaging | as implemented in the grouped study | reused unchanged |

PCR6+ is established prior work by Dezert, Dezert and Smarandache and is credited as such everywhere in this study. It is not presented as a new operator.

## 6. Historical PCR6 labelling

The existing `PCR6-R` rows in `EXP_FUSION_OPERATORS.md`, `EXP_DF40_OPERATORS.md` and the grouped reports were produced by folding the two-source rule over the three views, not by the direct three-source formula. PCR6 is not associative, so these are different operators; they differ by up to 2.082e-01 on random admissible opinions. Those reports are left unchanged, and every table in this study labels the fold **`PCR6-pairwise-R`** and the direct rule **`PCR6-R`**.

## 7. Folds, threshold rule, grid

Five existing source-grouped folds (`GroupShuffleSplit` on the source id, seeds 0-4) and the v3.1 macro objective over swap / reenactment / in-the-wild cells. Threshold: EER on FF++ val, computed per arm on its own scores — the frozen rule already in the protocol. Reliability grid: (0, 0.25, 0.5, 0.75, 1) per view with max = 1, the existing grid and candidate cap, unchanged.

## 8. Exposure

Unchanged from `GROUPED_REFERENCE_MANIFEST.md` §5 and repeated because it governs every claim here: CDFv2 and DeepfakeEval2024 informed development; DF40 has already informed operator design in two prior studies; FaceShifter was read once after the recipe was frozen. **No untouched confirmation set remains**, so outcome 6 of the brief — independent confirmation pending — applies to every result in this study regardless of what else it finds.

