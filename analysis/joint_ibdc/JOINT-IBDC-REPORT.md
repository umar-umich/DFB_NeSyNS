# Joint IBDC report

Brief: `tbiom/docs/discern-v2-joint-ibdc-brief.md`. Code in `training/joint_ibdc/`, outputs in
`analysis/joint_ibdc/`. The deployed configuration under test is **`42`**: warm+gate epoch 19 spatial
views with `vjepa2_lora_B_seed42`, held in **DISCERN_Ext** (`eval_adaptation/configs/warm_dicome_gate_seed42.yaml`);
the temporal branch and the fusion live in **DFB_NeSyNS**. No deployed config or checkpoint was modified.

`u = K/S` is called **evidential vacuity** throughout, never epistemic uncertainty.

## Verdicts

| item | verdict | evidence | one line |
|---|---|---|---|
| Fallback check | **three views** | `tbiom/reports/EXP_PCR6P_DEVELOPMENT.json` | Only one of the two required conditions holds, so the two-view fallback does not trigger. |
| Task 1, fusion equivalence | **PASS** | `analysis/joint_ibdc/task1_fusion_equivalence.json` | The torch fusion reproduces the deployed one exactly, 0.0 maximum absolute difference against a 1e-6 tolerance. |
| Task 1, unit tests | **PASS** | `analysis/joint_ibdc/task1_tests.txt` | 11 of 11 assertions, including the original IBDC's shared-ignorance failure reproduced by driving a gradient step. |
| Task 2, feature cache | TODO(run) | -- | Needs GPU; sequenced behind the I-JEPA extraction. |
| G0a, premise | **PASS** | `analysis/joint_ibdc/gate_g0a.json` | Conflict adds to vacuity when predicting the fused system's errors: +0.0312, 95% CI [+0.0201, +0.0424]. |
| G0b, activity | TODO(run) | -- | Blocked on Task 2. |
| Two-case diagnostic | **done** | `analysis/joint_ibdc/two_case_diagnostic.json` | Both directions of complementarity are large, and they split by forgery family. |
| Collapse check after J1 | TODO(run) | -- | Blocked on Task 2. |
| J2b pass rule | TODO(run) | -- | Blocked on Task 2. |

## The fallback check, and why the system identity mattered

The brief says to run two views if the PCR6+ "video plus one spatial view" row is within −0.005 of
the three-view system **and** Appearance has zero trust weight in at least 4 of 5 folds. On the
deployed configuration only the first holds.

| condition | deployed `42` | armB42 (NOT deployed) |
|---|---|---|
| T+OM minus three-view, development v3.1 | +0.0005 | +0.0000 |
| folds with Appearance weight 0 | **3 of 5** | 5 of 5 |
| fallback | **does not trigger** | would trigger |

`PCR6P_REFERENCE_MANIFEST.md` section 3 states that `42` is the deployed warm+gate epoch 19 spatial
path and that the armB tags are a different **training recipe**, not another seed of it. Reading
armB42 as the deployed system would have built this entire study on two views.

## Task 1

The torch fusion in `training/joint_ibdc/fusion_torch.py` is written independently of the deployed
numpy implementation rather than wrapping it, so the comparison is a cross-check and not a function
compared with itself. `test_ibdc.py` additionally asserts that the independently derived tuple
enumeration and keeping indexes agree element-for-element with the deployed spec.

| trust triple | max abs diff, mass | p | u |
|---|---|---|---|
| (0.25, 0.25, 1.0) | 0.0 | 0.0 | 0.0 |
| (0.25, 0.0, 1.0) | 0.0 | 0.0 | 0.0 |
| (1.0, 1.0, 1.0) | 0.0 | 0.0 | 0.0 |
| discount recomputed in torch from (p, u) | 2.2e-16 | -- | -- |

1,000 development videos, all five fitted per-fold triples plus the undiscounted case. Tolerance 1e-6.

## G0a, premise

Error AUROC on out-of-fold development opinions, 5-fold source-grouped cross-validation, mean over
the five fitted splits.

| feature set | error AUROC | pooled |
|---|---|---|
| 1. u_f alone | 0.7278 ± 0.0120 | 0.7232 |
| 2. kappa alone | 0.6283 ± 0.0653 | 0.6416 |
| 3. u_f and kappa | 0.7482 ± 0.0279 | 0.7545 |
| 4. u_f and the three kappa_ij | 0.7584 ± 0.0341 | 0.7648 |

**set 3 − set 1 = +0.0312, 95% CI [+0.0201, +0.0424]** over 1,000 source-grouped bootstrap draws.
The interval is above zero, so **G0a passes**.

### The finding the headline hides

The gain is not uniform across folds. It tracks whether Appearance survives the discount.

| split | Appearance weight | kappa alone | set 3 − set 1 |
|---|---|---|---|
| 0 | 0.25 | 0.6780 | +0.036 |
| 1 | 0.25 | 0.6800 | +0.035 |
| 2 | 0 | 0.5286 | −0.009 |
| 3 | 0 | 0.5714 | −0.013 |
| 4 | 0 | 0.6832 | +0.054 |

When Appearance is discounted to zero it is vacuous, so every kappa_ij containing it is identically
zero; sets 3 and 4 then collapse to the same two features and kappa alone falls towards chance. In
splits 2 and 3 conflict adds nothing at all. Since IBDC acts on kappa, this bounds how much the loss
can do in the folds where the discount has already removed a view, and it is the same 3-of-5 fact
that decided the fallback. It should be read alongside G0b, which measures how often the hinge is
active at all.

## Two-case diagnostic

Development pool, each view at its own FF++ validation EER threshold.

| view | tau | accuracy |
|---|---|---|
| T (video) | 0.5430 | 0.7098 |
| APP | 0.6395 | 0.7571 |
| OM | 0.3842 | 0.7613 |

- Video view right while **both** spatial views are wrong: **771 videos, 13.26%**
- Both spatial views right while the video view is wrong: **1,103 videos, 18.97%**

| source | n | video only | spatial only |
|---|---|---|---|
| CDFv3valG | 1664 | 245 | 389 |
| DF40train | 704 | **244** | 80 |
| FFpp_val | 700 | 8 | 14 |
| FFIWval | 500 | 30 | 96 |
| CDFv2val | 450 | 21 | **168** |
| DFDCPval | 450 | 30 | 148 |
| DFEval24val | 450 | 75 | 70 |
| DS11val | 448 | 47 | 69 |
| DS20val | 448 | 71 | 69 |

The two directions are not symmetric by family. On DF40train, which is reenactment-heavy, the video
view uniquely rescues 244 videos against 80 the other way, about 3 to 1 in its favour. On CDFv2val,
which is face swap, it is 21 against 168, about 8 to 1 the other way. Neither view dominates; they
fail on different forgery families, which is the complementarity the fusion exists to exploit.

## Still to run

Task 2's feature cache needs GPU and is sequenced behind the I-JEPA extraction. G0b, the collapse
check, all four runs (J0, J1, J2a, J2b), the λ sweep and every metric table depend on it.

## Task 2, cache equivalence: two views PASS, the temporal view FAILS

The brief says to stop if the deployed heads on the cache do not reproduce the deployed per-view
opinions to 1e-5. Two of the three views meet it; the third does not.

| view | reference | videos | max abs diff p | max abs diff u | verdict |
|---|---|---:|---:|---:|---|
| Appearance | development table | 700 (FF++ val) | 1.433e-08 | 1.013e-08 | **PASS** |
| Off-manifold | development table | 700 (FF++ val) | 1.047e-08 | 1.174e-08 | **PASS** |
| Appearance | development table | 1350 (VALmix) | 7.204e-08 | 3.427e-08 | **PASS** |
| Off-manifold | development table | 1350 (VALmix) | 1.869e-08 | 3.848e-08 | **PASS** |
| **V-JEPA 2 video** | `opinions/FFpp_val.csv` | 700 | **5.303e-04** | **1.045e-03** | **FAIL** |

### What it is not

- **Not instability in the re-extraction.** Two separate processes at batch 8 and batch 32 produce
  features differing by 3.815e-06, and both differ from the stored reference by the same 5.303e-04.
- **Not batch size.** Matching the deployed `BATCH = 32` changes nothing.
- **Not the autocast boundary.** Applying the head under bf16 autocast, as the deployed export does,
  makes it worse (1.074e-03), not better.
- **Not a stale checkpoint.** `opinions/FFpp_val.csv` was written 2026-09-17 11:49 and `best.pt` at
  11:24 the same morning, epoch 6.
- **Not the cache design.** The same kind of path reproduces the two spatial views to 1e-08.

The most likely explanation is bf16 kernel or library drift between the 2026-09-17 export and today,
through a frozen ViT-L encoder. That is consistent with everything above but it is an inference, not
something measured, and it is recorded as such.

### What it costs

| quantity | cached | deployed | difference |
|---|---:|---:|---:|
| FF++ val AUROC | 0.990395 | 0.990402 | −6.4e-06 |
| Spearman of the two rankings | | | 0.99994678 |
| decisions differing at tau 0.3 / 0.5 / 0.7 | | | 0 / 0 / 0 of 700 |

The median absolute difference is 1.9e-05, so the 5.3e-04 is a tail rather than the typical case, and
no video changes side at any threshold tried.

**This is reported as a gate failure rather than absorbed.** The numbers say the practical cost is
nil, but the brief set 1e-5 and the temporal view misses it by a factor of fifty, so the study does
not proceed past it on my judgement alone. The options are to accept the tolerance for the temporal
view with this evidence attached, to re-export the deployed temporal opinions from the cache so the
whole study is internally consistent at the cost of no longer matching the historical file, or to
investigate the 2026-09-17 environment further.

### The matched draws behave as designed

FF++ c23 train, V-JEPA 2 view, 3,600 videos, three draws. Where two draws happened to pick the same
setting for a video, the cached features are **bit-identical**; where they differ, the features move.

| pair | videos with the same setting | max feature diff there | videos with different settings | median diff there |
|---|---:|---:|---:|---:|
| 0 vs 1 | 783 | **0.000e+00** | 2817 | 0.186 |
| 0 vs 2 | 820 | **0.000e+00** | 2780 | 0.184 |
| 1 vs 2 | 797 | **0.000e+00** | 2803 | 0.184 |

The agreement rate, 21.8%, is what arm B's grid predicts: 0.4^2 + 6*0.1^2 = 0.22. So the forward is
deterministic, the drawn setting is the only thing varying between draws, and the augmentation is
doing real work rather than perturbing the features at the noise floor. Video order and labels are
identical across all three draws.


## Resolution: the temporal opinions are re-exported from the cache

The decision on the failed gate was **option 2**. The video view's development opinions are now
derived from the feature cache by applying the frozen head, so all three views in this study come
from one object. The deployed CSVs were **not** overwritten; they remain the reference of record for
the studies that already used them. New files live in `logs/joint_ibdc/opinions/`.

**The original check is now circular, and is replaced rather than re-passed.** Once the reference is
cache-derived, "the cache reproduces the reference" is true by construction and says nothing. What
can still be measured is the distance from the historical 2026-09-17 export, so that is what is
reported.

| set | videos | max abs dp | median abs dp | AUROC cache | AUROC historical | Spearman |
|---|---:|---:|---:|---:|---:|---:|
| VALmix | 1350 | 8.724e-04 | 4.261e-05 | 0.742139 | 0.741996 | 0.999994 |
| VALmix-v2 | 3764 | 8.754e-04 | 4.150e-05 | 0.821466 | 0.821441 | 0.999998 |
| FF++ val | 700 | 5.303e-04 | 1.883e-05 | 0.990395 | 0.990402 | 0.999947 |

VALmix-v2 contributes 3,764 videos rather than its 4,852 manifest videos because the protocol's
out-of-scope subsets are filtered; 1350 + 3764 + 700 = 5814, which is the development pool exactly.

### What the substitution changes

In the development table the spatial columns are **bit-identical** and only the video view moves:
`p_T` by at most 8.754e-04 (median 3.704e-05) and `u_T` by at most 1.388e-03 (median 7.329e-05).

G0a was re-run on the substituted table, because its verdict had been computed on the historical one
and a conclusion carried over without rechecking is a conclusion nobody has tested.

| G0a | video view | verdict | set 3 − set 1 | 95% CI |
|---|---|---|---:|---|
| `gate_g0a.json` | historical export | PASS | +0.0312 | [+0.0201, +0.0424] |
| `gate_g0a_cached.json` | cache-derived | PASS | +0.0312 | [+0.0201, +0.0424] |

Identical to four decimals, with the same fitted weights in every fold and thresholds moving by about
3e-04. The premise gate does not depend on which temporal opinions are used.

**Scope of this change.** It applies to the joint-IBDC study only. The I-JEPA branch study continues
to use the deployed temporal opinions, since it does not rebuild that cache, and the two must not be
quietly mixed.

## Prior art: joint training has already collapsed the views once here

`tbiom/reports/EXP_JOINT_ABLATION.md` (Experiment 6) fine-tuned the whole staged system together and
measured exactly the quantity the collapse check watches.

| arm | Δ drop APP | Δ drop OM |
|---|---:|---:|
| staged (this study's J0) | +0.0013 | **+0.0102** |
| joint, seed 42 | **−0.0004** | **+0.0003** |
| joint, seed 1337 | **−0.0005** | **−0.0004** |

The brief stops the study after J1 if a view with nonzero trust weight has a leave-one-out
contribution at most zero in J1 but positive in J0. On this precedent Appearance triggers that at
both seeds and Off-manifold triggers at seed 1337, with seed 42's +0.0003 barely above zero.
`DECISIONS_AND_OPTIONS.md` records the outcome already: *"not adopted: no gain, and it removes the
disagreement DS-R exploits."*

**It is a strong prior, not a foregone conclusion.** Experiment 6 trained CLIP LoRA, the β-VAE
projector, the temporal LoRA and every head. J1 here trains ONLY the evidential heads on frozen
cached features, which is far more constrained, so the collapse may not reproduce. It is recorded
here so that a J1 collapse is read as a reproduction of a known effect rather than a surprise, and so
that the absence of one is recognised as the more interesting result.

### The same document settles the trust-weight ambiguity

The brief says trust weights are "fixed at the deployed staged values during training", but those
values are fitted on DEVELOPMENT and differ per fold, so there is no single deployed value to fix
them at without importing development information into training. Experiment 6 met the same fork and
resolved it explicitly: it trained against the **equal-trust** fusion, "since DS-R's weights are
fitted on development, so they are kept out of the training objective".

This study follows that. **Equal trust in the training objective; weights refit out of fold on the
five development folds for evaluation**, which is what the brief's next sentence prescribes anyway.
G0b reports the per-fold deployed triples alongside equal trust, because the choice is not cosmetic:
discounting a view towards zero makes it vacuous, and a vacuous view contributes no conflict at all,
so kappa depends on which reading is taken. This is recorded as a reading of an ambiguous
instruction, not as the instruction.

### Option B, verified across both caches

Draw 0 of FF++ c23 train, spatial and temporal caches compared directly rather than each compared to
the rule.

| check | result |
|---|---|
| frames cross-checked against their clip's setting | **115,198 / 115,198 agree** |
| videos present in the spatial cache but missing a clip | 0 |
| videos whose frames carry more than one setting | 0 |
| frames per video | 32.0 over 3,600 videos |
| mix | 41.1% none, 9.3-10.5% each of six severities |

So a video's frames and its clip see the same processing, which is what makes "per augmentation
level" in G0b a well-defined breakdown. Pinned by `test_draws.py`.

### The cache gate extends to test sets, and a subset rule both studies must apply

The equivalence check was run on development sets first, but the test-set caches are where the
reported numbers come from, so it is run there too against the deployed per-frame exports in
`logs/tbiom/stage23/warmgate_sel_<set>.csv`.

| set | videos matched | worst abs diff | verdict |
|---|---:|---:|---|
| DFD | 3,431 / 3,431 | 2.830e-08 | PASS |
| DFDM | 1,896 / 1,896 | 2.642e-08 | PASS |
| DFDCP | 654 / 654 | 8.529e-08 | PASS |

**The deployed test tables drop three kinds of subset**, from `dsr_system.py:77`:

```python
keep = ~d.subset.str.contains("IP_LAP") & (d.subset != "fake-audio-driven") & ~d.subset.str.startswith("adv-")
```

IP_LAP is lip-sync-only, `fake-audio-driven` is KoDF's audio-driven split, and the `adv-` subsets are
KoDF's adversarial variants. Any test-set number computed WITHOUT this filter is not comparable to
the deployed one, however correct it looks on its own. Full coverage above (3431/3431, 1896/1896,
654/654) is what confirms the filter was applied to the reference and the cache consistently.

This applies to the I-JEPA study too: its cache holds every frame of KoDF and CDFv3 including these
subsets, because the exclusion belongs at analysis time rather than extraction time. Its test tables
must apply the same rule.
