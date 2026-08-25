# V1_GUARDS_REPORT — the §19 correctness guards, and what each would have caught

Deliverable for build spec §26. §19 calls these non-negotiable and requires them to pass before
Umar launches. Every row below is a test that runs, not a claim about the code.

**Status: 121 tests pass, 1 skips** (the end-to-end test needs `DISCERN_TEST_REFERENCE_ARTIFACT`
pointing at a Stage-A artifact). Reproduce:

```bash
cd /data/umar/Repos/DFB_NeSyNS/training
for f in test_fsvfm test_semantic_branch test_process_branch test_ds_fusion \
         test_v1_model test_applicability_gate test_risk_model test_reference \
         test_discern_v2 test_applicability test_integration; do
  /data/umar/miniconda3/envs/dfb_nesy/bin/python networks/discern_v2/$f.py
done
```

---

## 1. Gradient guards (§19)

| requirement | test | verified behaviour |
|---|---|---|
| LoRA nonzero in Stage B | `test_semantic_branch::_gradient_flow` | after a full AdamW step, LoRA parameters moved and the head has gradient |
| base CLIP zero except LoRA | same | every non-LoRA CLIP parameter is byte-identical and received no gradient |
| a full fine-tune cannot masquerade as LoRA | `assert_lora_only` | any trainable non-LoRA CLIP parameter raises |
| adapters cannot be silently frozen too | `assert_lora_only` | zero trainable LoRA parameters raises — otherwise the anchor could not learn and nothing else would say so |
| FS-VFM zero everywhere | `test_fsvfm::_zero_gradient` | after forward/backward/step, all 303,301,632 parameters have `.grad is None` and are byte-identical, while the trainable head still learns |
| FS-VFM stays in eval | `test_fsvfm::_stays_eval` | `train(True)` cannot leave eval — a frozen ViT in train mode runs drop-path, so the same image would give different `z_ref` each step |
| `P_R` nonzero in Stage A, zero after | `test_reference` (13 tests) | the reference receives gradient only in the offline fit; under supervised training it is provably unchanged |
| VAE zero | `test_process_branch::_frozen` | no gradient, no movement, eval preserved under `train()` |
| evidence heads nonzero in B | `test_process_branch`, `test_semantic_branch` | heads receive gradient in every branch |
| experts frozen after Stage B | `test_v1_model::_stage_freezes` | leaving stage B sets `requires_grad = False` on semantic, reference, process and the control |

## 2. Mechanism guards (§19)

| requirement | test | verified behaviour |
|---|---|---|
| the reference assertion runs in the real build path | `test_v1_model::_protocol_assert` | `assert_frozen_protocol()` is called from the model and propagates a leaky branch as an error |
| `L_ref` is nonzero and in Stage A's optimizer graph | Stage A smoke run | cosine loss falls 0.997 → 0.057 over the fit; a reference that received no gradient could not move |
| the saved checkpoint contains fitted weights and statistics | `fit_reference.py` artifact | `reference_state`, `calibrator_state`, `hidden_dim`, `latent_dim`, encoder fingerprint and provenance all travel with the weights |
| the restored model reproduces reference residuals | `test_reference`, plus the Stage A → branch chain | the branch rebuilds `P_R` at the recorded width and loads it; a width mismatch fails loudly rather than loading wrong weights |
| a partly-loaded FS-VFM is refused | `test_fsvfm::_partial_load_refused` | a checkpoint missing encoder blocks raises instead of leaving a partly-random "pretrained" prior |
| an unfitted calibrator cannot pass data through | `test_process_branch::_unfitted_refuses` | raises rather than silently behaving like an identity |
| a missing Stage-A artifact cannot become a 2-branch run | `test_v1_model::_refuses_missing_artifact` | `build_v1_model` raises; a run reporting itself as three branches while running two would survive into a results table |

## 3. Evidential guards (§7, §15, §16)

| requirement | test | verified behaviour |
|---|---|---|
| `sum_k belief_k + u == 1` | `Opinion.assert_normalized`, called on every fusion | an unnormalized opinion is refused, not fused |
| all eight §15 synthetic cases | `test_ds_fusion` (20 tests) | both vacuous; anchor-only; specialist-only; agreement; disagreement; near-total conflict; `q=0`; `q=1` — all finite, normalized, with sensible limits |
| combination order does not matter | `test_ds_fusion::_order` | max deviation across all permutations < 1e-4 |
| `q = 0` removes a specialist entirely | `test_ds_fusion::_q_zero`, `test_v1_model::_gating` | fusion becomes bit-identical to the anchor alone |
| total conflict is defined | `test_ds_fusion::_total_conflict` | falls back to the vacuous opinion and flags the sample; `V = 1.0` while `C > 0.99` |
| fused evidence is a valid Dirichlet | `test_ds_fusion::_back_to_edl` | `alpha >= 1` everywhere, probabilities sum to 1 |

## 4. Leakage guards (§19)

| requirement | test | verified behaviour |
|---|---|---|
| VAL_select and VAL_meta source-disjoint | `meta_split.py::verify` | 0 identities shared, checked on the identities themselves rather than on the group ids (checking the grouping with the grouping would be circular) |
| folds within VAL_meta source-disjoint | same | 0 identities shared between any pair of the 5 folds |
| no frames from one source cross partitions | union-find grouping | every id in a video name joins one group, so `802_885` cannot leave identity 885 free to cross |
| the gate never sees a label at inference | `ApplicabilityGate.forward` | takes features only; there is no argument through which a label could arrive |
| gate features are label-free | `test_applicability_gate::_label_free` | `dloss_ref`, `generator_id`, `family`, `label`, `dataset_name` are each refused by name |
| the risk model trains on out-of-fold `q` | `test_applicability_gate::_oof_is_out_of_sample` | every sample's `q` comes from a gate that did not see it |
| no threshold is tuned on evaluation data | `test_risk_model::_policy_transfers_unchanged` | applying the frozen policy to a shifted, riskier source yields coverage **below** the budget; coverage pinned at 90% everywhere would be the signature of retuning |
| the reference is fit on a frozen encoder | `fit_reference.py::assert_frozen_feature_space` | refuses a cache whose manifest says the encoder was tuned, and refuses an unprovenanced matrix unless overridden explicitly |
| caches cannot mix encoder states | `cache_encoder_features.py::merge_manifest` | a fingerprint mismatch is refused at write time |

## 5. Pipeline-fidelity guards (§6)

| requirement | result |
|---|---|
| our FS-VFM feature matches the official downstream path | **exact**: max abs diff 0.0, cosine 1.0 against the official `models_vit` loaded by the official recipe |
| FS-VFM normalization is not ImageNet | asserted against the shipped `pretrain_ds_mean_std.txt`; a file with disagreeing lines is refused |
| each branch normalizes independently | Branch B and C both start from `raw_frames` ([0,1]) and apply their own statistics; Branch A uses CLIP-normalized `spatial_frames` |

---

## 6. Guards that fired during the build

These are not hypothetical — each caught a real defect in this repo:

1. **`assert_frozen_protocol` caught a live one.** `ProcessResidualOperator` inherits
   `nn.Module`'s default `training=True` even though its VAE is constructed in eval, so a model
   built and used without an explicit `.train()` call reported a frozen branch sitting in train
   mode. Fixed at construction; the assertion now checks the VAE's own mode as well as the
   wrapper's.
2. **The artifact-width guard.** `ReferenceEvidenceBranch` rebuilds `P_R` before loading its
   weights, and the artifact did not record `hidden_dim` — so any reference not fit at the default
   width failed to load. `hidden_dim` now travels with the weights.
3. **The domain-audit holdout (pre-pivot, still relevant).** A holdout carved from FF++ train is
   still in-sample for the *reference*, because Stage A fits on all of train. Any audit comparing
   authentic distributions must use a slice the reference never saw.
4. **The parity probe's inverted AUROC.** At 8 video groups per class both crop arms scored
   AUROC ≈ 0.0 — consistently inverted, i.e. the probe separating identities rather than
   manipulations. The script now refuses to report a gap in that regime.
