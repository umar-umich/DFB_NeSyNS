# Where a Mamba (selective state-space) block would actually earn its place

Prepared 2026-09-19 for UMAR to circulate. Every claim below is tied to a number this project measured, not to what
the Mamba papers report. Reports cited: `tbiom/reports/EXP3B_TEMPORAL_PANEL.md`, `EXP_THREEVIEW_ABLATION.md`,
`EXP_UNCERTAINTY.md`, `EXP_DFDCP_DIAGNOSIS.md`, `EXP_JOINT_ABLATION.md`, `EXP_FREQBLENDER.md`,
`EXP_ACQUISITION_PILOT.md`, `FINAL_TABLE.md`.

## 1. The finding that should drive this decision

**Our "temporal" branch does not use time.** Shuffling the 16 frames of a clip before the V-JEPA 2 LoRA branch
changes almost nothing:

| set | ordered AUROC | shuffled AUROC | Δ | per-video score correlation |
|---|---:|---:|---:|---:|
| CDFv3 | 0.9263 | 0.9271 | −0.0009 | 0.82 |
| KoDF | 0.9656 | 0.9633 | +0.0023 | 0.95 |
| FF++ val | 0.9904 | 0.9869 | +0.0035 | 0.94 |

The frozen probes said the same for every family we tried (V-JEPA 2 ordered 0.7129 vs shuffled 0.7219; X-CLIP 0.6752
vs 0.6745; TimeSformer 0.6476 vs 0.6483; VideoMAE 0.6817 vs 0.6712). So the branch that carries our reenactment
performance is, empirically, a **set-of-frames appearance model with a video backbone**, and the token mean-pooling in
`train_branch.Branch.forward` is where the ordering goes.

That is the opening for a sequence model — and also the warning. Two readings, and they demand different work:

- **(a) The architecture throws order away.** Mean-pooling over V-JEPA's spatio-temporal tokens is an aggregation
  bottleneck; a selective SSM over the token sequence would keep it. Cheap to test.
- **(b) The data has no order left to use.** Our stored crops are ≤32 frames at **0.25–0.6 s spacing** — deliberate,
  because for most OOD sets we only ever had crops. Real manipulation dynamics (flicker, blend-boundary jitter, blink
  and micro-expression artefacts) live at 25–30 fps. At half-second spacing they are gone before any model sees them.

**Do not skip the test that separates (a) from (b).** It is a one-day experiment and it decides whether any Mamba
variant can work: re-extract dense clips (32 consecutive frames at native fps) for FF++ train and for the sets where
raw video still exists — FF++, Celeb-DF v2/v3, DeepfakeEval-2024 and the GenD releases have videos on disk; **DFDC
does not**, so at least one benchmark stays sparse whatever we build. Train the *existing* branch on dense clips and
re-run the shuffled control. If ordered − shuffled stays ≈ 0 at 30 fps, reading (b) holds and no sequence model will
save it; if a gap opens, reading (a) holds and the Mamba options below have something to model.

## 2. Options, ranked by what our evidence says they would buy

### Option A — Mamba as the temporal aggregator (replaces mean-pooling) — *best if the dense-clip test passes*

Bi-directional Mamba over V-JEPA's token sequence in time-major order, the evidential head reading the final state
instead of the mean. Linear in sequence length, so dense clips become affordable where attention would not be.

- **Why it could work here:** it attacks the exact bottleneck the shuffle test exposes, and it is the only option whose
  benefit is *measurable in advance* (the dense-clip test is its own go/no-go).
- **Interpretability we could actually claim:** a per-timestep state gives evidence accumulation over time — "the
  branch commits at frame k" — and a natural segment-level attribution that our current single pooled vector cannot
  produce. That is a figure DiCoME cannot draw.
- **Risks:** if reading (b) holds, this is a null result. Dense re-extraction changes the evaluation substrate, so
  every baseline row would need re-running on the same clips to stay comparable — budget that, it is the real cost.
- **Cost:** re-extraction ~1 day; branch training ~2 h/run; full re-evaluation ~1 day.

### Option B — Mamba for *time-varying reliability discounting* inside DS-R — *best novelty-to-risk ratio*

Today DS-R discounts each view by a **constant** weight per split (T 0.25, APP 0.25, OM 1.0), fitted on development
folds. Replace the constant with a small SSM that emits a reliability weight **per segment** from label-free inputs
(the view's own vacuity trajectory, branch disagreement, and the quality descriptors we already cache: blur, motion,
blockiness, luma, contrast), then discount and combine with the unchanged Dempster operator.

- **Why it could work here:** our worst measured cell is processed video — DFDCP at JPEG q40 (0.820), and Experiment 3
  traced that to the *branches* degrading, not the fusion. A static weight cannot express "this view is unreliable on
  this compressed segment". Experiment 7 also showed quality descriptors are cheap and already computed.
- **Novelty:** this is the strongest separation from DiCoME. It keeps the conflict-aware evidential story that is our
  contribution and makes the discounting *dynamic and input-conditioned* — "learned temporal reliability discounting
  for Dempster–Shafer fusion" is a claim with a mechanism, not a component swap.
- **Guard rails from our own results:** Experiment 6 showed that training the views jointly flattens complementarity
  (leave-one-out for OM fell from +0.0102 to +0.0003), and logistic stacking then catches DS-R. So the SSM must
  predict *weights only*, with the branches frozen, or we lose the disagreement the fusion exists to exploit. And
  Experiment 5 is the cautionary tale on metrics: FreqBlender improved development AUROC by +0.0092 with the CI above
  zero in 5/5 splits and was still refused because false positives rose +0.024. Any new fusion must report FPR/TPR at
  the frozen τ beside AUROC or it will not survive its own criteria.
- **Cost:** no re-extraction, no backbone training — cached opinions plus the quality cache. Pilot in ~2 days.

### Option C — Mamba as a sequence fusion over per-frame multi-view opinions — *risky, and it costs us our story*

An SSM reading the per-frame (p, u) of all three views and emitting a video-level opinion.

- **Against it:** it replaces the operator that is our novelty with a learned black box, and our numbers say the
  learned route is the fragile one — logistic stacking already loses to DS-R by 0.053 v3.1 under JPEG q40 (0.7564 vs
  0.8096), and Experiment 3 found the logistic comparator needed a clean+processed refit that DS-R did not. Expect a
  clean-data win and a robustness loss, which is the opposite of the paper's claim.
- Worth keeping as a **comparator**, not as the proposal.

### Option D — Mamba as a channel-mixing refiner on the off-manifold residual — *weak*

No measured gap points here. The β-VAE residual view is already our strongest single branch (0.8955 ten-set mean).
Adding a block without a deficit to fix is the kind of change reviewers read as component shopping.

## 3. What I would actually do, in order

1. **Dense-clip test (1 day, decides everything).** Re-extract 32 consecutive frames at native fps for FF++ train and
   the sets with raw video; retrain the *existing* branch; re-run the ordered-vs-shuffled control. This is a publishable
   result either way: "video foundation models on sparsely sampled face crops are order-blind" is a finding our panel
   already supports across four families.
2. **Option B pilot in parallel** (no GPU dependency, cached inputs), pre-registered with the same discipline we used
   for Experiments 5 and 7: frozen acceptance criteria, a matched non-learned control (the current constant weights), a
   null control, and FPR/TPR reported beside AUROC.
3. **Option A only if step 1 opens an order gap.** Then it is a real temporal model rather than a component swap.

## 4. Honest positioning against DiCoME

What already separates us, all measured: a video-level branch fused with frame-level views; conflict-aware DS with
reliability discounting that beats logistic stacking most where it matters (processing); fused vacuity as the best
error ranker (0.781 clean, 0.807 on pooled clean tests); a sealed-benchmark confirmation (FaceShifter 0.9549, above
every single view); and four pre-registered negative results — acquisition RL, joint training, residual operators,
FreqBlender — that a reviewer can check. Ten-set mean 0.9155 against 0.8945 for the best external baseline.

A Mamba block does not add to that by existing. It adds if it either (i) unlocks temporal order that we can *show* is
currently unused — which needs the dense-clip test first — or (ii) makes the evidential discounting input-conditioned,
which is the part of the framework that is ours. Everything else is a component swap, and component swaps are what
reviewers at TBIOM and CVPR ask "why" about.
