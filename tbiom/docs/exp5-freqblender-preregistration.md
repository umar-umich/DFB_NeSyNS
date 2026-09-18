# Experiment 5 (FreqBlender) — frozen before launch

Written 2026-09-17, **before** the FreqBlender arm was trained and before any of its outputs were scored. It fixes the
arms, the generator's provenance, the selection rule, the acceptance criteria and the disqualifier, per
`tbiom/docs/discern-v2-finish-line-brief.md` §Experiment 5.

## Question

Does adding frequency-domain synthetic fakes to the **spatial** branches help, with the temporal branch frozen?
FreqBlender is an experimental augmentation here, not an assumed novelty, and it is not part of the reference unless
it earns inclusion on development.

## The generator, and what it is not

| item | status |
|---|---|
| upstream implementation | `/data/umar/Repos/FreqBlender` @ `7c8bc7f`, **called in place, not edited or copied** |
| network, losses, DCT convention, crop, hyper-parameters | upstream (`conv_autoencoder`, `loss_function`, `face_cut`, `config/base.json`) |
| released FPNet weights | **not used** — behind a Baidu Pan account wall (audited in `freqblender-source-audit.md`) |
| FPNet used here | **trained locally** by `analysis/tbiom/freqblender/fpnet.py train` |
| authenticity teacher | **local, split-safe** ResNet-34 (`analysis/tbiom/train_authenticity_teacher.py`, FF++ train only, val AUROC 0.972, index 1 = fake), wrapped so it receives the CLIP normalisation it was trained with |
| facial-fidelity teacher | official MobileFaceNet + MTCNN shipped in the repository; face bank built from our 720 training-split real videos |
| generator training data | FF++ **train** ids only (720), real + Deepfakes pairs, both members of every pair in the same official split |
| epoch budget | 30 epochs of upstream's 100, fixed before launch: StepLR(5, 0.5) leaves the learning rate at 1/64 of its initial value by epoch 30. The loss curve is reported. |

Everything above must be reported as **a local reproduction of FreqBlender**, never as "the released FreqBlender".

The pseudo-fake is upstream's own construction, read off its authenticity loss:
`x = IDCT(DCT(real)·M1 + DCT(fake)·M2)` with `(M1, M2, M3) = FPNet(DCT(fake))` — the real face keeps its
facial-attribute band and borrows the forgery-trace band of a real manipulation; the quality band M3 is dropped.

## Amendment, 2026-09-18 — the arms changed, the criteria did not

The first attempt used `warm_dicome_gate_B_seed42` as the control and the same config with arm D as the treatment.
That pair is unusable: the warm+gate recipe sets `freeze_views: true`, so only the face-prior branch and its gate
train. The two checkpoints differ in 12 of 1,015 tensors, all in that branch; Appearance and Off-manifold come out
bit-identical, and the face prior is not part of the reference fusion. A training-data intervention cannot reach the
views the study is about, so the comparison returned Δ = 0.0000 everywhere (`tbiom/reports/void/`).

The arms below are therefore replaced by the `sbi_arm*` family, which trains the Appearance and Off-manifold views
under the same arm-B processing. **Nothing else changes**: the acceptance criteria, the disqualifier, the bootstrap
rule, the 10% fraction and the frozen temporal branch are exactly as written below, and they were written before any
FreqBlender arm was trained. The amendment is recorded here rather than folded silently into the text.

| arm | what it is | status |
|---|---|---|
| **Spatial-B** (control) | `sbi_armB_seed42`: the SBI-comparison arm B — trainable CLIP views, arm-B processing, no warm start, no freezing — at its development-selected epoch 8 (v3.1 rule, `sbiarmB_selection_v3.json`) | already trained; re-exported under the tag `fbB` so every row comes from that one checkpoint |
| **Spatial-B-Freq** | `sbi_armD_seed42`: byte-identical config with `training_arm: D` and `freq_fraction: 0.1` | training 2026-09-18 |

The reference detector's own Appearance and Off-manifold views stay frozen and untouched; this study asks whether
FreqBlender helps spatial views **that are actually being trained**, which is the question the brief poses.

## The two arms (original text, superseded above for the arm identities only)

| arm | what it is | status |
|---|---|---|
| **Spatial-B** (control) | `warm_dicome_gate_B_seed42`: the warm-start + gated detector with arm-B processing | already trained and exported; config differs from the reference run by `training_arm: B` alone |
| **Spatial-B-Freq** | `warm_dicome_gate_BFreq_seed42`: identical config with `training_arm: D`, `freq_fraction: 0.1` | to be trained |

Matched by construction: same initialisation (`init_views_from` the same DiCoME epoch-5 checkpoint), same seed, same
frozen views, same trainable modules, same losses, same optimiser, same 20-epoch budget, same class balance, same total
samples, same class-independent B processing applied **after** sample construction. The only difference is that 10% of
**fake** slots hold a FreqBlender pseudo-fake instead of a real manipulation. The fraction is fixed at 0.10; the brief
forbids sweeping it.

The β-VAE configuration and losses are untouched, and Appearance and Off-manifold share the CLIP feature path, so this
is one coupled spatial experiment, not two. The reference checkpoints are not modified.

**Temporal branch:** frozen at the reference B-augmented V-JEPA 2 LoRA checkpoint for both arms. No frame-level
pseudo-fake ever enters temporal training, and no synthetic spatial input's opinion is ever fused with a temporal
opinion from its original genuine clip: FreqBlender examples exist only inside spatial training, and evaluation fuses
the three views on the same real evaluation clips.

## Checkpoint selection

Both arms save checkpoints under the identical policy (top-3 by `val_auroc_video` plus `last`) and both are selected by
the unchanged v3.1 development rule (`analysis/tbiom/select_checkpoint_v3.py`, ties → earliest epoch) over their own
candidates. The control's candidates were {e02, e03, e09, last} and it selected **last (epoch 19)**. Each arm's
candidate set is reported.

## Evaluation

After spatial training, both arms export all three views on the same evaluation clips (real and fake videos), clean and
processed. DS-R is refitted **separately per arm** on the identical source-grouped development folds, with logistic
stacking on matched inputs as the mandatory comparator. Nothing is fitted on processed data, and no test set is read
before the development decision below is recorded.

## Acceptance criteria, frozen now, on development only

Let Δ = Spatial-B-Freq − Spatial-B at the **fused three-view level** (T frozen, DS-R primary, LR reported beside it),
on the v3.1 objective over the 5 source-grouped selection folds, with a paired source-group bootstrap (1000 draws per
split, one group resample per draw scoring both arms).

1. **Disqualifier (reenactment).** Arm C is the precedent: synthetic pseudo-fakes collapsed fused development
   reenactment to 0.527. FreqBlender is **disqualified regardless of any swap gain** if the fused development
   reenactment Δ is negative beyond the bootstrap noise band — operationally, mean Δ(reenactment) < 0 **and** the 95%
   CI upper bound < 0 in ≥ 3 of the 5 splits. This is checked at the fused level because freezing the temporal branch
   preserves its parameters but not the fused detector's reenactment behaviour: a swap-tuned spatial branch can still
   move the fused decision through DS-R.
2. **Inclusion.** FreqBlender is adopted only if it is not disqualified **and** it shows a development gain:
   mean Δ ≥ +0.005 on clean development **or** mean Δ ≥ +0.005 averaged over the two processed development conditions
   (JPEG q40, resize 0.60), **with** clean development loss ≤ 0.005 and clean development FPR at the fusion's own τ not
   worse by more than 0.02.
3. **Swaps alone do not qualify.** An arm whose swap cell improves while reenactment drops fails, even if the drop is
   inside the noise band and criterion 1 does not fire; clean swaps, reenactment, processed performance and FPR/TPR are
   reported together in one table so this is visible rather than argued.
4. **Compression robustness is not claimed** before it is demonstrated under criterion 2 on development.

Test sets, including CDFv3 reenactment, are read and reported **only after** the above verdict is recorded, and never
as the selection criterion. FaceShifter stays sealed.
