# P3a cost plan — what to change, and what not to

Status: **proposal for review** (2026-08-12). Nothing here is implemented yet. Written so it
can be checked against other sources; the claims that matter are flagged **[VERIFY]**.

---

## 1. The correction that changes the question

P3a does **not** fine-tune the diffusion model. The pilot plan specifies a *frozen* SD UNet
(now SD 3.5's MMDiT), and every operator pilot in this suite trains only a small evidential
head on top of a frozen operator:

| pilot | trainable | frozen operator |
|---|---:|---:|
| P2a | 25,806 | 83.6M (SDXL VAE) |
| P4 | 76,066 | 46.6M (StyleGAN-D + LGrad ResNet-50) |
| P3a (planned) | ~30–80K | ~2.5B (SD 3.5 MMDiT) |

So **LoRA adapters do not apply**. LoRA reduces the cost of *fine-tuning* by shrinking the
number of updated parameters and the optimiser state. Here there is no backward pass through
the 2.5B model at all — the cost is a frozen **forward** pass, once per batch. Adding LoRA
adapters would add parameters and make the forward marginally *slower*, not faster.

This matters because it redirects the whole optimisation: the lever is "how often, and at what
resolution, do we run MMDiT", not "how many of its parameters do we update".

## 2. Where the time actually goes

Training is 20 epochs x ~900 steps = ~18,000 steps. Measured on the two pilots now running:

- P2a (83.6M frozen operator): ~50 min/epoch -> ~15 h
- P4 (46.6M frozen operator, plus an input-gradient pass): ~50 min/epoch -> ~15 h

Both share their GPU with another lab member, so these are contended numbers.

A 2.5B frozen forward is ~30x the parameters of P4's stack. Naively that suggests days, but
parameter count is a poor proxy — MMDiT at a 32x32 latent is far cheaper per parameter than a
convolutional stack at 256x256.

### MEASURED 2026-08-12 (supersedes the estimate above)

Benchmarked on GPU 1 *while P4 was training on the same card*, so these are contended numbers,
not best-case:

| batch | ms/call | img/s | peak GiB |
|---:|---:|---:|---:|
| 32 | 478.5 | 66.9 | 7.9 |
| 128 | 1831.4 | 69.9 | 16.9 |

At the training batch size of 128: **1.83 s/step x 18,000 steps = ~9.2 h of operator time**,
on top of the base training. That puts P3a at roughly the same total cost as P2a and P4
(~15 h measured), **not days**.

The earlier worry was wrong for exactly the reason flagged as uncertain: the 2.5B parameter
count badly overstates MMDiT's cost here, because it runs on a 32x32 latent (256 tokens at
patch size 2) rather than at pixel resolution.

**Consequence: none of the speed-up options below are necessary.** Section 3 is retained as a
record of what was considered and why it was not needed.

## 3. Options, cheapest first

### A. Feed the operator un-augmented images, and precompute once (recommended)

The training augmentation is stochastic (`RandomHorizontalFlip`, `RandomAffine`,
`GaussianBlur`, `ColorJitter` — `src/dataset/base.py:67`). That is what currently forbids
caching: the operator sees a different image every epoch.

But there is a substantive argument that the operator *should not* see augmented images:

1. It measures a **generation-process** signature. Gaussian blur and colour jitter attack
   exactly the high-frequency evidence AEROBLADE- and LaRE2-style residuals depend on, so
   augmentation actively destroys the signal the operator exists to read.
2. **At test time there is no augmentation** (`init_augmentations(train=False)` is Identity).
   Feeding the operator clean images in training therefore *removes* a train/test mismatch
   rather than creating one — the operator would see the same distribution in both phases.
3. The semantic/CLIP branch keeps its augmentation unchanged, so the regularisation the
   backbone relies on is untouched.

If accepted, the operator's output becomes a deterministic function of each training image, so
it can be computed **once** for all 115,198 frames and cached. Training then costs the same as
P0-DS plus a table lookup, i.e. ~1–1.5 h instead of days, and the one-off precompute is a
single inference pass (~115K frames).

Cost: one design deviation, which must be recorded in `notes.md` and the paper. It applies
equally to P2a and P4 and would speed those up too, so it should be decided once for all three
rather than per-pilot.

**[VERIFY]** the claim that feeding a frozen forensic operator un-augmented input while the
main branch stays augmented is defensible, and whether any published operator-fusion detector
does this.

### B. Reduce the operator's working resolution

SD 3.5's VAE is f=8, so a 224x224 crop becomes a 28x28 latent; MMDiT cost scales roughly with
the square of the token count. **[VERIFY]** Halving the operator's input resolution should cut
its cost ~4x with unknown effect on the signal.

### C. Single timestep, already in the plan

The pilot plan already mandates one timestep (t=100 or t=250), not a sweep. No further saving
available here without deviating.

### D. A smaller diffusion model

SD 3.5 medium is already the smaller of the two SD 3.5 releases. Going smaller means leaving
the SD 3.x family; note Stability has **removed every SD 1.x/2.x repo from HuggingFace**, so
"just use SD 1.5" now means a community mirror with weaker provenance — the reason we moved to
3.5 in the first place.

### E. Do nothing, run it long

Acceptable if the benchmark says a single step is cheap. Rejected only if it turns into days on
a shared box.

## 4. Decision (2026-08-12, after measurement)

**Run P3a as specified — full augmentation, no precompute, no resolution reduction.** The
measured operator cost (~9.2 h over a full run) does not justify any deviation, and every
option in §3 carries a documentation or provenance cost that is now unwarranted.

Remaining sequence:

1. Wait for P2a and P4 to finish (~15 h each, running concurrently).
2. Run P3a for **one epoch** with full augmentation to confirm the end-to-end per-epoch cost
   matches the 9.2 h/run projection, rather than trusting the operator-only benchmark.
3. If it matches, let it run the remaining 19 epochs.

## 5. Things deliberately NOT proposed

- **LoRA on MMDiT** — no fine-tuning happens; see §1.
- **Multi-timestep / FIRE frequency decomposition** — that is P3b, conditional on P3a's verdict.
- **Changing training hyperparameters** (seed 42, 20 epochs, batch 128, LR 1e-4) — frozen by
  the pilot contract for comparability with P0-DS and P1a–P1d.
- **Quantising the operator** — would change its numerical output, making P3a's evidence not
  reproducible against the released weights.

## 6. Naming discipline (unchanged)

P3a uses SD 3.5, whose MMDiT is a **rectified-flow transformer predicting velocity**, with a
16-channel latent. LaRE2 is defined on a diffusion UNet's epsilon-residual. By the pilot plan's
own substitution rule — the one that forces "LGrad-inspired" for any swapped component — P3a
must be described as **"LaRE2-inspired"**, never as LaRE2.

(For contrast: P4 *is* "LGrad", because both its stages are the authors' original weights; only
the executor changed, from TensorFlow to a PyTorch port.)
