# Stage 1 — the measurement record

Four axes kept separate: **value**, **availability**, **reliability**, **source**, with evidence direction (`p_fake`, `vacuity`) carried alongside rather than folded into any of them. One row per (sample, cue).

Written to `logs/tbiom/stage1/record/record_<dataset>.parquet`.

## Coverage

| dataset | rows/cue | availability rate | reliability known |
|---|---:|---:|---:|
| FFpp_val | 22,396 | 1.0000 | 0% |
| VALmix | 40,142 | 1.0000 | 0% |
| CDFv2 | 16,572 | 1.0000 | 0% |
| CDFv3 | 173,295 | 1.0000 | 0% |
| DFD | 109,298 | 1.0000 | 0% |
| DFDC | 132,116 | 1.0000 | 0% |
| DFDCP | 17,222 | 1.0000 | 0% |
| DFEval24 | 23,209 | 1.0000 | 0% |

## The three axes, honestly

**Availability: 1.0000 mean across every cue and dataset.** Every cue is available on every sample. The axis is correctly populated from the measurement process and it is *constant*. That is a real property of this data, not a bug, and it has a consequence the writeup must carry: this system is never observed handling an unavailable cue, so Stage 4's missing-measurement result is about deliberately constructed conditions.

**Reliability: 0% known, by construction.** Nothing in the frozen model estimates measurement quality under operating conditions. The column is present and null rather than imputed from vacuity or from the score — either would collapse reliability back into the axes it is supposed to be separate from, which is the exact error this stage exists to prevent. Stage 2 fills it.

**Source: branch-level, with a shared-trunk marker.** `semantic` and `artifact` both carry `source_trunk = clip`; `fsvfm` is its own trunk. `source_region` is `full_face_crop` for all three because that is the truth — there are no regions in this architecture. The trunk field is the one that does work: Stage 4 measured that duplicating a branch raises fused certainty +0.0110 with no AUROC gain, and the CLIP-sharing pair is the standing instance of that in the live model.

**Value is complete for two cues of three.** `semantic` carries `norm_semantic`; `artifact` carries `recon_err`, `resid_norm` and `kl_manifold`. `fsvfm` carries no scalar: its measurement is the 1024-d feature and its norm is structurally constant (LayerNorm, cv 6.6e-08), so no scalar was invented for it. Any Stage 3 mechanism reading `value` must handle a cue whose value field is legitimately empty.
