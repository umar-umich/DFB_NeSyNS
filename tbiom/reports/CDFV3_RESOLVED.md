# The CDFv3 discrepancy — resolved, and it reverses a conclusion

## What it was

The pilot table records **P0-DS CDFv3 = 0.9516**. Re-scoring the same checkpoint on the same
5,418 videos gave **0.8409** — off by 0.11, while P1d reproduced to 0.0003 in the same run. That
asymmetry is what made it look like a measurement fault.

It is not. **CDFv3 has three manipulation families, and the pilot table's P0-DS cell was scored
on one of them.**

| family | methods | fake videos |
|---|---|---:|
| FaceSwap | BlendFace, Celeb-DF-v2, GHOST, HifiFace, InSwapper, MobileFaceSwap, SimSwap, UniFace | 1,740 |
| FaceReenact | DaGAN, FSRT, HyperReenact, LIA, LivePortrait, MCNET, TPSMM | 1,400 |
| TalkingFace | AniTalker, EchoMimic, EDTalk, FLOAT, IP_LAP, R3DPortrait, SadTalker | 2,100 |

All three share the same 178 real videos.

| arm | published | **FaceSwap only** | full CDFv3 | basis used |
|---|---:|---:|---:|---|
| P0-DS | 0.9516 | **0.9515** | 0.8409 | **FaceSwap only** |
| P1d | 0.8852 | 0.9354 | **0.8849** | **full set** |

0.9515 against a published 0.9516 is a match to 1e-4. **The table mixes bases inside a single
row**: P0-DS's cell is FaceSwap-only, P1d's is the full set. No measurement was wrong; the two
cells answer different questions and were subtracted from each other anyway.

## Why it matters more than a typo

The `mean Δ` row — the number that rejected every pilot — is a difference against that one
baseline cell. Correcting it shifts **every** pilot by the same `0.1107 / 7 = +0.0158`:

| pilot | published mean Δ | **corrected** | |
|---|---:|---:|---|
| P2a AEROBLADE | −0.0092 | **+0.0066** | beats P0-DS |
| P1d MR-VAE | −0.0127 | **+0.0031** | beats P0-DS |
| P3a LaRE2 | −0.0156 | +0.0002 | ties |
| P1a det. AE | −0.0157 | +0.0001 | ties |
| P1c WAE | −0.0195 | −0.0037 | below |
| P1b β-TCVAE | −0.0199 | −0.0041 | below |
| P4 LGrad | −0.0199 | −0.0041 | below |

**"Nothing beat P0-DS on mean AUROC" was an artifact of one mis-scoped cell.** Two pilots beat
it and two tie.

Independent cross-check: this correction predicts P1d at **+0.0031**; direct re-measurement of
both arms through one code path gave **+0.0033**. Agreement to 0.0002, from two routes that
share no arithmetic.

## The substantive finding underneath

The per-family split is not just bookkeeping — it says what P0-DS actually is.

| arm | FaceSwap | FaceReenact | TalkingFace | full |
|---|---:|---:|---:|---:|
| **P0-DS** | **0.9515** | 0.7874 | 0.7849 | 0.8409 |
| P1d (MR-VAE + rate) | 0.9354 | 0.8765 | 0.8488 | 0.8849 |
| Stage 3 (β-VAE + rate) | 0.9350 | 0.8637 | 0.8463 | 0.8804 |
| Stage 2 (MR-VAE proj) | 0.9288 | **0.8988** | **0.8958** | **0.9076** |

P0-DS is **best on FaceSwap and worst on the other two, by 8–11 points.** FaceSwap is the family
closest to its FF++ training corpus, which is face swapping. So P0-DS's apparent CDFv3 strength
was strength on the one family it was trained for, and scoring only that family is what made it
look dominant.

The rate-response and multi-rate arms give up ~0.016 on FaceSwap and gain **0.06–0.11** on
reenactment and talking-face — the modern generative families. Stage 2, which Stage 4 rejected on
mean AUROC, is the **best arm on both** of those.

## Consequences

1. **No P0-DS CDFv3 number from the pilot table may be quoted.** Use 0.8409 (full) or state
   FaceSwap-only explicitly. The same applies to any `mean Δ` computed from that row.
2. **Every published CDFv3 cell needs its basis verified**, not just P0-DS's. Only P0-DS and P1d
   could be checked here because only their exports exist; P1a/P1b/P1c/P2a/P3a/P4 are unverified
   and their corrected deltas above assume their cells used the full set, as P1d's did.
3. **Report CDFv3 per family.** A single CDFv3 AUROC averages three tasks of very different
   difficulty and hides the effect that actually distinguishes these arms.
4. **Stage 4's decision is unaffected** — all four arms there were measured through one code path
   on the full set. But its reasoning is enriched: Stage 3's CDFv3 gain over P0-DS (+0.0395) is
   almost entirely reenactment and talking-face, not face swapping.
