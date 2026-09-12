# SBI integration — pre-run verification

200 sampled frames from the FF++ train real split. A pass here is a correctness result, not evidence of generalisation.

## 1 — the on-the-fly path sees the h5's pixels

max abs difference over 50 frames: **0**, missing 0. Identical, so arm A differs from R0 in loader mechanics only, never in image content.

## 2 — landmarks, and the common eligible pool

**33 of 23039 (0.143%)** train real frames have no landmarks81 file — dlib's detector failed on them during preprocessing. The frames themselves are intact.

Within the eligible pool: wrong shape **0**, outside the 256x256 crop **0**.

**The filter is applied to every arm, not only the synthetic ones.** If arm C dropped the frames SBI cannot use while A and B kept them, the arms would no longer share a source pool, and part of any measured difference would be *which frames were trained on* — specifically the hard ones a landmark detector could not localise. The pool is written to `logs/tbiom/sbi/eligible_pool.txt` for the other arms to consume.

## 3 — generator, labels, failure rate

- label order asserted `[0, 1]` = (genuine, fake) on every item — **passes**. Upstream FreqBlender also treats index 1 as fake, so the conventions agree.
- generator failures: **0/60** (0.0%). Failures emit the genuine frame twice and are counted, rather than dropping the sample — dropping would filter difficult frames out of the synthetic arm only and quietly change the eligible source pool.
- genuine and fake members identical after synthesis: **0** (none, as required)

## 4 — normalisation is the detector's own, not a reimplementation

max deviation against the CLIP processor applied directly: **0.00e+00**. The dataset calls the same `vfm_preprocess` object the h5 path uses, so this cannot drift; the check guards against a future edit replacing it with hand-rolled constants.

## 5 — split safety of the source pool

train source videos: **720**; val source videos: **140**; overlap: **0**. Synthesis draws only from train sources, so no val identity can reach a generated example.

## Verdict

**All checks pass.** The integration is correct in the ways that would otherwise fail silently. This says nothing about whether SBI helps.
