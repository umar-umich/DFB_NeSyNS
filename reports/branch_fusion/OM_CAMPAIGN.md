# The off-manifold campaign beyond CLIP

Written unattended by `om_report.py`. Every number is read from a run record on disk;
nothing here is recalled from the session that set the campaign up.

## The question

The deployed Off-manifold view O is built on CLIP ViT-L/14 whose LoRA was trained WITH
the beta-VAE in the loop. The campaign asked whether that construction reproduces on
other trunks, and three successive answers were wrong for the same reason: the residual
was fitted on FROZEN features, so it could only ever be a function of an embedding that
was never asked to make it informative.

## Reference points

- frozen Y (arm 1, cached features): dev v3.1 0.8696
- A alone: dev v3.1 0.8736
- O, the CLIP off-manifold: dev v3.1 0.8252
- in-loop Y16, block 16 (spatial trunk): dev v3.1 0.8673

Per-video error correlation against the branches already deployed:

| pair | development (n=5814) | test pooled (n=27974) |
|---|---|---|
| frozen Y vs A | +0.789 | +0.786 |
| in-loop Y16 vs A | +0.678 | +0.669 |
| in-loop Y16 vs O | +0.440 | +0.386 |
| in-loop Y16 vs V | +0.100 | +0.197 |
| O's highest phi with anything | +0.476 | +0.415 |

Test-pooled AUROC: A 0.9392, S 0.9366, Y16 0.9296, O 0.9071, P 0.9057, V 0.8353, M 0.7281

Putting the encoder in the loop moved the off-manifold branch from +0.789 to +0.678
against A on development and from +0.786 to +0.669 on test -- the same shift on both
halves, so the mechanism is real. It is also not enough: the gate is 0.60, and a branch
at +0.67 still fails on the same videos A fails on. Y16 is a SUBSTITUTE for A, not an
addition, which is what the frozen arm was too.

## Depth

PE spatial trunk, off-manifold with the encoder in the loop (full dev, residual arm):
  b8    0.7831
  b12   0.8053
  b16   0.8673
  last  0.8664
  b20   0.8613
  spread across depths: 0.0842

block 16: seed 0 0.8673, seed 1 0.8547, |difference| 0.0126

Each depth gap against that noise -- a gap smaller than the seed movement is not a
result, however suggestive its ordering looks:
  b16   0.8673  vs  last  0.8664   gap 0.0009   INDISTINGUISHABLE
  last  0.8664  vs  b20   0.8613   gap 0.0051   INDISTINGUISHABLE
  b20   0.8613  vs  b12   0.8053   gap 0.0560   REAL (4.4x the noise)
  b12   0.8053  vs  b8    0.7831   gap 0.0221   REAL (1.8x the noise)

Some gaps are several times the seed movement, so depth IS measurable here -- but
only where the table says REAL. The blocks separated by less than the noise are one
result, not an ordering, and must not be reported as a ranking.

## Gates

Temporal off-manifold runs, PE-Core trunk, block 16 (development v3.1):

  ombvae_b16_cat                     off-manifold   0.8409  (epoch 9)
  ombvae_b16_cat_seq                 off-manifold   0.8497  (epoch 9)
  omnone_b16                         NO-OM CONTROL  0.8678  (epoch 9)
  omnone_b16_seq                     NO-OM CONTROL  0.8665  (epoch 6)

P16om: ombvae_b16_cat
  gate 1  0.8409 vs control 0.8678  -> FAIL
  gate 2  phi vs A +0.415 < 0.6  -> PASS
  P16om is reported, not registered: a branch that fails either gate would either duplicate A or be worse than the same network without the residual.
S16om: ombvae_b16_cat_seq
  gate 1  0.8497 vs control 0.8665  -> FAIL
  gate 2  phi vs A +0.500 < 0.6  -> PASS
  S16om is reported, not registered: a branch that fails either gate would either duplicate A or be worse than the same network without the residual.

## How to read a negative here

A branch that fails gate 2 is not a weak branch; it is a DUPLICATE of one already in
the system. A+V+O was selected in all five nested folds without correlation ever being
shown to the search, so the fusion already prefers decorrelated views. A new PE-trunk
branch has to leave that cluster to be worth a row, and standalone accuracy cannot tell
you whether it did.
