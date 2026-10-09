#!/bin/bash
# ARM 4 -- SCHEDULED, NOT LAUNCHED. OM on the V-JEPA 2 LoRA temporal feature.
#
# READ THIS BEFORE RUNNING IT. tbiom/reports/EXP4B_TEMPORAL_RESIDUAL.md ALREADY applied this exact
# residual family to V-JEPA 2: plain 0.7129 -> bvae 0.7151 -> bvae_real 0.7166 -> mrvae 0.7201.
# UMAR's decision of 2026-09-17 closed residual operators as reported negatives and lists "run more
# residual-operator variants" and "adapt more temporal encoders" under Do-not.
#
# The ONE thing that result does not settle: it was measured on a FROZEN probe topping out at 0.71,
# while the deployed V-JEPA LoRA branch scores far higher. So the honest framing for re-running it is
# "the earlier test used a weak feature", not "we never tried it" -- and the write-up must say so,
# or this reads as re-running a closed negative until it turns positive.
#
# COST: the expensive arm. No V-JEPA embeddings are cached anywhere, only opinions, so this needs a
# full extraction pass with the video model over all 13 datasets before any OM training can start.
set -u
echo "ARM 4 is a stub and re-opens a decision closed on 2026-09-17."
echo "Required first: a V-JEPA 2 LoRA embedding dump (13 datasets, video model, GPU hours)."
echo "Required also: an explicit decision to re-open, recorded in tbiom/docs/."
exit 2
