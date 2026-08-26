#!/usr/bin/env bash
# Re-run the Stage-1 membership gate against the FF++(+)DF40 anchor.
#
#   bash analysis/tbiom/stage1_new_anchor.sh <epoch> [out_root]
#
# WHY THIS HAS TO BE RE-RUN AT ALL. Stage 1 decided membership against an anchor trained on FF++
# c23 only. A DF40-trained anchor is a different model: it is stronger on exactly the generator
# families the experts were rescuing, so complementarity measured against the old anchor does not
# carry over. This is the one open result that could reopen membership.
#
# VALMIX ONLY, AND DF40-Dev DELIBERATELY NOT. DF40-Dev was the OOD probe for the FF++-only
# anchor. This anchor TRAINS on 32 DF40 methods, so DF40-Dev is no longer out of domain for it —
# rescue numbers there would compare an in-domain anchor against out-of-domain experts and flatter
# the anchor for a reason that has nothing to do with complementarity. VALmix stays valid: its
# three corpora (Celeb-DF-v2, DFDCP, Deepfake-Eval-2024) contribute nothing to either training
# set, which the manifest builder verifies.
#
# The EXPERTS are unchanged and already scored on VALmix, so only the anchor is re-scored here:
# its OOD rows and its own FF++ val threshold source. Each expert keeps its own threshold from
# its own FF++ val export, as the gate requires.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

EPOCH=${1:?epoch number}
E=$(printf '%03d' "$EPOCH")
OUT=${2:-logs/tbiom/stage1_newanchor/epoch_$E}
PY=/data/umar/miniconda3/envs/dfb_nesy/bin/python
RUN=logs/tbiom/v1_ffpp_df40_seed42
CKPT=$RUN/epoch_$E.pth
DEVICE=${DEVICE:-cuda:3}
V=logs/tbiom/valmix
S=logs/tbiom/score

[ -f "$CKPT" ] || { echo "no checkpoint at $CKPT"; exit 1; }
mkdir -p "$OUT"

# 1. the new anchor on VALmix (the OOD rows) and on FF++ val (its own operating threshold)
for pair in "VALmix:$OUT/valmix" "FaceForensics++:$OUT/ffppval"; do
  ds=${pair%%:*}; dest=${pair##*:}
  if ls "$dest"/*.parquet >/dev/null 2>&1; then echo "skip $ds (done)"; continue; fi
  echo "scoring new anchor on $ds"
  $PY -u training/eval_v1.py --checkpoint "$CKPT" --seed 42 --overwrite \
      --datasets "$ds" --output "$dest" \
      --batch-size 32 --workers 16 --device "$DEVICE" > "$dest.log" 2>&1 \
    || { echo "scoring $ds FAILED — see $dest.log"; exit 1; }
done

ANCHOR=$(ls "$OUT"/valmix/*.parquet | head -1)
THRESH=$(ls "$OUT"/ffppval/*.parquet | head -1)
echo "anchor  $ANCHOR"
echo "thresh  $THRESH"

# 2. the gate, experts unchanged
$PY analysis/tbiom/stage1_membership.py \
  --anchor clip_ffpp_df40 "$ANCHOR" --anchor-col p_sem \
  --threshold-source "$THRESH" --threshold-col p_sem \
  --expert fsvfm_preserve $V/fsvfm_preserve/profile_epoch_009.parquet p_direct \
           $S/fsvfm_preserve_ffppval/profile_epoch_009.parquet \
  --expert fsvfm_ordinary $V/fsvfm_ordinary/profile_epoch_009.parquet p_direct \
           $S/fsvfm_ordinary_ffppval/profile_epoch_009.parquet \
  --expert mrvae_rate     $V/fsvfm_preserve_rate/profile_epoch_009.parquet p_rate \
           $S/fsvfm_preserve_ffppval/profile_epoch_009.parquet \
  --out "$OUT"
echo "report -> $OUT/STAGE1_MEMBERSHIP.md"
