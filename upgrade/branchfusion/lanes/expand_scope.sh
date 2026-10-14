#!/usr/bin/env bash
# TD (V-JEPA 2) features for the CDFv3 and KoDF videos that protocol v3.1 excluded.
#
# WHY ONLY TD. App's `pe_core_lLN_*`, LFS's `clip_*` and STT's `pe_core_lLN-b16_*` caches ALL
# already hold the full 5418 CDFv3 and 2239 KoDF videos -- the exclusion was applied when their
# opinions were exported, not when the features were encoded, so those three only need a
# re-export. V-JEPA 2 filtered at EXTRACTION, so its cache genuinely stops at 5118 and 1395 and
# the missing 1144 videos have to be encoded.
#
# EXTRACTED AT `nolipadv`, the widest scope UMAR asked about, so both KoDF variants -- with and
# without adv-real/adv-fake -- are a subset operation at scoring rather than a second GPU pass.
set -u
cd /data/umar/Repos/DFB_NeSyNS
PY=/data/umar/miniconda3/envs/temporal_vjepa2/bin/python
L=logs/scope_expand
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 BF_SCOPE=nolipadv
say () { echo "[$(date -Is)] $*" >> "$L/chain.log"; }

say "=== TD features at scope nolipadv: CDFv3 (7 shards) and KoDF (2 shards) ==="
# GPU 1 is primary and takes CDFv3's seven shards four at a time; GPU 3 takes KoDF.
( for s in 0 1 2 3 4 5 6; do
    CUDA_VISIBLE_DEVICES=1 $PY analysis/tbiom/temporal/extract_video.py --model vjepa2 \
      --dataset CDFv3 --shard $s --nshards 7 >> "$L/vjepa2_CDFv3_s$s.log" 2>&1 \
      && say "CDFv3 shard $s ok" || say "CDFv3 shard $s FAILED"
  done ) &
( for s in 0 1; do
    CUDA_VISIBLE_DEVICES=3 $PY analysis/tbiom/temporal/extract_video.py --model vjepa2 \
      --dataset KoDF --shard $s --nshards 2 >> "$L/vjepa2_KoDF_s$s.log" 2>&1 \
      && say "KoDF shard $s ok" || say "KoDF shard $s FAILED"
  done ) &
wait
say "=== TD scope expansion finished ==="
