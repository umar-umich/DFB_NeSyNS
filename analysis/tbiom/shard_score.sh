#!/usr/bin/env bash
# Shard a DF40 scoring pass across N parallel processes, then merge the parquets.
#
#   bash analysis/tbiom/shard_score.sh <student_dir> <epoch> <out_root> <n_shards> [extra args...]
#
# WHY sharding and not a bigger batch. Measured on an idle H200 (forward-only, FS-VFM
# teacher+student):
#
#     batch  32   15.55 ms/frame   64.3 frames/s   1.93 GiB
#     batch 128   13.49 ms/frame   74.1 frames/s   4.20 GiB
#     batch 256   13.33 ms/frame   75.0 frames/s   7.22 GiB
#
# Batch 32 -> 256 buys +17% for 3.7x the memory and saturates by 128, so GPU compute is not the
# constraint and neither is memory (7 GiB of 143). The constraint is CPU-side: PNG decode plus a
# fresh dataset construction per method. More memory cannot fix a decode bottleneck; more
# PROCESSES can, because each brings its own dataloader workers. This machine has 384 cores at a
# load average under 10, so the headroom is real.
#
# Shards split by METHOD, never within one, so each shard's parquet is a set of whole methods and
# the merge is a concatenation with no risk of double-counting a video.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

STUDENT=${1:?student run dir}
EPOCH=${2:?epoch}
OUT=${3:?output root}
N=${4:-6}
shift 4 || true
EXTRA=("$@")

PY=/data/umar/miniconda3/envs/dfb_nesy/bin/python
DEVICE=${DEVICE:-cuda:3}
BATCH=${BATCH:-128}
WORKERS=${WORKERS:-12}
MAXB=${MAXB:-10}          # batches per method; scaled so batch*maxb matches the unsharded frames
METHODS=$(cat "${METHODS_FILE:-/tmp/devmethods.txt}")

read -r -a ALL <<< "$METHODS"
echo "sharding ${#ALL[@]} methods across $N processes on $DEVICE (batch $BATCH, $WORKERS workers)"

pids=()
for ((s=0; s<N; s++)); do
  shard=()
  for ((i=s; i<${#ALL[@]}; i+=N)); do shard+=("${ALL[$i]}"); done
  [ ${#shard[@]} -eq 0 ] && continue
  echo "  shard $s: ${#shard[@]} methods"
  $PY -u training/score_fpad.py --student "$STUDENT" --epoch "$EPOCH" --seed 42 \
      --datasets "${shard[@]}" --df40 --output "$OUT/shard_$s" \
      --batch-size "$BATCH" --workers "$WORKERS" --max-batches "$MAXB" \
      --device "$DEVICE" "${EXTRA[@]}" > "$OUT.shard_$s.log" 2>&1 &
  pids+=($!)
done

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "shards finished (fail=$fail)"

$PY - "$OUT" <<'PYEOF'
import sys, pathlib, pandas as pd
out = pathlib.Path(sys.argv[1])
parts = sorted(out.glob("shard_*/profile_epoch_*.parquet"))
if not parts:
    raise SystemExit(f"no shard parquets under {out} — check {out}.shard_*.log")
df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
dup = df.duplicated(subset=["dataset", "key"]).sum()
if dup:
    raise SystemExit(f"{dup} (dataset, key) rows appear in more than one shard — shards must "
                     f"split by METHOD, so a duplicate means the split was wrong and every "
                     f"pooled number would double-count them")
dest = out / f"profile_merged.parquet"
df.to_parquet(dest, index=False)
print(f"merged {len(parts)} shards -> {dest} ({len(df)} frames, "
      f"{df['dataset'].nunique()} methods)")
PYEOF
