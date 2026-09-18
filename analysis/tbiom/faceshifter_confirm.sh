#!/usr/bin/env bash
# Score FaceShifter ONCE with the final system, after Experiment 5 has fixed the recipe.
#
# The seal (tbiom/docs/SEALED_BENCHMARKS.md) is lifted by running this and nothing else: it builds the h5, exports
# the three views, fuses with the deployed DS-R weights and appends the row to the results table. No hyper-parameter,
# threshold or fusion weight is touched afterwards -- if anything about the recipe changes later, this number is
# discarded rather than recomputed.
set -uo pipefail
ROOT=/data/umar/Repos/DFB_NeSyNS; EXT=/data/umar/Repos/DISCERN_Ext
PYA=/data/umar/miniconda3/envs/discern_ext/bin/python
PYT=/data/umar/miniconda3/envs/discern_ext_timm/bin/python
PYV=/data/umar/miniconda3/envs/temporal_vjepa2/bin/python
S23=$ROOT/logs/tbiom/stage23; LOG=$ROOT/logs/tbiom/faceshifter
export CUDA_VISIBLE_DEVICES=1 NO_ALBUMENTATIONS_UPDATE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p "$LOG"; cd "$ROOT" || exit 2
say() { echo "[$(date '+%m-%d %H:%M')] $*"; }

[ -f "$ROOT/tbiom/reports/EXP_FREQBLENDER.md" ] || { say "EXP_FREQBLENDER.md is missing -- the recipe is not fixed yet"; exit 1; }

say "building the FaceShifter h5, splits and export configs"
$PYA analysis/tbiom/build_gend_test_h5.py --dataset FaceShifter --unseal > "$LOG/build.log" 2>&1 || {
  grep -q "refusing to overwrite" "$LOG/build.log" && say "h5 already present, continuing" || { say "build FAILED"; exit 1; }; }

say "exporting the appearance and off-manifold views"
CK=$EXT/runs/discern_ext_train/warm_dicome_gate_seed42/checkpoints/discern_ext-best-epoch=19-val_auroc_video=0.9956.ckpt
if [ ! -f "$S23/warmgate_sel_FaceShifter.csv" ]; then
  (cd "$EXT" && $PYT eval_adaptation/export_views.py --checkpoint "$CK" \
      --config eval_adaptation/configs/warmgate_FaceShifter.yaml --dataset FaceShifter \
      --device cuda:0 --batch-size 64 --out "$S23/warmgate_sel_FaceShifter.csv") > "$LOG/export_views.log" 2>&1 || {
    say "view export FAILED"; exit 1; }
fi

say "exporting the temporal branch"
$PYV analysis/tbiom/temporal/export_branch.py --branch vjepa2_lora_B_seed42 --datasets FaceShifter \
  > "$LOG/export_branch.log" 2>&1 || { say "branch export FAILED"; exit 1; }

say "scoring"
$PYA analysis/tbiom/faceshifter_score.py > "$LOG/score.log" 2>&1 && say "done -- see tbiom/reports/FACESHIFTER.md" \
  || { say "scoring FAILED"; exit 1; }
