#!/usr/bin/env bash
# O's CROSS-ENTROPY counterpart, exported rather than trained.
#
# `run1d_simplece_seed42` is already trained (best epoch 5, val video AUROC 0.9952) -- the same
# three views as arm B with every head swapped for a logit head and the loss swapped for plain
# cross-entropy. O's CE counterpart is that run's ARTIFACT view, so this is an export job, not
# the retraining it first looked like.
#
# `export_views.py` already handles the arm correctly: it has a dedicated `simple_ce` path so
# logits never go through `alpha = evidence + 1`, which for a logit of -2 would give alpha = -1.
set -u
cd /data/umar/Repos/DISCERN_Ext
PYT=/data/umar/miniconda3/envs/discern_ext_timm/bin/python
CK=runs/discern_ext_train/run1d_simplece_seed42/checkpoints/discern_ext-best-epoch=05-val_auroc_video=0.9952.ckpt
OUT=/data/umar/Repos/DFB_NeSyNS/logs/branchfusion/o_ce
M=/data/umar/Repos/DFB_NeSyNS/logs/branchfusion/orchestrator
L=/data/umar/Repos/DFB_NeSyNS/logs/branchfusion/lanes
mkdir -p "$OUT"
# NOT CUDA_VISIBLE_DEVICES: the checkpoint was saved from cuda:3, so torch.load asks for device
# 3 by index, and masking the GPUs makes that index invalid inside the process ("device_count()
# is 1"). `--device` selects the same card without renumbering it.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
say () { echo "[$(date -Is)] $*" >> "$M/orchestrator.log"; }

say "=== O CE export: artifact view from the simple-CE run, 16 datasets ==="
for DS in CDFv2 CDFv3 DFD DFDC DFDCP DFEval24 FFIW KoDF IDForge DFDM VALmix VALmixV2 FFpp_val UADFV DeepSpeak-1.1 DeepSpeak-2.0; do
  f="$OUT/${DS}.csv"
  [ -s "$f" ] && { say "O-CE $DS already exported"; continue; }
  $PYT eval_adaptation/export_views.py --checkpoint "$CK" \
      --config "eval_adaptation/configs/run1dce_${DS}.yaml" --dataset "$DS" \
      --out "$f" --device cuda:3 --num-workers 4 >> "$L/o_ce_export.log" 2>&1 \
    && say "O-CE $DS done" || say "O-CE $DS FAILED"
done
say "=== O CE export done ==="
