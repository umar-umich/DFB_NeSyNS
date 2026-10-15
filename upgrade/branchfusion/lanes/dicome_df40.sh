#!/usr/bin/env bash
# DiCoME's OWN repo checkpoint across every DF40 row, for the appendix.
#
# The DF40 table carries `DiCoME (our training)` -- our reproduction, exported through the
# stage-23 path. This adds DiCoME evaluated in its own repo by its own driver, so the appendix
# shows the baseline on its home ground as well as on ours.
#
# SAME CHECKPOINT AS THE TABLE 1 ROW: `ffpp_reproduce_seed42` epoch 1, which reproduces that
# row's CDFv2 0.9644 exactly. A method must not be two different models in two tables.
#
# `eval_df40.py` builds CLIP ViT-L/14 once and swaps the datamodule per row -- running
# `train_dicome.py test` 78 times would spend ~20 minutes on model construction alone. Its video
# rule is `key.rsplit("/", 1)[0]`, the full path minus the frame, which is ours; it does NOT use
# the bare `video_id` that collapses FFIW's fake and real halves together.
set -u
cd /data/umar/Repos/DiCoME
PY=/data/umar/miniconda3/envs/discern_ext_timm/bin/python
CK="runs/dicome_train/ffpp_reproduce_seed42/checkpoints/dicome-best-epoch=01-val_auroc_video=0.9960.ckpt"
OUT=/data/umar/Repos/DFB_NeSyNS/logs/tbiom/df40/dicome_repo_df40.json
L=/data/umar/Repos/DFB_NeSyNS/logs/scope_expand/dicome
M=/data/umar/Repos/DFB_NeSyNS/logs/branchfusion/orchestrator
mkdir -p "$L"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
say () { echo "[$(date -Is)] $*" >> "$M/orchestrator.log"; }

say "=== DiCoME repo checkpoint over every DF40 row ==="
# ONLY the 41 rows our table uses. The driver's full list is 78, and the extras are the pooled
# aggregates (EFSAll, FEAll, FSAll, FRAll) plus whole-face-synthesis families that are out of
# scope -- `EFSAll-cdf` alone is 245k frames, and the first seven rows of an unrestricted run
# were all aggregates, none of which appear in any table here.
ROWS="MRAA-cdf,MRAA-ff,blendface-cdf,blendface-ff,danet-cdf,danet-ff,deepfacelab,e4s-cdf,e4s-ff,facedancer-cdf,facedancer-ff,faceswap-cdf,faceswap-ff,facevid2vid-cdf,facevid2vid-ff,fomm-cdf,fomm-ff,fsgan-cdf,fsgan-ff,hyperreenact-cdf,hyperreenact-ff,inswap-cdf,inswap-ff,lia-cdf,lia-ff,mcnet-cdf,mcnet-ff,mobileswap-cdf,mobileswap-ff,one_shot_free-cdf,one_shot_free-ff,pirender-cdf,pirender-ff,sadtalker-cdf,sadtalker-ff,simswap-cdf,simswap-ff,tpsm-cdf,tpsm-ff,uniface-cdf,uniface-ff"
CUDA_VISIBLE_DEVICES=1 $PY eval_adaptation/eval_df40.py "$CK" --out "$OUT" --only "$ROWS" \
    >> "$L/df40.log" 2>&1 \
  && say "=== DiCoME DF40 done -> $OUT ===" || say "=== DiCoME DF40 FAILED -- see $L/df40.log ==="
