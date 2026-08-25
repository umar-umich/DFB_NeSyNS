#!/usr/bin/env bash
# Queue the official FS-VFM linear-probe baseline to start once BOTH Stage-A students finish.
#
# Waits rather than competing: the students are on the critical path to the Stage-2 gate, and
# running a third ViT-L job alongside them would slow both. Polls for the last checkpoint of each
# run and for the training processes to be gone, then trains and tests.
#
# EPOCHS defaults to 10, not the authors' 50. That is a deliberate, recorded deviation for the
# first submission deadline; rerun with EPOCHS=50 for the camera-ready and report both.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

PY=/data/umar/miniconda3/envs/dfb_nesy/bin/python
FSFM=/data/umar/Repos/FSFM-CVPR25/fsvfm/linearprobe/cross_dataset_DFD_and_DiFF
DATA=/data/umar/Datasets/fsvfm_lp
CKPT=/data/umar/Repos/DFB_NeSyNS/weights/FS-VFM/checkpoint-599.pth
EPOCHS=${EPOCHS:-10}
GPUS=${GPUS:-1,2}
NPROC=${NPROC:-2}
OUT=${OUT:-logs/fpad/fsvfm_lp_${EPOCHS}ep}
LAST=$(printf "epoch_%03d.pth" $((${STUDENT_EPOCHS:-10} - 1)))

echo "[queue] waiting for both students to write $LAST ..."
until [ -f "logs/fpad/studentA_ordinary_seed42/$LAST" ] \
   && [ -f "logs/fpad/studentA_preserve_seed42/$LAST" ] \
   && ! pgrep -f "train_fpad.py --stage A" >/dev/null; do
  sleep 120
done
echo "[queue] students done at $(date). Starting the linear probe ($EPOCHS epochs, GPUs $GPUS)."

mkdir -p "$OUT"
# Their build_transform reads the normalization from beside the checkpoint and falls back to
# ImageNet SILENTLY if absent, which is wrong for this model. Put it in place up front.
cp -n /data/umar/Repos/DFB_NeSyNS/weights/FS-VFM/pretrain_ds_mean_std.txt "$OUT/" 2>/dev/null

# Hyperparameters are the authors' ViT-L values from scripts_DFD/run_LP_DfD-ViT-L.sh, unchanged
# except --epochs. batch 128 x 2 GPUs = effective 256, which is the batch their blr assumes under
# the linear scaling rule; running on one GPU would halve the effective lr.
# torchrun, NOT `python -m torch.distributed.launch`. The 10-epoch run failed on
# `unrecognized arguments: --local-rank=1`: torch >= 2.0 passes --local-rank (hyphen) while this
# 2024-era code declares --local_rank (underscore). It does not matter, because
# `util/misc.py:233` reads LOCAL_RANK from the ENVIRONMENT, which is exactly what torchrun sets
# and torch.distributed.launch additionally duplicates as a flag. torchrun therefore runs the
# authors' code unchanged and keeps effective batch 128 x 2 = 256, which their blr assumes.
( cd "$FSFM" && CUDA_VISIBLE_DEVICES=$GPUS OMP_NUM_THREADS=1 \
  $(dirname $PY)/torchrun --nproc_per_node=$NPROC --master_port=${PORT:-29613} \
    main_linearprobe_DfD.py \
    --accum_iter 1 --apply_simple_augment --batch_size 128 --nb_classes 2 \
    --model vit_large_patch16 --epochs "$EPOCHS" --blr 1e-2 \
    --layer_decay 0 --weight_decay 0 --drop_path 0.1 --reprob 0.25 \
    --mixup 0.8 --cutmix 1.0 --dist_eval \
    --finetune "$CKPT" \
    --finetune_data_path "$DATA/FFpp_c23" \
    --output_dir "/data/umar/Repos/DFB_NeSyNS/$OUT" \
) 2>&1 | tee -a "$OUT/train.log"

BEST="$OUT/checkpoint-min_val_loss.pth"
[ -f "$BEST" ] || BEST=$(ls -t "$OUT"/checkpoint*.pth 2>/dev/null | head -1)
if [ -z "${BEST:-}" ] || [ ! -f "$BEST" ]; then
  echo "[queue] no checkpoint produced — see $OUT/train.log"; exit 1
fi
cp -n /data/umar/Repos/DFB_NeSyNS/weights/FS-VFM/pretrain_ds_mean_std.txt "$(dirname "$BEST")/" 2>/dev/null

echo "[queue] testing $BEST on our OOD sources with THEIR test code"
CUDA_VISIBLE_DEVICES=${GPUS%%,*} $PY analysis/fpad/fsvfm_lp_test.py \
  --resume "$BEST" --data-root "$DATA" \
  --out "docs/DiCoME_eval/fsvfm_linearprobe_${EPOCHS}ep.json" 2>&1 | tee -a "$OUT/test.log"
echo "[queue] done at $(date)"
