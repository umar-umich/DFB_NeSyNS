#!/usr/bin/env bash
# The ten TEST sets for the encoder-in-the-loop OM branch, on GPU 2.
# Dev extraction is already running on GPU 1; these are independent, so they run beside it rather
# than after it. FaceShifter is NOT in the list and must not be added: it is SEALED until the final
# recipe is fixed (tbiom/docs/SEALED_BENCHMARKS.md).
set -u
cd /data/umar/Repos/DFB_NeSyNS
PY=/data/umar/miniconda3/envs/discern_ext_timm/bin/python
LOG=logs/branchfusion/lanes/om_eil_tests.log
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
: > "$LOG"
for DS in CDFv2 CDFv3 DFD DFDC DFDCP DFEval24 FFIW KoDF IDForge DFDM; do
  echo "=== $DS $(date -Is)" >> "$LOG"
  $PY upgrade/extract_adapted.py --backbone pe_core_l --recipe LN --readout b16 \
      --ckpt logs/upgrade/om/om_pe_core_l_LN_bvae_s0_b16.pt \
      --cache-tag omEIL_b16 --dataset "$DS" --batch-size 96 >> "$LOG" 2>&1 \
    || echo "FAILED on $DS" >> "$LOG"
done
echo "OM_EIL_TESTS_DONE $(date -Is)" >> "$LOG"
