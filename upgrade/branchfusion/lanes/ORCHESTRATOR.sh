#!/usr/bin/env bash
# UNATTENDED ORCHESTRATOR -- runs the whole remaining campaign with nobody watching.
#
# UMAR disconnects 2026-10-09 ~23:30 and wants final results on return (hours to 2 days). The
# Claude session ends with his ssh session; only detached processes survive. So every decision that
# would otherwise need judgement is written down here as a RULE ON A NUMBER, and every stage
# records what it did so the reasoning can be audited rather than reconstructed.
#
# Authorised by UMAR before disconnecting: chain everything; commit code AND results locally, no
# push; keep GPUs 1, 2 and 3 regardless of other users.
#
# RULES THIS ENCODES
#   gate 1  a branch must beat its own NO-OM control at the same depth
#   gate 2  its per-video error phi against A must be below 0.60
#           (frozen Y was +0.789 and merely substituted for A; in-loop Y16 reached +0.678)
#   A branch failing either gate is REPORTED, not registered, and never enters the fusion screen.
#
#   depth   b12/b8/b16-seed1 are already running. If seed 1 moves b16 by more than the b16-b20 gap
#           (0.0060), the depth ordering is noise and no further depth run is launched -- that
#           conclusion is worth more than another layer.
#
# EVERY STAGE IS RESUMABLE: each writes a .done marker and is skipped if the marker exists, so this
# script can be re-run after a crash or a reboot without repeating finished work.
set -u
cd /data/umar/Repos/DFB_NeSyNS
PY=/data/umar/miniconda3/envs/discern_ext/bin/python
PYT=/data/umar/miniconda3/envs/discern_ext_timm/bin/python
PYV=/data/umar/miniconda3/envs/temporal_vjepa2/bin/python
L=logs/branchfusion/lanes
M=logs/branchfusion/orchestrator
mkdir -p "$M"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4

say () { echo "[$(date -Is)] $*" | tee -a "$M/orchestrator.log"; }
done_marker () { [ -f "$M/$1.done" ]; }
mark () { touch "$M/$1.done"; }

commit () {  # commit <message>
  # `logs/` matches the `*log*` ignore rule, so the decision log would never enter history and a
  # clone of this branch would show the results without the reasoning that produced them. Mirror it
  # into reports/, which is tracked, before every commit.
  mkdir -p reports/branch_fusion
  { echo "# Orchestrator decision log"; echo;
    echo "Written unattended. Every line is a decision the orchestrator made, or a number it made";
    echo "it on. The live copy is \`logs/branchfusion/orchestrator/orchestrator.log\`, which is";
    echo "gitignored; this is the committed mirror."; echo; echo '```';
    cat "$M/orchestrator.log" 2>/dev/null; echo '```'; } > reports/branch_fusion/ORCHESTRATOR_LOG.md
  for v in depth gate; do
    [ -f "$M/${v}_verdict.txt" ] && cp "$M/${v}_verdict.txt" "reports/branch_fusion/${v}_verdict.txt"
  done
  git add -A analysis reports upgrade 2>/dev/null
  git diff --cached --quiet 2>/dev/null && { say "nothing to commit for: $1"; return 0; }
  git commit -q -m "$1

Committed unattended by the orchestrator; see logs/branchfusion/orchestrator/orchestrator.log.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>" && say "committed: $1"
}

wait_for () {  # wait_for <pattern> <label> <max_minutes>
  local pat=$1 label=$2 max=$3 n=0
  while pgrep -f "$pat" > /dev/null 2>&1; do
    sleep 60; n=$((n + 1))
    [ $((n % 30)) -eq 0 ] && say "still waiting on $label (${n} min)"
    if [ "$n" -ge "$max" ]; then say "TIMEOUT waiting on $label after ${max} min -- moving on"; return 1; fi
  done
  say "$label is finished"; return 0
}

# DISK. /data was at 98% (659G free) when this launched. The test-set caches and checkpoints
# below add a few GB, but a two-day unattended run that fills the volume would corrupt whatever it
# was writing at the time. Stop cleanly instead, with the reason recorded.
disk_ok () {
  local free; free=$(df --output=avail -BG /data/umar | tail -1 | tr -dc '0-9')
  [ "${free:-0}" -ge 40 ] && return 0
  say "DISK BELOW 40G (${free}G free) -- stopping before writing more. Nothing is corrupted;"
  say "  free space and re-run this script, which resumes from its .done markers."
  return 1
}

say "=== orchestrator start ==="
say "GPUs: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"

# ---------------------------------------------------------------- 1. the runs already in flight
wait_for "adapt_om.py"            "spatial depth runs (b12/b8/b16 seed 1)" 600
wait_for "train_om_pecore.py"     "P+OM and its control"                   900
commit "fusion: depth probes and the P off-manifold runs"

# ---------------------------------------------------------------- 2. read the numbers, decide
$PY upgrade/branchfusion/om_report.py --stage depth > "$M/depth_verdict.txt" 2>&1 || true
say "depth verdict:"; sed 's/^/    /' "$M/depth_verdict.txt" | tee -a "$M/orchestrator.log"

# ------------------------------------------------- 2b. P+OM again, with the plain path kept
# WHY A SECOND P RUN. The first put ONLY the residual in front of the temporal stage, and it could
# not learn: training loss sat at 1.58 while the no-OM control fell past 1.00, and development held
# near chance. That is a predictable consequence of the construction rather than a fact about
# off-manifold features -- DISCERN gives f_t and f_a their own evidential heads and FUSES them, so
# the residual is a second view beside the plain feature. The alignment term drives f_c toward f_t
# by design, which squeezes f_a toward zero, so a network whose only input is f_a has its input
# removed by its own loss. `--om-mode concat` passes [f_t ; f_a] and is the faithful comparison.
# The residual-only run is kept and reported: it is the evidence for this paragraph.
if ! done_marker p_concat && [ -f "$M/concat.smokepass" ]; then
  say "launching P+OM (concat: [f_t ; f_a]) on GPU 3 -- the DISCERN-faithful construction"
  CUDA_VISIBLE_DEVICES=3 $PYV upgrade/branchfusion/train_om_pecore.py \
      --operator bvae --block 16 --om-mode concat \
      --model pecore_l14 --lora --augment B --seed 42 --workers 8 \
      >> "$L/om_P_b16_cat.log" 2>&1 && mark p_concat || say "P+OM concat FAILED"
  commit "fusion: the off-manifold residual as a second view rather than a replacement"
elif ! done_marker p_concat; then
  say "P+OM concat SKIPPED: no smoke pass recorded at $M/concat.smokepass"
  mark p_concat
fi

# ---------------------------------------------------------------- 3. branch S, encoder in loop
if ! done_marker s_inloop && [ ! -f "$M/s_inloop.smokepass" ]; then
  say "S in-loop is SKIPPED: its smoke test never recorded a pass."
  say "  The timestamp plumbing was written minutes before the session ended and a watcher was"
  say "  to mark $M/s_inloop.smokepass once the rehearsal reached its last stage. No marker means"
  say "  the rehearsal failed or never finished, and launching hours of GPU on unvalidated code"
  say "  unattended is exactly the mistake this campaign kept making."
  mark s_inloop
fi
if ! done_marker s_inloop; then
  say "launching S in-loop (real-timestamp temporal stage) on GPU 3"
  CUDA_VISIBLE_DEVICES=3 $PYV upgrade/branchfusion/train_om_pecore.py \
      --operator bvae --block 16 --temporal seq --om-mode concat \
      --model pecore_l14 --lora --augment B --seed 42 --workers 8 \
      >> "$L/om_S_inloop.log" 2>&1 && mark s_inloop || say "S in-loop FAILED (see $L/om_S_inloop.log)"
  # Its no-OM control, so gate 1 can be applied to S as well as to P.
  CUDA_VISIBLE_DEVICES=2 $PYV upgrade/branchfusion/train_om_pecore.py \
      --operator none --block 16 --temporal seq \
      --model pecore_l14 --lora --augment B --seed 42 --workers 6 \
      >> "$L/om_S_inloop_control.log" 2>&1 || say "S control FAILED"
  commit "fusion: branch S with the off-manifold front end and its control"
fi

# ---------------------------------------------------------------- 4. finish the Y16 test exports
if ! done_marker y16_tests; then
  disk_ok || exit 1
  say "completing the block-16 test-set extraction"
  for DS in CDFv2 CDFv3 DFD DFDC DFDCP DFEval24 FFIW KoDF IDForge DFDM; do
    [ -f "logs/upgrade/feats/omEIL_b16_${DS}.npz" ] && continue
    CUDA_VISIBLE_DEVICES=1 $PYT upgrade/extract_adapted.py --backbone pe_core_l --recipe LN \
        --readout b16 --ckpt logs/upgrade/om/om_pe_core_l_LN_bvae_s0_b16.pt \
        --cache-tag omEIL_b16 --dataset "$DS" --batch-size 96 >> "$L/om_eil_tests.log" 2>&1 \
      || say "extraction FAILED on $DS"
  done
  CUDA_VISIBLE_DEVICES=1 $PY upgrade/branchfusion/om_eil_export.py \
      --ckpt logs/upgrade/om/om_pe_core_l_LN_bvae_s0_b16.pt --cache omEIL_b16 --label Y16 \
      >> "$L/om_eil_export.log" 2>&1 && mark y16_tests
  commit "fusion: opinions for the encoder-in-loop off-manifold branch"
fi

# ---------------------------------------------------------------- 5. gates, then the fusion screen
$PY upgrade/branchfusion/om_report.py --stage gates > "$M/gate_verdict.txt" 2>&1 || true
say "gate verdict:"; sed 's/^/    /' "$M/gate_verdict.txt" | tee -a "$M/orchestrator.log"

if grep -q "^REGISTER:" "$M/gate_verdict.txt" 2>/dev/null; then
  for B in $(grep "^REGISTER:" "$M/gate_verdict.txt" | cut -d: -f2); do
    say "branch $B passed both gates -- entering the full subset screen"
    $PY upgrade/branchfusion/subsets.py --screen --branches "A,S,V,O,M,P,$B" \
        --sizes 1,2,3,4,5,6,7 >> "$L/screen_with_new.log" 2>&1 || say "screen FAILED for $B"
    $PY upgrade/branchfusion/per_dataset.py --branches "A,S,V,O,M,P,$B" \
        --sizes 1,2,3,4,5,6,7 --tag "with_$B" >> "$L/per_dataset_new.log" 2>&1 || true
  done
  commit "fusion: subset screen including the surviving off-manifold branch"
else
  say "no branch passed both gates; the screen is NOT re-run and the deployed selection stands"
fi

# ---------------------------------------------------------------- 6. correlation and the tables
$PY upgrade/branchfusion/decorrelation.py --branches A,S,V,O,M,P >> "$L/decorrelation_final.log" 2>&1 || true
$PY upgrade/branchfusion/tables.py >> "$L/tables_final.log" 2>&1 || true
$PY upgrade/branchfusion/om_report.py --stage final > reports/branch_fusion/OM_CAMPAIGN.md 2>&1 || true
commit "fusion: final correlation matrix, tables and the off-manifold campaign report"

say "=== orchestrator done ==="
say "read reports/branch_fusion/OM_CAMPAIGN.md first; the decision log is $M/orchestrator.log"
