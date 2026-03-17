#!/usr/bin/env bash
# =============================================================================
# ablation_run.sh  — NeSyDeFake Branch Ablation  (all 7 combinations)
# =============================================================================
#
# USAGE
# ─────
#   Single-GPU (e.g. GPU 7):
#       bash ablation_run.sh --mode single --gpus 7 --seeds "42 123 512"  
#
#   Multi-GPU DDP (e.g. GPUs 4,5,6):
#       bash ablation_run.sh --mode ddp --gpus 4,5,6 --port 29501
#
#   Run a specific experiment only (useful for re-running failures):
#       bash ablation_run.sh --mode ddp --gpus 4,5,6 --only T_S
#
# NOTES
# ─────
#   • Each experiment's logs go to a separate sub-directory so nothing is
#     overwritten between runs.
#   • Set DETECTOR_PATH and RESULTS_DIR below if your paths differ.
#   • Experiments run sequentially. Parallel runs on the same node are
#     possible by running this script in separate tmux/screen panes with
#     non-overlapping GPU lists and different --port values.
# =============================================================================

set -euo pipefail

# ── Configurable paths ────────────────────────────────────────────────────────
DETECTOR_PATH="./training/config/detector/nesy_defake.yaml"
RESULTS_DIR="./ablation_results"
LOG_DIR="${RESULTS_DIR}/logs_8_causal_semantic"   # subdir for this ablation set (change if running multiple sets)
PYTHON="python"          # or "python3"
TORCHRUN="torchrun"

# ── Defaults ──────────────────────────────────────────────────────────────────
MODE="single"            # single | ddp
GPUS="7"                 # comma-separated GPU IDs
MASTER_PORT="29501"
ONLY=""                  # if set, run only the experiment with this tag
SEEDS="42 123 256 512 1024"   # seeds to sweep; override with --seeds "s1 s2 ..."

# ── Parse CLI args ────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)   MODE="$2";        shift 2 ;;
        --gpus)   GPUS="$2";        shift 2 ;;
        --port)   MASTER_PORT="$2"; shift 2 ;;
        --only)   ONLY="$2";        shift 2 ;;
        --seeds)  SEEDS="$2";       shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Count GPUs for torchrun
GPU_COUNT=$(echo "$GPUS" | tr ',' '\n' | wc -l)

mkdir -p "${LOG_DIR}"

# ── Experiment definitions ────────────────────────────────────────────────────
# Format: "TAG|branch1 branch2 ..."
declare -a EXPERIMENTS=(
    "S_F|spatial frequency"
)
    # "T|temporal"
    # "S|spatial"
    # "F|frequency"
    # "T_S|temporal spatial"
    # "T_F|temporal frequency"
    # "S_F|spatial frequency"
    # "T_S_F|temporal spatial frequency"


# ── Helper: run one experiment ────────────────────────────────────────────────
run_experiment() {
    local TAG="$1"
    local BRANCHES="$2"          # space-separated branch names
    local SEED="$3"              # random seed

    # Skip if --only is set and tag doesn't match
    if [[ -n "$ONLY" && "$TAG" != "$ONLY" ]]; then
        return 0
    fi

    local BRANCH_ARGS=""
    for b in $BRANCHES; do
        BRANCH_ARGS="$BRANCH_ARGS $b"
    done
    BRANCH_ARGS="${BRANCH_ARGS# }"   # trim leading space

    local RUN_TAG="${TAG}_seed${SEED}"
    local STDOUT_LOG="${LOG_DIR}/exp_${RUN_TAG}.log"

    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  Experiment : ${RUN_TAG}  (branches: ${BRANCHES}, seed: ${SEED})"
    echo "  GPUs       : ${GPUS}  |  Mode: ${MODE}"
    echo "  Log        : ${STDOUT_LOG}"
    echo "════════════════════════════════════════════════════════════════"

    if [[ "$MODE" == "single" ]]; then
        # ── Single GPU ────────────────────────────────────────────────────
        CUDA_VISIBLE_DEVICES="${GPUS}" \
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
        TF_USE_LEGACY_KERAS=1 \
        $PYTHON training/train.py \
            --detector_path "${DETECTOR_PATH}" \
            --local_rank 0 \
            --task_target "ablation_${RUN_TAG}" \
            --active_branches $BRANCH_ARGS \
            --seed "${SEED}" \
            2>&1 | tee "${STDOUT_LOG}"

    elif [[ "$MODE" == "ddp" ]]; then
        # ── Multi-GPU DDP ─────────────────────────────────────────────────
        TORCH_DISTRIBUTED_DEBUG=DETAIL \
        CUDA_VISIBLE_DEVICES="${GPUS}" \
        TF_USE_LEGACY_KERAS=1 \
        $TORCHRUN \
            --nproc_per_node="${GPU_COUNT}" \
            --master_port="${MASTER_PORT}" \
            training/train.py \
            --detector_path "${DETECTOR_PATH}" \
            --ddp \
            --task_target "ablation_${RUN_TAG}" \
            --active_branches $BRANCH_ARGS \
            --seed "${SEED}" \
            2>&1 | tee "${STDOUT_LOG}"
    else
        echo "ERROR: unknown mode '${MODE}'. Use 'single' or 'ddp'."
        exit 1
    fi

    # Increment port so back-to-back DDP runs don't clash on the same host
    MASTER_PORT=$((MASTER_PORT + 1))

    echo "  ✓ Experiment ${RUN_TAG} complete."
}

# ── Main loop ─────────────────────────────────────────────────────────────────
echo ""
echo "NeSyDeFake Branch Ablation — $(date)"
echo "Mode: ${MODE} | GPUs: ${GPUS} | GPU count: ${GPU_COUNT}"
echo "Seeds: ${SEEDS}"
echo ""

for entry in "${EXPERIMENTS[@]}"; do
    TAG="${entry%%|*}"
    BRANCHES="${entry##*|}"
    for SEED in $SEEDS; do
        run_experiment "$TAG" "$BRANCHES" "$SEED"
    done
done

echo ""
echo "All experiments finished. Logs are in: ${LOG_DIR}"
echo "Run the results aggregator next:"
echo "   python ablation_results/aggregate_results.py --log_dir ${LOG_DIR}"