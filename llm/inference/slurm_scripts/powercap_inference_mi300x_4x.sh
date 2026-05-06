#!/bin/bash -l
###############################################################################
# Power-Capped Inference Benchmark — SGLang Serving (AMD MI300X)
#
# Self-contained SLURM job that:
#   1. Starts an SGLang inference server (tensor-parallel across GPUs)
#   2. Runs a warmup pass
#   3. Sweeps GPU power limits (default: 200–700W)
#   4. At each cap: logs GPU metrics at 100ms resolution, runs sglang.bench_serving
#
# All output (benchmark JSON + rocm_smi CSV) is written to a timestamped
# log directory under <WORKSPACE>/inference_logs/.
#
# After the job completes, analyze results with:
#   python analysis/analyze_inference_results.py <path_to_job_logs_dir> --gpu-type mi300x
###############################################################################

#SBATCH --job-name=powercap-inference
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --nodelist=aquavan1
#SBATCH --constraint=mi300x
#SBATCH --output=sbatch_inference_log.out
# #SBATCH --reservation=<your-reservation>   # Uncomment and set if needed

# ========================= USER CONFIGURATION ================================
# Set these paths and parameters to match your environment before submitting.

WORKSPACE="/path/to/your/workspace"                                   # <-- SET THIS
MODEL_DIR="$WORKSPACE/models/Meta-Llama-3-8B/meta-llama/Meta-Llama-3-8B"
SIF_PATH="$WORKSPACE/inference/sglang_v0.4.4.post1-rocm630.sif"
REPO_DIR="$WORKSPACE/inference"                                       # Path to this repo's inference/ folder

NUM_GPUS=4                    # Must match --gres and --ntasks above
NUM_PROMPTS=2048              # Number of prompts per benchmark run
SERVER_PORT=6000              # SGLang server port
POWER_LIMITS=(200 300 400 500 600 700)
PAUSE_SECONDS=${PAUSE_SECONDS:-600}  # Default 10 min; override via env

# AMD GPU visibility
export HIP_VISIBLE_DEVICES=0,1,2,3

# =============================================================================

unset SLURM_EXPORT_ENV

# ------------------ OUTPUT DIRECTORY ------------------
OUTPUT_BASE="${WORKSPACE}/inference_logs"
JOB_TAG="inference_${SLURM_JOBID}_${NUM_GPUS}xmi300x_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${OUTPUT_BASE}/${JOB_TAG}"
mkdir -p "$LOG_DIR"

echo "============================================="
echo "Power-Capped Inference Benchmark (MI300X)"
echo "============================================="
echo "Job ID:        $SLURM_JOBID"
echo "Partition:     $SLURM_JOB_PARTITION"
echo "Nodes:         $SLURM_JOB_NODELIST"
echo "Num GPUs:      $NUM_GPUS"
echo "Num Prompts:   $NUM_PROMPTS"
echo "Power Limits:  ${POWER_LIMITS[*]}"
echo "Pause Seconds: $PAUSE_SECONDS"
echo "Model:         $MODEL_DIR"
echo "Container:     $SIF_PATH"
echo "Dataset:       random synthetic prompts"
echo "Log Directory: $LOG_DIR"
echo "HIP_VISIBLE_DEVICES: $HIP_VISIBLE_DEVICES"
echo "============================================="

# Write job metadata
cat > "$LOG_DIR/job_info.log" << EOF
=== Job Info ===
Job ID: $SLURM_JOBID
Partition: $SLURM_JOB_PARTITION
Nodes: $SLURM_JOB_NODELIST
Num GPUs: $NUM_GPUS
GPU Type: MI300X
Num Prompts: $NUM_PROMPTS
Power Limits: ${POWER_LIMITS[*]}
Pause Seconds: $PAUSE_SECONDS
Model: $MODEL_DIR
Container: $SIF_PATH
Dataset: random synthetic prompts
Log Directory: $LOG_DIR
HIP_VISIBLE_DEVICES: $HIP_VISIBLE_DEVICES
EOF

# ------------------ CLEANUP TRAP ------------------
SERVER_PID=""
cleanup() {
    echo "Cleaning up..."
    if [ -n "$SERVER_PID" ]; then
        echo "Killing SGLang server (PID $SERVER_PID)"
        kill "$SERVER_PID" 2>/dev/null
        wait "$SERVER_PID" 2>/dev/null
    fi
    # Reset power overdrive to default
    echo "Resetting GPU power overdrive to default..."
    if [ "${SKIP_POWERCAP:-0}" = "1" ]; then
        echo "SKIP_POWERCAP=1 set; skipping rocm-smi reset."
    else
        sudo rocm-smi --resetpoweroverdrive || true
    fi
}
trap cleanup EXIT

# Print AMD GPU info
echo "Checking AMD GPU availability..."
rocm-smi || echo "Warning: rocm-smi not found"

# ------------------ START SGLANG SERVER ------------------
echo ""
echo "Starting SGLang server (TP=$NUM_GPUS) with ROCm backend..."
apptainer exec --rocm "$SIF_PATH" \
    python3 -m sglang.launch_server \
        --model-path "$MODEL_DIR" \
        --tp "$NUM_GPUS" \
        --host 0.0.0.0 \
        --trust-remote-code \
        --port "$SERVER_PORT" &
SERVER_PID=$!

# Wait for server readiness
echo "Waiting for server to become ready ..."
MAX_WAIT=600
WAITED=0
until curl -s "http://0.0.0.0:${SERVER_PORT}/v1/models" > /dev/null 2>&1; do
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "ERROR: Server did not start within ${MAX_WAIT}s. Aborting."
        exit 1
    fi
    sleep 5
    WAITED=$((WAITED + 5))
done
echo "Server is ready (waited ${WAITED}s)."

# ------------------ WARMUP ------------------
echo ""
echo "Running warmup (100 prompts, results discarded) ..."
if [ "${SKIP_POWERCAP:-0}" = "1" ]; then
    echo "SKIP_POWERCAP=1 set; skipping power cap for warmup."
else
    sudo rocm-smi --setpoweroverdrive 200
fi
apptainer exec --rocm --bind $TMPDIR "$SIF_PATH" \
    python3 -m sglang.bench_serving \
        --backend sglang \
        --disable-tqdm \
        --host 0.0.0.0 \
        --port "$SERVER_PORT" \
        --output-file "$TMPDIR/warmup.json" \
        --dataset-name random \
        --num-prompt 100 \
        --max-concurrency 256 &> /dev/null
echo "Warmup completed."

# ------------------ POWER-CAP SWEEP ------------------
for POWER_LIMIT in "${POWER_LIMITS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Power Limit: ${POWER_LIMIT}W"
    echo "=========================================="

    # Set power cap
    if [ "${SKIP_POWERCAP:-0}" = "1" ]; then
        echo "SKIP_POWERCAP=1 set; skipping rocm-smi power cap change."
    else
        sudo rocm-smi --setpoweroverdrive "${POWER_LIMIT}" \
            || { echo "ERROR: failed to set power cap via rocm-smi"; exit 1; }
    fi

    # Stabilization pause
    echo "Pausing ${PAUSE_SECONDS}s to stabilize ..."
    sleep "${PAUSE_SECONDS}"

    # Output file paths
    BENCH_FILE="${LOG_DIR}/benchmark_results_${NUM_GPUS}xmi300x_${POWER_LIMIT}Watts.json"
    ROCM_FILE="${LOG_DIR}/rocm_smi_${NUM_GPUS}xmi300x_${POWER_LIMIT}Watts.CSV"

    # Clean up stale files if any
    rm -f "$BENCH_FILE" "$ROCM_FILE"

    # ------------------ GPU LOGGING (rocm_smi.py via Apptainer) ------------------
    export AMD_SYSFS_LOG="$ROCM_FILE"
    ROCM_SMI_SCRIPT="$REPO_DIR/../training/slurm_scripts/rocm_smi.py"
    apptainer exec --rocm \
        --bind "$LOG_DIR:$LOG_DIR" \
        --bind "$REPO_DIR:$REPO_DIR" \
        --env "AMD_SYSFS_LOG=$AMD_SYSFS_LOG" \
        "$SIF_PATH" \
        python3 "$ROCM_SMI_SCRIPT" &
    LOGGER_PID=$!

    # Run benchmark
    echo "Running sglang.bench_serving (${NUM_PROMPTS} prompts) ..."
    apptainer exec --rocm --bind $TMPDIR "$SIF_PATH" \
        python3 -m sglang.bench_serving \
            --backend sglang \
            --host 0.0.0.0 \
            --port "$SERVER_PORT" \
            --output-file "$BENCH_FILE" \
            --num-prompt "$NUM_PROMPTS" \
            --random-input-len 8192 \
            --random-output-len 256 \
            --seed 42 \
            --dataset-name random

    # Stop GPU logger
    kill "$LOGGER_PID"
    wait "$LOGGER_PID" 2>/dev/null

    echo "Completed: ${POWER_LIMIT}W"
    echo "  Benchmark: $BENCH_FILE"
    echo "  GPU Log:   $ROCM_FILE"
done

echo ""
echo "============================================="
echo "All power caps completed."
echo "Results in: $LOG_DIR"
echo ""
echo "Analyze with:"
echo "  python analysis/analyze_inference_results.py $LOG_DIR --gpu-type mi300x"
echo "============================================="
