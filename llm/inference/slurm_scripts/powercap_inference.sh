#!/bin/bash -l
###############################################################################
# Power-Capped Inference Benchmark — SGLang Serving
#
# Self-contained SLURM job that:
#   1. Starts an SGLang inference server (tensor-parallel across GPUs)
#   2. Runs a warmup pass
#   3. Sweeps GPU power limits (default: 200–700W)
#   4. At each cap: logs GPU metrics at 100ms resolution, runs sglang.bench_serving
#
# All output (benchmark JSON + nvidia-smi CSV) is written to a timestamped
# log directory under <WORKSPACE>/inference_logs/.
#
# After the job completes, analyze results with:
#   python analysis/analyze_inference_results.py <path_to_job_logs_dir> --gpu-type <partition>
###############################################################################

#SBATCH --job-name=powercap-inference
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --partition=h200
#SBATCH --gres=gpu:4
#SBATCH --output=sbatch_inference_log.out
# #SBATCH --reservation=<your-reservation>   # Uncomment and set if needed

# ========================= USER CONFIGURATION ================================
# Submit this script from inside the llm/ directory:
#   cd /path/to/hpc-ai-perf-bench/llm
#   sbatch inference/slurm_scripts/powercap_inference.sh

WORKSPACE="$SLURM_SUBMIT_DIR"   # automatically set to the directory where sbatch is called
MODEL_DIR="$WORKSPACE/models/Meta-Llama-3-8B/meta-llama/Meta-Llama-3-8B"
SIF_PATH="$WORKSPACE/inference/sglang_v0.4.4.post1-cu124.sif"
REPO_DIR="$WORKSPACE/inference"                                       # Path to this repo's inference/ folder

NUM_GPUS=4                    # Must match --gres and --ntasks above
NUM_PROMPTS=2048              # Number of prompts per benchmark run
SERVER_PORT=6000              # SGLang server port
POWER_LIMITS=(200 300 400 500 600 700)
PAUSE_SECONDS=${PAUSE_SECONDS:-600}  # Default 10 min; override via env

# =============================================================================

unset SLURM_EXPORT_ENV

# Verify CUDA is visible inside the container before starting the server
echo "Checking GPU visibility inside container..."
GPU_COUNT=$(apptainer exec --nv "$SIF_PATH" python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
if [ "$GPU_COUNT" -eq 0 ] 2>/dev/null; then
    echo "ERROR: No CUDA GPUs visible inside container (torch.cuda.device_count()=0)"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    nvidia-smi -L
    exit 1
fi
echo "GPUs visible inside container: $GPU_COUNT"
apptainer exec --nv "$SIF_PATH" python3 -c "import torch; [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]" 2>/dev/null

# ------------------ OUTPUT DIRECTORY ------------------
OUTPUT_BASE="${WORKSPACE}/inference_logs"
JOB_TAG="inference_${SLURM_JOBID}_${NUM_GPUS}x${SLURM_JOB_PARTITION}_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${OUTPUT_BASE}/${JOB_TAG}"
mkdir -p "$LOG_DIR"

echo "============================================="
echo "Power-Capped Inference Benchmark"
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
echo "============================================="

# Write job metadata
cat > "$LOG_DIR/job_info.log" << EOF
=== Job Info ===
Job ID: $SLURM_JOBID
Partition: $SLURM_JOB_PARTITION
Nodes: $SLURM_JOB_NODELIST
Num GPUs: $NUM_GPUS
Num Prompts: $NUM_PROMPTS
Power Limits: ${POWER_LIMITS[*]}
Pause Seconds: $PAUSE_SECONDS
Model: $MODEL_DIR
Container: $SIF_PATH
Dataset: random synthetic prompts
Log Directory: $LOG_DIR
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
}
trap cleanup EXIT

# ------------------ START SGLANG SERVER ------------------
echo ""
echo "Starting SGLang server (TP=$NUM_GPUS) ..."
apptainer exec --nv \
    --bind "$MODEL_DIR:$MODEL_DIR" \
    --bind "$TMPDIR:$TMPDIR" \
    "$SIF_PATH" \
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
sudo nvidia-smi --power-limit=200
apptainer exec \
    --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    --bind "$TMPDIR:$TMPDIR" \
    --bind "$MODEL_DIR:$MODEL_DIR" \
    --bind "$REPO_DIR/sglang/bench_serving.py:/sgl-workspace/sglang/python/sglang/bench_serving.py" \
    "$SIF_PATH" \
    python3 -m sglang.bench_serving \
        --backend sglang \
        --disable-tqdm \
        --host 0.0.0.0 \
        --port "$SERVER_PORT" \
        --tokenizer "$MODEL_DIR" \
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
    sudo nvidia-smi --power-limit="${POWER_LIMIT}"

    # Stabilization pause
    echo "Pausing ${PAUSE_SECONDS}s to stabilize ..."
    sleep "${PAUSE_SECONDS}"

    # Output file paths
    BENCH_FILE="${LOG_DIR}/benchmark_results_${NUM_GPUS}x${SLURM_JOB_PARTITION}_${POWER_LIMIT}Watts.json"
    NVIDIA_FILE="${LOG_DIR}/nvidia_smi_${NUM_GPUS}x${SLURM_JOB_PARTITION}_${POWER_LIMIT}Watts.CSV"

    # Clean up stale files if any
    rm -f "$BENCH_FILE" "$NVIDIA_FILE"

    # Start GPU metrics logger (100ms resolution)
    nvidia-smi \
        --query-gpu=timestamp,index,name,uuid,pci.bus_id,power.draw,clocks.sm,clocks.mem,utilization.gpu,temperature.gpu \
        --format=csv,nounits -lms 100 -f "$NVIDIA_FILE" &
    LOGGER_PID=$!

    # Run benchmark
    echo "Running sglang.bench_serving (${NUM_PROMPTS} prompts) ..."
    apptainer exec \
        --env HF_HUB_OFFLINE=1 \
        --env TRANSFORMERS_OFFLINE=1 \
        --bind "$TMPDIR:$TMPDIR" \
        --bind "$LOG_DIR:$LOG_DIR" \
        --bind "$MODEL_DIR:$MODEL_DIR" \
        --bind "$REPO_DIR/sglang/bench_serving.py:/sgl-workspace/sglang/python/sglang/bench_serving.py" \
        "$SIF_PATH" \
        python3 -m sglang.bench_serving \
            --backend sglang \
            --host 0.0.0.0 \
            --port "$SERVER_PORT" \
            --tokenizer "$MODEL_DIR" \
            --output-file "$BENCH_FILE" \
            --num-prompt "$NUM_PROMPTS" \
            --random-input-len 1024 \
            --random-output-len 256 \
            --seed 42 \
            --dataset-name random

    # Stop GPU logger
    kill "$LOGGER_PID"
    wait "$LOGGER_PID" 2>/dev/null

    echo "Completed: ${POWER_LIMIT}W"
    echo "  Benchmark: $BENCH_FILE"
    echo "  GPU Log:   $NVIDIA_FILE"
done

echo ""
echo "============================================="
echo "All power caps completed."
echo "Results in: $LOG_DIR"
echo ""
echo "Analyze with:"
echo "  python analysis/analyze_inference_results.py $LOG_DIR --gpu-type ${SLURM_JOB_PARTITION}"
echo "============================================="
