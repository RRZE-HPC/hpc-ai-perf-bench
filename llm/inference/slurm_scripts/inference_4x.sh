#!/bin/bash -l
###############################################################################
# Inference Smoke Test — SGLang Serving (no power capping)
#
# Use this to verify the container, model weights, and server start correctly
# before running the full power-cap sweep.
#
# Starts an SGLang server and runs a single sglang.bench_serving pass.
# Submit from inside the llm/ directory:
#   cd /path/to/hpc-ai-perf-bench/llm
#   sbatch inference/slurm_scripts/inference_4x.sh
###############################################################################

#SBATCH --job-name=inference-test
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --partition=h200
#SBATCH --gres=gpu:4
#SBATCH --output=sbatch_inference_test_%j.out
# #SBATCH --reservation=<your-reservation>   # Uncomment and set if needed

# ========================= USER CONFIGURATION ================================

WORKSPACE="$SLURM_SUBMIT_DIR"   # automatically set to the directory where sbatch is called
MODEL_DIR="$WORKSPACE/models/Meta-Llama-3-8B/meta-llama/Meta-Llama-3-8B"
REPO_DIR="$WORKSPACE/inference"
SIF_PATH="$REPO_DIR/sglang_v0.4.4.post1-cu124.sif"

NUM_GPUS=4       # Must match --gres and --ntasks above
NUM_PROMPTS=256  # Reduced for smoke test
SERVER_PORT=6000

# =============================================================================

unset SLURM_EXPORT_ENV

# ------------------ PRE-FLIGHT CHECKS ------------------
if [ ! -f "$SIF_PATH" ]; then
    echo "ERROR: Container not found: $SIF_PATH"
    echo "Pull it with: apptainer pull docker://lmsysorg/sglang:v0.4.4.post1-cu124"
    exit 1
fi

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory not found: $MODEL_DIR"
    echo "Download with: huggingface-cli download meta-llama/Meta-Llama-3-8B --local-dir models/Meta-Llama-3-8B/meta-llama/Meta-Llama-3-8B"
    exit 1
fi

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
echo "Inference Smoke Test"
echo "============================================="
echo "Job ID:      $SLURM_JOBID"
echo "Partition:   $SLURM_JOB_PARTITION"
echo "Nodes:       $SLURM_JOB_NODELIST"
echo "Num GPUs:    $NUM_GPUS"
echo "Num Prompts: $NUM_PROMPTS"
echo "Model:       $MODEL_DIR"
echo "Container:   $SIF_PATH"
echo "Log Dir:     $LOG_DIR"
echo "============================================="

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

# ------------------ RUN BENCHMARK ------------------
BENCH_FILE="${LOG_DIR}/benchmark_results.json"

echo ""
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

echo ""
echo "============================================="
echo "Smoke test completed."
echo "Results: $BENCH_FILE"
echo "============================================="
