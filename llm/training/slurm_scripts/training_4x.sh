#!/bin/bash -l
###############################################################################
# Training Smoke Test — single run, no power capping
#
# Use this to verify the container, model weights, and training config work
# before running the full power-cap sweep.
#
# After completion, check training_logs/ for metrics.csv and gpu usage CSV.
###############################################################################

#SBATCH --job-name=training-test
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --partition=h200
#SBATCH --gres=gpu:4
#SBATCH --output=sbatch_training_test_%j.out

# ========================= USER CONFIGURATION ================================

WORKSPACE="$SLURM_SUBMIT_DIR"   # directory where sbatch is called (must be llm/)
MODEL_DIR="$WORKSPACE/models/Meta-Llama-3-8B/meta-llama/Meta-Llama-3-8B"
REPO_DIR="$WORKSPACE/training"
CONTAINER_PATH="$REPO_DIR/litgpt.sif"
TRAINING_CONFIG="$REPO_DIR/configs/llama3_8b_4gpu.yaml"

# =============================================================================

unset SLURM_EXPORT_ENV

# Pre-flight checks
if [ ! -f "$CONTAINER_PATH" ]; then
    echo "ERROR: Container not found: $CONTAINER_PATH"
    echo "  Build it with: apptainer build training/litgpt.sif training/litgpt.def"
    exit 1
fi
if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model weights not found: $MODEL_DIR"
    echo "  Download with: HF_TOKEN=<token> huggingface-cli download meta-llama/Meta-Llama-3-8B --local-dir models/Meta-Llama-3-8B/meta-llama/Meta-Llama-3-8B"
    exit 1
fi

FINAL_OUTPUT_DIR="${REPO_DIR}/training_logs"
export SCRATCH_DIR=$(mktemp -d)
export CHECKPOINT_PATH="$MODEL_DIR"
export TOKENIZER_PATH="$CHECKPOINT_PATH"

mkdir -p "$SCRATCH_DIR"

# Job-level logs directory
JOB_BASE="training_test_${SLURM_JOBID}_$(date +%Y%m%d_%H%M%S)"
LOCAL_JOB_LOGS_DIR="$SCRATCH_DIR/${JOB_BASE}"
FINAL_JOB_LOGS_DIR="$FINAL_OUTPUT_DIR/${JOB_BASE}"
mkdir -p "$LOCAL_JOB_LOGS_DIR"
mkdir -p "$FINAL_JOB_LOGS_DIR"

echo "============================================="
echo "Training Smoke Test"
echo "============================================="
echo "Job ID:       $SLURM_JOBID"
echo "Nodes:        $SLURM_JOB_NODELIST"
echo "Checkpoint:   $CHECKPOINT_PATH"
echo "Config:       $TRAINING_CONFIG"
echo "Scratch:      $SCRATCH_DIR"
echo "Logs:         $FINAL_JOB_LOGS_DIR"
echo "============================================="
cat "$TRAINING_CONFIG"
echo "============================================="

export HF_DATASETS_CACHE="$SCRATCH_DIR/.cache"
export TRITON_CACHE_DIR="$SCRATCH_DIR/.triton"
export HF_HOME="$SCRATCH_DIR/.cache"

# GPU info at start
nvidia-smi

# Start GPU metrics logger
nvidia-smi --query-gpu=timestamp,index,name,uuid,pci.bus_id,power.draw,clocks.sm,clocks.mem,utilization.gpu,temperature.gpu \
    --format=csv,nounits -lms 100 -f "$LOCAL_JOB_LOGS_DIR/nvidia_smi_gpu_usage.csv" &
LOGGER_PID=$!

# Run training
cd "$SCRATCH_DIR"
srun --export=ALL apptainer exec --nv \
    --bind "$SCRATCH_DIR:$SCRATCH_DIR" \
    --bind "$CHECKPOINT_PATH:$CHECKPOINT_PATH" \
    --bind "$REPO_DIR:$REPO_DIR" \
    --bind "$WORKSPACE:$WORKSPACE" \
    --bind "$REPO_DIR/litgpt:/workspace/litgpt" \
    --pwd "$SCRATCH_DIR" \
    "$CONTAINER_PATH" \
    python /usr/local/bin/litgpt pretrain Meta-Llama-3-8B \
        --config "$TRAINING_CONFIG" \
        --initial_checkpoint_dir "$CHECKPOINT_PATH" \
        --tokenizer_dir "$TOKENIZER_PATH"

TRAIN_EXIT=$?

# Stop GPU logger
kill $LOGGER_PID
wait $LOGGER_PID 2>/dev/null

# Collect metrics.csv
METRICS_FILE=$(find "$SCRATCH_DIR" -name "metrics.csv" -type f 2>/dev/null | head -1)
if [ -n "$METRICS_FILE" ]; then
    echo "Found metrics file: $METRICS_FILE"
    cp "$METRICS_FILE" "$LOCAL_JOB_LOGS_DIR/metrics.csv"
else
    echo "Warning: metrics.csv not found"
fi

# Sync logs to shared filesystem
rsync -av "$LOCAL_JOB_LOGS_DIR/" "$FINAL_JOB_LOGS_DIR/"

echo "============================================="
echo "Done. Exit code: $TRAIN_EXIT"
echo "Logs: $FINAL_JOB_LOGS_DIR"
echo "============================================="
exit $TRAIN_EXIT
