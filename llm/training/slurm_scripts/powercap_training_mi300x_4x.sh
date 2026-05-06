#!/bin/bash -l
###############################################################################
# Power-Capped Training Benchmark — 4x MI300X GPUs (AMD)
#
# Sweeps GPU power limits (default: 200–700W) and runs a LLaMA 3 8B
# pre-training workload at each cap. Logs GPU metrics (power, clocks,
# utilization, temperature) at 100ms resolution via rocm_smi.py.
#
# Logging strategy:
#   - Writes per-run logs to node-local scratch (SCRATCH_DIR)
#   - Syncs logs back to the shared filesystem after each power-cap run
#
# After the job completes, analyze results with:
#   python analysis/analyze_powercap_results.py <path_to_job_logs_dir>
###############################################################################

# NOTE on SLURM stdout/stderr output:
# This script intentionally does NOT set "#SBATCH --output".
# By default, SLURM will write stdout/stderr to a file like "slurm-%j.out"
# in the directory where you run "sbatch".
#
# If you want to control stdout/stderr, you can either:
#   1) Edit the header and add a line like:
#        #SBATCH --output=/some/path/training_%x_%j.out
#   2) Override at submit time:
#        sbatch -o /some/path/training_%x_%j.out slurm_scripts/powercap_training_mi300x_4x.sh

#SBATCH --job-name=powercap-training
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --nodelist=aquavan1
#SBATCH --constraint=mi300x
#SBATCH --output=sbatch_training_log.out
# #SBATCH --reservation=<your-reservation>   # Uncomment and set if needed

# ========================= USER CONFIGURATION ================================
# Set these paths to match your environment before submitting.

WORKSPACE="/path/to/your/workspace"                                   # <-- SET THIS TO ROOT OF REPOSITORY
MODEL_DIR="$WORKSPACE/models/Meta-Llama-3-8B/meta-llama/Meta-Llama-3-8B"
REPO_DIR="$WORKSPACE/training"   # Path to this repo's training/ folder
CONTAINER_PATH="$REPO_DIR/litgpt_amd.sif"

# In the current litgpt_amd.sif, litgpt is installed into a venv under /opt/venv.
VENV_PYTHON="/opt/venv/bin/python"
LITGPT_BIN="/opt/venv/bin/litgpt"

# =============================================================================

unset SLURM_EXPORT_ENV
NUM_GPUS=4

# ------------------ POWER LIMIT CONFIG ------------------
POWER_LIMITS=(200 300 400 500 600 700)
PAUSE_SECONDS=${PAUSE_SECONDS:-600}  # Default 10 min; override via env

# ------------------ ENV SETUP ------------------
FINAL_OUTPUT_DIR="${REPO_DIR}/training_logs"

export SCRATCH_DIR=$(mktemp -d)
export OUTPUT_BASE="$SCRATCH_DIR/outputbench"
export CHECKPOINT_PATH="$MODEL_DIR"
export TOKENIZER_PATH="$CHECKPOINT_PATH"
export TRAINING_CONFIG="$REPO_DIR/configs/llama3_8b_4gpu.yaml"

# AMD GPU visibility
export HIP_VISIBLE_DEVICES=0,1,2,3

mkdir -p "$SCRATCH_DIR"

# Job-level logs directory
JOB_BASE="llama_benchmark_${SLURM_JOBID}_${NUM_GPUS}x${SLURM_JOB_PARTITION}_$(date +%Y%m%d_%H%M%S)"
LOCAL_JOB_LOGS_DIR="$SCRATCH_DIR/${JOB_BASE}_logs"
mkdir -p "$LOCAL_JOB_LOGS_DIR"
FINAL_JOB_LOGS_DIR="$FINAL_OUTPUT_DIR/${JOB_BASE}_logs"
mkdir -p "$FINAL_JOB_LOGS_DIR"

echo "Using synthetic data generation for training input"
echo "Using checkpoint directory: $CHECKPOINT_PATH"
echo "Using tokenizer directory: $TOKENIZER_PATH"
echo "Using training config: $TRAINING_CONFIG below:"
cat "$TRAINING_CONFIG"
echo "Local logs directory (on compute node): $LOCAL_JOB_LOGS_DIR"
echo "Final logs directory (shared filesystem): $FINAL_JOB_LOGS_DIR"
echo "HIP_VISIBLE_DEVICES: $HIP_VISIBLE_DEVICES"

# Cache directories on scratch
export HF_DATASETS_CACHE="$SCRATCH_DIR/.cache"
export TRITON_CACHE_DIR="$SCRATCH_DIR/.triton"
export HF_HOME="$SCRATCH_DIR/.cache"

# Write job-level summary
cat > "$LOCAL_JOB_LOGS_DIR/job_info.log" << EOF
=== Job Info ===
Job ID: $SLURM_JOBID
Partition: $SLURM_JOB_PARTITION
Nodes: $SLURM_JOB_NODELIST
Num GPUs: $NUM_GPUS
GPU Type: MI300X
Power Limits: ${POWER_LIMITS[*]}
Pause Seconds: ${PAUSE_SECONDS}
Workspace: $WORKSPACE
Scratch: $SCRATCH_DIR
Data Path: synthetic
Checkpoint Path: $CHECKPOINT_PATH
Training Config: $TRAINING_CONFIG
Container: $CONTAINER_PATH
HIP_VISIBLE_DEVICES: $HIP_VISIBLE_DEVICES
EOF

cat "$TRAINING_CONFIG" >> "$LOCAL_JOB_LOGS_DIR/job_info.log"
cp "$LOCAL_JOB_LOGS_DIR/job_info.log" "$FINAL_JOB_LOGS_DIR/job_info.log"
echo "Initial job info written to local and final destinations"

# Print AMD GPU info
echo "ROCM_HOME: $ROCM_HOME"
echo "PATH: $PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
rocm-smi || echo "rocm-smi not found"

echo "Checking litgpt availability inside container..."
apptainer exec --rocm \
    --bind "$REPO_DIR:$REPO_DIR" \
    "$CONTAINER_PATH" \
    "$VENV_PYTHON" -c 'import sys; import litgpt; print(sys.executable); print(litgpt.__file__)' \
    || { echo "ERROR: litgpt is not importable inside the container ($CONTAINER_PATH) with $VENV_PYTHON. The current image seems to use /opt/venv; verify it exists and contains litgpt."; exit 1; }

# ------------------ LOOP OVER POWER CAPS ------------------
for POWER_LIMIT in "${POWER_LIMITS[@]}"; do
    echo "============================================================"
    echo "Setting GPU power limit to ${POWER_LIMIT}W via rocm-smi"
    echo "============================================================"
    if [ "${SKIP_POWERCAP:-0}" = "1" ]; then
        echo "SKIP_POWERCAP=1 set; skipping rocm-smi power cap change."
    else
        sudo rocm-smi --setpoweroverdrive "${POWER_LIMIT}" \
            || { echo "ERROR: failed to set power cap via rocm-smi"; exit 1; }
    fi
    echo "Pausing ${PAUSE_SECONDS}s to stabilize after cap change..."
    sleep "${PAUSE_SECONDS}"

    # Per-run directories
    RUN_DIR="$SCRATCH_DIR/run_${POWER_LIMIT}Watts"
    mkdir -p "$RUN_DIR"
    OUTPUT="$RUN_DIR/outputbench"
    mkdir -p "$OUTPUT"

    LOCAL_RUN_LOG_DIR="$LOCAL_JOB_LOGS_DIR/${POWER_LIMIT}Watts"
    mkdir -p "$LOCAL_RUN_LOG_DIR"
    FINAL_RUN_LOG_DIR="$FINAL_JOB_LOGS_DIR/${POWER_LIMIT}Watts"
    mkdir -p "$FINAL_RUN_LOG_DIR"

    # ------------------ GPU ALLOCATION LOG ------------------
    cat > "$LOCAL_RUN_LOG_DIR/gpu_allocation.log" << EOF
=== GPU Allocation Info ===
SLURM_JOB_GPUS: $SLURM_JOB_GPUS
SLURM_STEP_GPUS: $SLURM_STEP_GPUS
HIP_VISIBLE_DEVICES: $HIP_VISIBLE_DEVICES
ROCR_VISIBLE_DEVICES: $ROCR_VISIBLE_DEVICES
SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST
Power Limit: ${POWER_LIMIT}W
Scratch Dir: $RUN_DIR
EOF

    # ------------------ GPU LOGGING (rocm_smi.py via Apptainer) ------------------
    export AMD_SYSFS_LOG="$LOCAL_RUN_LOG_DIR/rocm_smi_gpu_usage.csv"
    ROCM_SMI_SCRIPT="$REPO_DIR/slurm_scripts/rocm_smi.py"
    apptainer exec --rocm \
        --bind "$SCRATCH_DIR:$SCRATCH_DIR" \
        --bind "$REPO_DIR:$REPO_DIR" \
        --env "AMD_SYSFS_LOG=$AMD_SYSFS_LOG" \
        "$CONTAINER_PATH" \
        "$VENV_PYTHON" "$ROCM_SMI_SCRIPT" &
    LOGGER_PID=$!

    # ------------------ APPTAINER EXEC ------------------
    cd "$RUN_DIR"
    srun --export=ALL apptainer exec --rocm \
        --bind "$SCRATCH_DIR:$SCRATCH_DIR" \
        --bind "$CHECKPOINT_PATH:$CHECKPOINT_PATH" \
        --bind "$OUTPUT:$OUTPUT" \
        --bind "$REPO_DIR:$REPO_DIR" \
        --bind "$WORKSPACE:$WORKSPACE" \
        --pwd "$RUN_DIR" \
        "$CONTAINER_PATH" \
        "$LITGPT_BIN" pretrain Meta-Llama-3-8B \
            --config "$TRAINING_CONFIG" \
            --initial_checkpoint_dir "$CHECKPOINT_PATH" \
            --tokenizer_dir "$TOKENIZER_PATH"

    # ------------------ STOP LOGGING ------------------
    kill $LOGGER_PID
    wait $LOGGER_PID 2>/dev/null

    # ------------------ COLLECT METRICS ------------------
    METRICS_FILE=$(find "$RUN_DIR" -name "metrics.csv" -type f 2>/dev/null | head -1)

    if [ -n "$METRICS_FILE" ]; then
        echo "Found metrics file: $METRICS_FILE"
        cp "$METRICS_FILE" "$LOCAL_RUN_LOG_DIR/metrics.csv"
    else
        echo "Warning: metrics.csv not found in $RUN_DIR"
        echo "Searching for any csv files..."
        find "$RUN_DIR" -name "*.csv" -type f
    fi

    # ------------------ SYNC LOGS TO SHARED FILESYSTEM ------------------
    echo "Syncing logs from local scratch to shared filesystem..."
    echo "Source: $LOCAL_RUN_LOG_DIR"
    echo "Destination: $FINAL_RUN_LOG_DIR"

    rsync -av "$LOCAL_RUN_LOG_DIR/" "$FINAL_RUN_LOG_DIR/"

    echo "Logs synced to: $FINAL_RUN_LOG_DIR"
    echo "Contents:"
    ls -la "$FINAL_RUN_LOG_DIR/"
done

# Reset power overdrive to default after the sweep
echo "Resetting GPU power overdrive to default..."
if [ "${SKIP_POWERCAP:-0}" = "1" ]; then
    echo "SKIP_POWERCAP=1 set; skipping rocm-smi reset."
else
    sudo rocm-smi --resetpoweroverdrive || true
fi

# After the job completes, run the analysis script:
# python analysis/analyze_powercap_results.py <FINAL_JOB_LOGS_DIR>
