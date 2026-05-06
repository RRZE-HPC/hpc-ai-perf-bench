#!/bin/bash -l
#SBATCH --job-name=sd_benchmark
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --error=outputs/slurm-%j.err
#SBATCH --time=1:59:00

unset SLURM_EXPORT_ENV

export https_proxy="http://proxy.nhr.fau.de:80"

STORAGE_DIR="$(ws_find h_benchmark)"

cd $STORAGE_DIR/ai_benchmark/computer_vision

# Create outputs directory if it doesn't exist
mkdir -p outputs

# Get arguments
SCRIPT=$1           # Path to main_new.py
ARGS=$2             # Script arguments (e.g., --batch_size 2)
DEVICES=${3:-1}     # Default 1 GPU
NODES=${4:-1}       # Default 1 node

echo "Running: $SCRIPT with args: $ARGS"
echo "Hardware: $DEVICES GPUs per node, $NODES nodes"

# Select container based on xformers option
CONTAINER_PATH="$STORAGE_DIR/nvidia_cuda12_9_xformers.sif"
echo "Using xformers container: $CONTAINER_PATH"

# Check if container exists
if [[ ! -f "$CONTAINER_PATH" ]]; then
    echo "Error: Container not found at $CONTAINER_PATH"
    echo "Please build the container first using the appropriate .def file"
    exit 1
fi

# # Run the training script with srun
# srun python3 $SCRIPT $ARGS

# Multi-GPU DDP launch (adjust cpus-per-task as needed)
srun --kill-on-bad-exit=1 \
    apptainer exec --nv \
        "$CONTAINER_PATH" \
        python3 $SCRIPT $ARGS
