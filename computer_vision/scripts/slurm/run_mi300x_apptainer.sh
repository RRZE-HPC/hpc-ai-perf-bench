#!/bin/bash -l
#SBATCH --job-name=sd_benchmark
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --error=outputs/slurm-%j.err
#SBATCH --time=1:45:00

unset SLURM_EXPORT_ENV

export https_proxy="http://proxy.nhr.fau.de:80"
export MIOPEN_DISABLE_CACHE=1
export TORCH_NCCL_ENABLE_MONITORING=0

# Define the path to the Apptainer image
APPTAINER_IMAGE="$WORK/AMD/amdMPI.sif"

# Change to the project directory (adjust if necessary inside/outside container)
cd /home/hpc/unrz/unrz109h/AIPerf/ai_benchmark/computer_vision
apptainer shell $APPTAINER_IMAGE

# Create outputs directory if it doesn't exist
mkdir -p outputs

# Get arguments
SCRIPT=$1      # Path to the python script to run inside the container
ARGS=$2        # Script arguments (e.g., --batch_size 2)
DEVICES=${3:-1}     # Default 1 GPU (Note: GPU handling might need specific apptainer flags)
DEVICE_LIST=$(seq -s, 0 $((DEVICES-1)))

export ROCR_VISIBLE_DEVICES=$DEVICE_LIST
echo $ROCR_VISIBLE_DEVICES

echo "Using Apptainer image: $APPTAINER_IMAGE"
echo "Running: $SCRIPT with args: $ARGS"
echo "Hardware: $DEVICES GPUs per node"

# Run the training script with srun and apptainer
apptainer exec --rocm $APPTAINER_IMAGE python3 $SCRIPT $ARGS


#srun apptainer exec --rocm $APPTAINER_IMAGE python3 $SCRIPT $ARGS

# srun apptainer exec --rocm --no-mount=proc --env LD_LIBRARY_PATH=/opt/conda/envs/py_3.10/lib:/lib:/usr/lib $APPTAINER_IMAGE python3 $SCRIPT $ARGS

# srun apptainer exec --rocm $APPTAINER_IMAGE \
#     bash -c 'export LD_LIBRARY_PATH=/opt/conda/envs/py_3.10/lib:/lib:/usr/lib && python3 "$@"' bash $SCRIPT $ARGS
