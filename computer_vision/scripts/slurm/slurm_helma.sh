#!/bin/bash -l

# Default values
BATCH_SIZE=${2:-"2,4,8"}  # Default batch sizes (comma-separated)
PARTITION=${3:-"h100"}  # Default partition (h100/h200)
NODES=${4:-1}       # Default 1 node
DEVICES=${5:-1}     # Default 1 GPU
TASKS_PER_NODE=${6:-$DEVICES}       # Default should be equal to DEVICES

echo "Running with: batch_size=$BATCH_SIZE, devices=$DEVICES, nodes=$NODES, partition=$PARTITION"

#
# --ntasks=$((NODES * DEVICES))
# Submit the job

# Build arguments string
ARGS="--batch_size $BATCH_SIZE"

sbatch --reservation=benchmark-v111dc15 --gres=gpu:$PARTITION:$DEVICES --exclusive --ntasks-per-node=$TASKS_PER_NODE --partition=$PARTITION --nodes=$NODES \
    scripts/slurm/run_helma_apptainer.sh $1 "$ARGS" $DEVICES $NODES


# How to run it
# sh scripts/slurm/slurm_helma.sh main_ldm.py "2,4,8" h100 1 4
# Single batch size: sh scripts/slurm/slurm_helma.sh main_ldm.py "48" h100 1 4