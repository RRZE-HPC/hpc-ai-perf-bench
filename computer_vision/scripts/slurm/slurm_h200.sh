#!/bin/bash -l

# Default values
BATCH_SIZE=${2:-2}  # Default batch size is 2
# MODEL=${3:-"vit_l_16"}
DEVICES=${3:-1}     # Default 1 GPU
NODES=${4:-1}       # Default 1 node
TASKS_PER_NODE=${5:-$DEVICES}       # Default should be equal to DEVICES

echo "Running with: batch_size=$BATCH_SIZE, devices=$DEVICES, nodes=$NODES"

#  
# --ntasks=$((NODES * DEVICES))
# Submit the job
sbatch --gres=gpu:h200:$DEVICES --exclusive --ntasks-per-node=$TASKS_PER_NODE --partition=h200 --nodes=$NODES \
    scripts/slurm/run_helma_apptainer.sh $1 "--batch_size $BATCH_SIZE" $DEVICES $NODES
