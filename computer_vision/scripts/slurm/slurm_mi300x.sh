#!/bin/bash -l

# Default values
BATCH_SIZE=${2:-2}  # Default batch size is 2
# MODEL=${3:-"vit_l_16"}
DEVICES=${3:-1}     # Default 1 GPU

echo "Running with: batch_size=$BATCH_SIZE, devices=$DEVICES"

#  
# --ntasks=$((NODES * DEVICES))
# Submit the job
# Submit the job
sbatch -w aquavan1 --ntasks-per-node=$DEVICES \
    scripts/slurm/run_mi300x.sh $1 "--batch_size $BATCH_SIZE" $DEVICES