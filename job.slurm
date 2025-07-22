#!/bin/bash

# Resource Allocation
#SBATCH --job-name=arc_solver
#SBATCH --partition=compute
#SBATCH --output=/mnt/storage/admindi/home/phsilva/slurm_outputs/%x_%a.out
#SBATCH --error=/mnt/storage/admindi/home/phsilva/slurm_outputs/%x_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=1:00:00

# Job Array
#SBATCH --array=1-400

SEED=1

# Environment Setup
export PYTHONPATH=/mnt/storage/admindi/home/phsilva/aeon:$PYTHONPATH
echo "PYTHONPATH set to: $PYTHONPATH"

# Job Execution
# Get the specific task ID for this job instance from the task_list.txt file.
TASK_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" task_list.txt)

echo "------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOB_ID"
echo "SLURM ARRAY TASK ID: $SLURM_ARRAY_TASK_ID"
echo "Assigned ARC Task ID: $TASK_ID"
echo "Assigned Seed: $SEED"
echo "------------------------------------------------"

# Run the Python solver, now passing the configured seed
/mnt/storage/admindi/home/phsilva/miniconda3/envs/aeon/bin/python script.py --task_id "$TASK_ID" --seed "$SEED" --output_dir "arc_results"

echo "Job finished."
