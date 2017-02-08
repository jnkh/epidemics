#!/bin/bash
##SBATCH -J consolidate                  # A single job name for the array
#SBATCH -n 1                       # Number of cores
#SBATCH -N 1                       # All cores on one machine
#SBATCH -p serial_requeue          # Partition
#SBATCH --mem 4000                 # Memory request (4Gb)
#SBATCH -t 0-2:00                  # Maximum execution time (D-HH:MM)
#SBATCH -o consolidate.out        # Standard output
#SBATCH -e consolidate.err        # Standard error

cd ~/physics/research/desai/epidemics/src
julia consolidate_runs.jl
# pwd > test.txt
# echo "${SLURM_ARRAY_TASK_ID}" > t.txt
