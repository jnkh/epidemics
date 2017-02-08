#!/bin/bash
##SBATCH -J test_array                  # A single job name for the array
#SBATCH -n 1                       # Number of cores
#SBATCH -N 1                       # All cores on one machine
#SBATCH -p serial_requeue          # Partition
#SBATCH --mem 4000                 # Memory request (4Gb)
#SBATCH -t 0-0:10                  # Maximum execution time (D-HH:MM)
#SBATCH -o test_%A_%a.out        # Standard output
#SBATCH -e test_%A_%a.err        # Standard error

DATE_STR="$(date +%Y%m%d%H%M%S)"
PREFIX="/n/regal/desai_lab/juliankh/tmp"
mkdir "${PREFIX}/${SLURM_ARRAY_TASK_ID}"
cd "${PREFIX}/${SLURM_ARRAY_TASK_ID}"
cp -r ~/juliankh/physics/research/desai/epidemics/src ./src
julia run_epidemics.jl
# pwd > test.txt
# echo "${SLURM_ARRAY_TASK_ID}" > t.txt