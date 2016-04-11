#!/bin/bash
#SBATCH -p general
#SBATCH -n 64
#SBATCH -t 0-05:00
#SBATCH --mem-per-cpu 8000
#SBATCH -o log.out
#SBATCH -e log.err

cd ~/physics/research/desai/epidemics/src
julia run_epidemics.jl
