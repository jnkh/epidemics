#!/bin/bash
#SBATCH -p general
#SBATCH -n 100
#SBATCH -t 10
#SBATCH --mem-per-cpu 4000
#SBATCH -o log.out
#SBATCH -e log.err

cd ~/physics/research/desai/epidemics/src
julia run_epidemics.jl
