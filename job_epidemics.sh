#!/bin/bash
#SBATCH -p general
#SBATCH -n 128
#SBATCH -t 0-8:00
#SBATCH --mem-per-cpu 4000
#SBATCH -o log.out
#SBATCH -e log.err

cd ~/physics/research/desai/epidemics/src
julia run_epidemics.jl
