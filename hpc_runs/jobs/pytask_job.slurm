#!/bin/bash
#SBATCH -o log.%x.%j.%N.out
#SBATCH -D .
#SBATCH -J psycourse_pytask
#SBATCH -p cm4_inter
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --hint=nomultithread
#SBATCH --mail-type=none
#SBATCH --time=06:00:00

# ensure pixi is available
export PATH="$HOME/.pixi/bin:$PATH"

# Avoid BLAS oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

srun --cpu-bind=cores pixi run pytask -n "${SLURM_CPUS_PER_TASK:-1}" --verbose 2
