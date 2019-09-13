#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name="GLYCOLYSIS_n2_nodata"
#SBATCH -p gpu 
#SBATCH -N 1
#SBATCH -p shared
#SBATCH --gres=gpu:1
#SBATCH -n 8
#SBATCH --time=94:00:00
#SBATCH --array=1

module purge
module load python/anaconda3

srun python ./TCA_PPP_GLYCOLYSIS_CELLWALL_FUNCTION.py $SLURM_ARRAY_TASK_ID 2 0

