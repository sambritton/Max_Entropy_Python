#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name="CELLWALL_n6_data_1e6"
#SBATCH -p gpu 
#SBATCH -N 1
#SBATCH -p shared
#SBATCH --gres=gpu:1
#SBATCH -n 8
#SBATCH --time=70:00:00
#SBATCH --array=1,2,3,4,5,6,7,8,9,10

module purge
module load python/anaconda3

srun python ./TCA_PPP_GLYCOLYSIS_CELLWALL_FUNCTION.py $SLURM_ARRAY_TASK_ID 6 1 1e-06 

