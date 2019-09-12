#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name="GLYCOLYSIS_n2_nodata"
#SBATCH -p gpu # This is the default partition, you can use any of the following; intel, batch, highmem, gpu
#SBATCH -N 1
#SBATCH -p shared
#SBATCH --gres=gpu:1
#SBATCH -n 8
#SBATCH --time=70:00:00
#SBATCH --array=1

module purge
module load python/anaconda3

srun python ./GLYCOLYSIS_TCA_GOGAT_FUNCTION.py $SLURM_ARRAY_TASK_ID 2 

