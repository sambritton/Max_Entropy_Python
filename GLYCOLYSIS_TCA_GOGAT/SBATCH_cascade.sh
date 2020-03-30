#!/bin/bash -l
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --job-name="glycolysis"
#SBATCH --nodes=1 #number of physical nodes, not core
#SBATCH --cpus-per-task=12 #number of cores per MPI rank
#SBATCH --time=168:00:00
#SBATCH --array=1,2,3,4,5,6,7,8,9,10
#SBATCH --account="emsla50173"
module purge

module purge
module load python/anaconda3

srun /home/scicons/cascade/apps/python/3.6/bin/python ./GLYCOLYSIS_TCA_GOGAT_FUNCTION.py $SLURM_ARRAY_TASK_ID 2 1 1e-04 

