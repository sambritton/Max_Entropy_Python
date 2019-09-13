#!/bin/csh
#$ -M @nd.edu
#$ -N  cpu_tca_glycolysis_1_n4_nodata	 # Specify job name
#$ -m abe
#$ -r y

module load python/3.7.3
module load pytorch/1.1.0


python ./GLYCOLYSIS_TCA_GOGAT_FUNCTION.py 1 4 0	
