#!/bin/csh
#$ -M @nd.edu
#$ -N  cpu_cellwall_1_n16_nodata	 # Specify job name
#$ -m abe
#$ -r y

module load python/3.7.3
module load pytorch/1.1.0


python ./TCA_PPP_GLYCOLYSIS_CELLWALL_FUNCTION.py 1 16 0	
