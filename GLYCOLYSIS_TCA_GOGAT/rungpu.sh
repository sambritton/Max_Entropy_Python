#!/bin/csh
#$ -M 	brit004@ucr.edu # Email address for job notification
#$ -m  abe		 # Send mail when job begins, ends and aborts
#$ -q  gpu 	 # Specify queue
#$ -l gpu_card=1
#s -pe smp 4         #specifies threads??? maybe
#$ -N  gpu_tca_glycolysis_6_n10_nodata	 # Specify job name
#$ -t 1       #specify number of data input files


module load python/3.7.3
module load pytorch/1.1.0


python ./GLYCOLYSIS_TCA_GOGAT_FUNCTION.py 6 10 0 1e-6	
