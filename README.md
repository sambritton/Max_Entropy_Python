# Max_Entropy_Python

This code compares two regulations methods for different pathways. 
Each pathway is in a folder. The functions utilized are in Basic_Functions
Equilibrator parses reactions from .dat folders
machine_learning_functions contains all functions to run a reinforcement learning program.
max_entropy_functions contains all functions to run a maximume entropy regulation program. 

To run:
Place all the code folders into a new folder, "new_folder". 
cd into new_folder inside python or make it your current working directory. 
Open one of the python files for a pathway, e.g. "GLUCONEOGENSIS"
Using "new_folder" as your cwd you can run the python file. 


To run with slurm:
cd into one of the pathway folders
type "sbatch SBATCH.sh"
the script runs multiple jobs for each n value. The simulation id's are in array. To run one, use --array=1, to run more use --array=1,2,...k, to run simulations 1-k for a specific n value. 

Necessary Arguments:
1: simulation number
2: n value

Additional Arguments:
3: use metabolite data (0 uses rule of thumb, 1 uses data) default 0
4: learning rate (1e-6, 1e-7, 1e-8) default 1e-8
5: epsilon greedy init (0-1) default 0.5
6: epsilon threshold (episodes before eps=eps/2) default 25
7: gamma (0-1) default 0.9 


To run a short simulation, cd to the GLYCOLYSIS_TCA_GOGAT folder. Use the command:
python ./GLYCOLYSIS_TCA_GOGAT_FUNCTION 1 4 0 1e-06 0.05 2 
