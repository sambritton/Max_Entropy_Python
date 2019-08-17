#!/usr/bin/env python
# coding: utf-8

# # Table of Contents

# # Develop Thermodynamic-kinetic Maximum Entropy Model

#cd Documents/cannon/Reaction_NoOxygen/Python_Notebook/
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import multiprocessing as mp

from scipy import stats
from scipy.optimize import curve_fit
import os
import re
import time

from PIL import Image
import matplotlib.image as mpimg
from IPython.display import display

import sys

cwd = os.getcwd()
print (cwd)

sys.path.insert(0, cwd+'\\Basic_Functions')
sys.path.insert(0, cwd+'\\GLYCOLYSIS_TCA_GOGAT')
sys.path.insert(0, cwd+'\\Basic_Functions\\equilibrator-api-v0.1.8\\build\\lib')

import max_entropy_functions
import machine_learning_functions


pd.set_option('display.max_columns', None,'display.max_rows', None)
#from ipynb_latex_setup import *

T = 298.15
R = 8.314e-03
RT = R*T
N_avogadro = 6.022140857e+23
VolCell = 1.0e-15
Concentration2Count = N_avogadro * VolCell
concentration_increment = 1/(N_avogadro*VolCell)


np.set_printoptions(suppress=True)#turn off printin
# In[3]:


with open( cwd + '\\GLYCOLYSIS_TCA_GOGAT\\GLYCOLYSIS_TCA_GOGAT.dat', 'r') as f:
  print(f.read())
  

# In[5]:


fdat = open(cwd + '\\GLYCOLYSIS_TCA_GOGAT\\GLYCOLYSIS_TCA_GOGAT.dat', 'r')
f#dat = open('TCA_PPP_Glycolysis.dat', 'r')

left ='LEFT'
right = 'RIGHT'
left_compartment = 'LEFT_COMPARTMENT'
right_compartment = 'RIGHT_COMPARTMENT'
enzyme_level = 'ENZYME_LEVEL'
deltag0 = 'DGZERO'
deltag0_sigma = 'DGZERO StdDev'
same_compartment = 'Same Compartment?'
full_rxn = 'Full Rxn'

reactions = pd.DataFrame(index=[],columns=[left, right, left_compartment, right_compartment, enzyme_level, deltag0, deltag0_sigma, same_compartment,full_rxn])
reactions.index.name='REACTION'
S_matrix = pd.DataFrame(index=[],columns=[enzyme_level])
S_matrix.index.name='REACTION'

for line in fdat:
    if (line.startswith('REACTION')):
        rxn_name = line[9:-1].lstrip()
        S_matrix.loc[rxn_name,enzyme_level] = 1.0
        reactions.loc[rxn_name,enzyme_level] = 1.0

    if (re.match("^LEFT\s",line)):
        line = line.upper()
        left_rxn = line[4:-1].lstrip()
        left_rxn = re.sub(r'\s+$', '', left_rxn) #Remove trailing white space
        reactions.loc[rxn_name,left] = left_rxn

    elif (re.match('^RIGHT\s',line)):
        line = line.upper()
        right_rxn = line[5:-1].lstrip()
        right_rxn = re.sub(r'\s+$', '', right_rxn) #Remove trailing white space
        reactions.loc[rxn_name,right] = right_rxn
        
    elif (line.startswith(left_compartment)):
        cpt_name = line[16:-1].lstrip()
        reactions.loc[rxn_name,left_compartment] = cpt_name
        reactants = re.split(' \+ ',left_rxn)
        for idx in reactants:
            values = re.split(' ', idx);
            if len(values) == 2:
                stoichiometry = np.float64(values[0]);
                molecule = values[1];
                if not re.search(':',molecule):
                    molecule = molecule + ':' + cpt_name
            else:
                stoichiometry = np.float64(-1.0);
                molecule = values[0]; 
                if not re.search(':',molecule):
                    molecule = molecule + ':' + cpt_name
            S_matrix.loc[rxn_name,molecule] = stoichiometry;


    elif (line.startswith(right_compartment)):
        cpt_name = line[17:-1].lstrip()
        reactions.loc[rxn_name,right_compartment] = cpt_name
        products = re.split(' \+ ',right_rxn)
        for idx in products:
            values = re.split(' ', idx);
            if len(values) == 2:
                stoichiometry = np.float64(values[0]);
                molecule = values[1];
                if not re.search(':',molecule):
                    molecule = molecule + ':' + cpt_name
            else:
                stoichiometry = np.float64(1.0);
                molecule = values[0];
                if not re.search(':',molecule):
                    molecule = molecule + ':' + cpt_name
            S_matrix.loc[rxn_name,molecule] = stoichiometry;

    elif (re.match("^ENZYME_LEVEL\s", line)):
        level = line[12:-1].lstrip()
        reactions.loc[rxn_name,enzyme_level] = float(level)
        S_matrix.loc[rxn_name,enzyme_level] = float(level)
                
    elif re.match('^COMMENT',line):
        continue
    elif re.match(r'//',line):
        continue
    elif re.match('^#',line):
        continue
        
#    elif (re.match("^[N,P]REGULATION\s", line)):
#        reg = line
#        reactions.loc[rxn_name,regulation] = reg
fdat.close()
S_matrix.fillna(0,inplace=True)
S_active = S_matrix[S_matrix[enzyme_level] > 0.0]
active_reactions = reactions[reactions[enzyme_level] > 0.0]
del S_active[enzyme_level]
# Delete any columns/metabolites that have all zeros in the S matrix:
S_active = S_active.loc[:, (S_active != 0).any(axis=0)]
np.shape(S_active.values)
display(S_active.shape)
display(S_active)
reactions[full_rxn] = reactions[left] + ' = ' + reactions[right]


# In[6]:


if (1):   
    for idx in reactions.index:
        #print(idx,flush=True)
        boltzmann_rxn_str = reactions.loc[idx,'Full Rxn']
        if re.search(':',boltzmann_rxn_str):
            all_cmprts = re.findall(':\S+', boltzmann_rxn_str)
            [s.replace(':', '') for s in all_cmprts] # remove all the ':'s 
            different_compartments = 0
            for cmpt in all_cmprts:
                if not re.match(all_cmprts[0],cmpt):
                    different_compartments = 1
            if ((not different_compartments) and (reactions[left_compartment].isnull or reactions[right_compartment].isnull)):
                reactions.loc[idx,left_compartment] = cmpt
                reactions.loc[idx,right_compartment] = cmpt
                reactions.loc[idx,same_compartment] = True
            if different_compartments:
                reactions.loc[idx,same_compartment] = False
        else:
            if (reactions.loc[idx,left_compartment] == reactions.loc[idx,right_compartment]):
                reactions.loc[idx,same_compartment] = True
            else:
                reactions.loc[idx,same_compartment] = False
display(reactions)                
            


# ## Calculate Standard Free Energies of Reaction 

# In[7]:


from equilibrator_api import *
from equilibrator_api.reaction_matcher import ReactionMatcher
reaction_matcher = ReactionMatcher()



#%%
if (0):
    eq_api = ComponentContribution(pH=7.0, ionic_strength=0.15) # loads data
    boltzmann_rxn_str = reactions.loc['CTP','Full Rxn']
    full_rxn_str_no_cmprt = re.sub(':\S+','', boltzmann_rxn_str)
    full_rxn_str_no_cmprt = re.sub('BETA-D-GLUCOSE','D-GLUCOSE',full_rxn_str_no_cmprt )
    print(full_rxn_str_no_cmprt)
    rxn = Reaction.parse_formula(full_rxn_str_no_cmprt)
    dG0_prime, dG0_uncertainty = eq_api.dG0_prime(rxn)
    display(dG0_prime, dG0_uncertainty)

if (1):
    eq_api=ComponentContribution(pH=7.0, ionic_strength=0.15) # loads data
    for idx in reactions.index:
       #print(idx, flush=True)
       boltzmann_rxn_str = reactions.loc[idx,'Full Rxn']
       #print("boltzmann_rxn_str")
       #print(boltzmann_rxn_str)
       
       full_rxn_str_no_cmprt = re.sub(':\S+','', boltzmann_rxn_str)
       #print("full_rxn_str_no_cmprt1")
       #print(full_rxn_str_no_cmprt)
       full_rxn_str_no_cmprt = re.sub('BETA-D-GLUCOSE','D-GLUCOSE',full_rxn_str_no_cmprt )
       #print("full_rxn_str_no_cmprt2")
       print(full_rxn_str_no_cmprt)

       rxn = reaction_matcher.match(full_rxn_str_no_cmprt)
       if not rxn.check_full_reaction_balancing():
         print('Reaction %s is not balanced:\n %s\n' % (full_rxn_str_no_cmprt), flush=True)
       dG0_prime, dG0_uncertainty = eq_api.dG0_prime(rxn)
       display(dG0_prime, dG0_uncertainty)
       reactions.loc[idx,deltag0] = dG0_prime
       reactions.loc[idx,deltag0_sigma] = dG0_uncertainty
if (0):
    eq_api = ComponentContribution(pH=7.0, ionic_strength=0.15)  # loads data
    rxn_list = []
    for idx in reactions.index:
       print(idx,flush=True)
       boltzmann_rxn_str = reactions.loc[idx,'Full Rxn']
       full_rxn_str_no_cmprt = re.sub(':\S+','', boltzmann_rxn_str)
       full_rxn_str_no_cmprt = re.sub('BETA-D-GLUCOSE','D-GLUCOSE',full_rxn_str_no_cmprt )
       print(full_rxn_str_no_cmprt)
       rxn = reaction_matcher.match(full_rxn_str_no_cmprt)
       if not rxn.check_full_reaction_balancing():
         print('Reaction %s is not balanced:\n %s\n' % (idx, full_rxn_str_no_cmprt), flush=True)
       rxn_list.append(rxn)
    dG0_prime, dG0_uncertainty = eq_api.dG0_prime_multi(rxn_list)
    display(dG0_prime)

reactions.loc['PYRt2m',deltag0] = -RT*np.log(10)
display(reactions)

# From Elad Noor regarding pyruvate => pyruvate reaction:
# Sorry for the delayed response. For formation energies you should use the function Compound.dG0_prime(). Here is some example code:
#    cm = CompoundMatcher()
#    df = cm.match('ATP')
#    c = Compound(Reaction.COMPOUND_DICT[df['CID'].iloc[0]])
#    print('Formation energy of ATP: %.2f' % c.dG0_prime(pH=7, pMg=0, I=0.1))
#The fact that reaction strings like 'ATP = ATP' worked was actually a bug and has just been fixed. If you parse the same string now, you will get an empty reaction.
#Regarding the compartments, the uncertainty does not depend on the aqueous conditions, and neither does the covariance between the uncertainty. Therefore, you can calculate all compartments at once or separately, it should not make a difference.


# ### Determine Pyruvate transport reaction manually

# In[ ]:


#Why do we set these 
reactions.loc['PYRt2m',deltag0] = -RT*np.log(10)
reactions.loc['PYRt2m',deltag0_sigma] = 0

reactions.loc['SUCD1m',deltag0] = 0


# ### Output the Standard Reaction Free Energies for use in a Boltzmann Simulation

# In[ ]:



#reaction_file = open('neurospora_aerobic_respiration.keq', 'w')
reaction_file = open(cwd+'\\GLYCOLYSIS_TCA_GOGAT\\GLYCOLYSIS_TCA_GOGA.keq', 'w')
for y in reactions.index:
    print('%s\t%e' % (y, np.exp(-reactions.loc[y,'DGZERO']/RT)),file=reaction_file)
reaction_file.close()    

#reaction_file = open('neurospora_aerobic_respiration.equilibrator.dat', 'w')
reaction_file = open('matlab_reaction_TCA2.dat', 'w')
for y in reactions.index:
    print("REACTION\t",y,file=reaction_file)
    #print(reaction_df[y])
    for x in reactions.columns:
        if x == "Full Rxn":
            continue
        if x == same_compartment:
            continue
#        if ((x == deltag0) and (reactions.loc[y,same_compartment] == False)):
#            continue
        if pd.notnull(reactions.loc[y,x]):
            print(x, reactions.loc[y,x],file=reaction_file)
    print("DGZERO-UNITS    KJ/MOL",file=reaction_file)
    print("//",file=reaction_file)
reaction_file.close()    


# ## Set Fixed Concentrations/Boundary Conditions

# In[10]:



conc = 'Conc'
variable = 'Variable'
metabolites = pd.DataFrame(index = S_active.columns, columns=[conc,variable])
metabolites[conc] = 0.001
metabolites[variable] = True

# Set the fixed metabolites:
metabolites.loc['ATP:MITOCHONDRIA',conc] = 9.600000e-03
metabolites.loc['ATP:MITOCHONDRIA',variable] = False
metabolites.loc['ADP:MITOCHONDRIA',conc] = 5.600000e-04
metabolites.loc['ADP:MITOCHONDRIA',variable] = False
metabolites.loc['ORTHOPHOSPHATE:MITOCHONDRIA',conc] = 2.000000e-02
metabolites.loc['ORTHOPHOSPHATE:MITOCHONDRIA',variable] = False

metabolites.loc['ATP:CYTOSOL',conc] = 9.600000e-03
metabolites.loc['ATP:CYTOSOL',variable] = False
metabolites.loc['ADP:CYTOSOL',conc] = 5.600000e-04
metabolites.loc['ADP:CYTOSOL',variable] = False
metabolites.loc['ORTHOPHOSPHATE:CYTOSOL',conc] = 2.000000e-02
metabolites.loc['ORTHOPHOSPHATE:CYTOSOL',variable] = False

metabolites.loc['NADH:MITOCHONDRIA',conc] = 8.300000e-05
metabolites.loc['NADH:MITOCHONDRIA',variable] = False
metabolites.loc['NAD+:MITOCHONDRIA',conc] = 2.600000e-03
metabolites.loc['NAD+:MITOCHONDRIA',variable] = False

metabolites.loc['NADH:CYTOSOL',conc] = 8.300000e-05
metabolites.loc['NADH:CYTOSOL',variable] = False
metabolites.loc['NAD+:CYTOSOL',conc] = 2.600000e-03
metabolites.loc['NAD+:CYTOSOL',variable] = False

#metabolites.loc['NADPH:MITOCHONDRIAL_MATRIX',conc] = 0.1
#metabolites.loc['NADPH:MITOCHONDRIAL_MATRIX',variable] = False
#metabolites.loc['NAPD+:MITOCHONDRIAL_MATRIX',conc] = 0.1
#metabolites.loc['NAPD+:MITOCHONDRIAL_MATRIX',variable] = False

#metabolites.loc['NADPH:CYTOSOL',conc] = 0.1
#metabolites.loc['NADPH:CYTOSOL',variable] = False
#metabolites.loc['NADP+:CYTOSOL',conc] = 0.1
#metabolites.loc['NADP+:CYTOSOL',variable] = False

metabolites.loc['ACETYL-COA:MITOCHONDRIA',conc] = 6.06E-04
metabolites.loc['ACETYL-COA:MITOCHONDRIA',variable] = True
#metabolites.loc['ACETYL-COA:CYTOSOL',conc] = 0.1
#metabolites.loc['ACETYL-COA:CYTOSOL',variable] = False

metabolites.loc['COA:MITOCHONDRIA',conc] = 1.400000e-03
metabolites.loc['COA:MITOCHONDRIA',variable] = False
#metabolites.loc['COA:CYTOSOL',conc] = 0.1
#metabolites.loc['COA:CYTOSOL',variable] = False
#metabolites.loc['COA:GLYOXYSOME',conc] = 0.1
#metabolites.loc['COA:GLYOXYSOME',variable] = False

metabolites.loc['CO2:MITOCHONDRIA',conc] = 1.000000e-04
metabolites.loc['CO2:MITOCHONDRIA',variable] = False
#metabolites.loc['CO2:CYTOSOL',conc] = 0.1
#metabolites.loc['CO2:CYTOSOL',variable] = False 

metabolites.loc['H2O:MITOCHONDRIA',conc] = 55.5
metabolites.loc['H2O:MITOCHONDRIA',variable] = False
metabolites.loc['H2O:CYTOSOL',conc] = 55.5
metabolites.loc['H2O:CYTOSOL',variable] = False 

# What should the concentration of oxygen in the mitochondrial be?
#metabolites.loc['OXYGEN:MITOCHONDRIAL_MATRIX','Conc',] = 1.0e-28
#metabolites.loc['OXYGEN:MITOCHONDRIA',conc] = 1.0e-04
#metabolites.loc['OXYGEN:MITOCHONDRIA',variable] = False
#metabolites.loc['OXYGEN:CYTOSOL',conc] = 55.5
#metabolites.loc['OXYGEN:CYTOSOL',variable] = False 

metabolites.loc['BETA-D-GLUCOSE:CYTOSOL',conc] = 2.000000e-03
metabolites.loc['BETA-D-GLUCOSE:CYTOSOL',variable] = False 

metabolites.loc['L-GLUTAMATE:MITOCHONDRIA',conc] = 9.60e-05
metabolites.loc['L-GLUTAMATE:MITOCHONDRIA',variable] = False 
metabolites.loc['L-GLUTAMINE:MITOCHONDRIA',conc] = 3.81e-03
metabolites.loc['L-GLUTAMINE:MITOCHONDRIA',variable] = False 

#%%
nvariables = metabolites[metabolites[variable]].count()
nvar = nvariables[variable]

metabolites.sort_values(by=variable, axis=0,ascending=False, inplace=True,)
display(metabolites)


# ## Prepare model for optimization

# - Adjust S Matrix to use only reactions with activity > 0, if necessary.
# - Water stoichiometry in the stiochiometric matrix needs to be set to zero since water is held constant.
# - The initial concentrations of the variable metabolites are random.
# - All concentrations are changed to log counts.
# - Equilibrium constants are calculated from standard free energies of reaction.
# - R (reactant) and P (product) matrices are derived from S.


# Make sure all the indices and columns are in the correct order:
active_reactions = reactions[reactions[enzyme_level] > 0.0]
#display(reactions)
display(metabolites.index)
Sactive_index = S_active.index
#Sactive_columns = S_active.columns
active_reactions.reindex(index = Sactive_index, copy = False)
S_active = S_active.reindex(columns = metabolites.index, copy = False)
S_active['H2O:MITOCHONDRIA'] = 0
S_active['H2O:CYTOSOL'] = 0

#####################################
#####################################
#THIS IS MAKING FLUX -> 0.0
where_are_NaNs = np.isnan(S_active)
S_active[where_are_NaNs] = 0

display(S_active[:])

S_mat = S_active.values

#display(nvar)

variable_concs = np.array(metabolites['Conc'].iloc[0:nvar].values, dtype=np.float64)
v_log_concs = -10 + 10*np.random.rand(nvar) #Vary between 1 M to 1.0e-10 M
v_concs = np.exp(v_log_concs)
v_log_counts_stationary = np.log(v_concs*Concentration2Count)
v_log_counts = v_log_counts_stationary
#display(v_log_counts)

fixed_concs = np.array(metabolites['Conc'].iloc[nvar:].values, dtype=np.float64)
fixed_counts = fixed_concs*Concentration2Count
f_log_counts = np.log(fixed_counts)
#display(f_log_counts)

Keq_constant = np.exp(-active_reactions[deltag0].astype('float')/RT)
#display(Keq_constant)
Keq_constant = Keq_constant.values

P_mat = np.where(S_mat>0,S_mat,0)
R_back_mat = np.where(S_mat<0, S_mat, 0)
E_regulation = np.ones(Keq_constant.size) # THis is the vector of enzyme activities, Range: 0 to 1.

#WARNING:::::::::::::::CHANGE BACK TO ZEROS
delta_increment_for_small_concs = (10**-50)*np.zeros(metabolites['Conc'].values.size);

mu0 = 1 #Dummy parameter for now; reserved for free energies of formation

#%% Basic test
import max_entropy_functions

variable_concs_begin = np.array(metabolites['Conc'].iloc[0:nvar].values, dtype=np.float64)
v_log_counts = np.log(variable_concs_begin*Concentration2Count)

from scipy.optimize import least_squares
#r_log_counts = -10 + 10*np.random.rand(v_log_counts.size)
#v_log_counts = r_log_counts
print('====== Without adjusting Keq_constant ======')


E_regulation = np.ones(Keq_constant.size) # THis is the vector of enzyme activities, Range: 0 to 1.
nvar = v_log_counts.size
#WARNING: INPUT LOG_COUNTS TO ALL FUNCTIONS. CONVERSION TO COUNTS IS DONE INTERNALLY
res_lsq1 = least_squares(max_entropy_functions.derivatives, v_log_counts, method='lm',xtol=1e-15, args=(f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, E_regulation))
res_lsq2 = least_squares(max_entropy_functions.derivatives, v_log_counts, method='dogbox',xtol=1e-15, args=(f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, E_regulation))
res_lsq3 = least_squares(max_entropy_functions.derivatives, v_log_counts, method='trf',xtol=1e-15, args=(f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, E_regulation))

rxn_flux = max_entropy_functions.oddsDiff(res_lsq1.x, f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, E_regulation)

display("Optimized Metabolites")
display(res_lsq1.x)
display(res_lsq2.x)
display(res_lsq3.x)
display("Reaction Flux")
display(rxn_flux)

# In[ ]:
down_regulate = True
has_been_up_regulated = 10*np.ones(Keq_constant.size)
begin_log_metabolites = np.append(res_lsq1.x,f_log_counts)
##########################################
##########################################
#####################TESTER###############

E_regulation = np.ones(Keq_constant.size) # THis is the vector of enzyme activities, Range: 0 to 1.
log_metabolites = np.append(res_lsq1.x,f_log_counts)
KQ_f = max_entropy_functions.odds(log_metabolites,mu0,S_mat, R_back_mat, P_mat, delta_increment_for_small_concs,Keq_constant);

Keq_inverse = np.power(Keq_constant,-1);
KQ_r = max_entropy_functions.odds(log_metabolites,mu0,-S_mat, P_mat, R_back_mat, delta_increment_for_small_concs,Keq_inverse,-1);

[RR,Jac] = max_entropy_functions.calc_Jac2(res_lsq1.x, f_log_counts, S_mat, delta_increment_for_small_concs, KQ_f, KQ_r, E_regulation)
A = max_entropy_functions.calc_A(res_lsq1.x,f_log_counts, S_mat, Jac, E_regulation )

[ccc,fcc] = max_entropy_functions.conc_flux_control_coeff(nvar, A, S_mat, rxn_flux, RR)

desired_conc=6.022140900000000e+05
React_Choice=15

newE = max_entropy_functions.calc_reg_E_step(E_regulation,React_Choice, nvar, res_lsq1.x, f_log_counts, desired_conc, 
                       S_mat, A, rxn_flux,KQ_f,True, has_been_up_regulated)
    
    
delta_S = max_entropy_functions.calc_deltaS(res_lsq1.x, f_log_counts, S_mat, KQ_f)

target_log_concs = np.ones(nvar) * 13.308368285158080
delta_S_metab = max_entropy_functions.calc_deltaS_metab(res_lsq1.x);

ipolicy = 4 #use ipolicy=1 or 4
reaction_choice = max_entropy_functions.get_enzyme2regulate(ipolicy, delta_S, delta_S_metab, ccc, KQ_f, E_regulation,res_lsq1.x,
                                                        has_been_up_regulated)

display(newE)
display(reaction_choice)

#%%
 
import machine_learning_functions

gamma = 0.9
num_samples = 10 #number of state samples theta_linear attempts to fit to in a single iteration
length_of_path = 5 #length of path after 1 forced step
epsilon_greedy = 0.00

#set variables in ML program
machine_learning_functions.Keq_constant = Keq_constant
machine_learning_functions.f_log_counts = f_log_counts

machine_learning_functions.P_mat = P_mat
machine_learning_functions.R_back_mat = R_back_mat
machine_learning_functions.S_mat = S_mat
machine_learning_functions.delta_increment_for_small_concs = delta_increment_for_small_concs
machine_learning_functions.desired_conc = desired_conc
machine_learning_functions.nvar = nvar
machine_learning_functions.mu0 = mu0

machine_learning_functions.gamma = gamma
machine_learning_functions.num_rxns = Keq_constant.size
machine_learning_functions.num_samples = num_samples
machine_learning_functions.length_of_path = length_of_path

    

#%%
import torch
N, D_in, H, D_out = 1, Keq_constant.size,  10*Keq_constant.size, 1

# Create random Tensors to hold inputs and outputs
x_in = torch.zeros(N, D_in)
y_in = torch.zeros(N, D_out)

# Create random Tensors to hold inputs and outputs
x = 10*torch.rand(1000,1, D_in)

# =============================================================================
# nn_model = torch.nn.Sequential(
#         torch.nn.Linear(D_in, 200),
#                 
#         torch.nn.ReLU(),
#         torch.nn.Conv1d(in_channels=1, out_channels=1,kernel_size=3,stride=2, padding=0),
#         torch.nn.ReLU(),
#         torch.nn.MaxPool1d(2),
#         torch.nn.Conv1d(in_channels=1, out_channels=1,kernel_size=3,stride=1, padding=0),
#         torch.nn.LeakyReLU(),
#         torch.nn.MaxPool1d(2),
#         
#         torch.nn.Linear(23, D_out)
#         )
# =============================================================================

nn_model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(H,D_out))
# =============================================================================
# 
# nn_model = torch.nn.Sequential(
#         torch.nn.Linear(D_in, D_out))
# =============================================================================

loss_fn = torch.nn.MSELoss(reduction='sum')
alpha = 1e-1
print(list(nn_model.parameters()))
#optimizer = torch.optim.Adam(nn_model.parameters(), lr=alpha)

optimizer = torch.optim.SGD(nn_model.parameters(), lr=1e-4, momentum=0.9)

#optimizer = torch.optim.LBFGS(nn_model.parameters(), lr=alpha, max_iter=10,tolerance_change=1e-300)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

#%% SGD UPDATE TEST
theta_linear=[]
updates = 25000 #attempted iterations to update theta_linear
v_log_counts = v_log_counts_stationary.copy()
episodic_loss = np.zeros(updates)
episodic_reward = np.zeros(updates)
episodic_prediction = np.zeros(updates)
episodic_prediction_changing = np.zeros(updates)
epsilon_greedy = 0.05

n_back_step = 8 #these steps use rewards. Total steps before n use state values

for update in range(0,updates):
    
    x_changing = 10*torch.rand(1000,1, D_in)

    
    #generate state to use
    state_is_valid = False
    state_sample = np.zeros(Keq_constant.size)
    for sample in range(0,len(state_sample)):
        state_sample[sample] = np.random.uniform(1,1)

    #annealing test
    if ((update %10 == 0) and (update != 0)):
        epsilon_greedy=epsilon_greedy/2
        print("RESET epsilon ANNEALING")
        print(epsilon_greedy)

    
    prediction_x_previous = nn_model(x)
    prediction_x_changing_previous = nn_model(x_changing)
    
    [sum_reward, average_loss] = machine_learning_functions.sarsa_n(nn_model,loss_fn, optimizer, scheduler, alpha, n_back_step, theta_linear, v_log_counts, state_sample, epsilon_greedy)
    print("TOTAL REWARD")
    print(sum_reward)
    print("ave loss")
    print(average_loss)
    
    prediction_x = nn_model(x)
    prediction_x_changing = nn_model(x_changing)
    
    total_prediction_diff = sum(abs(prediction_x - prediction_x_previous))
    total_prediction_changing_diff = sum(abs(prediction_x_changing - prediction_x_changing_previous))
    print("TOTALPREDICTION")
    print(total_prediction_diff)
    print(total_prediction_changing_diff)
    print(optimizer.state_dict)
    episodic_prediction[update] = total_prediction_diff
    episodic_prediction_changing[update] = total_prediction_changing_diff
    
    #print(list(nn_model.parameters()))
    #print("**********************************************************************")
    #print("EPISODE FINISHED")
    #print("sum")
    #print(sum_reward)
    episodic_loss[update]=average_loss
    episodic_reward[update]=sum_reward
    
#%% SAVE MODEL
    
#torch.save(nn_model.state_dict(), cwd+'\\GLYCOLYSIS_TCA_GOGAT\\'+'model.pth')
#%% LOAD MODEL
nn_model.load_state_dict(torch.load(cwd+'\\GLYCOLYSIS_TCA_GOGAT\\'+'model.pth'))

#%%
#np.savetxt(cwd+'\\GLYCOLYSIS_TCA_GOGAT\\'+'episodic_loss.txt', episodic_loss, fmt='%f')
#np.savetxt(cwd+'\\GLYCOLYSIS_TCA_GOGAT\\'+'episodic_reward.txt', episodic_reward, fmt='%f')


#%% Getting back the objects:
el = np.loadtxt(cwd+'\\GLYCOLYSIS_TCA_GOGAT\\'+'episodic_loss.txt', dtype=float)
er = np.loadtxt(cwd+'\\GLYCOLYSIS_TCA_GOGAT\\'+'episodic_reward.txt', dtype=float)


#%%
episodic_reward=episodic_reward[0:update]
episodic_loss=episodic_loss[0:update]
episodic_prediction = episodic_prediction[0:update]
episodic_prediction_changing = episodic_prediction_changing[0:update]
plt.plot(episodic_prediction_changing)
plt.plot(episodic_prediction)
#%%
plt.plot(episodic_loss)
plt.xlabel("epochs")
plt.ylabel("<L>")
#%%
plt.plot(episodic_reward)

plt.xlabel("epochs")
plt.ylabel("<R>")
#%%

v_log_concs = -10 + 10*np.random.rand(nvar) #Vary between 1 M to 1.0e-10 M
v_log_counts = np.log(v_concs*Concentration2Count)

E_regulation = np.ones(Keq_constant.size)

down_regulate = True
nvar = len(v_log_counts)

ipolicy=7 #USE 1 or 4

rxn_reset = 0 * np.ones(Keq_constant.size)
rxn_use_abs = 0 * np.ones(Keq_constant.size)
has_been_up_regulated = 1*np.zeros(Keq_constant.size)

React_Choice=0
#E_regulation = np.ones(Keq_constant.size)
attempts = 100000
i = 0
deltaS_value = 10
delta_S = np.ones(Keq_constant.size)
epr_old=10000
flux_vector_method_1 = np.zeros(attempts)
epr_vector_method_1=np.zeros(attempts)
final_choices1=np.zeros(attempts)

v_log_counts_matrix1 = np.zeros([v_log_counts.size, attempts])

use_abs_step = True
#somehow first iteration is off. 
#prod_indices in E_step are wrong. It is including 0. 
epsilon = 0.0

variable_concs_begin = np.array(metabolites['Conc'].iloc[0:nvar].values, dtype=np.float64)
#v_log_concs = -10 + 10*np.random.rand(nvar) #Vary between 1 M to 1.0e-10 M
#v_concs = np.exp(v_log_concs)
activity_matrix = np.ones([v_log_counts.size, 10])

total_reward=0
reward_vec_1= np.zeros(attempts)


ds_total_1=0.0  
ds_total_1_vec=[]
v_log_counts = np.log(variable_concs_begin*Concentration2Count)
while( (i < attempts) and (np.max(delta_S) > 0) ):
    
    res_lsq = least_squares(max_entropy_functions.derivatives, v_log_counts, method='lm',xtol=1e-15, args=(f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, E_regulation))
    #print("Finished optimizing")
    #Reset variable concentrations after optimization
    v_log_counts = res_lsq.x
    log_metabolites = np.append(v_log_counts, f_log_counts)
        
    #make calculations to regulate
    rxn_flux = max_entropy_functions.oddsDiff(v_log_counts, f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, E_regulation)
        
        
    KQ_f = max_entropy_functions.odds(log_metabolites, mu0,S_mat, R_back_mat, P_mat, delta_increment_for_small_concs,Keq_constant);
    Keq_inverse = np.power(Keq_constant,-1)
    KQ_r = max_entropy_functions.odds(log_metabolites, mu0,-S_mat, P_mat, R_back_mat, delta_increment_for_small_concs,Keq_inverse,-1);
    
    epr = max_entropy_functions.entropy_production_rate(KQ_f, KQ_r, E_regulation)
    delta_S = max_entropy_functions.calc_deltaS(v_log_counts,f_log_counts, S_mat, KQ_f)
    reward=0
    epr_vector_method_1[i]=epr
    if (i > 0):
            
        reward = machine_learning_functions.reward_value(v_log_counts, \
                                                         v_log_counts_matrix1[:,i-1],\
                                                         KQ_f, KQ_r, E_regulation,\
                                                         KQ_f, KQ_r, E_regulation,\
                                                         delta_S,delta_S_previous)

    total_reward+=reward
            
    reward_vec_1[i]=reward
    delta_S_metab = max_entropy_functions.calc_deltaS_metab(v_log_counts);
    
    [RR,Jac] = max_entropy_functions.calc_Jac2(v_log_counts, f_log_counts, S_mat, delta_increment_for_small_concs, KQ_f, KQ_r, E_regulation)
    A = max_entropy_functions.calc_A(v_log_counts, f_log_counts, S_mat, Jac, E_regulation )
        
    [ccc,fcc] = max_entropy_functions.conc_flux_control_coeff(nvar, A, S_mat, rxn_flux, RR)
    
    React_Choice = max_entropy_functions.get_enzyme2regulate(ipolicy, delta_S, delta_S_metab,
                                        ccc, KQ_f, E_regulation,v_log_counts, has_been_up_regulated)
        
    if (React_Choice == -1):
        print("FINISHED OPTIMIZING")
        break
            
    final_choices1[i]=React_Choice
        
    rxn_use_abs[React_Choice]=True
    desired_conc=6.022140900000000e+05
        
    oldE = E_regulation[React_Choice]
    old_delta_S=delta_S
        
    use_abs_step = rxn_use_abs[React_Choice]
    newE = max_entropy_functions.calc_reg_E_step(E_regulation, React_Choice, nvar, v_log_counts, 
                           f_log_counts, desired_conc, S_mat, A, rxn_flux, KQ_f, use_abs_step, 
                           has_been_up_regulated,
                           delta_S)
        
            
    print("rct_choice")
    print(React_Choice)
    print("newE")
    print(newE)
    deltaS_value = delta_S[React_Choice]
    
    flux_vector_method_1[i]=np.sum(rxn_flux)
        
    v_log_counts_matrix1[:,i] = v_log_counts
        
    #print ("sum_flux")
    #print(np.sum(rxn_flux))


    E_regulation[React_Choice] = newE
    ds = np.sum(delta_S[delta_S>0])
    ds_total_1_vec.append(ds)
        
    ds_total_1 += ds
    print("entropy_production_rate")
    print(epr)
        
    epr_old=epr
    print(delta_S)
    i += 1
    delta_S_previous = delta_S.copy()
        
        
reward_vec_1=reward_vec_1[0:i+1]
v_log_counts_matrix1 = v_log_counts_matrix1[:,0:i+1]
final_choices1=final_choices1[0:i+1]
epr_vector_method_1=epr_vector_method_1[0:i+1]
flux_vector_method_1=flux_vector_method_1[0:i+1]
opt_concs1 = v_log_counts
E_reg1 = E_regulation
rxn_flux_1 = rxn_flux
deltaS1=delta_S
#
#%%
#use policy_function
import random
attempts = 1000
epr_vector_method_2=np.zeros(attempts)
flux_vector_method_2=np.zeros(attempts)
delta_S = np.ones(Keq_constant.size)

epr_old=1000
KQ_f_old=np.ones(Keq_constant.size)
E_regulation_old=np.ones(Keq_constant.size)
epsilon = 0.0
final_choices2=np.zeros(attempts)
down_regulate = True

v_log_counts_matrix2 = np.zeros([v_log_counts.size, attempts])
v_log_counts = np.log(v_concs*Concentration2Count)
#v_log_counts_begin = begin_log_metabolites[0:nvar]

KQ_f_old=np.ones(Keq_constant.size)
KQ_r_old=np.ones(Keq_constant.size)

activity_matrix = np.ones([E_regulation.size, 20])

ds_total_2=0.0
ds_total_2_vec=[]
total_reward=0

reward_vec_2= np.zeros(attempts)
for test in range(0,1):
    #theta_linear=np.random.uniform(0,1,theta_linear.size)
    i = 0
    delta_S = np.ones(Keq_constant.size)
    E_regulation =np.ones(Keq_constant.size)
    print(theta_linear)
    while( (i < attempts) and (np.max(delta_S) > 0) ):
    #while( (i < attempts) ):
        
        res_lsq = least_squares(max_entropy_functions.derivatives, v_log_counts, method='lm',xtol=1e-15, args=(f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, E_regulation))
        #print("Finished optimizing")
        #Reset variable concentrations after optimization
        v_log_counts = res_lsq.x
        log_metabolites = np.append(v_log_counts, f_log_counts)
        
        #make calculations to regulate
        rxn_flux = max_entropy_functions.oddsDiff(v_log_counts, f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, E_regulation)
        KQ_f = max_entropy_functions.odds(log_metabolites, mu0,S_mat, R_back_mat, P_mat, delta_increment_for_small_concs,Keq_constant);
        Keq_inverse = np.power(Keq_constant,-1)
        KQ_r = max_entropy_functions.odds(log_metabolites, mu0,-S_mat, P_mat, R_back_mat, delta_increment_for_small_concs,Keq_inverse,-1);
    
        #Regulation
        #target_log_concs = np.ones(nvar) * 13.308368285158080
        delta_S = max_entropy_functions.calc_deltaS(v_log_counts,f_log_counts, S_mat, KQ_f)
        delta_S_metab = max_entropy_functions.calc_deltaS_metab(v_log_counts);
    
    
        [RR,Jac] = max_entropy_functions.calc_Jac2(v_log_counts, f_log_counts, S_mat, delta_increment_for_small_concs, KQ_f, KQ_r, E_regulation)
        A = max_entropy_functions.calc_A(v_log_counts, f_log_counts, S_mat, Jac, E_regulation )
        
        epr = max_entropy_functions.entropy_production_rate(KQ_f, KQ_r, E_regulation)
        
           
         
        
        #rxn_choices = [i for i in range(Keq_constant.size)]
        #React_Choice = random.choice(rxn_choices)
        
        #breakpoint()
        
        x = torch.zeros(1,1, E_regulation.size)
        y = torch.zeros(1, 1,1)
    
        for j in range(0,E_regulation.size):
            x[0][0][j] = E_regulation[j].copy()
        
        [React_Choice, reward] = machine_learning_functions.policy_function(nn_model,E_regulation, theta_linear, v_log_counts)
        
        print("reward")
        print(reward)
        
        if (React_Choice==-1):
            break
        final_choices2[i]=React_Choice
        print("rct_choice")
        print(React_Choice)
        desired_conc=6.022140900000000e+05
        
        oldE = E_regulation[React_Choice]
        newE = max_entropy_functions.calc_reg_E_step(E_regulation, React_Choice, nvar, 
                               v_log_counts, f_log_counts, desired_conc, S_mat, A, 
                               rxn_flux, KQ_f, False, has_been_up_regulated,\
                               delta_S)
        deltaS_value = delta_S[React_Choice]
        
        epr_vector_method_2[i]=epr
        if (oldE < newE):
            print("*****************************************************************************")
            #breakpoint()
        #print(newE)
                 

        
        ds = np.sum(delta_S[delta_S>0])
        ds_total_2 += ds
        ds_total_2_vec.append(ds)
        if (np.max(delta_S) > 0):
            total_reward+=reward
            
            reward_vec_2[i]=reward
            E_regulation[React_Choice] = newE
        flux_vector_method_2[i]=np.sum(rxn_flux)
        v_log_counts_matrix2[:,i] = v_log_counts
        #print("entropy_production_rate")
        #print("v_log_counts")
        #print(v_log_counts)
        print(epr)
        #print(np.max(rxn_flux))
        print("ds>0")
        print(np.sum(delta_S[delta_S>0]))
        #if (React_Choice==0):
        #print(delta_S_metab)
            
        #breakpoint()
        delta_S_previous = delta_S.copy()
        
        i = i+1
    activity_matrix[:,test] = E_regulation
    

reward_vec_2=reward_vec_2[0:i]
v_log_counts_matrix2 = v_log_counts_matrix2[:,0:i]
final_choices2=final_choices2[0:i]
epr_vector_method_2=epr_vector_method_2[0:i]
flux_vector_method_2=flux_vector_method_2[0:i]
opt_concs2 = v_log_counts
E_reg2 = E_regulation
rxn_flux_2 = rxn_flux
deltaS2=delta_S
#%% activity plot

fig = plt.figure(figsize=(figure_norm, figure_norm))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

for j in range(0, activity_matrix.shape[1]):
    ax1.plot(activity_matrix[:,j]-E_reg2)
ax2.plot(E_reg1)

#%%
   
fig = plt.figure(figsize=(figure_norm, figure_norm))
ax1 = fig.add_subplot(121)
for j in range(0, len(v_log_counts)):
    ax1.plot(v_log_counts_matrix1[j,:])
    
ax2 = fig.add_subplot(122)
for j in range(0, len(v_log_counts)):
    ax2.plot(v_log_counts_matrix2[j,:])

#%% KQ

tickSize=15
sns.set_style("ticks", {"xtick.major.size": tickSize, "ytick.major.size": tickSize})

figure_norm = 12 #convert to 17.1cm
figure_len_factor=4/3

figure_factor=17.1/8.3#ratio for sm journal

Fontsize_Title=20
Fontsize_Sub = 15
Fontsize_Leg = 15

fig = plt.figure(figsize=(figure_norm, figure_len_factor * figure_norm))
ax1 = fig.add_subplot(511)
ax2 = fig.add_subplot(512)
ax3 = fig.add_subplot(513)
ax4 = fig.add_subplot(514)
ax5 = fig.add_subplot(515)

ax1.plot(epr_vector_method_1,label='CCC')
ax2.plot(epr_vector_method_2,label='ML')

x=np.linspace(1,len(rxn_flux_2),len(rxn_flux_2))

sns.barplot(x=x,y=rxn_flux_1, palette="rocket",
            label='CCC', ax=ax3)
sns.barplot(x=x,y=rxn_flux_2, palette="rocket",
            label='ML',ax=ax4)

sns.barplot(x=x,y=deltaS1, palette="rocket", ax=ax3)
sns.barplot(x=x,y=deltaS2, palette="rocket", ax=ax4)


ax5.scatter(x,E_reg1, label='CCC')
ax5.scatter(x,E_reg2, label='ML')

ax1.set_xlabel('Iters',fontsize=Fontsize_Sub)
ax2.set_xlabel('Iters',fontsize=Fontsize_Sub)
ax1.set_ylabel('Entropy Production Rate',fontsize=Fontsize_Sub)
ax2.set_ylabel('Entropy Production Rate',fontsize=Fontsize_Sub)

ax3.set_xlabel('Reactions',fontsize=Fontsize_Sub)
ax4.set_xlabel('Reactions',fontsize=Fontsize_Sub)
ax3.set_ylabel('Flux(upper) & S_mat(lower)',fontsize=Fontsize_Sub)
ax4.set_ylabel('Flux(upper) & S_mat(lower)',fontsize=Fontsize_Sub)

ax5.set_ylabel('Reactions',fontsize=Fontsize_Sub)
ax5.set_ylabel('Regulation Value',fontsize=Fontsize_Sub)



ax1.legend(fontsize=Fontsize_Leg, loc='upper right')
ax2.legend(fontsize=Fontsize_Leg, loc='upper right')
ax3.legend(fontsize=Fontsize_Leg, loc='lower right')
ax4.legend(fontsize=Fontsize_Leg, loc='lower right')
ax5.legend(fontsize=Fontsize_Leg, loc='lower left')

ax1.set_xlim([0.0, len(epr_vector_method_1)])
ax2.set_xlim([0.0, len(epr_vector_method_2)])

#ax2.set_ylim([10/10000, 2/10])


#plt.tight_layout()

#%%

color1 = sns.xkcd_rgb["slate grey"]
color2 = sns.xkcd_rgb["grey"]
color3 = sns.xkcd_rgb["steel grey"]
fig = plt.figure(figsize=(figure_norm, figure_len_factor * figure_norm))
ax1 = fig.add_subplot(421)
ax2 = fig.add_subplot(422)
ax3 = fig.add_subplot(412)
ax4 = fig.add_subplot(413)
ax5 = fig.add_subplot(414)
sns.set_style("ticks", {"xtick.major.size": tickSize, "ytick.major.size": tickSize})




x_tick=np.linspace(0,Keq_constant.size-1,Keq_constant.size, dtype=int)
x_tick_list=list(x_tick)
x_tick_list_double = x_tick_list+x_tick_list
type1='CCC'
type2='ML'


ax1.plot(epr_vector_method_1,label='CCC')
ax2.plot(epr_vector_method_2,label='ML')


DataFlux = pd.DataFrame({'x':x_tick_list_double,
                           'data':list(rxn_flux_1)+list(rxn_flux_2),
                           ' ':len(rxn_flux_1)*[type1]+len(rxn_flux_2)*[type2]})
    
DataActivity = pd.DataFrame({'x':x_tick_list_double,
                           'data':list(E_reg1)+list(E_reg2),
                           ' ':len(E_reg1)*[type1]+len(E_reg2)*[type2]})
    
DataDeltaS = pd.DataFrame({'x':x_tick_list_double,
                           'data':list(deltaS1)+list(deltaS2),
                           ' ':len(deltaS1)*[type1]+len(deltaS2)*[type2]})

gflux=sns.catplot(x='x',y='data',data=DataActivity,hue=' ',
              kind='bar',color=color3,legend=False,
              edgecolor="0.1",height=6,ax=ax3)

gflux=sns.catplot(x='x',y='data',data=DataFlux,hue=' ',
              kind='bar',color=color3,legend=False,
              edgecolor="0.1",height=6,ax=ax4)

gflux=sns.catplot(x='x',y='data',data=DataDeltaS,hue=' ',
              kind='bar',color=color3,legend=False,
              edgecolor="0.1",height=6,ax=ax5)


ax1.set_xlabel('Iters',fontsize=Fontsize_Sub)
ax2.set_xlabel('Iters',fontsize=Fontsize_Sub)
ax1.set_ylabel('Entropy Production Rate',fontsize=Fontsize_Sub)
ax2.set_ylabel('Entropy Production Rate',fontsize=Fontsize_Sub)

ax1.set_xlim([0.0, len(epr_vector_method_1)])
ax2.set_xlim([0.0, len(epr_vector_method_2)])


ax3.set_xlabel('Reactions',fontsize = Fontsize_Sub)
ax3.set_ylabel('Activity',fontsize = Fontsize_Sub)

ax4.set_xlabel('Reactions',fontsize = Fontsize_Sub)
ax4.set_ylabel('Flux',fontsize = Fontsize_Sub)

ax5.set_xlabel('Reactions',fontsize = Fontsize_Sub)
ax5.set_ylabel(r'$\Delta$S_mat',fontsize = Fontsize_Sub)
#fig.suptitle(r'Fiber Density = 1 $\mu m^{-3}, ' '$ $W_{s} = %s$' %(temp) )

ax1.legend(fontsize=Fontsize_Leg, loc='upper right')
ax2.legend(fontsize=Fontsize_Leg, loc='upper right')
ax3.legend(fontsize=Fontsize_Leg,loc='lower left')
ax4.legend(fontsize=Fontsize_Leg,loc='lower left')
ax5.legend(fontsize=Fontsize_Leg,loc='upper left')

ax1.tick_params(labelsize=tickSize)
ax2.tick_params(labelsize=tickSize)
ax3.tick_params(labelsize=tickSize)
ax4.tick_params(labelsize=tickSize)
ax5.tick_params(labelsize=tickSize)

plt.tight_layout()

#%%
tickSize=25
sns.set_style("ticks", {"xtick.major.size": tickSize, "ytick.major.size": tickSize})

figure_norm = 12 #convert to 17.1cm
figure_len_factor=3/3

figure_factor=17.1/8.3#ratio for sm journal

Fontsize_Title=20
Fontsize_Sub = 20
Fontsize_Leg = 15

fig = plt.figure(figsize=(figure_norm, figure_len_factor * figure_norm))
ax1 = fig.add_subplot(111)

x1=np.linspace(1,len(final_choices1),len(final_choices1))
x2=np.linspace(1,len(final_choices2),len(final_choices2))

ax1.scatter(x1,final_choices1,label='CCC',alpha=0.5)
ax1.scatter(x2,final_choices2,label='ML',alpha=0.5)

ax1.legend(fontsize=Fontsize_Leg, loc='lower right')
#%%
fig = plt.figure(figsize=(figure_norm, figure_len_factor * figure_norm))
ax1 = fig.add_subplot(111)

ax = sns.distplot(x, fit=norm, kde=False)

#%% Plot matlab data for energy vs flux from two types of samples.
import numpy as np, pandas as pd; np.random.seed(0)
import seaborn as sns; sns.set(style="white", color_codes=True)
import scipy.io


matLHS = scipy.io.loadmat("sim_gibbs_flux.mat")
matUNIF = scipy.io.loadmat("sim_gibbs_flux_unif_sample.mat")

max_flux_LHS = np.max(matLHS["mid_sim_flux"])
max_gibbs_LHS = np.max(matLHS["sim_gibbs_prob"])
max_flux_UNIF = np.max(matUNIF["mid_sim_flux"])
max_gibbs_UNIF = np.max(matUNIF["sim_gibbs_prob"])


max_flux = np.max(matLHS["mid_sim_flux"])
max_gibbs = np.max(matLHS["sim_gibbs_prob"])
mean_gibbs = np.mean(matLHS["sim_gibbs_prob"])
min_gibbs = np.min(matLHS["sim_gibbs_prob"])
#g = sns.jointplot(mat["mid_sim_flux"]/max_flux, mat["sim_gibbs_prob"]/max_gibbs, kind="kde")

sns.kdeplot( matLHS["mid_sim_flux"].flatten())
sns.kdeplot( matUNIF["mid_sim_flux"].flatten())

#ax1.set_xlim([0.0, len(epr_vector_method_1)])
#ax1.set_ylim([np.min(mat["sim_gibbs_prob"]),np.max(mat["sim_gibbs_prob"])])

#%%
def calculate_rate_constants(log_counts, rxn_flux,KQ_inverse, R_back_mat, E_Regulation):
    
    rxn_flux_temp = rxn_flux.copy()
    KQ = np.power(KQ_inverse,-1)
    #Infer rate constants from reaction flux
    denominator = E_Regulation* np.exp(-R_back_mat.dot(log_counts))*(1-KQ_inverse)
    # A reaction near equilibrium is problematic because (1-KQ_inverse)->0
    # By setting these reactions to be 
    # rate constant = 1/product_concs we are setting the rate to 1, which
    # is the same as the thermodynamic rate = KQ.
    one_idx, = np.where(KQ_inverse > 0.9)
    denominator[one_idx] = E_Regulation[one_idx]* np.exp(-R_back_mat[one_idx,:].dot(log_counts));
    
    rxn_flux_temp[one_idx] = 1;
    fwd_rate_constants = rxn_flux_temp/denominator;
    
    return(fwd_rate_constants)


#%% Rate constants CCC

log_counts = np.append(opt_concs1,f_log_counts)
KQ_inverse = max_entropy_functions.odds(log_counts,mu0,S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, direction = -1)
forward_rate_constants_CCC = calculate_rate_constants(log_counts, rxn_flux_1, KQ_inverse, R_back_mat, E_reg1)
reverse_rate_constants_CCC = forward_rate_constants_CCC/Keq_constant
display(forward_rate_constants_CCC)

#Rate constants ML
log_counts = np.append(opt_concs2,f_log_counts)
KQ_inverse = max_entropy_functions.odds(log_counts,mu0,S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, direction = -1)
forward_rate_constants_ML = calculate_rate_constants(log_counts, rxn_flux_2, KQ_inverse, R_back_mat, E_reg2)
reverse_rate_constants_ML = forward_rate_constants_ML/Keq_constant
display(forward_rate_constants_ML)

plt.plot(forward_rate_constants_ML/forward_rate_constants_CCC)
plt.plot(reverse_rate_constants_ML/reverse_rate_constants_CCC)
# ## ODE Solvers: Python interface using libroadrunner to Sundials/CVODE
