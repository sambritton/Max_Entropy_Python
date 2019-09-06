# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 08:27:55 2019

@author: samuel_britton
"""

#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd

import os
import re
import sys

os.chdir("..")
cwd = os.getcwd()
sys.path.insert(0, cwd+'/Basic_Functions')
sys.path.insert(0, cwd+'/GLYCOLYSIS_TCA_GOGAT')
sys.path.insert(0, cwd+'/Basic_Functions/equilibrator-api-v0.1.8/build/lib')
    
import max_entropy_functions
import machine_learning_functions as me
from scipy.optimize import least_squares
import torch

def run(sim_number, n_back_step,learning_rate, epsilon, eps_threshold,\
        gamma = 0.9,updates = 250, penalty_reward_scalar=-1.0,): 
    
    pd.set_option('display.max_columns', None,'display.max_rows', None)
    
    
    T = 298.15
    R = 8.314e-03
    RT = R*T
    N_avogadro = 6.022140857e+23
    VolCell = 1.0e-15
    Concentration2Count = N_avogadro * VolCell
    concentration_increment = 1/(N_avogadro*VolCell)
    
    
    np.set_printoptions(suppress=True)#turn off printin
    # In[3]:
    
    
    with open( cwd + '/GLYCOLYSIS_TCA_GOGAT/GLYCOLYSIS_TCA_GOGAT.dat', 'r') as f:
      print(f.read())
      
    
    # In[5]:
    
    
    fdat = open(cwd + '/GLYCOLYSIS_TCA_GOGAT/GLYCOLYSIS_TCA_GOGAT.dat', 'r')
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
    print(S_active.shape)
    print(S_active)
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
    print(reactions)                
                
    
    
    # ## Calculate Standard Free Energies of Reaction 
    reactions.loc['CSm',deltag0] = -35.1166
    reactions.loc['ACONTm',deltag0] = 7.62949
    reactions.loc['ICDHxm',deltag0] = -2.872
    reactions.loc['AKGDam',deltag0] = -36.3549
    reactions.loc['SUCOASm',deltag0] = 1.924481
    reactions.loc['SUCD1m',deltag0] = 0
    reactions.loc['FUMm',deltag0] = -3.44873
    reactions.loc['MDHm',deltag0] = 29.9942
    reactions.loc['GAPD',deltag0] = 6.68673
    reactions.loc['PGK',deltag0] = -18.4733
    reactions.loc['TPI',deltag0] = 5.48642
    reactions.loc['FBA',deltag0] = 20.5096
    reactions.loc['PYK',deltag0] = -27.5366
    reactions.loc['PGM',deltag0] = 4.19953
    reactions.loc['ENO',deltag0] = -4.08222
    reactions.loc['HEX1',deltag0] = -17.0578
    reactions.loc['PGI',deltag0] = 2.52401
    reactions.loc['PFK',deltag0] = -15.4549
    reactions.loc['PYRt2m',deltag0] = -RT*np.log(10)
    reactions.loc['PDHm',deltag0] = -43.9219
    reactions.loc['GOGAT',deltag0] = 48.8552
    
    reactions.loc['CSm',deltag0_sigma] = 0.930552
    reactions.loc['ACONTm',deltag0_sigma] = 0.733847
    reactions.loc['ICDHxm',deltag0_sigma] = 7.62095
    reactions.loc['AKGDam',deltag0_sigma] = 7.97121
    reactions.loc['SUCOASm',deltag0_sigma] = 1.48197
    reactions.loc['SUCD1m',deltag0_sigma] = 2.31948
    reactions.loc['FUMm',deltag0_sigma] = 0.607693
    reactions.loc['MDHm',deltag0_sigma] = 0.422376
    reactions.loc['GAPD',deltag0_sigma] = 0.895659
    reactions.loc['PGK',deltag0_sigma] = 0.889982
    reactions.loc['TPI',deltag0_sigma] = 0.753116
    reactions.loc['FBA',deltag0_sigma] = 0.87227
    reactions.loc['PYK',deltag0_sigma] = 0.939774
    reactions.loc['PGM',deltag0_sigma] = 0.65542
    reactions.loc['ENO',deltag0_sigma] = 0.734193
    reactions.loc['HEX1',deltag0_sigma] = 0.715237
    reactions.loc['PGI',deltag0_sigma] = 0.596775
    reactions.loc['PFK',deltag0_sigma] = 0.886629
    reactions.loc['PYRt2m',deltag0_sigma] = 0
    reactions.loc['PDHm',deltag0_sigma] = 7.66459
    reactions.loc['GOGAT',deltag0_sigma] = 2.0508

    # ## Set Fixed Concentrations/Boundary Conditions
    
    conc = 'Conc'
    variable = 'Variable'
    conc_exp = 'Conc_Experimental'
    metabolites = pd.DataFrame(index = S_active.columns, columns=[conc,conc_exp,variable])
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
    
    metabolites.loc['ACETYL-COA:MITOCHONDRIA',conc] = 6.06E-04
    metabolites.loc['ACETYL-COA:MITOCHONDRIA',variable] = True
    
    
    metabolites.loc['COA:MITOCHONDRIA',conc] = 1.400000e-03
    metabolites.loc['COA:MITOCHONDRIA',variable] = False
    
    metabolites.loc['CO2:MITOCHONDRIA',conc] = 1.000000e-04
    metabolites.loc['CO2:MITOCHONDRIA',variable] = False
    
    metabolites.loc['H2O:MITOCHONDRIA',conc] = 55.5
    metabolites.loc['H2O:MITOCHONDRIA',variable] = False
    metabolites.loc['H2O:CYTOSOL',conc] = 55.5
    metabolites.loc['H2O:CYTOSOL',variable] = False 
    
    
    metabolites.loc['BETA-D-GLUCOSE:CYTOSOL',conc] = 2.000000e-03
    metabolites.loc['BETA-D-GLUCOSE:CYTOSOL',variable] = False 
    
    metabolites.loc['L-GLUTAMATE:MITOCHONDRIA',conc] = 9.60e-05
    metabolites.loc['L-GLUTAMATE:MITOCHONDRIA',variable] = False 
    metabolites.loc['L-GLUTAMINE:MITOCHONDRIA',conc] = 3.81e-03
    metabolites.loc['L-GLUTAMINE:MITOCHONDRIA',variable] = False 
    
    
    #When loading experimental concentrations, first copy current 
    #rule of thumb then overwrite with data values.
    metabolites[conc_exp] = metabolites[conc]
    metabolites.loc['(S)-MALATE:MITOCHONDRIA',conc_exp] = 1.68e-03
    metabolites.loc['BETA-D-GLUCOSE-6-PHOSPHATE:CYTOSOL',conc_exp] = 7.88e-03
    metabolites.loc['D-GLYCERALDEHYDE-3-PHOSPHATE:CYTOSOL',conc_exp] = 2.71e-04
    metabolites.loc['PYRUVATE:MITOCHONDRIA',conc_exp] = 3.66e-03
    metabolites.loc['ISOCITRATE:MITOCHONDRIA',conc_exp] = 1.000000e-03
    metabolites.loc['OXALOACETATE:MITOCHONDRIA',conc_exp] = 1.000000e-03
    metabolites.loc['3-PHOSPHO-D-GLYCEROYL_PHOSPHATE:CYTOSOL',conc_exp] = 1.000000e-03
    metabolites.loc['ACETYL-COA:MITOCHONDRIA',conc_exp] = 6.06e-04 
    metabolites.loc['CITRATE:MITOCHONDRIA',conc_exp] = 1.96e-03
    metabolites.loc['2-OXOGLUTARATE:MITOCHONDRIA',conc_exp] = 4.43e-04
    metabolites.loc['FUMARATE:MITOCHONDRIA',conc_exp] = 1.15e-04
    metabolites.loc['SUCCINYL-COA:MITOCHONDRIA',conc_exp] = 2.33e-04
    metabolites.loc['3-PHOSPHO-D-GLYCERATE:CYTOSOL',conc_exp] = 1.54e-03
    metabolites.loc['GLYCERONE_PHOSPHATE:CYTOSOL',conc_exp] = 3.060000e-03
    metabolites.loc['SUCCINATE:MITOCHONDRIA',conc_exp] = 5.69e-04
    metabolites.loc['PHOSPHOENOLPYRUVATE:CYTOSOL',conc_exp] = 1.84e-04
    metabolites.loc['D-FRUCTOSE_1,6-BISPHOSPHATE:CYTOSOL',conc_exp] = 1.52e-02
    metabolites.loc['D-FRUCTOSE_6-PHOSPHATE:CYTOSOL',conc_exp] = 2.52e-03
    metabolites.loc['PYRUVATE:CYTOSOL',conc_exp] = 3.66E-03
    metabolites.loc['2-PHOSPHO-D-GLYCERATE:CYTOSOL',conc_exp] = 9.180e-05
    
    #%%
    nvariables = metabolites[metabolites[variable]].count()
    nvar = nvariables[variable]
    
    metabolites.sort_values(by=variable, axis=0,ascending=False, inplace=True,)
    
    
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
    #display(metabolites.index)
    Sactive_index = S_active.index
    
    active_reactions.reindex(index = Sactive_index, copy = False)
    S_active = S_active.reindex(columns = metabolites.index, copy = False)
    S_active['H2O:MITOCHONDRIA'] = 0
    S_active['H2O:CYTOSOL'] = 0
    
    #####################################
    #####################################
    #THIS IS MAKING FLUX -> 0.0
    where_are_NaNs = np.isnan(S_active)
    S_active[where_are_NaNs] = 0
    
    #display(S_active[:])
    
    S_mat = S_active.values
    
    Keq_constant = np.exp(-active_reactions[deltag0].astype('float')/RT)
    #display(Keq_constant)
    Keq_constant = Keq_constant.values
    
    P_mat = np.where(S_mat>0,S_mat,0)
    R_back_mat = np.where(S_mat<0, S_mat, 0)
    E_regulation = np.ones(Keq_constant.size) # THis is the vector of enzyme activities, Range: 0 to 1.
    
    
    mu0 = 1 #Dummy parameter for now; reserved for free energies of formation
    
    #If no experimental data  is available, we can estimate using 'rule-of-thumb' data at 0.001
    use_experimental_data=False
    
    conc_type=conc
    if (use_experimental_data):
        conc_type=conc_exp
    
    variable_concs = np.array(metabolites[conc_type].iloc[0:nvar].values, dtype=np.float64)
    v_log_concs = -10 + 10*np.random.rand(nvar) #Vary between 1 M to 1.0e-10 M
    v_concs = np.exp(v_log_concs)
    v_log_counts_stationary = np.log(v_concs*Concentration2Count)
    v_log_counts = v_log_counts_stationary
    #display(v_log_counts)
    
    fixed_concs = np.array(metabolites[conc_type].iloc[nvar:].values, dtype=np.float64)
    fixed_counts = fixed_concs*Concentration2Count
    f_log_counts = np.log(fixed_counts)
    
    complete_target_log_counts = np.log(Concentration2Count * metabolites[conc_type].values)
    target_v_log_counts = complete_target_log_counts[0:nvar]
    target_f_log_counts = complete_target_log_counts[nvar:]
    
    #WARNING:::::::::::::::CHANGE BACK TO ZEROS
    delta_increment_for_small_concs = (10**-50)*np.zeros(metabolites[conc_type].values.size);
    
    variable_concs_begin = np.array(metabolites[conc_type].iloc[0:nvar].values, dtype=np.float64)
    
    #%% Basic test
    
    v_log_counts = np.log(variable_concs_begin*Concentration2Count)
    
    #r_log_counts = -10 + 10*np.random.rand(v_log_counts.size)
    #v_log_counts = r_log_counts
    print('====== Without adjusting Keq_constant ======')
    
    
    E_regulation = np.ones(Keq_constant.size) # THis is the vector of enzyme activities, Range: 0 to 1.
    nvar = v_log_counts.size
    #WARNING: INPUT LOG_COUNTS TO ALL FUNCTIONS. CONVERSION TO COUNTS IS DONE INTERNALLY
    res_lsq1 = least_squares(max_entropy_functions.derivatives, v_log_counts, method='lm',xtol=1e-15, args=(f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, E_regulation))
    res_lsq2 = least_squares(max_entropy_functions.derivatives, v_log_counts, method='dogbox',xtol=1e-15, args=(f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, E_regulation))
    
    rxn_flux = max_entropy_functions.oddsDiff(res_lsq1.x, f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, E_regulation)
    

    # In[ ]:
    begin_log_metabolites = np.append(res_lsq1.x,f_log_counts)
    ##########################################
    ##########################################
    #####################TESTER###############
    
    E_regulation = np.ones(Keq_constant.size) # THis is the vector of enzyme activities, Range: 0 to 1.
    log_metabolites = np.append(res_lsq1.x,f_log_counts)
    KQ_f = max_entropy_functions.odds(log_metabolites,mu0,S_mat, R_back_mat, P_mat, delta_increment_for_small_concs,Keq_constant)
    
    
    Keq_inverse = np.power(Keq_constant,-1)
    KQ_r = max_entropy_functions.odds(log_metabolites,mu0,-S_mat, P_mat, R_back_mat, delta_increment_for_small_concs,Keq_inverse,-1)
    
    [RR,Jac] = max_entropy_functions.calc_Jac2(res_lsq1.x, f_log_counts, S_mat, delta_increment_for_small_concs, KQ_f, KQ_r, E_regulation)
    A = max_entropy_functions.calc_A(res_lsq1.x,f_log_counts, S_mat, Jac, E_regulation )
    
    [ccc,fcc] = max_entropy_functions.conc_flux_control_coeff(nvar, A, S_mat, rxn_flux, RR)
    
    React_Choice=6
    
    newE = max_entropy_functions.calc_reg_E_step(E_regulation,React_Choice, nvar, res_lsq1.x, f_log_counts, complete_target_log_counts, 
                           S_mat, A, rxn_flux,KQ_f)
        
        
    
    delta_S_metab = max_entropy_functions.calc_deltaS_metab(res_lsq1.x, target_v_log_counts);
    
    ipolicy = 7 #use ipolicy=1 or 4
    reaction_choice = max_entropy_functions.get_enzyme2regulate(ipolicy, delta_S_metab, ccc, KQ_f, E_regulation, res_lsq1.x)                                                        

     #%%
    
    #device = torch.device("cpu")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    
    #set variables in ML program
    me.device=device
    me.v_log_counts_static = v_log_counts_stationary
    me.target_v_log_counts = target_v_log_counts
    me.complete_target_log_counts = complete_target_log_counts
    me.Keq_constant = Keq_constant
    me.f_log_counts = f_log_counts
    
    me.P_mat = P_mat
    me.R_back_mat = R_back_mat
    me.S_mat = S_mat
    me.delta_increment_for_small_concs = delta_increment_for_small_concs
    me.nvar = nvar
    me.mu0 = mu0
    
    me.gamma = gamma
    me.num_rxns = Keq_constant.size
    me.penalty_reward_scalar=penalty_reward_scalar
    
    #%%
    N, D_in, H, D_out = 1, Keq_constant.size,  50*Keq_constant.size, 1

    # Create random Tensors to hold inputs and outputs
    x = torch.rand(1000, D_in, device=device)

    nn_model = torch.nn.Sequential(
            torch.nn.Linear(D_in, H),
            torch.nn.Tanh(),
            torch.nn.Linear(H,D_out)).to(device)

    loss_fn = torch.nn.MSELoss(reduction='sum')
    #learning_rate=5e-6
    #optimizer = torch.optim.SGD(nn_model.parameters(), lr=learning_rate, momentum=0.9)
    optimizer = torch.optim.SGD(nn_model.parameters(), lr=learning_rate, momentum=0.9)
    
    #optimizer = torch.optim.Adam(nn_model.parameters(), lr=3e-4)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100, verbose=True, min_lr=1e-10,cooldown=10,threshold=1e-3)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100, verbose=True, min_lr=1e-10,cooldown=10,threshold=1e-4)
    
    #%% SGD UPDATE TEST

     #attempted iterations to update theta_linear
    v_log_counts = v_log_counts_stationary.copy()
    episodic_loss = []
    episodic_loss_max = []
    episodic_epr = []
    episodic_reward = []
    
    episodic_nn_step = []
    episodic_random_step = []
    
    epsilon_greedy_init = epsilon
    
    final_states=np.zeros(Keq_constant.size)
    epr_per_state=[]
    

    for update in range(0,updates):
        
        x_changing = 10*torch.rand(1000, D_in, device=device)
    
        
        #generate state to use
        state_sample = np.zeros(Keq_constant.size)
        for sample in range(0,len(state_sample)):
            state_sample[sample] = np.random.uniform(1,1)
    
        #annealing test
        if ((update % eps_threshold== 0) and (update != 0)):
            epsilon=epsilon/2
            print("RESET epsilon ANNEALING")
            print(epsilon)
    
        prediction_x_changing_previous = nn_model(x_changing)
        #nn_model.train()
        [sum_reward, average_loss,max_loss,final_epr,final_state, reached_terminal_state,\
         random_steps_taken,nn_steps_taken] = me.sarsa_n(nn_model,loss_fn, optimizer, scheduler, state_sample, n_back_step, epsilon)
        
        print('random,nn steps')
        print(random_steps_taken)
        print(nn_steps_taken)
        if (reached_terminal_state):
            final_states = np.vstack((final_states,final_state))
            epr_per_state.append(final_epr)
            
        scheduler.step(average_loss)
        print("TOTAL REWARD")
        print(sum_reward)
        print("ave loss")
        print(average_loss)
        print("max_loss")
        print(max_loss)
        
        print(optimizer.state_dict)
        print(scheduler.state_dict())
        prediction_x_changing = nn_model(x_changing)
        
        total_prediction_changing_diff = sum(abs(prediction_x_changing - prediction_x_changing_previous))
        print("TOTALPREDICTION")
        print(total_prediction_changing_diff)
        
        #print(list(nn_model.parameters()))
        #print("**********************************************************************")
        #print("EPISODE FINISHED")
        #print("sum")
        #print(sum_reward)
        episodic_epr.append(final_epr)
        
        episodic_loss.append(average_loss)
        
        episodic_loss_max.append(max_loss)
        episodic_reward.append(sum_reward)
        episodic_nn_step.append(nn_steps_taken)
        episodic_random_step.append(random_steps_taken)
        np.savetxt(cwd+'/GLYCOLYSIS_TCA_GOGAT/data/'+'temp_episodic_loss_'+str(n_back_step) +\
                   '_'+str(learning_rate)+'_'+str(eps_threshold)+'_eps'+str(epsilon_greedy_init)+'_'+str(sim_number)+\
                    '_penalty_reward_scalar_' + str(me.penalty_reward_scalar) +
                   '.txt', episodic_loss, fmt='%f')

        np.savetxt(cwd+'/GLYCOLYSIS_TCA_GOGAT/data/'+'temp_epr_'+str(n_back_step) +\
                   '_'+str(learning_rate)+'_'+str(eps_threshold)+'_eps'+str(epsilon_greedy_init)+'_'+str(sim_number)+\
                    '_penalty_reward_scalar_' + str(me.penalty_reward_scalar) +
                   '.txt', episodic_epr, fmt='%f')

        np.savetxt(cwd+'/GLYCOLYSIS_TCA_GOGAT/data/'+'temp_episodic_random_step_'+str(n_back_step)+\
                   '_'+str(learning_rate)+'_'+str(eps_threshold)+'_eps'+str(epsilon_greedy_init)+'_'+str(sim_number)+\
                    '_penalty_reward_scalar_' + str(me.penalty_reward_scalar) +
                   '.txt', episodic_random_step, fmt='%f')
        
        if (update > 200):
            if (max(episodic_loss[-100:])-min(episodic_loss[-100:]) < 0.05):
                break
        #save temporary copies to see
    
    #%%
    #gamma9 -> gamma=0.9
    #n8 -> n_back_step=8
    #k5 -> E=E-E/5 was used 
    #lr5e6 -> begin lr=0.5*e-6
    
    torch.save(nn_model, cwd+'/GLYCOLYSIS_TCA_GOGAT/models_final_data/'+'complete_model_gly_tca_gog_gamma9_n'+str(n_back_step)+'_k5_'\
               +'_lr'+str(learning_rate)+'_threshold'+str(eps_threshold)+'_eps'+str(epsilon_greedy_init)+'_sim'+str(sim_number)+\
                '_penalty_reward_scalar_' + str(me.penalty_reward_scalar) +\
                '.pth')
    
    #%%
    np.savetxt(cwd+'/GLYCOLYSIS_TCA_GOGAT/models_final_data/'+'episodic_loss_gamma9_n'+str(n_back_step)+'_k5_'\
               +'_lr'+str(learning_rate)+'_threshold'+str(eps_threshold)+'_eps'+str(epsilon_greedy_init)+'_sim'+str(sim_number)+\
                '_penalty_reward_scalar_' + str(me.penalty_reward_scalar) +\
                '.txt', episodic_loss, fmt='%f')
    np.savetxt(cwd+'/GLYCOLYSIS_TCA_GOGAT/models_final_data/'+'episodic_loss_max_gamma9_n'+str(n_back_step)+'_k5_'\
               +'_lr'+str(learning_rate)+'_threshold'+str(eps_threshold)+'_eps'+str(epsilon_greedy_init)+'_sim'+str(sim_number)+\
                '_penalty_reward_scalar_' + str(me.penalty_reward_scalar) +\
                '.txt', episodic_loss_max, fmt='%f')
    np.savetxt(cwd+'/GLYCOLYSIS_TCA_GOGAT/models_final_data/'+'episodic_reward_gamma9_n'+str(n_back_step)+'_k5_'\
               +'_lr'+str(learning_rate)+'_threshold'+str(eps_threshold)+'_eps'+str(epsilon_greedy_init)+'_sim'+str(sim_number)+\
                '_penalty_reward_scalar_' + str(me.penalty_reward_scalar) +\
                '.txt', episodic_reward, fmt='%f')
    
    np.savetxt(cwd+'/GLYCOLYSIS_TCA_GOGAT/models_final_data/'+'final_states_gamma9_n'+str(n_back_step)+'_k5_'\
               +'_lr'+str(learning_rate)+'_threshold'+str(eps_threshold)+'_eps'+str(epsilon_greedy_init)+'_sim'+str(sim_number)+\
                '_penalty_reward_scalar_' + str(me.penalty_reward_scalar) +\
                '.txt', final_states, fmt='%f')

    np.savetxt(cwd+'/GLYCOLYSIS_TCA_GOGAT/models_final_data/'+'epr_per_state_gamma9_n'+str(n_back_step)+'_k5_'\
               +'_lr'+str(learning_rate)+'_threshold'+str(eps_threshold)+'_eps'+str(epsilon_greedy_init)+'_sim'+str(sim_number)+\
                '_penalty_reward_scalar_'+str(me.penalty_reward_scalar) +'.txt', epr_per_state, fmt='%f')


#run(sim_number, n_back_step,learning_rate, epsilon, eps_threshold,\
#        gamma = 0.9,updates = 250, penalty_reward_scalar=False): 

updates=1000
gamma=0.9 
penalty_reward_scalar=0.1 #
epsilon_threshold=10
run(1,5,1e-7,1.0,epsilon_threshold, gamma, updates,penalty_reward_scalar)
