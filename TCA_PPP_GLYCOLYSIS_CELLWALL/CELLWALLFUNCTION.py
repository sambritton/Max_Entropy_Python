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
sys.path.insert(0, cwd+'/TCA_PPP_GLYCOLYSIS_CELLWALL')
sys.path.insert(0, cwd+'/Basic_Functions/equilibrator-api-v0.1.8/build/lib')
    
import max_entropy_functions
import machine_learning_functions as me
from scipy.optimize import least_squares
import torch

def run(argv): 

    #default values
    
    #If no experimental data  is available, we can estimate using 'rule-of-thumb' data at 0.001
    use_experimental_data=False
    learning_rate=1e-8 #3rd
    epsilon=0.5 #4th
    eps_threshold=25 #5th
    gamma = 0.9 #6th
    updates = 500
    penalty_reward_scalar=0.0

    #load input
    total = len(sys.argv)
    cmdargs = str(sys.argv)
    print ("The total numbers of args passed to the script: %d " % total)
    print ("Args list: %s " % cmdargs)
    print ("Script name: %s" % str(sys.argv[0]))
    for i in range(total):
        print ("Argument # %d : %s" % (i, str(sys.argv[i])))
    

    sim_number=int(sys.argv[1])
    n_back_step=int(sys.argv[2])
    if (total > 3):
        use_experimental_data=bool(int(sys.argv[3]))
    if (total > 4):
        learning_rate=float(sys.argv[4])
    if (total > 5):
        epsilon=float(sys.argv[5])
    if (total > 6):
        eps_threshold=float(sys.argv[6])
    if (total > 7):
        gamma=float(sys.argv[7])

    
    pd.set_option('display.max_columns', None,'display.max_rows', None)
    
    print("sim")
    print(sim_number)
    print("n_back_step")
    print(n_back_step)
    print("using experimental metabolite data")
    print(use_experimental_data)
    print("learning_rate")
    print(learning_rate)
    print("epsilon")
    print(epsilon)
    print("eps_threshold")
    print(eps_threshold)
    print("gamma")
    print(gamma)

    T = 298.15
    R = 8.314e-03
    RT = R*T
    N_avogadro = 6.022140857e+23
    VolCell = 1.0e-15
    Concentration2Count = N_avogadro * VolCell
    concentration_increment = 1/(N_avogadro*VolCell)
    
    
    np.set_printoptions(suppress=True)#turn off printin
    # In[3]:
    
    
    with open( cwd + '/TCA_PPP_GLYCOLYSIS_CELLWALL/TCA_PPP_Glycolysis_CellWall3b.dat', 'r') as f:
      print(f.read())
      
    
    # In[5]:
    
    
    fdat = open(cwd + '/TCA_PPP_GLYCOLYSIS_CELLWALL/TCA_PPP_Glycolysis_CellWall3b.dat', 'r')
    #fdat = open('TCA_PPP_Glycolysis.dat', 'r')
   
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
                
    reactions.loc['CSm',deltag0] = -35.8057
    reactions.loc['ACONTm',deltag0] = 7.62962
    reactions.loc['ICDHxm',deltag0] = -2.6492
    reactions.loc['AKGDam',deltag0] = -37.245
    reactions.loc['SUCOASm',deltag0] = 2.01842
    reactions.loc['SUCD1m',deltag0] = -379.579
    reactions.loc['FUMm',deltag0] = -3.44728
    reactions.loc['MDHm',deltag0] = 29.5419
    reactions.loc['GAPD',deltag0] = 5.24202
    reactions.loc['PGK',deltag0] = -18.5083
    reactions.loc['TPI',deltag0] = 5.49798
    reactions.loc['FBA',deltag0] = 21.4506
    reactions.loc['PYK',deltag0] = -27.3548
    reactions.loc['PGM',deltag0] = 4.17874
    reactions.loc['ENO',deltag0] = -4.0817
    reactions.loc['HEX1',deltag0] = -16.7776
    reactions.loc['PGI',deltag0] = 2.52206
    reactions.loc['PFK',deltag0] = -16.1049
    reactions.loc['PYRt2m',deltag0] = -RT*np.log(10)
    reactions.loc['PDHm',deltag0] = -44.1315
    reactions.loc['G6PDH2r',deltag0] = -3.89329
    reactions.loc['PGL',deltag0] = -22.0813
    reactions.loc['GND',deltag0] = 2.32254
    reactions.loc['RPE',deltag0] = -3.37
    reactions.loc['RPI',deltag0] = -1.96367
    reactions.loc['TKT2',deltag0] = -10.0342
    reactions.loc['TALA',deltag0] = -0.729232
    reactions.loc['FBA3',deltag0] = 13.9499
    reactions.loc['PFK_3',deltag0] = -9.33337
    reactions.loc['TKT1',deltag0] = -3.79303
    reactions.loc['Glutamine-fructose-6-phosphate aminotransferase',deltag0] = -13.4054
    reactions.loc['Glucosamine-6-phosphate N-acetyltransferase',deltag0] = -23.7065
    reactions.loc['N-acetylglucosamine-phosphate mutase',deltag0] = 4.65558
    reactions.loc['UDP N-acetylglucosamine pyrophosphorylase',deltag0] = 0.539147
    reactions.loc['Hyaluronan Synthase',deltag0] = -14.4143
    reactions.loc['Phosphoglucomutase',deltag0] = 7.41831
    reactions.loc['UTP-glucose-1-phosphate uridylyltransferase',deltag0] = 1.51043
    reactions.loc['1,3-beta-glucan synthase',deltag0] = -11.534
    reactions.loc['Citrate-oxaloacetate exchange',deltag0] = 0
    reactions.loc['CITRATE_LYASE',deltag0] = 10.0299
    reactions.loc['MDHc',deltag0] = -29.5419
    reactions.loc['MDH-NADPc',deltag0] = -29.7376
    reactions.loc['ME1c',deltag0] = 4.56191
    reactions.loc['ME2c',deltag0] = 4.75763
    reactions.loc['Pyruvate Carboxylase',deltag0] = -0.795825
    reactions.loc['Aldose 1-epimerase',deltag0] = 0
    reactions.loc['HEX1a',deltag0] = -16.7776
    reactions.loc['PGI-1',deltag0] = 2.52206
        
    reactions.loc['CSm',deltag0_sigma] = 0.930552
    reactions.loc['ACONTm',deltag0_sigma] = 0.733847
    reactions.loc['ICDHxm',deltag0_sigma] = 7.62095
    reactions.loc['AKGDam',deltag0_sigma] = 7.97121
    reactions.loc['SUCOASm',deltag0_sigma] = 1.48197
    reactions.loc['SUCD1m',deltag0_sigma] = 7.8098
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
    reactions.loc['G6PDH2r',deltag0_sigma] = 2.11855
    reactions.loc['PGL',deltag0_sigma] = 2.62825
    reactions.loc['GND',deltag0_sigma] = 7.60864
    reactions.loc['RPE',deltag0_sigma] = 1.16485
    reactions.loc['RPI',deltag0_sigma] = 1.16321
    reactions.loc['TKT2',deltag0_sigma] = 2.08682
    reactions.loc['TALA',deltag0_sigma] = 1.62106
    reactions.loc['FBA3',deltag0_sigma] = 7.36854
    reactions.loc['PFK_3',deltag0_sigma] = 7.3671
    reactions.loc['TKT1',deltag0_sigma] = 2.16133
    reactions.loc['Glutamine-fructose-6-phosphate aminotransferase',deltag0_sigma] = 3.08807
    reactions.loc['Glucosamine-6-phosphate N-acetyltransferase',deltag0_sigma] = 4.26738
    reactions.loc['N-acetylglucosamine-phosphate mutase',deltag0_sigma] = 3.06369
    reactions.loc['UDP N-acetylglucosamine pyrophosphorylase',deltag0_sigma] = 3.12527
    reactions.loc['Hyaluronan Synthase',deltag0_sigma] = 9.46851
    reactions.loc['Phosphoglucomutase',deltag0_sigma] = 1.09029
    reactions.loc['UTP-glucose-1-phosphate uridylyltransferase',deltag0_sigma] = 1.14644
    reactions.loc['1,3-beta-glucan synthase',deltag0_sigma] = 7.80447
    reactions.loc['Citrate-oxaloacetate exchange',deltag0_sigma] = 0
    reactions.loc['CITRATE_LYASE',deltag0_sigma] = 0.928303
    reactions.loc['MDHc',deltag0_sigma] = 0.422376
    reactions.loc['MDH-NADPc',deltag0_sigma] = 0.531184
    reactions.loc['ME1c',deltag0_sigma] = 7.60174
    reactions.loc['ME2c',deltag0_sigma] = 7.61042
    reactions.loc['Pyruvate Carboxylase',deltag0_sigma] = 7.60419
    reactions.loc['Aldose 1-epimerase',deltag0_sigma] = 0
    reactions.loc['HEX1a',deltag0_sigma] = 0.715237
    reactions.loc['PGI-1',deltag0_sigma] = 0.596775
        # ## Calculate Standard Free Energies of Reaction 
    # In[49]:
    
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
    
    metabolites.loc['UTP:CYTOSOL',conc] = 9.600000e-03
    metabolites.loc['UTP:CYTOSOL',variable] = False
    metabolites.loc['UDP:CYTOSOL',conc] = 5.600000e-04
    metabolites.loc['UDP:CYTOSOL',variable] = False
    metabolites.loc['DIPHOSPHATE:CYTOSOL',conc] = 2.000000e-02
    metabolites.loc['DIPHOSPHATE:CYTOSOL',variable] = False
    
    metabolites.loc['NADH:MITOCHONDRIA',conc] = 8.300000e-05 
    metabolites.loc['NADH:MITOCHONDRIA',variable] = False
    metabolites.loc['NAD+:MITOCHONDRIA',conc] = 2.600000e-03
    metabolites.loc['NAD+:MITOCHONDRIA',variable] = False
    
    metabolites.loc['NADH:CYTOSOL',conc] = 8.300000e-05 
    metabolites.loc['NADH:CYTOSOL',variable] = False
    metabolites.loc['NAD+:CYTOSOL',conc] = 2.600000e-03
    metabolites.loc['NAD+:CYTOSOL',variable] = False
    
    metabolites.loc['NADPH:CYTOSOL',conc] = 8.300000e-05 #also use 1.2e-4
    metabolites.loc['NADPH:CYTOSOL',variable] = False
    metabolites.loc['NADP+:CYTOSOL',conc] = 2.600000e-03 #also use 2.1e-6
    metabolites.loc['NADP+:CYTOSOL',variable] = False
    
    metabolites.loc['COA:MITOCHONDRIA',conc] = 1.400000e-03
    metabolites.loc['COA:MITOCHONDRIA',variable] = False
    metabolites.loc['COA:CYTOSOL',conc] = 1.400000e-03
    metabolites.loc['COA:CYTOSOL',variable] = False
    
    metabolites.loc['CO2:MITOCHONDRIA',conc] = 1.000000e-04
    metabolites.loc['CO2:MITOCHONDRIA',variable] = False
    metabolites.loc['CO2:CYTOSOL',conc] = 1.000000e-04
    metabolites.loc['CO2:CYTOSOL',variable] = False 
    
    metabolites.loc['H2O:MITOCHONDRIA',conc] = 55.5
    metabolites.loc['H2O:MITOCHONDRIA',variable] = False
    metabolites.loc['H2O:CYTOSOL',conc] = 55.5
    metabolites.loc['H2O:CYTOSOL',variable] = False 
    
    metabolites.loc['BETA-D-GLUCOSE:CYTOSOL',conc] = 2.0e-03
    metabolites.loc['BETA-D-GLUCOSE:CYTOSOL',variable] = False 
    
    metabolites.loc["CHITOBIOSE:CYTOSOL",conc] = 2.0e-09
    metabolites.loc["CHITOBIOSE:CYTOSOL",variable] = False 
    
    metabolites.loc['1,3-BETA-D-GLUCAN:CYTOSOL',conc] = 2.0e-09
    metabolites.loc['1,3-BETA-D-GLUCAN:CYTOSOL',variable] = False 
    
    metabolites.loc['L-GLUTAMINE:CYTOSOL',conc] = 2.0e-03
    metabolites.loc['L-GLUTAMINE:CYTOSOL',variable] = False 
    metabolites.loc['L-GLUTAMATE:CYTOSOL',conc] = 2.0e-04
    metabolites.loc['L-GLUTAMATE:CYTOSOL',variable] = False
    metabolites.loc['CELLOBIOSE:CYTOSOL',conc] = 2.0e-04
    metabolites.loc['CELLOBIOSE:CYTOSOL',variable] = False 
    
    metabolites.loc['N-ACETYL-D-GLUCOSAMINE:CYTOSOL',conc] = 1.0e-08
    metabolites.loc['N-ACETYL-D-GLUCOSAMINE:CYTOSOL',variable] = False 
    
    
    #When loading experimental concentrations, first copy current 
    #rule of thumb then overwrite with data values.
    metabolites[conc_exp] = metabolites[conc]
    metabolites.loc['2-OXOGLUTARATE:MITOCHONDRIA',conc_exp] = 0.0000329167257825644
    metabolites.loc['ISOCITRATE:MITOCHONDRIA',conc_exp] = 0.000102471198594958
    metabolites.loc['PHOSPHOENOLPYRUVATE:CYTOSOL',conc_exp] = 0.0000313819870767023
    metabolites.loc['D-GLYCERALDEHYDE-3-PHOSPHATE:CYTOSOL',conc_exp] = 0.0000321630949358949
    metabolites.loc['FUMARATE:MITOCHONDRIA',conc_exp] = 0.00128926137523035
    metabolites.loc['L-GLUTAMINE:CYTOSOL',conc_exp] = 0.0034421144256392
    metabolites.loc['PYRUVATE:MITOCHONDRIA',conc_exp] = 0.0000778160985710288
    metabolites.loc['PYRUVATE:CYTOSOL',conc_exp] = 0.0000778160985710288
    metabolites.loc['D-FRUCTOSE_6-PHOSPHATE:CYTOSOL',conc_exp] = 0.00495190614473117
    metabolites.loc['D-RIBOSE-5-PHOSPHATE:CYTOSOL',conc_exp] = 0.0000849533575412862
    metabolites.loc['CITRATE:MITOCHONDRIA',conc_exp] = 0.000485645834537379
    metabolites.loc['CITRATE:CYTOSOL',conc_exp] = 0.000485645834537379
    metabolites.loc['(S)-MALATE:MITOCHONDRIA',conc_exp] = 0.00213827060541153
    metabolites.loc['(S)-MALATE:CYTOSOL',conc_exp] = 0.00213827060541153
    metabolites.loc['SEDOHEPTULOSE_7-PHOSPHATE:CYTOSOL',conc_exp] = 0.00203246193132095
    metabolites.loc['D-RIBULOSE-5-PHOSPHATE:CYTOSOL',conc_exp] = 0.000468439334729429
    metabolites.loc['L-GLUTAMATE:CYTOSOL',conc_exp] = 0.00557167476932484
    metabolites.loc['SUCCINATE:MITOCHONDRIA',conc_exp] = 0.000942614767220802
    metabolites.loc['D-XYLULOSE-5-PHOSPHATE:CYTOSOL',conc_exp] = 0.000468439334729429
    
    nvariables = metabolites[metabolites[variable]].count()
    nvar = nvariables[variable]
    
    metabolites.sort_values(by=variable, axis=0,ascending=False, inplace=True,)
    print(metabolites)
    
    
    #%%
    nvariables = metabolites[metabolites[variable]].count()
    nvar = nvariables[variable]
    
    metabolites.sort_values(by=variable, axis=0,ascending=False, inplace=True,)
    print(metabolites)
    
    # ## Prepare model for optimization
    
    # - Adjust S Matrix to use only reactions with activity > 0, if necessary.
    # - Water stoichiometry in the stiochiometric matrix needs to be set to zero since water is held constant.
    # - The initial concentrations of the variable metabolites are random.
    # - All concentrations are changed to log counts.
    # - Equilibrium constants are calculated from standard free energies of reaction.
    # - R (reactant) and P (product) matrices are derived from S.
    
    # Make sure all the indices and columns are in the correct order:
    active_reactions = reactions[reactions[enzyme_level] > 0.0]
    #print(reactions)
    #print(metabolites.index)
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
    
    #print(S_active[:])
    
    S_mat = S_active.values
    
    Keq_constant = np.exp(-active_reactions[deltag0].astype('float')/RT)
    #print(Keq_constant)
    Keq_constant = Keq_constant.values
    
    P_mat = np.where(S_mat>0,S_mat,0)
    R_back_mat = np.where(S_mat<0, S_mat, 0)
    E_regulation = np.ones(Keq_constant.size) # THis is the vector of enzyme activities, Range: 0 to 1.
    
    
    mu0 = 1 #Dummy parameter for now; reserved for free energies of formation
    
    #If no experimental data  is available, we can estimate using 'rule-of-thumb' data at 0.001
    
    conc_type=conc
    if (use_experimental_data):
        conc_type=conc_exp
    
    variable_concs = np.array(metabolites[conc_type].iloc[0:nvar].values, dtype=np.float64)
    v_log_concs = -10 + 10*np.random.rand(nvar) #Vary between 1 M to 1.0e-10 M
    v_concs = np.exp(v_log_concs)
    v_log_counts_stationary = np.log(v_concs*Concentration2Count)
    v_log_counts = v_log_counts_stationary
    #print(v_log_counts)
    
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
    if (res_lsq1.success==False):
        res_lsq1 = least_squares(max_entropy_functions.derivatives, v_log_counts,method='dogbox',xtol=1e-15, args=(f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, E_regulation))
        if (res_lsq1.success==False):
            res_lsq1 = least_squares(max_entropy_functions.derivatives, v_log_counts,method='trf',xtol=1e-15, args=(f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, E_regulation))
            
    
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
    
    device = torch.device("cpu")
    
    
    
    
    
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
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100, verbose=True, min_lr=1e-10,cooldown=10,threshold=1e-5)
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
        np.savetxt(cwd+'/TCA_PPP_GLYCOLYSIS_CELLWALL/data/'+
                    'temp_episodic_loss_'+str(n_back_step) +
                    '_lr'+str(learning_rate)+
                    '_'+str(eps_threshold)+
                    '_eps'+str(epsilon_greedy_init)+
                    '_'+str(sim_number)+
                    '_penalty_reward_scalar_'+str(me.penalty_reward_scalar)+
                    '_use_experimental_metab_'+str(int(use_experimental_data))+ 
                    '.txt', episodic_loss, fmt='%f')

        np.savetxt(cwd+'/TCA_PPP_GLYCOLYSIS_CELLWALL/data/'+
                    'temp_epr_'+str(n_back_step) +
                    '_lr'+str(learning_rate)+
                    '_'+str(eps_threshold)+
                    '_eps'+str(epsilon_greedy_init)+
                    '_'+str(sim_number)+
                    '_penalty_reward_scalar_'+str(me.penalty_reward_scalar)+
                    '_use_experimental_metab_'+str(int(use_experimental_data))+
                    '.txt', episodic_epr, fmt='%f')

        np.savetxt(cwd+'/TCA_PPP_GLYCOLYSIS_CELLWALL/data/'+
                    'temp_episodic_random_step_'+str(n_back_step)+
                    '_lr'+str(learning_rate)+
                    '_'+str(eps_threshold)+
                    '_eps'+str(epsilon_greedy_init)+'_'+str(sim_number)+
                    '_penalty_reward_scalar_'+str(me.penalty_reward_scalar)+
                    '_use_experimental_metab_'+str(int(use_experimental_data))+
                    '.txt', episodic_random_step, fmt='%f')
        
        if (update > 200):
            if ((max(episodic_loss[-100:])-min(episodic_loss[-100:]) < 0.025) and (update > 350)):
                break
        
    
    #%%
    #gamma9 -> gamma=0.9
    #n8 -> n_back_step=8
    #k5 -> E=E-E/5 was used 
    #lr5e6 -> begin lr=0.5*e-6
    
    torch.save(nn_model, cwd+'/TCA_PPP_GLYCOLYSIS_CELLWALL/models_final_data/'+
                'complete_model_gly_tca_gog_gamma9_n'+str(n_back_step)+'_k5_'\
                '_lr'+str(learning_rate)+
                '_threshold'+str(eps_threshold)+
                '_eps'+str(epsilon_greedy_init)+
                '_penalty_reward_scalar_'+str(me.penalty_reward_scalar)+
                '_use_experimental_metab_' +str(int(use_experimental_data))+
                '_sim'+str(sim_number) + '.pth')
    
    
    np.savetxt(cwd+'/TCA_PPP_GLYCOLYSIS_CELLWALL/models_final_data/'+
                'episodic_loss_gamma9_n'+str(n_back_step)+'_k5_'
                '_lr'+str(learning_rate)+
                '_threshold'+str(eps_threshold)+
                '_eps'+str(epsilon_greedy_init)+
                '_penalty_reward_scalar_'+str(me.penalty_reward_scalar)+
                '_use_experimental_metab_'+str(int(use_experimental_data))+
                '_sim'+str(sim_number)+
                '.txt', episodic_loss, fmt='%f')

    np.savetxt(cwd+'/TCA_PPP_GLYCOLYSIS_CELLWALL/models_final_data/'+
                'episodic_loss_max_gamma9_n'+str(n_back_step)+'_k5_'+
                '_lr'+str(learning_rate)+
                '_threshold'+str(eps_threshold)+'_eps'+str(epsilon_greedy_init)+
                '_penalty_reward_scalar_'+str(me.penalty_reward_scalar)+
                '_use_experimental_metab_'+str(int(use_experimental_data))+
                '_sim'+str(sim_number)+
                '.txt', episodic_loss_max, fmt='%f')

    np.savetxt(cwd+'/TCA_PPP_GLYCOLYSIS_CELLWALL/models_final_data/'+
                'episodic_reward_gamma9_n'+str(n_back_step)+'_k5_'+
                '_lr'+str(learning_rate)+
               '_threshold'+str(eps_threshold)+
               '_eps'+str(epsilon_greedy_init)+
                '_penalty_reward_scalar_'+str(me.penalty_reward_scalar)+
                '_use_experimental_metab_'+str(int(use_experimental_data))+
                '_sim'+str(sim_number)+
                '.txt', episodic_reward, fmt='%f')
    
    np.savetxt(cwd+'/TCA_PPP_GLYCOLYSIS_CELLWALL/models_final_data/'+
                'final_states_gamma9_n'+str(n_back_step)+'_k5_'+
                '_lr'+str(learning_rate)+
                '_threshold'+str(eps_threshold)+
                '_eps'+str(epsilon_greedy_init)+
                '_penalty_reward_scalar_'+str(me.penalty_reward_scalar)+
                '_use_experimental_metab_'+str(int(use_experimental_data))+
                '_sim'+str(sim_number)+\
                '.txt', final_states, fmt='%f')

    np.savetxt(cwd+'/TCA_PPP_GLYCOLYSIS_CELLWALL/models_final_data/'+
                'epr_per_state_gamma9_n'+str(n_back_step)+'_k5_'+
                '_lr'+str(learning_rate)+
                '_threshold'+str(eps_threshold)+
                '_eps'+str(epsilon_greedy_init)+
                '_penalty_reward_scalar_'+str(me.penalty_reward_scalar)+
                '_use_experimental_metab_' +str(int(use_experimental_data))+
                '_sim'+str(sim_number)+
                '.txt', epr_per_state, fmt='%f')

if __name__ == '__main__':
    # Map command line arguments to function arguments.
    print(sys.argv[:])
    print(*sys.argv[:])
    run(sys.argv[:])    
    
    
    
    