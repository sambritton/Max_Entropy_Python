# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 18:58:22 2019

@author: samuel_britton
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 20:58:44 2019

@author: samuel_britton
"""
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
Temperature = 298.15
R_gas = 8.314e-03
RT = R_gas*Temperature

pathway_choice=1
use_exp_data=1

lr1=str('0.0001')
lr2=str('1e-05')
lr3=str('1e-06')
lrs=[lr1,lr2,lr3]
#lrs=[lr1,lr2]

Fontsize_Title=20
Fontsize_Sub = 15
Fontsize_Leg = 15
figure_norm=10
marker_size = 50


#cwd = 'C:\\Users\\samuel_britton\\Documents\\cannon\\Final_Pathways'
cwd = 'C:\\Users\\samuel_britton\\Documents\\cannon\\regulation_paper'

p1_string = '\\GLUCONEOGENESIS_noconstraint\\models_final_data' #RL not really applicable so far

#This is R2
p2_string = '\\GLYCOLYSIS_TCA_GOGAT\\models_final_data'
p3_string = '\\TCA_PPP_GLYCOLYSIS_GOGAT\\models_final_data'
p4_string = '\\TCA_PPP_GLYCOLYSIS_GOGAT_LOWERCONC\\models_final_data'
p5_string = '\\TCA_PPP_GLYCOLYSIS_GOGAT_NOPFK\\models_final_data' #changed for different NN architecture

#this is R1
p6_string = '\\GLYCOLYSIS_TCA_GOGAT_noconstraint\\models_final_data'
p7_string = '\\TCA_PPP_GLYCOLYSIS_GOGAT_noconstraint\\models_final_data'
p8_string = '\\TCA_PPP_GLYCOLYSIS_GOGAT_LOWERCONC_noconstraint\\models_final_data'#changed for different NN architecture
p9_string = '\\TCA_PPP_GLYCOLYSIS_GOGAT_NOPFK_noconstraint\\models_final_data_k10divisor'


p1_string_short = '\\GLUCONEOGENESIS'
p2_string_short = '\\GLYCOLYSIS_TCA_GOGAT'
p3_string_short = '\\TCA_PPP_GLYCOLYSIS_GOGAT'
p4_string_short = '\\TCA_PPP_GLYCOLYSIS_GOGAT_LOWERCONC'
p5_string_short = '\\TCA_PPP_GLYCOLYSIS_GOGAT_NOPFK'

p6_string_short = '\\GLYCOLYSIS_TCA_GOGAT_noconstraint'
p7_string_short = '\\TCA_PPP_GLYCOLYSIS_GOGAT_noconstraint'
p8_string_short = '\\TCA_PPP_GLYCOLYSIS_GOGAT_LOWERCONC_noconstraint'
p9_string_short = '\\TCA_PPP_GLYCOLYSIS_GOGAT_NOPFK_noconstraint'

bigg_reaction_abbr_p1 = pd.Series(['GLUCOSE_6_PHOSPHATASE', 'PGI', 'FBP', 'FBA', 'TPI', 'GAPD', 'PGK',
       'PGM', 'ENO', 'PEP_Carboxykinase', 'Pyruvate_Carboxylase'])

bigg_reaction_abbr_p2 = pd.Series(['CSm','ACONTm', 'ICDHxm', 'AKGDam', 'SUCOASm', 'SUCD1m', 'FUMm',
       'MDHm', 'GAPD', 'PGK', 'TPI', 'FBA', 'PYK', 'PGM', 'ENO', 'HEX1', 'PGI',
       'PFK', 'PYRt2m', 'PDHm', 'GOGAT'])

bigg_reaction_abbr_p3 = pd.Series(['CSm','ACONTm','ICDHxm','AKGDam','SUCOASm',
        'SUCD1m','FUMm','MDHm','GAPD','PGK','TPI','FBA','PYK','PGM','ENO',
        'HEX1','PGI','PFK','PYRt2m','PDHm','G6PDH2r','PGL','GND','RPE','RPI',
        'TKT2','TALA','TKT1','GOGAT'])

bigg_reaction_abbr_p4 = pd.Series(['CSM','ACONT','ICDH','AKGD','SUCOAS',
        'SUCD','FUM','MDH','GAPD','PGK','TPI','FBA','PYK','PGM','ENO',
        'HEX1','PGI','PFK','PYRt2M','PDH','G6PDH','PGL','GND','RPE','RPI',
        'TKT2','TALA','TKT1','GOGAT'])

bigg_reorder_p4 = [15, 16, 17, 11, 10, 8, 9, 13, 14, 12, 18, 19, 
                   20, 21, 22, 24, 23, 26, 25,
                   0, 1, 2, 3, 4, 5, 6, 7, 28]


deltag0='DGZERO'


dg0_gluconeogenesis = pd.DataFrame(index=[],columns=[deltag0])

dg0_gluconeogenesis.loc['GLUCOSE_6_PHOSPHATASE',deltag0] = -8.99819
dg0_gluconeogenesis.loc['PGI',deltag0] = -2.52206
dg0_gluconeogenesis.loc['FBP',deltag0] = -9.67092
dg0_gluconeogenesis.loc['FBA',deltag0] = -21.4506
dg0_gluconeogenesis.loc['TPI',deltag0] = -5.49798
dg0_gluconeogenesis.loc['GAPD',deltag0] = -5.24202
dg0_gluconeogenesis.loc['PGK',deltag0] = 18.5083
dg0_gluconeogenesis.loc['PGM',deltag0] = -4.17874
dg0_gluconeogenesis.loc['ENO',deltag0] = 4.0817
dg0_gluconeogenesis.loc['PEP_Carboxykinase',deltag0] = 2.37487
dg0_gluconeogenesis.loc['Pyruvate_Carboxylase',deltag0] = -0.795825

Keq_constant_gluconeogenesis = np.exp(-dg0_gluconeogenesis[deltag0].astype('float')/RT)

dg0_tca_gly = pd.DataFrame(index=[],columns=[deltag0])
dg0_tca_gly.loc['CSm',deltag0] = -35.8057
dg0_tca_gly.loc['ACONTm',deltag0] = 7.62962
dg0_tca_gly.loc['ICDHxm',deltag0] = -2.6492
dg0_tca_gly.loc['AKGDam',deltag0] = -37.245
dg0_tca_gly.loc['SUCOASm',deltag0] = 2.01842
dg0_tca_gly.loc['SUCD1m',deltag0] = 0
dg0_tca_gly.loc['FUMm',deltag0] = -3.44728
dg0_tca_gly.loc['MDHm',deltag0] = 29.5419
dg0_tca_gly.loc['GAPD',deltag0] = 5.24202
dg0_tca_gly.loc['PGK',deltag0] = -18.5083
dg0_tca_gly.loc['TPI',deltag0] = 5.49798
dg0_tca_gly.loc['FBA',deltag0] = 21.4506
dg0_tca_gly.loc['PYK',deltag0] = -27.3548
dg0_tca_gly.loc['PGM',deltag0] = 4.17874
dg0_tca_gly.loc['ENO',deltag0] = -4.0817
dg0_tca_gly.loc['HEX1',deltag0] = -16.7776
dg0_tca_gly.loc['PGI',deltag0] = 2.52206
dg0_tca_gly.loc['PFK',deltag0] = -16.1049
dg0_tca_gly.loc['PYRt2m',deltag0] = -RT*np.log(10)
dg0_tca_gly.loc['PDHm',deltag0] = -44.1315
dg0_tca_gly.loc['GOGAT',deltag0] = 48.1864

Keq_constant_tca_gly = np.exp(-dg0_tca_gly[deltag0].astype('float')/RT)

dg0_tca_gly_ppp = pd.DataFrame(index=[],columns=[deltag0])
dg0_tca_gly_ppp.loc['CSm',deltag0] = -35.8057
dg0_tca_gly_ppp.loc['ACONTm',deltag0] = 7.62962
dg0_tca_gly_ppp.loc['ICDHxm',deltag0] = -2.6492
dg0_tca_gly_ppp.loc['AKGDam',deltag0] = -37.245
dg0_tca_gly_ppp.loc['SUCOASm',deltag0] = 2.01842
dg0_tca_gly_ppp.loc['SUCD1m',deltag0] = 0
dg0_tca_gly_ppp.loc['FUMm',deltag0] = -3.44728
dg0_tca_gly_ppp.loc['MDHm',deltag0] = 29.5419
dg0_tca_gly_ppp.loc['GAPD',deltag0] = 5.24202
dg0_tca_gly_ppp.loc['PGK',deltag0] = -18.5083
dg0_tca_gly_ppp.loc['TPI',deltag0] = 5.49798
dg0_tca_gly_ppp.loc['FBA',deltag0] = 21.4506
dg0_tca_gly_ppp.loc['PYK',deltag0] = -27.3548
dg0_tca_gly_ppp.loc['PGM',deltag0] = 4.17874
dg0_tca_gly_ppp.loc['ENO',deltag0] = -4.0817
dg0_tca_gly_ppp.loc['HEX1',deltag0] = -16.7776
dg0_tca_gly_ppp.loc['PGI',deltag0] = 2.52206
dg0_tca_gly_ppp.loc['PFK',deltag0] = -16.1049
dg0_tca_gly_ppp.loc['PYRt2m',deltag0] = -RT*np.log(10)
dg0_tca_gly_ppp.loc['PDHm',deltag0] = -44.1315
dg0_tca_gly_ppp.loc['G6PDH2r',deltag0] = -3.89329
dg0_tca_gly_ppp.loc['PGL',deltag0] = -22.0813
dg0_tca_gly_ppp.loc['GND',deltag0] = 2.32254
dg0_tca_gly_ppp.loc['RPE',deltag0] = -3.37
dg0_tca_gly_ppp.loc['RPI',deltag0] = -1.96367
dg0_tca_gly_ppp.loc['TKT2',deltag0] = -10.0342
dg0_tca_gly_ppp.loc['TALA',deltag0] = -0.729232
dg0_tca_gly_ppp.loc['TKT1',deltag0] = -3.79303
dg0_tca_gly_ppp.loc['GOGAT',deltag0] = 48.1864

Keq_constant_tca_gly_ppp = np.exp(-dg0_tca_gly_ppp[deltag0].astype('float')/RT)

MCA_method_type1 = 'Method 1'
MCA_method_type2 = 'Method 2'

MCA_color_type1 = 'r'
MCA_color_type2 = 'r'
MCA_shape_type1 = 's'
MCA_shape_type2 = 'x'

RL_method_type1 = 'RL Unconstrained'
RL_method_type2 = 'RL Constrained'

RL_color_type1 = 'b'
RL_color_type2 = 'b'
RL_shape_type1 = 's'
RL_shape_type2 = 'x'


epr_method1_experimental_path1 = 0.96305606
epr_method2_experimental_path1 = 0.96305606
#activity_method2_experimental_path2=1
activity_method1_experimental_path1 = np.loadtxt(cwd+'\\GLUCONEOGENESIS\\'+'activities_experiment_method1.txt', dtype=float)
activity_method2_experimental_path1 = np.loadtxt(cwd+'\\GLUCONEOGENESIS\\'+'activities_experiment_method2.txt', dtype=float)
flux_method1_experimental_path1 = np.loadtxt(cwd+'\\GLUCONEOGENESIS\\'+'flux_experiment_method1.txt', dtype=float)
flux_method2_experimental_path1 = np.loadtxt(cwd+'\\GLUCONEOGENESIS\\'+'flux_experiment_method2.txt', dtype=float)
KQF_method1_experimental_path1 = np.loadtxt(cwd+'\\GLUCONEOGENESIS\\'+'KQF_experiment_method1.txt', dtype=float)
KQF_method2_experimental_path1 = np.loadtxt(cwd+'\\GLUCONEOGENESIS\\'+'KQF_experiment_method2.txt', dtype=float)
KQR_method1_experimental_path1 = np.loadtxt(cwd+'\\GLUCONEOGENESIS\\'+'KQF_experiment_method1.txt', dtype=float)
KQR_method2_experimental_path1 = np.loadtxt(cwd+'\\GLUCONEOGENESIS\\'+'KQF_experiment_method2.txt', dtype=float)


epr_method1_experimental_path2 = 4.1382
epr_method2_experimental_path2 = 4.13458403

activity_method1_experimental_path2 = np.loadtxt(cwd+'\\GLYCOLYSIS_TCA_GOGAT\\'+'activities_experiment_method1.txt', dtype=float)
activity_method2_experimental_path2 = np.loadtxt(cwd+'\\GLYCOLYSIS_TCA_GOGAT\\'+'activities_experiment_method2.txt', dtype=float)
flux_method1_experimental_path2 = np.loadtxt(cwd+'\\GLYCOLYSIS_TCA_GOGAT\\'+'flux_experiment_method1.txt', dtype=float)
flux_method2_experimental_path2 = np.loadtxt(cwd+'\\GLYCOLYSIS_TCA_GOGAT\\'+'flux_experiment_method2.txt', dtype=float)
KQF_method1_experimental_path2 = np.loadtxt(cwd+'\\GLYCOLYSIS_TCA_GOGAT\\'+'KQF_experiment_method1.txt', dtype=float)
KQF_method2_experimental_path2 = np.loadtxt(cwd+'\\GLYCOLYSIS_TCA_GOGAT\\'+'KQF_experiment_method2.txt', dtype=float)
KQR_method1_experimental_path2 = np.loadtxt(cwd+'\\GLYCOLYSIS_TCA_GOGAT\\'+'KQR_experiment_method1.txt', dtype=float)
KQR_method2_experimental_path2 = np.loadtxt(cwd+'\\GLYCOLYSIS_TCA_GOGAT\\'+'KQR_experiment_method2.txt', dtype=float)


epr_method1_experimental_path3 = 4.0369069
epr_method2_experimental_path3 = 3.9909399
activity_method1_experimental_path3 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'activities_experiment_method1_highhigh.txt', dtype=float)
activity_method2_experimental_path3 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'activities_experiment_method2_highhigh.txt', dtype=float)
flux_method1_experimental_path3 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'flux_experiment_method1_highhigh.txt', dtype=float)
flux_method2_experimental_path3 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'flux_experiment_method2_highhigh.txt', dtype=float)
KQF_method1_experimental_path3 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'KQF_experiment_method1_highhigh.txt', dtype=float)
KQF_method2_experimental_path3 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'KQF_experiment_method2_highhigh.txt', dtype=float)
KQR_method1_experimental_path3 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'KQR_experiment_method1_highhigh.txt', dtype=float)
KQR_method2_experimental_path3 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'KQR_experiment_method2_highhigh.txt', dtype=float)


epr_method1_experimental_path4 = 4.00244750
epr_method2_experimental_path4 = 3.9271418556
activity_method1_experimental_path4 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'activities_experiment_method1_highlow.txt', dtype=float)
activity_method2_experimental_path4 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'activities_experiment_method2_highlow.txt', dtype=float)
flux_method1_experimental_path4 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'flux_experiment_method1_highlow.txt', dtype=float)
flux_method2_experimental_path4 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'flux_experiment_method2_highlow.txt', dtype=float)
KQF_method1_experimental_path4 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'KQF_experiment_method1_highlow.txt', dtype=float)
KQF_method2_experimental_path4 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'KQF_experiment_method2_highlow.txt', dtype=float)
KQR_method1_experimental_path4 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'KQR_experiment_method1_highlow.txt', dtype=float)
KQR_method2_experimental_path4 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'KQR_experiment_method2_highlow.txt', dtype=float)


epr_method1_experimental_path5 = 0.01683762
epr_method2_experimental_path5 = 3.44467
activity_method1_experimental_path5 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'activities_experiment_method1_highhigh_PFKZERO.txt', dtype=float)
activity_method2_experimental_path5 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'activities_experiment_method2_highhigh_PFKZERO.txt', dtype=float)
flux_method1_experimental_path5 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'flux_experiment_method1_highhigh_PFKZERO.txt', dtype=float)
flux_method2_experimental_path5 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'flux_experiment_method2_highhigh_PFKZERO.txt', dtype=float)
KQF_method1_experimental_path5 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'KQF_experiment_method1_highhigh_PFKZERO.txt', dtype=float)
KQF_method2_experimental_path5 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'KQF_experiment_method2_highhigh_PFKZERO.txt', dtype=float)
KQR_method1_experimental_path5 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'KQR_experiment_method1_highhigh_PFKZERO.txt', dtype=float)
KQR_method2_experimental_path5 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'KQR_experiment_method2_highhigh_PFKZERO.txt', dtype=float)


begin_sim=1

if (pathway_choice==1):
    Keq_constant = Keq_constant_gluconeogenesis
    
    epr_to_plot_method1 = epr_method1_experimental_path1
    epr_to_plot_method2 = epr_method2_experimental_path1
    activity_to_plot_method1=activity_method1_experimental_path1
    activity_to_plot_method2=activity_method2_experimental_path1
    flux_to_plot_method1=flux_method1_experimental_path1
    flux_to_plot_method2=flux_method2_experimental_path1
    KQF_to_plot_method1 = KQF_method1_experimental_path1
    KQF_to_plot_method2 = KQF_method2_experimental_path1
    KQR_to_plot_method1 = KQR_method1_experimental_path1
    KQR_to_plot_method2 = KQR_method2_experimental_path1
    
    
    activity_to_plot_MCA = activity_to_plot_method1
    epr_to_plot_MCA = epr_to_plot_method1
    flux_to_plot_MCA = flux_to_plot_method1
    KQF_to_plot_MCA = KQF_to_plot_method1
    Pathway_Name='GLUCONEOGENESIS'
        
    #p_string_random = cwd + '\\GLYCOLYSIS_TCA_GOGAT_randomds'
    p_string = cwd + p1_string
    bigg_reaction_abbr_p = bigg_reaction_abbr_p1
    eps=0.5
    
    MCA_method_type = MCA_method_type1
    MCA_color_type = MCA_color_type1
    MCA_shape_type = MCA_shape_type1
    
    RL_method_type = RL_method_type1
    RL_color_type = RL_color_type1
    RL_shape_type = RL_shape_type1

if (pathway_choice==2):
    epr_to_plot_method1 = epr_method1_experimental_path2
    epr_to_plot_method2 = epr_method2_experimental_path2
    activity_to_plot_method1=activity_method1_experimental_path2
    activity_to_plot_method2=activity_method2_experimental_path2
    flux_to_plot_method1=flux_method1_experimental_path2
    flux_to_plot_method2=flux_method2_experimental_path2
    KQF_to_plot_method1 = KQF_method1_experimental_path2
    KQF_to_plot_method2 = KQF_method2_experimental_path2
    KQR_to_plot_method1 = KQR_method1_experimental_path2
    KQR_to_plot_method2 = KQR_method2_experimental_path2
    
    
    activity_to_plot_MCA = activity_to_plot_method2
    epr_to_plot_MCA = epr_to_plot_method2
    flux_to_plot_MCA = flux_to_plot_method2
    KQF_to_plot_MCA = KQF_to_plot_method2
    
    Pathway_Name='GLYCOLYSIS'
    Pathway_Name_nospace='GLYCOLYSIS_constraint'
        
    #p_string_random = cwd + '\\GLYCOLYSIS_TCA_GOGAT_randomds'
    p_string = cwd + p2_string
    bigg_reaction_abbr_p = bigg_reaction_abbr_p2
    eps=0.5
    
    MCA_method_type = MCA_method_type2
    MCA_color_type = MCA_color_type2
    MCA_shape_type = MCA_shape_type2
    
    RL_method_type = RL_method_type2
    RL_color_type = RL_color_type2
    RL_shape_type = RL_shape_type2

#TCA_PPP_GLY with high/high
if (pathway_choice==3):
    Keq_constant = Keq_constant_tca_gly_ppp
    
    epr_to_plot_method1 = epr_method1_experimental_path3
    epr_to_plot_method2 = epr_method2_experimental_path3
    activity_to_plot_method1=activity_method1_experimental_path3
    activity_to_plot_method2=activity_method2_experimental_path3
    flux_to_plot_method1=flux_method1_experimental_path3
    flux_to_plot_method2=flux_method2_experimental_path3
    KQF_to_plot_method1 = KQF_method1_experimental_path3
    KQF_to_plot_method2 = KQF_method2_experimental_path3
    KQR_to_plot_method1 = KQR_method1_experimental_path3
    KQR_to_plot_method2 = KQR_method2_experimental_path3
    
    
    activity_to_plot_MCA = activity_to_plot_method2
    epr_to_plot_MCA = epr_to_plot_method2
    flux_to_plot_MCA = flux_to_plot_method2
    KQF_to_plot_MCA = KQF_to_plot_method2
    
    Pathway_Name='GLYCOLYSIS_TCA_PPP, High/High'
    Pathway_Name_nospace='GLYCOLYSIS_TCA_PPP_high_high_constraint'
        
    #p_string_random = cwd + '\\TCA_PPP_GLYCOLYSIS_GOGAT_randomds_highhigh'
    p_string = cwd + p3_string
    bigg_reaction_abbr_p = bigg_reaction_abbr_p3
    eps=0.2
    
    MCA_method_type = MCA_method_type2
    MCA_color_type = MCA_color_type2
    MCA_shape_type = MCA_shape_type2
    
    RL_method_type = RL_method_type2
    RL_color_type = RL_color_type2
    RL_shape_type = RL_shape_type2
    


#TCA_PPP_GLY with high/low
if (pathway_choice==4):
    Keq_constant = Keq_constant_tca_gly_ppp
    
    epr_to_plot_method1 = epr_method1_experimental_path4
    epr_to_plot_method2 = epr_method2_experimental_path4
    activity_to_plot_method1=activity_method1_experimental_path4
    activity_to_plot_method2=activity_method2_experimental_path4
    flux_to_plot_method1=flux_method1_experimental_path4
    flux_to_plot_method2=flux_method2_experimental_path4
    KQF_to_plot_method1 = KQF_method1_experimental_path4
    KQF_to_plot_method2 = KQF_method2_experimental_path4
    KQR_to_plot_method1 = KQR_method1_experimental_path4
    KQR_to_plot_method2 = KQR_method2_experimental_path4
    
    
    activity_to_plot_MCA = activity_to_plot_method2
    epr_to_plot_MCA = epr_to_plot_method2
    flux_to_plot_MCA = flux_to_plot_method2
    KQF_to_plot_MCA = KQF_to_plot_method2
    
    Pathway_Name='GLYCOLYSIS_TCA_PPP, High/Low'
    Pathway_Name_nospace='GLYCOLYSIS_TCA_PPP_high_low_constraint'
        
    p_string = cwd + p4_string
    
    bigg_reaction_abbr_p = bigg_reaction_abbr_p3
    eps=0.2
        
    MCA_method_type = MCA_method_type2
    MCA_color_type = MCA_color_type2
    MCA_shape_type = MCA_shape_type2
    
    RL_method_type = RL_method_type2
    RL_color_type = RL_color_type2
    RL_shape_type = RL_shape_type2

#TCA_PPP_GLY ZERO_PFK
if (pathway_choice==5):
    Keq_constant = Keq_constant_tca_gly_ppp
    
    epr_to_plot_method1 = epr_method1_experimental_path5
    epr_to_plot_method2 = epr_method2_experimental_path5
    activity_to_plot_method1=activity_method1_experimental_path5
    activity_to_plot_method2=activity_method2_experimental_path5
    flux_to_plot_method1=flux_method1_experimental_path5
    flux_to_plot_method2=flux_method2_experimental_path5
    KQF_to_plot_method1 = KQF_method1_experimental_path5
    KQF_to_plot_method2 = KQF_method2_experimental_path5
    KQR_to_plot_method1 = KQR_method1_experimental_path5
    KQR_to_plot_method2 = KQR_method2_experimental_path5
    
    
    activity_to_plot_MCA = activity_to_plot_method2
    epr_to_plot_MCA = epr_to_plot_method2
    flux_to_plot_MCA = flux_to_plot_method2
    KQF_to_plot_MCA = KQF_to_plot_method2
    
    Pathway_Name='GLYCOLYSIS_TCA_PPP ZERO_PFK'
    Pathway_Name_nospace='GLYCOLYSIS_TCA_PPP_nopfk_high_high_constraint'
        
    p_string = cwd + p5_string
    
    bigg_reaction_abbr_p = bigg_reaction_abbr_p4
    eps=0.2
    
    MCA_method_type = MCA_method_type2
    MCA_color_type = MCA_color_type2
    MCA_shape_type = MCA_shape_type2
    
    RL_method_type = RL_method_type2
    RL_color_type = RL_color_type2
    RL_shape_type = RL_shape_type2

#Glycolysis with no constraint
if (pathway_choice==6):
    
    Keq_constant = Keq_constant_tca_gly
    
    epr_to_plot_method1 = epr_method1_experimental_path2
    epr_to_plot_method2 = epr_method2_experimental_path2
    activity_to_plot_method1=activity_method1_experimental_path2
    activity_to_plot_method2=activity_method2_experimental_path2
    flux_to_plot_method1=flux_method1_experimental_path2
    flux_to_plot_method2=flux_method2_experimental_path2
    KQF_to_plot_method1 = KQF_method1_experimental_path2
    KQF_to_plot_method2 = KQF_method2_experimental_path2
    KQR_to_plot_method1 = KQR_method1_experimental_path2
    KQR_to_plot_method2 = KQR_method2_experimental_path2
    
    
    activity_to_plot_MCA = activity_to_plot_method1
    epr_to_plot_MCA = epr_to_plot_method1
    flux_to_plot_MCA = flux_to_plot_method1
    KQF_to_plot_MCA = KQF_to_plot_method1
    
    Pathway_Name='GLYCOLYSIS no constraint'
    Pathway_Name_nospace='GLYCOLYSISnoconstraint'
        
    p_string_random = cwd + '\\GLYCOLYSIS_TCA_GOGAT_noconstraint'
    p_string = cwd + p6_string
    bigg_reaction_abbr_p = bigg_reaction_abbr_p2
    eps=0.5
    
    MCA_method_type = MCA_method_type1
    MCA_color_type = MCA_color_type1
    MCA_shape_type = MCA_shape_type1
    
    RL_method_type = RL_method_type1
    RL_color_type = RL_color_type1
    RL_shape_type = RL_shape_type1


#TCA_PPP_GLY with high/high no constraint
if (pathway_choice==7):
    Keq_constant = Keq_constant_tca_gly_ppp
    
    epr_to_plot_method1 = epr_method1_experimental_path3
    epr_to_plot_method2 = epr_method2_experimental_path3
    activity_to_plot_method1=activity_method1_experimental_path3
    activity_to_plot_method2=activity_method2_experimental_path3
    flux_to_plot_method1=flux_method1_experimental_path3
    flux_to_plot_method2=flux_method2_experimental_path3
    KQF_to_plot_method1 = KQF_method1_experimental_path3
    KQF_to_plot_method2 = KQF_method2_experimental_path3
    KQR_to_plot_method1 = KQR_method1_experimental_path3
    KQR_to_plot_method2 = KQR_method2_experimental_path3
    
    activity_to_plot_MCA = activity_to_plot_method1
    epr_to_plot_MCA = epr_to_plot_method1
    flux_to_plot_MCA = flux_to_plot_method1
    KQF_to_plot_MCA = KQF_to_plot_method1
    
    Pathway_Name='GLYCOLYSIS_TCA_PPP, High/High, no constraint'
    Pathway_Name_nospace='GLYCOLYSIS_TCA_PPP_high_high_noconstraint'
        
    p_string = cwd + p7_string
    bigg_reaction_abbr_p = bigg_reaction_abbr_p3
    eps=0.2

    MCA_method_type = MCA_method_type1
    MCA_color_type = MCA_color_type1
    MCA_shape_type = MCA_shape_type1
    
    RL_method_type = RL_method_type1
    RL_color_type = RL_color_type1
    RL_shape_type = RL_shape_type1

#TCA_PPP_GLY with high/low no constraint
if (pathway_choice==8):
    Keq_constant = Keq_constant_tca_gly_ppp
    
    epr_to_plot_method1 = epr_method1_experimental_path4
    epr_to_plot_method2 = epr_method2_experimental_path4
    activity_to_plot_method1=activity_method1_experimental_path4
    activity_to_plot_method2=activity_method2_experimental_path4
    flux_to_plot_method1=flux_method1_experimental_path4
    flux_to_plot_method2=flux_method2_experimental_path4
    KQF_to_plot_method1 = KQF_method1_experimental_path4
    KQF_to_plot_method2 = KQF_method2_experimental_path4
    KQR_to_plot_method1 = KQR_method1_experimental_path4
    KQR_to_plot_method2 = KQR_method2_experimental_path4
       
    activity_to_plot_MCA = activity_to_plot_method1
    epr_to_plot_MCA = epr_to_plot_method1
    flux_to_plot_MCA = flux_to_plot_method1
    KQF_to_plot_MCA = KQF_to_plot_method1
    Pathway_Name='GLYCOLYSIS_TCA_PPP, High/Low, no constraint'
    Pathway_Name_nospace='GLYCOLYSIS_TCA_PPP_high_low_noconstraint'
        
    p_string = cwd + p8_string
    
    bigg_reaction_abbr_p = bigg_reaction_abbr_p3
    eps=0.2
    
    MCA_method_type = MCA_method_type1
    MCA_color_type = MCA_color_type1
    MCA_shape_type = MCA_shape_type1
    
    RL_method_type = RL_method_type1
    RL_color_type = RL_color_type1
    RL_shape_type = RL_shape_type1

#TCA_PPP_GLY ZERO_PFK no constraint
if (pathway_choice==9):
    Keq_constant = Keq_constant_tca_gly_ppp
    
    epr_to_plot_method1 = epr_method1_experimental_path5
    epr_to_plot_method2 = epr_method2_experimental_path5
    activity_to_plot_method1=activity_method1_experimental_path5
    activity_to_plot_method2=activity_method2_experimental_path5
    flux_to_plot_method1=flux_method1_experimental_path5
    flux_to_plot_method2=flux_method2_experimental_path5
    KQF_to_plot_method1 = KQF_method1_experimental_path5
    KQF_to_plot_method2 = KQF_method2_experimental_path5
    
    KQR_to_plot_method1 = KQR_method1_experimental_path5
    KQR_to_plot_method2 = KQR_method2_experimental_path5
    
    activity_to_plot_MCA = activity_to_plot_method1
    epr_to_plot_MCA = epr_to_plot_method1
    flux_to_plot_MCA = flux_to_plot_method1
    KQF_to_plot_MCA = KQF_to_plot_method1
    
    Pathway_Name='GLYCOLYSIS_TCA_PPP ZERO_PFK, no constraint'
    
    Pathway_Name_nospace='GLYCOLYSIS_TCA_PPP_nopfk_high_high_noconstraint'
        
    p_string = cwd + p9_string
    
    bigg_reaction_abbr_p = bigg_reaction_abbr_p4
    eps=0.5
    begin_sim=1
    
    MCA_method_type = MCA_method_type1
    MCA_color_type = MCA_color_type1
    MCA_shape_type = MCA_shape_type1
    
    RL_method_type = RL_method_type1
    RL_color_type = RL_color_type1
    RL_shape_type = RL_shape_type1

def dedt(activity, KQF, KQR, KEQ):
    rate_f = activity*KQF
    rate_r = activity*KQR
    
    lhs = np.matmul(rate_f, np.log(KEQ))
    rhs = np.matmul(rate_r, np.log(KEQ))
    rxn_E_energy = -R_gas*Temperature*(lhs - rhs)
    return rxn_E_energy

def dgdt(activity, KQF, KQR, KEQ):
    rate_f = activity*KQF
    rate_r = activity*KQR
        
    #lhs_g = np.sum(rate_f* np.log(KQF),axis=1)
    #rhs_g = np.sum(rate_r* np.log(KQR),axis=1)
    lhs_g = np.sum(rate_f* np.log(KQF))
    rhs_g = np.sum(rate_r* np.log(KQF))
    
    rxn_G_energy = -R_gas*Temperature*(lhs_g - rhs_g)
    return rxn_G_energy

def dgdt_vec(activity, KQF, KQR, KEQ):
    rate_f = activity*KQF
    rate_r = activity*KQR
        
    lhs_g = np.sum(rate_f * np.log(KQF),axis=1)
    rhs_g = np.sum(rate_r * np.log(KQF),axis=1)
    
    rxn_G_energy = -R_gas*Temperature*(lhs_g - rhs_g)
    return rxn_G_energy

MCA_KQF1 = KQF_to_plot_method1
MCA_KQR1 = KQR_to_plot_method1   
MCA_activity1 = activity_to_plot_method1
MCA_epr1 = epr_to_plot_method1 * np.sum(flux_to_plot_method1)

MCA_dedt1 = dedt(MCA_activity1,MCA_KQF1,MCA_KQR1, Keq_constant)
MCA_dgdt1 = dgdt(MCA_activity1,MCA_KQF1,MCA_KQR1, Keq_constant)

MCA_KQF2 = KQF_to_plot_method2
MCA_KQR2 = KQR_to_plot_method2   
MCA_activity2 = activity_to_plot_method2
MCA_epr2 = epr_to_plot_method2 * np.sum(flux_to_plot_method2)
MCA_dedt2 = dedt(MCA_activity2,MCA_KQF2,MCA_KQR2, Keq_constant)
MCA_dgdt2 = dgdt(MCA_activity2,MCA_KQF2,MCA_KQR2, Keq_constant)


# =============================================================================
#             #dE/dt = -RT
#             ratef = temp_state[-length_of_back_count_epr:]*temp_KQF
#             lhs = np.matmul(ratef, np.log(Keq_constant))
#             
#             rater = temp_state[-length_of_back_count_epr:]*temp_KQR
#             rhs = np.matmul(rater, np.log(Keq_constant))
#             rxn_E_energy = -R_gas*Temperature*(lhs + rhs)
#             #rxn_energy = -np.matmul(rxn_flux ,np.log(Keq_constant) )
#             all_visited_energy_E = np.vstack((all_visited_energy_E,rxn_E_energy))
#             
# =============================================================================
            
            #dg/dt = -RT
            #ratef = temp_state[-length_of_back_count_epr:]*temp_KQF
            #lhs_g = np.sum(ratef* np.log(temp_KQF),axis=1)
            
            #rater = temp_state[-length_of_back_count_epr:]*temp_KQR
            #rhs_g = np.sum(rater* np.log(temp_KQR), axis=1)
            #rxn_G_energy = -R_gas*Temperature*(lhs_g + rhs_g)
            #all_visited_energy_G = np.vstack((all_visited_energy_G,rxn_G_energy))

#%% simple flux plot

colorChoice = sns.xkcd_rgb["black"]

color1 = sns.xkcd_rgb["slate grey"]
color2 = sns.xkcd_rgb["grey"]
color3 = sns.xkcd_rgb["steel grey"]

fig = plt.figure(figsize=(figure_norm, 0.5*figure_norm))
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

#method 1
sns.regplot(x=np.array([0]), y=np.array([flux_to_plot_method1[0]]), scatter=True, fit_reg=False, marker='x',
            ax=ax1,
            label='Method 1',
            color='r',
            scatter_kws={"s": 100})


for i in range(1,len(flux_to_plot_method1)):
    sns.regplot(x=np.array([i]), y=np.array([flux_to_plot_method1[i]]), scatter=True, fit_reg=False, marker='x',
                ax=ax1,
                color='r',
                scatter_kws={"s": 100})
    

#method 1
sns.regplot(x=np.array([0]), y=np.array([flux_to_plot_method2[0]]), scatter=True, fit_reg=False, marker='x',
            ax=ax2,
            label='Method 2',
            color='b',
            scatter_kws={"s": 100})


for i in range(1,len(flux_to_plot_method2)):
    sns.regplot(x=np.array([i]), y=np.array([flux_to_plot_method2[i]]), scatter=True, fit_reg=False, marker='x',
                ax=ax2,
                color='b',
                scatter_kws={"s": 100})


ax2.set_xticks( [i for i in range(0,len(flux_to_plot_method1)+1)] )
ax2.set_xticklabels(bigg_reaction_abbr_p, rotation='vertical', fontsize=Fontsize_Sub)

ax2.set_xlabel('Reactions',fontsize=Fontsize_Sub)
ax1.set_ylabel('Flux',fontsize=Fontsize_Sub)
ax2.set_ylabel('Flux',fontsize=Fontsize_Sub)

ax1.legend(fontsize=Fontsize_Leg, loc='best')
ax2.legend(fontsize=Fontsize_Leg, loc='best')

#%%


nvals=[2,4,6,8,10,12]
#nvals=[12]

p_len=345
path_lengths=[p_len,p_len,p_len,p_len,p_len,p_len]
num_sims = [11,11,11,11,11,11]
#num_sims = [21,21,21,21,21,21]
#num_sims = [4]

length_of_back_count=50

average_reward=[]
average_reward_total=[]
average_loss=[]

average_epr=[]
std_epr=[]

average_epr_normalized=[]
std_epr_normalized=[]

all_visited_epr_full=np.zeros(length_of_back_count)
all_visited_epr_relative_full=np.zeros(length_of_back_count)


reward_means=[]
reward_stds=[]
for id, lr in enumerate(lrs):
    reward_mean_n=[]
    reward_stds_n=[]
    for index_n,n in enumerate(nvals):
        
        path_length=path_lengths[index_n]    
        arr_reward=np.zeros(path_length)
        
        for i in range(begin_sim,num_sims[index_n]):
            tempr = np.loadtxt(p_string+'\\episodic_reward_gamma9_n'+str(n)+
                          '_k5__lr'+ lr +'_threshold25_eps'+str(eps)+'_penalty_reward_scalar_0.0'+
                          '_use_experimental_metab_'+str(use_exp_data)+'_sim'
                          +str(i)+'.txt',dtype=float)

            arr_reward = np.vstack((arr_reward,tempr[0:path_length]))
         

        sim_rew_set = np.mean(arr_reward[1:,-length_of_back_count:], axis=1)
        reward_mean_n.append(np.mean(sim_rew_set))
        reward_stds_n.append(np.std(sim_rew_set))
    reward_means.append(reward_mean_n)
    reward_stds.append(reward_stds_n)

lr_index=math.floor(np.argmax(reward_means) / len(nvals))
lr_choice = lrs[lr_index]
n_index = math.floor(np.argmax(reward_means) % len(nvals))
n_choice = nvals[n_index ]

#######################################################################################
max_reward_index = np.argmax(reward_means[lr_index])

#%% instead take max

using_one_sim = True
if (using_one_sim):
    nvals=[2,4,6,8,10,12]
    #nvals=[12]
    
    p_len=350
    path_lengths=[p_len,p_len,p_len,p_len,p_len,p_len]
    num_sims = [11,11,11,11,11,11]
    #num_sims = [21,21,21,21,21,21]
    
    average_reward=[]
    average_reward_total=[]
    average_loss=[]
    
    average_epr=[]
    std_epr=[]
    
    average_epr_normalized=[]
    std_epr_normalized=[]
    
    all_visited_epr_full=np.zeros(length_of_back_count)
    all_visited_epr_relative_full=np.zeros(length_of_back_count)
    
    current_max_reward=0.0
    
    for id, lr in enumerate(lrs):
        reward_mean_n=[]
        reward_stds_n=[]
        for index_n,n in enumerate(nvals):
            
            path_length=path_lengths[index_n]    
            arr_reward=np.zeros(path_length)
            
            for i in range(begin_sim,num_sims[index_n]):
                tempr = np.loadtxt(p_string+'\\episodic_reward_gamma9_n'+str(n)+
                              '_k5__lr'+ lr +'_threshold25_eps'+str(eps)+'_penalty_reward_scalar_0.0'+
                              '_use_experimental_metab_'+str(use_exp_data)+'_sim'
                              +str(i)+'.txt',dtype=float)
    
                tempr = tempr[0:path_length]
                
                [reward_unique, reward_indices_unique, reward_counts_unique] = np.unique(tempr[-length_of_back_count:], return_index=True,return_counts=True)
                #current_reward = reward_unique[np.argmax(reward_counts_unique)]
                
                current_reward = np.mean(tempr[-length_of_back_count:])
                
                temp = np.loadtxt(p_string+'\\episodic_loss_gamma9_n'+str(n)+
                                  '_k5__lr'+ lr_choice +'_threshold25_eps'+str(eps)+'_penalty_reward_scalar_0.0'+
                                  '_use_experimental_metab_'+str(use_exp_data)+'_sim'
                                  +str(i)+'.txt',dtype=float)
                temp = temp[0:path_length]

                #print(current_reward)
#                if (i==5 and n==8): 
                #fig = plt.figure()
                #ax1 = fig.add_subplot(1,1,1)
                #ax1.plot(tempr)
                #fig = plt.figure()
                #ax1 = fig.add_subplot(1,1,1)
                #ax1.plot(temp)                

                if (current_reward > current_max_reward):

                    print('new best')
                    print(current_reward)
                    print(i)
                    print(n)
                    print(lr)
                    current_max_reward=current_reward
                    lr_index = id
                    lr_choice=lr
                    n_index=index_n
                    n_choice=n
                    simulation_choice = i
                    fig = plt.figure()
                    ax1 = fig.add_subplot(1,1,1)
                    ax1.plot(tempr,color='r')



#%%
colorChoice = sns.xkcd_rgb["black"]

color1 = sns.xkcd_rgb["slate grey"]
color2 = sns.xkcd_rgb["grey"]
color3 = sns.xkcd_rgb["steel grey"]

fig = plt.figure(figsize=(figure_norm, 0.5*figure_norm))
ax1 = fig.add_subplot(1,1,1)
    




ax1.errorbar(nvals, reward_means[0], yerr=reward_stds[0],
             linestyle=':',color = color1, capsize=5,
             label='lr=1e-4')
ax1.errorbar(nvals, reward_means[1], yerr=reward_stds[1],
             linestyle='--',color = color2, capsize=5,
             label='lr=1e-5')
ax1.errorbar(nvals, reward_means[2], yerr=reward_stds[2],
             linestyle='-',color = color3, capsize=5,
             label='lr=1e-6')

fig.suptitle(Pathway_Name + "\nBest learning rate: " +lr
             + "\n Best n: " + str(n_choice),fontsize=Fontsize_Title, y=1.1)

ax1.set_xlabel("n",fontsize=Fontsize_Sub)
ax1.set_ylabel("Averaged Reward per Episode",fontsize=Fontsize_Sub)


ax1.legend(fontsize=Fontsize_Leg, loc='best')

#%% now lr is determined, take best n and use data

for index_n,n in enumerate(nvals):
    
    path_length=path_lengths[index_n]
    arr_loss=np.zeros(path_length)
    arr_reward=np.zeros(path_length)
    
    arr_epr = np.zeros(length_of_back_count)
    arr_epr_glucose = np.zeros(length_of_back_count)
    
    max_flux=np.zeros(length_of_back_count)
    
    if n == n_choice:
        for i in range(begin_sim,num_sims[index_n]):
            
            #if (using_one_sim):
            #    i = simulation_choice
            print(i)
            #episodic_loss_gamma9_n2_k5__lr1e-08_threshold25_eps0.5_penalty_reward_scalar_0.0_use_experimental_metab_0_sim7
            temp = np.loadtxt(p_string+'\\episodic_loss_gamma9_n'+str(n)+
                              '_k5__lr'+ lr_choice +'_threshold25_eps'+str(eps)+'_penalty_reward_scalar_0.0'+
                              '_use_experimental_metab_'+str(use_exp_data)+'_sim'
                              +str(i)+'.txt',dtype=float)
    
            
            #temp = np.sqrt(temp)
            temp[0:path_length] = temp[0:path_length] / max(temp[0:path_length])
            
            arr_loss = np.vstack((arr_loss,temp[0:path_length]))
            
            tempr = np.loadtxt(p_string+'\\episodic_reward_gamma9_n'+str(n)+
                              '_k5__lr'+ lr_choice +'_threshold25_eps'+str(eps)+'_penalty_reward_scalar_0.0'+
                              '_use_experimental_metab_'+str(use_exp_data)+'_sim'
                              +str(i)+'.txt',dtype=float)
    
            arr_reward = np.vstack((arr_reward,tempr[0:path_length]))
            
            temp_epr = np.loadtxt(p_string+'\\epr_per_state_gamma9_n'+str(n)+
                              '_k5__lr'+ lr_choice +'_threshold25_eps'+str(eps)+'_penalty_reward_scalar_0.0'+
                              '_use_experimental_metab_'+str(use_exp_data)+'_sim'
                              +str(i)+'.txt',dtype=float)
            
            temp_epr=temp_epr[0:path_length]
        
            all_visited_epr_full = np.vstack((all_visited_epr_full,temp_epr[-length_of_back_count:]))
            
            temp_state = np.loadtxt(p_string+'\\final_states_gamma9_n'+str(n)+
                          '_k5__lr'+ lr_choice +'_threshold25_eps'+str(eps)+'_penalty_reward_scalar_0.0'+
                          '_use_experimental_metab_'+str(use_exp_data)+'_sim'
                          +str(i)+'.txt',dtype=float)
            temp_state=temp_state[0:path_length+1]
            temp_state=temp_state[1:]
    
            temp_state=temp_state[-length_of_back_count:]
            
            temp_KQF = np.loadtxt(p_string+'\\final_KQF_gamma9_n'+str(n)+
                          '_k5__lr'+ lr_choice +'_threshold25_eps'+str(eps)+'_penalty_reward_scalar_0.0'+
                          '_use_experimental_metab_'+str(use_exp_data)+'_sim'
                          +str(i)+'.txt',dtype=float)
            temp_KQF=temp_KQF[0:path_length]
            temp_KQF=temp_KQF[-length_of_back_count:]
                    
            temp_KQR = np.loadtxt(p_string+'\\final_KQR_gamma9_n'+str(n)+
                          '_k5__lr'+ lr_choice +'_threshold25_eps'+str(eps)+'_penalty_reward_scalar_0.0'+
                          '_use_experimental_metab_'+str(use_exp_data)+'_sim'
                          +str(i)+'.txt',dtype=float)
            temp_KQR=temp_KQR[0:path_length]
            temp_KQR=temp_KQR[-length_of_back_count:]
            
            arr_epr = np.vstack((arr_epr, temp_epr[-length_of_back_count:]))
        
        
            #EPR/glucose = -RT*(1/rxn_flux(15)) * sum(rxn_flux.*log(KQ)) is supposed to calculated here
            rxn_flux = temp_state*(temp_KQF - temp_KQR)
            
            mrf = np.max(rxn_flux, axis=1)
            
            max_flux = np.vstack((max_flux,mrf))
            
        #fig = plt.figure(figsize=(figure_norm, figure_norm))
        #ax1 = fig.add_subplot(2,1,1)
        #ax2 = fig.add_subplot(2,1,2)
        #ax1.plot(np.mean(max_flux[1:,:], axis=0))
        #ax2.plot(np.mean(arr_epr[1:,:], axis=0))
        #ax1.set_xlabel("Episode",fontsize=Fontsize_Sub)
        #ax1.set_ylabel("maximum flux",fontsize=Fontsize_Sub)
        #ax2.set_xlabel("Episode",fontsize=Fontsize_Sub)
        #ax2.set_ylabel("EPR",fontsize=Fontsize_Sub)
        #for i in range(1,num_sims[index_n]):
        #    temp_arr_loss=arr_loss[i,:]
        #    ax1.plot(temp_arr_loss, label=str(i))
        #ax1.legend()
            
        #ax1 = fig.add_subplot(1,1,1)
        
        #for i in range(1,num_sims[index_n]):
        #    temp_arr_rew=arr_reward[i,:]
        #    ax1.plot(temp_arr_rew, label=str(i))
        #ax1.legend()
        
        
        average_loss = np.mean(arr_loss[1:,:],axis=0)
        
        average_reward.append(np.mean(arr_reward[1:,-length_of_back_count:],axis=0))  
        average_reward_total = np.mean(arr_reward[1:,:],axis=0)
        average_epr.append(np.mean(arr_epr[1:,:]))  
        std_epr.append(np.std(arr_epr[1:,:]))  
    
    
        
        
        


#%%
fig = plt.figure(figsize=(figure_norm, figure_norm))

ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

loss_to_plot = np.sqrt(average_loss)
ax1.plot(loss_to_plot )#, label='n='+str(nvals[max_reward_index]))
    

reward_to_plot = average_reward_total

ax2.plot(reward_to_plot,
             linestyle='-')

fig.suptitle(Pathway_Name,fontsize=Fontsize_Title)

ax1.annotate("A", xy=(-0.1, 1.0), xycoords="axes fraction",fontsize=Fontsize_Sub)
ax2.annotate("B", xy=(-0.1, 1.0), xycoords="axes fraction",fontsize=Fontsize_Sub)

ax1.set_xlabel("Episode",fontsize=Fontsize_Sub)
ax1.set_ylabel("RMSE Error",fontsize=Fontsize_Sub)
ax2.set_xlabel("Episode",fontsize=Fontsize_Sub)
ax2.set_ylabel("Averaged Normalized \nEpisodic Reward",fontsize=Fontsize_Sub)
#ax2.set_xticks(nvals)


#%% Just plot rewards
fig = plt.figure(figsize=(figure_norm, 0.5*figure_norm))
ax1 = fig.add_subplot(1,1,1)
   
max_reward = np.max(reward_to_plot)
min_reward = np.min(reward_to_plot)
ax1.plot( (reward_to_plot - min_reward) / (max_reward - min_reward),# min_reward,
             linestyle='-')

fig.suptitle(Pathway_Name,fontsize=Fontsize_Title)

ax1.set_xlabel("Episode",fontsize=Fontsize_Sub)
ax1.set_ylabel("Averaged Normalized \nEpisodic Reward",fontsize=Fontsize_Sub)
#ax2.set_xticks(nvals)

ax1.legend(fontsize=Fontsize_Leg, loc='upper right')



#%% save rewards 
reward_to_save = (reward_to_plot - min_reward) / (max_reward - min_reward)

if (pathway_choice == 1):
    np.savetxt(cwd + p1_string_short +'\\reward_experiment_RL_noconstraint.txt', reward_to_save, fmt='%1.30f')
    
if (pathway_choice == 6):
    np.savetxt(cwd + p2_string_short +'\\reward_experiment_RL_noconstraint.txt', reward_to_save, fmt='%1.30f')

if (pathway_choice == 7):
    np.savetxt(cwd + p3_string_short +'\\reward_experiment_RL_noconstraint_highhigh.txt', reward_to_save, fmt='%1.30f')

if (pathway_choice == 8):
    np.savetxt(cwd + p4_string_short +'\\reward_experiment_RL_noconstraint_highlow.txt', reward_to_save, fmt='%1.30f')
    
if (pathway_choice == 9):
    np.savetxt(cwd + p3_string_short +'\\reward_experiment_RL_noconstraint_highhigh_PFKZERO.txt', reward_to_save, fmt='%1.30f')
    

#%% plot best states


most_visited_state=[]
most_visited_states=[]
most_visited_flux=[]
most_visited_KQF=[]
all_visited_eprs=[]
most_visited_epr=[]
most_visited_norm_epr=[]
most_visited_energy=[]

for index_n,n in enumerate(nvals):
    if n == n_choice:
        path_length=path_lengths[index_n]
        most_visited_states=np.zeros((0,Keq_constant.shape[0]))
        most_visited_flux=np.zeros((0,Keq_constant.shape[0]))
        most_visited_KQF=np.zeros((0,Keq_constant.shape[0]))
        most_visited_KQR=np.zeros((0,Keq_constant.shape[0]))
            
        arr_epr = np.zeros((0,length_of_back_count))
        arr_norm_epr = np.zeros((0,length_of_back_count))
        arr_epr_glucose = np.zeros((0,length_of_back_count))
        
        max_flux=np.zeros(length_of_back_count)
        
        for i in range(begin_sim,num_sims[index_n]):
            if (using_one_sim):
                i = simulation_choice
            print(i)
            temp_state = np.loadtxt(p_string+'\\final_states_gamma9_n'+str(n)+
                          '_k5__lr'+lr_choice+'_threshold25_eps'+str(eps)+'_penalty_reward_scalar_0.0'+
                          '_use_experimental_metab_'+str(use_exp_data)+'_sim'
                          +str(i)+'.txt',dtype=float)
            
            temp_state=temp_state[1:path_length+1]
            
            #fig = plt.figure(figsize=(figure_norm, 0.5*figure_norm))
            #ax1 = fig.add_subplot(1,1,1)
            #ax1.plot(temp_state[-length_of_back_count:].T)
            

            
            most_visited_states = np.vstack((most_visited_states,temp_state[-length_of_back_count:]))
            
            temp_epr = np.loadtxt(p_string+'\\epr_per_state_gamma9_n'+str(n)+
                          '_k5__lr'+lr_choice+'_threshold25_eps'+str(eps)+'_penalty_reward_scalar_0.0'+
                          '_use_experimental_metab_'+str(use_exp_data)+'_sim'
                          +str(i)+'.txt',dtype=float)
            temp_epr=temp_epr[0:path_length]
    
            
            
            temp_state=temp_state[-length_of_back_count:]
            
            temp_KQF = np.loadtxt(p_string+'\\final_KQF_gamma9_n'+str(n)+
                          '_k5__lr'+ lr_choice +'_threshold25_eps'+str(eps)+'_penalty_reward_scalar_0.0'+
                          '_use_experimental_metab_'+str(use_exp_data)+'_sim'
                          +str(i)+'.txt',dtype=float)
            temp_KQF=temp_KQF[1:path_length+1]
            temp_KQF=temp_KQF[-length_of_back_count:]
                    
            temp_KQR = np.loadtxt(p_string+'\\final_KQR_gamma9_n'+str(n)+
                          '_k5__lr'+ lr_choice +'_threshold25_eps'+str(eps)+'_penalty_reward_scalar_0.0'+
                          '_use_experimental_metab_'+str(use_exp_data)+'_sim'
                          +str(i)+'.txt',dtype=float)
            temp_KQR=temp_KQR[1:path_length+1]
            temp_KQR=temp_KQR[-length_of_back_count:]
            
            rxn_flux = temp_state*(temp_KQF - temp_KQR)
            
            arr_epr = np.vstack((arr_epr, np.sum(rxn_flux, axis=1)*temp_epr[-length_of_back_count:]))
            arr_norm_epr = np.vstack((arr_norm_epr, temp_epr[-length_of_back_count:]))


            most_visited_flux = np.vstack((most_visited_flux,rxn_flux))
            most_visited_KQF = np.vstack((most_visited_KQF,temp_KQF))
            most_visited_KQR = np.vstack((most_visited_KQR,temp_KQR))


        [states_unique, indices_unique, counts_unique]=np.unique(most_visited_states[:,:], axis=0, return_index=True,return_counts=True)
        correct_row_index=indices_unique[np.argmax(counts_unique)]
        
        #[flux_unique, flux_indices_unique, flux_counts_unique]=np.unique(most_visited_flux[:,:], axis=0, return_index=True,return_counts=True)
        #[KQF_unique, KQF_indices_unique, KQF_counts_unique]=np.unique(most_visited_KQF[:,:], axis=0, return_index=True,return_counts=True)
        
        most_visited_state = states_unique[np.argmax(counts_unique),:]
        most_visited_state_alt = most_visited_states[correct_row_index,:]
        most_visited_flux = most_visited_flux[correct_row_index,:]
        most_visited_KQF = most_visited_KQF[correct_row_index,:]
        most_visited_KQR = most_visited_KQR[correct_row_index,:]
        most_visited_epr_vec=arr_epr.flatten()
        most_visited_epr = most_visited_epr_vec[correct_row_index] #do not redo these, they must be chosen from the last 50 states
        
        most_visited_norm_epr_vec = arr_norm_epr.flatten()
        most_visited_norm_epr = most_visited_norm_epr_vec[correct_row_index]

        most_visited_energy = -np.sum(most_visited_flux * np.log(Keq_constant) )

        most_visited_dedt = dedt(most_visited_state,most_visited_KQF,most_visited_KQR,Keq_constant )
        most_visited_dgdt = dgdt(most_visited_state,most_visited_KQF,most_visited_KQR,Keq_constant )
#states_unique = most_visited_states[:,:]

flux_method_RL = most_visited_flux

final_state_data = pd.DataFrame(columns=['Method','Activity','Reaction'])

counter=0
for i in range(0,(temp_state.shape[1])):
    final_state_data.loc[i] = ['RL', most_visited_state[i],i]
    
#save data for pathways 2,3,4
#%%

if (pathway_choice == 1):
    np.savetxt(cwd + p1_string_short +'\\activities_experiment_RL_noconstraint.txt', most_visited_state, fmt='%1.30f')
    np.savetxt(cwd + p1_string_short +'\\flux_experiment_RL_noconstraint.txt', most_visited_flux, fmt='%1.30f')
    np.savetxt(cwd + p1_string_short +'\\KQF_experiment_RL_noconstraint.txt', most_visited_KQF, fmt='%1.30f')
    np.savetxt(cwd + p1_string_short +'\\KQR_experiment_RL_noconstraint.txt', most_visited_KQR, fmt='%1.30f')
    np.savetxt(cwd + p1_string_short +'\\norm_EPR_experiment_RL_noconstraint.txt', np.array([most_visited_norm_epr]), fmt='%1.30f')
    
if (pathway_choice == 2):
    np.savetxt(cwd + p2_string_short +'\\activities_experiment_RL.txt', most_visited_state, fmt='%1.30f')
    np.savetxt(cwd + p2_string_short +'\\flux_experiment_RL.txt', most_visited_flux, fmt='%1.30f')
    np.savetxt(cwd + p2_string_short +'\\KQF_experiment_RL.txt', most_visited_KQF, fmt='%1.30f')
    np.savetxt(cwd + p2_string_short +'\\KQR_experiment_RL.txt', most_visited_KQR, fmt='%1.30f')
    np.savetxt(cwd + p2_string_short +'\\norm_EPR_experiment_RL.txt', np.array([most_visited_norm_epr]), fmt='%1.30f')
    
if (pathway_choice == 3):
    np.savetxt(cwd + p3_string_short +'\\activities_experiment_RL_highhigh.txt', most_visited_state, fmt='%1.30f')
    np.savetxt(cwd + p3_string_short +'\\flux_experiment_RL_highhigh.txt', most_visited_flux, fmt='%1.30f')
    np.savetxt(cwd + p3_string_short +'\\KQF_experiment_RL_highhigh.txt', most_visited_KQF, fmt='%1.30f')
    np.savetxt(cwd + p3_string_short +'\\norm_EPR_experiment_RL_highhigh.txt', np.array([most_visited_norm_epr]), fmt='%1.30f')
    
if (pathway_choice == 4):
    np.savetxt(cwd + p4_string_short +'\\activities_experiment_RL_highlow.txt', most_visited_state, fmt='%1.30f')
    np.savetxt(cwd + p4_string_short +'\\flux_experiment_RL_highlow.txt', most_visited_flux, fmt='%1.30f')
    np.savetxt(cwd + p4_string_short +'\\KQF_experiment_RL_highlow.txt', most_visited_KQF, fmt='%1.30f')
    np.savetxt(cwd + p4_string_short +'\\norm_EPR_experiment_RL_highlow.txt', np.array([most_visited_norm_epr]), fmt='%1.30f')
    
if (pathway_choice == 5):
    np.savetxt(cwd + p3_string_short +'\\activities_experiment_RL_highhigh_PFKZERO.txt', most_visited_state, fmt='%1.30f')
    np.savetxt(cwd + p3_string_short +'\\flux_experiment_RL_highhigh_PFKZERO.txt', most_visited_flux, fmt='%1.30f')
    np.savetxt(cwd + p3_string_short +'\\KQF_experiment_RL_highhigh_PFKZERO.txt', most_visited_KQF, fmt='%1.30f')    
    np.savetxt(cwd + p3_string_short +'\\norm_EPR_experiment_RL_highhigh_PFKZERO.txt', np.array([most_visited_norm_epr]), fmt='%1.30f')    
    
if (pathway_choice == 6):
    np.savetxt(cwd + p2_string_short +'\\activities_experiment_RL_noconstraint.txt', most_visited_state, fmt='%1.30f')
    np.savetxt(cwd + p2_string_short +'\\flux_experiment_RL_noconstraint.txt', most_visited_flux, fmt='%1.30f')
    np.savetxt(cwd + p2_string_short +'\\KQF_experiment_RL_noconstraint.txt', most_visited_KQF, fmt='%1.30f')
    np.savetxt(cwd + p2_string_short +'\\KQR_experiment_RL_noconstraint.txt', most_visited_KQR, fmt='%1.30f')
    np.savetxt(cwd + p2_string_short +'\\norm_EPR_experiment_RL_noconstraint.txt', np.array([most_visited_norm_epr]), fmt='%1.30f')
    
if (pathway_choice == 7):
    np.savetxt(cwd + p3_string_short +'\\activities_experiment_RL_noconstraint_highhigh.txt', most_visited_state, fmt='%1.30f')
    np.savetxt(cwd + p3_string_short +'\\flux_experiment_RL_noconstraint_highhigh.txt', most_visited_flux, fmt='%1.30f')
    np.savetxt(cwd + p3_string_short +'\\KQF_experiment_RL_noconstraint_highhigh.txt', most_visited_KQF, fmt='%1.30f')
    np.savetxt(cwd + p3_string_short +'\\KQR_experiment_RL_noconstraint_highhigh.txt', most_visited_KQR, fmt='%1.30f')
    np.savetxt(cwd + p3_string_short +'\\norm_EPR_experiment_RL_noconstraint_highhigh.txt', np.array([most_visited_norm_epr]), fmt='%1.30f')
    
if (pathway_choice == 8):
    np.savetxt(cwd + p4_string_short +'\\activities_experiment_RL_noconstraint_highlow.txt', most_visited_state, fmt='%1.30f')
    np.savetxt(cwd + p4_string_short +'\\flux_experiment_RL_noconstraint_highlow.txt', most_visited_flux, fmt='%1.30f')
    np.savetxt(cwd + p4_string_short +'\\KQF_experiment_RL_noconstraint_highlow.txt', most_visited_KQF, fmt='%1.30f')
    np.savetxt(cwd + p4_string_short +'\\KQR_experiment_RL_noconstraint_highlow.txt', most_visited_KQR, fmt='%1.30f')
    np.savetxt(cwd + p4_string_short +'\\norm_EPR_experiment_RL_noconstraint_highlow.txt', np.array([most_visited_norm_epr]), fmt='%1.30f')
    
    
if (pathway_choice == 9):
    np.savetxt(cwd + p3_string_short +'\\activities_experiment_RL_noconstraint_highhigh_PFKZERO.txt', most_visited_state, fmt='%1.30f')
    np.savetxt(cwd + p3_string_short +'\\flux_experiment_RL_noconstraint_highhigh_PFKZERO.txt', most_visited_flux, fmt='%1.30f')
    np.savetxt(cwd + p3_string_short +'\\KQF_experiment_RL_noconstraint_highhigh_PFKZERO.txt', most_visited_KQF, fmt='%1.30f')
    np.savetxt(cwd + p3_string_short +'\\KQR_experiment_RL_noconstraint_highhigh_PFKZERO.txt', most_visited_KQR, fmt='%1.30f')
    np.savetxt(cwd + p3_string_short +'\\norm_EPR_experiment_RL_noconstraint_highhigh_PFKZERO.txt', np.array([most_visited_norm_epr]), fmt='%1.30f')


#%%%

fig = plt.figure(figsize=(figure_norm, 0.25*figure_norm))
ax1 = fig.add_subplot(111)

ax1.set_title('Comparison: MCA Method 2 vs RL', fontsize=Fontsize_Title)
sns.boxplot(x='Reaction',
            y='Activity',
            data=final_state_data,
            hue='Method',
            palette=sns.light_palette((210, 90, 60), input="husl"),
            ax=ax1)

#To make legend, plot first with label for legend
sns.regplot(x=np.array([0]), y=np.array([activity_to_plot_method2[0]]), scatter=True, fit_reg=False, marker='x',
            ax=ax1,
            label='MCA: Method 2',
            color='r',
            scatter_kws={"s": 100})


for i in range(1,temp_state.shape[1]):
    sns.regplot(x=np.array([i]), y=np.array([activity_to_plot_method2[i]]), scatter=True, fit_reg=False, marker='x',
                ax=ax1,
                color='r',
                scatter_kws={"s": 100})


# =============================================================================
# sns.regplot(x=np.array([epr_method_RL]), y=np.array([1]), scatter=True, fit_reg=False, marker='o',
#                 ax=ax2,
#                 color='r',
#                scatter_kws={"s": 100})
# =============================================================================

ax1.set_xlabel('Reactions',fontsize=Fontsize_Sub)

ax1.set_xticklabels(bigg_reaction_abbr_p, rotation='vertical', fontsize=Fontsize_Sub)
ax1.set_ylabel('Enzyme Activity',fontsize=Fontsize_Sub)

ax1.legend(fontsize=Fontsize_Leg, loc='best')
#ax1.annotate("C", xy=(-0.1, 1.0), xycoords="axes fraction",fontsize=Fontsize_Sub)

#fig.tight_layout()  # otherwise the right y-label is slightly clipped

#%%

all_visited_states=[]
all_visited_fluxes=[]
all_visited_KQFs=[]

all_visited_epr=[]

length_of_back_count_epr=345

all_visited_epr = np.zeros((0,length_of_back_count_epr))
for index_n,n in enumerate(nvals):
    if n == n_choice:
        path_length=path_lengths[index_n]
        all_visited_states=np.zeros((0,temp_state.shape[1]))
        all_visited_fluxes=np.zeros((0,temp_state.shape[1]))
        all_visited_KQFs=np.zeros((0,temp_state.shape[1]))
        all_visited_energy_E=np.zeros((0,length_of_back_count_epr))
        all_visited_energy_G=np.zeros((0,length_of_back_count_epr))
            
        for i in range(begin_sim,num_sims[index_n]):
            if (using_one_sim):
                i = simulation_choice
            print(i)
            temp_state = np.loadtxt(p_string+'\\final_states_gamma9_n'+str(n)+
                          '_k5__lr'+lr_choice+'_threshold25_eps'+str(eps)+'_penalty_reward_scalar_0.0'+
                          '_use_experimental_metab_'+str(use_exp_data)+'_sim'
                          +str(i)+'.txt',dtype=float)
            temp_state=temp_state[1:path_length+1,:]
            
            all_visited_states = np.vstack((all_visited_states,temp_state[-length_of_back_count_epr:]))
            
            temp_epr = np.loadtxt(p_string+'\\epr_per_state_gamma9_n'+str(n)+
                          '_k5__lr'+lr_choice+'_threshold25_eps'+str(eps)+'_penalty_reward_scalar_0.0'+
                          '_use_experimental_metab_'+str(use_exp_data)+'_sim'
                          +str(i)+'.txt',dtype=float)
            temp_epr=temp_epr[0:path_length]
    
            
            
            temp_KQF = np.loadtxt(p_string+'\\final_KQF_gamma9_n'+str(n)+
                          '_k5__lr'+ lr_choice +'_threshold25_eps'+str(eps)+'_penalty_reward_scalar_0.0'+
                          '_use_experimental_metab_'+str(use_exp_data)+'_sim'
                          +str(i)+'.txt',dtype=float)
            temp_KQF=temp_KQF[1:path_length+1]
            temp_KQF=temp_KQF[-length_of_back_count_epr:]
                    
            temp_KQR = np.loadtxt(p_string+'\\final_KQR_gamma9_n'+str(n)+
                          '_k5__lr'+ lr_choice +'_threshold25_eps'+str(eps)+'_penalty_reward_scalar_0.0'+
                          '_use_experimental_metab_'+str(use_exp_data)+'_sim'
                          +str(i)+'.txt',dtype=float)
            temp_KQR=temp_KQR[1:path_length+1]
            temp_KQR=temp_KQR[-length_of_back_count_epr:]
            
            rxn_flux = temp_state[-length_of_back_count_epr:]*(temp_KQF - temp_KQR)
            
            all_visited_epr = np.vstack((all_visited_epr, np.sum(rxn_flux, axis=1) * temp_epr[-length_of_back_count_epr:]))

            all_visited_fluxes = np.vstack((all_visited_fluxes,rxn_flux))
            all_visited_KQFs = np.vstack((all_visited_KQFs,temp_KQF))
            
            #dE/dt = -RT
            ratef = temp_state[-length_of_back_count_epr:]*temp_KQF
            lhs = np.matmul(ratef, np.log(Keq_constant))
            
            rater = temp_state[-length_of_back_count_epr:]*temp_KQR
            rhs = np.matmul(rater, np.log(Keq_constant))
            rxn_E_energy = dedt(temp_state[-length_of_back_count_epr:],temp_KQF,temp_KQR,Keq_constant)
            #-R_gas*Temperature*(lhs - rhs)
            #rxn_energy = -np.matmul(rxn_flux ,np.log(Keq_constant) )
            all_visited_energy_E = np.vstack((all_visited_energy_E,rxn_E_energy))
            
            
            #dg/dt = -RT
            ratef = temp_state[-length_of_back_count_epr:]*temp_KQF
            lhs_g = np.sum(ratef* np.log(temp_KQF),axis=1)
            
            rater = temp_state[-length_of_back_count_epr:]*temp_KQR
            rhs_g = np.sum(rater* np.log(temp_KQR), axis=1)
            
            rxn_G_energy = dgdt_vec(temp_state[-length_of_back_count_epr:],temp_KQF,temp_KQR,Keq_constant)
            #-R_gas*Temperature*(lhs_g - rhs_g)
            all_visited_energy_G = np.vstack((all_visited_energy_G,rxn_G_energy))
            
            if (using_one_sim):
                break
        
        [states_unique, indices_unique, counts_unique]=np.unique(all_visited_states[:,:], axis=0, return_index=True,return_counts=True)
        correct_row_index=indices_unique[np.argmax(counts_unique)]
        
        all_visited_state = states_unique[np.argmax(counts_unique),:]
        all_visited_state_alt = all_visited_states[correct_row_index,:]
        all_visited_flux = all_visited_fluxes[correct_row_index,:]
        all_visited_KQF = all_visited_KQFs[correct_row_index,:]
        all_visited_epr_vec=all_visited_epr.flatten()

#states_unique = all_visited_states[:,:]


all_visited_norm_epr = all_visited_epr_vec / np.sum(all_visited_fluxes, axis=1)

all_visited_energy_E_total = all_visited_energy_E.flatten()
all_visited_energy_G_total = all_visited_energy_G.flatten()

temp_table_info = np.vstack((all_visited_epr_vec, all_visited_norm_epr, all_visited_energy_E_total  ))
temp_table_info_final_choice = np.vstack((most_visited_epr, most_visited_norm_epr, most_visited_energy  ))

#%%
if (pathway_choice == 1):
    np.savetxt(cwd + p1_string_short +'\\de_dt_RL_noconstraint.txt', all_visited_energy_E_total, fmt='%1.30f')
    np.savetxt(cwd + p1_string_short +'\\dg_dt_RL_noconstraint.txt', all_visited_energy_G_total, fmt='%1.30f')
    
if (pathway_choice == 2):
    np.savetxt(cwd + p2_string_short +'\\de_dt_RL_constraint.txt', all_visited_energy_E_total, fmt='%1.30f')
    np.savetxt(cwd + p2_string_short +'\\dg_dt_RL_constraint.txt', all_visited_energy_G_total, fmt='%1.30f')
    
if (pathway_choice == 3):
    np.savetxt(cwd + p3_string_short +'\\de_dt_RL_constraint_highhigh.txt', all_visited_energy_E_total, fmt='%1.30f')
    np.savetxt(cwd + p3_string_short +'\\dg_dt_RL_constraint_highhigh.txt', all_visited_energy_G_total, fmt='%1.30f')
    
if (pathway_choice == 4):
    np.savetxt(cwd + p4_string_short +'\\de_dt_RL_constraint_highlow.txt', all_visited_energy_E_total, fmt='%1.30f')
    np.savetxt(cwd + p4_string_short +'\\dg_dt_RL_constraint_highlow.txt', all_visited_energy_G_total, fmt='%1.30f')
    
if (pathway_choice == 5):
    np.savetxt(cwd + p3_string_short +'\\de_dt_RL_constraint_highhigh_PFKZERO.txt', all_visited_energy_E_total, fmt='%1.30f')
    np.savetxt(cwd + p3_string_short +'\\dg_dt_RL_constraint_highhigh_PFKZERO.txt', all_visited_energy_G_total, fmt='%1.30f')
    
if (pathway_choice == 6):
    np.savetxt(cwd + p2_string_short +'\\de_dt_RL_noconstraint.txt', all_visited_energy_E_total, fmt='%1.30f')
    np.savetxt(cwd + p2_string_short +'\\dg_dt_RL_noconstraint.txt', all_visited_energy_G_total, fmt='%1.30f')
    
if (pathway_choice == 7):
    np.savetxt(cwd + p3_string_short +'\\de_dt_RL_noconstraint_highhigh.txt', all_visited_energy_E_total, fmt='%1.30f')
    np.savetxt(cwd + p3_string_short +'\\dg_dt_RL_noconstraint_highhigh.txt', all_visited_energy_G_total, fmt='%1.30f')
    
if (pathway_choice == 8):
    np.savetxt(cwd + p4_string_short +'\\de_dt_RL_noconstraint_highlow.txt', all_visited_energy_E_total, fmt='%1.30f')
    np.savetxt(cwd + p4_string_short +'\\dg_dt_RL_noconstraint_highlow.txt', all_visited_energy_G_total, fmt='%1.30f')
    
    
if (pathway_choice == 9):
    np.savetxt(cwd + p3_string_short +'\\de_dt_RL_noconstraint_highhigh_PFKZERO.txt', all_visited_energy_E_total, fmt='%1.30f')
    np.savetxt(cwd + p3_string_short +'\\dg_dt_RL_noconstraint_highhigh_PFKZERO.txt', all_visited_energy_G_total, fmt='%1.30f')


#%%

import pandas as pd

## convert your array into a dataframe
df_data = pd.DataFrame (temp_table_info.T)
df_data_final = pd.DataFrame (temp_table_info_final_choice.T)

filepath_data = cwd+'\\epr_nepr_dedt'+Pathway_Name_nospace + '.xlsx'
filepath_data_final = cwd+'\\final_choice_epr_nepr_dedt'+Pathway_Name_nospace + '.xlsx'



df_data.to_excel(filepath_data, index=False)
df_data_final.to_excel(filepath_data_final, index=False)

#%% Plot epr vs dE/dt

fig = plt.figure(figsize=(figure_norm, figure_norm))


fig.suptitle(Pathway_Name,fontsize=Fontsize_Title)
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)




ax1.scatter(all_visited_energy_E_total, all_visited_energy_G_total,alpha=.25, color='grey')

ax1.scatter(most_visited_dedt, most_visited_dgdt, 
            color = RL_color_type,
            marker = RL_shape_type,
            label = RL_method_type,
            s=marker_size)

#ax1.scatter(MCA_energy, MCA_epr, 
#            color = MCA_color_type,
#            marker = MCA_shape_type,
#           label = MCA_method_type,
#           s=marker_size)

#TEMP
if (pathway_choice != 9):
    ax1.scatter(MCA_dedt1, MCA_dgdt1, 
                color = MCA_color_type1,
                alpha=0.95,
                marker = MCA_shape_type1,
                label = MCA_method_type1,
                s=marker_size)

ax1.scatter(MCA_dedt1, MCA_dgdt1, 
            color = MCA_color_type2,
            alpha=0.95,
            marker = MCA_shape_type2,
            label = MCA_method_type2,
            s=marker_size)

#normed version
ax2.scatter(all_visited_energy_E_total,all_visited_energy_G_total,alpha=0.25, color='grey')

ax2.scatter(most_visited_dedt, most_visited_dgdt, 
            color = RL_color_type,
            alpha=0.95,
            label = RL_method_type,
            marker = RL_shape_type,
            s=marker_size)

#ax2.scatter(MCA_energy, MCA_norm_epr, 
#            color = MCA_color_type,
#            marker = MCA_shape_type,
#            label = MCA_method_type,
#           s=marker_size)

#TEMP
if (pathway_choice !=9):
    ax2.scatter(MCA_dedt1, MCA_dgdt1, 
                color = MCA_color_type1,
                alpha=0.95,
                marker = MCA_shape_type1,
                label = MCA_method_type1,
                s=marker_size)

    ax2.scatter(MCA_dedt1, MCA_dgdt2, 
                color = MCA_color_type2,
                alpha=0.95,
                marker = MCA_shape_type2,
                label = MCA_method_type2,
                s=marker_size)

#-epr = dG/dt
ax1.set_xlabel('dE/dt',fontsize=Fontsize_Sub)
ax1.set_ylabel('dG/dt',fontsize=Fontsize_Sub)

ax2.set_xlabel('dE/dt',fontsize=Fontsize_Sub)
ax2.set_ylabel('Normed EPR',fontsize=Fontsize_Sub)

ax1.legend(fontsize=Fontsize_Leg, loc='upper left')
ax2.legend(fontsize=Fontsize_Leg, loc='lower right')
fig.tight_layout()  # otherwise the right y-label is slightly clipped



#%%

final_state_data = pd.DataFrame(columns=['Method','Activity','Reaction'])

counter=0
for i in range(0,(temp_state.shape[1])):
    final_state_data.loc[i] = ['Local Control', most_visited_state[i],i]
  
         
            
fig = plt.figure(figsize=(figure_norm, 0.25*figure_norm))
ax1 = fig.add_subplot(111)

ax1.set_title('Comparison: MCA Method 1 and Local Control ', fontsize=Fontsize_Title)
sns.boxplot(x='Reaction',
            y='Activity',
            data=final_state_data,
            hue='Method',
            palette=sns.light_palette((210, 90, 60), input="husl"),
            ax=ax1)

#To make legend, plot first with label for legend
sns.regplot(x=np.array([0]), y=np.array([activity_to_plot_method1[0]]), scatter=True, fit_reg=False, marker='x',
            ax=ax1,
            label='MCA: Method 1',
            color='r',
            scatter_kws={"s": 100})


for i in range(1,temp_state.shape[1]):
    sns.regplot(x=np.array([i]), y=np.array([activity_to_plot_method1[i]]), scatter=True, fit_reg=False, marker='x',
                ax=ax1,
                color='r',
                scatter_kws={"s": 100})


# =============================================================================
# sns.regplot(x=np.array([epr_method_RL]), y=np.array([1]), scatter=True, fit_reg=False, marker='o',
#                 ax=ax2,
#                 color='r',
#                scatter_kws={"s": 100})
# =============================================================================

ax1.set_xlabel('Reactions',fontsize=Fontsize_Sub)

ax1.set_xticklabels(bigg_reaction_abbr_p, rotation='vertical', fontsize=Fontsize_Sub)
ax1.set_ylabel('Enzyme Activity',fontsize=Fontsize_Sub)

ax1.legend(fontsize=Fontsize_Leg, loc='best')