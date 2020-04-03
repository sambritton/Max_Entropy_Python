# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 09:47:44 2019

@author: samuel_britton
"""

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.gridspec as gridspec
Temperature = 298.15
R_gas = 8.314e-03
RT = R_gas*Temperature

pathway_choice=4

use_exp_data=1

lr1=str('0.0001')
lr2=str('1e-05')
lr3=str('1e-06')
lrs=[lr1,lr2,lr3]

Fontsize_Title=25
Fontsize_Sub = 20
Fontsize_Leg = 15
figure_norm=10

marker_size = 50


colorChoice = sns.xkcd_rgb["black"]

color1 = sns.xkcd_rgb["red"]
color2 = sns.xkcd_rgb["black"]
color3 = sns.xkcd_rgb["green"]

MCA_method_type1 = 'Local MCA'
MCA_method_type2 = 'Unrestricted MCA'

MCA_color_type1 = color1
MCA_color_type2 = color3

MCA_shape_type1 = '+'
MCA_shape_type2 = 'x'

RL_method_type1 = 'RL'
RL_method_type2 = 'RL'

RL_color_type1 = color2
RL_color_type2 = 'b'
RL_shape_type1 = 's'
RL_shape_type2 = 's'

#cwd = 'C:\\Users\\samuel_britton\\Documents\\cannon\\Final_Pathways'
cwd = 'C:\\Users\\samuel_britton\\Documents\\cannon\\regulation_paper'

p1_string = '\\GLUCONEOGENESIS\\models_final_data'

p2_string = '\\GLYCOLYSIS_TCA_GOGAT\\models_final_data'
p3_string = '\\TCA_PPP_GLYCOLYSIS_GOGAT\\models_final_data'
p4_string = '\\TCA_PPP_GLYCOLYSIS_GOGAT_LOWERCONC\\models_final_data'
p5_string = '\\TCA_PPP_GLYCOLYSIS_GOGAT_NOPFK\\models_final_data'


p6_string = '\\GLYCOLYSIS_TCA_GOGAT_noconstraint\\models_final_data'
p7_string = '\\TCA_PPP_GLYCOLYSIS_GOGAT_noconstraint\\models_final_data'
p8_string = '\\TCA_PPP_GLYCOLYSIS_GOGAT_LOWERCONC_noconstraint\\models_final_data'
p9_string = '\\TCA_PPP_GLYCOLYSIS_GOGAT_NOPFK_noconstraint\\models_final_data'


bigg_reaction_abbr_p1 = pd.Series(['G6Pase', 'PGI', 'FBP', 'FBA', 'TPI', 'GAPD', 'PGK',
       'PGM', 'ENO', 'PEPCK', 'PC'])

bigg_reorder_p1 = [0,1,2,3,5,6,7,8,9,10]
able_to_regulate_p1 = [2,9]
not_able_to_regulate_p1 = bigg_reorder_p1.copy()
for elt in able_to_regulate_p1:
    not_able_to_regulate_p1.remove(elt)

bigg_reaction_abbr_p2 = pd.Series(['CSM','ACONT', 'ICDH', 'AKGD', 'SUCOAS', 'SUCD', 'FUM',
       'MDH', 'GAPD', 'PGK', 'TPI', 'FBA', 'PYK', 'PGM', 'ENO', 'HEX1', 'PGI',
       'PFK', 'PYRt2M', 'PDH', 'GOGAT'])


bigg_reorder_p2 = [15, 16, 17, 11, 10, 8, 9, 13, 14, 12, 18, 19, 0, 1, 2, 3, 4, 5, 6, 7, 20]

able_to_regulate_p2 = [0,2,5,6,7,9,11]
not_able_to_regulate_p2 = bigg_reorder_p2.copy()
for elt in able_to_regulate_p2:
    not_able_to_regulate_p2.remove(elt)

bigg_reaction_abbr_p3 = pd.Series(['CSM','ACONT','ICDH','AKGD','SUCOAS',
        'SUCD','FUM','MDH','GAPD','PGK','TPI','FBA','PYK','PGM','ENO',
        'HEX1','PGI','PFK','PYRt2M','PDH','G6PDH','PGL','GND','RPE','RPI',
        'TKT1','TALA','TKT2','GOGAT'])

bigg_reorder_p3 = [15, 16, 17, 11, 10, 8, 9, 13, 14, 12, 18, 19, 
                   20, 21, 22, 24, 23, 25, 26, 27,
                   0, 1, 2, 3, 4, 5, 6, 7, 28]

able_to_regulate_p3 = [0,2,5,6,7,9,11,12,17,19]
not_able_to_regulate_p3 = bigg_reorder_p3.copy()
for elt in able_to_regulate_p3:
    not_able_to_regulate_p3.remove(elt)

bigg_reorder_p4 = [15, 16, 17, 11, 10, 8, 9, 13, 14, 12, 18, 19, 
                   20, 21, 22, 24, 23, 25, 26, 27,
                   0, 1, 2, 3, 4, 5, 6, 7, 28]

able_to_regulate_p4 = [0,2,5,6,7,9,11,12,17,19]
not_able_to_regulate_p4 = bigg_reorder_p4.copy()
for elt in able_to_regulate_p4:
    not_able_to_regulate_p4.remove(elt)

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
dg0_tca_gly_ppp.loc['TALA',deltag0] = -0.729232
dg0_tca_gly_ppp.loc['TKT1',deltag0] = -3.79303
dg0_tca_gly_ppp.loc['TKT2',deltag0] = -10.0342
dg0_tca_gly_ppp.loc['GOGAT',deltag0] = 48.1864

Keq_constant_tca_gly_ppp = np.exp(-dg0_tca_gly_ppp[deltag0].astype('float')/RT)


epr_method1_experimental_path1 = np.loadtxt(cwd+'\\GLUCONEOGENESIS\\'+'EPR_experiment_method1.txt', dtype=float)
epr_method2_experimental_path1 = np.loadtxt(cwd+'\\GLUCONEOGENESIS\\'+'EPR_experiment_method2.txt', dtype=float)

activity_method1_experimental_path1 = np.loadtxt(cwd+'\\GLUCONEOGENESIS\\'+'activities_experiment_method1.txt', dtype=float)
activity_method2_experimental_path1 = np.loadtxt(cwd+'\\GLUCONEOGENESIS\\'+'activities_experiment_method2.txt', dtype=float)
flux_method1_experimental_path1 = np.loadtxt(cwd+'\\GLUCONEOGENESIS\\'+'flux_experiment_method1.txt', dtype=float)
flux_method2_experimental_path1 = np.loadtxt(cwd+'\\GLUCONEOGENESIS\\'+'flux_experiment_method2.txt', dtype=float)
KQF_method1_experimental_path1 = np.loadtxt(cwd+'\\GLUCONEOGENESIS\\'+'KQF_experiment_method1.txt', dtype=float)
KQF_method2_experimental_path1 = np.loadtxt(cwd+'\\GLUCONEOGENESIS\\'+'KQF_experiment_method2.txt', dtype=float)

KQR_method1_experimental_path1 = np.loadtxt(cwd+'\\GLUCONEOGENESIS\\'+'KQR_experiment_method1.txt', dtype=float)
KQR_method2_experimental_path1 = np.loadtxt(cwd+'\\GLUCONEOGENESIS\\'+'KQR_experiment_method2.txt', dtype=float)

reward_methodRL_noconstraint_experimental_path1 = np.loadtxt(cwd+'\\GLUCONEOGENESIS\\'+'reward_experiment_RL_noconstraint.txt', dtype=float)

activity_methodRL_noconstraint_experimental_path1 = np.loadtxt(cwd+'\\GLUCONEOGENESIS\\'+'activities_experiment_RL_noconstraint.txt', dtype=float)
flux_methodRL_noconstraint_experimental_path1 = np.loadtxt(cwd+'\\GLUCONEOGENESIS\\'+'flux_experiment_RL_noconstraint.txt', dtype=float)

KQF_methodRL_noconstraint_experimental_path1 = np.loadtxt(cwd+'\\GLUCONEOGENESIS\\'+'KQF_experiment_RL_noconstraint.txt', dtype=float)
KQR_methodRL_noconstraint_experimental_path1 = np.loadtxt(cwd+'\\GLUCONEOGENESIS\\'+'KQR_experiment_RL_noconstraint.txt', dtype=float)

norm_EPR_methodRL_noconstraint_path1 = np.loadtxt(cwd+'\\GLUCONEOGENESIS\\'+'norm_EPR_experiment_RL_noconstraint.txt', dtype=float)

dg_dt_methodRL_noconstraint_path1 = np.loadtxt(cwd+'\\GLUCONEOGENESIS\\'+'dg_dt_RL_noconstraint.txt', dtype=float)
de_dt_methodRL_noconstraint_path1 = np.loadtxt(cwd+'\\GLUCONEOGENESIS\\'+'de_dt_RL_noconstraint.txt', dtype=float)



#Path 2
#activity_method2_experimental_path2=1
epr_method1_experimental_path2 = np.loadtxt(cwd+'\\GLYCOLYSIS_TCA_GOGAT\\'+'EPR_experiment_method1.txt', dtype=float)
epr_method2_experimental_path2 = np.loadtxt(cwd+'\\GLYCOLYSIS_TCA_GOGAT\\'+'EPR_experiment_method2.txt', dtype=float)

activity_method1_experimental_path2 = np.loadtxt(cwd+'\\GLYCOLYSIS_TCA_GOGAT\\'+'activities_experiment_method1.txt', dtype=float)
activity_method2_experimental_path2 = np.loadtxt(cwd+'\\GLYCOLYSIS_TCA_GOGAT\\'+'activities_experiment_method2.txt', dtype=float)
flux_method1_experimental_path2 = np.loadtxt(cwd+'\\GLYCOLYSIS_TCA_GOGAT\\'+'flux_experiment_method1.txt', dtype=float)
flux_method2_experimental_path2 = np.loadtxt(cwd+'\\GLYCOLYSIS_TCA_GOGAT\\'+'flux_experiment_method2.txt', dtype=float)

KQF_method1_experimental_path2 = np.loadtxt(cwd+'\\GLYCOLYSIS_TCA_GOGAT\\'+'KQF_experiment_method1.txt', dtype=float)
KQF_method2_experimental_path2 = np.loadtxt(cwd+'\\GLYCOLYSIS_TCA_GOGAT\\'+'KQF_experiment_method2.txt', dtype=float)
KQR_method1_experimental_path2 = np.loadtxt(cwd+'\\GLYCOLYSIS_TCA_GOGAT\\'+'KQR_experiment_method1.txt', dtype=float)
KQR_method2_experimental_path2 = np.loadtxt(cwd+'\\GLYCOLYSIS_TCA_GOGAT\\'+'KQR_experiment_method2.txt', dtype=float)

reward_methodRL_noconstraint_experimental_path2 = np.loadtxt(cwd+'\\GLYCOLYSIS_TCA_GOGAT\\'+'reward_experiment_RL_noconstraint.txt', dtype=float)

activity_methodRL_experimental_path2 = np.loadtxt(cwd+'\\GLYCOLYSIS_TCA_GOGAT\\'+'activities_experiment_RL.txt', dtype=float)
activity_methodRL_noconstraint_experimental_path2 = np.loadtxt(cwd+'\\GLYCOLYSIS_TCA_GOGAT\\'+'activities_experiment_RL_noconstraint.txt', dtype=float)
flux_methodRL_experimental_path2 = np.loadtxt(cwd+'\\GLYCOLYSIS_TCA_GOGAT\\'+'flux_experiment_RL.txt', dtype=float)
flux_methodRL_noconstraint_experimental_path2 = np.loadtxt(cwd+'\\GLYCOLYSIS_TCA_GOGAT\\'+'flux_experiment_RL_noconstraint.txt', dtype=float)

KQF_methodRL_experimental_path2 = np.loadtxt(cwd+'\\GLYCOLYSIS_TCA_GOGAT\\'+'KQF_experiment_RL.txt', dtype=float)
KQF_methodRL_noconstraint_experimental_path2 = np.loadtxt(cwd+'\\GLYCOLYSIS_TCA_GOGAT\\'+'KQF_experiment_RL_noconstraint.txt', dtype=float)
KQR_methodRL_noconstraint_experimental_path2 = np.loadtxt(cwd+'\\GLYCOLYSIS_TCA_GOGAT\\'+'KQR_experiment_RL_noconstraint.txt', dtype=float)

norm_EPR_methodRL_noconstraint_path2 = np.loadtxt(cwd+'\\GLYCOLYSIS_TCA_GOGAT\\'+'norm_EPR_experiment_RL_noconstraint.txt', dtype=float)

dg_dt_methodRL_noconstraint_path2 = np.loadtxt(cwd+'\\GLYCOLYSIS_TCA_GOGAT\\'+'dg_dt_RL_noconstraint.txt', dtype=float)
de_dt_methodRL_noconstraint_path2 = np.loadtxt(cwd+'\\GLYCOLYSIS_TCA_GOGAT\\'+'de_dt_RL_noconstraint.txt', dtype=float)


#high-high
epr_method1_experimental_path3 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'EPR_experiment_method1_highhigh.txt', dtype=float)
epr_method2_experimental_path3 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'EPR_experiment_method2_highhigh.txt', dtype=float)

activity_method1_experimental_path3 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'activities_experiment_method1_highhigh.txt', dtype=float)
activity_method2_experimental_path3 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'activities_experiment_method2_highhigh.txt', dtype=float)
flux_method1_experimental_path3 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'flux_experiment_method1_highhigh.txt', dtype=float)
flux_method2_experimental_path3 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'flux_experiment_method2_highhigh.txt', dtype=float)

KQF_method1_experimental_path3 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'KQF_experiment_method1_highhigh.txt', dtype=float)
KQF_method2_experimental_path3 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'KQF_experiment_method2_highhigh.txt', dtype=float)
KQR_method1_experimental_path3 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'KQR_experiment_method1_highhigh.txt', dtype=float)
KQR_method2_experimental_path3 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'KQR_experiment_method2_highhigh.txt', dtype=float)


reward_methodRL_noconstraint_experimental_path3 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'reward_experiment_RL_noconstraint_highhigh.txt', dtype=float)
activity_methodRL_experimental_path3 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'activities_experiment_RL_highhigh.txt', dtype=float)
activity_methodRL_noconstraint_experimental_path3 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'activities_experiment_RL_noconstraint_highhigh.txt', dtype=float)
flux_methodRL_experimental_path3 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'flux_experiment_RL_highhigh.txt', dtype=float)
flux_methodRL_noconstraint_experimental_path3 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'flux_experiment_RL_noconstraint_highhigh.txt', dtype=float)

KQF_methodRL_experimental_path3 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'KQF_experiment_RL_highhigh.txt', dtype=float)
KQF_methodRL_noconstraint_experimental_path3 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'KQF_experiment_RL_noconstraint_highhigh.txt', dtype=float)
#KQR_methodRL_experimental_path3 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'KQR_experiment_RL_highhigh.txt', dtype=float)
KQR_methodRL_noconstraint_experimental_path3 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'KQR_experiment_RL_noconstraint_highhigh.txt', dtype=float)

norm_EPR_methodRL_constraint_path3 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'norm_EPR_experiment_RL_highhigh.txt', dtype=float)
norm_EPR_methodRL_noconstraint_path3 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'norm_EPR_experiment_RL_noconstraint_highhigh.txt', dtype=float)

dg_dt_methodRL_constraint_path3 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'dg_dt_RL_constraint_highhigh.txt', dtype=float)
de_dt_methodRL_constraint_path3 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'de_dt_RL_constraint_highhigh.txt', dtype=float)

dg_dt_methodRL_noconstraint_path3 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'dg_dt_RL_noconstraint_highhigh.txt', dtype=float)
de_dt_methodRL_noconstraint_path3 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'de_dt_RL_noconstraint_highhigh.txt', dtype=float)



#high-low

epr_method1_experimental_path4 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'EPR_experiment_method1_highlow.txt', dtype=float)
epr_method2_experimental_path4 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'EPR_experiment_method2_highlow.txt', dtype=float)

activity_method1_experimental_path4 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'activities_experiment_method1_highlow.txt', dtype=float)
activity_method2_experimental_path4 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'activities_experiment_method2_highlow.txt', dtype=float)
flux_method1_experimental_path4 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'flux_experiment_method1_highlow.txt', dtype=float)
flux_method2_experimental_path4 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'flux_experiment_method2_highlow.txt', dtype=float)

KQF_method1_experimental_path4 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'KQF_experiment_method1_highlow.txt', dtype=float)
KQF_method2_experimental_path4 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'KQF_experiment_method2_highlow.txt', dtype=float)
KQR_method1_experimental_path4 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'KQR_experiment_method1_highlow.txt', dtype=float)
KQR_method2_experimental_path4 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'KQR_experiment_method2_highlow.txt', dtype=float)

activity_methodRL_experimental_path4 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT_LOWERCONC\\'+'activities_experiment_RL_highlow.txt', dtype=float)
activity_methodRL_noconstraint_experimental_path4 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT_LOWERCONC\\'+'activities_experiment_RL_noconstraint_highlow.txt', dtype=float)
flux_methodRL_experimental_path4 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT_LOWERCONC\\'+'flux_experiment_RL_highlow.txt', dtype=float)
flux_methodRL_noconstraint_experimental_path4 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT_LOWERCONC\\'+'flux_experiment_RL_noconstraint_highlow.txt', dtype=float)


reward_methodRL_noconstraint_experimental_path4 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT_LOWERCONC\\'+'reward_experiment_RL_noconstraint_highlow.txt', dtype=float)
KQF_methodRL_experimental_path4 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT_LOWERCONC\\'+'KQF_experiment_RL_highlow.txt', dtype=float)
KQF_methodRL_noconstraint_experimental_path4 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT_LOWERCONC\\'+'KQF_experiment_RL_noconstraint_highlow.txt', dtype=float)
#KQR_methodRL_experimental_path4 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT_LOWERCONC\\'+'KQR_experiment_RL_highlow.txt', dtype=float)
KQR_methodRL_noconstraint_experimental_path4 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT_LOWERCONC\\'+'KQR_experiment_RL_noconstraint_highlow.txt', dtype=float)

norm_EPR_methodRL_constraint_path4 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT_LOWERCONC\\'+'norm_EPR_experiment_RL_highlow.txt', dtype=float)
norm_EPR_methodRL_noconstraint_path4 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT_LOWERCONC\\'+'norm_EPR_experiment_RL_noconstraint_highlow.txt', dtype=float)

dg_dt_methodRL_constraint_path4 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT_LOWERCONC\\'+'dg_dt_RL_constraint_highlow.txt', dtype=float)
de_dt_methodRL_constraint_path4 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT_LOWERCONC\\'+'de_dt_RL_constraint_highlow.txt', dtype=float)

dg_dt_methodRL_noconstraint_path4 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT_LOWERCONC\\'+'dg_dt_RL_noconstraint_highlow.txt', dtype=float)
de_dt_methodRL_noconstraint_path4 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT_LOWERCONC\\'+'de_dt_RL_noconstraint_highlow.txt', dtype=float)


#high high no pfk
epr_method1_experimental_path5 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'EPR_experiment_method1_highhigh_PFKZERO.txt', dtype=float)
epr_method2_experimental_path5 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'EPR_experiment_method2_highhigh_PFKZERO.txt', dtype=float)

activity_method1_experimental_path5 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'activities_experiment_method1_highhigh_PFKZERO.txt', dtype=float)
activity_method2_experimental_path5 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'activities_experiment_method2_highhigh_PFKZERO.txt', dtype=float)
flux_method1_experimental_path5 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'flux_experiment_method1_highhigh_PFKZERO.txt', dtype=float)
flux_method2_experimental_path5 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'flux_experiment_method2_highhigh_PFKZERO.txt', dtype=float)

KQF_method1_experimental_path5 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'KQF_experiment_method1_highhigh_PFKZERO.txt', dtype=float)
KQF_method2_experimental_path5 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'KQF_experiment_method2_highhigh_PFKZERO.txt', dtype=float)
KQR_method1_experimental_path5 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'KQR_experiment_method1_highhigh_PFKZERO.txt', dtype=float)
KQR_method2_experimental_path5 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'KQR_experiment_method2_highhigh_PFKZERO.txt', dtype=float)


reward_methodRL_noconstraint_experimental_path5 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'reward_experiment_RL_noconstraint_highhigh_PFKZERO.txt', dtype=float)
activity_methodRL_experimental_path5 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'activities_experiment_RL_highhigh_PFKZERO.txt', dtype=float)
activity_methodRL_noconstraint_experimental_path5 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'activities_experiment_RL_noconstraint_highhigh_PFKZERO.txt', dtype=float)
flux_methodRL_experimental_path5 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'flux_experiment_RL_highhigh_PFKZERO.txt', dtype=float)
flux_methodRL_noconstraint_experimental_path5 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'flux_experiment_RL_noconstraint_highhigh_PFKZERO.txt', dtype=float)

KQF_methodRL_experimental_path5 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'KQF_experiment_RL_highhigh_PFKZERO.txt', dtype=float)
KQF_methodRL_noconstraint_experimental_path5 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'KQF_experiment_RL_noconstraint_highhigh_PFKZERO.txt', dtype=float)
#KQR_methodRL_experimental_path5 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'KQR_experiment_RL_highhigh_PFKZERO.txt', dtype=float)
KQR_methodRL_noconstraint_experimental_path5 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'KQR_experiment_RL_noconstraint_highhigh_PFKZERO.txt', dtype=float)

norm_EPR_methodRL_constraint_path5 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'norm_EPR_experiment_RL_highhigh_PFKZERO.txt', dtype=float)
norm_EPR_methodRL_noconstraint_path5 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'norm_EPR_experiment_RL_noconstraint_highhigh_PFKZERO.txt', dtype=float)

dg_dt_methodRL_constraint_path5 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'dg_dt_RL_constraint_highhigh_PFKZERO.txt', dtype=float)
de_dt_methodRL_constraint_path5 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'de_dt_RL_constraint_highhigh_PFKZERO.txt', dtype=float)

dg_dt_methodRL_noconstraint_path5 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'dg_dt_RL_noconstraint_highhigh_PFKZERO.txt', dtype=float)
de_dt_methodRL_noconstraint_path5 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'de_dt_RL_noconstraint_highhigh_PFKZERO.txt', dtype=float)



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

    epr_to_plot_methodRL_noconstraint = norm_EPR_methodRL_noconstraint_path1

    activity_to_plot_methodRL_noconstraint = activity_methodRL_noconstraint_experimental_path1

    flux_to_plot_methodRL_noconstraint = flux_methodRL_noconstraint_experimental_path1
    
    KQF_to_plot_methodRL_noconstraint = KQF_methodRL_noconstraint_experimental_path1
    KQR_to_plot_methodRL_noconstraint = KQR_methodRL_noconstraint_experimental_path1
    
    de_dt_to_plot_methodRL_noconstraint = de_dt_methodRL_noconstraint_path1
    dg_dt_to_plot_methodRL_noconstraint = dg_dt_methodRL_noconstraint_path1
    
    Pathway_Name='GLUCONEOGENESIS'
        
    #p_string_random = cwd + '\\GLYCOLYSIS_TCA_GOGAT_randomds'
    bigg_reorder = bigg_reorder_p1
    p_string = cwd + p1_string
    bigg_reaction_abbr_p = bigg_reaction_abbr_p1
    eps=0.5
    
    MCA_method_type = MCA_method_type1
    MCA_color_type = MCA_color_type1
    MCA_shape_type = MCA_shape_type1
    
    RL_method_type = RL_method_type1
    RL_color_type = RL_color_type1
    RL_shape_type = RL_shape_type1
    
    
    regulate_indices = able_to_regulate_p1
    do_not_regulate_indices = not_able_to_regulate_p1


if (pathway_choice==2):
    
    Keq_constant = Keq_constant_tca_gly
    
    epr_to_plot_method1 = epr_method1_experimental_path2
    epr_to_plot_method2 = epr_method2_experimental_path2
    activity_to_plot_method1=activity_method1_experimental_path2
    activity_to_plot_method2=activity_method2_experimental_path2
    flux_to_plot_method1=flux_method1_experimental_path2
    flux_to_plot_method2=flux_method2_experimental_path2
    
    KQF_to_plot_method1=KQF_method1_experimental_path2
    KQF_to_plot_method2=KQF_method2_experimental_path2
    KQR_to_plot_method1=KQR_method1_experimental_path2
    KQR_to_plot_method2=KQR_method2_experimental_path2
    
    epr_to_plot_methodRL_noconstraint = norm_EPR_methodRL_noconstraint_path2
    activity_to_plot_methodRL = activity_methodRL_experimental_path2
    activity_to_plot_methodRL_noconstraint = activity_methodRL_noconstraint_experimental_path2
    flux_to_plot_methodRL = flux_methodRL_experimental_path2
    flux_to_plot_methodRL_noconstraint = flux_methodRL_noconstraint_experimental_path2
    
    KQF_to_plot_methodRL_noconstraint = KQF_methodRL_noconstraint_experimental_path2
    KQR_to_plot_methodRL_noconstraint = KQR_methodRL_noconstraint_experimental_path2
    
    de_dt_to_plot_methodRL_noconstraint = de_dt_methodRL_noconstraint_path2
    dg_dt_to_plot_methodRL_noconstraint = dg_dt_methodRL_noconstraint_path2
    Pathway_Name='Glycolysis-TCA'
        
    bigg_reorder = bigg_reorder_p2
    p_string = cwd + p2_string
    bigg_reaction_abbr_p = bigg_reaction_abbr_p2
    
    
    regulate_indices = able_to_regulate_p2
    do_not_regulate_indices = not_able_to_regulate_p2
    
    eps=0.5


#TCA_PPP_GLY with high/high
if (pathway_choice==3):
    
    Keq_constant = Keq_constant_tca_gly_ppp
    
    epr_to_plot_method1 = epr_method1_experimental_path3
    epr_to_plot_method2 = epr_method2_experimental_path3
    activity_to_plot_method1=activity_method1_experimental_path3
    activity_to_plot_method2=activity_method2_experimental_path3
    flux_to_plot_method1=flux_method1_experimental_path3
    flux_to_plot_method2=flux_method2_experimental_path3
    
    KQF_to_plot_method1=KQF_method1_experimental_path3
    KQF_to_plot_method2=KQF_method2_experimental_path3
    
    KQR_to_plot_method1=KQR_method1_experimental_path3
    KQR_to_plot_method2=KQR_method2_experimental_path3
    
    
    activity_to_plot_methodRL = activity_methodRL_experimental_path3
    activity_to_plot_methodRL_noconstraint = activity_methodRL_noconstraint_experimental_path3
    flux_to_plot_methodRL = flux_methodRL_experimental_path3
    flux_to_plot_methodRL_noconstraint = flux_methodRL_noconstraint_experimental_path3
    KQF_to_plot_methodRL = KQF_methodRL_experimental_path3
    KQF_to_plot_methodRL_noconstraint = KQF_methodRL_noconstraint_experimental_path3
    KQR_to_plot_methodRL_noconstraint = KQR_methodRL_noconstraint_experimental_path3
    
    epr_to_plot_methodRL_constraint = norm_EPR_methodRL_constraint_path3
    epr_to_plot_methodRL_noconstraint = norm_EPR_methodRL_noconstraint_path3
    
    de_dt_to_plot_methodRL_noconstraint = de_dt_methodRL_noconstraint_path3
    dg_dt_to_plot_methodRL_noconstraint = dg_dt_methodRL_noconstraint_path3
    de_dt_to_plot_methodRL_constraint = de_dt_methodRL_constraint_path3
    dg_dt_to_plot_methodRL_constraint = dg_dt_methodRL_constraint_path3
    
    Pathway_Name='Glycolysis-PPP-TCA: High/High'
        
    bigg_reorder = bigg_reorder_p3
    p_string = cwd + p3_string
    bigg_reaction_abbr_p = bigg_reaction_abbr_p3
    
    regulate_indices = able_to_regulate_p3
    do_not_regulate_indices = not_able_to_regulate_p3
    
    eps=0.2


#TCA_PPP_GLY with high/low
if (pathway_choice==4):
    KQF_init_unregulated = np.array([79.57448718, 79.57448718, 79.57448718, 79.57448718, 79.57448718,
       79.57448718, 79.57448718, 79.57448718, 79.57448718, 79.57448718,
       39.54099841, 39.54099841, 79.57448718, 79.57448718, 79.57448718,
       40.07116773, 38.48068739, 39.54099841, 79.57448718, 79.57448718,
        2.07373378,  2.07373378,  2.07373378,  1.66250854,  1.29983334,
        1.29983334,  1.29983334,  1.29983334,  1.        ])
    flux_init_unregulated = np.array([79.56192034, 79.56192034, 79.56192034, 79.56192034, 79.56192034,
       79.56192034, 79.56192034, 79.56192034, 79.56192034, 79.56192034,
       39.5157082 , 39.5157082 , 79.56192034, 79.56192034, 79.56192034,
       40.04621214, 38.45470033, 39.5157082 , 79.56192034, 79.56192034,
        1.59151181,  1.59151181,  1.59151181,  1.06100787,  0.53050394,
        0.53050394,  0.53050394,  0.53050394, -0.        ])
    
    Keq_constant = Keq_constant_tca_gly_ppp
    
    epr_to_plot_method1 = epr_method1_experimental_path4
    epr_to_plot_method2 = epr_method2_experimental_path4
    activity_to_plot_method1=activity_method1_experimental_path4
    activity_to_plot_method2=activity_method2_experimental_path4
    flux_to_plot_method1=flux_method1_experimental_path4
    flux_to_plot_method2=flux_method2_experimental_path4
    
    KQF_to_plot_method1=KQF_method1_experimental_path4
    KQF_to_plot_method2=KQF_method2_experimental_path4
    KQR_to_plot_method1=KQR_method1_experimental_path4
    KQR_to_plot_method2=KQR_method2_experimental_path4
    
    activity_to_plot_methodRL = activity_methodRL_experimental_path4
    activity_to_plot_methodRL_noconstraint = activity_methodRL_noconstraint_experimental_path4
    flux_to_plot_methodRL = flux_methodRL_experimental_path4
    flux_to_plot_methodRL_noconstraint = flux_methodRL_noconstraint_experimental_path4
    
    KQF_to_plot_methodRL = KQF_methodRL_experimental_path4
    KQF_to_plot_methodRL_noconstraint = KQF_methodRL_noconstraint_experimental_path4
    KQR_to_plot_methodRL_noconstraint = KQR_methodRL_noconstraint_experimental_path4
    
    
    epr_to_plot_methodRL_constraint = norm_EPR_methodRL_constraint_path4
    epr_to_plot_methodRL_noconstraint = norm_EPR_methodRL_noconstraint_path4
    
    de_dt_to_plot_methodRL_noconstraint = de_dt_methodRL_noconstraint_path4
    dg_dt_to_plot_methodRL_noconstraint = dg_dt_methodRL_noconstraint_path4
    de_dt_to_plot_methodRL_constraint = de_dt_methodRL_constraint_path4
    dg_dt_to_plot_methodRL_constraint = dg_dt_methodRL_constraint_path4
    
    Pathway_Name='Glycolysis-PPP-TCA: High/Low'
        
    
    bigg_reorder = bigg_reorder_p3
    p_string = cwd + p4_string
    
    bigg_reaction_abbr_p = bigg_reaction_abbr_p3
    
    regulate_indices = able_to_regulate_p3
    do_not_regulate_indices = not_able_to_regulate_p3
    
    eps=0.2




#TCA_PPP_GLY no pfk with high/high
if (pathway_choice==5):
    
    Keq_constant = Keq_constant_tca_gly_ppp
    
    epr_to_plot_method1 = epr_method1_experimental_path5
    epr_to_plot_method2 = epr_method2_experimental_path5
    activity_to_plot_method1=activity_method1_experimental_path5
    activity_to_plot_method2=activity_method2_experimental_path5
    flux_to_plot_method1=flux_method1_experimental_path5
    flux_to_plot_method2=flux_method2_experimental_path5
    
    KQF_to_plot_method1=KQF_method1_experimental_path5
    KQF_to_plot_method2=KQF_method2_experimental_path5
    KQR_to_plot_method1=KQR_method1_experimental_path5
    KQR_to_plot_method2=KQR_method2_experimental_path5
    
    
    activity_to_plot_methodRL = activity_methodRL_experimental_path5
    activity_to_plot_methodRL_noconstraint = activity_methodRL_noconstraint_experimental_path5
    flux_to_plot_methodRL = flux_methodRL_experimental_path5
    flux_to_plot_methodRL_noconstraint = flux_methodRL_noconstraint_experimental_path5
    
    KQF_to_plot_methodRL = KQF_methodRL_experimental_path5
    KQF_to_plot_methodRL_noconstraint = KQF_methodRL_noconstraint_experimental_path5
    KQR_to_plot_methodRL_noconstraint = KQR_methodRL_noconstraint_experimental_path5
    
    epr_to_plot_methodRL_constraint = norm_EPR_methodRL_constraint_path5
    epr_to_plot_methodRL_noconstraint = norm_EPR_methodRL_noconstraint_path5
    
    de_dt_to_plot_methodRL_noconstraint = de_dt_methodRL_noconstraint_path5
    dg_dt_to_plot_methodRL_noconstraint = dg_dt_methodRL_noconstraint_path5
    de_dt_to_plot_methodRL_constraint = de_dt_methodRL_constraint_path5
    dg_dt_to_plot_methodRL_constraint = dg_dt_methodRL_constraint_path5


    
    Pathway_Name='Glycolysis-PPP-TCA: High/High, No PFK'
    
    bigg_reorder = bigg_reorder_p4
    p_string = cwd + p5_string
    bigg_reaction_abbr_p = bigg_reaction_abbr_p3
    
    regulate_indices = able_to_regulate_p4
    do_not_regulate_indices = not_able_to_regulate_p4
    
    eps=0.2

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

MCA_activity1 = activity_to_plot_method1
MCA_KQF1 = KQF_to_plot_method1
MCA_KQR1 = KQR_to_plot_method1

MCA_norm_epr1 = epr_to_plot_method1
MCA_de_dt1 = dedt(MCA_activity1,MCA_KQF1,MCA_KQR1, Keq_constant)
MCA_dg_dt1 = dgdt(MCA_activity1,MCA_KQF1,MCA_KQR1, Keq_constant)
#-np.sum(flux_to_plot_method1 * np.log(Keq_constant) )

MCA_activity2 = activity_to_plot_method2
MCA_KQF2 = KQF_to_plot_method2
MCA_KQR2 = KQR_to_plot_method2

MCA_norm_epr2 = epr_to_plot_method2
MCA_de_dt2 = dedt(MCA_activity2,MCA_KQF2,MCA_KQR2, Keq_constant)
MCA_dg_dt2 = dgdt(MCA_activity2,MCA_KQF2,MCA_KQR2, Keq_constant)

RL_noconstraint_activity = activity_to_plot_methodRL_noconstraint
RL_noconstraint_KQF= KQF_to_plot_methodRL_noconstraint
RL_noconstraint_KQR= KQR_to_plot_methodRL_noconstraint

RL_noconstraint_dg_dt = dgdt(RL_noconstraint_activity,RL_noconstraint_KQF,RL_noconstraint_KQR,Keq_constant)
#-epr_to_plot_methodRL_noconstraint * np.sum(flux_to_plot_methodRL_noconstraint)
RL_noconstraint_norm_epr = epr_to_plot_methodRL_noconstraint
RL_noconstraint_de_dt = dedt(RL_noconstraint_activity,RL_noconstraint_KQF,RL_noconstraint_KQR,Keq_constant)
#-np.sum(flux_to_plot_methodRL_noconstraint * np.log(Keq_constant))

RL_de_dt_population_noconstraint = de_dt_to_plot_methodRL_noconstraint
RL_dg_dt_population_noconstraint = dg_dt_to_plot_methodRL_noconstraint

#RL_constraint_activity = activity_to_plot_methodRL
#RL_constraint_dg_dt = -epr_to_plot_methodRL_constraint * np.sum(flux_to_plot_methodRL)
#RL_constraint_norm_epr = epr_to_plot_methodRL_constraint
#RL_constraint_de_dt = -np.sum(flux_to_plot_methodRL * np.log(Keq_constant))
#RL_de_dt_population_constraint = de_dt_to_plot_methodRL_constraint
#RL_dg_dt_population_constraint = dg_dt_to_plot_methodRL_constraint

x_scatter = [i for i in range(0,len(bigg_reorder))]

#%%plot for gluconeogenesis
  
  
color3 = sns.xkcd_rgb["steel grey"]
fig = plt.figure(figsize=(figure_norm, 0.75*figure_norm))
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)

KQF_method1 = KQF_to_plot_method1[bigg_reorder]
dg_method1 = -np.log(KQF_method1)

activity_method1 = activity_to_plot_method1[bigg_reorder]

flux_method1 = flux_to_plot_method1[bigg_reorder]


sns.regplot(x=np.array([0]), y=np.array([dg_method1[0]]), scatter=True, fit_reg=False, marker='x',
            ax=ax1,
            label='',
            color=color3,
            scatter_kws={"s": 100})
for i in range(1,len(bigg_reorder)):
    sns.regplot(x=np.array([i]), y=np.array([dg_method1[i]]), scatter=True, fit_reg=False, marker='x',
                ax=ax1,
                color=MCA_color_type1,
                scatter_kws={"s": 100})


sns.regplot(x=np.array([0]), y=np.array([activity_method1[0]]), scatter=True, fit_reg=False, marker=MCA_shape_type1,
            ax=ax2,
            label='',
            color=color3,
            scatter_kws={"s": 100})
for i in range(1,len(bigg_reorder)):
    sns.regplot(x=np.array([i]), y=np.array([activity_method1[i]]), scatter=True, fit_reg=False, marker='x',
                ax=ax2,
                color=MCA_color_type2,
                scatter_kws={"s": 100})


sns.regplot(x=np.array([0]), y=np.array([flux_method1[0]]), scatter=True, fit_reg=False, marker='x',
            ax=ax3,
            label='',
            color=color3,
            scatter_kws={"s": 100})


for i in range(1,len(bigg_reorder)):
    sns.regplot(x=np.array([i]), y=np.array([flux_method1[i]]), scatter=True, fit_reg=False, marker='x',
                ax=ax3,
                color=color3,
                scatter_kws={"s": 100})
    
#fig.suptitle(Pathway_Name,fontsize=Fontsize_Title)

ax3.set_xticks( [i for i in range(0,len(bigg_reorder)+1)] )
ax3.set_xticklabels(bigg_reaction_abbr_p[bigg_reorder], rotation='vertical', fontsize=Fontsize_Sub)

ax3.set_xlabel('Reactions',fontsize=Fontsize_Sub)
ax1.set_ylabel(r'$\Delta{G}/RT$',fontsize=Fontsize_Sub)
ax2.set_ylabel('Enzyme Activity',fontsize=Fontsize_Sub)
ax3.set_ylabel('Flux',fontsize=Fontsize_Sub)

ax1.annotate("A", xy=(-0.1, 1.0), xycoords="axes fraction",fontsize=Fontsize_Sub)
ax2.annotate("B", xy=(-0.1, 1.0), xycoords="axes fraction",fontsize=Fontsize_Sub)
ax3.annotate("C", xy=(-0.1, 1.0), xycoords="axes fraction",fontsize=Fontsize_Sub)
#fig.tight_layout()

#%% simple flux plot

fig = plt.figure(figsize=(figure_norm, 0.25*figure_norm))
ax1 = fig.add_subplot(1,1,1)

flux_method1 = flux_to_plot_method1[bigg_reorder]
flux_method2 = flux_to_plot_method2[bigg_reorder]

flux_method_RL_noconstraint = flux_to_plot_methodRL_noconstraint[bigg_reorder]

ax1.scatter(x_scatter,flux_method1,
            color = MCA_color_type1,
            alpha=1,
            marker = MCA_shape_type1,
            label = MCA_method_type1,
            s=2*marker_size)

ax1.scatter(x_scatter,flux_method2,
            color = MCA_color_type2,
            alpha=1,
            marker = MCA_shape_type2,
            label = MCA_method_type2,
            s=2*marker_size)


ax1.scatter(x_scatter, flux_method_RL_noconstraint, 
            color = RL_color_type1,
            alpha=1,
            facecolors='none',
            marker = RL_shape_type1,
            label = RL_method_type1,
            s=2*marker_size) 

#fig.suptitle(Pathway_Name,fontsize=Fontsize_Title, y=1.0)

ax1.set_xticks( [i for i in range(0,len(bigg_reorder))] )

ax1.set_xticklabels(bigg_reaction_abbr_p[bigg_reorder], rotation='vertical', 
                    fontsize=Fontsize_Sub,
                    weight='normal')

for elt in regulate_indices:
    ax1.get_xticklabels()[elt].set_weight('bold')



ax1.set_xlabel('Reactions',fontsize=Fontsize_Sub)
ax1.set_ylabel('Flux',fontsize=Fontsize_Sub)
ax1.legend(fontsize=Fontsize_Leg, loc='best')

#%%reward plot

fig = plt.figure(figsize=(figure_norm, 0.75*figure_norm))
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

color_xkcd_1 = sns.xkcd_rgb["pale red"]
color_xkcd_2 = sns.xkcd_rgb["medium green"]
color_xkcd_3 = sns.xkcd_rgb["denim blue"]

flux_method_RL_noconstraint = flux_to_plot_methodRL_noconstraint[bigg_reorder]
reward_scatter_1=[i for i in range(0,len(reward_methodRL_noconstraint_experimental_path5))]


ax1.plot(reward_methodRL_noconstraint_experimental_path1,
            color = color_xkcd_1, label='Gluconeogenesis')
ax1.plot(reward_methodRL_noconstraint_experimental_path2,
            color = color_xkcd_2, label='Glycolysis-TCA')

ax2.plot(reward_methodRL_noconstraint_experimental_path3,
            color = color_xkcd_1, label='High/High')
ax2.plot(reward_methodRL_noconstraint_experimental_path4,
            color = color_xkcd_2, label='High/Low')
ax2.plot(reward_methodRL_noconstraint_experimental_path5,
            color = color_xkcd_3, label='High/High PFK=0')

#fig.suptitle('Pathway Rewards',fontsize=Fontsize_Title)

ax2.set_xlabel("Episode",fontsize=Fontsize_Sub)
ax1.set_ylabel("Averaged Normalized \nEpisodic Reward",fontsize=Fontsize_Sub)
ax2.set_ylabel("Averaged Normalized \nEpisodic Reward",fontsize=Fontsize_Sub)
#ax2.set_xticks(nvals)

ax1.legend(fontsize=Fontsize_Leg, loc='lower right')
ax2.legend(fontsize=Fontsize_Leg, loc='lower right')

ax1.annotate("A", xy=(-0.1, 1.05), xycoords="axes fraction",fontsize=Fontsize_Sub)
ax2.annotate("B", xy=(-0.1, 1.05), xycoords="axes fraction",fontsize=Fontsize_Sub)

#%% activity energy, dgdt/dedt plot

colorChoice = sns.xkcd_rgb["black"]

color1 = sns.xkcd_rgb["slate grey"]
color2 = sns.xkcd_rgb["grey"]
color3 = sns.xkcd_rgb["steel grey"]

#fig = plt.figure(figsize=(figure_norm, 2*figure_norm))
#ax1 = fig.add_subplot(2,1,1)
#ax2 = fig.add_subplot(2,1,2)
#ax3 = fig.add_subplot(3,1,3)

fig = plt.figure(figsize=(figure_norm, figure_norm),constrained_layout=True)
spec2 = gridspec.GridSpec(ncols=1, nrows=3, figure=fig)
ax1 = fig.add_subplot(spec2[0, 0])
ax2 = fig.add_subplot(spec2[1, 0])
ax3 = fig.add_subplot(spec2[2, 0])

activity_method1 = activity_to_plot_method1[bigg_reorder]
activity_method2 = activity_to_plot_method2[bigg_reorder]

activity_method_RL_noconstraint = activity_to_plot_methodRL_noconstraint[bigg_reorder]

KQF_method1 = KQF_to_plot_method1[bigg_reorder]
KQF_method2 = KQF_to_plot_method2[bigg_reorder]

dg_method1 = -np.log(KQF_method1)
dg_method2 = -np.log(KQF_method2)

KQF_method_RL_noconstraint = KQF_to_plot_methodRL_noconstraint[bigg_reorder]

dg_methodRL_noconstraint = -np.log(KQF_method_RL_noconstraint)


ax1.scatter(x_scatter,activity_method1,
            color = MCA_color_type1,
            alpha=1,
            marker = MCA_shape_type1,
            label = MCA_method_type1,
            s=2*marker_size)

ax1.scatter(x_scatter,activity_method2,
            color = MCA_color_type2,
            alpha=1,
            marker = MCA_shape_type2,
            label = MCA_method_type2,
            s=2*marker_size)

ax1.scatter(x_scatter, activity_method_RL_noconstraint, 
            color = RL_color_type1,
            alpha=1,
            facecolors='none',
            marker = RL_shape_type1,
            label = RL_method_type1,
            s=2*marker_size) 

ax2.scatter(x_scatter,dg_method1,
            color = MCA_color_type1,
            alpha=1,
            marker = MCA_shape_type1,
            label = MCA_method_type1,
            s=2*marker_size)

ax2.scatter(x_scatter,dg_method2,
            color = MCA_color_type2,
            alpha=1,
            marker = MCA_shape_type2,
            label = MCA_method_type2,
            s=2*marker_size)

ax2.scatter(x_scatter, dg_methodRL_noconstraint, 
            color = RL_color_type1,
            alpha=1,
            facecolors='none',
            marker = RL_shape_type1,
            label = RL_method_type1,
            s=2*marker_size) 

#fig.suptitle(Pathway_Name,fontsize=Fontsize_Title)

ax2.set_xticks( [i for i in range(0,len(bigg_reorder)+1)] )
ax2.set_xticklabels(bigg_reaction_abbr_p[bigg_reorder], rotation='vertical', fontsize=Fontsize_Sub)

for elt in regulate_indices:
    ax2.get_xticklabels()[elt].set_weight('bold')

ax2.set_xlabel('Reactions',fontsize=Fontsize_Sub)
ax1.set_ylabel('Enzyme Activity',fontsize=Fontsize_Sub)
ax2.set_ylabel(r'$\Delta{G}/RT$',fontsize=Fontsize_Sub)

ax1.annotate("A", xy=(-0.1, 1.0), xycoords="axes fraction",fontsize=Fontsize_Sub)
ax2.annotate("B", xy=(-0.1, 1.0), xycoords="axes fraction",fontsize=Fontsize_Sub)
ax3.annotate("C", xy=(-0.1, 1.0), xycoords="axes fraction",fontsize=Fontsize_Sub)


if (pathway_choice!=5):
    ax3.scatter(MCA_de_dt1, MCA_dg_dt1, 
                color = MCA_color_type1,
                alpha=1,
                marker = MCA_shape_type1,
                label = MCA_method_type1,
                s=2*marker_size)

ax3.scatter(MCA_de_dt2, MCA_dg_dt2, 
            color = MCA_color_type2,
            alpha=1,
            marker = MCA_shape_type2,
            label = MCA_method_type2,
            s=2*marker_size)


ax3.scatter(RL_noconstraint_de_dt, RL_noconstraint_dg_dt, 
            color = RL_color_type1,
            alpha=1,
            facecolors='none',
            marker = RL_shape_type1,
            label = RL_method_type1,
            s=3*marker_size)


ax3.scatter(de_dt_to_plot_methodRL_noconstraint,
            dg_dt_to_plot_methodRL_noconstraint,
            alpha=0.5, color='grey', marker='o',
            s=0.325*marker_size, 
            label='RL Population')

#-epr = dG/dt
ax3.set_xlabel('dE/dt',fontsize=Fontsize_Sub)

ax3.set_ylabel('dG/dt',fontsize=Fontsize_Sub)


ax1.legend(fontsize=Fontsize_Leg, loc='best')
ax2.legend(fontsize=Fontsize_Leg, loc='best')
ax3.legend(fontsize=Fontsize_Leg, loc='lower right')

#%% initial energy high low


fig = plt.figure(figsize=(figure_norm, 0.5*figure_norm))


#fig.suptitle(Pathway_Name,fontsize=Fontsize_Title)
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

KQF_init_unregulated_reorder = KQF_init_unregulated[bigg_reorder]
dg_init_unregulated_reorder = -np.log(KQF_init_unregulated_reorder)

ax1.scatter(x_scatter,dg_init_unregulated_reorder,
            color = 'grey',
            alpha=1,
            marker = 'o',
            label = MCA_method_type1,
            s=2*marker_size)

flux_reorder = flux_init_unregulated[bigg_reorder]
ax2.scatter(x_scatter,flux_reorder,
            color = 'grey',
            alpha=1,
            marker = 'o',
            label = MCA_method_type1,
            s=2*marker_size)


#fig.suptitle(Pathway_Name,fontsize=Fontsize_Title)

ax2.set_xticks( [i for i in range(0,len(bigg_reorder)+1)] )
ax2.set_xticklabels(bigg_reaction_abbr_p[bigg_reorder], rotation='vertical', fontsize=Fontsize_Sub)

for elt in regulate_indices:
    ax2.get_xticklabels()[elt].set_weight('bold')

ax2.set_xlabel('Reactions',fontsize=Fontsize_Sub)
ax1.set_ylabel(r'$\Delta{G}/RT$',fontsize=Fontsize_Sub)
ax2.set_ylabel(r'Flux',fontsize=Fontsize_Sub)


ax1.annotate("A", xy=(-0.1, 1.0), xycoords="axes fraction",fontsize=Fontsize_Sub)
ax2.annotate("B", xy=(-0.1, 1.0), xycoords="axes fraction",fontsize=Fontsize_Sub)
#%%

#high/high
KQF_method1_high_high = KQF_method1_experimental_path3[bigg_reorder_p3]
KQF_method2_high_high = KQF_method2_experimental_path3[bigg_reorder_p3]

dg_method1_high_high = -np.log(KQF_method1_high_high)
dg_method2_high_high = -np.log(KQF_method2_high_high)

KQF_method_RL_high_high = KQF_methodRL_experimental_path3[bigg_reorder_p3]
KQF_method_RL_noconstraint_high_high = KQF_methodRL_noconstraint_experimental_path3[bigg_reorder_p3]

dg_methodRL_high_high = -np.log(KQF_method_RL_high_high)
dg_methodRL_noconstraint_high_high = -np.log(KQF_method_RL_noconstraint_high_high)

#high/low
KQF_method1_high_low = KQF_method1_experimental_path4[bigg_reorder_p3]
KQF_method2_high_low = KQF_method2_experimental_path4[bigg_reorder_p3]

dg_method1_high_low = -np.log(KQF_method1_high_low)
dg_method2_high_low = -np.log(KQF_method2_high_low)

KQF_method_RL_high_low = KQF_methodRL_experimental_path4[bigg_reorder_p3]
KQF_method_RL_noconstraint_high_low = KQF_methodRL_noconstraint_experimental_path4[bigg_reorder_p3]

dg_methodRL_high_low = -np.log(KQF_method_RL_high_low)
dg_methodRL_noconstraint_high_low = -np.log(KQF_method_RL_noconstraint_high_low)

#high/high no pfk
KQF_method1_high_high_nopfk = KQF_method1_experimental_path5[bigg_reorder_p4]
KQF_method2_high_high_nopfk = KQF_method2_experimental_path5[bigg_reorder_p4]

dg_method1_high_high_nopfk = -np.log(KQF_method1_high_high_nopfk)
dg_method2_high_high_nopfk = -np.log(KQF_method2_high_high_nopfk)

KQF_method_RL_high_high_nopfk = KQF_methodRL_experimental_path5[bigg_reorder_p4]
KQF_method_RL_noconstraint_high_high_nopfk = KQF_methodRL_noconstraint_experimental_path5[bigg_reorder_p4]

dg_methodRL_high_high_nopfk = -np.log(KQF_method_RL_high_high_nopfk)
dg_methodRL_noconstraint_high_high_nopfk = -np.log(KQF_method_RL_noconstraint_high_high_nopfk)



temp_table = np.vstack((bigg_reaction_abbr_p3[bigg_reorder_p3],
                        dg_method1_high_low, dg_method2_high_low,dg_methodRL_noconstraint_high_low,
                        dg_method1_high_high, dg_method2_high_high, dg_methodRL_noconstraint_high_high,
                        dg_method1_high_high_nopfk,   dg_method2_high_high_nopfk, dg_methodRL_noconstraint_high_high_nopfk))
#%%

import pandas as pd

## convert your array into a dataframe
df = pd.DataFrame (temp_table.T)

filepath = cwd+'\\energies.xlsx'

df.to_excel(filepath, index=False)


#%% Now for activity

#high/high
act_method1_high_high = activity_method1_experimental_path3[bigg_reorder_p3]
act_method2_high_high = activity_method2_experimental_path3[bigg_reorder_p3]

act_methodRL_high_high = activity_methodRL_experimental_path3[bigg_reorder_p3]
act_methodRL_noconstraint_high_high = activity_methodRL_noconstraint_experimental_path3[bigg_reorder_p3]

#high/low

act_method1_high_low = activity_method1_experimental_path4[bigg_reorder_p3]
act_method2_high_low = activity_method2_experimental_path4[bigg_reorder_p3]

act_methodRL_high_low = activity_methodRL_experimental_path4[bigg_reorder_p3]
act_methodRL_noconstraint_high_low = activity_methodRL_noconstraint_experimental_path4[bigg_reorder_p3]



#high/high no pfk

act_method1_high_high_nopfk = activity_method1_experimental_path5[bigg_reorder_p4]
act_method2_high_high_nopfk = activity_method2_experimental_path5[bigg_reorder_p4]

act_methodRL_high_high_nopfk = activity_methodRL_experimental_path5[bigg_reorder_p4]
act_methodRL_noconstraint_high_high_nopfk = activity_methodRL_noconstraint_experimental_path5[bigg_reorder_p4]


temp_table_act = np.vstack((bigg_reaction_abbr_p3[bigg_reorder_p3],
                            act_method1_high_low, act_method2_high_low, act_methodRL_noconstraint_high_low,
                        act_method1_high_high, act_method2_high_high, act_methodRL_noconstraint_high_high,
                        act_method1_high_high_nopfk,  act_method2_high_high_nopfk, act_methodRL_noconstraint_high_high_nopfk))
#%%

import pandas as pd

## convert your array into a dataframe
df_act = pd.DataFrame (temp_table_act.T)

filepath = cwd+'\\activities.xlsx'

df_act.to_excel(filepath, index=False)


#%% Now for activity

#high/high
flux_method1_high_high = flux_method1_experimental_path3[bigg_reorder_p3]
flux_method2_high_high = flux_method2_experimental_path3[bigg_reorder_p3]

flux_methodRL_high_high = flux_methodRL_experimental_path3[bigg_reorder_p3]
flux_methodRL_noconstraint_high_high = flux_methodRL_noconstraint_experimental_path3[bigg_reorder_p3]

#high/low
flux_method1_high_low = flux_method1_experimental_path4[bigg_reorder_p3]
flux_method2_high_low = flux_method2_experimental_path4[bigg_reorder_p3]

flux_methodRL_high_low = flux_methodRL_experimental_path4[bigg_reorder_p3]
flux_methodRL_noconstraint_high_low = flux_methodRL_noconstraint_experimental_path4[bigg_reorder_p3]


#high/high no pfk
flux_method1_high_high_nopfk = flux_method1_experimental_path5[bigg_reorder_p4]
flux_method2_high_high_nopfk = flux_method2_experimental_path5[bigg_reorder_p4]

flux_methodRL_high_high_nopfk = flux_methodRL_experimental_path5[bigg_reorder_p4]
flux_methodRL_noconstraint_high_high_nopfk = flux_methodRL_noconstraint_experimental_path5[bigg_reorder_p4]


temp_table_flux = np.vstack((bigg_reaction_abbr_p3[bigg_reorder_p3],
                        flux_method1_high_low,  flux_method2_high_low, flux_methodRL_noconstraint_high_low,
                        flux_method1_high_high, flux_method2_high_high, flux_methodRL_noconstraint_high_high,
                        flux_method1_high_high_nopfk,  flux_method2_high_high_nopfk, flux_methodRL_noconstraint_high_high_nopfk))
#%%

import pandas as pd

## convert your array into a dataframe
df_flux = pd.DataFrame (temp_table_flux.T)

filepath_flux = cwd+'\\flux_data.xlsx'

df_flux.to_excel(filepath_flux, index=False)

#%%
KQF_method1 = KQF_method1_experimental_path2[bigg_reorder_p2]
KQF_method2 = KQF_method2_experimental_path2[bigg_reorder_p2]

dg_method1 = -np.log(KQF_method1)
dg_method2 = -np.log(KQF_method2)

KQF_method_RL = KQF_methodRL_experimental_path2[bigg_reorder_p2]
KQF_method_RL_noconstraint = KQF_methodRL_noconstraint_experimental_path2[bigg_reorder_p2]

dg_methodRL = -np.log(KQF_method_RL)
dg_methodRL_noconstraint = -np.log(KQF_method_RL_noconstraint)



temp_table = np.vstack((bigg_reaction_abbr_p2[bigg_reorder_p2],
                        dg_method1, dg_method2,dg_methodRL_noconstraint))
#%%

import pandas as pd

## convert your array into a dataframe
df = pd.DataFrame (temp_table.T)

filepath = cwd+'\\energies_gly.xlsx'

df.to_excel(filepath, index=False)

#%%
flux_method1 = flux_method1_experimental_path2[bigg_reorder_p2]
flux_method2 = flux_method2_experimental_path2[bigg_reorder_p2]

flux_method_RL_noconstraint = flux_methodRL_noconstraint_experimental_path2[bigg_reorder_p2]


temp_table = np.vstack((bigg_reaction_abbr_p2[bigg_reorder_p2],
                        flux_method1, flux_method2,flux_method_RL_noconstraint))
#%%

import pandas as pd

## convert your array into a dataframe
df = pd.DataFrame (temp_table.T)

filepath = cwd+'\\flux_gly.xlsx'

df.to_excel(filepath, index=False)

#%%
act_method1 = activity_method1_experimental_path2[bigg_reorder_p2]
act_method2 = activity_method2_experimental_path2[bigg_reorder_p2]

act_method_RL_noconstraint = activity_methodRL_noconstraint_experimental_path2[bigg_reorder_p2]


temp_table = np.vstack((bigg_reaction_abbr_p2[bigg_reorder_p2],
                        act_method1, act_method2,act_method_RL_noconstraint))
#%%

import pandas as pd

## convert your array into a dataframe
df = pd.DataFrame (temp_table.T)

filepath = cwd+'\\act_gly.xlsx'

df.to_excel(filepath, index=False)


#%%
KQF_method1 = KQF_method1_experimental_path1[bigg_reorder_p1]
KQF_method2 = KQF_method2_experimental_path1[bigg_reorder_p1]

dg_method1 = -np.log(KQF_method1)
dg_method2 = -np.log(KQF_method2)

KQF_method_RL_noconstraint = KQF_methodRL_noconstraint_experimental_path1[bigg_reorder_p1]

dg_methodRL_noconstraint = -np.log(KQF_method_RL_noconstraint)



temp_table = np.vstack((bigg_reaction_abbr_p1[bigg_reorder_p1],
                        dg_method1, dg_method2,dg_methodRL_noconstraint))
#%%

import pandas as pd

## convert your array into a dataframe
df = pd.DataFrame (temp_table.T)

filepath = cwd+'\\energies_gluco.xlsx'

df.to_excel(filepath, index=False)

#%%
flux_method1 = flux_method1_experimental_path1[bigg_reorder_p1]
flux_method2 = flux_method2_experimental_path1[bigg_reorder_p1]

flux_method_RL_noconstraint = flux_methodRL_noconstraint_experimental_path1[bigg_reorder_p1]


temp_table = np.vstack((bigg_reaction_abbr_p1[bigg_reorder_p1],
                        flux_method1, flux_method2,flux_method_RL_noconstraint))
#%%

import pandas as pd

## convert your array into a dataframe
df = pd.DataFrame (temp_table.T)

filepath = cwd+'\\flux_gluco.xlsx'

df.to_excel(filepath, index=False)

#%%
act_method1 = activity_method1_experimental_path1[bigg_reorder_p1]
act_method2 = activity_method2_experimental_path1[bigg_reorder_p1]

act_method_RL_noconstraint = activity_methodRL_noconstraint_experimental_path1[bigg_reorder_p1]


temp_table = np.vstack((bigg_reaction_abbr_p1[bigg_reorder_p1],
                        act_method1, act_method2,act_method_RL_noconstraint))
#%%

import pandas as pd

## convert your array into a dataframe
df = pd.DataFrame (temp_table.T)

filepath = cwd+'\\act_gluco.xlsx'

df.to_excel(filepath, index=False)