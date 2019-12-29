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
Temperature = 298.15
R_gas = 8.314e-03
RT = R_gas*Temperature

pathway_choice=2

use_exp_data=1

lr1=str('0.0001')
lr2=str('1e-05')
lr3=str('1e-06')
lrs=[lr1,lr2,lr3]

Fontsize_Title=20
Fontsize_Sub = 15
Fontsize_Leg = 15
figure_norm=10


cwd = 'C:\\Users\\samuel_britton\\Documents\\cannon\\Final_Pathways_Publication'

p1_string = '\\GLUCONEOGENESIS\\models_final_data'

p2_string = '\\GLYCOLYSIS_TCA_GOGAT\\models_final_data'
p3_string = '\\TCA_PPP_GLYCOLYSIS_GOGAT\\models_final_data'
p4_string = '\\TCA_PPP_GLYCOLYSIS_GOGAT_LOWERCONC\\models_final_data'
p5_string = '\\TCA_PPP_GLYCOLYSIS_GOGAT_NOPFK\\models_final_data'


p6_string = '\\GLYCOLYSIS_TCA_GOGAT_noconstraint\\models_final_data'
p7_string = '\\TCA_PPP_GLYCOLYSIS_GOGAT_noconstraint\\models_final_data'
p8_string = '\\TCA_PPP_GLYCOLYSIS_GOGAT_LOWERCONC_noconstraint\\models_final_data'
p9_string = '\\TCA_PPP_GLYCOLYSIS_GOGAT_NOPFK_noconstraint\\models_final_data'


bigg_reaction_abbr_p1 = pd.Series(['GLUCOSE_6_PHOSPHATASE', 'PGI', 'FBP', 'FBA', 'TPI', 'GAPD', 'PGK',
       'PGM', 'ENO', 'PEP_Carboxykinase', 'Pyruvate_Carboxylase'])

bigg_reorder_p1 = [0,1,2,3,4,5,6,7,8,9,10]


bigg_reaction_abbr_p2 = pd.Series(['CSM','ACONT', 'ICDH', 'AKGD', 'SUCOAS', 'SUCD', 'FUM',
       'MDH', 'GAPD', 'PGK', 'TPI', 'FBA', 'PYK', 'PGM', 'ENO', 'HEX1', 'PGI',
       'PFK', 'PYRt2', 'PDH', 'GOGAT'])

bigg_reorder_p2 = [15, 16, 17, 11, 10, 8, 9, 13, 14, 12, 18, 19, 0, 1, 2, 3, 4, 5, 6, 7, 20]

bigg_reaction_abbr_p3 = pd.Series(['CSM','ACONT','ICDH','AKGD','SUCOAS',
        'SUCD','FUM','MDH','GAPD','PGK','TPI','FBA','PYK','PGM','ENO',
        'HEX1','PGI','PFK','PYRt2M','PDH','G6PDH','PGL','GND','RPE','RPI',
        'TKT2','TALA','TKT1','GOGAT'])

bigg_reorder_p3 = [15, 16, 17, 11, 10, 8, 9, 13, 14, 12, 18, 19, 
                   20, 21, 22, 24, 23, 27, 26, 25,
                   0, 1, 2, 3, 4, 5, 6, 7, 28]


bigg_reaction_abbr_p5 = pd.Series(['CSM','ACONT','ICDH','AKGD','SUCOAS',
        'SUCD','FUM','MDH','GAPD','PGK','TPI','FBA','PYK','PGM','ENO',
        'HEX1','PGI','PFK','PYRt2M','PDH','G6PDH','PGL','GND','RPE','RPI',
        'TKT2','TALA','TKT1','GOGAT'])

bigg_reorder_p5 = [15, 16, 17, 11, 10, 8, 9, 13, 14, 12, 18, 19, 
                   20, 21, 22, 24, 23, 27, 26, 25,
                   0, 1, 2, 3, 4, 5, 6, 7, 28]


epr_method1_experimental_path1 = 0.96305606
epr_method2_experimental_path1 = 0.96305606
#activity_method2_experimental_path2=1
activity_method1_experimental_path1 = np.loadtxt(cwd+'\\GLUCONEOGENESIS\\'+'activities_experiment_method1.txt', dtype=float)
activity_method2_experimental_path1 = np.loadtxt(cwd+'\\GLUCONEOGENESIS\\'+'activities_experiment_method2.txt', dtype=float)
flux_method1_experimental_path1 = np.loadtxt(cwd+'\\GLUCONEOGENESIS\\'+'flux_experiment_method1.txt', dtype=float)
flux_method2_experimental_path1 = np.loadtxt(cwd+'\\GLUCONEOGENESIS\\'+'flux_experiment_method2.txt', dtype=float)
KQF_method1_experimental_path1 = np.loadtxt(cwd+'\\GLUCONEOGENESIS\\'+'KQF_experiment_method1.txt', dtype=float)
KQF_method2_experimental_path1 = np.loadtxt(cwd+'\\GLUCONEOGENESIS\\'+'KQF_experiment_method2.txt', dtype=float)


epr_method1_experimental_path2 = 4.1392
epr_method2_experimental_path2 = 4.13458403
#activity_method2_experimental_path2=1
activity_method1_experimental_path2 = np.loadtxt(cwd+'\\GLYCOLYSIS_TCA_GOGAT\\'+'activities_experiment_method1.txt', dtype=float)
activity_method2_experimental_path2 = np.loadtxt(cwd+'\\GLYCOLYSIS_TCA_GOGAT\\'+'activities_experiment_method2.txt', dtype=float)
flux_method1_experimental_path2 = np.loadtxt(cwd+'\\GLYCOLYSIS_TCA_GOGAT\\'+'flux_experiment_method1.txt', dtype=float)
flux_method2_experimental_path2 = np.loadtxt(cwd+'\\GLYCOLYSIS_TCA_GOGAT\\'+'flux_experiment_method2.txt', dtype=float)

KQF_method1_experimental_path2 = np.loadtxt(cwd+'\\GLYCOLYSIS_TCA_GOGAT\\'+'KQF_experiment_method1.txt', dtype=float)
KQF_method2_experimental_path2 = np.loadtxt(cwd+'\\GLYCOLYSIS_TCA_GOGAT\\'+'KQF_experiment_method2.txt', dtype=float)

activity_methodRL_experimental_path2 = np.loadtxt(cwd+'\\GLYCOLYSIS_TCA_GOGAT\\'+'activities_experiment_RL.txt', dtype=float)
activity_methodRL_noconstraint_experimental_path2 = np.loadtxt(cwd+'\\GLYCOLYSIS_TCA_GOGAT\\'+'activities_experiment_RL_noconstraint.txt', dtype=float)
flux_methodRL_experimental_path2 = np.loadtxt(cwd+'\\GLYCOLYSIS_TCA_GOGAT\\'+'flux_experiment_RL.txt', dtype=float)
flux_methodRL_noconstraint_experimental_path2 = np.loadtxt(cwd+'\\GLYCOLYSIS_TCA_GOGAT\\'+'flux_experiment_RL_noconstraint.txt', dtype=float)

KQF_methodRL_experimental_path2 = np.loadtxt(cwd+'\\GLYCOLYSIS_TCA_GOGAT\\'+'KQF_experiment_RL.txt', dtype=float)
KQF_methodRL_noconstraint_experimental_path2 = np.loadtxt(cwd+'\\GLYCOLYSIS_TCA_GOGAT\\'+'KQF_experiment_RL_noconstraint.txt', dtype=float)


epr_method1_experimental_path3 = 4.0677
epr_method2_experimental_path3 = 3.9918
activity_method1_experimental_path3 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'activities_experiment_method1_highhigh.txt', dtype=float)
activity_method2_experimental_path3 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'activities_experiment_method2_highhigh.txt', dtype=float)
flux_method1_experimental_path3 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'flux_experiment_method1_highhigh.txt', dtype=float)
flux_method2_experimental_path3 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'flux_experiment_method2_highhigh.txt', dtype=float)
KQF_method1_experimental_path3 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'KQF_experiment_method1_highhigh.txt', dtype=float)
KQF_method2_experimental_path3 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'KQF_experiment_method2_highhigh.txt', dtype=float)

activity_methodRL_experimental_path3 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'activities_experiment_RL_highhigh.txt', dtype=float)
activity_methodRL_noconstraint_experimental_path3 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'activities_experiment_RL_noconstraint_highhigh.txt', dtype=float)
flux_methodRL_experimental_path3 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'flux_experiment_RL_highhigh.txt', dtype=float)
flux_methodRL_noconstraint_experimental_path3 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'flux_experiment_RL_noconstraint_highhigh.txt', dtype=float)
KQF_methodRL_experimental_path3 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'KQF_experiment_RL_highhigh.txt', dtype=float)
KQF_methodRL_noconstraint_experimental_path3 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'KQF_experiment_RL_noconstraint_highhigh.txt', dtype=float)



epr_method1_experimental_path4 = 3.9249
epr_method2_experimental_path4 = 3.8930
activity_method1_experimental_path4 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'activities_experiment_method1_highlow.txt', dtype=float)
activity_method2_experimental_path4 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'activities_experiment_method2_highlow.txt', dtype=float)
flux_method1_experimental_path4 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'flux_experiment_method1_highlow.txt', dtype=float)
flux_method2_experimental_path4 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'flux_experiment_method2_highlow.txt', dtype=float)

KQF_method1_experimental_path4 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'KQF_experiment_method1_highlow.txt', dtype=float)
KQF_method2_experimental_path4 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'KQF_experiment_method2_highlow.txt', dtype=float)

activity_methodRL_experimental_path4 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT_LOWERCONC\\'+'activities_experiment_RL_highlow.txt', dtype=float)
activity_methodRL_noconstraint_experimental_path4 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT_LOWERCONC\\'+'activities_experiment_RL_noconstraint_highlow.txt', dtype=float)
flux_methodRL_experimental_path4 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT_LOWERCONC\\'+'flux_experiment_RL_highlow.txt', dtype=float)
flux_methodRL_noconstraint_experimental_path4 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT_LOWERCONC\\'+'flux_experiment_RL_noconstraint_highlow.txt', dtype=float)

KQF_methodRL_experimental_path4 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT_LOWERCONC\\'+'KQF_experiment_RL_highlow.txt', dtype=float)
KQF_methodRL_noconstraint_experimental_path4 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT_LOWERCONC\\'+'KQF_experiment_RL_noconstraint_highlow.txt', dtype=float)


epr_method1_experimental_path5 = 3.44467
epr_method2_experimental_path5 = 3.44467
activity_method1_experimental_path5 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'activities_experiment_method1_highhigh_PFKZERO.txt', dtype=float)
activity_method2_experimental_path5 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'activities_experiment_method2_highhigh_PFKZERO.txt', dtype=float)
flux_method1_experimental_path5 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'flux_experiment_method1_highhigh_PFKZERO.txt', dtype=float)
flux_method2_experimental_path5 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'flux_experiment_method2_highhigh_PFKZERO.txt', dtype=float)

KQF_method1_experimental_path5 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'KQF_experiment_method1_highhigh_PFKZERO.txt', dtype=float)
KQF_method2_experimental_path5 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'KQF_experiment_method2_highhigh_PFKZERO.txt', dtype=float)

activity_methodRL_experimental_path5 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'activities_experiment_RL_highhigh_PFKZERO.txt', dtype=float)
activity_methodRL_noconstraint_experimental_path5 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'activities_experiment_RL_noconstraint_highhigh_PFKZERO.txt', dtype=float)
flux_methodRL_experimental_path5 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'flux_experiment_RL_highhigh_PFKZERO.txt', dtype=float)
flux_methodRL_noconstraint_experimental_path5 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'flux_experiment_RL_noconstraint_highhigh_PFKZERO.txt', dtype=float)

KQF_methodRL_experimental_path5 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'KQF_experiment_RL_highhigh_PFKZERO.txt', dtype=float)
KQF_methodRL_noconstraint_experimental_path5 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'KQF_experiment_RL_noconstraint_highhigh_PFKZERO.txt', dtype=float)



if (pathway_choice==1):
    epr_to_plot_method1 = epr_method1_experimental_path1
    epr_to_plot_method2 = epr_method2_experimental_path1
    activity_to_plot_method1=activity_method1_experimental_path1
    activity_to_plot_method2=activity_method2_experimental_path1
    flux_to_plot_method1=flux_method1_experimental_path1
    flux_to_plot_method2=flux_method2_experimental_path1
    KQF_to_plot_method1=KQF_method1_experimental_path1
    KQF_to_plot_method2=KQF_method2_experimental_path1
    Pathway_Name='GLUCONEOGENESIS'
        
    bigg_reorder = bigg_reorder_p1
    p_string = cwd + p1_string
    bigg_reaction_abbr_p = bigg_reaction_abbr_p1
    eps=0.5


if (pathway_choice==2):
    epr_to_plot_method1 = epr_method1_experimental_path2
    epr_to_plot_method2 = epr_method2_experimental_path2
    activity_to_plot_method1=activity_method1_experimental_path2
    activity_to_plot_method2=activity_method2_experimental_path2
    flux_to_plot_method1=flux_method1_experimental_path2
    flux_to_plot_method2=flux_method2_experimental_path2
    
    KQF_to_plot_method1=KQF_method1_experimental_path2
    KQF_to_plot_method2=KQF_method2_experimental_path2
    
    
    activity_to_plot_methodRL = activity_methodRL_experimental_path2
    activity_to_plot_methodRL_noconstraint = activity_methodRL_noconstraint_experimental_path2
    flux_to_plot_methodRL = flux_methodRL_experimental_path2
    flux_to_plot_methodRL_noconstraint = flux_methodRL_noconstraint_experimental_path2
    
    KQF_to_plot_methodRL = KQF_methodRL_experimental_path2
    KQF_to_plot_methodRL_noconstraint = KQF_methodRL_noconstraint_experimental_path2
    Pathway_Name='GLYCOLYSIS_TCA'
        
    bigg_reorder = bigg_reorder_p2
    p_string = cwd + p2_string
    bigg_reaction_abbr_p = bigg_reaction_abbr_p2
    eps=0.5


#TCA_PPP_GLY with high/high
if (pathway_choice==3):
    epr_to_plot_method1 = epr_method1_experimental_path3
    epr_to_plot_method2 = epr_method2_experimental_path3
    activity_to_plot_method1=activity_method1_experimental_path3
    activity_to_plot_method2=activity_method2_experimental_path3
    flux_to_plot_method1=flux_method1_experimental_path3
    flux_to_plot_method2=flux_method2_experimental_path3
    
    KQF_to_plot_method1=KQF_method1_experimental_path3
    KQF_to_plot_method2=KQF_method2_experimental_path3
    
    
    activity_to_plot_methodRL = activity_methodRL_experimental_path3
    activity_to_plot_methodRL_noconstraint = activity_methodRL_noconstraint_experimental_path3
    flux_to_plot_methodRL = flux_methodRL_experimental_path3
    flux_to_plot_methodRL_noconstraint = flux_methodRL_noconstraint_experimental_path3
    KQF_to_plot_methodRL = KQF_methodRL_experimental_path3
    KQF_to_plot_methodRL_noconstraint = KQF_methodRL_noconstraint_experimental_path3
    
    Pathway_Name='GLYCOLYSIS_TCA_PPP, High/High'
        
    bigg_reorder = bigg_reorder_p3
    p_string = cwd + p3_string
    bigg_reaction_abbr_p = bigg_reaction_abbr_p3
    eps=0.2


#TCA_PPP_GLY with high/low
if (pathway_choice==4):
    epr_to_plot_method1 = epr_method1_experimental_path4
    epr_to_plot_method2 = epr_method2_experimental_path4
    activity_to_plot_method1=activity_method1_experimental_path4
    activity_to_plot_method2=activity_method2_experimental_path4
    flux_to_plot_method1=flux_method1_experimental_path4
    flux_to_plot_method2=flux_method2_experimental_path4
    
    KQF_to_plot_method1=KQF_method1_experimental_path4
    KQF_to_plot_method2=KQF_method2_experimental_path4
    
    
    activity_to_plot_methodRL = activity_methodRL_experimental_path4
    activity_to_plot_methodRL_noconstraint = activity_methodRL_noconstraint_experimental_path4
    flux_to_plot_methodRL = flux_methodRL_experimental_path4
    flux_to_plot_methodRL_noconstraint = flux_methodRL_noconstraint_experimental_path4
    
    KQF_to_plot_methodRL = KQF_methodRL_experimental_path4
    KQF_to_plot_methodRL_noconstraint = KQF_methodRL_noconstraint_experimental_path4
    
    Pathway_Name='GLYCOLYSIS_TCA_PPP, High/Low'
        
    
    bigg_reorder = bigg_reorder_p3
    p_string = cwd + p4_string
    
    bigg_reaction_abbr_p = bigg_reaction_abbr_p3
    eps=0.2




#TCA_PPP_GLY no pfk with high/high
if (pathway_choice==5):
    epr_to_plot_method1 = epr_method1_experimental_path5
    epr_to_plot_method2 = epr_method2_experimental_path5
    activity_to_plot_method1=activity_method1_experimental_path5
    activity_to_plot_method2=activity_method2_experimental_path5
    flux_to_plot_method1=flux_method1_experimental_path5
    flux_to_plot_method2=flux_method2_experimental_path5
    
    KQF_to_plot_method1=KQF_method1_experimental_path5
    KQF_to_plot_method2=KQF_method2_experimental_path5
    
    
    activity_to_plot_methodRL = activity_methodRL_experimental_path5
    activity_to_plot_methodRL_noconstraint = activity_methodRL_noconstraint_experimental_path5
    flux_to_plot_methodRL = flux_methodRL_experimental_path5
    flux_to_plot_methodRL_noconstraint = flux_methodRL_noconstraint_experimental_path5
    KQF_to_plot_methodRL = KQF_methodRL_experimental_path5
    KQF_to_plot_methodRL_noconstraint = KQF_methodRL_noconstraint_experimental_path5
    
    Pathway_Name='GLYCOLYSIS_TCA_PPP No PFK, High/High'
        
    bigg_reorder = bigg_reorder_p5
    p_string = cwd + p5_string
    bigg_reaction_abbr_p = bigg_reaction_abbr_p5
    eps=0.2
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
                color=color3,
                scatter_kws={"s": 100})


sns.regplot(x=np.array([0]), y=np.array([activity_method1[0]]), scatter=True, fit_reg=False, marker='x',
            ax=ax2,
            label='',
            color=color3,
            scatter_kws={"s": 100})
for i in range(1,len(bigg_reorder)):
    sns.regplot(x=np.array([i]), y=np.array([activity_method1[i]]), scatter=True, fit_reg=False, marker='x',
                ax=ax2,
                color=color3,
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
    
fig.suptitle(Pathway_Name,fontsize=Fontsize_Title)

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

colorChoice = sns.xkcd_rgb["black"]

color1 = sns.xkcd_rgb["slate grey"]
color2 = sns.xkcd_rgb["grey"]
color3 = sns.xkcd_rgb["steel grey"]

fig = plt.figure(figsize=(figure_norm, 0.5*figure_norm))
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

flux_method1 = flux_to_plot_method1[bigg_reorder]
flux_method2 = flux_to_plot_method2[bigg_reorder]

flux_method_RL = flux_to_plot_methodRL[bigg_reorder]
flux_method_RL_noconstraint = flux_to_plot_methodRL_noconstraint[bigg_reorder]



#method 1 vs RLnoconstraint
sns.regplot(x=np.array([0]), y=np.array([flux_method1[0]]), scatter=True, fit_reg=False, marker='x',
            ax=ax1,
            label='Method 1',
            color='r',
            scatter_kws={"s": 100})


for i in range(1,len(bigg_reorder)):
    sns.regplot(x=np.array([i]), y=np.array([flux_method1[i]]), scatter=True, fit_reg=False, marker='x',
                ax=ax1,
                color='r',
                scatter_kws={"s": 100})
 
sns.regplot(x=np.array([0]), y=np.array([flux_method_RL_noconstraint[0]]), scatter=True, fit_reg=False, marker='o',
            ax=ax1,
            label='RL Unconstrained',
            color=color3,
            scatter_kws={"s": 100})
for i in range(1,len(bigg_reorder)):
    sns.regplot(x=np.array([i]), y=np.array([flux_method_RL_noconstraint[i]]), scatter=True, fit_reg=False, marker='o',
                ax=ax1,
                color=color3,
                scatter_kws={"s": 100})

#method 2 vs RL
sns.regplot(x=np.array([0]), y=np.array([flux_method2[0]]), scatter=True, fit_reg=False, marker='x',
            ax=ax2,
            label='Method 2',
            color='b',
            scatter_kws={"s": 100})
for i in range(1,len(bigg_reorder)):
    sns.regplot(x=np.array([i]), y=np.array([flux_method2[i]]), scatter=True, fit_reg=False, marker='x',
                ax=ax2,
                color='b',
                scatter_kws={"s": 100})
    

sns.regplot(x=np.array([0]), y=np.array([flux_method_RL[0]]), scatter=True, fit_reg=False, marker='s',
            ax=ax2,
            label='RL Constrained',
            color=color3,
            scatter_kws={"s": 100})
for i in range(1,len(bigg_reorder)):
    sns.regplot(x=np.array([i]), y=np.array([flux_method_RL[i]]), scatter=True, fit_reg=False, marker='s',
                ax=ax2,
                color=color3,
                scatter_kws={"s": 100})



fig.suptitle(Pathway_Name,fontsize=Fontsize_Title)

ax2.set_xticks( [i for i in range(0,len(bigg_reorder)+1)] )
ax2.set_xticklabels(bigg_reaction_abbr_p[bigg_reorder], rotation='vertical', fontsize=Fontsize_Sub)

ax2.set_xlabel('Reactions',fontsize=Fontsize_Sub)
ax1.set_ylabel('Flux',fontsize=Fontsize_Sub)
ax2.set_ylabel('Flux',fontsize=Fontsize_Sub)

ax1.legend(fontsize=Fontsize_Leg, loc='best')
ax2.legend(fontsize=Fontsize_Leg, loc='best')
ax1.annotate("A", xy=(-0.1, 1.0), xycoords="axes fraction",fontsize=Fontsize_Sub)
ax2.annotate("B", xy=(-0.1, 1.0), xycoords="axes fraction",fontsize=Fontsize_Sub)

#%% simple activity plot

colorChoice = sns.xkcd_rgb["black"]

color1 = sns.xkcd_rgb["slate grey"]
color2 = sns.xkcd_rgb["grey"]
color3 = sns.xkcd_rgb["steel grey"]

fig = plt.figure(figsize=(figure_norm, 0.5*figure_norm))
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

activity_method1 = activity_to_plot_method1[bigg_reorder]
activity_method2 = activity_to_plot_method2[bigg_reorder]

activity_method_RL = activity_to_plot_methodRL[bigg_reorder]
activity_method_RL_noconstraint = activity_to_plot_methodRL_noconstraint[bigg_reorder]



#method 1 vs RLnoconstraint
sns.regplot(x=np.array([0]), y=np.array([activity_method1[0]]), scatter=True, fit_reg=False, marker='x',
            ax=ax1,
            label='Method 1',
            color='r',
            scatter_kws={"s": 100})


for i in range(1,len(bigg_reorder)):
    sns.regplot(x=np.array([i]), y=np.array([activity_method1[i]]), scatter=True, fit_reg=False, marker='x',
                ax=ax1,
                color='r',
                scatter_kws={"s": 100})
 
sns.regplot(x=np.array([0]), y=np.array([activity_method_RL_noconstraint[0]]), scatter=True, fit_reg=False, marker='o',
            ax=ax1,
            label='RL Unconstrained',
            color=color3,
            scatter_kws={"s": 100})
for i in range(1,len(bigg_reorder)):
    sns.regplot(x=np.array([i]), y=np.array([activity_method_RL_noconstraint[i]]), scatter=True, fit_reg=False, marker='o',
                ax=ax1,
                color=color3,
                scatter_kws={"s": 100})

#method 2 vs RL
sns.regplot(x=np.array([0]), y=np.array([activity_method2[0]]), scatter=True, fit_reg=False, marker='x',
            ax=ax2,
            label='Method 2',
            color='b',
            scatter_kws={"s": 100})
for i in range(1,len(bigg_reorder)):
    sns.regplot(x=np.array([i]), y=np.array([activity_method2[i]]), scatter=True, fit_reg=False, marker='x',
                ax=ax2,
                color='b',
                scatter_kws={"s": 100})
    

sns.regplot(x=np.array([0]), y=np.array([activity_method_RL[0]]), scatter=True, fit_reg=False, marker='s',
            ax=ax2,
            label='RL Constrained',
            color=color3,
            scatter_kws={"s": 100})
for i in range(1,len(bigg_reorder)):
    sns.regplot(x=np.array([i]), y=np.array([activity_method_RL[i]]), scatter=True, fit_reg=False, marker='s',
                ax=ax2,
                color=color3,
                scatter_kws={"s": 100})



fig.suptitle(Pathway_Name,fontsize=Fontsize_Title)

ax2.set_xticks( [i for i in range(0,len(bigg_reorder)+1)] )
ax2.set_xticklabels(bigg_reaction_abbr_p[bigg_reorder], rotation='vertical', fontsize=Fontsize_Sub)

ax2.set_xlabel('Reactions',fontsize=Fontsize_Sub)
ax1.set_ylabel('Enzyme Activity',fontsize=Fontsize_Sub)
ax2.set_ylabel('Enzyme Activity',fontsize=Fontsize_Sub)

ax1.legend(fontsize=Fontsize_Leg, loc='best')
ax2.legend(fontsize=Fontsize_Leg, loc='best')
ax1.annotate("A", xy=(-0.1, 1.0), xycoords="axes fraction",fontsize=Fontsize_Sub)
ax2.annotate("B", xy=(-0.1, 1.0), xycoords="axes fraction",fontsize=Fontsize_Sub)



#%% simple energy plot

colorChoice = sns.xkcd_rgb["black"]

color1 = sns.xkcd_rgb["slate grey"]
color2 = sns.xkcd_rgb["grey"]
color3 = sns.xkcd_rgb["steel grey"]

fig = plt.figure(figsize=(figure_norm, 0.5*figure_norm))
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

KQF_method1 = KQF_to_plot_method1[bigg_reorder]
KQF_method2 = KQF_to_plot_method2[bigg_reorder]

dg_method1 = -np.log(KQF_method1)
dg_method2 = -np.log(KQF_method2)

KQF_method_RL = KQF_to_plot_methodRL[bigg_reorder]
KQF_method_RL_noconstraint = KQF_to_plot_methodRL_noconstraint[bigg_reorder]

dg_methodRL = -np.log(KQF_method_RL)
dg_methodRL_noconstraint = -np.log(KQF_method_RL_noconstraint)


#method 1 vs RLnoconstraint
sns.regplot(x=np.array([0]), y=np.array([dg_method1[0]]), scatter=True, fit_reg=False, marker='x',
            ax=ax1,
            label='Method 1',
            color='r',
            scatter_kws={"s": 100})


for i in range(1,len(bigg_reorder)):
    sns.regplot(x=np.array([i]), y=np.array([dg_method1[i]]), scatter=True, fit_reg=False, marker='x',
                ax=ax1,
                color='r',
                scatter_kws={"s": 100})
 
sns.regplot(x=np.array([0]), y=np.array([dg_methodRL_noconstraint[0]]), scatter=True, fit_reg=False, marker='o',
            ax=ax1,
            label='RL Unconstrained',
            color=color3,
            scatter_kws={"s": 100})
for i in range(1,len(bigg_reorder)):
    sns.regplot(x=np.array([i]), y=np.array([dg_methodRL_noconstraint[i]]), scatter=True, fit_reg=False, marker='o',
                ax=ax1,
                color=color3,
                scatter_kws={"s": 100})

#method 2 vs RL
sns.regplot(x=np.array([0]), y=np.array([dg_method2[0]]), scatter=True, fit_reg=False, marker='x',
            ax=ax2,
            label='Method 2',
            color='b',
            scatter_kws={"s": 100})
for i in range(1,len(bigg_reorder)):
    sns.regplot(x=np.array([i]), y=np.array([dg_method2[i]]), scatter=True, fit_reg=False, marker='x',
                ax=ax2,
                color='b',
                scatter_kws={"s": 100})
    

sns.regplot(x=np.array([0]), y=np.array([dg_methodRL[0]]), scatter=True, fit_reg=False, marker='s',
            ax=ax2,
            label='RL Constrained',
            color=color3,
            scatter_kws={"s": 100})
for i in range(1,len(bigg_reorder)):
    sns.regplot(x=np.array([i]), y=np.array([dg_methodRL[i]]), scatter=True, fit_reg=False, marker='s',
                ax=ax2,
                color=color3,
                scatter_kws={"s": 100})



fig.suptitle(Pathway_Name,fontsize=Fontsize_Title)

ax2.set_xticks( [i for i in range(0,len(bigg_reorder)+1)] )
ax2.set_xticklabels(bigg_reaction_abbr_p[bigg_reorder], rotation='vertical', fontsize=Fontsize_Sub)

ax2.set_xlabel('Reactions',fontsize=Fontsize_Sub)
ax1.set_ylabel(r'$\Delta{G}/RT$',fontsize=Fontsize_Sub)
ax2.set_ylabel(r'$\Delta{G}/RT$',fontsize=Fontsize_Sub)

ax1.legend(fontsize=Fontsize_Leg, loc='best')
ax2.legend(fontsize=Fontsize_Leg, loc='best')

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
KQF_method1_high_high_nopfk = KQF_method1_experimental_path5[bigg_reorder_p5]
KQF_method2_high_high_nopfk = KQF_method2_experimental_path5[bigg_reorder_p5]

dg_method1_high_high_nopfk = -np.log(KQF_method1_high_high_nopfk)
dg_method2_high_high_nopfk = -np.log(KQF_method2_high_high_nopfk)

KQF_method_RL_high_high_nopfk = KQF_methodRL_experimental_path5[bigg_reorder_p5]
KQF_method_RL_noconstraint_high_high_nopfk = KQF_methodRL_noconstraint_experimental_path5[bigg_reorder_p5]

dg_methodRL_high_high_nopfk = -np.log(KQF_method_RL_high_high_nopfk)
dg_methodRL_noconstraint_high_high_nopfk = -np.log(KQF_method_RL_noconstraint_high_high_nopfk)



temp_table = np.vstack((dg_methodRL_noconstraint_high_low, dg_method1_high_low, dg_methodRL_high_low, dg_method2_high_low,
                        dg_methodRL_noconstraint_high_high, dg_method1_high_high, dg_methodRL_high_high, dg_method2_high_high,
                        dg_methodRL_noconstraint_high_high_nopfk,dg_method1_high_high_nopfk, dg_methodRL_high_high_nopfk,  dg_method2_high_high_nopfk))
#%%

import pandas as pd

## convert your array into a dataframe
df = pd.DataFrame (temp_table.T)

filepath = cwd+'\\energies.xlsx'

df.to_excel(filepath, index=False)


#%% Now for activity

#high/high
act_method1_high_high = activity_method1_experimental_path3[bigg_reorder_p3]
act_method2_high_high = activity_method1_experimental_path3[bigg_reorder_p3]

act_methodRL_high_high = activity_methodRL_experimental_path3[bigg_reorder_p3]
act_methodRL_noconstraint_high_high = activity_methodRL_noconstraint_experimental_path3[bigg_reorder_p3]

#high/low

act_method1_high_low = activity_method1_experimental_path4[bigg_reorder_p3]
act_method2_high_low = activity_method1_experimental_path4[bigg_reorder_p3]

act_methodRL_high_low = activity_methodRL_experimental_path4[bigg_reorder_p3]
act_methodRL_noconstraint_high_low = activity_methodRL_noconstraint_experimental_path4[bigg_reorder_p3]



#high/high no pfk

act_method1_high_high_nopfk = activity_method1_experimental_path5[bigg_reorder_p5]
act_method2_high_high_nopfk = activity_method1_experimental_path5[bigg_reorder_p5]

act_methodRL_high_high_nopfk = activity_methodRL_experimental_path5[bigg_reorder_p5]
act_methodRL_noconstraint_high_high_nopfk = activity_methodRL_noconstraint_experimental_path5[bigg_reorder_p5]


temp_table_act = np.vstack((act_methodRL_noconstraint_high_low, act_method1_high_low, act_methodRL_high_low, act_method2_high_low,
                        act_methodRL_noconstraint_high_high, act_method1_high_high, act_methodRL_high_high, act_method2_high_high,
                        act_methodRL_noconstraint_high_high_nopfk, act_method1_high_high_nopfk, act_methodRL_high_high_nopfk,  act_method2_high_high_nopfk))
#%%

import pandas as pd

## convert your array into a dataframe
df_act = pd.DataFrame (temp_table_act.T)

filepath = cwd+'\\activities.xlsx'

df_act.to_excel(filepath, index=False)


#%% Now for activity

#high/high
flux_method1_high_high = flux_method1_experimental_path3[bigg_reorder_p3]
flux_method2_high_high = flux_method1_experimental_path3[bigg_reorder_p3]

flux_methodRL_high_high = flux_methodRL_experimental_path3[bigg_reorder_p3]
flux_methodRL_noconstraint_high_high = flux_methodRL_noconstraint_experimental_path3[bigg_reorder_p3]

#high/low
flux_method1_high_low = flux_method1_experimental_path4[bigg_reorder_p3]
flux_method2_high_low = flux_method1_experimental_path4[bigg_reorder_p3]

flux_methodRL_high_low = flux_methodRL_experimental_path4[bigg_reorder_p3]
flux_methodRL_noconstraint_high_low = flux_methodRL_noconstraint_experimental_path4[bigg_reorder_p3]


#high/high no pfk
flux_method1_high_high_nopfk = flux_method1_experimental_path5[bigg_reorder_p5]
flux_method2_high_high_nopfk = flux_method1_experimental_path5[bigg_reorder_p5]

flux_methodRL_high_high_nopfk = flux_methodRL_experimental_path5[bigg_reorder_p5]
flux_methodRL_noconstraint_high_high_nopfk = flux_methodRL_noconstraint_experimental_path5[bigg_reorder_p5]


temp_table_flux = np.vstack((flux_methodRL_noconstraint_high_low, flux_method1_high_low, flux_methodRL_high_low, flux_method2_high_low,
                        flux_methodRL_noconstraint_high_high, flux_method1_high_high, flux_methodRL_high_high, flux_method2_high_high,
                        flux_methodRL_noconstraint_high_high_nopfk, flux_method1_high_high_nopfk, flux_methodRL_high_high_nopfk,  flux_method2_high_high_nopfk))
#%%

import pandas as pd

## convert your array into a dataframe
df_flux = pd.DataFrame (temp_table_flux.T)

filepath_flux = cwd+'\\flux_data.xlsx'

df_flux.to_excel(filepath_flux, index=False)
