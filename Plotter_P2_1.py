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

pathway_choice=3
use_exp_data=1

lr1=str('0.0001')
lr2=str('1e-05')
lr3=str('1e-06')
lrs=[lr1,lr2,lr3]
lrs=[lr1,lr2]

Fontsize_Title=20
Fontsize_Sub = 15
Fontsize_Leg = 15
figure_norm=10


cwd = 'C:\\Users\\samuel_britton\\Documents\\cannon\\Final_Pathways_Publication'



p1_string = '\\GLUCONEOGENESIS\\models_final_data'

#This is R2
p2_string = '\\GLYCOLYSIS_TCA_GOGAT\\models_final_data'
p3_string = '\\TCA_PPP_GLYCOLYSIS_GOGAT\\models_final_data'
p4_string = '\\TCA_PPP_GLYCOLYSIS_GOGAT_LOWERCONC\\models_final_data'
p5_string = '\\TCA_PPP_GLYCOLYSIS_GOGAT_NOPFK\\models_final_data_altarch' #changed for different NN architecture

#this is R1
p6_string = '\\GLYCOLYSIS_TCA_GOGAT_noconstraint\\models_final_data'
p7_string = '\\TCA_PPP_GLYCOLYSIS_GOGAT_noconstraint\\models_final_data'
p8_string = '\\TCA_PPP_GLYCOLYSIS_GOGAT_LOWERCONC_noconstraint\\models_final_data'
p9_string = '\\TCA_PPP_GLYCOLYSIS_GOGAT_NOPFK_noconstraint\\models_final_data'



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

epr_method1_experimental_path1 = 0.96305606
epr_method2_experimental_path1 = 0.96305606
#activity_method2_experimental_path2=1
activity_method1_experimental_path1 = np.loadtxt(cwd+'\\GLUCONEOGENESIS\\'+'activities_experiment_method1.txt', dtype=float)
activity_method2_experimental_path1 = np.loadtxt(cwd+'\\GLUCONEOGENESIS\\'+'activities_experiment_method2.txt', dtype=float)
flux_method1_experimental_path1 = np.loadtxt(cwd+'\\GLUCONEOGENESIS\\'+'flux_experiment_method1.txt', dtype=float)
flux_method2_experimental_path1 = np.loadtxt(cwd+'\\GLUCONEOGENESIS\\'+'flux_experiment_method2.txt', dtype=float)


epr_method1_experimental_path2 = 4.1392
epr_method2_experimental_path2 = 4.13458403
#activity_method2_experimental_path2=1
activity_method1_experimental_path2 = np.loadtxt(cwd+'\\GLYCOLYSIS_TCA_GOGAT\\'+'activities_experiment_method1.txt', dtype=float)
activity_method2_experimental_path2 = np.loadtxt(cwd+'\\GLYCOLYSIS_TCA_GOGAT\\'+'activities_experiment_method2.txt', dtype=float)
flux_method1_experimental_path2 = np.loadtxt(cwd+'\\GLYCOLYSIS_TCA_GOGAT\\'+'flux_experiment_method1.txt', dtype=float)
flux_method2_experimental_path2 = np.loadtxt(cwd+'\\GLYCOLYSIS_TCA_GOGAT\\'+'flux_experiment_method2.txt', dtype=float)


epr_method1_experimental_path3 = 4.0677
epr_method2_experimental_path3 = 3.9918
activity_method1_experimental_path3 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'activities_experiment_method1_highhigh.txt', dtype=float)
activity_method2_experimental_path3 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'activities_experiment_method2_highhigh.txt', dtype=float)
flux_method1_experimental_path3 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'flux_experiment_method1_highhigh.txt', dtype=float)
flux_method2_experimental_path3 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'flux_experiment_method2_highhigh.txt', dtype=float)


epr_method1_experimental_path4 = 3.9249
epr_method2_experimental_path4 = 3.8930
activity_method1_experimental_path4 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'activities_experiment_method1_highlow.txt', dtype=float)
activity_method2_experimental_path4 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'activities_experiment_method2_highlow.txt', dtype=float)
flux_method1_experimental_path4 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'flux_experiment_method1_highlow.txt', dtype=float)
flux_method2_experimental_path4 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'flux_experiment_method2_highlow.txt', dtype=float)

epr_method1_experimental_path5 = 3.44467
epr_method2_experimental_path5 = 3.44467
activity_method1_experimental_path5 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'activities_experiment_method1_highhigh_PFKZERO.txt', dtype=float)
activity_method2_experimental_path5 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'activities_experiment_method2_highhigh_PFKZERO.txt', dtype=float)
flux_method1_experimental_path5 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'flux_experiment_method1_highhigh_PFKZERO.txt', dtype=float)
flux_method2_experimental_path5 = np.loadtxt(cwd+'\\TCA_PPP_GLYCOLYSIS_GOGAT\\'+'flux_experiment_method2_highhigh_PFKZERO.txt', dtype=float)


if (pathway_choice==1):
    epr_to_plot_method1 = epr_method1_experimental_path1
    epr_to_plot_method2 = epr_method2_experimental_path1
    activity_to_plot_method1=activity_method1_experimental_path1
    activity_to_plot_method2=activity_method2_experimental_path1
    flux_to_plot_method1=flux_method1_experimental_path1
    flux_to_plot_method2=flux_method2_experimental_path1
    Pathway_Name='GLUCONEOGENESIS'
        
    #p_string_random = cwd + '\\GLYCOLYSIS_TCA_GOGAT_randomds'
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
    Pathway_Name='GLYCOLYSIS'
        
    #p_string_random = cwd + '\\GLYCOLYSIS_TCA_GOGAT_randomds'
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
    Pathway_Name='GLYCOLYSIS_TCA_PPP, High/High'
        
    #p_string_random = cwd + '\\TCA_PPP_GLYCOLYSIS_GOGAT_randomds_highhigh'
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
    Pathway_Name='GLYCOLYSIS_TCA_PPP, High/Low'
        
    p_string = cwd + p4_string
    
    bigg_reaction_abbr_p = bigg_reaction_abbr_p3
    eps=0.2

#TCA_PPP_GLY ZERO_PFK
if (pathway_choice==5):
    epr_to_plot_method1 = epr_method1_experimental_path5
    epr_to_plot_method2 = epr_method2_experimental_path5
    activity_to_plot_method1=activity_method1_experimental_path5
    activity_to_plot_method2=activity_method2_experimental_path5
    flux_to_plot_method1=flux_method1_experimental_path5
    flux_to_plot_method2=flux_method2_experimental_path5
    Pathway_Name='GLYCOLYSIS_TCA_PPP ZERO_PFK'
        
    p_string = cwd + p5_string
    
    bigg_reaction_abbr_p = bigg_reaction_abbr_p4
    eps=0.2

#Glycolysis with no constraint
if (pathway_choice==6):
    epr_to_plot_method1 = epr_method1_experimental_path2
    epr_to_plot_method2 = epr_method2_experimental_path2
    activity_to_plot_method1=activity_method1_experimental_path2
    activity_to_plot_method2=activity_method2_experimental_path2
    flux_to_plot_method1=flux_method1_experimental_path2
    flux_to_plot_method2=flux_method2_experimental_path2
    Pathway_Name='GLYCOLYSIS no constraint'
        
    p_string_random = cwd + '\\GLYCOLYSIS_TCA_GOGAT_noconstraint'
    p_string = cwd + p6_string
    bigg_reaction_abbr_p = bigg_reaction_abbr_p2
    eps=0.5


#TCA_PPP_GLY with high/high no constraint
if (pathway_choice==7):
    epr_to_plot_method1 = epr_method1_experimental_path3
    epr_to_plot_method2 = epr_method2_experimental_path3
    activity_to_plot_method1=activity_method1_experimental_path3
    activity_to_plot_method2=activity_method2_experimental_path3
    flux_to_plot_method1=flux_method1_experimental_path3
    flux_to_plot_method2=flux_method2_experimental_path3
    Pathway_Name='GLYCOLYSIS_TCA_PPP, High/High, no constraint'
        
    p_string = cwd + p7_string
    bigg_reaction_abbr_p = bigg_reaction_abbr_p3
    eps=0.2


#TCA_PPP_GLY with high/low no constraint
if (pathway_choice==8):
    epr_to_plot_method1 = epr_method1_experimental_path4
    epr_to_plot_method2 = epr_method2_experimental_path4
    activity_to_plot_method1=activity_method1_experimental_path4
    activity_to_plot_method2=activity_method2_experimental_path4
    flux_to_plot_method1=flux_method1_experimental_path4
    flux_to_plot_method2=flux_method2_experimental_path4
    Pathway_Name='GLYCOLYSIS_TCA_PPP, High/Low, no constraint'
        
    p_string = cwd + p8_string
    
    bigg_reaction_abbr_p = bigg_reaction_abbr_p3
    eps=0.2

#TCA_PPP_GLY ZERO_PFK no constraint
if (pathway_choice==9):
    epr_to_plot_method1 = epr_method1_experimental_path5
    epr_to_plot_method2 = epr_method2_experimental_path5
    activity_to_plot_method1=activity_method1_experimental_path5
    activity_to_plot_method2=activity_method2_experimental_path5
    flux_to_plot_method1=flux_method1_experimental_path5
    flux_to_plot_method2=flux_method2_experimental_path5
    Pathway_Name='GLYCOLYSIS_TCA_PPP ZERO_PFK, no constraint'
        
    p_string = cwd + p9_string
    
    bigg_reaction_abbr_p = bigg_reaction_abbr_p4
    eps=0.2


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

p_len=350
path_lengths=[p_len,p_len,p_len,p_len,p_len,p_len]
num_sims = [11,11,11,11,11,11]
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
        
        for i in range(1,num_sims[index_n]):
            tempr = np.loadtxt(p_string+'\\episodic_reward_gamma9_n'+str(n)+
                          '_k5__lr'+ lr +'_threshold25_eps'+str(eps)+'_penalty_reward_scalar_0.0'+
                          '_use_experimental_metab_'+str(use_exp_data)+'_sim'
                          +str(i)+'.txt',dtype=float)

            arr_reward = np.vstack((arr_reward,tempr[0:path_length]))
         
        #fig = plt.figure(figsize=(figure_norm, 0.5*figure_norm))
        #ax1 = fig.add_subplot(1,1,1)
        #ax1.plot(np.mean(arr_reward[1:,:], axis=0))
        
        sim_rew_set = np.mean(arr_reward[1:,-length_of_back_count:], axis=1)
        reward_mean_n.append(np.mean(sim_rew_set))
        reward_stds_n.append(np.std(sim_rew_set))
    reward_means.append(reward_mean_n)
    reward_stds.append(reward_stds_n)

lr_index=math.floor(np.argmax(reward_means) / len(nvals))
lr = lrs[lr_index]
n_choice = nvals[math.floor(np.argmax(reward_means) % len(nvals)) ]

#######################################################################################
max_reward_index = np.argmax(reward_means[lr_index])
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
    
    for i in range(1,num_sims[index_n]):
        #episodic_loss_gamma9_n2_k5__lr1e-08_threshold25_eps0.5_penalty_reward_scalar_0.0_use_experimental_metab_0_sim7
        temp = np.loadtxt(p_string+'\\episodic_loss_gamma9_n'+str(n)+
                          '_k5__lr'+ lr +'_threshold25_eps'+str(eps)+'_penalty_reward_scalar_0.0'+
                          '_use_experimental_metab_'+str(use_exp_data)+'_sim'
                          +str(i)+'.txt',dtype=float)

        
        #temp = np.sqrt(temp)
        temp[0:path_length] = temp[0:path_length] / max(temp[0:path_length])
        arr_loss = np.vstack((arr_loss,temp[0:path_length]))
        
        tempr = np.loadtxt(p_string+'\\episodic_reward_gamma9_n'+str(n)+
                          '_k5__lr'+ lr +'_threshold25_eps'+str(eps)+'_penalty_reward_scalar_0.0'+
                          '_use_experimental_metab_'+str(use_exp_data)+'_sim'
                          +str(i)+'.txt',dtype=float)

        arr_reward = np.vstack((arr_reward,tempr[0:path_length]))
        
        temp_epr = np.loadtxt(p_string+'\\epr_per_state_gamma9_n'+str(n)+
                          '_k5__lr'+ lr +'_threshold25_eps'+str(eps)+'_penalty_reward_scalar_0.0'+
                          '_use_experimental_metab_'+str(use_exp_data)+'_sim'
                          +str(i)+'.txt',dtype=float)
        
        temp_epr=temp_epr[0:path_length]
    
        all_visited_epr_full = np.vstack((all_visited_epr_full,temp_epr[-length_of_back_count:]))
        
        temp_state = np.loadtxt(p_string+'\\final_states_gamma9_n'+str(n)+
                      '_k5__lr'+ lr +'_threshold25_eps'+str(eps)+'_penalty_reward_scalar_0.0'+
                      '_use_experimental_metab_'+str(use_exp_data)+'_sim'
                      +str(i)+'.txt',dtype=float)
        temp_state=temp_state[0:path_length+1]
        temp_state=temp_state[1:]

        temp_state=temp_state[-length_of_back_count:]
        
        temp_KQF = np.loadtxt(p_string+'\\final_KQF_gamma9_n'+str(n)+
                      '_k5__lr'+ lr +'_threshold25_eps'+str(eps)+'_penalty_reward_scalar_0.0'+
                      '_use_experimental_metab_'+str(use_exp_data)+'_sim'
                      +str(i)+'.txt',dtype=float)
        temp_KQF=temp_KQF[0:path_length]
        temp_KQF=temp_KQF[-length_of_back_count:]
                
        temp_KQR = np.loadtxt(p_string+'\\final_KQR_gamma9_n'+str(n)+
                      '_k5__lr'+ lr +'_threshold25_eps'+str(eps)+'_penalty_reward_scalar_0.0'+
                      '_use_experimental_metab_'+str(use_exp_data)+'_sim'
                      +str(i)+'.txt',dtype=float)
        temp_KQR=temp_KQR[0:path_length]
        temp_KQR=temp_KQR[-length_of_back_count:]
        
        arr_epr = np.vstack((arr_epr, temp_epr[-length_of_back_count:]))
    
    
        #EPR/glucose = -RT*(1/rxn_flux(15)) * sum(rxn_flux.*log(KQ)) is supposed to calculated here
        rxn_flux = temp_state*(temp_KQF - temp_KQR)
        
        mrf = np.max(rxn_flux, axis=1)
        
        max_flux = np.vstack((max_flux,mrf))
        
        epr_glucose = -RT*(1/rxn_flux[:,15] * np.sum(rxn_flux*np.log(temp_KQF), axis=1))
        
        arr_epr_glucose = np.vstack((arr_epr_glucose, epr_glucose))
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
    
    
    average_loss.append(np.mean(arr_loss[1:,:],axis=0))
    
    average_reward.append(np.mean(arr_reward[1:,-length_of_back_count:],axis=0))  
    average_reward_total.append(np.mean(arr_reward[1:,:],axis=0))  
    average_epr.append(np.mean(arr_epr[1:,:]))  
    std_epr.append(np.std(arr_epr[1:,:]))  


    #rxn_flux = temp_state*(temp_KQF - temp_KQR)
    #average_epr_normalized.append(np.mean(arr_epr[1:,:]/temp_KQF[:,15]))  
    #std_epr_normalized.append(np.std(arr_epr[1:,:]/temp_KQF[:,15]))  
    
    average_epr_normalized.append(np.mean(arr_epr_glucose[1:,:]))  
    std_epr_normalized.append(np.std(arr_epr_glucose[1:,:]))
    
    
    


#%%
ave_reward_n=[]
std_reward_n=[]
for index_n,n in enumerate(nvals):
    ave_reward_n.append(np.mean(average_reward[index_n]))
    std_reward_n.append(np.std(average_reward[index_n]))


fig = plt.figure(figsize=(figure_norm, figure_norm))

ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

loss_to_plot = np.sqrt((average_loss[max_reward_index]) / max((average_loss[max_reward_index])))
ax1.plot(loss_to_plot )#, label='n='+str(nvals[max_reward_index]))
    

reward_to_plot = average_reward_total[max_reward_index]

ax2.plot(reward_to_plot,
             linestyle='-')

fig.suptitle(Pathway_Name,fontsize=Fontsize_Title)

ax1.annotate("A", xy=(-0.1, 1.0), xycoords="axes fraction",fontsize=Fontsize_Sub)
ax2.annotate("B", xy=(-0.1, 1.0), xycoords="axes fraction",fontsize=Fontsize_Sub)

ax1.set_xlabel("Episode",fontsize=Fontsize_Sub)
ax1.set_ylabel("RMSE Error",fontsize=Fontsize_Sub)
ax2.set_xlabel("Episode",fontsize=Fontsize_Sub)
ax2.set_ylabel("Averaged Episodic Reward",fontsize=Fontsize_Sub)
#ax2.set_xticks(nvals)

ax1.legend(fontsize=Fontsize_Leg, loc='upper right')

#%% Just plot rewards
fig = plt.figure(figsize=(figure_norm, 0.5*figure_norm))
ax1 = fig.add_subplot(1,1,1)
    
loss_to_plot = np.sqrt((average_loss[max_reward_index]) / max((average_loss[max_reward_index])))
 


ax1.errorbar(nvals, ave_reward_n, yerr=std_reward_n,
             linestyle=':',
             label='n='+str(nvals[max_reward_index]))

fig.suptitle(Pathway_Name,fontsize=Fontsize_Title)

ax1.set_xlabel("n",fontsize=Fontsize_Sub)
ax1.set_ylabel("Averaged Reward per Episode",fontsize=Fontsize_Sub)
#ax2.set_xticks(nvals)

ax1.legend(fontsize=Fontsize_Leg, loc='upper right')

#%%

for index_n,n in enumerate(nvals[:]):
    plt.plot(average_reward[index_n], label=str(nvals[index_n]))
    plt.legend()
#%%

for index_n,n in enumerate(nvals[:]):
    plt.plot(average_loss[index_n], label=str(nvals[index_n]))
    plt.legend()
#%% plot best states


length_of_back_count=50
most_visited_state=[]
most_visited_states=[]
most_visited_flux=[]
most_visited_KQF=[]
all_visited_eprs=[]
most_visited_epr=[]

for index_n,n in enumerate(nvals):
    if n == nvals[max_reward_index]:
        path_length=path_lengths[index_n]
        most_visited_states=np.zeros((0,temp_state.shape[1]))
        most_visited_flux=np.zeros((0,temp_state.shape[1]))
        most_visited_KQF=np.zeros((0,temp_state.shape[1]))
            
        arr_epr = np.zeros((0,length_of_back_count))
        arr_epr_glucose = np.zeros(length_of_back_count)
        
        max_flux=np.zeros(length_of_back_count)
        
        for i in range(1,num_sims[index_n]):
            
            print(i)
            temp_state = np.loadtxt(p_string+'\\final_states_gamma9_n'+str(n)+
                          '_k5__lr'+str(lr)+'_threshold25_eps'+str(eps)+'_penalty_reward_scalar_0.0'+
                          '_use_experimental_metab_'+str(use_exp_data)+'_sim'
                          +str(i)+'.txt',dtype=float)
            
            temp_state=temp_state[1:path_length+1]
            
            #fig = plt.figure(figsize=(figure_norm, 0.5*figure_norm))
            #ax1 = fig.add_subplot(1,1,1)
            #ax1.plot(temp_state[-length_of_back_count:].T)
            

            
            most_visited_states = np.vstack((most_visited_states,temp_state[-length_of_back_count:]))
            
            temp_epr = np.loadtxt(p_string+'\\epr_per_state_gamma9_n'+str(n)+
                          '_k5__lr'+str(lr)+'_threshold25_eps'+str(eps)+'_penalty_reward_scalar_0.0'+
                          '_use_experimental_metab_'+str(use_exp_data)+'_sim'
                          +str(i)+'.txt',dtype=float)
            temp_epr=temp_epr[0:path_length]
    
            
            
            temp_state=temp_state[-length_of_back_count:]
            
            temp_KQF = np.loadtxt(p_string+'\\final_KQF_gamma9_n'+str(n)+
                          '_k5__lr'+ lr +'_threshold25_eps'+str(eps)+'_penalty_reward_scalar_0.0'+
                          '_use_experimental_metab_'+str(use_exp_data)+'_sim'
                          +str(i)+'.txt',dtype=float)
            temp_KQF=temp_KQF[1:path_length+1]
            temp_KQF=temp_KQF[-length_of_back_count:]
                    
            temp_KQR = np.loadtxt(p_string+'\\final_KQR_gamma9_n'+str(n)+
                          '_k5__lr'+ lr +'_threshold25_eps'+str(eps)+'_penalty_reward_scalar_0.0'+
                          '_use_experimental_metab_'+str(use_exp_data)+'_sim'
                          +str(i)+'.txt',dtype=float)
            temp_KQR=temp_KQR[1:path_length+1]
            temp_KQR=temp_KQR[-length_of_back_count:]
            
            arr_epr = np.vstack((arr_epr, temp_epr[-length_of_back_count:]))

        
            #EPR/glucose = -RT*(1/rxn_flux(15)) * sum(rxn_flux.*log(KQ)) is supposed to calculated here
            rxn_flux = temp_state*(temp_KQF - temp_KQR)

            most_visited_flux = np.vstack((most_visited_flux,rxn_flux))
            most_visited_KQF = np.vstack((most_visited_KQF,temp_KQF))
            
            #rxn_flux[rxn_flux==0]=1e-16
            #epr_glucose = -RT*(1/rxn_flux[:,15] * np.sum(rxn_flux * np.log(temp_KQF), axis=1))
            
            #epr_glucose = temp_epr[-length_of_back_count:]/temp_KQF[:,15]
            arr_epr_glucose = np.vstack((arr_epr_glucose, epr_glucose))

        #most_visited_states=most_visited_states[1:,:]
        #most_visited_flux=most_visited_flux[1:,:]
        #most_visited_KQF=most_visited_KQF[1:,:]


        [states_unique, indices_unique, counts_unique]=np.unique(most_visited_states[:,:], axis=0, return_index=True,return_counts=True)
        correct_row_index=indices_unique[np.argmax(counts_unique)]
        
        #[flux_unique, flux_indices_unique, flux_counts_unique]=np.unique(most_visited_flux[:,:], axis=0, return_index=True,return_counts=True)
        #[KQF_unique, KQF_indices_unique, KQF_counts_unique]=np.unique(most_visited_KQF[:,:], axis=0, return_index=True,return_counts=True)
        
        most_visited_state = states_unique[np.argmax(counts_unique),:]
        most_visited_state_alt = most_visited_states[correct_row_index,:]
        most_visited_flux = most_visited_flux[correct_row_index,:]
        most_visited_KQF = most_visited_KQF[correct_row_index,:]
        most_visited_epr_vec=arr_epr.flatten()
        most_visited_epr = most_visited_epr_vec[correct_row_index]
        

flux_method_RL = most_visited_flux

final_state_data = pd.DataFrame(columns=['Method','Activity','Reaction'])

counter=0
for i in range(0,(temp_state.shape[1])):
    final_state_data.loc[i] = ['RL', most_visited_state[i],i]
    
#save data for pathways 2,3,4


if (pathway_choice == 2):
    np.savetxt(cwd + p2_string_short +'\\activities_experiment_RL.txt', most_visited_state, fmt='%1.30f')
    np.savetxt(cwd + p2_string_short +'\\flux_experiment_RL.txt', most_visited_flux, fmt='%1.30f')
    np.savetxt(cwd + p2_string_short +'\\KQF_experiment_RL.txt', most_visited_KQF, fmt='%1.30f')
    
if (pathway_choice == 3):
    np.savetxt(cwd + p3_string_short +'\\activities_experiment_RL_highhigh.txt', most_visited_state, fmt='%1.30f')
    np.savetxt(cwd + p3_string_short +'\\flux_experiment_RL_highhigh.txt', most_visited_flux, fmt='%1.30f')
    np.savetxt(cwd + p3_string_short +'\\KQF_experiment_RL_highhigh.txt', most_visited_KQF, fmt='%1.30f')
    
if (pathway_choice == 4):
    np.savetxt(cwd + p4_string_short +'\\activities_experiment_RL_highlow.txt', most_visited_state, fmt='%1.30f')
    np.savetxt(cwd + p4_string_short +'\\flux_experiment_RL_highlow.txt', most_visited_flux, fmt='%1.30f')
    np.savetxt(cwd + p4_string_short +'\\KQF_experiment_RL_highlow.txt', most_visited_KQF, fmt='%1.30f')
    
if (pathway_choice == 5):
    np.savetxt(cwd + p3_string_short +'\\activities_experiment_RL_highhigh_PFKZERO.txt', most_visited_state, fmt='%1.30f')
    np.savetxt(cwd + p3_string_short +'\\flux_experiment_RL_highhigh_PFKZERO.txt', most_visited_flux, fmt='%1.30f')
    np.savetxt(cwd + p3_string_short +'\\KQF_experiment_RL_highhigh_PFKZERO.txt', most_visited_KQF, fmt='%1.30f')    
    
if (pathway_choice == 6):
    np.savetxt(cwd + p2_string_short +'\\activities_experiment_RL_noconstraint.txt', most_visited_state, fmt='%1.30f')
    np.savetxt(cwd + p2_string_short +'\\flux_experiment_RL_noconstraint.txt', most_visited_flux, fmt='%1.30f')
    np.savetxt(cwd + p2_string_short +'\\KQF_experiment_RL_noconstraint.txt', most_visited_KQF, fmt='%1.30f')
    
if (pathway_choice == 7):
    np.savetxt(cwd + p3_string_short +'\\activities_experiment_RL_noconstraint_highhigh.txt', most_visited_state, fmt='%1.30f')
    np.savetxt(cwd + p3_string_short +'\\flux_experiment_RL_noconstraint_highhigh.txt', most_visited_flux, fmt='%1.30f')
    np.savetxt(cwd + p3_string_short +'\\KQF_experiment_RL_noconstraint_highhigh.txt', most_visited_KQF, fmt='%1.30f')
    
if (pathway_choice == 8):
    np.savetxt(cwd + p4_string_short +'\\activities_experiment_RL_noconstraint_highlow.txt', most_visited_state, fmt='%1.30f')
    np.savetxt(cwd + p4_string_short +'\\flux_experiment_RL_noconstraint_highlow.txt', most_visited_flux, fmt='%1.30f')
    np.savetxt(cwd + p4_string_short +'\\KQF_experiment_RL_noconstraint_highlow.txt', most_visited_KQF, fmt='%1.30f')
    
    
if (pathway_choice == 9):
    np.savetxt(cwd + p3_string_short +'\\activities_experiment_RL_noconstraint_highhigh_PFKZERO.txt', most_visited_state, fmt='%1.30f')
    np.savetxt(cwd + p3_string_short +'\\flux_experiment_RL_noconstraint_highhigh_PFKZERO.txt', most_visited_flux, fmt='%1.30f')
    np.savetxt(cwd + p3_string_short +'\\KQF_experiment_RL_noconstraint_highhigh_PFKZERO.txt', most_visited_KQF, fmt='%1.30f')
#%%

mean_arr_epr_glucose = np.mean(arr_epr_glucose,axis=1)
std_arr_epr_glucose = np.std(arr_epr_glucose,axis=1)

mean_arr_epr = np.mean(arr_epr,axis=1)
std_arr_epr = np.std(arr_epr,axis=1)




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

most_visited_states=[]
most_visited_fluxes=[]
most_visited_KQFs=[]

all_visited_eprs=[]

length_of_back_count_epr=350

all_visited_epr=np.zeros(length_of_back_count_epr)
for index_n,n in enumerate(nvals):
    if n == nvals[max_reward_index]:
        path_length=path_lengths[index_n]
        most_visited_states=np.zeros((0,temp_state.shape[1]))
        most_visited_fluxes=np.zeros((0,temp_state.shape[1]))
        most_visited_KQFs=np.zeros((0,temp_state.shape[1]))
        all_visited_energy=np.zeros((0,temp_state.shape[1]))
            
        for i in range(1,num_sims[index_n]):
        #for i in range(2,3):
                    
            print(i)
            temp_state = np.loadtxt(p_string+'\\final_states_gamma9_n'+str(n)+
                          '_k5__lr'+str(lr)+'_threshold25_eps'+str(eps)+'_penalty_reward_scalar_0.0'+
                          '_use_experimental_metab_'+str(use_exp_data)+'_sim'
                          +str(i)+'.txt',dtype=float)
            temp_state=temp_state[1:path_length+1,:]
            
           

            
            most_visited_states = np.vstack((most_visited_states,temp_state[-length_of_back_count_epr:]))
            
            temp_epr = np.loadtxt(p_string+'\\epr_per_state_gamma9_n'+str(n)+
                          '_k5__lr'+str(lr)+'_threshold25_eps'+str(eps)+'_penalty_reward_scalar_0.0'+
                          '_use_experimental_metab_'+str(use_exp_data)+'_sim'
                          +str(i)+'.txt',dtype=float)
            temp_epr=temp_epr[0:path_length]
    
            all_visited_epr = np.vstack((all_visited_epr,temp_epr[-length_of_back_count_epr:]))
            
            
            temp_KQF = np.loadtxt(p_string+'\\final_KQF_gamma9_n'+str(n)+
                          '_k5__lr'+ lr +'_threshold25_eps'+str(eps)+'_penalty_reward_scalar_0.0'+
                          '_use_experimental_metab_'+str(use_exp_data)+'_sim'
                          +str(i)+'.txt',dtype=float)
            temp_KQF=temp_KQF[1:path_length+1]
            temp_KQF=temp_KQF[-length_of_back_count_epr:]
                    
            temp_KQR = np.loadtxt(p_string+'\\final_KQR_gamma9_n'+str(n)+
                          '_k5__lr'+ lr +'_threshold25_eps'+str(eps)+'_penalty_reward_scalar_0.0'+
                          '_use_experimental_metab_'+str(use_exp_data)+'_sim'
                          +str(i)+'.txt',dtype=float)
            temp_KQR=temp_KQR[1:path_length+1]
            temp_KQR=temp_KQR[-length_of_back_count_epr:]
            
            rxn_flux = temp_state[-length_of_back_count_epr:]*(temp_KQF - temp_KQR)

            most_visited_fluxes = np.vstack((most_visited_fluxes,rxn_flux))
            most_visited_KQFs = np.vstack((most_visited_KQFs,temp_KQF))
            
            rxn_energy = rxn_flux * np.log(temp_KQF)
            all_visited_energy = np.vstack((all_visited_energy,rxn_energy))
            
        
        #most_visited_states=most_visited_states[1:,:]
        #most_visited_fluxes=most_visited_fluxes[1:,:]
        #most_visited_KQFs=most_visited_KQFs[1:,:]
        
        [states_unique, indices_unique, counts_unique]=np.unique(most_visited_states[:,:], axis=0, return_index=True,return_counts=True)
        [flux_unique, flux_indices_unique, flux_counts_unique]=np.unique(most_visited_fluxes[:,:], axis=0, return_index=True,return_counts=True)
        [KQF_unique, KQF_indices_unique, KQF_counts_unique]=np.unique(most_visited_KQFs[:,:], axis=0, return_index=True,return_counts=True)
        
        most_visited_state = states_unique[np.argmax(counts_unique),:]
        most_visited_flux = flux_unique[np.argmax(counts_unique),:]
        most_visited_KQF = KQF_unique[np.argmax(counts_unique),:]

most_visited_energy = -np.sum(most_visited_flux * np.log(most_visited_KQF) )
states_unique = most_visited_states[:,:]
#take the corresponding unique epr 

#all_visited_epr = all_visited_epr[1:,:].flatten()[indices_unique]
all_visited_epr = all_visited_epr[1:,:].flatten()


#%% Plot 

fig = plt.figure(figsize=(figure_norm, 0.5*figure_norm))


fig.suptitle(Pathway_Name,fontsize=Fontsize_Title, y=1.0)
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)




sns.distplot(all_visited_epr, rug=False, kde=False,norm_hist=False, ax=ax1)

sns.regplot(x=np.array([epr_to_plot_method2]), y=np.array([1]), scatter=True, fit_reg=False, marker='x',
                ax=ax1,
                label='MCA',
                color='r',
               scatter_kws={"s": 100})

average_final_states = np.mean(states_unique, axis=0)
std_final_states = np.std(states_unique, axis=0)

x_vals = np.zeros(shape=states_unique.shape)
for i in range(states_unique.shape[1]-1):
    x_vals[:,i+1]=i+1


#ax2.scatter(x_vals, states_unique.T)

x_vals = [i for i in range(0,average_final_states.size)]
ax2.errorbar(x_vals, average_final_states, yerr=std_final_states,
             linestyle=':',alpha=1.0)


ax1.annotate("A", xy=(-0.1, 1.0), xycoords="axes fraction",fontsize=Fontsize_Sub)
ax2.annotate("B", xy=(-0.1, 1.0), xycoords="axes fraction",fontsize=Fontsize_Sub)


ax1.set_ylabel('EPR Counts',fontsize=Fontsize_Sub)
ax1.set_xlabel('Entropy Production Rate, EPR',fontsize=Fontsize_Sub)

ax2.set_ylabel('Activity \n mean and std',fontsize=Fontsize_Sub)
#ax2.set_ylabel('Activity unique \n states in last ' +str(length_of_back_count_epr)+ ' \n terminal states',fontsize=Fontsize_Sub)
ax2.set_xlabel('Reactions',fontsize=Fontsize_Sub)
ax1.legend(fontsize=Fontsize_Leg, loc='upper left')

fig.tight_layout()  # otherwise the right y-label is slightly clipped

#%% Try using max epr and backcalculate states for plot

fig = plt.figure(figsize=(figure_norm, 0.5*figure_norm))


fig.suptitle(Pathway_Name,fontsize=Fontsize_Title)
ax1 = fig.add_subplot(111)

all_visited_energy_total = -np.sum(all_visited_energy, axis=1)
args_energy = np.argsort(all_visited_energy_total,axis=0)

state_temp = most_visited_states[:,:]
state_temp= state_temp[args_energy,:]



num_regulated = np.sum((state_temp<1), axis=1)

#ax2.errorbar(x_vals, 1-max_average_final_states, yerr=max_std_final_states,
#             linestyle=':',alpha=1.0, color=color)
#ax2.plot(num_regulated)
ax1.scatter(num_regulated,all_visited_energy_total[args_energy],alpha=1.0)

num_regulated_most_visited = np.sum((most_visited_state<1))
ax1.scatter(num_regulated_most_visited, most_visited_energy, color='r')


ax1.set_xlabel('Number of Reactions Regulated',fontsize=Fontsize_Sub)
ax1.set_ylabel('Energy Value',fontsize=Fontsize_Sub)

ax1.set_xticks( [i for i in range(min(num_regulated),max(num_regulated)+1)] )

#%% Try using max epr and backcalculate states for plot

fig = plt.figure(figsize=(figure_norm, 0.5*figure_norm))


fig.suptitle(Pathway_Name,fontsize=Fontsize_Title)
ax1 = fig.add_subplot(111)


args_epr = np.argsort(all_visited_epr,axis=0)

state_temp = most_visited_states[:,:]
state_temp= state_temp[args_epr,:]



num_regulated = np.sum((state_temp<1), axis=1)

#ax2.errorbar(x_vals, 1-max_average_final_states, yerr=max_std_final_states,
#             linestyle=':',alpha=1.0, color=color)
#ax2.plot(num_regulated)
ax1.scatter(num_regulated,all_visited_epr[args_epr],alpha=1.0)



ax1.set_xlabel('Number of Reactions Regulated',fontsize=Fontsize_Sub)
ax1.set_ylabel('EPR Value',fontsize=Fontsize_Sub)

ax1.set_xticks( [i for i in range(min(num_regulated),max(num_regulated)+1)] )

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