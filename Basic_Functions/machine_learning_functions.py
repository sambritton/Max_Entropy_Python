# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 13:48:04 2019

@author: samuel_britton
"""

#%% Learning Test

##Terms::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#States: defined by how much each rxn is regulated (vector of 21 floats from 0-1)
#Action: defined by which rxn we choose to regulate. 
#Policy: function S_mat->A (each )


#%% Variables to set before use
Keq_constant=[]
f_log_counts=[]

P_mat=[]
R_back_mat=[]
S_mat=[]
delta_increment_for_small_concs=[]
desired_conc=[]
nvar=[]
mu0=[]

gamma=[]
num_rxns=[]
num_samples=[]
length_of_path=[]
#%% use functions 
import max_entropy_functions
import numpy as np
import pandas as pd
import random
from scipy.optimize import least_squares
from scipy.optimize import minimize


#NOTE Sign Switched, so we maximize (-epr)
def entropy_production_rate(KQ_f, KQ_r, E_Regulation):
    #breakpoint()
    KQ_f_reg = E_Regulation * KQ_f
    KQ_r_reg = E_Regulation * KQ_r
    sumOdds = np.sum(KQ_f_reg) + np.sum(KQ_r_reg)
    epr = +np.sum(KQ_f_reg * np.log(KQ_f_reg))/sumOdds + np.sum(KQ_r_reg * np.log(KQ_r_reg))/sumOdds
    return epr

    #old matlab code
    #KQ_reg = E_Regulation(active).*KQ(active);
    #KQ_inverse_reg = E_Regulation(active).*KQ_inverse(active);
    #sumOdds = sum(KQ_reg) + sum(KQ_inverse_reg);
    #entropy_production_rate = -sum(KQ_reg.*log(KQ(active)))/sumOdds - sum(KQ_inverse_reg.*log(KQ_inverse(active)))/sumOdds;

#try theta_linear * (entropy production rate)

#Physically this is trying to minimizing the free energy change in each reaction. 
def state_value(theta_linear, delta_s, KQ_f, KQ_r, E_Regulation):
    #epr = entropy_production_rate(KQ_f, KQ_r, E_Regulation)
    KQ_f_reg = E_Regulation * KQ_f
    KQ_r_reg = E_Regulation * KQ_r
    #sumOdds = np.sum(KQ_f_reg) + np.sum(KQ_r_reg)
    norm_LR = np.sum(theta_linear * KQ_f_reg/np.sum(KQ_f_reg) * np.log(KQ_f_reg))
    norm_RL = np.sum(theta_linear * KQ_r_reg/np.sum(KQ_r_reg) * np.log(KQ_r_reg))
   
    
    
    val = norm_LR + norm_RL
    
    #print("norm_LR")
    #print(norm_LR)
    #print("norm_RL")
    #rint(norm_RL)
    #NOTE CLEAR WHICH TO USE
    #val = np.sum(theta_linear * (-epr))
    #val = np.sum(theta_linear * (-delta_s))
    return -val

#define reward as entropy production rate
def reward_value(KQ_f, KQ_r, E_Regulation):
    val = -entropy_production_rate(KQ_f, KQ_r, E_Regulation)
    return val

     
def fun_opt_theta(theta_linear,state_delta_s_matrix, target_value_vec, KQ_f_matrix, KQ_r_matrix, state_sample_matrix):
    
    #breakpoint()
    totalvalue=0
    for sample in range(0,num_samples):
        delta_s = state_delta_s_matrix[:,sample].copy()
        KQ_f = KQ_f_matrix[:,sample].copy()
        KQ_r = KQ_r_matrix[:,sample].copy()
        regulation_sample = state_sample_matrix[:,sample].copy()
        
        sv = state_value(theta_linear, delta_s, KQ_f, KQ_r, regulation_sample)
        
        totalvalue+= (sv - target_value_vec[sample])**2
    totalvalue=totalvalue/2.0

        
    return (totalvalue)

#input theta_linear use policy_function to update theta_linear
def update_theta( theta_linear, v_log_counts_static, *args):
    #First we sample m states from E
    
    varargin = args
    nargin = len(varargin)
    epsilon_greedy = 0.0
    if (nargin == 1):
        epsilon_greedy = varargin[0]

    reaction_size_to_regulate = num_rxns
    if (epsilon_greedy > 0.0):
        reaction_size_to_regulate = 1
    
    has_been_up_regulated = 10*np.ones(num_rxns)
    
    estimate_value_vec=np.zeros(num_samples)
    best_action_vec=np.zeros(num_samples)
    state_sample_matrix= np.zeros(shape=(num_rxns, num_samples))
    state_delta_s_matrix= np.zeros(shape=(num_rxns, num_samples))
    KQ_f_matrix= np.zeros(shape=(num_rxns, num_samples))
    KQ_r_matrix= np.zeros(shape=(num_rxns, num_samples))
    
    for i in range(0,num_samples):
        state_sample = np.zeros(num_rxns)
        for sample in range(0,len(state_sample)):
            state_sample[sample] = np.random.uniform(0,1)
        state_sample_matrix[:,i] = state_sample#save for future use
        #print("state_sample")
        #print(state_sample)
        #breakpoint()
        #after sample of regulation is taken, optimize concentrations   
        res_lsq = least_squares(max_entropy_functions.derivatives, v_log_counts_static, method='lm',xtol=1e-15, args=(f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, state_sample))
        v_log_counts = res_lsq.x
        log_metabolites = np.append(v_log_counts, f_log_counts)
        
        rxn_flux = max_entropy_functions.oddsDiff(v_log_counts, f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, state_sample)
        KQ_f = max_entropy_functions.odds(log_metabolites, mu0,S_mat, R_back_mat, P_mat, delta_increment_for_small_concs,Keq_constant);
        KQ_f_matrix[:,i] = KQ_f
        Keq_inverse = np.power(Keq_constant,-1)
        KQ_r = max_entropy_functions.odds(log_metabolites, mu0,-S_mat, P_mat, R_back_mat, delta_increment_for_small_concs,Keq_inverse,-1);
        KQ_r_matrix[:,i] = KQ_r
        
        delta_S = max_entropy_functions.calc_deltaS(v_log_counts,f_log_counts, S_mat, KQ_f)
        state_delta_s_matrix[:,i] = delta_S
        #delta_S_metab = calc_delaS_metab(v_log_counts);
        [RR,Jac] = max_entropy_functions.calc_Jac2(v_log_counts, f_log_counts, S_mat, delta_increment_for_small_concs, KQ_f, KQ_r, state_sample)
        A = max_entropy_functions.calc_A(v_log_counts, f_log_counts, S_mat, Jac, state_sample )
        
        #Now we take an action (regulate a reaction) and calculate the new E value that 
        #would occur if we had regulated the state_sample
        estimate_value=-np.inf
        
        
        average_path_val=np.zeros(reaction_size_to_regulate)
        for act in range(0, reaction_size_to_regulate):
        
            #exploring starts method
            v_log_counts_path = v_log_counts.copy()
            value_path=np.zeros(length_of_path)
            act_path=np.zeros(length_of_path)
            path_state_sample = state_sample.copy()#these will be reset in the path loop
            
            #make iniial change for beginning of path
            React_Choice=act
            newE = max_entropy_functions.calc_reg_E_step(path_state_sample, React_Choice, nvar, 
                                   v_log_counts, f_log_counts, desired_conc, S_mat, A, 
                                   rxn_flux, KQ_f, False, has_been_up_regulated)
            path_state_sample[React_Choice] = newE
            
            #re-optimize
            new_res_lsq = least_squares(max_entropy_functions.derivatives, v_log_counts_path, method='lm',xtol=1e-15, args=(f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, path_state_sample))
            v_log_counts_path = new_res_lsq.x
            
            #if using epsilon, reaction_size_to_regulate==1, so cancel exploring starts
            if (epsilon_greedy > 0.0):
                v_log_counts_path = v_log_counts.copy()
                value_path=np.zeros(length_of_path)
                act_path=np.zeros(length_of_path)
                path_state_sample = state_sample.copy()#these will be reset in the path loop
                
                #make iniial change for beginning of path

                #re-optimize
                new_res_lsq = least_squares(max_entropy_functions.derivatives, v_log_counts_path, method='lm',xtol=1e-15, args=(f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, path_state_sample))
                v_log_counts_path = new_res_lsq.x
            
            #now generate the next step based on newE
            for path in range(0,length_of_path):
                #print("Path")
                #print(path)
                #React_Choice = act#regulate each reaction.
                #Here we use the state sample since we will reset it
                React_Choice = policy_function(path_state_sample,theta_linear, v_log_counts_path, epsilon_greedy)#regulate each reaction.
                
                newE = max_entropy_functions.calc_reg_E_step(path_state_sample, React_Choice, nvar, 
                                       v_log_counts, f_log_counts, desired_conc, S_mat, A,
                                       rxn_flux, KQ_f, False, has_been_up_regulated)
                
                #now generate the next step based on newE
                
                path_state_sample[React_Choice] = newE
                
                #re-optimize
                new_res_lsq = least_squares(max_entropy_functions.derivatives, v_log_counts_path, method='lm',xtol=1e-15, args=(f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, path_state_sample))
                
                v_log_counts_path = new_res_lsq.x
                new_log_metabolites = np.append(v_log_counts_path, f_log_counts)
            
                KQ_f_new = max_entropy_functions.odds(new_log_metabolites, mu0,S_mat, R_back_mat, P_mat, delta_increment_for_small_concs,Keq_constant);
                Keq_inverse = np.power(Keq_constant,-1)
                KQ_r_new = max_entropy_functions.odds(log_metabolites, mu0,-S_mat, P_mat, R_back_mat, delta_increment_for_small_concs,Keq_inverse,-1);
            
                new_delta_S = max_entropy_functions.calc_deltaS(v_log_counts_path,f_log_counts, S_mat, KQ_r_new)
                
               
                value_current_state = state_value(theta_linear, new_delta_S, KQ_f_new, KQ_r_new, path_state_sample)
                
                #Should the reward value be taken at path_state_sample
                #or state_sample??
                action_value = reward_value(KQ_f, KQ_r, state_sample) + (gamma) * value_current_state#note, action is using old KQ values
    
                q_ = action_value
                    
                #print("reaction")
                #print(React_Choice)
                #print("q")
                #print(q_)
                
                act_path[path]=React_Choice
                value_path[path]=q_
            
            #after the path is set, average to get q(a)
            average_path_val[act] = np.mean(value_path)
        
        #print("action_path")
        #print(act_path)
        #print("value_path")
        #print(value_path)
        #print("bestAction")
        #print(best_act)
        #print("bestValue")
        #print(best_val)
    
        #after every possible action for the state had been taken,
        #choose the optimal path for the sample. 
        estimate_value_vec[i] = np.max(average_path_val)   #estimate_value now represents the target value of the best possible action from the trail state.
        best_action_vec[i] = np.argmax(average_path_val)
        #print("estimate_value_vec")
        #print(estimate_value_vec)
        #print("best_actoin_vec")
        #print(best_action_vec)
    #Estimate New Theta
    #using all samples we compare how our value function matches, and update 
    #theta_linear accordingly. 
    #theta_linear = theta_linear + 0.01*()
    #breakpoint()
    res_min = minimize(fun_opt_theta, theta_linear, method='nelder-mead', args=(state_delta_s_matrix, estimate_value_vec, KQ_f_matrix, KQ_r_matrix, state_sample_matrix))
         
    return res_min.x  

#%%  
#input state, return action

def policy_function(state, theta_linear, v_log_counts_path, *args ):
    #last input argument should be epsilon for use when using greedy-epsilon algorithm. 
    
    varargin = args
    nargin = len(varargin)
    epsilon_greedy = 0.0
    if (nargin == 1):
        epsilon_greedy = varargin[0]
        

    rxn_choices = [i for i in range(num_rxns)]
    #MAYBE REMOVE
    has_been_up_regulated = 10*np.ones(num_rxns)
    
    res_lsq = least_squares(max_entropy_functions.derivatives, v_log_counts_path, method='lm',xtol=1e-15, args=(f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, state))
    v_log_counts = res_lsq.x
    log_metabolites = np.append(v_log_counts, f_log_counts)
        
    rxn_flux = max_entropy_functions.oddsDiff(v_log_counts, f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, state)
    KQ_f = max_entropy_functions.odds(log_metabolites, mu0,S_mat, R_back_mat, P_mat, delta_increment_for_small_concs,Keq_constant);
    Keq_inverse = np.power(Keq_constant,-1)
    KQ_r = max_entropy_functions.odds(log_metabolites, mu0,-S_mat, P_mat, R_back_mat, delta_increment_for_small_concs,Keq_inverse,-1);
    
    [RR,Jac] = max_entropy_functions.calc_Jac2(v_log_counts, f_log_counts, S_mat, delta_increment_for_small_concs, KQ_f, KQ_r, state)
    A = max_entropy_functions.calc_A(v_log_counts, f_log_counts, S_mat, Jac, state )
        
    reaction_size_to_regulate = num_rxns
    init_action_val = -np.inf
    action_choice=1
    
    for act in range(0,reaction_size_to_regulate):
        #print("in policy_function action loop")
        #print("v_log_counts")
        #print(np.max(v_log_counts))
        #print(np.min(v_log_counts))
        #print("end:")
        React_Choice = act#regulate each reaction.
        
        newE = max_entropy_functions.calc_reg_E_step(state, React_Choice, nvar, v_log_counts, f_log_counts,
                               desired_conc, S_mat, A, rxn_flux, KQ_f, False, has_been_up_regulated)
        #now generate the next step based on newE
        trial_state_sample = state.copy()#DO NOT MODIFY STATE
        trial_state_sample[React_Choice] = newE
            
        #re-optimize
        new_res_lsq = least_squares(max_entropy_functions.derivatives, v_log_counts, method='lm',xtol=1e-15, args=(f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, trial_state_sample))
            
        new_v_log_counts = new_res_lsq.x
        new_log_metabolites = np.append(new_v_log_counts, f_log_counts)
        
        KQ_f_new = max_entropy_functions.odds(new_log_metabolites, mu0,S_mat, R_back_mat, P_mat, delta_increment_for_small_concs,Keq_constant);    
        Keq_inverse = np.power(Keq_constant,-1)
        KQ_r_new = max_entropy_functions.odds(new_log_metabolites, mu0,-S_mat, P_mat, R_back_mat, delta_increment_for_small_concs,Keq_inverse,-1);

        new_delta_S = max_entropy_functions.calc_deltaS(new_v_log_counts,f_log_counts, S_mat, KQ_f_new)
    
        value_current_state = state_value(theta_linear, new_delta_S, KQ_f_new, KQ_r_new, trial_state_sample)

        action_value = reward_value(KQ_f, KQ_r, state) + gamma * value_current_state#note, action is using old KQ values
        if ((np.max(v_log_counts) > 200) or (np.min(v_log_counts) < -200)):
            print("v_log_counts_path")
            print(v_log_counts_path)
            print(newE)
            print(React_Choice) 
            print("v_log_counts")
            print(v_log_counts)

        #trial_state_sample * KQ_f_new = KQ_f_reg relation to KQ_f??
        #take smallest option after regulation
        if (action_value > init_action_val):
            init_action_val = action_value
            action_choice = act

            #KQ_f_reg = trial_state_sample * KQ_f_new
            #KQ_r_reg = trial_state_sample * KQ_r_new
            #sumOdds = np.sum(KQ_f_reg) + np.sum(KQ_r_reg)    
            #norm_LR = np.sum(theta_linear * KQ_f_reg/np.sum(KQ_f_reg) * np.log(KQ_f_reg))
            #norm_RL = np.sum(theta_linear * KQ_r_reg/np.sum(KQ_r_reg) * np.log(KQ_r_reg))

            #breakpoint()
            #print("act")
            #print(action_choice)
            #print("action_value")
            #print(action_value) 
            #print("value_current_state")
            #print(value_current_state)
            #print("sumOdds")
            #print(norm_LR)
            #print(KQ_f_reg * np.log(KQ_f_reg))
            #print(KQ_r_reg * np.log(KQ_r_reg))
    
    rxn_choices.remove(action_choice)
    random_choice = random.choice(rxn_choices)
    
    unif_rand = np.random.uniform(0,1)
    if (unif_rand < epsilon_greedy):
        print("****************************************************************")
        print("****************************************************************")
        print("USING EPSILON GREEDY")
        print(action_choice)
        print("replaced by")
        print(random_choice)
        
        action_choice = random_choice
    
    return action_choice