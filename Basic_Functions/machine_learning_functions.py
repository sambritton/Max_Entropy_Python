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

penalty_reward = -5.0

#%% use functions 
import max_entropy_functions
import numpy as np
import pandas as pd
import random
import time
from scipy.optimize import least_squares
from scipy.optimize import minimize
import multiprocessing as mp
from multiprocessing import Pool
import torch

Method = 'lm'


def generate_states(num_states_generate, steps_between_states, seed_state, v_log_counts_static ):
    
    has_been_up_regulated = 10*np.ones(num_rxns)
    
    all_states = np.zeros(shape=(num_rxns, num_states_generate * steps_between_states))
    all_unique_states=[]
    num_unique_states=0
    state = seed_state.copy()
    state_id=0
    
    while (num_unique_states < num_states_generate):
        
        res_lsq = least_squares(max_entropy_functions.derivatives, v_log_counts_static, method=Method,xtol=1e-15, args=(f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, state))
        if (res_lsq.optimality > 1e-8):
            print("warning not optimizing")
        v_log_counts = res_lsq.x
        log_metabolites = np.append(v_log_counts, f_log_counts)
        
        rxn_flux = max_entropy_functions.oddsDiff(v_log_counts, f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, state)
        KQ_f = max_entropy_functions.odds(log_metabolites, mu0,S_mat, R_back_mat, P_mat, delta_increment_for_small_concs,Keq_constant);
    
        Keq_inverse = np.power(Keq_constant,-1)
        KQ_r = max_entropy_functions.odds(log_metabolites, mu0,-S_mat, P_mat, R_back_mat, delta_increment_for_small_concs,Keq_inverse,-1);
    
        delta_S = max_entropy_functions.calc_deltaS(v_log_counts,f_log_counts, S_mat, KQ_f)
    
        [RR,Jac] = max_entropy_functions.calc_Jac2(v_log_counts, f_log_counts, S_mat, delta_increment_for_small_concs, KQ_f, KQ_r, state)
        A = max_entropy_functions.calc_A(v_log_counts, f_log_counts, S_mat, Jac, state )
        
    
        EPR = entropy_production_rate(KQ_f, KQ_r, state)

        step_possible = False #initial value
        regulated_reaction = -1 * np.ones(num_rxns)
        activity_choices = np.ones(num_rxns)
        for act in range(0, num_rxns):
        
            #make iniial change for beginning of path
            React_Choice=act
            initialE = state[act]
            newE = max_entropy_functions.calc_reg_E_step(state, React_Choice, nvar, 
                               v_log_counts, f_log_counts, desired_conc, S_mat, A, 
                               rxn_flux, KQ_f, False, has_been_up_regulated)
    
            if ((newE < 1) and (0 < newE) and (newE != initialE)):
                activity_choices[act] = newE
                regulated_reaction[act] = act
                step_possible = True
        
        regulated_reaction, = np.where(regulated_reaction>-1) #indices
        index_choice = random.choice(regulated_reaction)
        
        #modify state
        state[index_choice] = activity_choices[index_choice]
        
        
        #reset to seed state if you cannot move
        if (step_possible == False):
            state = seed_state.copy()
        
        all_states[:,state_id] = state
        all_unique_states = np.unique(all_states, axis=1)
        all_unique_states = all_unique_states[:,1:]
        num_unique_states = all_unique_states.shape[1]
        
        
        all_states[:,0:num_unique_states] = all_unique_states
        state_id = num_unique_states
    return all_unique_states


def test_state_validity(state, v_log_counts_static):
    is_state_valid = False
    
    has_been_up_regulated = 10*np.ones(num_rxns)
    res_lsq = least_squares(max_entropy_functions.derivatives, v_log_counts_static, method=Method,xtol=1e-15, args=(f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, state))
    if (res_lsq.optimality > 1e-5):
        is_state_valid=False
        return is_state_valid
        print("warning not optimizing")
        breakpoint()
        
    v_log_counts = res_lsq.x
    log_metabolites = np.append(v_log_counts, f_log_counts)
        
    rxn_flux = max_entropy_functions.oddsDiff(v_log_counts, f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, state)
    KQ_f = max_entropy_functions.odds(log_metabolites, mu0,S_mat, R_back_mat, P_mat, delta_increment_for_small_concs,Keq_constant);
    
    Keq_inverse = np.power(Keq_constant,-1)
    KQ_r = max_entropy_functions.odds(log_metabolites, mu0,-S_mat, P_mat, R_back_mat, delta_increment_for_small_concs,Keq_inverse,-1);
    
    #delta_S = max_entropy_functions.calc_deltaS(v_log_counts,f_log_counts, S_mat, KQ_f)
    
    [RR,Jac] = max_entropy_functions.calc_Jac2(v_log_counts, f_log_counts, S_mat, delta_increment_for_small_concs, KQ_f, KQ_r, state)
    A = max_entropy_functions.calc_A(v_log_counts, f_log_counts, S_mat, Jac, state )
        
    
    EPR = entropy_production_rate(KQ_f, KQ_r, state)

    step_possible = False #initial value
    for act in range(0, num_rxns):
        
        #make iniial change for beginning of path
        React_Choice=act

        initialE = state[act]
        newE = max_entropy_functions.calc_reg_E_step(state, React_Choice, nvar, 
                               v_log_counts, f_log_counts, desired_conc, S_mat, A, 
                               rxn_flux, KQ_f, False, has_been_up_regulated)
    
        if ((newE < 1) and (0 < newE) and (newE != initialE)):
            step_possible = True
            regulated_reaction = act
            
            #print(regulated_reaction)
            #print(KQ_f)
        
    #breakpoint()
    #conditions for valid state
    #EPR positive
    #regulation step possible for some reaction
    if ( (EPR > 0) and (step_possible == True) ):
        is_state_valid = True
    return is_state_valid

#NOTE Sign Switched, so we maximize (-epr)

#returns vector
    
def delta_delta_s_vec(delta_S, delta_S_metab, ccc, KQ, E_regulation, v_log_counts):
    S_index, = np.where(delta_S > -np.inf) 
    sm_idx, = np.where(delta_S_metab > 0.0) #reactions that have bad values 

    S_index_neg, = np.where(delta_S < 0.0) 
    sm_idx_neg, = np.where(delta_S_metab < 0.0) #reactions that have bad values 
    
    temp = ccc[np.ix_(sm_idx, S_index)]#np.ix_ does index outer product
    temp_neg = ccc[np.ix_(sm_idx_neg, S_index_neg)]#np.ix_ does outer product
                
    temp2 = (temp > 0) #ccc>0 means derivative is positive (dlog(conc)/dlog(activity)>0) 
    #this means regulation (decrease in activity) will result in decrease in conc
            
            
    temp2_neg = (temp_neg > 0)
            
    #row represents rxn, col represents metabolite
    temp_x = (temp * temp2)#Do not use matmul, use element wise mult.
    temp_x_neg = (temp_neg * temp2_neg)
            
    dx = np.multiply(v_log_counts[sm_idx].T, temp_x.T)
    dx_neg = np.multiply(v_log_counts[sm_idx_neg].T, temp_x_neg.T)

    #dx_neg = v_counts[sm_idx_neg].T*temp_x_neg
    
    #Change in enzyme activity
    
    DeltaAlpha = 0.001; # must be small enough such that the arguement
                                # of the log below is > 0
    DeltaDeltaS = -np.log(1 - DeltaAlpha*np.divide(dx, v_log_counts[sm_idx]))
            
    alternate_vector = np.sum(DeltaDeltaS, axis=1) #sum along metabolites
    
    return alternate_vector       
            
def entropy_production_rate_vec(KQ_f, KQ_r, E_Regulation, *args):
    
    
    varargin = args
    nargin = len(varargin)

    theta=np.ones(E_Regulation.size)
    if (nargin == 1):
        theta=varargin[0].copy()
    if (nargin == len(E_Regulation)):
        theta=varargin.copy()

    
    KQ_f_reg = E_Regulation * KQ_f
    KQ_r_reg = E_Regulation * KQ_r
    sumOdds = np.sum(KQ_f_reg) + np.sum(KQ_r_reg)

    kq_ge1_idx, = np.where(KQ_f >= 1)
    kq_le1_idx, = np.where(KQ_f < 1)
    kq_inv_ge1_idx, = np.where(KQ_r > 1)
    kq_inv_le1_idx, = np.where(KQ_r <= 1)
    #epr = +np.sum(KQ_f_reg * safe_ln(KQ_f_reg))/sumOdds + np.sum(KQ_r_reg * safe_ln(KQ_r_reg))/sumOdds
    

    val=np.zeros(E_Regulation.size)
    if (kq_ge1_idx.size>0):
        val[kq_ge1_idx] += theta[kq_ge1_idx] * (KQ_f_reg[kq_ge1_idx] * np.log(KQ_f[kq_ge1_idx]))/sumOdds
        
    if (kq_le1_idx.size>0):
        val[kq_le1_idx] -= theta[kq_le1_idx] * (KQ_f_reg[kq_le1_idx] * np.log(KQ_f[kq_le1_idx]))/sumOdds
        
    if (kq_ge1_idx.size>0):
        val[kq_inv_le1_idx] -= theta[kq_inv_le1_idx] * (KQ_r_reg[kq_inv_le1_idx] * np.log(KQ_r[kq_inv_le1_idx]))/sumOdds
        
    if (kq_le1_idx.size>0):
        val[kq_inv_ge1_idx] += theta[kq_inv_ge1_idx] * (KQ_r_reg[kq_inv_ge1_idx] * np.log(KQ_r[kq_inv_ge1_idx]))/sumOdds
        
    
    val_temp = max_entropy_functions.entropy_production_rate(KQ_f, KQ_r, E_Regulation)
    if ((theta==1).all()):
        if (val_temp - np.sum(val)>0.1):
            breakpoint()
    #val = +theta * (KQ_f_reg[kq_ge1_idx] * np.log(KQ_f[kq_ge1_idx]))/sumOdds \
    #      -theta * (KQ_f_reg[kq_le1_idx] * np.log(KQ_f[kq_le1_idx]))/sumOdds \
    #      -theta * (KQ_r_reg[kq_inv_le1_idx] * np.log(KQ_r[kq_inv_le1_idx]))/sumOdds \
    #      +theta * (KQ_r_reg[kq_inv_ge1_idx] * np.log(KQ_r[kq_inv_ge1_idx]))/sumOdds
          

    return val

def entropy_production_rate(KQ_f, KQ_r, E_Regulation):
    #breakpoint()
    #KQ_f_reg = E_Regulation * KQ_f
    #KQ_r_reg = E_Regulation * KQ_r
    #sumOdds = np.sum(KQ_f_reg) + np.sum(KQ_r_reg)
    #epr = +np.sum(KQ_f_reg * np.log(KQ_f_reg))/sumOdds + np.sum(KQ_r_reg * np.log(KQ_r_reg))/sumOdds
    
    #norm_LR = np.sum(KQ_f_reg/sumOdds * np.log(KQ_f))
    #norm_RL = np.sum(KQ_r_reg/sumOdds * np.log(KQ_r))
    
    vec = entropy_production_rate_vec(KQ_f, KQ_r, E_Regulation)
    val = np.sum(vec)
    #val = (norm_LR + norm_RL)
    return val

    #old matlab code
    #KQ_reg = E_Regulation(active).*KQ(active);
    #KQ_inverse_reg = E_Regulation(active).*KQ_inverse(active);
    #sumOdds = sum(KQ_reg) + sum(KQ_inverse_reg);
    #entropy_production_rate = -sum(KQ_reg.*log(KQ(active)))/sumOdds - sum(KQ_inverse_reg.*log(KQ_inverse(active)))/sumOdds;

#try theta_linear * (entropy production rate)

#Physically this is trying to minimizing the free energy change in each reaction. 
#we use the negative since maximizing (-f(x)) is the same as minimizing (f(x))
def state_value(nn_model, theta_linear, delta_s, delta_s_metab, KQ_f, KQ_r, E_Regulation, v_log_counts):
    
    
    #Problem::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #in this function, KQ_f_reg is not changing much b/c E_Regulation is multiplying it.
    #when it does change, it does so by proportions
    #This makes theta_linear have no effect on individual reactions
    
    #As is, theta chooses lowest overall flux between forward and backward
    
    #KQ_f_reg = E_Regulation * KQ_f
    #KQ_r_reg = E_Regulation * KQ_r
    
    
    #sumOdds = np.sum(KQ_f_reg) + np.sum(KQ_r_reg)
    #norm_LR = np.sum(theta_linear * KQ_f_reg/sumOdds * np.log(KQ_f))
    #norm_RL = np.sum(theta_linear * KQ_r_reg/sumOdds * np.log(KQ_r))
   
    #print("norm_LR")
    #print(norm_LR)
    #print("norm_RL")
    #print(norm_RL)
    
    #vec = entropy_production_rate_vec(KQ_f, KQ_r, E_Regulation, theta_linear)
    #val = np.sum(vec)
    
    #ALT
    #temp_vec = delta_s.copy()
    #temp_vec[temp_vec<0]=0
    #vec = theta_linear * temp_vec
    #val = np.sum(vec)
    
    #NN
# =============================================================================
#     x = torch.zeros(1,1, E_Regulation.size)
#     for i in range(0,E_Regulation.size):
#         x[0][0][i] = E_Regulation[i].copy()
#     y_pred = nn_model(x)
#     val=y_pred
# =============================================================================
    
    x = torch.zeros(1,1, E_Regulation.size)
    count=0

    for i in range(0,E_Regulation.size):
        val = E_Regulation[i].copy()
        val = val**(0.5)
        x[0][0][i] = np.log(1.0 + np.exp(20.0)*val)
        count+=1
    y_pred = nn_model(x)
    val=y_pred
    
    #print(val)
    
    #breakpoint()
    #print(list(nn_model.parameters()))
    #val = np.sum(theta_linear * alternate_value_vector)
    
    #state_value_vec = delta_s[delta_s>0]
    
    #val = np.sum(theta_linear * state_value_vec)
    #vec = np.append(delta_s, delta_s_metab).copy()
    
    
    #vec = delta_s_metab.copy()
    
    #vec[vec<0]=0
    #val = np.sum(theta_linear * vec)
    return val

#define reward as entropy production rate
    #usually positive, so 
def reward_value(v_log_counts_future, v_log_counts_old, KQ_f, KQ_r, E_Regulation,\
                 delta_s_next,delta_s_previous):
    
    #val_old = max_entropy_functions.calc_deltaS_metab(v_log_counts_old)
    #val_future = max_entropy_functions.calc_deltaS_metab(v_log_counts_future)
    
    #want to maximize the change in loss function for positive values. 
    
    
    val_future=delta_s_next.copy()
    val_future[val_future<0.0]=0
        
    val_old=delta_s_previous.copy()
    val_old[val_old<0.0]=0
    
    #reward = np.sum((val_old) - (val_new))
    
    val_old[val_old<0.0] = 0
    val_future[val_future<0] = 0
    
    final_reward=0.0
    reward_s = np.sum(val_old - val_future) #this is positive if val_future is less, so we need
    
    
    if ((reward_s <= 0.0)):
        final_reward = penalty_reward
        
    if (reward_s > 0.0):
        final_reward = -0.01
        #if negative -> take fastest path
        #if positive -> take slowest path
        
    if ((delta_s_next<=0.0).all()):
        epr = entropy_production_rate(KQ_f, KQ_r, E_Regulation)
        final_reward = 1.0 * epr
        #breakpoint()
       
    #to maximize it. This will make the 
    
    #val_new=delta_s_next.copy()
    #val_new[val_new<0]=0
        
    #val_old=delta_s_previous.copy()
    #val_old[val_old<0]=0
    
    #reward = np.sum((val_old) - (val_new))
    #Reward = How much did delta_S decrease
    
    #reward = np.sum(val_old) - np.sum(val_new)

    
    return final_reward

     
def fun_opt_theta(theta_linear,state_delta_s_matrix, target_value_vec, KQ_f_matrix, KQ_r_matrix, state_sample_matrix):
    
    totalvalue=0
    for sample in range(0,num_samples):
        delta_s = state_delta_s_matrix[:,sample].copy()
        KQ_f = KQ_f_matrix[:,sample].copy()
        KQ_r = KQ_r_matrix[:,sample].copy()
        regulation_sample = state_sample_matrix[:,sample].copy()
        
        sv = state_value(theta_linear, delta_s, KQ_f, KQ_r, regulation_sample)
        target_value = target_value_vec[sample]
        
        totalvalue+= (sv - target_value)**2
    totalvalue=totalvalue/2.0
    
    #print("theta_linear")
    #print(totalvalue)
    #print(theta_linear)   
    
    return (totalvalue)

#input theta_linear use policy_function to update theta_linear
def update_theta( theta_linear, v_log_counts_static, *args):
    #First we sample m states from E
    #First args input is epsilon
    #Second args input is a matrix of states to fit to.
    
    varargin = args
    nargin = len(varargin)
    
    use_same_states = False
    epsilon_greedy = 0.0
    state_sample_matrix_input=[]
    if (nargin >= 1):
        epsilon_greedy = varargin[0]
        if (nargin >= 2):
            use_same_states = True
            state_sample_matrix_input = varargin[1]

    reaction_size_to_regulate = num_rxns
    if (epsilon_greedy > 0.0):
        reaction_size_to_regulate = 1
    
    #breakpoint() 
    has_been_up_regulated = 10*np.ones(num_rxns)
    
    estimate_value_vec=np.zeros(num_samples)
    best_action_vec=np.zeros(num_samples)
    state_sample_matrix= np.zeros(shape=(num_rxns, num_samples))
    if (use_same_states == True):
        state_sample_matrix = state_sample_matrix_input
        
    state_delta_s_matrix= np.zeros(shape=(num_rxns, num_samples))
    KQ_f_matrix= np.zeros(shape=(num_rxns, num_samples))
    KQ_r_matrix= np.zeros(shape=(num_rxns, num_samples))
    
    for i in range(0,num_samples):
        state_sample = state_sample_matrix[:,i] #if input, use it
        
        if (use_same_states == False):
            state_sample = np.zeros(num_rxns)
            for sample in range(0,len(state_sample)):
                state_sample[sample] = np.random.uniform(0,1)
            state_sample_matrix[:,i] = state_sample#save for future use
        
        #after sample of regulation is taken, optimize concentrations   
        res_lsq = least_squares(max_entropy_functions.derivatives, v_log_counts_static, method=Method,xtol=1e-15, args=(f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, state_sample))
        v_log_counts = res_lsq.x
        log_metabolites = np.append(v_log_counts, f_log_counts)
        
        rxn_flux = max_entropy_functions.oddsDiff(v_log_counts, f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, state_sample)
        KQ_f = max_entropy_functions.odds(log_metabolites, mu0,S_mat, R_back_mat, P_mat, delta_increment_for_small_concs,Keq_constant);
        KQ_f_matrix[:,i] = KQ_f
        Keq_inverse = np.power(Keq_constant,-1)
        KQ_r = max_entropy_functions.odds(log_metabolites, mu0,-S_mat, P_mat, R_back_mat, delta_increment_for_small_concs,Keq_inverse,-1);
        KQ_r_matrix[:,i] = KQ_r
        
        
        initial_reward = reward_value(KQ_f, KQ_r, state_sample)
        
        delta_S = max_entropy_functions.calc_deltaS(v_log_counts,f_log_counts, S_mat, KQ_f)
        state_delta_s_matrix[:,i] = delta_S
        #delta_S_metab = calc_deltaS_metab(v_log_counts);
        [RR,Jac] = max_entropy_functions.calc_Jac2(v_log_counts, f_log_counts, S_mat, delta_increment_for_small_concs, KQ_f, KQ_r, state_sample)
        A = max_entropy_functions.calc_A(v_log_counts, f_log_counts, S_mat, Jac, state_sample )
        
        #Now we take an action (regulate a reaction) and calculate the new E value that 
        #would occur if we had regulated the state_sample
        
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
            new_res_lsq = least_squares(max_entropy_functions.derivatives, v_log_counts_path, method=Method,xtol=1e-15, args=(f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, path_state_sample))
            v_log_counts_path = new_res_lsq.x
            
            #if using epsilon, reaction_size_to_regulate==1, so cancel exploring starts
            if (epsilon_greedy > 0.0):
                v_log_counts_path = v_log_counts.copy()
                value_path=np.zeros(length_of_path)
                act_path=np.zeros(length_of_path)
                path_state_sample = state_sample.copy()#these will be reset in the path loop
                
                #make iniial change for beginning of path

                #re-optimize
                new_res_lsq = least_squares(max_entropy_functions.derivatives, v_log_counts_path, method=Method,xtol=1e-15, args=(f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, path_state_sample))
                v_log_counts_path = new_res_lsq.x
            
            initial_reward_reset = initial_reward.copy
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
                new_res_lsq = least_squares(max_entropy_functions.derivatives, v_log_counts_path, method=Method,xtol=1e-15, args=(f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, path_state_sample))
                
                v_log_counts_path = new_res_lsq.x
                new_log_metabolites = np.append(v_log_counts_path, f_log_counts)
            
                KQ_f_new = max_entropy_functions.odds(new_log_metabolites, mu0,S_mat, R_back_mat, P_mat, delta_increment_for_small_concs,Keq_constant);
                Keq_inverse = np.power(Keq_constant,-1)
                KQ_r_new = max_entropy_functions.odds(log_metabolites, mu0,-S_mat, P_mat, R_back_mat, delta_increment_for_small_concs,Keq_inverse,-1);
            
                new_delta_S = max_entropy_functions.calc_deltaS(v_log_counts_path,f_log_counts, S_mat, KQ_r_new)
                
                
                current_reward = reward_value(KQ_f_new, KQ_r_new, path_state_sample)
        
               
                value_current_state = state_value(theta_linear, new_delta_S, KQ_f_new, KQ_r_new, path_state_sample)
                
                #Should the reward value be taken at path_state_sample (in first step?)
                #or state_sample??
                #use initial_reward or initial_reward_reset??
                action_value = initial_reward + (gamma) * value_current_state#note, action is using old KQ values
                
                #after using reward, set to next reward since we loop on the path
                initial_reward_reset = current_reward
                q_ = action_value
                    
                #print("reaction")
                #print(React_Choice)
                #print("q")
                #print(q_)
                
                act_path[path]=React_Choice
                value_path[path]=q_
            
            #after the path is set, average to get q(a)
            average_path_val[act] = np.mean(value_path)
            
            #reward only added once: value = reward(current state) + sum(gamma * state_value_j)
            #average_path_val[act] = reward_value(KQ_f, KQ_r, state_sample) + np.mean(value_path)
        
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
    #res_min = minimize(fun_opt_theta, theta_linear, method='nelder-mead', args=(state_delta_s_matrix, estimate_value_vec, KQ_f_matrix, KQ_r_matrix, state_sample_matrix))
    
    def con(t):
        return t
    constraint = {'type': 'ineq', 'fun': con }
    res_min = minimize(fun_opt_theta, theta_linear, constraints=constraint, args=(state_delta_s_matrix, estimate_value_vec, KQ_f_matrix, KQ_r_matrix, state_sample_matrix))
         
    print("estimate_value_vec")
    print(estimate_value_vec)
    print("best_action_vec")
    print(best_action_vec)
    
    val1 = fun_opt_theta(theta_linear, state_delta_s_matrix, estimate_value_vec, KQ_f_matrix, KQ_r_matrix, state_sample_matrix)
    val2 = fun_opt_theta(res_min.x, state_delta_s_matrix, estimate_value_vec, KQ_f_matrix, KQ_r_matrix, state_sample_matrix)
    #print(val1)
    #print(val2)
    #breakpoint()
    return res_min.x  

def update_theta_SGD( step_size, theta_linear, v_log_counts_static, *args):
    #First we sample m states from E
    #First args input is epsilon
    #Second args input is a matrix of states to fit to.
    
    varargin = args
    nargin = len(varargin)
    
    use_same_states = False
    epsilon_greedy = 0.0
    state_sample=[]
    if (nargin >= 1):
        epsilon_greedy = varargin[0]
        if (nargin >= 2):
            use_same_states = True
            state_sample = varargin[1]

    reaction_size_to_regulate = num_rxns
    #if (epsilon_greedy > 0.0):
    #    reaction_size_to_regulate = 1
    
    has_been_up_regulated = 10*np.ones(num_rxns)
    res_lsq = least_squares(max_entropy_functions.derivatives, v_log_counts_static, method=Method,xtol=1e-15, args=(f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, state_sample))
    v_log_counts = res_lsq.x
    log_metabolites = np.append(v_log_counts, f_log_counts)
        
    rxn_flux_init = max_entropy_functions.oddsDiff(v_log_counts, f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, state_sample)
    KQ_f_init = max_entropy_functions.odds(log_metabolites, mu0,S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant);
    
    Keq_inverse = np.power(Keq_constant,-1)
    KQ_r_init = max_entropy_functions.odds(log_metabolites, mu0,-S_mat, P_mat, R_back_mat, delta_increment_for_small_concs, Keq_inverse,-1);
    
    
        
    delta_S_init = max_entropy_functions.calc_deltaS(v_log_counts,f_log_counts, S_mat, KQ_f_init)
    
    [RR,Jac] = max_entropy_functions.calc_Jac2(v_log_counts, f_log_counts, S_mat, delta_increment_for_small_concs, KQ_f_init, KQ_r_init, state_sample)
    A_init = max_entropy_functions.calc_A(v_log_counts, f_log_counts, S_mat, Jac, state_sample )
    
    #Now we take an action (regulate a reaction) and calculate the new E value that 
    #would occur if we had regulated the state_sample
        
    average_path_val=np.zeros(reaction_size_to_regulate)
    last_action_value=np.zeros(reaction_size_to_regulate)
    
    for act in range(0, reaction_size_to_regulate):
        
        #each action begins from the initial state, so ocpy necessary variables
        A_act = A_init.copy()
        act_state_sample = state_sample.copy()
        rxn_flux_act = rxn_flux_init.copy()
        #exploring starts method
        value_path = np.zeros(length_of_path)
        act_path = np.zeros(length_of_path)
        
        #make iniial change for beginning of path
        React_Choice = act
        newE_act = max_entropy_functions.calc_reg_E_step(act_state_sample, React_Choice, nvar, 
                               v_log_counts, f_log_counts, desired_conc, S_mat, A_act, 
                               rxn_flux_act, KQ_f_init, False, has_been_up_regulated)
        
        act_state_sample[React_Choice] = newE_act
            
        #re-optimize
        new_res_lsq = least_squares(max_entropy_functions.derivatives, v_log_counts, method=Method,xtol=1e-15, args=(f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, act_state_sample))
        v_log_counts_act = new_res_lsq.x
        rxn_flux_act = max_entropy_functions.oddsDiff(v_log_counts_act, f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, act_state_sample)
        
        log_metabolites_act = np.append(v_log_counts_act, f_log_counts)
    
        KQ_f_act = max_entropy_functions.odds(log_metabolites_act, mu0,S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant);
        KQ_r_act = max_entropy_functions.odds(log_metabolites_act, mu0,-S_mat, P_mat, R_back_mat, delta_increment_for_small_concs, Keq_inverse,-1);
      
        delta_S_act = max_entropy_functions.calc_deltaS(v_log_counts_act,f_log_counts, S_mat, KQ_f_act)
    
        [RR_act, Jac_act] = max_entropy_functions.calc_Jac2(v_log_counts_act, f_log_counts, S_mat, delta_increment_for_small_concs, KQ_f_act, KQ_r_act, act_state_sample)
        A_act = max_entropy_functions.calc_A(v_log_counts_act, f_log_counts, S_mat, Jac_act, act_state_sample )
    
        #if using epsilon, reaction_size_to_regulate==1, so cancel exploring starts
        #if (epsilon_greedy > 0.0):
        #    value_path=np.zeros(length_of_path)
        #   act_path=np.zeros(length_of_path)
        #   act_state_sample = state_sample.copy()#these will be reset in the path loop
            
            #make iniial change for beginning of path
            #re-optimize
        #    new_res_lsq = least_squares(max_entropy_functions.derivatives, v_log_counts, method=Method,xtol=1e-15, args=(f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, act_state_sample))
        #    v_log_counts_act = new_res_lsq.x
        
        
        #WARNING: THESE VARIABLES RESET VALUE IN PATH LOOP BELOW
        #For each path, we will reset the variables as we move along the path. 
        #For that reason, these variables must be set outside the path loop 
        initial_reward_act = reward_value(KQ_f_act, KQ_r_act, act_state_sample)
        
        initial_reward_reset = initial_reward_act.copy
        
        v_log_counts_path = v_log_counts_act.copy()
        path_state_sample = act_state_sample.copy()
        A_path = A_act.copy()  
        rxn_flux_path = rxn_flux_act.copy()
        
        KQ_f_path = KQ_f_act.copy()
        
        KQ_r_path = KQ_r_act.copy()
        value_current_state_act = state_value(theta_linear, delta_S_act, KQ_f_act, KQ_r_act, act_state_sample)
                
        #action_value_init_act = initial_reward_act + gamma * value_current_state_act
        action_value_init_act = initial_reward_act
        
        #now generate the next path based on act_state_sample
        for path in range(0,length_of_path):

            React_Choice = policy_function(path_state_sample, theta_linear, v_log_counts_path, epsilon_greedy)#regulate each reaction.
            
            newE_path = max_entropy_functions.calc_reg_E_step(path_state_sample, React_Choice, nvar, 
                                   v_log_counts_path, f_log_counts, desired_conc, S_mat, A_path,
                                   rxn_flux_path, KQ_f_path, False, has_been_up_regulated)
                
            #now generate the next step based on newE
            
            path_state_sample[React_Choice] = newE_path
            #print(path_state_sample)
            #breakpoint()
            #re-optimize
            new_res_lsq = least_squares(max_entropy_functions.derivatives, v_log_counts_path, method=Method,xtol=1e-15, args=(f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, path_state_sample))
            
            v_log_counts_path = new_res_lsq.x
            log_metabolites_path = np.append(v_log_counts_path, f_log_counts)
        
            rxn_flux_path = max_entropy_functions.oddsDiff(v_log_counts_path, f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, path_state_sample)
        
            KQ_f_path = max_entropy_functions.odds(log_metabolites_path, mu0,S_mat, R_back_mat, P_mat, delta_increment_for_small_concs,Keq_constant);
            KQ_r_path = max_entropy_functions.odds(log_metabolites_path, mu0,-S_mat, P_mat, R_back_mat, delta_increment_for_small_concs,Keq_inverse,-1);
            
            [RR_path, Jac_path] = max_entropy_functions.calc_Jac2(v_log_counts_path, f_log_counts, S_mat, delta_increment_for_small_concs, KQ_f_path, KQ_r_path, path_state_sample)
            A_path = max_entropy_functions.calc_A(v_log_counts_path, f_log_counts, S_mat, Jac_path, path_state_sample )
    
        
            path_delta_S = max_entropy_functions.calc_deltaS(v_log_counts_path, f_log_counts, S_mat, KQ_f_path)
            
            current_reward = reward_value(KQ_f_path, KQ_r_path, path_state_sample)
        
            value_current_state_path = state_value(theta_linear, path_delta_S, KQ_f_path, KQ_r_path, path_state_sample)
                
            #Should the reward value be taken at path_state_sample (in first step?)
            #or state_sample??
            
            #AVE VERSION :
            #action_value = (current_reward + (gamma) * value_current_state_path)#note, action is using old KQ values
            
            #ALTERNATE
            #path starts at zero, so add one
            #action_value = (gamma**(path+1)) * (current_reward + (gamma) * value_current_state_path)#note, action is using old KQ values
            
            action_value = (gamma**(path+1)) * (current_reward)
            if (path == (length_of_path-1)):
                action_value = (gamma**(path+1)) * value_current_state_path #last value should be state
                #print(action_value)
                last_action_value[act]=action_value
                
                #after using reward, set to next reward since we loop on the path
            initial_reward_reset = current_reward     

            act_path[path] = React_Choice
            value_path[path] = action_value
            #breakpoint()
            #after the path is set, average to get q(a)
        
        #the last action value in value_path is gamma^n * (estimate state value)
        value_path_including_init = np.append(action_value_init_act, value_path)
        
        #AVE VERSION
        #average_path_val[act] = np.mean(value_path_including_init)
        
        #ALTERNATE
        average_path_val[act] = np.sum(value_path_including_init)
            
        #after path is generated:
        #average_path_val[act] holds reward
        #value_path[:] holds action values: gamma^n * value
        #sum everything for that path
        #average_path_val[act] += np.sum(value_path)
        
    #after every possible action for the state had been taken,
    #choose the optimal path for the sample. 
    
    #estimate value is r_(t+1) + sum{gamma^n * x_(t+1)_j }
    #breakpoint()
    #print(last_action_value)
    print("min->max order of best last actions")
    print(np.argsort(last_action_value))
    print("min->max order of best actions")
    print(np.argsort(average_path_val))
    #breakpoint()
    print("last_action_value")
    print(last_action_value)
    print("average_path_val")
    print(average_path_val)
    estimate_value = np.max(average_path_val)   #estimate_value now represents the target value of the best possible action from the trail state.
    best_action = np.argmax(average_path_val)
    
    print("best_action")
    print(best_action)
    #print("average_path_val")
    #print(average_path_val)
    #previous value is using x_t
    previous_value = state_value(theta_linear, delta_S_init, KQ_f_init, KQ_r_init, state_sample)
    
    step_value = step_size * (estimate_value - previous_value)
    
    vec = entropy_production_rate_vec(KQ_f_init, KQ_r_init, state_sample)
    x_t = -(vec) #this is -EPR before summing it. 

    #new_theta = theta_linear + step_value * (x_t)
    new_theta = SGD_UPDATE(theta_linear, step_size, estimate_value, previous_value, x_t)
    #enforce positive values
    #new_theta[new_theta<0] = 0
    return new_theta


def update_theta_SGD_TD( threshold,nn_model,loss_fn, optimizer, step_size, n_back_step, theta_linear, v_log_counts_static, state_sample, epsilon_greedy):
    #First we sample m states from E
    #First args input is epsilon
    #Second args input is a matrix of states to fit to.
    
    sum_reward_episode=0
    end_of_path = 1000 #this is the maximum length a path can take
    KQ_f_matrix= np.zeros(shape=(num_rxns, end_of_path+1))
    KQ_r_matrix= np.zeros(shape=(num_rxns, end_of_path+1))
    states_matrix= np.zeros(shape=(num_rxns, end_of_path+1))
    delta_S_matrix = np.zeros(shape=(num_rxns, end_of_path+1))
    
    delta_S_metab_matrix = np.zeros(shape=(nvar, end_of_path+1))
    v_log_counts_matrix = np.zeros(shape=(nvar, end_of_path+1))
    
    alt_value_matrix = np.zeros(shape=(num_rxns, end_of_path+1))

    has_been_up_regulated = 10*np.ones(num_rxns)
    res_lsq = least_squares(max_entropy_functions.derivatives, v_log_counts_static, method=Method,xtol=1e-15, args=(f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, state_sample))
    v_log_counts = res_lsq.x
    log_metabolites = np.append(v_log_counts, f_log_counts)
        
    rxn_flux_init = max_entropy_functions.oddsDiff(v_log_counts, f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, state_sample)
    KQ_f_init = max_entropy_functions.odds(log_metabolites, mu0,S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant);
    
    Keq_inverse = np.power(Keq_constant,-1)
    KQ_r_init = max_entropy_functions.odds(log_metabolites, mu0,-S_mat, P_mat, R_back_mat, delta_increment_for_small_concs, Keq_inverse,-1);
    
        
    delta_S_init = max_entropy_functions.calc_deltaS(v_log_counts,f_log_counts, S_mat, KQ_f_init)
    
    [RR,Jac] = max_entropy_functions.calc_Jac2(v_log_counts, f_log_counts, S_mat, delta_increment_for_small_concs, KQ_f_init, KQ_r_init, state_sample)
    A_init = max_entropy_functions.calc_A(v_log_counts, f_log_counts, S_mat, Jac, state_sample )
            
    
    delta_S_metab_init = max_entropy_functions.calc_deltaS_metab(v_log_counts);
        
    [ccc,fcc] = max_entropy_functions.conc_flux_control_coeff(nvar, A_init, S_mat, rxn_flux_init, RR)
        
    
    alt_value_vec = delta_delta_s_vec(delta_S_init, delta_S_metab_init, ccc, KQ_f_init, state_sample, v_log_counts)
    
    initial_reward = np.sum(delta_S_init[delta_S_init>0])
    #reward_value(v_log_counts,v_log_counts, KQ_f_init, KQ_r_init, state_sample,\
    #                              delta_S_init,delta_S_init)
    
    v_log_counts_matrix[:,0] = v_log_counts.copy()
    states_matrix[:,0] = state_sample.copy()
    delta_S_matrix[:,0] = delta_S_init.copy()
    delta_S_metab_matrix[:,0] = delta_S_metab_init.copy()
    KQ_f_matrix[:,0] = KQ_f_init.copy()
    KQ_r_matrix[:,0] = KQ_r_init.copy()
    alt_value_matrix[:,0] = alt_value_vec.copy()
    initial_reward_reset = initial_reward
    
    v_log_counts_path = v_log_counts.copy()
    v_log_counts_path_previous = v_log_counts.copy()
    path_state_sample = state_sample.copy() #this is modified
    A_path = A_init.copy()  
    rxn_flux_path = rxn_flux_init.copy()
        
    KQ_f_path = KQ_f_init.copy()
    
    KQ_r_path = KQ_r_init.copy()
                
    value_path = np.zeros(end_of_path+1)   
    act_path = np.zeros(end_of_path+1) 
    
    value_path[0] = initial_reward
    state_tau=[]
    KQ_f_tau=[]
    KQ_r_tau=[]
    delta_S_tau=[]
    path_delta_S_previous = delta_S_init.copy()
    path_delta_S = delta_S_init.copy()
    #breakpoint()
    t = 0
    tau_true = 0
    while tau_true < (end_of_path - 1):
        
        if (t < end_of_path):

            [React_Choice, policy_reward] = policy_function(nn_model, path_state_sample, theta_linear, v_log_counts_path, epsilon_greedy)#regulate each reaction.
            if (React_Choice == -1):
                end_of_path=t
                break
                #breakpoint()
                
            
            newE_path = max_entropy_functions.calc_reg_E_step(path_state_sample, React_Choice, nvar, 
                               v_log_counts_path, f_log_counts, desired_conc, S_mat, A_path,
                               rxn_flux_path, KQ_f_path, False, has_been_up_regulated,\
                               path_delta_S)
                
            #now generate the next step based on newE
                
            path_state_sample[React_Choice] = newE_path
            #re-optimize
            new_res_lsq = least_squares(max_entropy_functions.derivatives, v_log_counts_path, method=Method,xtol=1e-15, args=(f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, path_state_sample))
                
            v_log_counts_path = new_res_lsq.x
            log_metabolites_path = np.append(v_log_counts_path, f_log_counts)
        
            rxn_flux_path = max_entropy_functions.oddsDiff(v_log_counts_path, f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, path_state_sample)
            
            KQ_f_path = max_entropy_functions.odds(log_metabolites_path, mu0,S_mat, R_back_mat, P_mat, delta_increment_for_small_concs,Keq_constant);
            KQ_r_path = max_entropy_functions.odds(log_metabolites_path, mu0,-S_mat, P_mat, R_back_mat, delta_increment_for_small_concs,Keq_inverse,-1);
                
            [RR_path, Jac_path] = max_entropy_functions.calc_Jac2(v_log_counts_path, f_log_counts, S_mat, delta_increment_for_small_concs, KQ_f_path, KQ_r_path, path_state_sample)
            A_path = max_entropy_functions.calc_A(v_log_counts_path, f_log_counts, S_mat, Jac_path, path_state_sample )
        
            epr_path = max_entropy_functions.entropy_production_rate(KQ_f_path, KQ_r_path, path_state_sample)
            
            path_delta_S = max_entropy_functions.calc_deltaS(v_log_counts_path, f_log_counts, S_mat, KQ_f_path)
            
            delta_S_metab_path = max_entropy_functions.calc_deltaS_metab(v_log_counts_path)
            
            [ccc_path, fcc] = max_entropy_functions.conc_flux_control_coeff(nvar, A_path, S_mat, rxn_flux_path, RR_path)
            
            alt_value_path_vec = delta_delta_s_vec(path_delta_S, delta_S_metab_path, ccc_path, KQ_f_path, path_state_sample, v_log_counts_path)
    
            action_value = reward_value(v_log_counts_path, v_log_counts_path_previous, KQ_f_path, KQ_r_path, path_state_sample,\
                                        path_delta_S,path_delta_S_previous)
            
            sum_reward_episode+=action_value
            #print("REWARD")
            #print(action_value)
            
            if (np.abs(policy_reward-action_value) >0.01):
                breakpoint()
            path_delta_S_previous = path_delta_S.copy()#set after calculation
            v_log_counts_path_previous = v_log_counts_path.copy()

            act_path[t+1] = React_Choice
            #print("setting  ")
            #print(t+1)
            value_path[t+1] = action_value
            states_matrix[:,t+1] = path_state_sample.copy()
            KQ_f_matrix[:,t+1] = KQ_f_path.copy()
            KQ_r_matrix[:,t+1] = KQ_r_path.copy()
            delta_S_matrix[:,t+1] = path_delta_S.copy()
            delta_S_metab_matrix[:,t+1] = delta_S_metab_path.copy()
            v_log_counts_matrix[:,t+1] = v_log_counts_path.copy()
            alt_value_matrix[:,t+1] = alt_value_path_vec.copy()
            
            #last_state = states_matrix[:,t]
            current_state = states_matrix[:,t+1]
            
            

            #We stop the path if we have no more positive loss function values, or if we revisit a state. 
            if ((path_delta_S<=0.0).all()):
                end_of_path=t+1
                
                print("**************************************Path Length ds<0******************************************")
                print(end_of_path)
                print("Final STATE")
                print(path_state_sample)
                print(rxn_flux_path)
                print(epr_path)
                

            for state in range(0,t+1):
                last_state = states_matrix[:,state]
                if ((current_state==last_state).all()):
                    end_of_path=t+1
                    
                    print("**************************************Path Length******************************************")
                    print(end_of_path)
                    print("Final STATE")
                    print(path_state_sample)
                    print(rxn_flux_path)
                    print(epr_path)

        
        #tau = t - n_back_step + 1
        #instead of estimating state tau = t-n+1
        #collect estimation for all states from 0 to t.
        #using temp_t taking values from [0,t]. This will take an estimate for all the 
        #states that we have visiting in the current path. We'll use a random selection of them to train on. 
        
        #idea: Faster to select states, then generate estimate values after selection. 
        
        #number of samples to use cannot exceed the number of states visited so far: (t-n_back_step+1)
        
        #batch_size = min(2*n_back_step, t - n_back_step + 1)
        

        ##BEGIN OLD METHOD
        tau_true = t - n_back_step + 1
        tau = t - n_back_step + 1
                
        if (tau >=0):
            #breakpoint()
            estimate_value = 0
# =============================================================================
#                     
#             x = torch.zeros(1, 1, Keq_constant.size)
#             y = torch.zeros(1, 1, 1)
#             
# =============================================================================
            
            x = torch.zeros(1, 1, Keq_constant.size)
            y = torch.zeros(1, 1, 1)
            
            #sum must go from i = tau+1 until the value min( tau+n, LOP)
            #We therefore need to increment the range values by one. 
            for i in range(tau + 1, min(tau + n_back_step, end_of_path)+1):    
                estimate_value += (gamma**(i-tau-1)) * value_path[i] 
                #print("estimate")
                #print(estimate_value)
            if ((tau + n_back_step) < end_of_path):
                state_tau_n = states_matrix[:, tau + n_back_step].copy()
                KQ_f_tau_n = KQ_f_matrix[:, tau + n_back_step].copy()
                KQ_r_tau_n = KQ_r_matrix[:, tau + n_back_step].copy()
                delta_S_tau_n = delta_S_matrix[:, tau + n_back_step].copy()
                    
                v_log_counts_tau_n = v_log_counts_matrix[:, tau + n_back_step].copy()
                alt_value_tau_n = alt_value_matrix[:, tau + n_back_step].copy()
                delta_S_metab_tau_n = delta_S_metab_matrix[:, tau + n_back_step].copy()
                value_tau_n = state_value(nn_model,theta_linear, delta_S_tau_n, delta_S_metab_tau_n, KQ_f_tau_n, KQ_r_tau_n, state_tau_n, v_log_counts_tau_n)
                        
                estimate_value += (gamma**(n_back_step)) * value_tau_n
                    #print("estimate_tau")
                    #print(estimate_value)
                    
# =============================================================================
#             state_tau = states_matrix[:, tau].copy()
#             for rxn in range (0,state_tau.size):
#                 x[0,0,rxn] = state_tau[rxn].copy()
#                 
# =============================================================================
            
            #v_log_counts_tau = v_log_counts_matrix[:, tau].copy()
            
            state_tau = states_matrix[:, tau].copy()
            

            for i in range(0,state_tau.size):
                val = state_tau[i].copy()
                val = val**(0.5)
                x[0][0][i] = np.log(1.0 + np.exp(20.0)*val)
                
# =============================================================================
#             for met in range (0,v_log_counts_tau.size):
#                 x[0,0,met] = v_log_counts_tau[met].copy()
#                 
# =============================================================================
            y[0][0][0] = estimate_value
                
            y_pred = nn_model(x)
                            
            loss = loss_fn(y_pred, y)
            print(loss.item())
            nn_model.zero_grad()
            loss.backward(retain_graph=True)
                #print("difference, target -> pred")
                #print(y)
                #print(y_pred)
                #breakpoint()
        
            
            optimizer.step()
            optimizer.zero_grad()
        
        ##END OLD METHOD
        
        
        ##NEW BATCH METHOD
        ##NEW BATCH METHOD
        ##NEW BATCH METHOD
# =============================================================================
# 
#         tau_true = t-n_back_step+1
#         loss_max = np.inf
#         
#         while (loss_max > threshold):
#             no_update=True
#             #train on all current data points. 
#             old_loss=0.0
#             current_loss=0.0
#             print("training")
#             print(t)
#             min_val=0 #standard method uses min_t. 
#             for test_t in range(min_val,t+1):
#                 
#                 tau = t - n_back_step + 1
#                 
#                 if (tau >=0):
#                     no_update=False
#                     #breakpoint()
#                     estimate_value = 0
#                     
#                     x = torch.zeros(1, 1, Keq_constant.size)
#                     y = torch.zeros(1, 1, 1)
#                     #sum must go from i = tau+1 until the value min( tau+n, LOP)
#                     #We therefore need to increment the range values by one. 
#                     for i in range(tau + 1, min(tau + n_back_step, end_of_path)+1):    
#                         estimate_value += (gamma**(i-tau-1)) * value_path[i] 
#                         #print("estimate")
#                         #print(estimate_value)
#                     if ((tau + n_back_step) < end_of_path):
#                         state_tau_n = states_matrix[:, tau + n_back_step].copy()
#                         KQ_f_tau_n = KQ_f_matrix[:, tau + n_back_step].copy()
#                         KQ_r_tau_n = KQ_r_matrix[:, tau + n_back_step].copy()
#                         delta_S_tau_n = delta_S_matrix[:, tau + n_back_step].copy()
#                         
#                         alt_value_tau_n = alt_value_matrix[:, tau + n_back_step].copy()
#                         delta_S_metab_tau_n = delta_S_metab_matrix[:, tau + n_back_step].copy()
#                         value_tau_n = state_value(nn_model,theta_linear, delta_S_tau_n, delta_S_metab_tau_n, KQ_f_tau_n, KQ_r_tau_n, state_tau_n, alt_value_tau_n)
#                         
#                         estimate_value += (gamma**(n_back_step)) * value_tau_n
#                         #print("estimate_tau")
#                         #print(estimate_value)
#                     state_tau = states_matrix[:, tau].copy()
#                     for rxn in range (0,Keq_constant.size):
#                         x[0,0,rxn] = state_tau[rxn].copy()
#                     y[0][0][0] = estimate_value
#                 
#                     y_pred = nn_model(x)
#                             
#                     loss = loss_fn(y_pred, y)
#                     print(loss.item())
#                     nn_model.zero_grad()
#                     loss.backward(retain_graph=True)
#                     #print("difference, target -> pred")
#                     #print(y)
#                     #print(y_pred)
#                     #breakpoint()
#         
#             
#                     optimizer.step()
#                     optimizer.zero_grad()
#                     
#                     current_loss = loss.item()
#                     if (current_loss>old_loss):
#                         loss_max = current_loss
#                         old_loss = current_loss
#             
#             #print("max loss")
#             #print(loss_max)
#             #breakpoint()
#             if (no_update==True):
#                 loss_max=-np.inf
# =============================================================================
    
        ##NEW BATCH METhOD
        ##NEW BATCH METhOD
        ##NEW BATCH METhOD

        t+=1
    
    return sum_reward_episode

def SGD_UPDATE(theta_linear, step_size, estimate_value, previous_value, x_t):

    step_value = step_size * (estimate_value - previous_value)
    
    #vec = entropy_production_rate_vec(KQ_f_init, KQ_r_init, state_sample)
    #x_t = -(vec) #this is -EPR before summing it. 

    new_theta = theta_linear + step_value * (x_t)
    
    #new_theta[new_theta<0] = 0
    return new_theta
#%%
#input state, return action
    
def policy_function(nn_model,state, theta_linear, v_log_counts_path, *args ):
    #last input argument should be epsilon for use when using greedy-epsilon algorithm. 
    
    varargin = args
    nargin = len(varargin)
    epsilon_greedy = 0.0
    if (nargin == 1):
        epsilon_greedy = varargin[0]
        
    rxn_choices = [i for i in range(num_rxns)]
    #MAYBE REMOVE
    has_been_up_regulated = 10*np.ones(num_rxns)
    
    #res_lsq = least_squares(max_entropy_functions.derivatives, v_log_counts_path, method=Method,xtol=1e-15, args=(f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, state))
    
    #if (res_lsq.optimality > 0.00001):
    #    breakpoint()
    v_log_counts = v_log_counts_path.copy()#res_lsq.x
    log_metabolites = np.append(v_log_counts, f_log_counts)
        
    rxn_flux = max_entropy_functions.oddsDiff(v_log_counts, f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, state)
    KQ_f = max_entropy_functions.odds(log_metabolites, mu0,S_mat, R_back_mat, P_mat, delta_increment_for_small_concs,Keq_constant);
    Keq_inverse = np.power(Keq_constant,-1)
    KQ_r = max_entropy_functions.odds(log_metabolites, mu0,-S_mat, P_mat, R_back_mat, delta_increment_for_small_concs,Keq_inverse,-1);
    
    [RR,Jac] = max_entropy_functions.calc_Jac2(v_log_counts, f_log_counts, S_mat, delta_increment_for_small_concs, KQ_f, KQ_r, state)
    A = max_entropy_functions.calc_A(v_log_counts, f_log_counts, S_mat, Jac, state )
    
    delta_S = max_entropy_functions.calc_deltaS(v_log_counts,f_log_counts, S_mat, KQ_f)
    
    delta_S_metab = max_entropy_functions.calc_deltaS_metab(v_log_counts);
        
    [ccc,fcc] = max_entropy_functions.conc_flux_control_coeff(nvar, A, S_mat, rxn_flux, RR)
        
    alt_value_vec = delta_delta_s_vec(delta_S, delta_S_metab, ccc, KQ_f, state, v_log_counts)
    
    
    
    reaction_size_to_regulate = num_rxns
    init_action_val = -np.inf
    action_choice=1
    #breakpoint()
    
    #initial_reward = reward_value(v_log_counts,v_log_counts, KQ_f, KQ_r, state)
        
    action_value_vec = np.zeros(reaction_size_to_regulate)
    state_value_vec = np.zeros(reaction_size_to_regulate)
    E_test_vec = np.zeros(reaction_size_to_regulate)
    old_E_test_vec = np.zeros(reaction_size_to_regulate)
    current_reward_vec = np.zeros(reaction_size_to_regulate)
    #print("BEGIN ACTIONS")
    
    for act in range(0,reaction_size_to_regulate):
        #print("in policy_function action loop")
        #print("v_log_counts")
        #print(np.max(v_log_counts))
        #print(np.min(v_log_counts))
        #print("end:")
        React_Choice = act#regulate each reaction.
        
        old_E = state[act]
        newE = max_entropy_functions.calc_reg_E_step(state, React_Choice, nvar, v_log_counts, f_log_counts,
                               desired_conc, S_mat, A, rxn_flux, KQ_f, False, has_been_up_regulated,\
                               delta_S)
        
        
        #if (( newE <1 ) and (newE > 0) and (newE != oldE)):
            
        #now generate the next step based on newE
        trial_state_sample = state.copy()#DO NOT MODIFY STATE
        trial_state_sample[React_Choice] = newE
            
        #re-optimize
        new_res_lsq = least_squares(max_entropy_functions.derivatives, v_log_counts, method=Method,xtol=1e-15, args=(f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, trial_state_sample))
            
        new_v_log_counts = new_res_lsq.x
        #breakpoint()
        new_log_metabolites = np.append(new_v_log_counts, f_log_counts)
        rxn_flux_new = max_entropy_functions.oddsDiff(new_v_log_counts, f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, trial_state_sample)
    
        KQ_f_new = max_entropy_functions.odds(new_log_metabolites, mu0,S_mat, R_back_mat, P_mat, delta_increment_for_small_concs,Keq_constant);    
        Keq_inverse = np.power(Keq_constant,-1)
        KQ_r_new = max_entropy_functions.odds(new_log_metabolites, mu0,-S_mat, P_mat, R_back_mat, delta_increment_for_small_concs,Keq_inverse,-1);

        
        delta_S_new = max_entropy_functions.calc_deltaS(new_v_log_counts,f_log_counts, S_mat, KQ_f_new)
    
        delta_S_metab_new = max_entropy_functions.calc_deltaS_metab(new_v_log_counts);
        
        [RR_new,Jac_new] = max_entropy_functions.calc_Jac2(new_v_log_counts, f_log_counts, S_mat, delta_increment_for_small_concs, KQ_f_new, KQ_r_new, trial_state_sample)
    
        A_new = max_entropy_functions.calc_A(new_v_log_counts, f_log_counts, S_mat, Jac_new, trial_state_sample )
    

        [ccc_new,fcc_new] = max_entropy_functions.conc_flux_control_coeff(nvar, A_new, S_mat, rxn_flux_new, RR_new)
        
        alt_value_vec_new = delta_delta_s_vec(delta_S_new, delta_S_metab_new, ccc_new, KQ_f_new, trial_state_sample, new_v_log_counts)
    
        #if (act == 13):
        #    breakpoint()
    

        value_current_state = state_value(nn_model, theta_linear, delta_S_new, delta_S_metab_new, KQ_f_new, KQ_r_new, trial_state_sample, new_v_log_counts)
        #breakpoint()
        #print("value_current_state")
        #print(value_current_state)
        current_reward = reward_value(new_v_log_counts, v_log_counts, KQ_f_new, KQ_r_new, trial_state_sample,\
                                      delta_S_new, delta_S)

        #best action is defined by the current chosen state and value 
        #should it be the next state value after?
        action_value = current_reward + (gamma) * value_current_state #note, action is using old KQ values
        
        
        action_value_vec[act] = action_value
        old_E_test_vec[act] = old_E
        E_test_vec[act] = newE
        state_value_vec[act] = value_current_state
        current_reward_vec[act] = current_reward #Should have smaller EPR
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
            #print("actoin_value")
            #print(action_value)
        
        if (current_reward == penalty_reward):
            rxn_choices.remove(act)
            #breakpoint()
    
    
    #randomly choose one of the top choices if there is a tie. 
    action_choice = np.random.choice(np.flatnonzero(action_value_vec == action_value_vec.max()))
    
    #action_value_vec
# =============================================================================
#     print("min->max order of best actions")
#     print(np.argsort(action_value_vec))
#     print("action_value_vec")
#     print(action_value_vec)
#     print(action_value_vec[action_choice])
#     print(current_reward_vec[action_choice])
#     if (current_reward_vec[action_choice]<-1):
#         breakpoint()
# 
# 
# 
#     print("oldE_test_vec")
#     print(old_E_test_vec)
#     print("current_reward_vec")
#     print(current_reward_vec)
#     
#     print("state_value_vec")
#     print(state_value_vec)
#     print("E_test_vec")
#     
#     print(E_test_vec)
# =============================================================================
    #breakpoint()
    #breakpoint()
    #rxn_choices.remove(action_choice)
    #breakpoint()
    unif_rand = np.random.uniform(0,1)
    if (unif_rand < epsilon_greedy):
        if (len(rxn_choices)>1):
            rxn_choices.remove(action_choice)
        print("USING EPSILON GREEDY")
        print(action_choice)       
        random_choice = random.choice(rxn_choices)
        action_choice = random_choice

        print("replaced by")
        print(random_choice)
        
    #if ((current_reward_vec<0).all()):
    #    print("STOP SIMULAITON, no more rewards")
    #    #breakpoint()
    #    action_choice=-1
    if ([action_choice] == 20):
        breakpoint()
    
        
    
    return [action_choice,current_reward_vec[action_choice]]