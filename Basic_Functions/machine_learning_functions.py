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
import multiprocessing as mp
from multiprocessing import Pool

Method = 'trf'

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
def entropy_production_rate_vec(KQ_f, KQ_r, E_Regulation, *args):
    
    varargin = args
    nargin = len(varargin)
    method = 0
    theta=[]
    if (nargin == 1):
        method = 1
        theta=varargin[0].copy()
    if (nargin == len(E_Regulation)):
        method = 1
        theta=varargin.copy()
    
    #print(theta)
    KQ_f_reg = E_Regulation * KQ_f
    KQ_r_reg = E_Regulation * KQ_r
    sumOdds = np.sum(KQ_f_reg) + np.sum(KQ_r_reg)
    
    val = []
    if (method == 0):
        val = KQ_f_reg/sumOdds * np.log(KQ_f) + KQ_r_reg/sumOdds * np.log(KQ_r)
    else:
        val = theta * (KQ_f_reg/sumOdds * np.log(KQ_f) + KQ_r_reg/sumOdds * np.log(KQ_r))
    
    #WHY NO MINUS?
    EPR_matlab = +np.sum(KQ_f_reg*np.log(KQ_f))/sumOdds + np.sum(KQ_r_reg*np.log(KQ_r))/sumOdds
    
    #print('val')
    #print(np.sum(val))
    #print("m")
    #print(entropy_production_rate)
    if ((np.abs(EPR_matlab-np.sum(val) > 1e-5)) and (nargin==0) ):
        breakpoint()
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
def state_value(theta_linear, delta_s, KQ_f, KQ_r, E_Regulation):
    
    
    #Problem::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #in this function, KQ_f_reg is not changing much b/c E_Regulation is multiplying it.
    #when it does change, it does so by proportions
    #This makes theta_linear have no effect on individual reactions
    
    #As is, theta chooses lowest overall flux between forward and backward
    
    KQ_f_reg = E_Regulation * KQ_f
    KQ_r_reg = E_Regulation * KQ_r
    
    
    sumOdds = np.sum(KQ_f_reg) + np.sum(KQ_r_reg)
    norm_LR = np.sum(theta_linear * KQ_f_reg/sumOdds * np.log(KQ_f))
    norm_RL = np.sum(theta_linear * KQ_r_reg/sumOdds * np.log(KQ_r))
   
    #print("norm_LR")
    #print(norm_LR)
    #print("norm_RL")
    #print(norm_RL)
    vec = entropy_production_rate_vec(KQ_f, KQ_r, E_Regulation, theta_linear)
    
    val = -np.sum(vec)
    val_prev = -(norm_LR + norm_RL)
    if (np.abs(val - val_prev)>1e-5):
        breakpoint()

    return val

#define reward as entropy production rate
    #usually positive, so 
def reward_value(KQ_f, KQ_r, E_Regulation):
    val = -entropy_production_rate(KQ_f, KQ_r, E_Regulation)
    return val

     
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
        #delta_S_metab = calc_delaS_metab(v_log_counts);
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


def update_theta_SGD_TD( step_size, n_back_step, theta_linear, v_log_counts_static, state_sample, epsilon_greedy):
    #First we sample m states from E
    #First args input is epsilon
    #Second args input is a matrix of states to fit to.
    KQ_f_matrix= np.zeros(shape=(num_rxns, length_of_path+1))
    KQ_r_matrix= np.zeros(shape=(num_rxns, length_of_path+1))
    states_matrix= np.zeros(shape=(num_rxns, length_of_path+1))
    delta_S_matrix = np.zeros(shape=(num_rxns, length_of_path+1))

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
    
    
    initial_reward = reward_value(KQ_f_init, KQ_r_init, state_sample)
    
    
    states_matrix[:,0] = state_sample
    delta_S_matrix[:,0] = delta_S_init
    KQ_f_matrix[:,0] = KQ_f_init
    KQ_r_matrix[:,0] = KQ_r_init
        
    initial_reward_reset = initial_reward.copy
    
    v_log_counts_path = v_log_counts.copy()
    path_state_sample = state_sample.copy() #this is modified
    A_path = A_init.copy()  
    rxn_flux_path = rxn_flux_init.copy()
        
    KQ_f_path = KQ_f_init.copy()
    
    KQ_r_path = KQ_r_init.copy()
                
    value_path = np.zeros(length_of_path+1)   
    act_path = np.zeros(length_of_path+1) 
    value_path[0] = initial_reward
    state_tau=[]
    KQ_f_tau=[]
    KQ_r_tau=[]
    delta_S_tau=[]
    
    t = 0
    tau = 0
    while tau < (length_of_path - 1):
        if (t < length_of_path):

            React_Choice = policy_function(path_state_sample, theta_linear, v_log_counts_path, epsilon_greedy)#regulate each reaction.
            
            newE_path = max_entropy_functions.calc_reg_E_step(path_state_sample, React_Choice, nvar, 
                               v_log_counts_path, f_log_counts, desired_conc, S_mat, A_path,
                               rxn_flux_path, KQ_f_path, False, has_been_up_regulated)
                
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
        
            
            path_delta_S = max_entropy_functions.calc_deltaS(v_log_counts_path, f_log_counts, S_mat, KQ_f_path)
            
            action_value = reward_value(KQ_f_path, KQ_r_path, path_state_sample)
            
            #if (path == (length_of_path-1)):
                #action_value = state_value(theta_linear, path_delta_S, KQ_f_path, KQ_r_path, path_state_sample)
            
            act_path[t+1] = React_Choice
            value_path[t+1] = action_value
            states_matrix[:,t+1] = path_state_sample.copy()
            KQ_f_matrix[:,t+1] = KQ_f_path.copy()
            KQ_r_matrix[:,t+1] = KQ_r_path.copy()
            delta_S_matrix[:,t+1] = path_delta_S.copy()
            
        tau = t - n_back_step + 1
            
        if (tau >=0):
            estimate_value = 0
            
            #sum must go from i = tau+1 until the value min( tau+n, LOP)
            #We therefore need to increment the range values by one. 
            for i in range(tau + 1, min(tau + n_back_step+1, length_of_path+1)):    
                estimate_value += (gamma**(i-tau-1)) * value_path[i] 
                
            if ((tau + n_back_step) < length_of_path):
                state_tau_n = states_matrix[:, tau + n_back_step].copy()
                KQ_f_tau_n = KQ_f_matrix[:, tau + n_back_step].copy()
                KQ_r_tau_n = KQ_r_matrix[:, tau + n_back_step].copy()
                delta_S_tau_n = delta_S_matrix[:, tau + n_back_step].copy()
                value_tau_n = state_value(theta_linear, delta_S_tau_n, KQ_f_tau_n, KQ_r_tau_n, state_tau_n)
        
                estimate_value += (gamma**(n_back_step)) + value_tau_n
                
                #breakpoint()
                
            #now calcuale new weights with estimate value
            state_tau = states_matrix[:, tau].copy()
            KQ_f_tau = KQ_f_matrix[:, tau].copy()
            KQ_r_tau = KQ_r_matrix[:, tau].copy()
            previous_value = state_value(theta_linear, delta_S_tau, KQ_f_tau, KQ_r_tau, state_tau)
    
            vec = entropy_production_rate_vec(KQ_f_tau, KQ_r_tau, state_tau)
            x_t = -(vec) #this is -EPR before summing it. 

            #new_theta = theta_linear + step_value * (x_t)
            theta_linear = SGD_UPDATE(theta_linear, step_size, estimate_value, previous_value, x_t)        
            
            
        #after complete loop is done, increment t.
        t+=1
            
    
    return theta_linear

def SGD_UPDATE(theta_linear, step_size, estimate_value, previous_value, x_t):

    step_value = step_size * (estimate_value - previous_value)
    
    #vec = entropy_production_rate_vec(KQ_f_init, KQ_r_init, state_sample)
    #x_t = -(vec) #this is -EPR before summing it. 

    new_theta = theta_linear + step_value * (x_t)
    
    new_theta[new_theta<0] = 0
    return new_theta
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
    
    res_lsq = least_squares(max_entropy_functions.derivatives, v_log_counts_path, method=Method,xtol=1e-15, args=(f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, state))
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
    #breakpoint()
    
    initial_reward = reward_value(KQ_f, KQ_r, state)
        
    action_value_vec = np.zeros(reaction_size_to_regulate)
    state_value_vec = np.zeros(reaction_size_to_regulate)
    E_test_vec = np.zeros(reaction_size_to_regulate)
    old_E_test_vec = np.zeros(reaction_size_to_regulate)
    current_reward_vec = np.zeros(reaction_size_to_regulate)
    for act in range(0,reaction_size_to_regulate):
        #print("in policy_function action loop")
        #print("v_log_counts")
        #print(np.max(v_log_counts))
        #print(np.min(v_log_counts))
        #print("end:")
        React_Choice = act#regulate each reaction.
        
        old_E = state[act]
        newE = max_entropy_functions.calc_reg_E_step(state, React_Choice, nvar, v_log_counts, f_log_counts,
                               desired_conc, S_mat, A, rxn_flux, KQ_f, False, has_been_up_regulated)
        
        
        #if (( newE <1 ) and (newE > 0) and (newE != oldE)):
            
        #now generate the next step based on newE
        trial_state_sample = state.copy()#DO NOT MODIFY STATE
        trial_state_sample[React_Choice] = newE
            
        #re-optimize
        new_res_lsq = least_squares(max_entropy_functions.derivatives, v_log_counts, method=Method,xtol=1e-15, args=(f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, trial_state_sample))
            
        new_v_log_counts = new_res_lsq.x
        #breakpoint()
        new_log_metabolites = np.append(new_v_log_counts, f_log_counts)
        
        KQ_f_new = max_entropy_functions.odds(new_log_metabolites, mu0,S_mat, R_back_mat, P_mat, delta_increment_for_small_concs,Keq_constant);    
        Keq_inverse = np.power(Keq_constant,-1)
        KQ_r_new = max_entropy_functions.odds(new_log_metabolites, mu0,-S_mat, P_mat, R_back_mat, delta_increment_for_small_concs,Keq_inverse,-1);

        new_delta_S = max_entropy_functions.calc_deltaS(new_v_log_counts,f_log_counts, S_mat, KQ_f_new)
    
        value_current_state = state_value(theta_linear, new_delta_S, KQ_f_new, KQ_r_new, trial_state_sample)
        
        current_reward = reward_value(KQ_f_new, KQ_r_new, trial_state_sample)

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
                
    #print("min->max order of best actions")
    #print(np.argsort(action_value_vec))
    #print("action_value_vec")
    #print(action_value_vec)
    #print(action_value_vec[action_choice])
    #print("state_value_vec")
    #print(state_value_vec)
    #print("E_test_vec")
    #print(E_test_vec)
    #print("oldE_test_vec")
    #print(old_E_test_vec)
    #print("current_reward_vec")
    #print(current_reward_vec)
    rxn_choices.remove(action_choice)
    random_choice = random.choice(rxn_choices)
    #breakpoint()
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