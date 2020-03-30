# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 13:48:04 2019

@author: samuel_britton
"""

#%% Learning Test

##Terms::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#States: defined by how much each rxn is regulated (vector of num_rxns floats from 0-1)
#Action: defined by which rxn we choose to regulate. 
#Policy: function S_mat->A (each )


#%% Variables to set before use
### STATIC ###
complete_target_log_counts=[]
device=[]
Keq_constant=[]
f_log_counts=[]
P_mat=[]
R_back_mat=[]
S_mat=[]
delta_increment_for_small_concs=[]
v_log_counts_static=[]
target_v_log_counts=[]

nvar=[]
mu0=[]

gamma=[]
num_rxns=[]

penalty_exclusion_reward = -10.0
penalty_reward_scalar=0.0

range_of_activity_scale = 1.0
log_scale_activity = 0.4
alternative_reward=False

#%% used functions 
import max_entropy_functions
import numpy as np
import pandas as pd
import random
import time
from scipy.optimize import least_squares
from scipy.optimize import minimize
import multiprocessing as mp
from multiprocessing import Pool
from itertools import repeat
import torch


def state_value(nn_model, x):    
    
    scale_to_one = np.log(range_of_activity_scale + (1**log_scale_activity))
    x_scaled = (1.0 / scale_to_one) * torch.log(1.0 + (x**log_scale_activity))
    val = nn_model( x_scaled )
    
    return val

#want to maximize the change in loss function for positive values. 
#what i really want is to make this function continuous. 
def reward_value(v_log_counts_future, v_log_counts_old,\
                 KQ_f_new, KQ_r_new, E_Regulation_new, E_Regulation_old):
    final_reward=0.0

    #here we use the mean for the scaling. The logic is as follows:
    #https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/

    scale_old_max = np.max(v_log_counts_old - target_v_log_counts)
    scale_old_min = np.min(v_log_counts_old - target_v_log_counts)
    scale_old = (scale_old_max + scale_old_min)/2.0
    
    e_val_old = np.exp(v_log_counts_old - target_v_log_counts - scale_old)
    e_val_old = scale_old + np.log(np.sum(e_val_old))
      
    scale_future_max = np.max(v_log_counts_future - target_v_log_counts)
    scale_future_min = np.min(v_log_counts_future - target_v_log_counts)
    scale_future = (scale_future_max + scale_future_min)/2.0

    e_val_future = np.exp(v_log_counts_future - target_v_log_counts - scale_future)
    e_val_future = scale_future  + np.log(np.sum(e_val_future))

    reward_s = (e_val_old - e_val_future)
    final_reward=reward_s     
    if ((  scale_future_max <=0.0)):
        #The final reward is meant to maximize the EPR value. However, there was some residual error
        #that must be taken into account. We therefore add the last reward_s to the EPR value. 
        
        epr_future = max_entropy_functions.entropy_production_rate(KQ_f_new, KQ_r_new, E_Regulation_new)
        final_reward = 1.0 * epr_future + reward_s 
        
    return final_reward


def sarsa_n(nn_model, loss_fn, optimizer, scheduler, state_sample, n_back_step, epsilon_greedy):
    total_time_cpu=0
    total_time_nn=0
    #reset for each episode. policy will add
    random_steps_taken=0
    nn_steps_taken=0
    
    final_state=[]
    final_KQ_f=[]
    final_KQ_r=[]
    reached_terminal_state=False
    average_loss=[]
    
    final_reward=0
    sum_reward_episode = 0
    end_of_path = 5000 #this is the maximum length a path can take
    KQ_f_matrix = np.zeros(shape=(num_rxns, end_of_path+1))
    KQ_r_matrix = np.zeros(shape=(num_rxns, end_of_path+1))
    states_matrix = np.zeros(shape=(num_rxns, end_of_path+1))
    delta_S_metab_matrix = np.zeros(shape=(nvar, end_of_path+1))
    v_log_counts_matrix = np.zeros(shape=(nvar, end_of_path+1))
    
    states_matrix[:,0] = state_sample
    
    
    res_lsq = least_squares(max_entropy_functions.derivatives, v_log_counts_static, method='lm',
                            xtol=1e-15, 
                            args=(f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, states_matrix[:,0]))

    v_log_counts_matrix[:,0] = res_lsq.x.copy()
    log_metabolites = np.append(v_log_counts_matrix[:,0], f_log_counts)
        
    rxn_flux_init = max_entropy_functions.oddsDiff(v_log_counts_matrix[:,0], f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, states_matrix[:,0])
    KQ_f_matrix[:,0] = max_entropy_functions.odds(log_metabolites, mu0,S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant);
    
    Keq_inverse = np.power(Keq_constant,-1)
    KQ_r_matrix[:,0] = max_entropy_functions.odds(log_metabolites, mu0,-S_mat, P_mat, R_back_mat, delta_increment_for_small_concs, Keq_inverse,-1);
        
    delta_S_metab_matrix[:,0] = max_entropy_functions.calc_deltaS_metab(v_log_counts_matrix[:,0], target_v_log_counts);
              
    reward_vec = np.zeros(end_of_path+1)   
    
    reward_vec[0] = 0.0
    rxn_flux_path=rxn_flux_init.copy()
    
    for t in range(0,end_of_path):
        if (t < end_of_path):
            #This represents the choice from the current policy. 
            [React_Choice,reward_vec[t+1],\
            KQ_f_matrix[:,t+1], KQ_r_matrix[:,t+1],\
            v_log_counts_matrix[:,t+1],\
            states_matrix[:,t+1],\
            delta_S_metab_matrix[:,t+1],\
            used_random_step,time_cpu,time_nn] = policy_function(nn_model, states_matrix[:,t], v_log_counts_matrix[:,t], epsilon_greedy)#regulate each reaction.                
            
            total_time_cpu+=time_cpu
            total_time_nn+=time_nn

            if (used_random_step):
                random_steps_taken+=1
            else:
                nn_steps_taken+=1
                    
            if (React_Choice==-1):
                print("bad reaction choice, using action = -1")
                break

            rxn_flux_path = max_entropy_functions.oddsDiff(v_log_counts_matrix[:,t+1], f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, states_matrix[:,t+1])

            epr_path = max_entropy_functions.entropy_production_rate(KQ_f_matrix[:,t+1], KQ_r_matrix[:,t+1], states_matrix[:,t+1])
            sum_reward_episode += reward_vec[t+1]

            current_state = states_matrix[:,t+1].copy()
            
            #We stop the path if we have no more positive loss function values, or if we revisit a state. 
            if ((delta_S_metab_matrix[:,t+1]<=0.0).all()):
                end_of_path=t+1 #stops simulation at step t+1
                
                reached_terminal_state = True
                final_state=states_matrix[:,t+1].copy()
                final_KQ_f=KQ_f_matrix[:,t+1].copy()
                final_KQ_r=KQ_r_matrix[:,t+1].copy()
                final_reward=epr_path
                print("**************************************Path Length ds<0******************************************")
                print(end_of_path)
                print("Final STATE")
                print(states_matrix[:,t+1])
                print(rxn_flux_path)
                print("original epr")
                print(epr_path)
                print("all rewards")
                print(reward_vec[0:t+1])                

        ##BEGIN LEARNING
        tau = t - n_back_step + 1
                
        if (tau >=0):
            #breakpoint()
            estimate_value = torch.zeros(1, device=device)

            for i in range(tau + 1, min(tau + n_back_step, end_of_path)+1):    
                estimate_value += (gamma**(i-tau-1)) * reward_vec[i]

            if ((tau + n_back_step) < end_of_path):
                begin_nn = time.time()
                value_tau_n = state_value(nn_model, torch.from_numpy(states_matrix[:, tau + n_back_step]).float().to(device) )
                end_nn = time.time()
                total_time_nn+=end_nn-begin_nn
                estimate_value += (gamma**(n_back_step)) * value_tau_n
            
            begin_nn = time.time()
            value_tau = state_value(nn_model, torch.from_numpy(states_matrix[:, tau]).float().to(device) )
            end_nn = time.time()
            total_time_nn+=end_nn-begin_nn            

            if (value_tau.requires_grad == False):
                breakpoint()
            if (estimate_value.requires_grad == True):
                estimate_value.detach_()
            
            #WARNING
            #loss ordering should be input with requires_grad == True,
            #followed by target with requires_grad == False
            #breakpoint()
            begin_nn = time.time()
            loss = loss_fn( value_tau, estimate_value) #MSE
            
            optimizer.zero_grad()
            loss.backward()
            clipping_value = 1.0
            torch.nn.utils.clip_grad_norm_(nn_model.parameters(), clipping_value)

            optimizer.step()
            end_nn = time.time()
            total_time_nn+=end_nn-begin_nn
            average_loss.append(loss.item())

        if (tau >= (end_of_path-1)):
            break
    
    #after episode is finished, take average loss
    average_loss_episode = np.mean(average_loss)
    print("index of max error on path")
    print(average_loss.index(max(average_loss)))
    return [sum_reward_episode, average_loss_episode,max(average_loss),final_reward, final_state, final_KQ_f,final_KQ_r,\
            reached_terminal_state, random_steps_taken,nn_steps_taken]


#%%
#input must be able to determine optimization routine. We need the follwing variables:
#0: state, type:np.array(float), purpose: current enzyme activities
#1: v_log_counts, type:np.array(float), purpose:initial guess for optimization
#2: f_log_counts, type:np.array(float), purpose:fixed metabolites
#3: mu0, type: , purpose: non as of now
#4: S_mat, type: np.array(float), purpose: stoichiometric matrix (rxn by matabolites)
#5: R_back_mat, type: np.array(float), purpose: reverse stoichiometric matrix (rxn by matabolites)
    #note could be calculated from S_mat: R_back_mat = np.where(S_mat<0, S_mat, 0)
#6: P_mat, type: np.array(float), purpose: forward stoichiometric matrix (rxn by matabolites), 
    # note could be calculated from S_mat: P_mat = np.where(S_mat>0,S_mat,0)
#7: Keq_constant, type: np.array(float), purpose: equilibrium constants

#This function shoud run for each reaction (index = 0 : nrxn-1)
#It will apply regulation to the state (enzyme activities) and calculate resulting steady state metabolite concentrations
def potential_step(index, other_args):
    React_Choice=index
    
    state, v_log_counts, f_log_counts,\
    mu0, S_mat, R_back_mat, P_mat, \
    delta_increment_for_small_concs, Keq_constant = other_args
    
    
    
    newE = max_entropy_functions.calc_new_enzyme_simple(state, React_Choice)
    trial_state_sample = state.copy()#DO NOT MODIFY ORIGINAL STATE
    trial_state_sample[React_Choice] = newE

    new_res_lsq = least_squares(max_entropy_functions.derivatives, v_log_counts, method='lm',
                                xtol=1e-15, 
                                args=(f_log_counts, mu0, S_mat, R_back_mat, P_mat, 
                                      delta_increment_for_small_concs, Keq_constant, trial_state_sample))

    new_v_log_counts = new_res_lsq.x
    
    #minimal output is the new steady state concentrations. We can recalculate the trial_state_sample to avoid sending it out
    return [new_v_log_counts]
    
#%%policy inputs states->actions. 
#here, taking an action corresponds to finding new steady state metabolite concentrations and other variables related to them.
def policy_function(nn_model, state, v_log_counts_path, *args ):
    #last input argument should be epsilon for use when using greedy-epsilon algorithm. 
    varargin = args
    nargin = len(varargin)
    epsilon_greedy = 0.0
    if (nargin == 1):
        epsilon_greedy = varargin[0]
        
    used_random_step=False
    
    rxn_choices = [i for i in range(num_rxns)]
        
    unif_rand = np.random.uniform(0,1)
    if ( (unif_rand < epsilon_greedy) and (len(rxn_choices) > 0)):
        used_random_step=True
        random_choice = random.choice(rxn_choices)
        final_action = random_choice
        used_random_step=1

        res_lsq = least_squares(max_entropy_functions.derivatives, v_log_counts_path, method='lm',
                        xtol=1e-15, 
                        args=(f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, state))


        final_v_log_counts = res_lsq.x
        
        new_log_metabolites = np.append(final_v_log_counts, f_log_counts)
        final_state = state.copy()
        newE = max_entropy_functions.calc_new_enzyme_simple(state, final_action)
        final_state = state.copy()#DO NOT MODIFY ORIGINAL STATE
        final_state[final_action] = newE


        final_delta_s_metab = max_entropy_functions.calc_deltaS_metab(final_v_log_counts, target_v_log_counts)
        final_KQ_f = max_entropy_functions.odds(new_log_metabolites, mu0,S_mat, R_back_mat, P_mat, delta_increment_for_small_concs,Keq_constant)
        Keq_inverse = np.power(Keq_constant,-1)
        final_KQ_r = max_entropy_functions.odds(new_log_metabolites, mu0,-S_mat, P_mat, R_back_mat, delta_increment_for_small_concs,Keq_inverse,-1)
        
        value_current_state = state_value(nn_model,  torch.from_numpy(final_state).float().to(device) )
        value_current_state = value_current_state.item()
        final_reward = reward_value(final_v_log_counts, v_log_counts_path, \
                                    final_KQ_f, final_KQ_r,\
                                    final_state, state)

    else:
        #In this, we must choose base on the best prediction base on environmental feedback

        v_log_counts = v_log_counts_path
            
        log_metabolites = np.append(v_log_counts, f_log_counts)
            
        rxn_flux = max_entropy_functions.oddsDiff(v_log_counts, f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, state)
        KQ_f = max_entropy_functions.odds(log_metabolites, mu0,S_mat, R_back_mat, P_mat, delta_increment_for_small_concs,Keq_constant)
        Keq_inverse = np.power(Keq_constant,-1)
        KQ_r = max_entropy_functions.odds(log_metabolites, mu0,-S_mat, P_mat, R_back_mat, delta_increment_for_small_concs,Keq_inverse,-1)
        
        [RR,Jac] = max_entropy_functions.calc_Jac2(v_log_counts, f_log_counts, S_mat, delta_increment_for_small_concs, KQ_f, KQ_r, state)
        A = max_entropy_functions.calc_A(v_log_counts, f_log_counts, S_mat, Jac, state )
        
        
        delta_S_metab = max_entropy_functions.calc_deltaS_metab(v_log_counts, target_v_log_counts)
        
        [ccc,fcc] = max_entropy_functions.conc_flux_control_coeff(nvar, A, S_mat, rxn_flux, RR)
        
        indices = [i for i in range(0,len(Keq_constant))]

        #minimal varialbes to run optimization
        variables=[state, v_log_counts, f_log_counts,\
                mu0, S_mat, R_back_mat, P_mat, 
                delta_increment_for_small_concs, Keq_constant]
        
        with Pool() as pool:
            async_result = pool.starmap(potential_step, zip(indices, repeat(variables)))
            pool.close()
            pool.join()
        end = time.time()
        
        temp_action_value = -np.inf
        for act in range(0,len(async_result)):

            new_v_log_counts = async_result[act][0] #output from pool
            new_log_metabolites = np.append(new_v_log_counts, f_log_counts)

            trial_state_sample = state.copy()

            newE = max_entropy_functions.calc_new_enzyme_simple(state, act)
            trial_state_sample = state.copy()#DO NOT MODIFY ORIGINAL STATE
            trial_state_sample[act] = newE
            new_delta_S_metab = max_entropy_functions.calc_deltaS_metab(new_v_log_counts, target_v_log_counts)

            KQ_f_new = max_entropy_functions.odds(new_log_metabolites, mu0,S_mat, R_back_mat, P_mat, delta_increment_for_small_concs,Keq_constant);
            KQ_r_new = max_entropy_functions.odds(new_log_metabolites, mu0,-S_mat, P_mat, R_back_mat, delta_increment_for_small_concs,Keq_inverse,-1);
            
            value_current_state = state_value(nn_model,  torch.from_numpy(trial_state_sample).float().to(device) )
            value_current_state = value_current_state.item()

            current_reward = reward_value(new_v_log_counts, v_log_counts, \
                                        KQ_f_new, KQ_r_new,\
                                        trial_state_sample, state)

            action_value = current_reward + (gamma) * value_current_state

            if (action_value > temp_action_value):
                #then a new action is the best. 
                temp_action_value = action_value

                #set best output variables 
                final_action = act
                final_reward = current_reward
                final_KQ_f = KQ_f_new
                final_KQ_r = KQ_r_new
                final_v_log_counts = new_v_log_counts
                final_state = trial_state_sample
                final_delta_s_metab = new_delta_S_metab

    return [final_action,\
            final_reward,\
            final_KQ_f,\
            final_KQ_r,\
            final_v_log_counts,\
            final_state,\
            final_delta_s_metab,used_random_step,0.0,0.0]
            
    
    
