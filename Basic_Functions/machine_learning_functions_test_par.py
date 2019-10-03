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
v_log_counts_static=[]
target_v_log_counts=[]
complete_target_log_counts=[]
device=[]
Keq_constant=[]
f_log_counts=[]

P_mat=[]
R_back_mat=[]
S_mat=[]
delta_increment_for_small_concs=[]

nvar=[]
mu0=[]

gamma=[]
num_rxns=[]

penalty_exclusion_reward = -10.0
penalty_reward_scalar=0.0

range_of_activity_scale = 1.0
log_scale_activity = 0.4
alternative_reward=False


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
from itertools import repeat
import torch


Method1 = 'dogbox'
Method2 = 'lm'
Method3 = 'trf'

#Physically this is trying to minimizing the free energy change in each reaction. 
def state_value(nn_model, x):
    
    #val = nn_model(torch.log(1.0 + torch.exp(range_of_activity_scale*x)))
    
    scale_to_one = np.log(range_of_activity_scale + (1**log_scale_activity))
    x_scaled = (1.0 / scale_to_one) * torch.log(1.0 + (x**log_scale_activity))
    val = nn_model( x_scaled )
    
    return val

#want to maximize the change in loss function for positive values. 
#what i really want is to make this function continuous. 
def reward_value(v_log_counts_future, v_log_counts_old,\
                 KQ_f_new, KQ_r_new, E_Regulation_new, E_Regulation_old):
    final_reward=0.0

    #val_old = max_entropy_functions.calc_deltaS_metab(v_log_counts_old, target_v_log_counts)
    #val_future = max_entropy_functions.calc_deltaS_metab(v_log_counts_future, target_v_log_counts)
    
    #https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/

    #here we use the mean for the scaling. The logic is as follows:
    #
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

    if ( (np.isnan(e_val_future).any()) or (np.isnan(e_val_old).any()) ):
        print(v_log_counts_future)
        print(v_log_counts_old)
    #if (( (v_log_counts_future>50).any()) or (v_log_counts_future<-50).any()):
    #    print(min(np.exp(v_log_counts_future - target_v_log_counts - scale_future)))
    #    print(max(np.exp(v_log_counts_future - target_v_log_counts - scale_future)))
    #   print(scale_future)

    reward_s = e_val_old - e_val_future
    #val_old[val_old<0.0] = 0
    #val_future[val_future<0] = 0

    #reward_s = np.sum(val_old - val_future) #this is positive if val_future is less, so we need
    
    #originally, does nothing, but turns to penalty when regulating a new reaction
    psi = 1.0 
    
    #reward_s = e_val_old-e_val_future
    num_regulated_new = np.sum(E_Regulation_new==1)
    num_regulated_old = np.sum(E_Regulation_old==1)

    if (num_regulated_new != num_regulated_old):
        #then you regulated a new reaction:
        psi = penalty_reward_scalar

    if ((reward_s < 0.0)):
        final_reward = penalty_exclusion_reward
        
    if (reward_s >= 0.0):
        final_reward = psi * reward_s
        #if negative (-0.01) -> take fastest path
        #if positive (0.01) -> take slowest path
        
    if ((  scale_future_max <=0.0)):
        #The final reward is meant to maximize the EPR value. However, there was some residual error in ds_metab
        #that must be taken into account. We therefore add the last reward_s to the EPR value. 
        
        epr_future = max_entropy_functions.entropy_production_rate(KQ_f_new, KQ_r_new, E_Regulation_new)
        final_reward = 1.0 * epr_future + psi*reward_s 
        
    return final_reward


def sarsa_n(nn_model, loss_fn, optimizer, scheduler, state_sample, n_back_step, epsilon_greedy):
    
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
    end_of_path = 1000 #this is the maximum length a path can take
    KQ_f_matrix = np.zeros(shape=(num_rxns, end_of_path+1))
    KQ_r_matrix = np.zeros(shape=(num_rxns, end_of_path+1))
    states_matrix = np.zeros(shape=(num_rxns, end_of_path+1))
    delta_S_metab_matrix = np.zeros(shape=(nvar, end_of_path+1))
    v_log_counts_matrix = np.zeros(shape=(nvar, end_of_path+1))
    
    states_matrix[:,0] = state_sample
    
    
    res_lsq = least_squares(max_entropy_functions.derivatives, v_log_counts_static, method=Method1,
                            bounds=(-500,500),xtol=1e-15, 
                            args=(f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, states_matrix[:,0]))
    if (res_lsq.success==False):
        res_lsq = least_squares(max_entropy_functions.derivatives, v_log_counts_static, method=Method2,xtol=1e-15, args=(f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, states_matrix[:,0]))
        if (res_lsq.success==False):
            res_lsq = least_squares(max_entropy_functions.derivatives, v_log_counts_static, method=Method3,xtol=1e-15, args=(f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, states_matrix[:,0]))
    
    v_log_counts_matrix[:,0] = res_lsq.x.copy()
    #v_log_counts_matrix[:,0]=v_log_counts_static.copy()
    log_metabolites = np.append(v_log_counts_matrix[:,0], f_log_counts)
        
    rxn_flux_init = max_entropy_functions.oddsDiff(v_log_counts_matrix[:,0], f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, states_matrix[:,0])
    KQ_f_matrix[:,0] = max_entropy_functions.odds(log_metabolites, mu0,S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant);
    
    Keq_inverse = np.power(Keq_constant,-1)
    KQ_r_matrix[:,0] = max_entropy_functions.odds(log_metabolites, mu0,-S_mat, P_mat, R_back_mat, delta_increment_for_small_concs, Keq_inverse,-1);
    
    #[RR,Jac] = max_entropy_functions.calc_Jac2(v_log_counts_matrix[:,0], f_log_counts, S_mat, delta_increment_for_small_concs, KQ_f_matrix[:,0], KQ_r_matrix[:,0], states_matrix[:,0])
    #A_init = max_entropy_functions.calc_A(v_log_counts_matrix[:,0], f_log_counts, S_mat, Jac, states_matrix[:,0] )
    
    delta_S_metab_matrix[:,0] = max_entropy_functions.calc_deltaS_metab(v_log_counts_matrix[:,0], target_v_log_counts);
        
    #[ccc,fcc] = max_entropy_functions.conc_flux_control_coeff(nvar, A_init, S_mat, rxn_flux_init, RR)
          
    reward_vec = np.zeros(end_of_path+1)   
    
    reward_vec[0] = 0.0
    rxn_flux_path=rxn_flux_init.copy()
    #A_path = A_init.copy()
    
    for t in range(0,end_of_path):
        #breakpoint()
        if (t < end_of_path):

            #This represents the choice from the current policy. 
            [React_Choice,reward_vec[t+1],\
            KQ_f_matrix[:,t+1], KQ_r_matrix[:,t+1],\
            v_log_counts_matrix[:,t+1],\
            states_matrix[:,t+1],\
            delta_S_metab_matrix[:,t+1],\
            used_random_step] = policy_function(nn_model, states_matrix[:,t], v_log_counts_matrix[:,t], epsilon_greedy)#regulate each reaction.                
            
            if (used_random_step):
                random_steps_taken+=1
            else:
                nn_steps_taken+=1
                    
            if (React_Choice==-1):
                break

            rxn_flux_path = max_entropy_functions.oddsDiff(v_log_counts_matrix[:,t+1], f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, states_matrix[:,t+1])
            

            epr_path = max_entropy_functions.entropy_production_rate(KQ_f_matrix[:,t+1], KQ_r_matrix[:,t+1], states_matrix[:,t+1])
            

            sum_reward_episode += reward_vec[t+1]

            #last_state = states_matrix[:,t]
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
                
                

# =============================================================================
#             for state in range(0,t+1):
#                 last_state = states_matrix[:,state].copy()
#                 if ((current_state==last_state).all()):
#                     end_of_path=t+1
#                     
#                     print("**************************************Path Length******************************************")
#                     print(end_of_path)
#                     print("Final STATE")
#                     print(states_matrix[:,state])
#                     print(rxn_flux_path)
#                     print(epr_path)
# =============================================================================

        ##BEGIN LEARNING
        tau = t - n_back_step + 1
                
        if (tau >=0):
            #breakpoint()
            estimate_value = torch.zeros(1, device=device)

            for i in range(tau + 1, min(tau + n_back_step, end_of_path)+1):    
                estimate_value += (gamma**(i-tau-1)) * reward_vec[i]

            if ((tau + n_back_step) < end_of_path):
                value_tau_n = state_value(nn_model, torch.from_numpy(states_matrix[:, tau + n_back_step]).float().to(device) )
                
                
                if len(value_tau_n.shape) ==2:
                    value_tau_n = value_tau_n[0]
                elif len(value_tau_n.shape) ==3:
                    value_tau_n = value_tau_n[0][0]
                
                estimate_value += (gamma**(n_back_step)) * value_tau_n
            
            value_tau = state_value(nn_model, torch.from_numpy(states_matrix[:, tau]).float().to(device) )
            if len(value_tau.shape) == 2:
                value_tau = value_tau[0]
            elif len(value_tau.shape) == 3:
                value_tau = value_tau[0][0]
            #nn_model.eval()
            

            if (value_tau.requires_grad == False):
                breakpoint()
            if (estimate_value.requires_grad == True):
                estimate_value.detach_()
            

            
            #WARNING
            #loss ordering should be input with requires_grad == True,
            #followed by target with requires_grad == False
            #breakpoint()
            loss = torch.sqrt(loss_fn( value_tau, estimate_value)) #RMSE
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

            average_loss.append(loss.item())
            

        if (tau >= (end_of_path-1)):
            break
    
    #after episode is finished, take average loss
    average_loss_episode = np.mean(average_loss)
    #print(average_loss)
    print("index of max error on path")
    print(average_loss.index(max(average_loss)))
    return [sum_reward_episode, average_loss_episode,max(average_loss),final_reward, final_state, final_KQ_f,final_KQ_r,\
            reached_terminal_state, random_steps_taken,nn_steps_taken]


#%%
#input state, return action
    
def potential_step(index, other_args):
    React_Choice=index
    
    nn_model,state, nvar, v_log_counts, f_log_counts,\
    complete_target_log_counts, A, rxn_flux, KQ_f,\
    delta_S_metab,\
    mu0, S_mat, R_back_mat, P_mat, \
    delta_increment_for_small_concs, Keq_constant = other_args
    
    
    newE = max_entropy_functions.calc_reg_E_step(state, React_Choice, nvar, v_log_counts, f_log_counts,
                               complete_target_log_counts, S_mat, A, rxn_flux, KQ_f,\
                               delta_S_metab)

    trial_state_sample = state.copy()#DO NOT MODIFY ORIGINAL STATE
    trial_state_sample[React_Choice] = newE
        #re-optimize
    new_res_lsq = least_squares(max_entropy_functions.derivatives, v_log_counts, method=Method1,
                                bounds=(-500,500),xtol=1e-15, 
                                args=(f_log_counts, mu0, S_mat, R_back_mat, P_mat, 
                                      delta_increment_for_small_concs, Keq_constant, trial_state_sample))

    new_v_log_counts = new_res_lsq.x

    
    new_log_metabolites = np.append(new_v_log_counts, f_log_counts)

    new_delta_S_metab = max_entropy_functions.calc_deltaS_metab(new_v_log_counts, target_v_log_counts)

    KQ_f_new = max_entropy_functions.odds(new_log_metabolites, mu0,S_mat, R_back_mat, P_mat, delta_increment_for_small_concs,Keq_constant);    
    Keq_inverse = np.power(Keq_constant,-1)
    KQ_r_new = max_entropy_functions.odds(new_log_metabolites, mu0,-S_mat, P_mat, R_back_mat, delta_increment_for_small_concs,Keq_inverse,-1);

    value_current_state = state_value(nn_model,  torch.from_numpy(trial_state_sample).float().to(device) )
    value_current_state = value_current_state.item()

    current_reward = reward_value(new_v_log_counts, v_log_counts, \
                                  KQ_f_new, KQ_r_new,\
                                  trial_state_sample, state)

    action_value = current_reward + (gamma) * value_current_state #note, action is using old KQ values

    return [action_value, current_reward,KQ_f_new,KQ_r_new,new_v_log_counts,trial_state_sample,new_delta_S_metab]
    
def policy_function(nn_model, state, v_log_counts_path, *args ):
    #last input argument should be epsilon for use when using greedy-epsilon algorithm. 
    
    varargin = args
    nargin = len(varargin)
    epsilon_greedy = 0.0
    if (nargin == 1):
        epsilon_greedy = varargin[0]
        
    rxn_choices = [i for i in range(num_rxns)]
    
    res_lsq = least_squares(max_entropy_functions.derivatives, v_log_counts_path, method=Method1,
                            bounds=(-500,500),xtol=1e-15, 
                            args=(f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, state))
    if (res_lsq.optimality>1e-05):
        res_lsq = least_squares(max_entropy_functions.derivatives, v_log_counts_path, method=Method2,xtol=1e-15, args=(f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, state))
        if (res_lsq.optimality>1e-05):
            res_lsq = least_squares(max_entropy_functions.derivatives, v_log_counts_path, method=Method3,xtol=1e-15, args=(f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, state))
    

    #v_log_counts = v_log_counts_path.copy()
    v_log_counts = res_lsq.x
    
    if (np.sum(np.abs(v_log_counts - v_log_counts_path)) > 0.001):
        print("ERROR IN POLICY V_COUNT OPTIMIZATION")
    log_metabolites = np.append(v_log_counts, f_log_counts)
        
    rxn_flux = max_entropy_functions.oddsDiff(v_log_counts, f_log_counts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, state)
    KQ_f = max_entropy_functions.odds(log_metabolites, mu0,S_mat, R_back_mat, P_mat, delta_increment_for_small_concs,Keq_constant);
    Keq_inverse = np.power(Keq_constant,-1)
    KQ_r = max_entropy_functions.odds(log_metabolites, mu0,-S_mat, P_mat, R_back_mat, delta_increment_for_small_concs,Keq_inverse,-1);
    
    [RR,Jac] = max_entropy_functions.calc_Jac2(v_log_counts, f_log_counts, S_mat, delta_increment_for_small_concs, KQ_f, KQ_r, state)
    A = max_entropy_functions.calc_A(v_log_counts, f_log_counts, S_mat, Jac, state )
    
    
    delta_S_metab = max_entropy_functions.calc_deltaS_metab(v_log_counts, target_v_log_counts)
    delta_S = max_entropy_functions.calc_deltaS(v_log_counts, f_log_counts, S_mat, KQ_f)
    
    [ccc,fcc] = max_entropy_functions.conc_flux_control_coeff(nvar, A, S_mat, rxn_flux, RR)
    
    
    indices = [i for i in range(0,len(Keq_constant))]
    action_value_vec = np.zeros(num_rxns)
    state_value_vec = np.zeros(num_rxns)
    E_test_vec = np.zeros(num_rxns)
    old_E_test_vec = np.zeros(num_rxns)
    current_reward_vec = np.zeros(num_rxns)

    variables=[nn_model,state, nvar, v_log_counts, f_log_counts,\
               complete_target_log_counts, A, rxn_flux, KQ_f,\
               delta_S_metab,\
               mu0, S_mat, R_back_mat, P_mat, 
               delta_increment_for_small_concs, Keq_constant]

    start = time.time()
    
    with Pool() as pool:
        async_result = pool.starmap(potential_step, zip(indices, repeat(variables)))
        pool.close()
        pool.join()
    end = time.time()

    #only choose from non penalty rewards        
    for act in range(0,len(async_result)):
        if (async_result[act][1] == penalty_exclusion_reward):
            rxn_choices.remove(act)
        action_value_vec[act] = async_result[act][0]
        current_reward_vec[act] = async_result[act][1]
    
    action_choice_index = np.random.choice(np.flatnonzero(action_value_vec[rxn_choices] == action_value_vec[rxn_choices].max()))
    action_choice = rxn_choices[action_choice_index]


    arr_choice_index = np.flatnonzero(action_value_vec[rxn_choices] == action_value_vec[rxn_choices].max())
    arr_choice=np.asarray(rxn_choices)[arr_choice_index]
    
    arr_choice_reg = np.flatnonzero(state[arr_choice]<1)
    if (arr_choice_reg.size>1):
        print('using tie breaker')
        print(arr_choice[arr_choice_reg])
        action_choice = np.random.choice(arr_choice[arr_choice_reg])
    


    used_random_step=False
    unif_rand = np.random.uniform(0,1)
    if ( (unif_rand < epsilon_greedy) and (len(rxn_choices) > 0)):
        #if (len(rxn_choices)>1):
        #    rxn_choices.remove(action_choice)
        #print("USING EPSILON GREEDY")
        #print(action_choice)       
        used_random_step=True
        random_choice = random.choice(rxn_choices)
        action_choice = random_choice
        used_random_step=1

    if (current_reward_vec == penalty_exclusion_reward).all():
        print("OUT OF REWARDS")
        action_choice=-1

    #async_result order
    #[action_value, current_reward,KQ_f_new,KQ_r_new,new_v_log_counts,trial_state_sample,new_delta_S_metab]
    return [action_choice,async_result[action_choice][1],\
            async_result[action_choice][2],async_result[action_choice][3],\
            async_result[action_choice][4],\
            async_result[action_choice][5],\
            async_result[action_choice][6],used_random_step]
        
    #return [action_choice,current_reward_vec[action_choice],\
    #        KQ_f_matrix[:,action_choice],KQ_r_matrix[:,action_choice],\
    #        v_log_counts_matrix[:,action_choice],\
    #        states_matrix[:,action_choice],\
    #        delta_S_metab_matrix[:,action_choice],used_random_step]
    
    
    
    
    