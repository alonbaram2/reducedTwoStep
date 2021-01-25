'''Mixture agent with forgetting, perseveration and bias.'''

import numpy as np
from numba import njit

import RL_utils as ru

# Agent info.

name = 'MF_MBi'
param_names  = ['G_td', 'G_mb', 'alpQ', 'lbd' , 'alpT', 'bias', 'pers']
param_ranges = ['pos' , 'pos' , 'unit', 'unit', 'unit', 'unc' , 'unc' ]
n_params = len(param_names)
    
# Session likelihood ----------------------------------------------

def session_likelihood(session, params):
    '''Compute the data likelihood for session given params.'''

    choices, second_steps, outcomes = ru.unpack_trial_data(session)
    Q_tot = compute_values(choices, second_steps, outcomes, params)
        
    return ru.session_log_likelihood(choices, Q_tot)

# Compute values    ----------------------------------------------

@njit()
def compute_values(choices, second_steps, outcomes, params):

    # Unpack parameters.
    n_trials = len(choices)
    G_td, G_mb, alpQ, lbd, alpT, bias, pers  = params

    #Variables.
    Q = np.zeros((2,n_trials)) # First step TD values.
    V = np.zeros((2,n_trials)) # Second step TD values.
    T = np.zeros((2,n_trials)) # Transition probabilities.

    # Initialize first trial transition probabilities.
    T[:,0] = [0.5,0.5]

    for i, (c, s, o) in enumerate(zip(choices[:-1], second_steps, outcomes)): # loop over trials.

        n = 1 - c  # Action not chosen at first step.
        r = 1 - s  # State not reached at second step.

        # Update action values and transition probabilities.

        Q[n,i+1] = Q[n,i] 
        V[r,i+1] = V[r,i]
        T[n,i+1] = T[n,i]

        Q[c,i+1] = (1.-alpQ)*Q[c,i] + alpQ*((1.-lbd)*V[s,i] + lbd*o) # First step TD update.
        V[s,i+1] = (1.-alpQ)*V[s,i] + alpQ*o  # Second step TD update.

        T[c,i+1] = (1.-alpT)*T[c,i] + alpT*s  # Transition prob. update.

    # Evaluate net action value.

    M = T*V[1,:] + (1.-T)*V[0,:] # Model based action values.
    Q_net = G_td*Q + G_mb*M      # Mixture of model based and model free values.
    
    # Apply bias and perseveration.

    K = np.zeros((2,n_trials)) # Modifier to values due to bias and perseveration.
    K[1,: ] += bias                    # Apply choice bias
    K[1,1:] += pers*(choices[:-1]-0.5) # Apply perseveration.
    Q_tot = Q_net + K 

    return Q_tot
