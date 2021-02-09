import sys
import numpy as np

log_max_float = np.log(sys.float_info.max/2.1) # Log of largest floating point value.

def log_safe(x):
    '''Return log(x) protected against giving -inf for very small x.'''
    return np.log(((1e-200)/2)+(1-(1e-200))*x)

def softmax_probs(Q):
    '''Compute softmax probabilities for binary choice with temp =1.'''
    P = np.zeros(Q.shape)
    TdQ = Q[1,:]-Q[0,:]
    TdQ[TdQ > log_max_float] = log_max_float # Protection agairt overflow in exponential.    
    P[0,:] = 1./(1. + np.exp(TdQ))
    P[1,:] = 1. - P[0,:]
    return P

def session_log_likelihood(choices, Q):
    '''Evaluate session log likelihood given choices, action values,
    using softmax decision rule with temp=1.'''
    choice_probs = softmax_probs(Q)
    session_log_likelihood = np.sum(log_safe(choice_probs[choices,np.arange(len(choices))]))
    return session_log_likelihood

def unpack_trial_data(session_df):
    '''Return the choices, second-step states and outcomes from a
    dataframe as numpy arrays.'''
    return (session_df['choice' ].to_numpy(),
            session_df['state'  ].to_numpy(),
            session_df['outcome'].to_numpy())
