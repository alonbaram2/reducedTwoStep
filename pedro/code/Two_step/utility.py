import pickle
import sys
import math
import numpy as np
from numba import jit

log_max_float = np.log(sys.float_info.max/2.1) # Log of largest floating point value.

def log_safe(x):
    '''Return log of x protected against giving -inf for very small values of x.'''
    return np.log(((1e-200)/2)+(1-(1e-200))*x)

@jit
def exp_mov_ave(data, tau = 8., initValue = 0., alpha = None):
    '''Exponential Moving average for 1d data.  The decay of the exponential can 
    either be specified with a time constant tau or a learning rate alpha.'''
    if not alpha: alpha = 1. - np.exp(-1./tau)
    mov_ave = np.zeros(np.size(data)+1)
    mov_ave[0] = initValue
    for i, x in enumerate(data):
        mov_ave[i+1] = (1.-alpha)*mov_ave[i] + alpha*x 
    return mov_ave[1:]


def nans(shape, dtype=float):
    '''return array of nans of specified shape.'''
    a = np.empty(shape, dtype)
    a.fill(np.nan)
    return a


def projection(u,v):
    '''For vectors u and v, returns the projection of v along u.
    '''
    u_dot_u = np.dot(u,u)
    if  u_dot_u == 0:
        return np.zeros(len(u))
    else:
        return u*np.dot(u,v)/u_dot_u


def check_common_transition_fraction(sessions):
    '''Sanity check that common transitions are happening at correct frequency.'''
    sIDs = set([s.subject_ID for s in sessions])
    for sID in sIDs:
        a_sessions = [s for s in sessions if s.subject_ID == sID]
        transitions_CR = np.hstack([s.trial_data['transitions'] == s.blocks['trial_trans_state'] for s in a_sessions])
        print(('Subject: {}  mean transitions: {}'.format(sID, np.mean(transitions_CR))))


def norm_correlate(a, v, mode='Full'):
    '''Calls numpy correlate after normalising the inputs.'''
    a = (a - np.mean(a)) / (np.std(a) * len(a))
    v = (v - np.mean(v)) /  np.std(v)
    return np.correlate(a, v, mode)


def nansem(x,dim = 0, ddof = 1):
    '''Standard error of the mean ignoring nans along dimension dim.'''
    return np.sqrt(np.nanvar(x,dim)/(np.sum(~np.isnan(x),dim) - ddof))


def save_item(item, file_name):
    '''Save an item using pickle.'''
    with open(file_name+'.pkl', 'wb') as f:
        pickle.dump(item, f)

def load_item(file_name):
    '''Unpickle and return specified item.'''
    with open(file_name+'.pkl', 'rb') as f:
        return pickle.load(f)