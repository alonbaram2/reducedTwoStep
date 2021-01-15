import numpy as np
import datetime
from scipy import signal
from copy import deepcopy

def get_IDs(IDs, event_list):
    return [IDs[val] for val in event_list]

def event_name(event_code, IDs):
    'Get event name from event ID.'
    return list(IDs.keys())[list(IDs.values()).index(event_code)]

def exp_mov_ave(data, tau = 8., initValue = 0.5):
    'Exponential Moving average for 1d data.'
    m = np.exp(-1./tau)
    i = 1 - m
    mov_ave = np.zeros(np.size(data)+1)
    mov_ave[0] = initValue
    for k, sample in enumerate(data):
        mov_ave[k+1] = mov_ave[k] * m + i * sample 
    return mov_ave[1::]

def CTSO_unpack(CTSO, order = 'CTSO', dtype = int):
    'Return elements of CTSO dictionary in specified order and data type.'
    o_dict = {'C': 'choices', 'T': 'transitions', 'S': 'second_steps', 'O': 'outcomes'}
    if dtype == int:
        return [CTSO[o_dict[i]] for i in order]
    else:
        return [CTSO[o_dict[i]].astype(dtype) for i in order]

def nans(shape, dtype=float):
    'return array of nans of specified shape.'
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
    ''' Sanity check that common transitions are happening at correct frequency.
    '''
    sIDs = set([s.subject_ID for s in sessions])
    for sID in sIDs:
        a_sessions = [s for s in sessions if s.subject_ID == sID]
        transitions_CR = np.hstack([s.CTSO['transitions'] == s.blocks['trial_trans_state'] for s in a_sessions])
        print(('Subject: {}  mean transitions: {}'.format(sID, np.mean(transitions_CR))))

def norm_correlate(a, v, mode='Full'):
    'Calls numpy correlate after normaising the inputs.'
    a = (a - np.mean(a)) / (np.std(a) * len(a))
    v = (v - np.mean(v)) /  np.std(v)
    return np.correlate(a, v, mode)

def nansem(x,dim = 0, ddof = 1):
    'Standard error of the mean ignoring nans along dimension dim.'
    return np.sqrt(np.nanvar(x,dim)/(np.sum(~np.isnan(x),dim) - ddof))

def select_trials(session, selection_type, select_n = 20, first_n_mins = False,
                  block_type = 'all'):
    ''' Select specific trials for analysis.  

    The first selection step is specified by selection_type:

    'end' : Only final select_n trials of each block are selected.

    'xtr' : Select all trials except select_n trials following transition reversal.

    'all' : All trials are included.

    The first_n_mins argument can be used to select only trials occuring within
    a specified number of minutes of the session start.

    The block_type argument allows additional selection for only 'neutral' or 'non_neutral' blocks.
    '''

    assert selection_type in ['end', 'xtr', 'all'], 'Invalid trial select type.'

    if selection_type == 'xtr': # Select all trials except select_n following transition reversal.
        trials_to_use = np.ones(session.n_trials, dtype = bool)
        trans_change = np.hstack((True, ~np.equal(session.blocks['transition_states'][:-1], \
                                                  session.blocks['transition_states'][1:])))
        start_trials = session.blocks['start_trials'] + [session.blocks['end_trials'][-1] + select_n]
        trials_to_use[0] = False
        for i in range(len(trans_change)):
            if trans_change[i]:
                trials_to_use[start_trials[i]:start_trials[i] + select_n] = False

    elif selection_type == 'end': # Select only select_n trials before block transitions.
        trials_to_use = np.zeros(session.n_trials, dtype = bool)
        for b in session.blocks['start_trials'][1:]:
            trials_to_use[b - 1 - select_n:b -1] = True

    elif selection_type == 'all': # Use all trials.
        trials_to_use = np.ones(session.n_trials, dtype = bool)
        
    if first_n_mins:  #  Restrict analysed trials to only first n minutes. 
        time_selection = session.trial_start_times[:session.n_trials] < (60 * first_n_mins)
        trials_to_use = trials_to_use & time_selection

    if not block_type == 'all': #  Restrict analysed trials to blocks of certain types.
        if block_type == 'neutral':       # Include trials only from neutral blocks.
            block_selection = session.blocks['trial_rew_state'] == 1
        elif block_type == 'non_neutral': # Include trials only from non-neutral blocks.
            block_selection = session.blocks['trial_rew_state'] != 1
        trials_to_use = trials_to_use & block_selection

    return trials_to_use