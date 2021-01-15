import numpy as np
from copy import deepcopy
from . import utility as ut
from . import RL_utils as ru
from . import RL_plotting as rp
from . import plotting as pl
from . import model_fitting as mf


class simulated_session():
    '''Stores agent parameters and simulated data, supports plotting as for experimental
    session class.
    '''
    def __init__(self, task, agent, params_T, n_trials = 1000):
        '''Simulate session with current agent and task parameters.'''
        self.param_names = agent.param_names
        self.true_params_T = params_T
        try: # Not possible for e.g. unit range params_T with value 0 or 1.
            self.true_params_U = ru.transTU(params_T, agent.param_ranges)
        except Exception: 
            self.true_params_U = None
        self.n_trials = n_trials
        choices, second_steps, outcomes = agent.simulate(task, params_T, n_trials)
        
        self.CTSO = {'choices'      : choices,
                     'transitions'  : (choices == second_steps).astype(int),
                     'second_steps' : second_steps,
                     'outcomes'     : outcomes}

        if hasattr(task,'blocks'):
            self.blocks = deepcopy(task.blocks)

    def plot(self):pl.plot_session(self)

    def select_trials(self, selection_type, select_n = 20, first_n_mins = False, block_type = 'all'):
        return ut.select_trials(self, selection_type, select_n, first_n_mins, block_type)
            
def simulate_sessions(agent, task,  n_sessions = 10, n_trials = 1000,
                      pop_params = None, randomize_params = True):
    '''Simulate a population of sessions.  If list of integers is passed as the n_trials argument,
    a set of sessions with number of trials given by the list elements is simulated, 
    overriding the n_sessions argument.
    By default agent parameters are randomised for each session using agent.randomize_params().
    If randomise_params is False and pop_params are provided, the pop_params means are used as the 
    agent parameters.'''

    sessions = []
    if pop_params:
        agent.pop_params = pop_params
    if not type(n_trials) == list:
        n_trials = [n_trials] * n_sessions
    for n_t in n_trials:
        if randomize_params:
            agent.randomize_params()
        elif pop_params:
            agent.set_params_U(pop_params['means'])
        sessions.append(simulated_session(task, agent, n_t))
    return sessions

def sim_sessions_from_pop_fit(task,agent,population_fit, use_MAP = False, enlarge = 1):
    '''Simulate sessions using parameter values from population fit.
    If use_MAP is true, simulated sessions use the MAP paramter values,
    otherwise parameter values are drawn randomly from the population 
    level distributions.  The number of trials in the simulated sessions
    matches those in the orginal dataset.
    The enlarge parameter can be used to produce larger datasets by simulating multiple 
    sessions for each session in the real data set.
    '''
    assert population_fit['param_names'] == agent.param_names, 'Agent parameters do not match fit.'
    agent.pop_params = {'SDs'  : population_fit['pop_params']['SDs'],
                        'means': population_fit['pop_params']['means']}
    sessions = []
    for i in range(enlarge):
        for n_trials, MAP_fit in zip(population_fit['n_trials'],
                                     population_fit['MAP_fits']):
            if use_MAP:
                agent.set_params_U(MAP_fit['params_U'])
            else:
                agent.randomize_params()
            sessions.append(simulated_session(task, agent, n_trials))
    return sessions

def sim_session_from_pop_fit_means(task, agent, population_fit, n_trials = 10000):
    '''Simulate a single session with agent parameter values set to the mean
    values of the population level distribution.
    '''
    assert population_fit['param_names'] == agent.param_names, 'Agent parameters do not match fit.'
    params_T = ru.trans_UT(population_fit['pop_params']['means'], agent.param_ranges)
    return simulated_session(task, agent, params_T, n_trials)
