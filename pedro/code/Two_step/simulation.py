import numpy as np
from random import randint, random
from copy import deepcopy
from functools import partial
from scipy.stats import multivariate_normal as mv_normal
from . import utility as ut
from . import model_fitting as mf
from . import model_plotting as mp
from . import plotting as pl
from .session import session
from . import parallel_processing as pp

#------------------------------------------------------------------------------------
# Two-step task.
#------------------------------------------------------------------------------------

class Extended_two_step:
    '''Two step task with reversals in both which side is good and the transition matrix.'''
    def __init__(self, neutral_reward_probs = False):
        # Parameters
        self.norm_prob = 0.8 # Probability of normal transition.
        self.neutral_reward_probs = neutral_reward_probs

        if neutral_reward_probs: 
            self.reward_probs = np.array([[0.4, 0.4],  # Reward probabilities in each reward block type.
                                          [0.4, 0.4],
                                          [0.4, 0.4]])
        else:
            self.reward_probs = np.array([[0.2, 0.8],  # Reward probabilities in each reward block type.
                                          [0.4, 0.4],
                                          [0.8, 0.2]])
        self.threshold = 0.75 
        self.tau = 8.  # Time constant of moving average.
        self.min_block_length = 40       # Minimum block length.
        self.min_trials_post_criterion = 20  # Number of trials after transition criterion reached before transtion occurs.
        self.mov_ave = _exp_mov_ave(tau = self.tau, init_value = 0.5)   # Moving average of agents choices.
        self.reset()

    def reset(self, n_trials = 1000, stim = False):
        self.transition_block = _with_prob(0.5)      # True for A blocks, false for B blocks.
        self.reward_block =     randint(0,2)        # 0 for left good, 1 for neutral, 2 for right good.
        self.block_trials = 0                       # Number of trials into current block.
        self.cur_trial = 0                          # Current trial number.
        self.trans_crit_reached = False             # True if transition criterion reached in current block.
        self.trials_post_criterion = 0              # Current number of trials past criterion.
        self.trial_number = 1                       # Current trial number.
        self.n_trials = n_trials                    # Session length.
        self.mov_ave.reset()
        self.end_session   = False
        self.stim_trials = _get_stim_trials(n_trials+1) if stim else None # Trials on which stimulation is simulated.
        self.blocks = {'start_trials'      : [0],
                       'end_trials'        : [],
                       'reward_states'     : [self.reward_block],      # 0 for left good, 1 for neutral, 2 for right good.
                       'transition_states' : [self.transition_block]}  # 1 for A blocks, 0 for B blocks.

    def trial(self, choice):
        # Update moving average.
        self.mov_ave.update(choice)
        second_step = int((choice == _with_prob(self.norm_prob))
                           == self.transition_block)
        self.block_trials += 1
        self.cur_trial += 1
        outcome = int(_with_prob(self.reward_probs[self.reward_block, second_step]))
        # Check for block transition.
        block_transition = False
        if self.trans_crit_reached:
            self.trials_post_criterion +=1
            if (self.trials_post_criterion >= self.min_trials_post_criterion) & \
               (self.block_trials >= self.min_block_length):
               block_transition = True
        else: # Check if transition criterion reached.
            if self.reward_block == 1 or self.neutral_reward_probs: #Neutral block
                if (self.block_trials > 20) & _with_prob(0.04):
                    self.trans_crit_reached = True
            elif self.transition_block ^ (self.reward_block == 2): # High is good option
                if self.mov_ave.ave > self.threshold:
                    self.trans_crit_reached = True
            else:                                                  # Low is good option
                if self.mov_ave.ave < (1. -self.threshold):
                    self.trans_crit_reached = True                

        if block_transition:
            self.block_trials = 0
            self.trials_post_criterion = 0
            self.trans_crit_reached = False
            old_rew_block = self.reward_block
            if old_rew_block == 1:                      # End of neutral block always transitions to one side 
                self.reward_block = _with_prob(0.5) * 2  # being good without reversal of transition probabilities.
            else: # End of block with one side good, 50% chance of change in transition probs.
                if _with_prob(0.5): #Reversal in transition probabilities.
                    self.transition_block = not self.transition_block
                    if _with_prob(0.5): # 50% chance of transition to neutral block.
                        self.reward_block = 1
                else: # No reversal in transition probabilities.
                    if _with_prob(0.5):
                        self.reward_block = 1 # Transition to neutral block.
                    else:
                        self.reward_block = 2 - old_rew_block # Invert reward probs.
            self.blocks['start_trials'].append(self.cur_trial)
            self.blocks['end_trials'].append(self.cur_trial)
            self.blocks['reward_states'].append(self.reward_block)
            self.blocks['transition_states'].append(self.transition_block)

        if self.cur_trial >= self.n_trials: #End of session.
            self.end_session = True
            self.blocks['end_trials'].append(self.cur_trial + 1)

            self.blocks['trial_trans_state'] = np.zeros(self.n_trials, dtype = bool) #Boolean array indication state of tranistion matrix for each trial.
            self.blocks['trial_rew_state']   = np.zeros(self.n_trials, dtype = int)

            for start_trial,end_trial, trans_state, reward_state in \
                    zip(self.blocks['start_trials'],self.blocks['end_trials'], \
                        self.blocks['transition_states'], self.blocks['reward_states']):
                self.blocks['trial_trans_state'][start_trial - 1:end_trial-1] = trans_state   
                self.blocks['trial_rew_state'][start_trial - 1:end_trial-1]  = reward_state   

        if self.stim_trials is not None:
            return (second_step, outcome, self.stim_trials[self.cur_trial])
        else:
            return (second_step, outcome)

class _exp_mov_ave:
    'Exponential moving average class.'
    def __init__(self, tau=None, init_value=0., alpha = None):
        if alpha is None: alpha = 1 - np.exp(-1/tau)
        self._alpha = alpha
        self._m = 1 - alpha
        self.init_value = init_value
        self.reset()

    def reset(self, init_value = None):
        if init_value:
            self.init_value = init_value
        self.ave = self.init_value

    def update(self, sample):
        self.ave = (self.ave*self._m) + (self._alpha*sample)


def _with_prob(prob):
    'return true / flase with specified probability .'
    return random() < prob

def _get_stim_trials(n_trials, min_ISI=2, mean_TPST=6):
    ''' Generate pattern of stim trials disributed with min_ISI + exponential disribution
    of trials between stim trials to give mean_TPST trials per stim trial.'''
    stim_prob = 1. / (mean_TPST - min_ISI) 
    trials_since_last_stim = 0
    stim_trials = np.zeros(n_trials, bool)
    for i in range(n_trials):
        trials_since_last_stim += 1
        if ((trials_since_last_stim > min_ISI) and _with_prob(stim_prob)): 
            trials_since_last_stim = 0
            stim_trials[i] = True
    return stim_trials

#------------------------------------------------------------------------------------
# Simulation.
#------------------------------------------------------------------------------------

class simulated_session(session):
    '''Stores agent parameters and simulated data, supports plotting as for experimental
    session class.
    '''
    def __init__(self, agent, params_T, n_trials = 1000, task = Extended_two_step()):
        '''Simulate session with current agent and task parameters.'''
        self.param_names = agent.param_names
        self.true_params_T = params_T
        self.subject_ID = -1 
        try: # Not possible for e.g. unit range params_T with value 0 or 1.
            self.true_params_U = mf.transTU(params_T, agent.param_ranges)
        except Exception: 
            self.true_params_U = None
        self.n_trials = n_trials
        choices, second_steps, outcomes = agent.simulate(task, params_T, n_trials)
        
        self.trial_data = {'choices'      : choices,
                           'transitions'  : (choices == second_steps).astype(int),
                           'second_steps' : second_steps,
                           'outcomes'     : outcomes}

        if hasattr(task,'blocks'):
            self.blocks = deepcopy(task.blocks)

        if task.stim_trials is not None:
            self.stim_trials = task.stim_trials[:-1]
 

def sim_sessions_from_pop_fit(agent, population_fit, n_ses=10, n_trials=1000,
                              task = Extended_two_step()):
    '''Simulate sessions using parameter values drawn from the population distribution specified
    by population_fit. alternatively a dictionary of means and variances for each paramter can be
    specified.'''
    assert population_fit['param_names'] == agent.param_names, 'Agent parameters do not match fit.'
    _sim_func_ = partial(_sim_func, population_fit, agent, n_trials, task)
    sessions = pp.map(_sim_func_, range(n_ses))
    return sessions

def _sim_func(population_fit, agent, n_trials, task, i):
        params_T = mf._sample_params_T(population_fit)
        return simulated_session(agent, params_T, n_trials, task)

def sim_ses_from_pop_means(agent, population_fit, n_trials = 10000, task = Extended_two_step()):
    '''Simulate a single session with agent parameter values set to the mean
    values of the population level distribution.
    '''
    assert population_fit['param_names'] == agent.param_names, 'Agent parameters do not match fit.'
    params_T = mf._trans_UT(population_fit['pop_dists']['means'], agent.param_ranges)
    return simulated_session(agent, params_T, n_trials, task)

#---------------------------------------------------------------------------------------
# Regression fit to RL simulation.
#---------------------------------------------------------------------------------------

def LR_fit_to_RL_sim(RL_agents, LR_agent, sessions=None, RL_fits=None, LR_fit=None, fig_no=1,
                     n_trials=500, n_ses=4000, use_pop_dists=True):
    '''Fit RL agents to sessions, simulate data for each RL agent with fitted
     parameters. Fit logistic regression model to data and simulated sessions
     and plot logistic regression fits.'''
    if RL_fits is None:
        print('Fitting RL agents to data.')
        RL_fits = [mf.fit_population(sessions, RL_agent) for RL_agent in RL_agents]
    if LR_fit is None:
        print('Fitting LR agent to data.')
        LR_fit = mf.fit_population(sessions, LR_agent)
    print('Fitting to simulated data.')
    LR_sim_fits = []
    for RL_agent, RL_fit in zip(RL_agents, RL_fits):
        if use_pop_dists: # Simulate sessions with parameters drawn from population distribution.
            simulated_sessions = sim_sessions_from_pop_fit(RL_agent, RL_fit, n_ses, n_trials)
            LR_sim_fits.append(mf.fit_population(simulated_sessions, LR_agent))
        else: # Simulate sessions with parameters set to population mean.
            simulated_sessions = [sim_ses_from_pop_means(RL_agent, RL_fit, n_trials)
                                  for i in range(n_ses)]
            LR_sim_fits.append([mf.fit_session(sim_session,LR_agent)['params_U']
                                for sim_session in simulated_sessions])
    mp.model_fit_plot(LR_fit, fig_no = fig_no)
    plt.figure(fig_no).gca().set_prop_cycle(plt.cycler('color', ['gold', 'deeppink', 'm', 'y', 'k']))
    for LR_sim_fit, RL_agent in zip(LR_sim_fits, RL_agents):
            if use_pop_dists:
                plt.plot(np.arange(LR_agent.n_params)+0.5, 
                LR_sim_fit['pop_dists']['means'], linestyle = '', marker = 'o',
                markeredgecolor = 'none', markersize = 7, label = RL_agent.name)
            else:
                plt.errorbar(np.arange(LR_agent.n_params)+0.5, np.mean(LR_sim_fit,0),
                sem(LR_sim_fit,0), linestyle = '', capsize = 0,  elinewidth = 3,
                marker = 'o', markeredgecolor = 'none', markersize = 7, label = RL_agent.name)
    plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1.028))
    plt.xlim(LR_agent.n_params-3,LR_agent.n_params)
    plt.ylim(-0.4,0.6)