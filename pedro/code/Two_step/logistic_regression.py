import numpy as np
from random import choice
from sklearn.utils import resample
from functools import partial
from . import utility as ut
from . import model_fitting as mf 
from . import model_plotting as mp
from . import parallel_processing as pp
from .group_comparison import _print_P_values

# -------------------------------------------------------------------------------------
# logistic_regression_model
# -------------------------------------------------------------------------------------

class _logistic_regression_model():
    '''
    Superclass for logistic regression models which provides generic likelihood and 
    likelihood gradient evaluation.  To implement a specific logistic regression model 
    this class is subclassed with a _get_session_predictors function which takes a
    session as its argument and returns the array of predictors.  

    The trial_select variable can be used to specify rules for including only a subset 
    of trials in the analysis.  Set this variable to False to use all trials. include
    '''

    def __init__(self):

        self.n_params = 1 + len(self.predictors)

        self.param_ranges = ('all_unc', self.n_params)
        self.param_names  = ['bias'] + self.predictors

        if not hasattr(self, 'trial_select'): # Selection 
            self.trial_select = False

        self.calculates_gradient = True
        self.type = 'log_reg'

    def _select_trials(self, session):
        if 'selection_type' in self.trial_select.keys():
            selected_trials = session.select_trials(self.trial_select['selection_type'],
                                                    self.trial_select['select_n'])
        else:
            selected_trials=np.ones(session.n_trials,bool)
        if 'trial_mask' in self.trial_select.keys():
            trial_mask = getattr(session, self.trial_select['trial_mask'])
            if self.trial_select['invert_mask']:
                trial_mask = ~trial_mask
            selected_trials = selected_trials & trial_mask
        return selected_trials
        
    def session_likelihood(self, session, params_T, eval_grad = False):

        bias = params_T[0]
        weights = params_T[1:]

        choices = session.trial_data['choices']

        if not hasattr(session,'predictors'):
            predictors = self._get_session_predictors(session) # Get array of predictors
        else:
            predictors = session.predictors

        assert predictors.shape[0] == session.n_trials, 'predictor array does not match number of trials.'
        assert predictors.shape[1] == len(weights), 'predictor array does not match number of weights.'

        if self.trial_select: # Only use subset of trials.
            selected_trials = self._select_trials(session)
            choices = choices[selected_trials]
            predictors = predictors[selected_trials,:]

        # Evaluate session log likelihood.

        Q = np.dot(predictors,weights) + bias
        Q[Q < -ut.log_max_float] = -ut.log_max_float # Protect aganst overflow in exp.
        P = 1./(1. + np.exp(-Q))  # Probability of making choice 1
        Pc = 1 - P - choices + 2. * choices * P  # Probability of chosen action.

        session_log_likelihood = sum(ut.log_safe(Pc)) 

        # Evaluate session log likelihood gradient.

        if eval_grad:
            dLdQ  = - 1 + 2 * choices + Pc - 2 * choices * Pc
            dLdB = sum(dLdQ) # Likelihood gradient w.r.t. bias paramter.
            dLdW = sum(np.tile(dLdQ,(len(weights),1)).T * predictors, 0) # Likelihood gradient w.r.t weights.
            session_log_likelihood_gradient = np.append(dLdB,dLdW)
            return (session_log_likelihood, session_log_likelihood_gradient)
        else:
            return session_log_likelihood

# -------------------------------------------------------------------------------------
# Kernels only.
# -------------------------------------------------------------------------------------


class kernels_only(_logistic_regression_model):

    '''
    Equivilent to RL agent using only bias, choice kernel (stay), and second step kernel (side)
    '''

    def __init__(self):

        self.name = 'kernels_only'

        self.predictors = ['choice', 'side']

        _logistic_regression_model.__init__(self)


    def _get_session_predictors(self, session):
        '''Calculate and return values of predictor variables for all trials in session.
        '''
        
        choices, second_steps = session.unpack_trial_data('CS', float)

        predictors = np.array((choices, second_steps)).T - 0.5
        predictors = np.vstack((np.zeros(2),predictors[:-1,:]))  # First trial events predict second trial etc.

        return predictors

# -------------------------------------------------------------------------------------
# Configurable logistic regression Model.
# -------------------------------------------------------------------------------------

# Dictionary specifying predictors for commonly used logistic regression models.
pred_dict = {'standard': ['correct','choice','outcome', 'trans_CR', 'trCR_x_out'],
             '-correct': [          'choice','outcome', 'trans_CR', 'trCR_x_out']}

class config_log_reg(_logistic_regression_model):

    '''
    Configurable logistic regression agent. Arguments:

    predictors - The basic set of predictors used is specified with predictors argument.  

    lags        - By default each predictor is only used at a lag of -1 (i.e. one trial predicting the next).
                 The lags argument is used to specify the use of additional lags for specific predictors:
                 e.g. lags = {'outcome': 3, 'choice':2} specifies that the outcomes on the previous 3 trials
                 should be used as predictors, while the choices on the previous 2 trials should be used.  If an
                 interger is provided as the lags argument all predictors are given this number of lags.

    norm        - Set to True to normalise predictors such that each has the same mean absolute value.

    orth        - The orth argument is used to specify an orthogonalization scheme.  
                 orth = [('trans_CR', 'choice'), ('trCR_x_out', 'correct')] will orthogonalize trans_CR relative
                 to 'choice' and 'trCR_x_out' relative to 'correct'.

    trial_mask  - Subselect trials based on session attribute with specified name.  E.g. if 
                trial_mask is set to 'stim_choices', the variable session.stim_choices 
                (which must be a boolean array of length n_trials) will be used to select
                trials for each session fit. The additional invert_mask variable can be 
                used to invert the mask.  Used for subselecting trials with e.g. optogenetic
                stimulation.
    '''


    def __init__(self, predictors = 'standard', lags = {}, norm = False, orth = False, 
                 trial_mask = None, invert_mask = False, mov_ave_CR = False):

        self.name = 'config_lr'
        self.orth = orth 
        self.norm = norm
        self.mov_ave_CR = mov_ave_CR

        if type(predictors) == list:
            self.base_predictors = predictors # predictor names ignoring lags.
        else:
            self.base_predictors = pred_dict[predictors]

        if type(lags) == int:
            lags = {p:lags for p in predictors}

        self.predictors = [] # predictor names including lags.
        for predictor in self.base_predictors:
            if predictor in list(lags.keys()):
                for i in range(lags[predictor]):
                    self.predictors.append(predictor + '-' + str(i + 1)) # Lag is indicated by value after '-' in name.
            else:
                self.predictors.append(predictor) # If no lag specified, defaults to 1.

        self.n_predictors = len(self.predictors)

        self.trial_select = {'selection_type': 'xtr',
                             'select_n'      : 20}      

        if mov_ave_CR: # Use moving average of recent transitions to evaluate common vs rare transitions. 
            self.trial_select['selection_type'] = 'all'
            self.tau = 10.      

        if trial_mask:
            self.trial_select['trial_mask']  = trial_mask
            self.trial_select['invert_mask'] = invert_mask

        _logistic_regression_model.__init__(self)

    def _get_session_predictors(self, session):
        'Calculate and return values of predictor variables for all trials in session.'

        # Evaluate base (non-lagged) predictors from session events.

        choices, transitions_AB, second_steps, outcomes = session.unpack_trial_data(dtype = bool)
        trans_state = session.blocks['trial_trans_state']    # Trial by trial state of the tranistion matrix (A vs B)
        transitions_CR = transitions_AB == trans_state
        transition_CR_x_outcome = transitions_CR == outcomes 
        correct = -0.5*(session.blocks['trial_rew_state']-1)* \
                       (2*session.blocks['trial_trans_state']-1) 

        if self.mov_ave_CR:
            trans_mov_ave = np.zeros(len(choices))
            trans_mov_ave[1:] = (5./3.) * ut.exp_mov_ave(transitions_AB - 0.5, self.tau, 0.)[:-1] # Average of 0.5 for constant 0.8 transition prob.
            transitions_CR = 2 * (transitions_AB - 0.5) * trans_mov_ave
            transition_CR_x_outcome = 2. * transitions_CR * (outcomes - 0.5) 
            choices_0_mean = 2 * (choices - 0.5)
        else:  
            transitions_CR = transitions_AB == trans_state
            transition_CR_x_outcome = transitions_CR == outcomes 

        bp_values = {} 

        for p in self.base_predictors:

            if p == 'correct':  # 0.5, 0, -0.5 for high poke being correct, neutral, incorrect option.
                bp_values[p] =  correct
      
            elif p == 'side': # 0.5, -0.5 for left, right side reached at second step. 
                bp_values[p] = second_steps - 0.5

            elif p ==  'choice': # 0.5, - 0.5 for choices high, low.
                bp_values[p] = choices - 0.5
                    
            elif p ==  'outcome': # 0.5 , -0.5 for  rewarded , not rewarded.
                bp_values[p] = (outcomes == choices) - 0.5

            elif p ==  'trans_CR': # 0.5, -0.5 for common, rare transitions.  
                if self.mov_ave_CR:            
                    bp_values[p] = transitions_CR * choices_0_mean 
                else: 
                    bp_values[p] = ((transitions_CR) == choices)  - 0.5               

            elif p == 'trCR_x_out': # 0.5, -0.5 for common, rare transitions inverted by trial outcome.
                if self.mov_ave_CR: 
                    bp_values[p] = transition_CR_x_outcome * choices_0_mean 
                else:
                    bp_values[p] = (transition_CR_x_outcome  == choices) - 0.5

            elif p == 'rew_com':  # Rewarded common transition predicts repeating choice.
                bp_values[p] = ( outcomes &  transitions_CR) * (choices - 0.5)

            elif p == 'rew_rare':  # Rewarded rare transition predicts repeating choice.
                bp_values[p] = ( outcomes & ~transitions_CR) * (choices - 0.5)   

            elif p == 'non_com':  # Non-rewarded common transition predicts repeating choice.
                bp_values[p] = (~outcomes &  transitions_CR) * (choices - 0.5)

            elif p == 'non_rare':  # Non-Rewarded rare transition predicts repeating choice.
                bp_values[p] = (~outcomes & ~transitions_CR) * (choices - 0.5)
                
        # predictor orthogonalization.

        if self.orth: 
            for A, B in self.orth: # Remove component of predictor A that is parrallel to predictor B. 
                bp_values[A] = bp_values[A] - ut.projection(bp_values[B], bp_values[A])

        # predictor normalization.
        if self.norm:
            for p in self.base_predictors:
                bp_values[p] = bp_values[p] * 0.5 / np.mean(np.abs(bp_values[p]))

        # Generate lagged predictors from base predictors.

        predictors = np.zeros([session.n_trials, self.n_predictors])

        for i,p in enumerate(self.predictors):  
            if '-' in p: # Get lag from predictor name.
                lag = int(p.split('-')[1]) 
                bp_name = p.split('-')[0]
            else:        # Use default lag.
                lag = 1
                bp_name = p
            predictors[lag:, i] = bp_values[bp_name][:-lag]

        return predictors

# -------------------------------------------------------------------------------------
# Logistic regression analyses.
# -------------------------------------------------------------------------------------

def logistic_regression(sessions, predictors='standard', fig_no=1, title=None, per_sub=False):
    ''' Run and plot logistic regression analysis on specified sessions using
    logistic regression model with specified predictors. 
    '''
    if predictors == 'lagged':
        model = config_log_reg(['rew_com', 'rew_rare', 'non_com', 'non_rare', 'choice'], lags=3)
    else:
        model = config_log_reg(predictors)
    population_fit = mf.fit_population(sessions, model)
    if per_sub:
        mp.per_subject_fit_plot(population_fit, fig_no)
    elif predictors == 'lagged':
        mp.lagged_fit_plot(population_fit, fig_no=fig_no)
    else:
        mp.model_fit_plot(population_fit, fig_no, title=title)


def longditudinal_log_reg(experiment, predictors = 'standard', epoch_len = 1, fig_no = 1, title=None):
    model = config_log_reg(predictors)
    longdit_fit = mf.longditudinal_fit(experiment, model,
                           epoch_len = epoch_len)
    mp.longditudinal_fit_plot(longdit_fit,fig_no = fig_no, title=title)

# -------------------------------------------------------------------------------------
# Bootstrap significance testing.
# -------------------------------------------------------------------------------------

def predictor_significance_test(sessions, agent, n_perms = 1000):
    '''Test whether logistic regression predictor loadings are significantly different
    from zero by fitting model to bootstrap resampled populations of sessions.'''

    mf._precalculate_fits(sessions, agent) # Store first round fits on sessions.
    
    sessions_agent_list = [(sessions,agent)]*n_perms

    def eval_P_values():
        pop_means = np.array([bf['pop_dists']['means'] for bf in bootstrap_fits])
        P_values = np.min((np.mean(pop_means > 0, 0), np.mean(pop_means < 0, 0)),0)*2.
        P_value_dict = dict(zip(agent.param_names, P_values))
        _print_P_values(P_value_dict)

    bootstrap_fits = []
    for i, bs_fit in enumerate(pp.imap(_permute_and_fit, sessions_agent_list, ordered = False)):
        bootstrap_fits.append(bs_fit)
        print('Fitted permuted dataset {} of {}'.format(i+1, n_perms))
        if i > 0 and i%10 == 9:
            eval_P_values()

    eval_P_values()
    for session in sessions: del(session.fit) # Clear precalcuated fits.

def _permute_and_fit(sessions_agent):
    sessions, agent = sessions_agent
    bs_sessions = resample(sessions)
    bs_fits = [session.fit for session in bs_sessions]   
    return mf.fit_population(bs_sessions, agent, session_fits = bs_fits)