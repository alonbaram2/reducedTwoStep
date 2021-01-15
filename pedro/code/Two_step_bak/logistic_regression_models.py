import numpy as np
import time
import pylab as plt
from .RL_utils import softmax_bin as softmax
from . import utility as ut
from . import RL_utils as ru
from . import RL_plotting as rp

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


    def _select_trials(self, session):
        selected_trials = session.select_trials(self.trial_select['selection_type'],
                                                self.trial_select['select_n'])
        if 'trial_mask' in self.trial_select.keys():
            trial_mask = getattr(session, self.trial_select['trial_mask'])
            if self.trial_select['invert_mask']:
                trial_mask = ~trial_mask
            selected_trials = selected_trials & trial_mask
        return selected_trials
        
    def session_likelihood(self, session, params_T, eval_grad = False):

        bias = params_T[0]
        weights = params_T[1:]

        choices = session.CTSO['choices']

        if not hasattr(session,'predictors'):
            predictors = self._get_session_predictors(session) # Get array of predictors
        else:
            predictors = session.predictors

        assert predictors.shape[0] == session.n_trials,  'predictor array does not match number of trials.'
        assert predictors.shape[1] == len(weights), 'predictor array does not match number of weights.'

        if self.trial_select: # Only use subset of trials.
            selected_trials = self._select_trials(session)
            choices = choices[selected_trials]
            predictors = predictors[selected_trials,:]

        # Evaluate session log likelihood.

        Q = np.dot(predictors,weights) + bias
        P = ru.logistic(Q)  # Probability of making choice 1
        Pc = 1 - P - choices + 2. * choices * P  

        session_log_likelihood = sum(ru.protected_log(Pc)) 

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
        
        choices, second_steps = ut.CTSO_unpack(session.CTSO, 'CS', float)

        predictors = np.array((choices, second_steps)).T - 0.5
        predictors = np.vstack((np.zeros(2),predictors[:-1,:]))  # First trial events predict second trial etc.

        return predictors

# -------------------------------------------------------------------------------------
# Configurable logistic regression Model.
# -------------------------------------------------------------------------------------

class config_log_reg(_logistic_regression_model):

    '''
    Configurable logistic regression agent. Arguments:

    predictors - The basic set of predictors used is specified with predictors argument.  

    lags        - By default each predictor is only used at a lag of -1 (i.e. one trial predicting the next).
                 The lags argument is used to specify the use of additional lags for specific predictors:
                 e.g. lags = {'outcome': 3, 'choice':2} specifies that the outcomes on the previous 3 trials
                 should be used as predictors, while the choices on the previous 2 trials should be used

    norm        - Set to True to normalise predictors such that each has the same mean absolute value.

    orth        - The orth argument is used to specify an orthogonalization scheme.  
                 orth = [('trans_CR', 'choice'), ('trCR_x_out', 'correct')] will orthogonalize trans_CR relative
                 to 'choice' and 'trCR_x_out' relative to 'correct'.

    mov_ave_CR   - Specifies whether transitions are classified common or rare based on block structue (False)
                 or based on a moving average of recent choices.

    trial_mask  - Subselect trials based on session attribute with specified name.  E.g. if 
                trial_mask is set to 'stim_choices', the variable session.stim_choices 
                (which must be a boolean array of length n_trials) will be used to select
                trials for each session fit. The additional invert_mask variable can be 
                used to invert the mask.  Used for subselecting trials with e.g. optogenetic
                stimulation.
    '''


    def __init__(self, predictors = ['side', 'correct','choice','outcome',
                'trans_CR', 'trCR_x_out'], lags = {}, norm = False, orth = False, 
                mov_ave_CR = False, trial_mask = None, invert_mask = False):

        self.name = 'config_lr'
        self.base_predictors = predictors # predictor names ignoring lags.
        self.orth = orth 
        self.norm = norm

        self.predictors = [] # predictor names including lags.
        for predictor in self.base_predictors:
            if predictor in list(lags.keys()):
                for i in range(lags[predictor]):
                    self.predictors.append(predictor + '-' + str(i + 1)) # Lag is indicated by value after '-' in name.
            else:
                self.predictors.append(predictor) # If no lag specified, defaults to 1.

        self.n_predictors = len(self.predictors)

        self.mov_ave_CR = mov_ave_CR 

        if mov_ave_CR: # Use moving average of recent transitions to evaluate 
            self.tau = 10.  # common vs rare transitions.

        if trial_mask or not mov_ave_CR:
            self.trial_select = {'selection_type': 'xtr',
                                 'select_n'      : 20}
            if trial_mask:
                self.trial_select['trial_mask']  = trial_mask
                self.trial_select['invert_mask'] = invert_mask


        _logistic_regression_model.__init__(self)

    def _get_session_predictors(self, session):
        '''Calculate and return values of predictor variables for all trials in session.
        '''

        # Evaluate base (non-lagged) predictors from session events.

        choices, transitions_AB, second_steps, outcomes = ut.CTSO_unpack(session.CTSO, dtype = bool)
        trans_state = session.blocks['trial_trans_state']    # Trial by trial state of the tranistion matrix (A vs B)

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

            if p == 'correct':  # 0.5, 0, -1 for high poke being correct, neutral, incorrect option.
                bp_values[p] = 0.5 * (session.blocks['trial_rew_state'] - 1) * \
                              (2 * session.blocks['trial_trans_state'] - 1)  
      
            elif p == 'side': # 0.5, -0.5 for left, right side reached at second steplt. 
                bp_values[p] = second_steps - 0.5

            elif p == 'side_x_out': # 0.5, -0.5.  Side predictor invered by trial outcome.
                bp_values[p] = (second_steps == outcomes) - 0.5

            # The following predictors all predict stay probability rather than high vs low.
            # e.g the outcome predictor represents the effect of outcome on stay probabilty.
            # This is implemented by inverting the predictor dependent on the choice made on the trial.

            elif p ==  'choice': # 0.5, - 0.5 for choices high, low.
                bp_values[p] = choices - 0.5

            elif p == 'good_side': # 0.5, 0, -0.5 for reaching good, neutral, bad second link state.
                bp_values[p] = 0.5 * (session.blocks['trial_rew_state'] - 1) * (2 * (second_steps == choices) - 1)
                    
            elif p ==  'outcome': # 0.5 , -0.5 for  rewarded , not rewarded.
                bp_values[p] = (outcomes == choices) - 0.5

            elif p ==  'block':     # 0.5, -0.5 for A , B blocks.
                bp_values[p] = (trans_state == choices) - 0.5

            elif p == 'block_x_out': # 0.5, -0.5 for A , B blocks inverted by trial outcome.
                bp_values[p] = ((outcomes == trans_state) == choices) - 0.5

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

            elif p ==  'trans_CR_rew': # 0.5, -0.5, for common, rare transitions on rewarded trials, otherwise 0.
                    if self.mov_ave_CR: 
                        bp_values[p] = transitions_CR * choices_0_mean * outcomes
                    else: 
                        bp_values[p] = (((transitions_CR) == choices)  - 0.5) * outcomes

            elif p ==  'trans_CR_non_rew': # 0.5, -0.5, for common, rare transitions on non-rewarded trials, otherwise 0.
                    if self.mov_ave_CR: 
                        bp_values[p] = transitions_CR * choices_0_mean * ~outcomes
                    else: 
                        bp_values[p] = (((transitions_CR) == choices)  - 0.5) * ~outcomes

            elif p ==  'out_dep_bias': # 0.5 , -0.5 for  rewarded , not rewarded.
                bp_values[p] = outcomes - 0.5
                
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
                lag = int(plt.split('-')[1]) 
                bp_name = plt.split('-')[0]
            else:        # Use default lag.
                lag = 1
                bp_name = p
            predictors[lag:, i] = bp_values[bp_name][:-lag]

        return predictors
















