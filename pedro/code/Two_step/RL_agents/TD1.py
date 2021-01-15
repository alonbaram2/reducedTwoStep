from ._RL_agent import *

class TD1(RL_agent):
    'TD1 (direct reinforcement) agent.'

    def __init__(self, kernels = ['bias', 'ck', 'ssk']):
        self.name = 'TD1'
        self.param_names  = ['alp', 'iTemp']
        self.param_ranges = ['unit', 'pos' ]
        RL_agent.__init__(self, kernels)

    @jit
    def session_likelihood(self, session, params_T, get_DVs = False):

        # Unpack trial events.
        choices, second_steps, outcomes = session.unpack_trial_data('CSO')

        # Unpack parameters.
        alp, iTemp = params_T[:2]  

        #Variables.        
        Q = np.zeros([2,session.n_trials])# First step TD values.

        for i, (c, s, o) in enumerate(zip(choices[:-1], second_steps, outcomes)): # loop over trials.

            n = 1 - c  # Action not chosen at first step.

            # Update action values. 

            Q[c,i+1]  = (1.-alp)*Q[c,i] +  alp*o  # First step TD update.

        # Evaluate net action values and likelihood. 

        Q_net = self.apply_kernels(Q, choices, second_steps, params_T)

        if get_DVs: return self.get_DVs(session, params_T, Q_net, Q)
        else:       return session_log_likelihood(choices, Q_net, iTemp)