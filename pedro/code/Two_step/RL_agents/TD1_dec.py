from ._RL_agent import *

class TD1_dec(RL_agent):
    'TD1 (direct reinforcement) agent with forgetting.'

    def __init__(self, kernels = ['bias', 'ck', 'ssk']):
        self.name = 'TD1_dec'
        self.param_names  = ['alp' , 'iTemp', 'dec ']
        self.param_ranges = ['unit', 'pos'  , 'unit']
        RL_agent.__init__(self, kernels)

    @jit
    def session_likelihood(self, session, params_T, get_DVs = False):

        # Unpack trial events.
        choices, second_steps, outcomes = session.unpack_trial_data('CSO')

        # Unpack parameters.
        alp, iTemp, dec = params_T[:3]  

        #Variables.        
        Q = np.zeros([2,session.n_trials])# First step TD values.

        for i, (c, s, o) in enumerate(zip(choices[:-1], second_steps, outcomes)): # loop over trials.

            n = 1 - c  # Action not chosen at first step.

            # Update action values. 

            Q[n,i+1] = Q[n,i] * (1.-dec)          # First step forgetting.
            Q[c,i+1] = (1.-alp) * Q[c,i] +  alp*o # First step TD update.

        # Evaluate net action values and likelihood. 

        Q_net = self.apply_kernels(Q, choices, second_steps, params_T)

        if get_DVs: return self.get_DVs(session, params_T, Q_net, Q)
        else:       return session_log_likelihood(choices, Q_net, iTemp)

    def simulate(self, task, params_T, n_trials):

        # Unpack parameters.
        alp, iTemp, dec = params_T[:3]  

        #Variables.        
        Q = np.zeros([2,n_trials+1]) ## First step TD values.
        Q_net = np.zeros(2)
        choices, second_steps, outcomes = (np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int))

        task.reset(n_trials)
        for i in range(n_trials):

            # Generate trial events.
            c = choose(softmax(Q_net, iTemp)) 
            s, o = task.trial(c)
            n = 1 - c  # Action not chosen at first step.
            choices[i], second_steps[i], outcomes[i]  = (c, s, o)

            # Update action values. 

            Q[n,i+1] = Q[n,i] * (1.-dec)               # First step forgetting.
            Q[c,i+1]  = (1.-alp) * Q[c,i] +  alp * o  # First step TD update.

            Q_net = self.apply_kernels(Q[:,i+1], c, s, params_T)

        return choices, second_steps, outcomes