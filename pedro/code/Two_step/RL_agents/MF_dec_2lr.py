from ._RL_agent import *

class MF_dec_2lr(RL_agent):
    'Model-free agent with forgetting, seperate learning rates at first and second step.'

    def __init__(self, kernels = ['bias', 'ck', 'ssk']):
        self.name = 'MF_dec_2lr'
        self.param_names  = ['alpQ', 'alpV', 'iTemp', 'lbd' , 'decQ', 'decV']
        self.param_ranges = ['unit', 'unit', 'pos'  , 'unit', 'unit', 'unit']
        RL_agent.__init__(self, kernels)

    @jit
    def session_likelihood(self, session, params_T, get_DVs = False):

        # Unpack trial events.
        choices, second_steps, outcomes = session.unpack_trial_data('CSO')

        # Unpack parameters.
        alpQ, alpV, iTemp, lbd, decQ, decV = params_T[:6] # Q value decay parameter.

        #Variables.
        Q = np.zeros([2,session.n_trials]) # First step TD values.
        V = np.zeros([2,session.n_trials]) # Second step TD values.

        for i, (c, s, o) in enumerate(zip(choices[:-1], second_steps, outcomes)): # loop over trials.

            n = 1 - c  # Action not chosen at first step.
            r = 1 - s  # State not reached at second step.

            # Update action values. 

            Q[n,i+1] = Q[n,i] * (1.-decQ)   # First step forgetting.
            V[r,i+1] = V[r,i] * (1.-decV)   # Second step forgetting.

            Q[c,i+1] = (1.-alpQ)*Q[c,i] + alpQ*((1.-lbd)*V[s,i] + lbd*o) # First step TD update.
            V[s,i+1] = (1.-alpV)*V[s,i] + alpV*o # Second step TD update.

        # Evaluate net action values and likelihood. 

        Q_net = self.apply_kernels(Q, choices, second_steps, params_T)

        if get_DVs: return self.get_DVs(session, params_T, Q_net, Q)
        else:       return session_log_likelihood(choices, Q_net, iTemp)

    def simulate(self, task, params_T, n_trials):

        # Unpack parameters.
        alpQ, alpV, iTemp, lbd, decQ, decV = params_T[:6]

        Q = np.zeros([2,n_trials+1])  ## First step TD values.
        V = np.zeros([2,n_trials+1])  # Model free second step action values.
        Q_net = np.zeros(2)
        choices, second_steps, outcomes = (np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int))

        task.reset(n_trials)
        for i in range(n_trials):

            # Generate trial events.
            c = choose(softmax(Q_net, iTemp)) 
            s, o = task.trial(c)
            n = 1 - c  # Action not chosen at first step.
            r = 1 - s  # State not reached at second step.
            choices[i], second_steps[i], outcomes[i]  = (c, s, o)

            # Update action values. 

            Q[n,i+1] = Q[n,i] * (1.-decQ)   # First step forgetting.
            V[r,i+1] = V[r,i] * (1.-decV)   # Second step forgetting.

            Q[c,i+1] = (1.-alpQ)*Q[c,i] + alpQ*((1.-lbd)*V[s,i] + lbd*o) # First step TD update.
            V[s,i+1] = (1.-alpV)*V[s,i] + alpV*o # Second step TD update.

            Q_net = self.apply_kernels(Q[:,i+1], c, s, params_T)

        return choices, second_steps, outcomes