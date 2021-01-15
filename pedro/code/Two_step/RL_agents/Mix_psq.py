from ._RL_agent import *

class Mix_psq(RL_agent):
    '''Mixture agent with decays.'''

    def __init__(self, kernels = ['bias', 'ck', 'ssk']):
        self.name = 'Mix_psq'
        self.param_names  = ['alpQ', 'decQ', 'lbd' , 'alpT', 'decT', 'G_td', 'G_mb']
        self.param_ranges = ['unit', 'unit', 'unit', 'unit', 'unit', 'pos' , 'pos' ]
        RL_agent.__init__(self, kernels)

    @jit
    def session_likelihood(self, session, params_T, get_DVs = False):

        # Unpack trial events.
        choices, second_steps, outcomes = session.unpack_trial_data('CSO')

        # Unpack parameters.
        alpQ, decQ, lbd, alpT, decT, G_td, G_mb = params_T[:7]

        #Variables.
        Q = np.zeros([2,2,session.n_trials]) # First step TD values. (c,ps,t)
        V = np.zeros([2,session.n_trials]) # Second step TD values.
        T = np.zeros([2,session.n_trials]) # Transition probabilities.
        T[:,0] = 0.5 # Initialize first trial transition probabilities.
        ps = 0 # Previous trial second step state.

        for i, (c, s, o) in enumerate(zip(choices[:-1], second_steps, outcomes)): # loop over trials.

            n = 1 - c  # Action not chosen at first step.
            r = 1 - s  # State not reached at second step.

            # Update action values and transition probabilities.

            Q[:,:,i+1] = Q[:,:,i] * (1.-decQ) # First step forgetting.
            V[r,i+1] = V[r,i] * (1.-decQ) # Second step forgetting.

            Q[c,ps,i+1] = (1.-alpQ)*Q[c,ps,i] + alpQ*((1.-lbd)*V[s,i] + lbd*o) # First step TD update.
            V[s,i+1] = (1.-alpQ)*V[s,i] + alpQ*o  # Second step TD update.

            T[n,i+1] = T[n,i] - decT*(T[n,i]-0.5) # Transition prob. forgetting.
            T[c,i+1] = (1.-alpT)*T[c,i] + alpT*s  # Transition prob. update.

            ps = s

        # Evaluate net action values and likelihood. 

        M = T*V[1,:] + (1.-T)*V[0,:] # Model based action values.
        Q_net = G_mb*M      # Mixture of model based and model free values.
        Q_net[:,1:] += G_td*Q[:,second_steps[:-1],np.arange(1,session.n_trials)]
        Q_net = self.apply_kernels(Q_net, choices, second_steps, params_T)

        if get_DVs: return self.get_DVs(session, params_T, Q_net, Q, M)
        else:       return session_log_likelihood(choices, Q_net)