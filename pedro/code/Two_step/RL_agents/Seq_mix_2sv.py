from ._RL_agent import *

class Seq_mix_2sv(RL_agent):
    '''Mixture agent which learr model free action values for sequenes at second step,
    but also second step state values which are used for model-based lookahead. Seperate
    learing rates used at first and second steps.'''

    def __init__(self, kernels = ['bias', 'ck', 'ssk']):
        self.name = 'Seq_mix_2sv'
        self.param_names  = ['alpQ', 'alpV', 'lbd' , 'decQ', 'decV', 'alpT', 'decT', 'G_td', 'G_mb']
        self.param_ranges = ['unit', 'unit', 'unit', 'unit', 'unit', 'unit', 'unit', 'pos' , 'pos' ]
        RL_agent.__init__(self, kernels)

    @jit
    def session_likelihood(self, session, params_T, get_DVs = False):

        # Unpack trial events.
        choices, second_steps, outcomes = session.unpack_trial_data('CSO')

        # Unpack parameters.
        alpQ, alpV, lbd, decQ, decV, alpT, decT, G_td, G_mb = params_T[:9]

        #Variables.
        Q1 = np.zeros([2,session.n_trials])   # First step TD action values.
        Q2 = np.zeros([2,2,session.n_trials]) # Model free second step action values [c,s,trial]
        V  = np.zeros([2,session.n_trials])   # Second step state values             [s,trial].
        T  = np.zeros([2,session.n_trials])    # Transition probabilities.
        T[:,0] = 0.5 # Initialize first trial transition probabilities.

        for i, (c, s, o) in enumerate(zip(choices[:-1], second_steps, outcomes)): # loop over trials.

            n = 1 - c  # Action not chosen at first step.
            r = 1 - s  # State not reached at second step.

            # Update action values and transition probabilities.

            Q1[n,i+1]   = Q1[n,i]   * (1.-decQ) # First step forgetting.
            Q2[:,:,i+1] = Q2[:,:,i] * (1.-decQ) # Second step forgetting.
            V[r,i+1] = V[r,i] * (1.-decV) # Second step forgetting.

            Q1[c,i+1] = (1.-alpQ)*Q1[c,i] + alpQ*(Q2[c,s,i]*(1.-lbd)+lbd*o) # First step TD update.
      
            Q2[c,s,i+1] = (1.-alpQ)*Q2[c,s,i] + alpQ*o # Second step TD update.
            V[s,i+1]    = (1.-alpV)*V[s,i]    + alpV*o # Second step TD update.

            T[n,i+1] = T[n,i] - decT*(T[n,i]-0.5) # Transition prob. forgetting.
            T[c,i+1] = (1.-alpT)*T[c,i] + alpT*s  # Transition prob. update.

        # Evaluate net action values and likelihood. 

        M = T*V[1,:] + (1.-T)*V[0,:] # Model based action values.
        Q_net = G_td*Q1 + G_mb*M # Mixture of model based and model free values. 
        Q_net = self.apply_kernels(Q_net, choices, second_steps, params_T)

        if get_DVs: return self.get_DVs(session, params_T, Q_net, Q, M)
        else:       return session_log_likelihood(choices, Q_net)