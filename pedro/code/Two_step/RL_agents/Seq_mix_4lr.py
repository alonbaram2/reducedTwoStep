from ._RL_agent import *

class Seq_mix_4lr(RL_agent):
    '''Mixture agent which learns model free action values for both sequences and
    second step actions and '''

    def __init__(self, kernels = ['bias', 'ck', 'ssk']):
        self.name = 'Seq_mix_4lr'
        self.param_names  = ['alpQ1', 'decQ1', 'alpQ2', 'decQ2', 'alpS', 'decS', 'alpV',
                             'decV', 'seq', 'lbd', 'alpT', 'decT', 'G_td', 'G_mb']
        self.param_ranges = ['unit']*12 + ['pos']*2
        RL_agent.__init__(self, kernels)

    @jit
    def session_likelihood(self, session, params_T, get_DVs = False):

        # Unpack trial events.
        choices, second_steps, outcomes = session.unpack_trial_data('CSO')

        # Unpack parameters.
        alpQ1, decQ1, alpQ2, decQ2, alpS, decS, alpV, decV, seq, lbd, alpT, decT, G_td, G_mb = params_T[:14]

        #Variables.
        Q1 = np.zeros([2,session.n_trials])   # First step TD action values.
        Q2 = np.zeros([2,session.n_trials])   # Second step TD action values.
        S = np.zeros([2,2,session.n_trials])  # Sequence values.
        V  = np.zeros([2,session.n_trials])   # Second step state values.
        T = np.zeros([2,session.n_trials])    # Transition probabilities.
        T[:,0] = 0.5 # Initialize first trial transition probabilities.

        for i, (c, s, o) in enumerate(zip(choices[:-1], second_steps, outcomes)): # loop over trials.

            n = 1 - c  # Action not chosen at first step.
            r = 1 - s  # State not reached at second step.

            # Update action values and transition probabilities.

            # Forgetting.

            Q1[n,i+1]  = Q1[n,i]  * (1.-decQ1) 
            Q2[r,i+1]  = Q2[r,i]  * (1.-decQ2) 
            S[:,:,i+1] = S[:,:,i] * (1.-decS)  
            V[r,i+1]   = V[r,i]   * (1.-decV)  

            # Learning

            Q1[c,i+1]  = (1.-alpQ1)*Q1[c,i] + \
                         alpQ1*((1.-lbd)*(seq*S[c,s,i]+(1.-seq)*Q2[s,i])+lbd*o) 
            S[c,s,i+1] = (1.-alpS) * S[c,s,i] + alpS*o 
            V[  s,i+1] = (1.-alpV) * V[  s,i] + alpV*o 
            Q2[ s,i+1] = (1.-alpQ2)* Q2[ s,i] + alpQ2*o

            T[n,i+1] = T[n,i] - decT*(T[n,i]-0.5) # Transition prob. forgetting.
            T[c,i+1] = (1.-alpT)*T[c,i] + alpT*s  # Transition prob. update.

        # Evaluate net action values and likelihood. 

        M = T*V[1,:] + (1.-T)*V[0,:] # Model based action values.
        Q_net = G_td*Q1 + G_mb*M # Mixture of model based and model free values. 
        Q_net = self.apply_kernels(Q_net, choices, second_steps, params_T)

        if get_DVs: return self.get_DVs(session, params_T, Q_net, Q, M)
        else:       return session_log_likelihood(choices, Q_net)