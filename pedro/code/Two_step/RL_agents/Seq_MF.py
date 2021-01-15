from ._RL_agent import *

class Seq_MF(RL_agent):
    'Model free agent which learr action values for sequenes at second step.'

    def __init__(self, kernels = ['bias', 'ck', 'ssk']):
        self.name = 'Seq_MF'
        self.param_names  = ['alp', 'iTemp', 'lbd' , 'dec' ]
        self.param_ranges = ['unit', 'pos' , 'unit', 'unit']
        RL_agent.__init__(self, kernels)

    @jit
    def session_likelihood(self, session, params_T, get_DVs = False):

        # Unpack trial events.
        choices, second_steps, outcomes = session.unpack_trial_data('CSO')

        # Unpack parameters.
        alp, iTemp, lbd, dec = params_T[:4]

        #Variables.
        Q = np.zeros([2,session.n_trials])   # First step TD action values.
        V = np.zeros([2,2,session.n_trials]) # Model free second step action values [c,s,trial]
        
        for i, (c, s, o) in enumerate(zip(choices[:-1], second_steps, outcomes)): # loop over trials.

            n = 1 - c  # Action not chosen at first step.
            r = 1 - s  # State not reached at second step.

            # Update model free action values. 

            Q[n,i+1]   = Q[n,i]   * (1.-dec) # First step forgetting.
            V[:,:,i+1] = V[:,:,i] * (1.-dec) # Second step forgetting.

            Q[c,i+1]   = (1.-alp)*Q[c,i]   + alp*(V[c,s,i]*(1.-lbd)+lbd*o) # First step TD update.
            V[c,s,i+1] = (1.-alp)*V[c,s,i] + alp*o  # Second step TD update.

        Q_net = self.apply_kernels(Q, choices, second_steps, params_T)

        if get_DVs: return self.get_DVs(session, params_T, Q_net, Q)
        else:       return session_log_likelihood(choices, Q_net, iTemp)