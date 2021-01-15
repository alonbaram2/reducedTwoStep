from ._RL_agent import *

class KO(RL_agent):
    'Kernels only agent.'

    def __init__(self, kernels = ['bs', 'ck', 'rb']):
        self.name = 'KO'
        self.param_names  = []
        self.param_ranges = []
        RL_agent.__init__(self, kernels)

    @jit
    def session_likelihood(self, session, params_T, get_DVs = False):

        # Unpack trial events.
        choices, second_steps, outcomes = session.unpack_trial_data('CSO')

        Q_net = self.apply_kernels(np.zeros([2,session.n_trials]), choices, second_steps, params_T)

        if get_DVs: return self.get_DVs(session, params_T, Q_net, Q)
        else:       return session_log_likelihood(choices, Q_net)

