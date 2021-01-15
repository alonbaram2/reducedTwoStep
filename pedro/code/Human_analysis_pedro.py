from Two_step import *
import matplotlib.pyplot as plt
my_exp = he.experiment('fmri')

my_sessions = my_exp.get_sessions('all','all')

my_exp.plot_subject(xx)
import matplotlib.pyplot as plt

#Reversal analysis:
pl.reversal_analysis(my_sessions)
pl.reversal_analysis(my_sessions, by_type=True)
plt.suptitle('Analysis of Reversals')

#Stay probability plots:
pl.stay_probability_analysis(my_sessions)
pl.stay_probability_analysis(my_sessions),plt.ylim(0.5,1)

#Logistic regression:
lr.logistic_regression(my_sessions)
lr.logistic_regression(my_sessions, per_sub = True)

plt.ylim(-5,10)

# Number of rewards:
def total_rewards(sessions):
    '''Return a dictionary of subject IDs and corresponding number of rewards in sessions'''
    subject_IDs = set([s.subject_ID for s in sessions])
    return {sID: sum([session.rewards for session in sessions if session.subject_ID == sID])
            for sID in subject_IDs}

#INDIVIDUAL PARAMETERS:
lr_model = lr.config_log_reg()
population_fit = mf.fit_population(my_sessions, lr_model) 

session_fits = [p['params_U'] for p in population_fit['session_fits']] 

[fit[lr_model.param_names.index('trCR_x_out')] for fit in session_fits]

[s.subject_ID for s in my_sessions]

[s.number for s in my_sessions]

#PERMUTATION TESTS:
pp.enable_multiprocessing()

exp_1 = he.experiment('name_of_exp_1')
exp_2 = he.experiment('name_of_exp_2')
sessions_1 = exp_1.get_sessions('all','all')
sessions_2 = exp_2.get_sessions('all','all')
agent = lr.config_log_reg()
gc.model_fit_test(sessions_1, sessions_2, agent, 'cross_subject')
gc.model_fit_test(sessions_1, sessions_2, agent, 'within_subject')

gc.model_fit_test(sessions_1, sessions_2, agent, 'within_subject', n_perms = 5000)
