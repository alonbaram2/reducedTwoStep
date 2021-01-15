import numpy as np
from random import shuffle, choice
import pylab as plt
import os
import time
import sys
from scipy.stats import ttest_ind, ttest_rel, sem
from collections import OrderedDict
from pprint import pprint
from . import utility as ut
from . import model_plotting as mp
from . import plotting as pl
from . import model_fitting as mf
from . import parallel_processing as pp


def group_info(sessions):
    return {'n_subjects'  : len(set([s.subject_ID for s in sessions])),
            'n_sessions' : len(sessions),
            'n_blocks'   : sum([len(s.blocks['start_trials']) - 1 for s in sessions]),
            'n_trials'   : sum([s.n_trials for s in sessions])}


def run_tests(sessions_A, sessions_B, RL_agent, LR_agent, perm_type,
              n_perms=5000, test_time=None, file_name=None):
    ''' Run a suite of different comparisons on two groups of sessions.'''

    test_data = {
        'group_A_info'     : group_info(sessions_A),
        'group_B_info'     : group_info(sessions_B),
        'perm_type'        : perm_type,
        'n_perms'          : n_perms,
        'trial_rate'       : trial_rate_test(sessions_A, sessions_B, perm_type, 
                                        test_time, n_perms),
        'reversal_analysis': reversal_test(sessions_A, sessions_B, perm_type, n_perms)}
    
    output_test_data(test_data, file_name)

    test_data['LR_fit'] = model_fit_test(sessions_A, sessions_B, LR_agent, perm_type,
                                         n_perms, fig_no=False)

    output_test_data(test_data, file_name)

    test_data['RL_fit'] = model_fit_test(sessions_A, sessions_B, RL_agent, perm_type,
                                         n_perms, fig_no=False)

    output_test_data(test_data, file_name)
    return test_data


def output_test_data(test_data, file_name=None):
    if file_name:
        ut.save_item(test_data, file_name)
        file = open(file_name+'.txt', 'w+')
    else:
        file = sys.stdout
    if 'group_A_info' in test_data.keys():
        print('\nGroup A info:', file=file)
        print(test_data['group_A_info'], file=file)
        print('\nGroup B info:', file=file)
        print(test_data['group_B_info'], file=file)
    if 'trial_rate' in test_data.keys():
        print('\nP value for number of trials in first {} minutes: {}'
              .format(test_data['trial_rate']['test_time'],
                      test_data['trial_rate']['p_val']), file=file)
    if 'reversal_analysis' in test_data.keys():    
        print('\nReversal analysis P values: P_0: {}, tau: {}'
              .format(test_data['reversal_analysis']['block_end_P_value'],
                      test_data['reversal_analysis']['tau_P_value']), file=file)
    if 'LR_fit' in test_data.keys():   
        print('\nLogistic regression fit:', file=file)
        _print_P_values(test_data['LR_fit']['P_values'], 
                        n_perms=test_data['LR_fit']['n_perms'], file=file)
    if 'RL_fit' in test_data.keys():     
        print('\nRL fit:', file=file)
        _print_P_values(test_data['RL_fit']['P_values'],
                        n_perms=test_data['RL_fit']['n_perms'], file=file)
    if file_name: file.close()


def plots(sessions_A, sessions_B, RL_agent, LR_agent=None, title=None,
          test_time=None, test_data=None, per_sub=False, fig_no=0):
    if test_data:
        RL_fit_A = test_data['RL_fit']['true_fit']['fit_A']
        RL_fit_B = test_data['RL_fit']['true_fit']['fit_B']
        LR_fit_A = test_data['LR_fit']['true_fit']['fit_A']
        LR_fit_B = test_data['LR_fit']['true_fit']['fit_B']
    else:
        RL_fit_A = mf.fit_population(sessions_A, RL_agent)
        RL_fit_B = mf.fit_population(sessions_B, RL_agent)
        LR_fit_A = mf.fit_population(sessions_A, LR_agent)
        LR_fit_B = mf.fit_population(sessions_B, LR_agent)

    trial_rate_comparison(sessions_A, sessions_B, test_time, fig_no+1, title, plot_cum=True)
    reversal_comparison(sessions_A, sessions_B,  fig_no+2, title, by_type=True)
    model_fit_comp_plot(LR_fit_A, LR_fit_B, fig_no+3, title)
    model_fit_comp_plot(RL_fit_A, RL_fit_B, fig_no+4, title)
    abs_preference_comparison(sessions_A, sessions_B, RL_fit_A, RL_fit_B, RL_agent,
                              fig_no+5, title)
    if per_sub:
        per_subject_fit_comparison(sessions_A, sessions_B, RL_agent, fig_no+6, title)
        per_subject_fit_comparison(sessions_A, sessions_B, LR_agent, fig_no+7, title)

# -------------------------------------------------------------------------------------
# Group comparison plots.
# -------------------------------------------------------------------------------------

def model_fit_comparison(sessions_A, sessions_B, agent, fig_no=1, title=None, ebars='pm95'):
    ''' Fit the two groups of sessions with the specified agent and plot the results on the same axis.'''
    eval_BIC = ebars == 'pm95'
    fit_A = mf.fit_population(sessions_A, agent, eval_BIC=eval_BIC)
    fit_B = mf.fit_population(sessions_B, agent, eval_BIC=eval_BIC)
    model_fit_comp_plot(fit_A, fit_B, fig_no=fig_no, sub_medians=True, ebars=ebars)
    if title:plt.suptitle(title)
    if agent.type == 'RL':  
        abs_preference_comparison(sessions_A, sessions_B, fit_A, fit_B, agent, fig_no+1000, title)


def model_fit_comp_plot(fit_1, fit_2, fig_no=1, title=None, clf=True, sub_medians=True, ebars='SD'):
    'Compare two different model fits.'
    mp.model_fit_plot(fit_1, fig_no, col='b', clf=clf , x_offset=-0.11, sub_medians=sub_medians, ebars=ebars)
    mp.model_fit_plot(fit_2, fig_no, col='r', clf=False, x_offset= 0.11, title=title, 
                      sub_medians=sub_medians, ebars=ebars)


def fit_evolution_comparison(fit_1, fit_2, fig_no=1, title=None):
    'Compare two different model fits.'
    mp.parameter_evolution_plot(fit_1, fig_no=fig_no, clf=True)
    mp.parameter_evolution_plot(fit_2, fig_no=fig_no, clf=False)


def per_subject_fit_comparison(sessions_A, sessions_B, agent, fig_no=1, title=None):
    'Perform per subject fits on the two groups and plot subject mean parameter values.'
    sub_fits_A = mf.per_subject_fit(sessions_A, agent)
    sub_fits_B = mf.per_subject_fit(sessions_B, agent)
    mp.per_subject_fit_plot(sub_fits_A, fig_no, col='b', ebars=False, x_offset=-0.11, title=title)
    mp.per_subject_fit_plot(sub_fits_B, fig_no, col='r', ebars=False, x_offset= 0.11, clf=False)
    sub_A_means_U = np.vstack([sub_fit['pop_dists']['means'] for sub_fit in sub_fits_A])
    sub_B_means_U = np.vstack([sub_fit['pop_dists']['means'] for sub_fit in sub_fits_B])
    if [sf['sID'] for sf in sub_fits_A] == [sf['sID'] for sf in sub_fits_B]:
        p_vals = np.round(ttest_rel(sub_A_means_U, sub_B_means_U).pvalue,3)
        print('Paired t-test P values:')
    else:
        p_vals = np.round(ttest_ind(sub_A_means_U, sub_B_means_U).pvalue,3)
        print('Independent t-test P values:')
    for param_name, p_val in zip(sub_fits_A[0]['param_names'], p_vals):
        print('{}: {:.3f}'.format(param_name, p_val))


def trial_rate_comparison(sessions_A, sessions_B, test_time=None, fig_no=1,
                          title=None, ebars='SEM', plot_cum=True):
    '''Plot trials per minute for each group, and dashed vertical line at test time if specified.'''
    pl.trials_per_minute(sessions_A, col='b', fig_no=fig_no, ebars=ebars, plot_cum=plot_cum)
    pl.trials_per_minute(sessions_B, col='r', fig_no=fig_no, clf=False, ebars=ebars, plot_cum=plot_cum)
    if test_time:
        plt.plot([test_time,test_time], plt.ylim(),':k')
    if title:
        plt.title(title)


def reversal_comparison(sessions_A, sessions_B,  fig_no=1, title=None, by_type=False):
    '''Plot choice trajectories around reversals for both groups.'''
    pl.reversal_analysis(sessions_A, cols=0, fig_no=fig_no, by_type=by_type)
    pl.reversal_analysis(sessions_B, cols=1, fig_no=fig_no, by_type=by_type, clf=False)
    if title: plt.title(title)


def p_correct_comparison(sessions_A, sessions_B, fig_no=1, title=None, last_n=15, verbose=False):
    ''' Compare fraction of correct choices at end on non neutral blocks.  Plot shows 
    data point for each animal and population mean and SEM. last_n specifies how many
     trials at the end of each block to include, set to 'all' to include all trials in 
     non-neutral blocks.
    '''
    if verbose and title:print(title)
    p_corrects_A = pl.per_animal_end_of_block_p_correct(sessions_A, col='b', fig_no=fig_no,
                                                        last_n=last_n, verbose=verbose)
    p_corrects_B = pl.per_animal_end_of_block_p_correct(sessions_B, col='r', fig_no=fig_no,
                                            last_n=last_n, verbose=verbose, clf=False)
    plt.xlim(-0.5,0.4)
    if title: plt.title(title)
    if set([s.subject_ID for s in sessions_A]) == set([s.subject_ID for s in sessions_B]):
        print('Paired t-test P value: {}'.format(ttest_rel(p_corrects_A, p_corrects_B)[1]))
    else:
        print('Independent t-test P value: {}'.format(ttest_ind(p_corrects_A, p_corrects_B)[1]))
 

def abs_preference_comparison(sessions_A, sessions_B, population_fit_A, population_fit_B, agent,
                               fig_no=1, title=None):
    ''' Plot mean absolute preference of model based and model free system based on population fits.'''
    mean_preference_mb_A, mean_preference_mf_A = mp.abs_preference_plot(sessions_A, population_fit_A, agent, to_plot=False)
    mean_preference_mb_B, mean_preference_mf_B = mp.abs_preference_plot(sessions_B, population_fit_B, agent, to_plot=False)
    plt.figure(fig_no, figsize=[2.5,2.3]).clf()
    plt.bar([1  , 3],[mean_preference_mb_A, mean_preference_mf_A])
    plt.bar([1.8,3.8],[mean_preference_mb_B, mean_preference_mf_B],color='r')
    plt.xticks([1.8, 3.8], ['Model based', 'Model free'])
    plt.xlim(0.8,4.8)
    plt.ylabel('Mean abs. preference')
    plt.locator_params(nbins=5, axis='y')
    if title:plt.title(title)

def reaction_times_second_step_comparison(sessions_A, sessions_B, fig_no=1, title=None):
    '''Permuation test for significance level of differences in seccond step reaction 
    times between groups.  Reports P values for common and rare transition RTs and for
    the difference between common and rare reaction times.''' 

    RT_common_A, RT_rare_A = pl.reaction_times_second_step(sessions_A, fig_no=False)
    RT_common_B, RT_rare_B = pl.reaction_times_second_step(sessions_B, fig_no=False)
    plt.figure(fig_no, figsize=[2.5,2.3]).clf()
    plt.bar([1  , 3], [RT_common_A, RT_rare_A])
    plt.bar([1.8,3.8],[RT_common_B, RT_rare_B],color='r')
    plt.xticks([1.8, 3.8], ['Common', 'Rare'])
    plt.xlim(0.8,4.8)
    plt.ylabel('Reaction time')
    plt.ylim(min(RT_common_A,RT_common_B) * 0.8, max(RT_rare_A, RT_rare_B) * 1.1)
    plt.locator_params(nbins=5, axis='y')
    plt.tight_layout()
    if title:plt.title(title)


# -------------------------------------------------------------------------------------
# Permutation tests.
# -------------------------------------------------------------------------------------

def model_fit_test(sessions_A, sessions_B, agent,  perm_type, n_perms=1000,
                   n_true_fit=5, fig_no=1, title=None, file_name=None):

    '''Permutation test for significant differences in model fits between two groups of 
    sessions.  Outline of procedure:
    1. Perform model fitting seperately on both groups of sessions.
    2. Evaluate distance metric (KL divergence or difference of means) between fits
    for each parameter.
    3. Generate ensemble of resampled datasets in which sessions are randomly allocated
    to A or B.
    4. Perform model fitting and evalute distance metric for each resampled dataset to
    get a distribution of the distance metric under the null hypothesis that there is
    no difference between groups.
    5. Compare the true distance metric for each parameter with the null distribution
    to get a P value.'''

    mf._precalculate_fits(sessions_A + sessions_B, agent) # Store first round fits on sessions.

    test_var_names = agent.param_names[:] # Names of variables being permutation tested.
    if agent.type == 'RL': 
        test_var_names += ['Model-based influence','Model-free influence' ]

    print('Fitting original dataset.')
    fit_test_data = {'test_var_names':test_var_names,
                     'true_fits' : pp.map(_fit_dataset,
                                      [(sessions_A, sessions_B, agent)]*n_true_fit)}

    perm_datasets = [_permuted_dataset(sessions_A, sessions_B, perm_type) + [agent] 
                     for _ in range(n_perms)]

    fit_test_data['perm_fits'] = []

    for i, perm_fit in enumerate(pp.imap(_fit_dataset, perm_datasets, ordered=False)):
        fit_test_data['perm_fits'].append(perm_fit)
        print('Fitted permuted dataset {} of {}'.format(i+1, n_perms))
        if i > 0 and i%10 == 9: _model_fit_P_values(fit_test_data, file_name)

    for session in sessions_A + sessions_B: del(session.fit) # Clear precalcuated fits.

    if fig_no: _model_fit_test_plot(fit_test_data, fig_no=fig_no, title=title)
    
    return fit_test_data


def _fit_dataset(fit_data):
    # Evaluate and store fits for one dataset consisting of two sets of sessions,
    # along with distances between each parameter value.
    sessions_A, sessions_B, agent = fit_data   
    session_fits_A = [session.fit for session in sessions_A]
    session_fits_B = [session.fit for session in sessions_B] 
    fit_A = mf.fit_population(sessions_A, agent, session_fits=session_fits_A, verbose=False)
    fit_B = mf.fit_population(sessions_B, agent, session_fits=session_fits_B, verbose=False)
    differences = fit_A['pop_dists']['means']-fit_B['pop_dists']['means']
    if agent.type == 'RL':
        prefs_A = mp.abs_preference_plot(sessions_A, fit_A, agent, to_plot=False)
        prefs_B = mp.abs_preference_plot(sessions_B, fit_B, agent, to_plot=False)
        differences = np.hstack([differences, prefs_A-prefs_B])
    distances = np.abs(differences)
    return {'fit_A': fit_A,
            'fit_B': fit_B,
            'differences': differences,
            'distances': distances}


def _model_fit_P_values(fit_test_data, file_name=None):
    '''Evaluate P values from distances between true and permuted datasets'''
    true_distances = np.median([f['distances'] for f in fit_test_data['true_fits']], axis=0)
    perm_distances = np.array([f['distances'] for f in fit_test_data['perm_fits']])
    true_differences = np.median([f['differences'] for f in fit_test_data['true_fits']], axis=0)
    perm_differences = np.array([f['differences'] for f in fit_test_data['perm_fits']])
    P_values   = np.mean(perm_distances > true_distances, 0)
    diff_ranks = np.mean(perm_differences > true_differences, 0)
    n_perms = len(fit_test_data['perm_fits'])
    P_value_dict = OrderedDict([(pn,pv) for pn, pv in
                                zip(fit_test_data['test_var_names'], P_values)])
    diff_rank_dict = OrderedDict([(pn,dr) for pn, dr in
                                zip(fit_test_data['test_var_names'], diff_ranks)])
    fit_test_data.update({'true_distances'  : true_distances,
                          'perm_distances'  : perm_distances,
                          'true_differences': true_differences,
                          'perm_differences': perm_differences,
                          'P_values'        : P_value_dict,
                          'diff_ranks'      : diff_rank_dict,
                          'n_perms'         : n_perms})
    if file_name: ut.save_item(fit_test_data, file_name)
    _print_P_values(fit_test_data['P_values'], n_perms, file_name)
    _print_P_values(fit_test_data['diff_ranks'], None, file_name,
                                  dict_name='Diff ranks', append=True) 


def _print_P_values(P_value_dict, n_perms=None, file_name=None, file=None,
                    dict_name='P values', append=False):
    if file_name or file: 
        _print_P_values(P_value_dict, n_perms) # Print to standard out then print to file.
    file = open(file_name + '.txt', 'a' if append else 'w') if file_name else file
    print(dict_name + (' ({} permutations):'.format(n_perms) if n_perms else ':'), file=file)
    name_len = max([len(name) for name in P_value_dict.keys()])
    for pn, pv in P_value_dict.items():
        print('   ' + pn.ljust(name_len) + ': {:.3f}'.format(pv), file=file)
    if file_name: file.close() 


def _model_fit_test_plot(fit_test_data, fig_no=1, title=None):
    plt.figure(fig_no).clf()
    mp.model_fit_plot(fit_test_data['true_fits'][0]['fit_A'], fig_no, col='b',
                      clf=False, x_offset=-0.11, half_height=True)
    mp.model_fit_plot(fit_test_data['true_fits'][0]['fit_B'], fig_no, col='r',
                      clf=False, x_offset= 0.11, half_height=True)
    plt.subplot(2,1,2)
    param_names = fit_test_data['true_fits'][0]['fit_A']['param_names']
    n_params = len(param_names)
    perm_distances = fit_test_data['perm_distances'][:,:n_params]
    true_distances = fit_test_data['true_distances'][:n_params]
    perm_distances.sort(axis=0)
    x = np.arange(n_params)
    z = np.zeros(n_params)
    plt.errorbar(x+0.4, z, (z, perm_distances[-1,:]),
                  linestyle='', linewidth=2, color='k')
    plt.errorbar(x+0.6, z, (z, perm_distances[round(fit_test_data['n_perms']*0.95)-1,:]),
                  linestyle='', linewidth=2, color='b')
    plt.plot(x+0.5, true_distances, linestyle='', color='r', marker='.') 
    plt.xticks(x+0.5, param_names, rotation=-45, ha='left')
    plt.ylabel('Distance')
    text_ypos = plt.ylim()[1]*1.1
    plt.ylim(0,text_ypos*1.2)
    for i, p_val in enumerate(list(fit_test_data['P_values'].values())[:n_params]):
        plt.text(i+0.4, text_ypos, str(round(p_val,3)))
    if title: plt.suptitle(title)
    
# -------------------------------------------------------------------------------------

def session_fit_test(sessions_A, sessions_B, agent,  perm_type,
                     n_perms=1000, use_median=False):
    ''' A test for differences in model fits between two groups of subjects which fits
    a single population distribution to both sets of sessions combined and then looks 
    for differences in the distribution of MAP fits between the two groups.'''

    all_sessions = sessions_A + sessions_B

    all_sessions_fit =  mf.fit_population(all_sessions, agent)

    for i, session_fit in enumerate(all_sessions_fit['session_fits']):
        all_sessions[i].session_fit = session_fit

    true_session_fits_A = np.array([s.session_fit['params_T'] for s in sessions_A])
    true_session_fits_B = np.array([s.session_fit['params_T'] for s in sessions_B])

    ave_func = np.median if use_median else np.mean

    true_fit_dists = np.abs(ave_func(true_session_fits_A, 0) - ave_func(true_session_fits_B, 0))

    shuffled_fit_dists = np.zeros([n_perms, agent.n_params])

    for i in range(n_perms):
        print('Evaluating permuted sessions, round: {} of {}'.format(i+1, n_perms))
        perm_ses_A, perm_ses_B = _permuted_dataset(sessions_A, sessions_B, perm_type)
        shuffled_session_fits_A = np.array([s.session_fit['params_T'] for s in perm_ses_A])
        shuffled_session_fits_B = np.array([s.session_fit['params_T'] for s in perm_ses_B])
        shuffled_fit_dists[i,:] = np.abs(ave_func(shuffled_session_fits_A, 0) -
                                         ave_func(shuffled_session_fits_B, 0))

        p_vals = np.mean(shuffled_fit_dists>=true_fit_dists, axis=0)

    return p_vals


def reversal_test(sessions_A, sessions_B, perm_type, n_perms=1000, by_type=False):
    ''' Permutation test for differences in the fraction correct at end of blocks and
    the time constant of adaptation to block transitions.'''
    fit_A = pl.reversal_analysis(sessions_A, return_fits=True, by_type=by_type)
    fit_B = pl.reversal_analysis(sessions_B, return_fits=True, by_type=by_type)

    def _rev_fit_dist(fit_A, fit_B):
        '''Evaluate absolute difference in asymtotic choice probability and reversal 
        time constants for pair of fits to reversal choice trajectories.'''
        if fit_A['rew_rev']:  # Fit includes seperate analyses by reversal type.
            return np.abs([fit_A['p_1']              - fit_B['p_1'],
                           fit_A['both_rev']['tau']  - fit_B['both_rev']['tau'],
                           fit_A['rew_rev']['tau']   - fit_B['rew_rev']['tau'],
                           fit_A['trans_rev']['tau'] - fit_B['trans_rev']['tau']])
        else:
            return np.abs([fit_A['p_1']              - fit_B['p_1'],
                           fit_A['both_rev']['tau']  - fit_B['both_rev']['tau'], 0., 0.])

    true_rev_fit_dist = _rev_fit_dist(fit_A,fit_B)
    permuted_rev_fit_dist = np.zeros([n_perms, 4])
    print('Reversal analysis permutation test:')
    for i in range(n_perms):
        if i > 0 and i%10 == 9:
            print('Fitting permuted sessions, round: {} of {}'.format(i+1, n_perms))

        perm_ses_A, perm_ses_B = _permuted_dataset(sessions_A, sessions_B, perm_type)
        shuffled_fit_A = pl.reversal_analysis(perm_ses_A, return_fits=True, by_type=by_type)
        shuffled_fit_B = pl.reversal_analysis(perm_ses_B, return_fits=True, by_type=by_type)
        permuted_rev_fit_dist[i,:] = _rev_fit_dist(shuffled_fit_A, shuffled_fit_B)

    p_vals = np.mean(permuted_rev_fit_dist>=true_rev_fit_dist, axis=0) 

    print('Block end choice probability P value   : {}'.format(p_vals[0]))
    print('All reversals tau P value              : {}'.format(p_vals[1]))
    if by_type:
        print('Reward probability reversal tau P value: {}'.format(p_vals[2]))
        print('Trans. probability reversal tau P value: {}'.format(p_vals[3]))
    return {'block_end_P_value': p_vals[0], 'tau_P_value' : p_vals[1]}


def block_length_test(sessions_A, sessions_B, perm_type, n_perms=1000):
    '''Test whether the length of non-neutral blocks is significantly different
    between two groups of sessions.'''
    for session in sessions_A + sessions_B:
        start_trials = session.blocks['start_trials']
        block_lengths = np.array(start_trials[1:]) - np.array(start_trials[:-1])
        not_neutral = session.blocks['reward_states'][:-1] != 1
        session.block_lengths = block_lengths[not_neutral]

    def block_length_dist(sessions_A, sessions_B):
        mean_BL_A = np.mean(np.hstack([session.block_lengths for session in sessions_A]))
        mean_BL_B = np.mean(np.hstack([session.block_lengths for session in sessions_B]))
        return np.abs(mean_BL_A - mean_BL_B)

    true_dist = block_length_dist(sessions_A, sessions_B)
    perm_dists = np.zeros(n_perms)
    for i in range(n_perms):
        perm_ses_A, perm_ses_B = _permuted_dataset(sessions_A, sessions_B, perm_type)
        perm_dists[i] = block_length_dist(perm_ses_A, perm_ses_B)

    print('Block length difference P value: {:.4f}'.format(np.mean(perm_dists>=true_dist))) 
    for session in sessions_A + sessions_B:
        del(session.block_lengths)


def trial_rate_test(sessions_A, sessions_B, perm_type, test_time=None, n_perms=1000): 
    ''' Evaluate whether number of trials per session in first test_time minutes is 
    different between groups.'''

    if not test_time: # Time at which to test for different in number of trials.
        test_time = np.ceil(max([s.duration for s in sessions_A + sessions_B])/60)

    for session in sessions_A + sessions_B:
        session.n_trials_test = sum(session.times['trial_start'] < (60 * test_time))

    true_n_trials_diff = np.abs(sum([s.n_trials_test for s in sessions_A]) - \
                                sum([s.n_trials_test for s in sessions_B]))
    perm_n_trials_diff = np.zeros(n_perms)
    for i in range(n_perms):
        perm_ses_A, perm_ses_B = _permuted_dataset(sessions_A, sessions_B, perm_type)
        perm_n_trials_diff[i] = np.abs(sum([s.n_trials_test for s in perm_ses_A]) - \
                                       sum([s.n_trials_test for s in perm_ses_B]))
    p_val = np.mean(perm_n_trials_diff>=true_n_trials_diff)
    print('Trial number difference P value: {}'.format(p_val))
    return {'test_time': test_time,
            'p_val'    : p_val} 


def reaction_times_second_step_test(sessions_A, sessions_B, perm_type, n_perms=1000):
    '''Permuation test for significance level of differences in seccond step reaction 
    times between groups.  Reports P values for common and rare transition RTs and for
    the difference between common and rare reaction times.''' 

    def RT_distances(sessions_A, sessions_B):
        RT_common_A, RT_rare_A = pl.reaction_times_second_step(sessions_A, fig_no=False)
        RT_common_B, RT_rare_B = pl.reaction_times_second_step(sessions_B, fig_no=False)
        CR_diff_A = RT_rare_A-RT_common_A
        CR_diff_B = RT_rare_B-RT_common_B
        return np.abs([RT_common_A-RT_common_B, RT_rare_A-RT_rare_B, CR_diff_A-CR_diff_B])

    true_distances = RT_distances(sessions_A, sessions_B)
    perm_distances = np.zeros([n_perms, 3])
    for i in range(n_perms):
        if i > 0 and i%100 == 99:print('Permuatation: {} of {}'.format(i+1, n_perms))
        perm_ses_A, perm_ses_B = _permuted_dataset(sessions_A, sessions_B, perm_type)
        perm_distances[i,:] = RT_distances(perm_ses_A, perm_ses_B)

    P_values = np.mean(perm_distances > true_distances, 0)

    print('Common RT P value:{}'.format(P_values[0]))
    print('Rare   RT P value:{}'.format(P_values[1]))
    print('Diff.  RT P value:{}'.format(P_values[2]))

#---------------------------------------------------------------------------------------------------
#  Permuted dataset generation.
#---------------------------------------------------------------------------------------------------

def _permuted_dataset(sessions_A, sessions_B, perm_type='ignore_subject'):
    ''' Generate permuted datasets by randomising assignment of sessions between groups A and B.
    perm_type argument controls how permutations are implemented:
    'within_subject' - Permute sessions within subject such that each permuted group has the same
                     number of session from each subject as the true datasets.
    'cross_subject' - All sessions from a given subject are assigned to one or other of the permuted datasets.
    'ignore_subject' - The identity of the subject who generated each session is ignored in the permutation.
    'within_group' - Permute subjects within groups that are subsets of all subjects.  
                     Animal assignment to groups is specified by groups argument which should be 
                     a list of lists of animals in each grouplt.
    '''
    assert perm_type in ('within_subject', 'cross_subject', 'ignore_subject',
                         'within_sub_&_cyc'), 'Invalid permutation type.'
    all_sessions = sessions_A + sessions_B
    all_subjects = list(set([s.subject_ID for s in all_sessions]))

    if perm_type == 'ignore_subject':  # Shuffle sessions ignoring which subject each session is from.        
        shuffle(all_sessions)
        perm_ses_A = all_sessions[:len(sessions_A)]
        perm_ses_B = all_sessions[len(sessions_A):]

    elif perm_type == 'cross_subject':  # Permute subjects across groups (used for cross subject tests.)
        n_subj_A     = len(set([s.subject_ID for s in sessions_A]))        
        shuffle(all_subjects)   
        perm_ses_A = [s for s in all_sessions if s.subject_ID in all_subjects[:n_subj_A]]
        perm_ses_B = [s for s in all_sessions if s.subject_ID in all_subjects[n_subj_A:]]
    
    elif perm_type == 'within_subject': # Permute sessions keeping number from each subject in each group constant.
        perm_ses_A = []
        perm_ses_B = []
        for subject in all_subjects:
            subject_sessions_A = [s for s in sessions_A if s.subject_ID == subject]
            subject_sessions_B = [s for s in sessions_B if s.subject_ID == subject]
            all_subject_sessions = subject_sessions_A + subject_sessions_B
            shuffle(all_subject_sessions)
            perm_ses_A += all_subject_sessions[:len(subject_sessions_A)]
            perm_ses_B += all_subject_sessions[len(subject_sessions_A):]
    
    elif perm_type == 'within_sub_&_cyc': # Permute sessions keeping number from each subject and cycle in each group constant.
        perm_ses_A = []
        perm_ses_B = []
        all_cycles = list(set([s.cycle for s in all_sessions]))
        for subject in all_subjects:
            for cycle in all_cycles:
                sub_cyc_sessions_A = [s for s in sessions_A if 
                    s.subject_ID == subject and s.cycle == cycle]
                sub_cyc_sessions_B = [s for s in sessions_B if 
                    s.subject_ID == subject and s.cycle == cycle]
                all_sub_cyc_sessions = sub_cyc_sessions_A + sub_cyc_sessions_B
                shuffle(all_sub_cyc_sessions)
                perm_ses_A += all_sub_cyc_sessions[:len(sub_cyc_sessions_A)]
                perm_ses_B += all_sub_cyc_sessions[len(sub_cyc_sessions_A):]

    return [perm_ses_A, perm_ses_B]
