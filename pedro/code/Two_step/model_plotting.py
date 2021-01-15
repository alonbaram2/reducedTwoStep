
import pylab as plt
import numpy as np
import scipy as sp
from collections import OrderedDict
from random import shuffle
import matplotlib.cm as cm
from . import plotting as pl
from . import utility as ut
from . import model_fitting as mf

#--------------------------------------------------------------------------------------
# Plotting model fits.
#--------------------------------------------------------------------------------------

def model_fit_plot(population_fit, fig_no=1, clf=True, col='b', x_offset=0., scatter=True,
                   ebars='SD', title=None, half_height=False, sub_medians=False):
    ''' Plot the results of a population fit, for logistic regression fits all predictor
    loadings are plotted on single axis.  For RL fits where parameters are transformed
    to enforce constraints, seperate axes are used for parameters with different ranges.
    '''

    plt.figure(fig_no, figsize=[7,2.3])
    if clf:plt.clf()
    title = population_fit['agent_name'] if title is None else title
    plt.suptitle(title)

    def _plot(y, yerr, MAP_params, rng, param_names):
        n_ses, n_params = MAP_params.shape
        if ebars:
            plt.errorbar(np.arange(n_params)+0.5+x_offset, y, yerr, linestyle='',
                     linewidth=2, color=col, marker='_', markersize=10)
        else:
            plt.plot(np.arange(n_params)+0.5+x_offset, y, linestyle='', marker='.',
                     markersize=6, color=col)
        if scatter:
            for i in range(n_params):
                plt.scatter(i+0.4+x_offset+0.2*np.random.rand(n_ses), MAP_params[:,i],
                            s=4,  facecolor=col, edgecolors='none', lw=0)
        if rng == 'unc':
            plt.plot(plt.xlim(),[0,0],'k')
        elif rng == 'unit': 
            plt.ylim(0,1)
        elif rng == 'pos':
            plt.ylim(0, max(plt.ylim()[1], np.max(MAP_params) * 1.2))

        plt.locator_params('y', nbins=5)
        plt.xlim(0,n_params)
        plt.xticks(np.arange(n_params)+0.5, param_names, rotation=-45, ha='left')

    pop_dists = population_fit['pop_dists']

    if sub_medians: # Plot subject median MAP fits rather than all session MAP fits.
        MAP_params = _sub_median_MAP(population_fit)
    else:
        MAP_params = np.array([sf['params_T'] for sf in population_fit['session_fits']])

    if ebars == 'SD': # Use population level distribution SDs for errorbars.
        ebars_U = pop_dists['SDs']
    elif ebars == 'pm95': # Use 95% confidence intervals of population distribution means.
        ebars_U = 1.96*np.sqrt(-1/population_fit['iBIC']['means_hessian'])
    else:
        ebars_U = np.zeros(len(pop_dists['SDs'])) 

    if (population_fit['param_ranges'][0] == 'all_unc' or
        all([pr == 'unc' for pr in population_fit['param_ranges']])): # Logistic regression fit.
        if half_height: plt.subplot(2,1,1)
        _plot(pop_dists['means'], ebars_U, MAP_params, 'unc', population_fit['param_names'])
        plt.ylabel('Log odds')

    else: # Reinforcement learning model fit.
        param_ranges = population_fit['param_ranges']

        #Transform parameters into constrained space.
        means_T = mf._trans_UT(pop_dists['means'], param_ranges)
        upp_1SD = mf._trans_UT(pop_dists['means'] + ebars_U, param_ranges)
        low_1SD = mf._trans_UT(pop_dists['means'] - ebars_U, param_ranges)
        yerr_T = np.vstack([means_T - low_1SD, upp_1SD - means_T])

        axes, ax_left = ([], 0.125)
        ax_bottom, ax_height = (0.53, 0.4) if half_height else (0.1, 0.8)
        for rng in ['unit', 'pos', 'unc']:
            rng_mask = np.array([r == rng for r in param_ranges])
            param_names = [p_name for p_name, rm in zip(population_fit['param_names'], rng_mask) if rm]
            subplot_MAP_params = MAP_params[:,rng_mask]   
            ax_width = np.mean(rng_mask) * 0.655  
            if rng == 'unc': ax_left += 0.02
            axes.append(plt.axes([ax_left, ax_bottom, ax_width, ax_height]))
            _plot(means_T[rng_mask], yerr_T[:,rng_mask], subplot_MAP_params, rng, param_names)
            ax_left = ax_left + ax_width + 0.05

        axes[0].set_ylabel('Parameter value')

def _sub_median_MAP(population_fit):
    'Return array of median MAP session fits per subject.'
    subject_IDs = set([sf['sID'] for sf in population_fit['session_fits']])
    subject_medians = np.zeros([len(subject_IDs), len(population_fit['param_names'])])
    for i,subject_ID in enumerate(subject_IDs):
        sub_session_fits = [sf for sf in population_fit['session_fits'] if sf['sID'] == subject_ID]
        subject_medians[i,:] = np.median([sf['params_T'] for sf in sub_session_fits],0)
    return subject_medians


def lagged_fit_plot(fit, fig_no=1):
    'Model fit plot for logistic regression model with lagged predictors.'
    plt.figure(fig_no).clf()
    param_means = fit['pop_dists']['means']
    param_SDs  = fit['pop_dists']['SDs']
    param_lags = np.array([int(pn.split('-')[1]) if len(pn.split('-')) > 1 else -1
                           for pn in fit['param_names']])
    param_base = [pn.split('-')[0] for pn in fit['param_names']]
    base_params = sorted(list(set(param_base)), key = param_base.index)
    color_idx = np.linspace(0, 1, len(base_params))
    for i, base_param in zip(color_idx, base_params):
        p_mask = np.array([pb == base_param for pb in param_base])
        if sum(p_mask) > 1:
            plt.errorbar(-param_lags[p_mask], param_means[p_mask], param_SDs[p_mask],
                         label = base_param, color = plt.cm.jet(i))
    plt.plot([-max(param_lags) - 0.5, -0.5], [0,0], 'k')
    plt.xlim([-max(param_lags) - 0.5, -0.5])
    plt.xlabel('Lag (trials)')
    plt.legend(bbox_to_anchor=(0, 1), loc=2, borderaxespad=0., fontsize = 'small')


def per_subject_fit_plot(fit, fig_no = 1, clf = True, col = None, ebars = False,
                      x_offset = 0, sort_param = None, title = None):
    '''Plot model fitting results by subject.  fit argument can be either a single
    population_fit containing many subjects, or a set of list of population fits each
    containing data for a single subject.
    '''
    if type(fit) == list: 
        # fit is list of seperate subjects population_fits.
        subject_fits = fit 
        if sort_param:  # Order subjects by value of specified param.            
            sp_index  = subject_fits[0]['param_names'].index(sort_param)
            sub_order = np.argsort([sf['pop_dists']['means'][sp_index]
                                    for sf in subject_fits])
            subject_fits = [subject_fits[i] for i in sub_order]
        n_params = len(subject_fits[0]['pop_dists']['means'])
        sub_x  = np.linspace(0, 1, len(subject_fits)) # Values for each subject.
        colors = cm.jet(sub_x)
        plt.figure(fig_no, figsize = [7,2.3])
        if clf:plt.clf()
        for subject_fit, c, x in zip(subject_fits, colors, sub_x):
            if col:c = col
            model_fit_plot(subject_fit, fig_no, col = c, clf = False, title = title,
                           x_offset = 0.2*x + x_offset, scatter = False, ebars = False)
        if sort_param: return [sf['sID'] for sf in subject_fits]
    else:
        # fit is single population fit containing many subjects sessions.
        population_fit = fit 
        MAP_params_U   = np.array([session_fit['params_U'] for session_fit  in population_fit['session_fits'] ])
        n_ses, n_params = np.shape(MAP_params_U)    
        subjects  = list(set([f['sID'] for f in population_fit['session_fits']]))
        n_subjects = len(subjects)
        sub_means = np.zeros((n_subjects,n_params))
        sub_SDs   = np.zeros((n_subjects,n_params))
        for i,subject in enumerate(subjects):
            sub_MAP_params_U = np.array([session_fit['params_U'] for session_fit in
                                         population_fit['session_fits']
                                         if session_fit['sID'] == subject ])
            sub_means[i,:] = np.mean(sub_MAP_params_U,0)
            sub_SDs[i,:]   = np.std(sub_MAP_params_U,0)

        if sort_param:  # Order subjects by value of specified param.            
            sp_index  = population_fit['param_names'].index(sort_param)
            sub_order = np.argsort(sub_means[:,sp_index])
        else:
            sub_order = np.arange(n_subjects)

        plt.figure(fig_no, figsize = [7,2.3])
        if clf:plt.clf()
        sub_x = np.linspace(0, 1, n_subjects) 
        colors = cm.jet(sub_x)
        for s, x, c in zip(sub_order, sub_x, colors):
            plt.errorbar(np.arange(n_params)+0.4+0.2*x, sub_means[s,:], sub_SDs[s,:],
                linestyle = '', marker = 'o', markersize = 4, markeredgecolor='none',
                linewidth = 2, color = c, capsize = 0, label = subjects[s])
            if sort_param:
                print('Subject: {} Loading: {:.3f}'.format(subjects[s], sub_means[s,sp_index]))
        plt.plot([0,n_params],[0,0],'k')
        plt.xlim(0,n_params)
        plt.xticks(np.arange(n_params)+0.5, population_fit['param_names'])
        plt.ylabel('Log odds')
        if title:
            plt.title(title)
        if sort_param: return [subjects[s] for s in sub_order]


def flat_fits_plot(session_fits, fig_no=1):
    'Plot a set of non-hierarchical session fits'
    colors = cm.jet(np.linspace(0, 1,len(session_fits)))
    plt.figure(fig_no).clf()
    x = np.arange(len(session_fits[0]['params_T']))+0.5
    for color, fit in zip(colors, session_fits):
        plt.plot(x,fit['params_T'], linestyle='None', marker='.', color=color)
    plt.xticks(x, fit['param_names'])

def session_fit_plot(session_fit, fig_no=1):
    if fig_no: plt.figure(fig_no).clf()
    x = np.arange(len(session_fit['params_T']))+0.5
    plt.plot(x,session_fit['params_T'], linestyle='None', marker='o')
    plt.xticks(x, session_fit['param_names'])

def longditudinal_fit_plot(longdit_fit, fig_no = 1, clf = True, col = 'b', title=None):
    'Plot parameters over training epochs from longditudinal fit'
    param_names = longdit_fit[0]['param_names']
    n_params = len(param_names)
    n_epochs = len(longdit_fit)
    epoch_start_days = [ef['start_day'] for ef in longdit_fit]
    param_means = np.zeros((n_params, n_epochs)) 
    param_SDs   = np.zeros((n_params, n_epochs)) 
    for i, epoch_fit in enumerate(longdit_fit):
        param_means[:,i] = epoch_fit['pop_dists']['means']
        param_SDs  [:,i] = epoch_fit['pop_dists']['SDs']
    plt.figure(fig_no)
    if clf:plt.clf()
    for i in range(n_params):
        plt.subplot(n_params, 1, i + 1)
        plt.errorbar(epoch_start_days, param_means[i,:], yerr = param_SDs[i,:], linewidth = 1.5, color = col)
        plt.plot([0.5,epoch_start_days[-1] + 0.5],[0,0],'k')
        plt.xlim(0.5,epoch_start_days[-1] + 0.5)
        plt.ylabel(param_names[i])
    plt.xlabel('Days')
    if title: plt.suptitle(title)
    

def session_action_values(session, agent, params_T, xlim = None, fig_no = 1, fill = True):
    '''Plot action values and preferences for model based and model free system.
    Preferences are difference in action values scaled by mixture parameter and 
    softmax inverse temparature.
    '''
    DVs = agent.session_likelihood(session, params_T, get_DVs = True)
    plt.figure(fig_no).clf()
    plt.subplot(4,1,1)
    pl.session_plot(session)
    if xlim: plt.xlim(xlim)
    plt.subplot(4,1,2)
    plt.plot(-DVs['Q_td'][0,:], '.-r', markersize = 3)
    plt.plot( DVs['Q_td'][1,:], '.-r', markersize = 3)
    plt.plot([0,session.n_trials],[0,0],'k')
    plt.xlim(0,session.n_trials)
    plt.ylim(max(np.abs(np.array(plt.ylim()))) * np.array([-1,1]))
    plt.ylabel('Model free values')
    if xlim: plt.xlim(xlim)
    plt.subplot(4,1,3)
    plt.plot(-DVs['Q_mb'][0,:], '.-g', markersize = 3)
    plt.plot( DVs['Q_mb'][1,:], '.-g', markersize = 3)
    plt.plot([0,session.n_trials],[0,0],'k')
    plt.xlim(0,session.n_trials)
    plt.ylim(max(np.abs(np.array(plt.ylim()))) * np.array([-1,1]))
    plt.ylabel('Model based values')
    if xlim: plt.xlim(xlim)
    plt.subplot(4,1,4)
    if fill:
        plt.fill_between(np.arange(session.n_trials), DVs['P_mf'], color = 'r', alpha = 0.5)
        plt.fill_between(np.arange(session.n_trials), DVs['P_mb'], color = 'g', alpha = 0.5)
    else: 
        plt.plot(DVs['P_mb'], '.-g', markersize = 3)
        plt.plot(DVs['P_mf'], '.-r', markersize = 3)
    plt.plot([0,session.n_trials],[0,0],'k')
    plt.xlim(0,session.n_trials)
    if xlim: plt.xlim(xlim)
    plt.ylabel('Preference')
    plt.xlabel('Trials')
    mean_abs_mb = np.mean(np.abs(DVs['P_mb']))
    mean_abs_td = np.mean(np.abs(DVs['P_mf']))
    print('Model-based mean abs. preference: {:.3}'.format(mean_abs_mb))
    print('Model-free  mean abs. preference: {:.3}'.format(mean_abs_td))
    print('Fraction model based            : {:.3}'.format(mean_abs_mb/(mean_abs_mb + mean_abs_td)))
    print('MB MF preference correlation: {:.3}'.format(
          np.corrcoef(DVs['P_mf'],DVs['P_mb'])[0,1]))

def abs_preference_plot(sessions, population_fit, agent, kernels = True, to_plot = True):
    ses_mean_preference_mb = np.zeros(len(sessions))
    ses_mean_preference_mf = np.zeros(len(sessions))
    #ses_mean_preference_k  = np.zeros(len(sessions))
    for i, (session, session_fit) in enumerate(zip(sessions,population_fit['session_fits'])):
        DVs = agent.session_likelihood(session, session_fit['params_T'], get_DVs = True)
        ses_mean_preference_mb[i] = np.mean(np.abs(DVs['P_mb']))
        ses_mean_preference_mf[i] = np.mean(np.abs(DVs['P_mf']))
        #ses_mean_preference_k[i]  = np.mean(np.abs(DVs['P_k']))
    mean_preference_mb = np.mean(ses_mean_preference_mb)
    mean_preference_mf = np.mean(ses_mean_preference_mf)
    #mean_preference_k = np.mean(ses_mean_preference_k)
    if to_plot:
        plt.figure(to_plot, figsize = [2.5,2.3]).clf()
        if False:#kernels:
            plt.bar([1,2,3],[mean_preference_mb, mean_preference_mf, mean_preference_k])        
            plt.xlim(0.8,4)
            plt.xticks([1.4, 2.4, 3.4], ['Model based', 'Model free', 'kernels'])
        else:
            plt.bar([1,2],[mean_preference_mb, mean_preference_mf])        
            plt.xlim(0.8,3)
            plt.xticks([1.4, 2.4], ['Model based', 'Model free'])

        plt.ylabel('Mean abs. preference')
    else:
        return np.array([mean_preference_mb, mean_preference_mf])


def parameter_evolution_plot(population_fit, fig_no = 1, title = None, clf = True):
    'Plot evolution of model parameters over step of EM fitting.'
    fit_evo = population_fit['fit_evo']
    n_params = len(population_fit['param_names'])    
    plt.figure(fig_no, figsize = [8,12])
    cols = (('r','m'),('b','c'))[int(clf)]
    if clf: plt.clf()
    # Plot Parameter evolution
    param_means = np.array([d['means']      for d in fit_evo['dists']])
    param_SDs = np.sqrt(np.array([d['vars'] for d in fit_evo['dists']]))
    x = np.arange(len(fit_evo['prob']))
    for i in range(n_params):
        plt.subplot(n_params,2,2*i+1)
        plt.locator_params(nbins=4, axis='y')
        if population_fit['param_ranges'][0] == 'all_unc':
            p_range = ['all_unc']
        else:
            p_range = [population_fit['param_ranges'][i]]*len(x)
        plt.plot(x, mf._trans_UT(param_means[:,i], p_range), color = cols[0], linewidth=1)
        low_1SD =  mf._trans_UT(param_means[:,i] - param_SDs[:,i], p_range)
        upp_1SD =  mf._trans_UT(param_means[:,i] + param_SDs[:,i], p_range)
        plt.fill_between(x, low_1SD, upp_1SD, alpha = 0.2, color = cols[0])
        plt.ylabel(population_fit['param_names'][i])
        plt.xlim(0,x[-1])
        plt.xticks(x[::2],'')
    plt.xlabel('EM round')
    plt.xticks(x[::2],x[::2])
    # Plot goodness of fit evolution.
    MAP_prob = np.array(fit_evo['prob'])
    MAP_rel_change = (MAP_prob[:-1]-MAP_prob[1:])/MAP_prob[1:]
    n_plot = 3 if 'iLik' in fit_evo.keys() else 2
    x = np.arange(len(fit_evo['prob']))
    plt.subplot(6,2,2)
    plt.plot(x, MAP_prob, color = cols[0])#-MAP_prob[-1],'b')
    plt.title('{:.0f}'.format(MAP_prob[-1]), loc = 'left', fontsize = 'small')
    plt.locator_params(nbins=4, axis='y')
    plt.xlim(0,x[-1])
    plt.xticks(x[::2],'')
    plt.ylabel('MAP prob.')
    if 'iLik' in fit_evo.keys():
        iLik = np.array(fit_evo['iLik'])
        iLik_rel_change = (iLik[:-1]-iLik[1:])/iLik[1:]
        plt.subplot(6,2,4)
        plt.plot(x, iLik, color = cols[1])#-iLik[-1],'g')
        plt.title('{:.0f}'.format(iLik[-1]), loc = 'left', fontsize = 'small')
        plt.ylabel('Int. Lik.')
        plt.xlim(0,x[-1])
        plt.locator_params(nbins=4, axis='y')
        plt.xticks(x[::2],'')
        plt.subplot(6,2,6)
        plt.semilogy(x[1:], iLik_rel_change, color = cols[1])
        plt.plot([0,x[-1]],[1e-4,1e-4],'g:')
    plt.subplot(6,2,6)
    plt.semilogy(x[1:], MAP_rel_change, color = cols[0])
    plt.plot([0,x[-1]],[1e-3,1e-3],'b:')
    plt.xlabel('EM round')
    plt.ylabel('Rel. change')
    plt.xlim(0,x[-1])
    plt.xticks(x[::2])
    if title: plt.suptitle(title)


def true_vs_fitted_session_params(population_fit, sim_sessions, fig_no = 1):
    ''' For a set of simulated sessions plot the fitted parameter values
    against the true paramter values to assess goodness of fit.'''

    #Unpack true and fitted parameter values into arrays.
    true_params_U = np.array([session.true_params_U for session in sim_sessions])
    true_params_T = np.array([session.true_params_T for session in sim_sessions])

    MAP_params_U   = np.array([session_fit['params_U'] for session_fit  in population_fit['session_fits'] ]).T
    MAP_params_T   = np.array([session_fit['params_T'] for session_fit  in population_fit['session_fits'] ]).T

    n_params = np.shape(true_params_U)[0]
    n_sessions = np.shape(true_params_U)[1]
    cols = (np.arange(n_sessions) + 0.01)/n_sessions
    plt.figure(fig_no).clf()
  
    for true_params_U_i, true_params_T_i,  MAP_params_U_i, MAP_params_T_i, i in \
        zip(true_params_U, true_params_T,  MAP_params_U, MAP_params_T, list(range(n_params))):
 
        plt.subplot(2, n_params, i + 1)
        plt.scatter(true_params_U_i, MAP_params_U_i, c = cols, cmap = 'hsv', vmin = 0., vmax = 1.)
        plt.plot([min(true_params_U_i), max(true_params_U_i)], [min(true_params_U_i), max(true_params_U_i)] , 'k')
        plt.locator_params(axis = 'x', nbins = 4, tight = True)

        plt.subplot(2, n_params, 1 * n_params + i + 1)
        plt.scatter(true_params_T_i, MAP_params_T_i, c = cols, cmap = 'hsv', vmin = 0., vmax = 1.)
        plt.plot([min(true_params_T_i), max(true_params_T_i)], [min(true_params_T_i), max(true_params_T_i)] , 'k')
        plt.locator_params(axis = 'x', nbins = 4, tight = True)

    plt.subplot(2, n_params, 1)
    plt.title('MAP Fits: Unconstrained space')
    plt.subplot(2, n_params, 1 + n_params)
    plt.title('MAP Fits: True space')

    for i in range(n_params):
        plt.subplot(2, n_params, i + 1 + (2 - 1) * n_params)
        plt.xlabel(population_fit['param_names'][i])

# -------------------------------------------------------------------------------------
# Fit correlation plots.
# -------------------------------------------------------------------------------------

def session_fit_correlations(population_fit, fig_no = 1, diag_zero = False, vmax = 1, use_abs = False):
    'Evaluate and plot correlation matrix between MAP fit parameters.'
    MAP_params_U   = np.array([session_fit['params_U'] for session_fit  in population_fit['session_fits'] ])
    if use_abs:
        MAP_params_U = np.abs(MAP_params_U)
    R = np.corrcoef(MAP_params_U.T)
    if diag_zero:
        np.fill_diagonal(R, 0)
    n_params = len(population_fit['param_names'])
    plt.figure(fig_no).clf()
    plt.pcolor(R, vmin = 0, vmax = vmax)
    plt.colorbar()
    plt.xticks(np.arange(n_params)+0.5, population_fit['param_names'])
    plt.yticks(np.arange(n_params)+0.5, population_fit['param_names'])


def within_and_cross_subject_correlations(subject_fits, fig_no = 1):
    n_params = len(subject_fits[0]['pop_dists']['means'])
    subject_means = np.array([f['pop_dists']['means'] for f in subject_fits])
    cross_subject_corr = np.corrcoef(subject_means.T)
    within_subject_corrs = []
    for subject_fit in subject_fits:
        MAP_params_U   = np.array([session_fit['params_U'] for session_fit  in subject_fit['session_fits'] ])
        within_subject_corrs.append(np.corrcoef(MAP_params_U.T))
    ave_within_subject_corr = np.mean(np.array(within_subject_corrs),0)
    plt.figure(fig_no).clf()
    plt.subplot(1,2,1)
    plt.pcolor(cross_subject_corr)
    plt.colorbar()
    plt.xticks(np.arange(n_params)+0.5, subject_fits[0]['param_names'])
    plt.yticks(np.arange(n_params)+0.5, subject_fits[0]['param_names'])
    plt.title('Cross subject correlations')
    plt.subplot(1,2,2)
    plt.pcolor(ave_within_subject_corr, vmin = 0, vmax = 0.5)
    plt.colorbar()
    plt.xticks(np.arange(n_params)+0.5, subject_fits[0]['param_names'])
    plt.yticks(np.arange(n_params)+0.5, subject_fits[0]['param_names'])
    plt.title('Within subject correlations')


def parameter_autocor(sessions, population_fit, param = 'side'):
    ''' Evaluate within and cross subject variability in 
    specified parameter and autocorrelation across sessions.
    '''

    assert len(population_fit['session_fits']) == len(sessions), \
        'Population fit does not match number of sessions.'

    sp_index = population_fit['param_names'].index(param)
    for i, session_fit in enumerate(population_fit['session_fits']):
        sessions[i].side_loading = session_fit['params_U'][sp_index]

    sIDs = list(set([s.subject_ID for s in sessions]))

    plt.figure(1).clf()
    plt.subplot2grid((2,2),(0,0), colspan = 2)
    subject_means = [] 
    subject_SDs = []
    cor_len = 20
    subject_autocorrelations = np.zeros([len(sIDs), 2 * cor_len + 1])
    subject_shuffled_autocor = np.zeros([len(sIDs), 2 * cor_len + 1, 1000])
    for i, sID in enumerate(sIDs):
        a_sessions = sorted([s for s in sessions if s.subject_ID == sID],
                            key = lambda s:s.day)
        sl = [s.side_loading for s in a_sessions]
        plt.plot(sl)
        subject_means.append(np.mean(sl))
        subject_SDs.append(np.std(sl))
        sl = (np.array(sl) - np.mean(sl)) / np.std(sl)
        autocor = np.correlate(sl, sl, 'full') / len(sl)
        subject_autocorrelations[i,:] = autocor[autocor.size/2 - cor_len:
                                                autocor.size/2 + cor_len + 1]
        for j in range(1000):
            shuffle(sl)
            autocor = np.correlate(sl, sl, 'full') / len(sl)
            subject_shuffled_autocor[i,:,j] = autocor[autocor.size/2 - cor_len:
                                                autocor.size/2 + cor_len + 1]


    mean_shuffled_autocors = np.mean(subject_shuffled_autocor,0)
    mean_shuffled_autocors.sort(1)

    plt.xlabel('Day')
    plt.ylabel('Subject rotational bias')
    plt.subplot2grid((2,2),(1,0))
    plt.fill_between(list(range(-cor_len, cor_len + 1)),mean_shuffled_autocors[:,10],
                   mean_shuffled_autocors[:,-10], color = 'k', alpha = 0.2)
    plt.plot(list(range(-cor_len, cor_len + 1)),np.mean(subject_autocorrelations,0),'b.-', markersize = 5)
    plt.xlabel('Lag (days)')
    plt.ylabel('Correlation')
    plt.subplot2grid((2,2),(1,1))
    plt.bar([0.5,1.5], [np.mean(subject_SDs), np.std(subject_means)])
    plt.xticks([1,2], ['Within subject', 'Cross subject'])
    plt.xlim(0.25,2.5)
    plt.ylabel('Standard deviation')


def predictor_correlations(sessions,agent, fig_no = 1):
    'Plot correlation matrix between predictors in logistic regression models.'
    predictors = []
    for session in sessions:
        predictors.append(agent._get_session_predictors(session))
    predictors = np.vstack(predictors)
    R = np.corrcoef(predictors.T)
    n_params = len(agent.param_names)-1
    plt.figure(fig_no).clf()
    plt.pcolor(R)#, vmin = 0, vmax = vmax)
    plt.colorbar()
    plt.xticks(np.arange(n_params)+0.5, agent.param_names[1:])
    plt.yticks(np.arange(n_params)+0.5, agent.param_names[1:])
