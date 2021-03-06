
import pylab as plt
import numpy as np
import scipy as sp
from collections import OrderedDict
from random import shuffle
import matplotlib.cm as cm
from . import plotting as pl
from . import utility as ut
from . import RL_utils as ru

def fit_evolution_plot(population_fit):
    'Plot evoluation of population parameters over rounds of EM.'

    param_evol = population_fit['param_evol']
    param_means = np.array(param_evol['means_T'])
    lower_conf_int = np.array(param_evol['lower_conf_T'])
    upper_conf_int = np.array(param_evol['upper_conf_T'])

    n_params = len(param_evol['means_T'][0])    
    plt.figure(1)
    plt.clf()
    plt.subplot(n_params + 1, 1, 0)
    posterior_prob = np.array(param_evol['liks'])
        
    x = np.arange(param_means.shape[0])
    plt.plot(x, posterior_prob)
    plt.ylabel('Posterior prob.')
    for i in range(n_params):
        plt.subplot(n_params + 1,1,i + 1)
        plt.plot(x, param_means[:,i], 'k')
        plt.plot(x, upper_conf_int[:,i], 'Grey')
        plt.plot(x, lower_conf_int[:,i], 'Grey')
        plt.ylabel(population_fit['param_names'][i])
        plt.ylim(min(lower_conf_int[:,i]) - 0.1, max(upper_conf_int[:,i]) + 0.1)


def true_vs_fitted_session_params(population_fit):
    ''' For a set of simulated sessions plot the fitted parameter values
    against the true paramter values to assess goodness of fit.  If the 
    population fit includes ML fits, these are plotted as well as the MAP_fit
    fits and some metrics of the difference in fit quality are provided.'''

    #Unpack true and fitted parameter values into arrays.
    true_params_U = np.array(population_fit['true_values']['params_U']).T   
    true_params_T = np.array(population_fit['true_values']['params_T']).T 


    MAP_params_U   = np.array([MAP_fit['params_U'] for MAP_fit  in population_fit['MAP_fits'] ]).T
    MAP_params_T   = np.array([MAP_fit['params_T'] for MAP_fit  in population_fit['MAP_fits'] ]).T

    n_params = np.shape(true_params_U)[0]
    n_sessions = np.shape(true_params_U)[1]
    cols = (np.arange(n_sessions) + 0.01)/n_sessions
    n_rows = 2
    plt.figure(2)
    plt.clf()

    if not population_fit['ML_fits'] is None:

        ML_params_U   = np.array([ML_fit['params_U'] for ML_fit  in population_fit['ML_fits'] ]).T
        ML_params_T   = np.array([ML_fit['params_T'] for ML_fit  in population_fit['ML_fits'] ]).T

        #Evaluate and print fit errors.
        ML_mean_abs_error_T  = np.mean(np.abs(ML_params_T   - true_params_T), 1)
        MAP_mean_abs_error_T = np.mean(np.abs(MAP_params_T  - true_params_T), 1)
        mean_abs_error_MAP_bonus = ML_mean_abs_error_T - MAP_mean_abs_error_T
        ML_error_of_mean_T  = np.abs(np.mean(ML_params_T, 1)  - np.mean(true_params_T, 1))
        MAP_error_of_mean_T = np.abs(np.mean(MAP_params_T, 1) - np.mean(true_params_T, 1))
        error_of_mean_MAP_bonus = ML_error_of_mean_T - MAP_error_of_mean_T
        print(('MAP bonus - mean absolute error: {}'.format(mean_abs_error_MAP_bonus)))
        print(('MAP bonus - error of mean      : {}'.format(error_of_mean_MAP_bonus)))

        n_rows = 4

        for true_params_U_i, true_params_T_i, ML_params_U_i, ML_params_T_i, i in \
           zip(true_params_U, true_params_T, ML_params_U, ML_params_T, list(range(n_params))):

            plt.subplot(4, n_params, 2 * n_params + i + 1)
            plt.scatter(true_params_U_i, ML_params_U_i, c = cols, cmap = 'hsv', vmin = 0., vmax = 1.)
            plt.plot([min(true_params_U_i), max(true_params_U_i)], [min(true_params_U_i), max(true_params_U_i)] , 'k')
            plt.locator_params(axis = 'x', nbins = 4, tight = True)

            plt.subplot(4, n_params, 3 * n_params + i + 1)
            plt.scatter(true_params_T_i, ML_params_T_i, c = cols, cmap = 'hsv', vmin = 0., vmax = 1.)
            plt.plot([min(true_params_T_i), max(true_params_T_i)], [min(true_params_T_i), max(true_params_T_i)] , 'k')
            plt.locator_params(axis = 'x', nbins = 4, tight = True)

        plt.subplot(4, n_params, 1 + 2 * n_params)
        plt.title('ML Fits: Unconstrained space')
        plt.subplot(4, n_params, 1 + 3 * n_params)
        plt.title('ML Fits: True space')

  
    for true_params_U_i, true_params_T_i,  MAP_params_U_i, MAP_params_T_i, i in \
        zip(true_params_U, true_params_T,  MAP_params_U, MAP_params_T, list(range(n_params))):
 
        plt.subplot(n_rows, n_params, i + 1)
        plt.scatter(true_params_U_i, MAP_params_U_i, c = cols, cmap = 'hsv', vmin = 0., vmax = 1.)
        plt.plot([min(true_params_U_i), max(true_params_U_i)], [min(true_params_U_i), max(true_params_U_i)] , 'k')
        plt.locator_params(axis = 'x', nbins = 4, tight = True)

        plt.subplot(n_rows, n_params, 1 * n_params + i + 1)
        plt.scatter(true_params_T_i, MAP_params_T_i, c = cols, cmap = 'hsv', vmin = 0., vmax = 1.)
        plt.plot([min(true_params_T_i), max(true_params_T_i)], [min(true_params_T_i), max(true_params_T_i)] , 'k')
        plt.locator_params(axis = 'x', nbins = 4, tight = True)

    plt.subplot(n_rows, n_params, 1)
    plt.title('MAP Fits: Unconstrained space')
    plt.subplot(n_rows, n_params, 1 + n_params)
    plt.title('MAP Fits: True space')

    for i in range(n_params):
        plt.subplot(n_rows, n_params, i + 1 + (n_rows - 1) * n_params)
        plt.xlabel(population_fit['param_names'][i])


def dists_and_hists_plot(population_fit, fig_no = 1):
    ''' Plot population distributions and histogram of MAP parameter estimates in true space.
    If simulated data, plot true and fitted data seperately.
    '''
    fig = plt.figure(fig_no)
    #fig.tight_layout()
    plt.clf()
    pop_params = population_fit['pop_params']
    param_ranges = pop_params['ranges']
    MAP_params_T   = np.array([MAP_fit['params_T'] for MAP_fit  in population_fit['MAP_fits'] ])
    n_params = np.shape(MAP_params_T)[1]

    n_rows = 1
    if False:#population_fit.has_key('pop_params_true') & \
        #(n_params == len(population_fit['pop_params_true']['means'])):
        n_rows = 2 
        pop_params_true = population_fit['pop_params_true']
        if isinstance(pop_params_true['SDs'], float):
            pop_params_true['SDs'] = np.ones(n_params) * pop_params_true['SDs']
        true_params_T = np.array(population_fit['true_values']['params_T'])
        for i in range(n_params):
            plt.subplot(2,n_params,i+1)
            _plot_dist_and_hist(pop_params_true['means'][i], pop_params_true['SDs'][i],
                               true_params_T[:,i], pop_params['ranges'][i], 'r')
            plt.locator_params(nbins = 4)
            plt.subplot(2, n_params, n_params + i + 1)
            _plot_dist_and_hist(pop_params_true['means'][i], pop_params_true['SDs'][i],
                               None, pop_params['ranges'][i], 'r')

    for i in range(n_params):
        plt.subplot(n_rows, n_params, (n_rows-1) * n_params + i + 1)
        _plot_dist_and_hist(pop_params['means'][i], pop_params['SDs'][i],
                           MAP_params_T[:,i], pop_params['ranges'][i],'k')
        plt.locator_params(nbins = 4)
        plt.yticks([])
        plt.xlabel(population_fit['param_names'][i])
        if param_ranges[i] == 'unit':
            plt.xlim(0,1)

def pop_fit_comparison(population_fit_A, population_fit_B, fig_no = 1, normalize = False):
    'Plot multiple population fits on the same axis for comparison.'
    plt.figure(fig_no)
    plt.clf()
    pop_params_A = population_fit_A['pop_params']
    pop_params_B = population_fit_B['pop_params']
    n_params = len(pop_params_A['ranges'])
    for i in range(n_params):
        plt.subplot(1, n_params, i + 1)
        _plot_dist(pop_params_A['means'][i], pop_params_A['SDs'][i], 
                   pop_params_A['ranges'][i], col = 'b', normalize = normalize)
        _plot_dist(pop_params_B['means'][i], pop_params_B['SDs'][i], 
                   pop_params_B['ranges'][i], col = 'r', normalize = normalize)
        plt.locator_params('x', nbins = 3)
        plt.xlabel(population_fit_A['param_names'][i])
        plt.yticks([])
    for i in range(n_params):
        plt.subplot(1, n_params, i + 1)
        plt.ylim([0,plt.ylim()[1] * 1.1])
            

def _plot_dist_and_hist(mean_U, SD_U, vals_T, rnge, col = 'r'):
    '''For population distribution specified by mean and SD in unconstrained space, 
    and individual subject parameter values specified in true space, plot in true
    space population distribution and histogram of subject values.'''

    if vals_T is not None:
        #Make histogram of subject param values.
        if rnge == 'unit': 
            bins = np.arange(0,1.1,0.05)
        else:
            bins = 10
        hist,bin_edges = np.histogram(vals_T, bins)
        bin_width = bin_edges[1] - bin_edges[0]

        plt.bar(bin_edges[0:-1],hist/(len(vals_T)*bin_width), width = bin_width)
        _plot_dist(mean_U, SD_U, rnge, col = col, ls = '-')
    else:
        _plot_dist(mean_U, SD_U, rnge, col = col, ls = '--')

def scatter_plot_comp(fit_1, fit_2, fig_no = 1):
    pop_scatter_plot(fit_1, fig_no, col = 'b', clf = True , x_offset = -0.1)
    pop_scatter_plot(fit_2, fig_no, col = 'r', clf = False, x_offset =  0.1)

def pop_scatter_plot(population_fit, fig_no = 1, col = 'b', clf = True, subplot = None,
                     x_offset = 0., sort_param = None, subjects = None, title = None):
    pop_params = population_fit['pop_params']
    MAP_params_U   = np.array([MAP_fit['params_U'] for MAP_fit  in population_fit['MAP_fits'] ])
    n_ses, n_params = np.shape(MAP_params_U)    
    if subplot:
        plt.subplot(subplot[0], subplot[1], subplot[2])
    else:
        plt.figure(fig_no)
        if clf:plt.clf()
    if col == 'sID': # Color session MAP fits by animals ID: 
        if not subjects:
            subjects  = list(set([f['sID'] for f in population_fit['MAP_fits']]))
        sub_means = np.zeros((len(subjects),n_params))
        sub_SDs   = np.zeros((len(subjects),n_params))
        for i,subject in enumerate(subjects):
            sub_MAP_params_U = np.array([MAP_fit['params_U'] for MAP_fit  in 
                               population_fit['MAP_fits'] if MAP_fit['sID'] == subject ])
            sub_means[i,:] = np.mean(sub_MAP_params_U,0)
            sub_SDs[i,:]   = np.std(sub_MAP_params_U,0)
        if sort_param:  # Order subjects by value of specified param.            
            sp_index = population_fit['param_names'].index(sort_param)
            sub_order = np.argsort(sub_means[:,sp_index])
        else:
            sub_order = np.arange(len(subjects))
        sub_x = np.linspace(0, 1, len(subjects)) 
        colors = cm.jet(sub_x)
        for s, x, c in zip(sub_order, sub_x, colors):
            plt.errorbar(np.arange(n_params)+0.4+0.2*x, sub_means[s,:], sub_SDs[s,:],
                linestyle = '', marker = 'o', markersize = 4, markeredgecolor='none',
                linewidth = 2, color = c, capsize = 0, label = subjects[s])
            if sort_param:
                print('Subject: {} Loading: {:.3f}'.format(subjects[s], sub_means[s,sp_index]))
        plt.legend(bbox_to_anchor=(1.2, 1))
    else: # Use specified color for plot.    
        plt.errorbar(np.arange(n_params)+0.5+x_offset, pop_params['means'], pop_params['SDs'],linestyle = '', linewidth = 2, color = col)
        for i in range(n_params):
            param_vals = MAP_params_U[:,i]
            plt.scatter(i+0.4+x_offset+0.2*np.random.rand(n_ses),param_vals, s = 4,  facecolor= col, edgecolors='none', lw = 0)
    plt.plot([0,n_params],[0,0],'k')
    plt.xlim(0,n_params)
    plt.xticks(np.arange(n_params)+0.5, population_fit['param_names'])
    plt.ylabel('Log odds')
    if title:
        plt.title(title)
    if sort_param:
        return [subjects[s] for s in sub_order]


def _plot_dist(mean_U, SD_U, rnge, col = 'r', ls = '-', normalize = False):
    #Transform population distribution from true to unconstrained space.
    if rnge == 'unc':
        T = np.arange(mean_U - 3 * SD_U, mean_U + 3 * SD_U, 6 * SD_U / 100.)
        dUdT = 1.
        U = T
        x_ticks = np.linspace(np.ceil(T[0]),np.floor(T[-1]),3)
        if x_ticks[0] == x_ticks[2]:
            x_ticks = np.round([T[0],T[-1]],1)
    elif rnge == 'unit':
        T = np.arange(0.001,0.999,0.001)
        U = ru.inverse_sigmoid(T)
        dUdT = ru.inv_sigmoid_grad(T)
        x_ticks = [0,0.5,1]
    elif rnge == 'pos':
        T_range = np.exp([mean_U - 3 * SD_U, mean_U + 3 * SD_U])
        T = np.linspace(T_range[0], T_range[1],100)
        U = np.log(T)
        dUdT = 1./T
        x_ticks = np.linspace(np.ceil(T[0]),np.floor(T[-1]),3)
    spacing = T[1] - T[0]
    dist = splt.stats.norm(mean_U,SD_U).pdf(U) * dUdT
    if normalize:
        dist = dist / np.max(dist)
    else:
        dist = dist / (dist.sum() * spacing)
    if col:
        plt.plot(T, dist, color = col, linestyle = ls, linewidth = 1.5)
    else:
        plt.plot(T, dist, linestyle = ls, linewidth = 1.5)
    #plt.xticks(x_ticks)
    #plt.xlim(T[0]-0.02,T[-1]+0.02)

def logistic_regression_plot(population_fit, n_back, fig_no = 1):
    ''' Plot weight vectors for fits of logistic regression model.'''
    pop_params = population_fit['pop_params']
    bias = np.array([pop_params['means'][0], pop_params['SDs'][0]])
    means = pop_params['means'][1:].reshape(-1, n_back)
    SDs = pop_params['SDs'][1:].reshape(-1, n_back)
    x = np.arange(- 1, - n_back - 1, -1)
    n_plot = shape(SDs)[0]
    y_max = np.max(means + SDs) + 0.2
    y_min = np.min(means - SDs) - 0.2
    plt.figure(fig_no)
    for mn, sd, i in zip(means, SDs, list(range(n_plot))):
        plt.subplot(n_plot, 1, i + 1)
        plt.errorbar(x,mn, yerr = sd)
        plt.plot((-n_back - 0.5, - 0.5), (0,0), 'k')
        plt.xlim(-n_back - 0.5, - 0.5)
        plt.ylim(y_min, y_max)
        plt.xticks(x)
    plt.xlabel('Trials back')

def logistic_regression_plot_2(population_fit, n_back = 4, fig_no = 2, line_style = '-'):
    pop_params = population_fit['pop_params']
    plt.figure(fig_no)
    plt.subplot(3,1,1)
    x = np.arange(- 1, - n_back - 1, -1)
    bias = np.array([pop_params['means'][0], pop_params['SDs'][0]])
    means = pop_params['means'][1:].reshape(-1, n_back)
    SDs = pop_params['SDs'][1:].reshape(-1, n_back)
    plt.errorbar(x, means[0,:], yerr = SDs[0,:],color = 'b', linestyle = line_style)
    plt.errorbar(x, means[1,:], yerr = SDs[1,:],color = 'g', linestyle = line_style)
    plt.ylabel('Log odds', {'horizontalalignment' : 'center'})
    plt.xlim([-4.2,-0.5])
    plt.xticks([-4,-3,-2,-1])
    plt.subplot(3,1,2)
    plt.errorbar(x, means[2,:], yerr = SDs[2,:],color = 'b', linestyle = line_style)
    plt.errorbar(x, means[3,:], yerr = SDs[3,:],color = 'g', linestyle = line_style)
    plt.xlim([-4.2,-0.5])
    plt.xticks([-4,-3,-2,-1])
    plt.ylabel('Log odds', {'horizontalalignment' : 'center'})
    if shape(means)[0] > 4:
        plt.subplot(3,1,3)
        plt.errorbar(x, means[4,:], yerr = SDs[4,:],color = 'k', linestyle = line_style)
        plt.errorbar(x, means[5,:], yerr = SDs[5,:],color = 'm', linestyle = line_style)
        plt.errorbar(-0.75, bias[0], yerr = bias[1],color = 'r', linestyle = line_style)
        plt.xlim([-4.2,-0.5])
        plt.xticks([-4,-3,-2,-1])
        plt.xlabel('Trials back')
        plt.ylabel('Log odds', {'horizontalalignment' : 'center'})

def agent_param_dists_plot(agent, fig_no):
        plt.figure(fig_no)
        plt.clf()
        means = agent.pop_params['means']
        SDs   = agent.pop_params['SDs']
        if isinstance(SDs, float):
            SDs = np.ones(agent.n_params) * SDs
        for mean, SD, rnge, i in zip(means, SDs, agent.param_ranges,
                                    list(range(agent.n_params))):
            plt.subplot(1,agent.n_params, i + 1)
            _plot_dist(mean, SD, rnge)

def longditudinal_fit_plot(epoch_fits, fig_no = 1, clf = True, col = 'b'):
    param_names = epoch_fits[0]['param_names']
    n_params = len(param_names)
    n_epochs = len(epoch_fits)
    epoch_start_days = [ef['start_day'] for ef in epoch_fits]
    param_means = np.zeros((n_params, n_epochs)) 
    param_SDs   = np.zeros((n_params, n_epochs)) 
    for i, epoch_fit in enumerate(epoch_fits):
        param_means[:,i] = epoch_fit['pop_params']['means']
        param_SDs  [:,i] = epoch_fit['pop_params']['SDs']
    for i in range(n_params):
        plt.subplot(n_params, 1, i + 1)
        plt.errorbar(epoch_start_days, param_means[i,:], yerr = param_SDs[i,:], linewidth = 1.5, color = col)
        plt.plot([0,epoch_start_days[-1] + 2],[0,0],'k')
        plt.xlim(0,epoch_start_days[-1] + 2)
        plt.ylabel(param_names[i])
    plt.xlabel('Days')

def subject_fits_plot(subject_fits, fig_no = 1, col = None, black_1st = False, ebars = False):
    n_params = len(subject_fits[0]['pop_params']['means'])
    sub_x = np.linspace(0, 1, len(subject_fits)) # Values for each 
    colors = cm.rainbow(sub_x)
    if black_1st:
        colors[0,:]= np.array([0,0,0,1]) # Set first color to black.
    plt.figure(fig_no)
    plt.clf()
    for subject_fit, c, x in zip(subject_fits, colors, sub_x):
        pop_params = subject_fit['pop_params']
        if col:c = col
        if ebars:
            plt.errorbar(np.arange(n_params)+0.4+0.2*x, pop_params['means'], pop_params['SDs'],linestyle = '', marker = 'o', linewidth = 2, color = c)
        else:
            plt.plot(np.arange(n_params)+0.4+0.2*x, pop_params['means'], linestyle = '', marker = 'o', color = c)
    plt.plot([0,n_params],[0,0],'k')
    plt.xlim(0,n_params)
    plt.xticks(np.arange(n_params)+0.5, subject_fits[0]['param_names'])    
    
def MAP_fit_correlations(population_fit, fig_no = 1, diag_zero = False, vmax = 1, use_abs = False):
    'Evaluate and plot correlation matrix between MAP fit parameters.'
    MAP_params_U   = np.array([MAP_fit['params_U'] for MAP_fit  in population_fit['MAP_fits'] ])
    if use_abs:
        MAP_params_U = np.abs(MAP_params_U)
    R = np.corrcoef(MAP_params_U.T)
    if diag_zero:
        np.fill_diagonal(R, 0)
    n_params = len(population_fit['param_names'])
    plt.figure(fig_no)
    plt.clf()
    plt.pcolor(R, vmin = 0, vmax = vmax)
    plt.colorbar()
    plt.xticks(np.arange(n_params)+0.5, population_fit['param_names'])
    plt.yticks(np.arange(n_params)+0.5, population_fit['param_names'])

def within_and_cross_subject_correlations(subject_fits, fig_no = 1):
    n_params = len(subject_fits[0]['pop_params']['means'])
    subject_means = np.array([f['pop_params']['means'] for f in subject_fits])
    cross_subject_corr = np.corrcoef(subject_means.T)
    within_subject_corrs = []
    for subject_fit in subject_fits:
        MAP_params_U   = np.array([MAP_fit['params_U'] for MAP_fit  in subject_fit['MAP_fits'] ])
        within_subject_corrs.append(np.corrcoef(MAP_params_U.T))
    ave_within_subject_corr = np.mean(np.array(within_subject_corrs),0)
    plt.figure(fig_no)
    plt.clf()
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

def lagged_log_reg_plot(population_fit, n_back = 4, n_non_lag = 1, fig_no = 1, clf = True):
    pop_params = population_fit['pop_params']
    n_params = (len(pop_params['means']) - n_non_lag)/n_back
    means = pop_params['means']
    SDs = pop_params['SDs']
    plt.figure(fig_no)
    if clf:plt.clf()
    x = list(range(1,n_back + 1))
    for i in range(n_params):
        param_means = means[n_non_lag + i::n_params]
        param_SDs   =   SDs[n_non_lag + i::n_params]
        plt.subplot(n_params, 1, i + 1)
        plt.errorbar(x, param_means, param_SDs)
        plt.ylabel(population_fit['param_names'][i+n_non_lag][:-2])
        plt.plot([0.5,n_back + 0.5],[0,0],'k')
        plt.locator_params(nbins = 4)
        plt.xticks(list(range(1,n_back + 1)),[])
        plt.xlim(0.5,n_back + 0.5)
    plt.xticks(list(range(1,n_back + 1)),-np.arange(1,n_back + 1))

def named_params_plot(fits, fig_no = 1, params = [], cols = None, black_1st = False, x_space = 0.2, plot_MAP = True):
    ''' Takes list of population_fits and plot the population parameters by name, rather
    than order, to allow comparison of parameter values across models. To plot subset of 
    params, provide list of parameter names as params argument.
    '''
    if not params:
        params = []
        for fit in fits:
            params += fit['param_names']
        params = list(OrderedDict.fromkeys(params))
    n_params = len(params)
    if plot_MAP:
        n_ses = len(fits[0]['MAP_fits'])
        all_MAP_fits = [np.array([MAP_fit['params_U'] for MAP_fit  in fit['MAP_fits']]) for fit in fits]
    sub_x = np.linspace(0, 1, len(fits)) 
    if cols:
        colors = cols
    else:
        colors = cm.rainbow(sub_x)
    if black_1st:
        colors[0,:]= np.array([0,0,0,1]) # Set first color to black.
    plt.figure(fig_no)
    plt.clf()
    for i, param_name in enumerate(params):
        for j, fit in enumerate(fits):
            if param_name in fit['param_names']:               
                param_no = fit['param_names'].index(param_name)
                plt.errorbar(i+sub_x[j]*x_space-x_space/2., \
                    fit['pop_params']['means'][param_no], fit['pop_params']['SDs'][param_no], \
                    linestyle = '', linewidth = 2, marker = '.', color = colors[j])
                if plot_MAP:
                    param_vals = all_MAP_fits[j][:,param_no]
                    plt.scatter(i+sub_x[j]*x_space-x_space+x_space*np.random.rand(n_ses),param_vals,
                              s = 4,  facecolor = colors[j], edgecolors ='none', lw = 0)

    plt.plot([-0.5,n_params - 0.5],[0,0],'k')
    plt.xticks(np.arange(n_params), params)
    plt.xlim(-0.5,n_params - 0.5)

def predictor_correlations(sessions,agent, fig_no = 1):
    ''' Evaluate and plot correlation matrix between predictors in 
    logistic regression models.
    '''
    predictors = []
    for session in sessions:
        predictors.append(agent._get_session_predictors(session))
    predictors = np.vstack(predictors)
    R = np.corrcoef(predictors.T)
    n_params = len(agent.param_names)-1
    plt.figure(fig_no)
    plt.clf()
    plt.pcolor(R)#, vmin = 0, vmax = vmax)
    plt.colorbar()
    plt.xticks(np.arange(n_params)+0.5, agent.param_names[1:])
    plt.yticks(np.arange(n_params)+0.5, agent.param_names[1:])

def session_action_values(session, agent, params_T, xlim = None, fig_no = 1, fill = True):
    '''Plot action values and preferences for model based and model free system.
    Preferences are difference in action values scaled by mixture parameter and 
    softmax inverse temparature.
    '''
    trial_data = agent.session_likelihood(session, params_T, return_trial_data = True)
    plt.figure(fig_no)
    plt.clf()
    plt.subplot(4,1,1)
    pl.choice_mov_ave(session, 'current_axis', False)
    if xlim: plt.xlim(xlim)
    plt.subplot(4,1,2)
    plt.plot(-trial_data['Q_td'][:,0], '.-r', markersize = 3)
    plt.plot( trial_data['Q_td'][:,1], '.-r', markersize = 3)
    plt.plot([0,session.n_trials],[0,0],'k')
    plt.xlim(0,session.n_trials)
    plt.ylim(np.max(np.abs(np.array(plt.ylim()))) * np.array([-1,1]))
    plt.ylabel('Model free values')
    if xlim: plt.xlim(xlim)
    plt.subplot(4,1,3)
    plt.plot(-trial_data['Q_mb'][:,0], '.-g', markersize = 3)
    plt.plot( trial_data['Q_mb'][:,1], '.-g', markersize = 3)
    plt.plot([0,session.n_trials],[0,0],'k')
    plt.xlim(0,session.n_trials)
    plt.ylim(np.max(np.abs(np.array(plt.ylim()))) * np.array([-1,1]))
    plt.ylabel('Model based values')
    if xlim: plt.xlim(xlim)
    plt.subplot(4,1,4)
    if fill:
        plt.fill_between(np.arange(session.n_trials),  -trial_data['P_mb'], color = 'g', alpha = 0.5)
        plt.fill_between(np.arange(session.n_trials),  -trial_data['P_td'], color = 'r', alpha = 0.5)
    else: 
        plt.plot(-trial_data['P_mb'], '.-g', markersize = 3)
        plt.plot(-trial_data['P_td'], '.-r', markersize = 3)
    plt.plot([0,session.n_trials],[0,0],'k')
    plt.xlim(0,session.n_trials)
    if xlim: plt.xlim(xlim)
    plt.ylabel('Preference')
    plt.xlabel('Trials')
    mean_abs_mb = np.mean(np.abs(trial_data['P_mb']))
    mean_abs_td = np.mean(np.abs(trial_data['P_td']))
    print(('Model-based mean abs. preference: {}'.format(mean_abs_mb)))
    print(('Model-free  mean abs. preference: {}'.format(mean_abs_td)))
    print(('Fraction model based            : {}'.format(mean_abs_mb/(mean_abs_mb + mean_abs_td))))

def abs_preference_plot(sessions, population_fit, agent, kernels = True, to_plot = True):
    ses_mean_preference_mb = np.zeros(len(sessions))
    ses_mean_preference_td = np.zeros(len(sessions))
    ses_mean_preference_k  = np.zeros(len(sessions))
    for i, (session, MAP_fit) in enumerate(zip(sessions,population_fit['MAP_fits'])):
        trial_data = agent.session_likelihood(session, MAP_fit['params_T'], return_trial_data = True)
        ses_mean_preference_mb[i] = np.mean(np.abs(trial_data['P_mb']))
        ses_mean_preference_td[i] = np.mean(np.abs(trial_data['P_td']))
        ses_mean_preference_k[i]  = np.mean(np.abs(trial_data['P_k']))
    mean_preference_mb = np.mean(ses_mean_preference_mb)
    mean_preference_td = np.mean(ses_mean_preference_td)
    mean_preference_k = np.mean(ses_mean_preference_k)
    if to_plot:
        plt.figure(to_plot)
        plt.clf()
        if kernels:
            plt.bar([1,2,3],[mean_preference_mb, mean_preference_td, mean_preference_k])        
            plt.xlim(0.8,4)
            plt.xticks([1.4, 2.4, 3.4], ['Model based', 'Model free', 'kernels'])
        else:
            plt.bar([1,2],[mean_preference_mb, mean_preference_td])        
            plt.xlim(0.8,3)
            plt.xticks([1.4, 2.4], ['Model based', 'Model free'])

        plt.ylabel('Mean abs. preference')
    else:
        return(mean_preference_mb, mean_preference_td)


def parameter_autocor(sessions, population_fit, param = 'side'):
    ''' Evaluate within and cross subject variability in 
    specified parameter and autocorrelation across sessions.
    '''

    assert len(population_fit['MAP_fits']) == len(sessions), \
        'Population fit does not match number of sessions.'

    sp_index = population_fit['param_names'].index(param)
    for i, MAP_fit in enumerate(population_fit['MAP_fits']):
        sessions[i].side_loading = MAP_fit['params_U'][sp_index]

    sIDs = list(set([s.subject_ID for s in sessions]))

    plt.figure(1)
    plt.clf()
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


def eval_calibration(sessions, agent, population_fit, use_MAP = True, trial_mask = None,
                     n_bins = 10, fixed_widths = False, to_plot = False,
                     model_output = 'choice_probs', orthog_against = None):
    '''Caluculate real choice probabilities as function of model choice probabilities
    or model action value differences.'''

    assert model_output in ['choice_probs', 'P_net', 'P_mb', 'P_td']

    session_fits = population_fit['MAP_fits']

    assert len(session_fits[0]['params_T']) == agent.n_params, 'agent n_params does not match population_fit.'
    assert len(sessions) == len(session_fits), 'Number of fits does not match number of sessions.'
    assert population_fit['agent_name'] == agent.name, 'Agent name different from that used for fits.'

    # Create arrays containing model choice probabilites and true choices for each trial.
    session_choices, session_model_prediction = ([],[])
    for fit, session in zip(session_fits, sessions):
        if use_MAP:
            params_T = fit['params_T']
        else:
            params_T = ru.sample_params_T_from_pop_params(population_fit['pop_params'], agent) 
        session_choices.append(session.CTSO['choices'].tolist())
        model_time_series = agent.session_likelihood(session, params_T, return_trial_data = True)
        if orthog_against is not None:
            session_model_prediction.append(model_time_series[model_output] - 
                ut.projection(model_time_series[orthog_against], model_time_series[model_output]))
        else:
            session_model_prediction.append(model_time_series[model_output])
    
    choices = np.hstack(session_choices)

    if model_output == 'choice_probs':  
        model_prediction = np.vstack(session_model_prediction)[:, 1]
    else:
        model_prediction = np.concatenate(session_model_prediction)

    

    if trial_mask is not None: # Select subset of trials using binary mask.
        choices = choices[trial_mask]
        model_prediction = model_prediction[trial_mask]
    # Calculate true vs model choice probs.
    true_probs  = np.zeros(n_bins)
    model_probs = np.zeros(n_bins)
    if fixed_widths: # Bins of equal width in model choice probability.
        bin_edges = np.linspace(0, 1, n_bins + 1)
    else: # Bins of equal trial number.
        choices = choices[np.argsort(model_prediction)]
        model_prediction.sort()
        bin_edges = model_prediction[np.round(np.linspace(0,len(model_prediction) - 1, n_bins + 1)).astype(int)]
        bin_edges[0] = bin_edges[0] - 1e-6 
    for b in range(n_bins):
        true_probs[b] = np.mean(choices[np.logical_and(
                            bin_edges[b] < model_prediction,
                            model_prediction <= bin_edges[b + 1])])
        model_probs[b] = np.mean(model_prediction[np.logical_and(
                            bin_edges[b] < model_prediction,
                            model_prediction <= bin_edges[b + 1])])
    
    calibration = {'true_probs': true_probs, 'model_prediction': model_probs}
    
    if to_plot: 
        calibration_plot(calibration) 
    
    if model_output == 'choice_probs':
        print(('Fraction correct: {}'.format(sum((model_prediction > 0.5) == choices.astype(bool)) / len(choices))))
        chosen_probs = np.hstack([model_prediction[choices == 1], 1. - model_prediction[choices == 0]])
        print(('Geometric mean choice prob: {}'.format(np.exp(np.mean(np.log(chosen_probs))))))
    
    return calibration

    
def calibration_plot(calibration, clf = True):
    if 'calibration' in list(calibration.keys()): #Allow population_fit to be passed in.
        calibration = calibration['calibration']
    plt.figure(1)
    if clf:plt.clf()
    plt.plot(calibration['true_probs'], calibration['model_prediction'], 'o-')
    plt.plot([0,1],[0,1],'k',linestyle =':')
    plt.xlabel('True choice probability')
    plt.ylabel('Model choice probability')







