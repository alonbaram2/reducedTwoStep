import numpy as np
import time
from copy import deepcopy
import pylab as plt
from functools import partial
from multiprocessing import Pool
from . import RL_utils as ru
from . import RL_plotting as rp
from . import plotting as pl
from . import session as ss
from . import utility as ut
from . import model_fitting as mf

# -------------------------------------------------------------------------------------
# Model comparison.
# -------------------------------------------------------------------------------------

def BIC_model_comparison(population_fits):
    ''' Compare goodness of different fits using integrated BIC'''    
    sorted_fits = sorted(population_fits, key = lambda fit: fit['BIC_score'])
    print('BIC_scores:')
    for fit in sorted_fits:
        print('{} : '.format(round(fit['BIC_score'])) + fit['agent_name'])
    print('The best fitting model is: ' + sorted_fits[0]['agent_name'])
 


def model_comparison_robustness(sessions, agents, task, n_eval = 100, n_sim = 100):
    ''' Model comparison includeing an estimation of how robust is the conlusion about
    which model is best.
    The approach taken is as follows:
    1. Evaluate the quality off fit of the models to the data provided using the
    specified metric (e.g. BIC score)
    2. Using the best fitting model generate a population of simulated datasets, each of
    which is the same size as the real dataset.
    3. Fit all model to each simulated dataset and evaluate the BIC scores for the fit.
    4. Plot the distibutions of BIC scores for each model, and the distribution of BIC 
    score difference between the best fitting model and each other model.
    '''
    print('Fitting real data.')
    model_fits = [mf.fit_population(sessions, agent, eval_BIC = n_eval) for agent in agents]
    best_agent_n = np.argmin([fit['BIC_score'] for fit in model_fits])
    best_agent = agents[best_agent_n]
    best_agent_fit =  model_fits[best_agent_n]
    simulated_datasets = []
    for i in range(n_sim):
        simulated_datasets.append(sim_sessions_from_pop_fit(task, best_agent,
                                                               best_agent_fit, use_MAP = False))

    # simulated_data_fits, i, n_fits = ([], 1, len(agents) * n_sim )
    # for agent in agents:
    #     agent_simdata_fits = []
    #     init_params = None 
    #     for sim_data in simulated_datasets:
    #         print('Simulated dataset fit {} of {}'.format(i, n_fits))
    #         agent_simdata_fits.append(mf.fit_population(sim_data, agent, 
    #                                   eval_BIC = n_eval, pop_init_params = init_params))
    #         init_params = agent_simdata_fits[-1]['pop_params'] 
    #         i += 1
    #     simulated_data_fits.append(agent_simdata_fits)

    fit_func = partial(fit_agent_to_sim_data, simulated_datasets = simulated_datasets, n_eval = n_eval)
    simulated_data_fits = mp_pool.map(fit_func, agents)

    mod_comp = {'agents'              : agents,
                'sessions'            : sessions,
                'task'                : task,
                'best_agent_n'        : best_agent_n,
                'model_fits'          : model_fits,
                'simulated_datasets'  : simulated_datasets,
                'simulated_data_fits' : simulated_data_fits}

    plot_BIC_dists(mod_comp)

    return mod_comp

def fit_agent_to_sim_data(agent, simulated_datasets, n_eval):

    agent_simdata_fits = []
    init_params = None 
    for sim_data in simulated_datasets:
        agent_simdata_fits.append(mf.fit_population(sim_data, agent, 
                                  eval_BIC = n_eval, pop_init_params = init_params))
        init_params = agent_simdata_fits[-1]['pop_params'] 
    return agent_simdata_fits


def plot_BIC_dists(mod_comp, n_bins = 100):
    'Plot results of model comparison.'
    agents = mod_comp['agents']

    sim_data_BIC_scores = np.array([[fit['BIC_score'] for fit in agent_simdata_fits] for 
                                     agent_simdata_fits in mod_comp['simulated_data_fits']])


    BIC_diffs = sim_data_BIC_scores - np.tile(sim_data_BIC_scores[mod_comp['best_agent_n'],:],(len(agents),1))

    BIC_score_range = (sim_data_BIC_scores.min() - 1, sim_data_BIC_scores.max() + 1)
    BIC_diffs_range = (BIC_diffs.min() - 1, BIC_diffs.max() + 1)
    cols = plt.cm.rainbow(np.linspace(0,1,len(agents)))
    plt.figure(1)
    plt.clf()
    plt.subplot(2,1,1)
    for i, agent in enumerate(agents):
        plt.hist(sim_data_BIC_scores[i,:], n_bins, BIC_score_range, color = cols[i],
               histtype='stepfilled', alpha = 0.5, label= agent.name)
    y_lim = plt.ylim()
    for i, agent in enumerate(agents):
        plt.plot([mod_comp['model_fits'][i]['BIC_score']],[y_lim[1]/2.],'o', color = cols[i])
    plt.ylim(np.array(y_lim)*1.1)
    plt.legend()
    plt.xlabel('BIC score')
    plt.subplot(2,1,2)
    for i, agent in enumerate(agents):
        if not BIC_diffs[i,0] == 0:
            plt.hist(BIC_diffs[i,:], n_bins, BIC_diffs_range, color = cols[i],
                   histtype='stepfilled', alpha = 0.5, label= agent.name)
    plt.ylim(np.array(plt.ylim())*1.1)
    plt.xlabel('BIC score difference')


def plot_fit_consistency(population_fits, plot_true = True, fig_no = 1):

    fit_means = np.array([pf['pop_params']['means'] for pf in population_fits])
    fit_SDs = np.array([pf['pop_params']['SDs'] for pf in population_fits])
    true_means = population_fits[0]['pop_params_true']['means']
    true_SDs = population_fits[0]['pop_params_true']['SDs']
    n_params = fit_means.shape[1]
    n_fits = fit_means.shape[0]
    x = np.arange(n_fits)/n_fits
    ymin = np.min(fit_means - fit_SDs) - 0.2
    ymax = np.max(fit_means + fit_SDs) + 0.2

    if not len(true_means) == n_params:
        plot_true = False

    plt.figure(fig_no)
    plt.clf()

    for i in range(n_params):
        plt.subplot(1, n_params, i + 1)
        if plot_true:
            plt.plot([0.45,0.45], [true_means[i] - true_SDs[i], true_means[i] + true_SDs[i]], 'r', linewidth = 2)         
        for f in range(n_fits):
            plt.plot([x[f],x[f]], [fit_means[f,i] - fit_SDs[f,i], fit_means[f,i] + fit_SDs[f,i]],'b')
            plt.locator_params(axis = 'y', nbins = 4)
            plt.xticks([])
            plt.xlabel(population_fits[0]['param_names'][i])
            plt.ylim(ymin,ymax)


def test_population_fitting(task, agent, n_sessions = 8, n_trials = 1000, pop_params = None):
    '''Simulate a set of sessions using parameters drawn from normal distributions
    specified by pop_params.  Then fit the agent model to the simulated data and plot
    correspondence between true and fitted paramter values.
    '''
    sessions = simulate_sessions(task, agent, n_sessions, n_trials, pop_params)
    ML_fits, MAP_fits, pop_params = mf.fit_population(sessions, agent, max_iter = 15)
    rp.plot_true_fitted_params(sessions, ML_fits, MAP_fits)
    return (sessions, ML_fits, MAP_fits, pop_params)