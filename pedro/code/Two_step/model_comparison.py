import numpy as np
import pylab as plt
from functools import partial
from collections import Counter
from scipy.stats import sem
from sklearn.utils import resample
from scipy.stats import chi2
from . import utility as ut
from . import model_plotting as mp
from . import model_fitting as mf
from . import simulation as sm
from . import RL_agents as rl
from . import parallel_processing as pp

def BIC_model_comparison(sessions, agents, n_draws=1000, n_repeats=1, fig_no=1,
                         file_name=None, log_Y=False):
    ''' Compare goodness of different fits using integrated BIC.'''    
    if n_repeats > 1: 
        fit_func = partial(mf.repeated_fit_population, sessions, n_draws=n_draws, n_repeats=n_repeats)
        fiterator = map(fit_func, agents) # Save parallel processessing for repeated fits of same agent.
    else:
        fit_func = partial(mf.fit_population, sessions, eval_BIC={'n_draws':n_draws})
        fiterator = pp.imap(fit_func, agents, ordered=False) # Use parallel processing for seperate agents.
    population_fits = []
    for i,fit in enumerate(fiterator):
        print('Fit {} of {}, agent: '.format(i+1, len(agents)) + fit['agent_name'])
        population_fits.append(fit)
        if file_name: ut.save_item(population_fits, file_name)
    BIC_comparison_plot(population_fits, fig_no, log_Y)
    return population_fits

def BIC_comparison_plot(population_fits, fig_no=1, log_Y=False, plot_rep_fits=False):
    '''Plot the results of a BIC model comparison'''
    sorted_fits = sorted(population_fits, key = lambda fit: fit['iBIC']['score'])
    print('BIC_scores:')
    for fit in sorted_fits:
        s =   '{:.3f} : '.format(fit['iBIC']['best_prob']) if 'best_prob' in fit['iBIC'].keys() else ''
        print('{:.0f} : '.format(round(fit['iBIC']['score'])) + s + fit['agent_name'])
    print('The best fitting model is: ' + sorted_fits[0]['agent_name'])
    if fig_no:
        BIC_scores = np.array([fit['iBIC']['score'] for fit in sorted_fits])
        BIC_deltas = BIC_scores - BIC_scores[0]
        agent_names = [fit['agent_name'] for fit in sorted_fits]
        x = np.flipud(np.arange(1,len(agent_names)+1))
        if 'BIC_95_conf' in fit['iBIC'].keys():
            ebars = np.array([np.abs(fit['iBIC']['BIC_95_conf'] - fit['iBIC']['score'])
                              for fit in sorted_fits]).T
        else: 
            ebars = -2*np.array([np.abs(fit['iBIC']['lik_95_conf'] - fit['iBIC']['int_lik'])
                                for fit in sorted_fits]).T
        plt.figure(fig_no).clf()
        plt.bar(x, BIC_deltas, color = 'k')
        plt.errorbar(x + 0.4, BIC_deltas, ebars, color = 'r', linestyle = '', elinewidth = 2)
        if 'repeated_fits' in population_fits[0].keys() and plot_rep_fits: # Scatter plot repeated fits.
            for fit, xi in zip(sorted_fits, x):
                rep_fit_iBICs = np.array([f['iBIC']['score'] for f in fit['repeated_fits']])
                plt.scatter(xi+0.4+np.linspace(-0.2,0.2,len(rep_fit_iBICs)), rep_fit_iBICs - BIC_scores[0])
        plt.xticks(x + 0.6/len(agent_names), agent_names, rotation = -45, ha = 'left')
        plt.xlim(0.75,len(agent_names)+1)
        plt.ylim(0, BIC_deltas[-1]*1.2)
        plt.ylabel('∆ BIC')
        plt.figtext(0.13,0.92,'Best BIC score: {}'.format(int(BIC_scores[0])))
        plt.tight_layout()
        if log_Y:
            plt.gca().set_yscale('log')
            plt.ylim(10,plt.ylim()[1])
            
def per_subject_BIC_comparison(sessions, agents, repeats=5, fig_no=1):
    'Model comparison on a per subject basis using non-hierarchical fit.'
    subject_IDs = list(set([s.subject_ID for s in sessions]))
    subject_sessions = [[s for s in sessions if s.subject_ID == sID] for sID in subject_IDs]
    fit_func = partial(_eval_BICs, agents=agents, repeats=repeats)
    sub_comps = {} # BIC score comparisons for each subjects sessions.
    for i, (sID,sub_comp) in enumerate(zip(subject_IDs, pp.imap(fit_func, subject_sessions, ordered=False))):
        print('Fitting subject {} of {}. '.format(i+1, len(subject_IDs)))
        sub_comps[sID] = sub_comp
    _best_agent_histogram(sub_comps, 'BIC', fig_no)
    return sub_comps

def _best_agent_histogram(sub_comps, metric='BIC', fig_no=1):
    'Histogram of the number of subjects for which each agent is the best fit.'
    assert metric in ('BIC', 'lik'), "metric argument must be 'BIC' or 'lik'."
    if metric =='BIC':
        best_agents = [min(sub_comp, key=lambda x: sub_comp[x]['BIC']) 
                       for sub_comp in sub_comps.values()]
    elif metric == 'lik':
        best_agents = [max(sub_comp, key=lambda x: sub_comp[x]['lik']) 
                       for sub_comp in sub_comps.values()]
    best_agent_counts = sorted(list(Counter(best_agents).items()), key = lambda x:x[1])
    plt.figure(fig_no).clf()
    x = np.arange(len(best_agent_counts))
    plt.bar(x, [c[1] for c in best_agent_counts])
    plt.xticks(x+0.5,[c[0] for c in best_agent_counts],rotation = -45)
    plt.ylim(0, best_agent_counts[-1][1]+1)
    plt.xlim(-0.2, x[-1]+1.2)
    plt.ylabel('Subjects best fit')


def _eval_BICs(sessions, agents, repeats=5):
    'Evaluate the BIC scores for each agent using flat (non-hierarchical) fits.'
    agent_comp = {}
    for agent in agents:
        session_fits = [mf.fit_session_con(session, agent, repeats=repeats) for session in sessions]
        log_lik = np.sum([fit['prob'] for fit in session_fits])
        BIC = -2*log_lik + agent.n_params*np.sum(np.sum([np.log(s.n_trials) for s in sessions]))
        agent_comp[agent.name] = {'fits': session_fits, 'lik': log_lik, 'BIC': BIC}
    return agent_comp

def likelihood_ratio_plot(sessions, kernels=['bs','ck'], repeats=20, fig_no=1, title=None):
    '''Scatter plot sessions showing the log likelihood ratios between:
    x-axis: Mixture agent and a kernels only agent with no reward driven learning.
    y-axis: Mixture agent and model-free agent.
    Sessions are colour codes:
    red  : sessions with no evidence of reward driven learning.
    black: session with  
    '''
    agent_KO = rl.KO(kernels) # Kernel only agent, no RL.
    agent_MF = rl.MF(kernels) # Model free agent.
    agent_MB = rl.MB(kernels) # Model free agent.
    agent_MX = rl.MF_MB(kernels) # Mixture agent.
    fits_KO = [mf.fit_session_con(session, agent_KO, repeats=repeats) for session in sessions] 
    fits_MF = [mf.fit_session_con(session, agent_MF, repeats=repeats) for session in sessions] 
    fits_MB = [mf.fit_session_con(session, agent_MB, repeats=repeats) for session in sessions] 
    fits_MX = [mf.fit_session_con(session, agent_MX, repeats=repeats) for session in sessions] 
    log_lik_KO = np.array([fit['prob'] for fit in fits_KO])
    log_lik_MF = np.array([fit['prob'] for fit in fits_MF])
    log_lik_MB = np.array([fit['prob'] for fit in fits_MB])
    log_lik_MX = np.array([fit['prob'] for fit in fits_MX])
    MF_KO_gain = log_lik_MF-log_lik_KO
    MX_KO_gain = log_lik_MX-log_lik_KO 
    MX_MF_gain = log_lik_MX-log_lik_MF
    MX_MB_gain = log_lik_MX-log_lik_MB
    for g in [MF_KO_gain, MX_KO_gain, MX_MF_gain, MX_MB_gain]:
        g[g<0] = 1e-6
    MF_KO_Pval = chi2.sf(MF_KO_gain*2,agent_MF.n_params - agent_KO.n_params) # Likelihood ratio test P values for MF agent better than KO agent.
    MX_KO_Pval = chi2.sf(MX_KO_gain*2,agent_MX.n_params - agent_KO.n_params) 
    MX_MF_Pval = chi2.sf(MX_MF_gain*2,agent_MX.n_params - agent_MF.n_params) 
    MX_MB_Pval = chi2.sf(MX_MB_gain*2,agent_MX.n_params - agent_MB.n_params) 
    KO_ses = (MF_KO_Pval>0.05) & (MX_KO_Pval>0.05) # Sessions with no evidence of reward sensitivity.
    MF_ses = ~KO_ses & (MX_MF_Pval>0.05) # Sessions with model free RL only.
    MX_ses = ~KO_ses & (MX_MF_Pval<0.05)
    MB_ses = ~KO_ses & (MX_MF_Pval<0.05) & (MX_MB_Pval>0.05)
    plt.figure(fig_no).clf()
    plt.scatter(MX_KO_gain[KO_ses], MX_MF_gain[KO_ses], color='r') 
    plt.scatter(MX_KO_gain[MF_ses], MX_MF_gain[MF_ses], color='k') 
    plt.scatter(MX_KO_gain[MX_ses], MX_MF_gain[MX_ses], color='b') 
    plt.scatter(MX_KO_gain[MB_ses], MX_MF_gain[MB_ses], color='g')
    plt.xlabel('(log) likelihood ratio: MF+MB / no-reward-learning')
    plt.ylabel('(log) likelihood ratio: MF+MB / MF')
    plt.ylim(-2,plt.ylim()[1])
    plt.xlim(-2,plt.xlim()[1])
    if title:plt.title(title)

#---------------------------------------------------------------------------------------
# Model calibration - evaluate real vs predicted choice probabilities.
#---------------------------------------------------------------------------------------

def eval_calibration(sessions, agent, population_fit, use_MAP=True, n_bins=10,
                     fixed_widths=False, to_plot=False):
    '''Caluculate real choice probabilities as function of model choice probabilities.'''

    session_fits = population_fit['session_fits']

    assert len(session_fits[0]['params_T']) == agent.n_params, 'agent n_params does not match population_fit.'
    assert len(sessions) == len(session_fits), 'Number of fits does not match number of sessions.'
    assert population_fit['agent_name'] == agent.name, 'Agent name different from that used for fits.'

    # Create arrays containing model choice probabilites and true choices for each trial.
    session_choices, session_choice_probs = ([],[])
    for fit, session in zip(session_fits, sessions):
        if use_MAP:
            params_T = fit['params_T']
        else:
            params_T = mf._sample_params_T(population_fit) 
        session_choices.append(session.trial_data['choices'])
        DVs = agent.session_likelihood(session, params_T, get_DVs = True)
        session_choice_probs.append(DVs['choice_probs'])

    choices = np.hstack(session_choices)
    choice_probs = np.hstack(session_choice_probs)[1,:]

    # Calculate true vs model choice probs.
    true_probs  = np.zeros(n_bins)
    model_probs = np.zeros(n_bins)
    if fixed_widths: # Bins of equal width in model choice probability.
        bin_edges = np.linspace(0, 1, n_bins + 1)
    else: # Bins of equal trial number.
        choices = choices[np.argsort(choice_probs)]
        choice_probs.sort()
        bin_edges = choice_probs[np.round(np.linspace(0,len(choice_probs)-1,
                                 n_bins+1)).astype(int)]
        bin_edges[0] = bin_edges[0] - 1e-6 
    for b in range(n_bins):
        true_probs[b] = np.mean(choices[np.logical_and(
                            bin_edges[b] < choice_probs,
                            choice_probs <= bin_edges[b + 1])])
        model_probs[b] = np.mean(choice_probs[np.logical_and(
                            bin_edges[b] < choice_probs,
                            choice_probs <= bin_edges[b + 1])])
    
    calibration = {'true_probs': true_probs, 'choice_probs': model_probs}
    
    if to_plot: 
        calibration_plot(calibration, fig_no = to_plot) 

    print(('Fraction correct: {}'.format(sum((choice_probs > 0.5) == choices.astype(bool)) / len(choices))))
    chosen_probs = np.hstack([choice_probs[choices == 1], 1. - choice_probs[choices == 0]])
    print(('Geometric mean choice prob: {}'.format(np.exp(np.mean(np.log(chosen_probs))))))
    
    return calibration

def calibration_plot(calibration, clf=True, fig_no=1):
    if 'calibration' in list(calibration.keys()): #Allow population_fit to be passed in.
        calibration = calibration['calibration']
    plt.figure(fig_no)
    if clf:plt.clf()
    plt.plot(calibration['true_probs'], calibration['choice_probs'], 'o-')
    plt.plot([0,1],[0,1],'k',linestyle =':')
    plt.xlabel('True choice probability')
    plt.ylabel('Model choice probability')