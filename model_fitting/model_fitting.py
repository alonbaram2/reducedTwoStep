import numpy as np
import pandas as pd
import pylab as plt
import scipy.optimize as op
from scipy.stats import gamma, beta, norm
from functools import partial

# Prior distributions -----------------------------------------------

beta_prior  = beta(a=2, b=2)      # Prior for unit range parameters.
gamma_prior = gamma(a=2, scale=2) # Prior for positive range parameters.
norm_prior  = norm(scale=2)       # Prior for unconstrained range paramters.

def plot_priors():
    '''Plot the prior distribution PDFs.'''
    plt.figure(1, figsize=[9,3], clear=True)
    plt.subplot(1,3,1)
    x = np.arange(0,1,0.01)
    plt.plot(x, beta_prior.pdf(x))
    plt.ylim(ymin=0)
    plt.title('Unit')
    plt.subplot(1,3,2)
    x = np.arange(0,10,0.1)
    plt.plot(x, gamma_prior.pdf(x))
    plt.ylim(ymin=0)
    plt.title('Positive')
    plt.subplot(1,3,3)
    x = np.arange(-5,5,0.1)
    plt.plot(x, norm_prior.pdf(x))
    plt.ylim(ymin=0)
    plt.title('Unconstrained')
    plt.tight_layout()

# Fit session -------------------------------------------------------

def fit_session(session, agent, repeats=10, use_prior=True):
    '''ML or MAP fit of session using constrained optimisation.'''
    if use_prior: #
        fit_func = partial(_neg_log_posterior_prob, session=session, agent=agent)
    else:
        fit_func =  partial(_neg_log_likelihood, session=session, agent=agent)
    bounds = [{'unc':(None, None), 'unit':(0.,1.), 'pos':(0,None)}[param_range]
              for param_range in agent.param_ranges]
    fits = []
    for r in range(repeats): # Number of fits to perform with different starting conditions.
        fits.append(op.minimize(fit_func, _get_init_params(agent.param_ranges), method='L-BFGS-B', 
                                bounds=bounds, options={'disp': True}))
    fit = fits[np.argmin([f['fun'] for f in fits])]  # Select best fit out of repeats.
    if use_prior:
        logpostprob = - fit['fun']
        loglik = agent.session_likelihood(session, fit['x'])
    else:
        logpostprob = None
        loglik = - fit['fun']
    n_trials = len(session)
    # Return fit as single row dataframe.
    info_df = pd.DataFrame({
        'subject'     : [int(session['subject'].unique())],
        'sessions'    : [ f"{session['session_n'].iloc[0]}-{session['session_n'].iloc[-1]}"],
        'agent'       : [agent.name],
        'loglik'      : [loglik],
        'logpostprob' : [logpostprob],
        'BIC'         : [-2*loglik + np.log(n_trials)*agent.n_params],
        'AIC'         : [-2*loglik + 2*agent.n_params]})
    info_df.columns = pd.MultiIndex.from_tuples([(c, '') for c in info_df])
    fitted_params = pd.DataFrame(
        {('params',n):[v] for n,v in zip(agent.param_names, fit['x'])})
    param_ranges = pd.DataFrame(
        {('param_ranges',n):[r] for n,r in zip(agent.param_names, agent.param_ranges)})
    return pd.concat([info_df, fitted_params, param_ranges], axis=1)

def _neg_log_likelihood(params, session, agent): 
    return -agent.session_likelihood(session, params)

def _neg_log_posterior_prob(params, session, agent):
    loglik = agent.session_likelihood(session, params)
    priorprob = _log_prior_prob(params, agent.param_ranges)
    return -loglik - priorprob

def _log_prior_prob(params, param_ranges):
    '''Return the log prior probability of a set of parameters.'''
    priorprobs = np.hstack((
        beta_prior.logpdf( params[np.array([r=='unit' for r in param_ranges])]),
        gamma_prior.logpdf(params[np.array([r=='pos'  for r in param_ranges])]),
        norm_prior.logpdf( params[np.array([r=='unc'  for r in param_ranges])])))
    priorprobs[priorprobs<-1000] = -1000 # Protect against -inf.
    return np.sum(priorprobs)

def _get_init_params(param_ranges):
    ''' Get initial parameters by sampling from prior probability distributions.'''
    params = []
    for rng in param_ranges:
        if rng == 'unit':
            params.append(beta_prior.rvs())
        elif rng == 'pos':
            params.append(gamma_prior.rvs())
        elif rng == 'unc':
            params.append(norm_prior.rvs())
    return np.array(params)

# Fit subjects -------------------------------------------------------

def fit_subjects(data_df, agent, repeats=10, use_prior=True):
    '''Fit each subject individually and return data frame with subject fits.'''
    subjects = data_df.subject.unique()
    fit_dfs = []
    for subject in subjects:
        print(f'Fitting subject: {subject}')
        subject_df = data_df[data_df['subject'] == subject]
        fit_dfs.append(fit_session(subject_df, agent, repeats, use_prior))
    return pd.concat(fit_dfs, ignore_index=True)