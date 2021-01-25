import numpy as np
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

# Fitting ------------------------------------------------------------

def fit_session(session, agent, repeats=10, use_prior=False):
    '''ML or MAP fit of session using constrained optimisation.'''
    if use_prior:
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
    return {'agent_name' : agent.name,            
            'param_names': agent.param_names,
            'param_ranges': agent.param_ranges,
            'n_params'   : agent.n_params,
            'params'     : fit['x'],
            'loglik'     : loglik, 
            'logpostprob': logpostprob,
            'n_trials'   : n_trials,
            'BIC'        : -2*loglik + np.log(n_trials)*agent.n_params,
            'AIC'        : -2*loglik + 2*agent.n_params}

def _neg_log_likelihood(params, session, agent): 
    return -agent.session_likelihood(session, params)

def _neg_log_posterior_prob(params, session, agent):
    loglik = agent.session_likelihood(session, params)
    priorprob = _log_prior_prob(params, agent)
    return -loglik - priorprob

def _log_prior_prob(params, agent):
    priorprobs = np.hstack((
        beta_prior.logpdf( params[np.array([r=='unit' for r in agent.param_ranges])]),
        gamma_prior.logpdf(params[np.array([r=='pos' for r in agent.param_ranges])]),
        norm_prior.logpdf( params[np.array([r=='unc' for r in agent.param_ranges])])))
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