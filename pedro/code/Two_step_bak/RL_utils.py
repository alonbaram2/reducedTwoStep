'''Utility functions and classes for Reinforcement learning modelling.'''

import numpy as np
import random
import math
from scipy.misc import derivative
import sys

log_max_float = np.log(sys.float_info.max/2.1) # Log of largest possible floating point number.

def softmax(Q,T):
    "Softmax choice probs given values Q and inverse temp T."
    QT = Q * T
    QT[QT > log_max_float] = log_max_float # Protection against overflow in exponential.    
    expQT = np.exp(QT)
    return expQT/expQT.sum()

def array_softmax(Q,T):
    '''Array based calculation of softmax probabilities for binary choices.
    Q: Action values - array([n_trials,2])
    T: Inverse temp  - float.'''
    P = np.zeros(Q.shape)
    TdQ = -T*(Q[:,0]-Q[:,1])
    TdQ[TdQ > log_max_float] = log_max_float # Protection against overflow in exponential.    
    P[:,0] = 1./(1. + np.exp(TdQ))
    P[:,1] = 1. - P[:,0]
    return P

def softmax_bin(Q,T):
    '''Faster calculation of softmax for binary choices.'''
    P = 1./(1. + np.exp(-T*(Q[0]-Q[1])))
    return np.array([P, 1. - P])

def logistic(Q):
    'Logistic functionl used in logistic regression.'
    Q[Q < -log_max_float] = -log_max_float
    return 1./(1. + np.exp(-Q))

def choose(P):
    "Takes vector of probabilities P summing to 1, returns integer s with prob P[s]"
    return sum(np.cumsum(P)<np.random.rand(1))

def sigmoid(x):return 1./(1.+np.exp(-x))

def sigmoid_grad(x): 
    'return dy/dx for y = sigmoid(x)'
    return np.exp(x)/((np.exp(x)+1)**2)

def inverse_sigmoid(y):return -np.log((1./y)-1)

def inv_sigmoid_grad(y):
    'return dx/dy for x = inverse_sigmoid(y)'
    return 1./(y-y**2)

def protected_log(x):
    'Return log of x protected against giving -inf for very small values of x.'
    return np.log(((1e-200)/2)+(1-(1e-200))*x)

def random_params(param_ranges, return_T = False):
        if param_ranges[0] == 'all_unc':
            return np.random.normal(size = param_ranges[1])
        params_T = np.zeros(len(param_ranges))
        for i, r in enumerate(param_ranges):
            if r == 'unit':
                params_T[i] = 0.001 + 0.998 * random.random() # Uniformly distributed between 0 and 1.
            elif r == 'pos':
                params_T[i] = math.exp(random.random() * 3)
            elif r == 'unc':
                params_T[i] = random.normalvariate(0,1)   # Normally distributed with mean = 1 and SD = 1.
        if return_T:
            return params_T
        else:
            return trans_TU(params_T, param_ranges)

def Hess_diag(fun, x, dx = 1e-4):
    '''Evaluate the diagonal elements of the hessian matrix using the 3 point
    central difference formula with spacing of dx between points.'''
    n = len(x)
    v = fun(x)
    hessdiag = np.zeros(n)
    for i in range(n):
        x_ph = np.zeros(n)
        x_ph[i] += dx
        x_ph += x
        x_mh = np.zeros(n)
        x_mh[i] -= dx
        x_mh += x
        hessdiag[i] = (fun(x_ph) - 2. * v + fun(x_mh)) / (dx ** 2)
    return hessdiag

def smooth_abs(x,a):
    'smooth aproximation to the absolute value of x.'
    return (1./a)*(np.log(1 + np.exp(-a*x)) + np.log(1 + np.exp(a*x)))

def smooth_l1(x, a):
    'smooth aproximation to the L1 norm of x.'
    return np.sum(smooth_abs(x,a))

def smooth_l1_grad(x,a):
    'Grad of smooth aproximation to the L1 norm.'
    return (1./(1. + np.exp(-a*x)) - 1./(1. + np.exp(a*x)))

def sample_params_T_from_pop_params(pop_params, agent):
    ''' Draw a sample of paramter values in true space given population
    level prior distribution.
    '''
    sample_params_U = np.random.normal(pop_params['means'], pop_params['SDs'])
    sample_params_T = trans_UT(sample_params_U, agent.param_ranges)
    return sample_params_T

# -------------------------------------------------------------------------------------
# Transformations between unconstrained and transformed space.
# -------------------------------------------------------------------------------------

def trans_UT(values_U, param_ranges):
    'Transform parameters from unconstrained to true space.'
    if param_ranges[0] == 'all_unc':
        return values_U
    values_T = []
    for value, rng in zip(values_U, param_ranges):
        if rng   == 'unit':
            if value < -16.:
                value = -16.
            values_T.append(1./(1. + math.exp(-value)))  # Don't allow values smaller than 1e-7
        elif rng == 'pos':
            if value > 16.:
                value = 16.
            values_T.append(math.exp(value))  # Don't allow values bigger than ~ 1e7.
        elif rng == 'unc':
            values_T.append(value)
    return np.array(values_T)

def trans_TU(values_T, param_ranges):
    'Transform parameters from true to unconstrained space.'
    if param_ranges[0] == 'all_unc':
        return values_T
    values_U = []
    for value, rng in zip(values_T, param_ranges):
        if rng   == 'unit':
            values_U.append(-math.log((1./value)-1))
        elif rng == 'pos':
            values_U.append(math.log(value))
        elif rng == 'unc':
            values_U.append(value)
    return np.array(values_U)

def multi_trans_UT(multi_params, param_ranges):
    ''' Converts 2d array or list of lists of params from unconstrained 
    to true space. '''
    if param_ranges[0] == 'all_unc':
        return multi_params
    if isinstance(multi_params,np.ndarray):
        multi_params = multi_params.tolist()
    param_array_T = np.array([trans_UT(params_U, param_ranges) 
                              for params_U in multi_params])
    return param_array_T

def trans_grad_TU(values_T, gradients_T, param_ranges):
    'Transform gradient wrt paramters from true to unconstrained space.'
    if param_ranges[0] == 'all_unc':
        return gradients_T
    gradients_U = []
    for x, dLdx, rng in zip(values_T, gradients_T, param_ranges):
        if rng   == 'unit':
            gradients_U.append(x * (1-x) * dLdx)
        elif rng == 'pos':
            gradients_U.append(x * dLdx)
        elif rng == 'unc':
            gradients_U.append(dLdx)
    return np.array(gradients_U)


