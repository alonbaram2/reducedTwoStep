import numpy as np
import pylab as plt
import seaborn as sns
import pandas as pd
from scipy.stats import ttest_rel

# Configure plotting defaults>

plt.rcParams['pdf.fonttype'] = 42
plt.rc("axes.spines", top=False, right=False)

# Plot fits. ---------------------------------------------------

def plot_fits(fits):
    '''Plot a set of fits, grouped by training stage and 
    plotting paramters with different ranges on seperate axes.
    Prints paired t-test P values for differences in paramteres
    between training stages.'''

    def plot_selected_params(rng, ax):
        '''Plot those paramters whose range is rng on axis ax.'''
        params = fits.params[fits.param_ranges==rng].dropna(axis=1)
        data = pd.concat([fits.sessions,  params], axis=1)
        sns.boxplot(x='param', y='value', hue='sessions', ax=ax,
            palette='crest', fliersize=0,
            data=data.melt(id_vars='sessions', var_name='param'))
        sns.stripplot(x='param', y='value', hue='sessions', ax=ax, 
            palette='crest', edgecolor='gray', linewidth=1, dodge=True, 
            data=data.melt(id_vars='sessions', var_name='param'))
    
    fig, [ax1, ax2, ax3] = plt.subplots(1,3, figsize=(9,3))
    plot_selected_params('pos' , ax1)
    ax1.set_ylim(ymin=0)
    ax1.legend_.remove()
    plot_selected_params('unit', ax2)
    ax2.set_ylim(0,1)
    ax2.legend_.remove()
    plot_selected_params('unc' , ax3)
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.tight_layout()

    # Test for significant differences between stages.

    stages = fits.sessions.unique().tolist()
    p_values = pd.DataFrame(index=fits.params.columns)
    for i in range(len(stages)-1):
        p_values[f'{stages[i]} vs {stages[i+1]}'] = ttest_rel(
            fits.params[fits.sessions==stages[i]], 
            fits.params[fits.sessions==stages[i+1]]).pvalue
    print('Paired t-test P values:')
    print(p_values)

# Plot action values. -----------------------------------------------

def plot_action_values(session, agent, fit):
    '''Plot action values for a given session agent and fit.'''
    Q_tot, Q_mf, Q_mb, Q_k =  agent.return_values(session, fit)
    plt.figure(figsize=[9,3], clear=True)
    x = np.arange(Q_mf.shape[1])+1
    plt.plot(x, Q_mf[0,:], label='Q_mf bottom')
    plt.plot(x, Q_mf[1,:], label='Q_mf top')
    plt.plot(x, Q_mb[0,:], label='Q_mb bottom')
    plt.plot(x, Q_mb[1,:], label='Q_mb top')
    plt.xlim(0, x[-1])
    plt.xlabel('Trial')
    plt.ylabel('Value')
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
