import pandas as pd
from os.path import dirname, abspath, join

import MF_MB_agent
import data_import as di
import model_fitting as mf
import model_plotting as mp

# Load data.
data_dir = join(dirname(dirname(abspath(__file__))), 'bhv_data')

try: # Load data from .pkl if available.
    data = pd.read_pickle('data.pkl')
    print('Loaded data from data.pkl')
except FileNotFoundError: # Load data from data dir files.
    data = di.load_experiment(data_dir) 
    data.to_pickle('data.pkl') # Save data as pkl for faster future loading.

sessions_1_4  = data[data['session_n'].isin(range(1,5))]
sessions_5_8  = data[data['session_n'].isin(range(5,9))]
sessions_9_12 = data[data['session_n'].isin(range(9,13))]

# Compute RL model fits.

try: # Load previously saved fits if available. 
    fits = pd.read_pickle('fits.pkl')
    print('Loaded fits from fits.pkl')
except FileNotFoundError: # Fit RL model to data and save fits.
    fits_1_4  = mf.fit_subjects(sessions_1_4, MF_MB_agent, repeats=10, use_prior=True)
    fits_5_8  = mf.fit_subjects(sessions_5_8, MF_MB_agent, repeats=10, use_prior=True)
    fits_9_12 = mf.fit_subjects(sessions_9_12, MF_MB_agent, repeats=10, use_prior=True)
    fits = pd.concat([fits_1_4, fits_5_8, fits_9_12], ignore_index=True)
    fits.to_pickle('fits.pkl') # Save fits as pkl.

# Plot fits for all subjects across training stages.

mp.plot_fits(fits)

# Plot action values for one subject over sessions 1-4.

subject_data =  data[(data['session_n'].isin(range(1,5))) & (data['subject']==2)]
subject_fit = mf.fit_session(subject_data, MF_MB_agent)
mp.plot_action_values(subject_data, MF_MB_agent, subject_fit)