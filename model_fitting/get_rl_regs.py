import pandas as pd
from os.path import dirname, abspath, join
import MF_MB_agent
import RL_utils as ru

data_dir = join(dirname(dirname(abspath(__file__))), 'bhv_data')
data = pd.read_pickle('data.pkl')
fits = pd.read_pickle('fits.pkl')

for subject_ID in range(1,29):
    
    fits_1_4 = fits[(fits['subject']==subject_ID) & (fits['sessions']=='1-4')]
    fits_5_8 = fits[(fits['subject']==subject_ID) & (fits['sessions']=='5-8')]
    fits_9_12 = fits[(fits['subject']==subject_ID) & (fits['sessions']=='9-12')]
    
    params_1_4 = fits_1_4['params'].to_numpy().flatten()
    params_5_8 = fits_5_8['params'].to_numpy().flatten()
    params_9_12 = fits_9_12['params'].to_numpy().flatten()
    
    sessions_1_4  = data[(data['session_n'].isin(range(1,5))) & (data['subject']==subject_ID)]
    sessions_5_8  = data[(data['session_n'].isin(range(5,9))) & (data['subject']==subject_ID)]
    sessions_9_12 = data[(data['session_n'].isin(range(9,13))) & (data['subject']==subject_ID)]
    
    choices_1_4, second_steps_1_4, outcomes_1_4 = ru.unpack_trial_data(sessions_1_4)
    choices_5_8, second_steps_5_8, outcomes_5_8 = ru.unpack_trial_data(sessions_5_8)
    choices_9_12, second_steps_9_12, outcomes_9_12 = ru.unpack_trial_data(sessions_9_12)
    
    Q_tot_1_4, Q_MF_1_4, Q_MB_1_4, K_1_4, PE_1st_MF_1_4, PE_1st_MB_1_4, PE_2nd_1_4 = MF_MB_agent.compute_values(choices_1_4, second_steps_1_4, outcomes_1_4, params_1_4)    
    Q_tot_5_8, Q_MF_5_8, Q_MB_5_8, K_5_8, PE_1st_MF_1_4, PE_1st_MB_5_8, PE_2nd_5_8 = MF_MB_agent.compute_values(choices_5_8, second_steps_5_8, outcomes_5_8, params_5_8)    
    Q_tot_9_12, Q_MF_9_12, Q_MB_9_12, K_9_12, PE_1st_MF_9_12, PE_1st_MB_9_12, PE_2nd_9_12 = MF_MB_agent.compute_values(choices_9_12, second_steps_9_12, outcomes_9_12, params_9_12)    
    