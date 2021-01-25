import os
import pandas as pd
import numpy as np

# Load subject ----------------------------------------------------------

def load_subject(subject_ID, data_path):
    '''Load all data from one subject and return as a single dataframe.'''

    print(f'Loading subject ID: {subject_ID :02d}')

    session_dfs = []

    for session_n in range(1,13):
        file_name = f'sub-{subject_ID:02d}_task-simple2step_run-{session_n:02d}_events.tsv'

        # Import data.
        import_df = pd.read_csv(os.path.join(data_path, file_name), sep='\t')

        # Make data frame of sessions choices, second-steps and outcomes.
                
        choices  = import_df['choice1'].dropna().to_numpy(int) # True if high, flase if low.
        states   = import_df['state'  ].dropna().to_numpy(int)[::2] # True if left, false if right.     
        outcomes = import_df['reward' ].dropna().to_numpy(int) # True if rewarded,  flase if no reward.

        trial_n = np.arange(len(choices)) + 1

        session_dfs.append(pd.DataFrame({'session_n': session_n, 
                                        'trial_n'  : trial_n,
                                        'choice'   : choices, 
                                        'state'    : states, 
                                        'outcome'  : outcomes}))

    return pd.concat(session_dfs, ignore_index=True)