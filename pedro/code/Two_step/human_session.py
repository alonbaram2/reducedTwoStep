
import numpy as np
import os
from . import utility as ut
from . import plotting as pl
from .session import session

class human_session(session):
    'Class containing data from a human session.'
    def __init__(self, file_name, data_path, IDs):
        # Session information.
        print(file_name)
        self.file_name  = file_name
        self.subject_ID =  int(file_name.split('_')[0])
        self.date = file_name.split('_')[3]
        self.IDs = IDs
        self.number = int(file_name.split('_')[1][7:])

        # Import data.
        with open(os.path.join(data_path, file_name), 'r') as data_file:
            data_lines = [line.strip() for line in data_file if line[0].isdigit()]

        #  Extract time stamps and data strings.

        time_stamps = np.array([float(line.split(' :')[0]) for line in data_lines])
        line_strings   = [line.split(' :')[1] for line in data_lines]

        trial_block_info = [ls for ls in line_strings if ls[:6] == 'trial-'] # Lines  carrying trial block info.

        # Store data and summary info.

        self.rewards  = sum([ls == IDs['reward'] for ls in line_strings])
        self.n_trials = len(trial_block_info)

        self.fraction_rewarded = self.rewards /  self.n_trials

        # -------------------------------------------------------------------------------------------
        # Make dictionary of choices, transitions, second steps and outcomes on each trial.
        #--------------------------------------------------------------------------------------------
 
        choice_strings      = [l for l in line_strings if l in (self.IDs['choice-down'], self.IDs['choice-up'   ])]
        second_step_strings = [l for l in line_strings if l in (self.IDs['choice-left'], self.IDs['choice-right'])]
        outcome_strings     = [l for l in line_strings if l in (self.IDs['reward'], self.IDs['non-reward'])]

        assert len(choice_strings) == len(second_step_strings) == len(outcome_strings) == self.n_trials, \
               'Unable to read file ' + file_name + ' as number of choices, second steps or outcomes does not match number of trials.'

        choices      = np.array([ c == self.IDs['choice-up']   for c in choice_strings], bool) # True if high, flase if low.
        second_steps = np.array([ c == self.IDs['choice-left'] for c in second_step_strings], bool) # True if left, false if right.     
        outcomes     = np.array([ c == self.IDs['reward']   for c in outcome_strings], bool) # True if rewarded,  flase if no reward.
        transitions  = choices == second_steps  # True if high --> left or low --> right  (A type),
                                                # flase if high --> right or low --> left (B type).

        self.trial_data = {'choices'      : choices.astype(int), #Store as integer arrays.
                           'transitions'  : transitions.astype(int),
                           'second_steps' : second_steps.astype(int),
                           'outcomes'     : outcomes.astype(int)}

        #--------------------------------------------------------------------------------------------
        # Extract block information.
        #--------------------------------------------------------------------------------------------

        trial_rew_state     = np.array([int(t.split('/')[1].split(':')[1]) for t in trial_block_info], int) # Reward state for each trial.
        trial_trans_state = np.array([int(t.split('/')[2].split(':')[1]) for t in trial_block_info], bool)  # Transition state for each trial.
        block_start_mask = (np.not_equal(trial_rew_state[1:], trial_rew_state[:-1]) | np.not_equal(trial_trans_state[1:], trial_trans_state[:-1]))
        start_trials = [0] + (np.where(block_start_mask)[0] + 1).astype(int).tolist() # Block start trials (trials numbered from 0).
        transition_states = trial_trans_state[np.array(start_trials)].astype(int)     # Transition state for each block.
        reward_states = trial_rew_state[np.array(start_trials)]                       # Reward state for each block.
                
        self.blocks = {'start_trials'      : start_trials,
                       'end_trials'        : start_trials[1:] + [self.n_trials],
                       'start_times'       : None,
                       'reward_states'     : reward_states,      # 0 for left good, 1 for neutral, 2 for right good.
                       'transition_states' : transition_states,  # 1 for A blocks, 0 for B blocks.
                       'trial_trans_state' : trial_trans_state,
                       'trial_rew_state'   : trial_rew_state}  

