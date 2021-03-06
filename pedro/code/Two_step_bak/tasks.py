import numpy as np
from random import randint, random

# -------------------------------------------------------------------------------------
# Extended two step
# -------------------------------------------------------------------------------------

def with_prob(prob):
    'return true / flase with specified probability .'
    return random() < prob

class Extended_two_step:
    '''Two step task with reversals in both which side is good and the transition matrix.'''
    def __init__(self, neutral_reward_probs = False):
        # Parameters
        self.norm_prob = 0.8 # Probability of normal transition.
        self.neutral_reward_probs = neutral_reward_probs

        if neutral_reward_probs: 
            self.reward_probs = np.array([[0.4, 0.4],  # Reward probabilities in each reward block type.
                                          [0.4, 0.4],
                                          [0.4, 0.4]])
        else:
            self.reward_probs = np.array([[0.8, 0.2],  # Reward probabilities in each reward block type.
                                          [0.4, 0.4],
                                          [0.2, 0.8]])
        self.threshold = 0.75 
        self.tau = 8.  # Time constant of moving average.
        self.min_block_length = 40       # Minimum block length.
        self.min_trials_post_criterion = 20  # Number of trials after transition criterion reached before transtion occurs.
        self.mov_ave = exp_mov_ave(tau = self.tau, init_value = 0.5)   # Moving average of agents choices.
        self.reset()

    def reset(self, n_trials = 1000):
        self.transition_block = with_prob(0.5)      # True for A blocks, false for B blocks.
        self.reward_block =     randint(0,2)        # 0 for left good, 1 for neutral, 2 for right good.
        self.block_trials = 0                       # Number of trials into current block.
        self.cur_trial = 0                         # Current trial number.
        self.trans_crit_reached = False             # True if transition criterion reached in current block.
        self.trials_post_criterion = 0              # Current number of trials past criterion.
        self.trial_number = 1                       # Current trial number.
        self.n_trials = n_trials                    # Session length.
        self.mov_ave.reset()
        self.end_session   = False
        self.blocks = {'start_trials'      : [0],
                       'end_trials'        : [],
                       'reward_states'     : [self.reward_block],      # 0 for left good, 1 for neutral, 2 for right good.
                       'transition_states' : [self.transition_block]}  # 1 for A blocks, 0 for B blocks.

    def trial(self, choice):
        # Update moving average.
        self.mov_ave.update(choice)
        second_step = (choice ==  with_prob(self.norm_prob)) == self.transition_block
        self.block_trials += 1
        self.cur_trial += 1
        outcome = int(with_prob(self.reward_probs[self.reward_block, second_step]))
        # Check for block transition.
        block_transition = False
        if self.trans_crit_reached:
            self.trials_post_criterion +=1
            if (self.trials_post_criterion >= self.min_trials_post_criterion) & \
               (self.block_trials >= self.min_block_length):
               block_transition = True
        else: # Check if transition criterion reached.
            if self.reward_block == 1 or self.neutral_reward_probs: #Neutral block
                if (self.block_trials > 20) & with_prob(0.04):
                    self.trans_crit_reached = True
            elif self.transition_block ^ (self.reward_block == 0): # High is good option
                if self.mov_ave.ave > self.threshold:
                    self.trans_crit_reached = True
            else:                                                  # Low is good option
                if self.mov_ave.ave < (1. -self.threshold):
                    self.trans_crit_reached = True                

        if block_transition:
            self.block_trials = 0
            self.trials_post_criterion = 0
            self.trans_crit_reached = False
            old_rew_block = self.reward_block
            if old_rew_block == 1:                      # End of neutral block always transitions to one side 
                self.reward_block = with_prob(0.5) * 2  # being good without reversal of transition probabilities.
            else: # End of block with one side good, 50% chance of change in transition probs.
                if with_prob(0.5): #Reversal in transition probabilities.
                    self.transition_block = not self.transition_block
                    if with_prob(0.5): # 50% chance of transition to neutral block.
                        self.reward_block = 1
                else: # No reversal in transition probabilities.
                    if with_prob(0.5):
                        self.reward_block = 1 # Transition to neutral block.
                    else:
                        self.reward_block = 2 - old_rew_block # Invert reward probs.
            self.blocks['start_trials'].append(self.cur_trial)
            self.blocks['end_trials'].append(self.cur_trial)
            self.blocks['reward_states'].append(self.reward_block)
            self.blocks['transition_states'].append(self.transition_block)

        if self.cur_trial >= self.n_trials: #End of session.
            self.end_session = True
            self.blocks['end_trials'].append(self.cur_trial + 1)

            self.blocks['trial_trans_state'] = np.zeros(self.n_trials, dtype = bool) #Boolean array indication state of tranistion matrix for each trial.
            self.blocks['trial_rew_state']   = np.zeros(self.n_trials, dtype = int)

            for start_trial,end_trial, trans_state, reward_state in \
                    zip(self.blocks['start_trials'],self.blocks['end_trials'], \
                        self.blocks['transition_states'], self.blocks['reward_states']):
                self.blocks['trial_trans_state'][start_trial - 1:end_trial-1] = trans_state   
                self.blocks['trial_rew_state'][start_trial - 1:end_trial-1]  = reward_state   

        return (second_step, outcome)


class exp_mov_ave:
    'Exponential moving average class.'
    def __init__(self, tau, init_value):
        self.tau = tau
        self.init_value = init_value
        self.reset()

    def reset(self, init_value = None, tau = None):
        if tau:
            self.tau = tau
        if init_value:
            self.init_value = init_value
        self.ave = self.init_value
        self._m = np.exp(-1./self.tau)
        self._i = 1 - self._m

    def update(self, sample):
        self.ave = (self.ave * self._m) + (self._i * sample)

