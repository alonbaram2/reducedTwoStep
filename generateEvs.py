import os
import os.path as op
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import ipdb


class subject:
    'Class containing data from a single subject'
    def __init__(self,root,subj_tag):
        self.subj_name = f'sub-{subj_tag}'
        self.subj_tag  = subj_tag
        self.path_data       = op.join(root,self.subj_name)
        self.path_evs        = op.join(root,'derivatives','evs',self.subj_name)
        self.path_func_data  = op.join(self.path_data,'func')
        self.path_fsl        = op.join(root,'derivatives','fsl',self.subj_name)
        self.path_fmriprep   = op.join(root,'derivatives','fmriprep',self.subj_name)
        

class run_:
    'Class containing data from a single run, for a single subject'
    def __init__(self,subject,run_tag):
        self.subj_tag = subject.subj_tag
        self.run_tag = run_tag
        self.events_file = op.join(subject.path_func_data,subject.subj_name + '_task-' + task_name + '_run-' + run_tag + '_events.tsv')
        self.bold_file = op.join(subject.path_func_data,subject.subj_name + '_task-' + task_name + '_run-' + run_tag + '_bold.nii.gz')
        self.bold_file_dict = op.join(subject.path_func_data,subject.subj_name + '_task-' + task_name + '_run-' + run_tag + '_bold.json')
        self.feat_preproc_dir = op.join(subject.path_fsl,'func',f'preproc_run-{run_tag}.feat')
        self.evs_dir = op.join(subject.path_evs,'run-' + run_tag)        
        
    def generateEvs(self):
        def df_to_csv_3durations(df, evs_dir, reg_name, paramReg):
            # saves a dataFrame in 3-col regressors format, using 3 different durations:
            # dur2next, dur2fb, durStick
            df.to_csv(op.join(evs_dir, reg_name + '_dur2next.tsv'), \
                                   columns = ['onset','duration',paramReg], \
                                       sep = '\t', index = False, header = False)                
            df.to_csv(op.join(evs_dir, reg_name + '_dur2fb.tsv'), \
                                   columns = ['onset','duration2fb',paramReg], \
                                       sep = '\t', index = False, header = False)                            
            df.to_csv(op.join(self.evs_dir, reg_name + '_durStick.tsv'), \
                                   columns = ['onset','durationStick',paramReg], \
                                       sep = '\t', index = False, header = False)
            
        print(f"generate EVs sub-{self.subj_tag} run-{self.run_tag}")
        # check if EVs dir exists, if not create it
        os.makedirs(self.evs_dir, exist_ok = True)
        # get events data
        self.events_df = pd.read_table(self.events_file)
        # add a column of ones to dataFrame, to use as main effect regressor
        self.events_df['columnOfOnes'] = np.ones(self.events_df.shape[0])
        # duration for stick regressors - 100ms long
        self.events_df['durationStick'] = np.ones(self.events_df.shape[0]) / 10
        # first create EVs for the single events
        events_list = ['stim1','choice1','stim2','choice2','outcome','ITI_onset']
        for event in events_list:
            df_tmp = self.events_df.loc[self.events_df['event'] == event]
            df_to_csv_3durations(df_tmp, self.evs_dir, event, 'columnOfOnes')  
                            
        # now do the same for regressors including more than one event
        # all stims
        event = 'stim12' # as stim12_dur2next.tsv for short duration and stim12_dur2fb.tsv for longer duration
        df_tmp = self.events_df.loc[(self.events_df['event'] == 'stim1') \
                                              | (self.events_df['event'] == 'stim2')]
        df_to_csv_3durations(df_tmp, self.evs_dir, event, 'columnOfOnes')                
             
        # all choices
        event = 'choice12'
        df_tmp = self.events_df.loc[(self.events_df['event'] == 'choice1') \
                                              | (self.events_df['event'] == 'choice2')]
        df_to_csv_3durations(df_tmp, self.evs_dir, event, 'columnOfOnes') 
            
        # all events
        event = 'allEvents'        
        df_to_csv_3durations(self.events_df, self.evs_dir, event, 'columnOfOnes') 
                                    
        ## main effect regressors of different states - for use in RSA aand decoding
        
        # state0@stim2
        event = 'state0_stim2'
        df_tmp = self.events_df.loc[(self.events_df['event'] == 'stim2') \
                                              & (self.events_df['state'] == 0)]
        df_to_csv_3durations(df_tmp, self.evs_dir, event, 'columnOfOnes') 
                
        # state1@stim2
        event = 'state1_stim2'
        df_tmp = self.events_df.loc[(self.events_df['event'] == 'stim2') \
                                              & (self.events_df['state'] == 1)]
        df_to_csv_3durations(df_tmp, self.evs_dir, event, 'columnOfOnes')        

        # state0@choice2
        event = 'state0_choice2'
        df_tmp = self.events_df.loc[(self.events_df['event'] == 'choice2') \
                                              & (self.events_df['state'] == 0)]
        df_to_csv_3durations(df_tmp, self.evs_dir, event, 'columnOfOnes')

        # state1@choice2
        event = 'state1_choice2'
        df_tmp = self.events_df.loc[(self.events_df['event'] == 'choice2') \
                                              & (self.events_df['state'] == 1)]
        df_to_csv_3durations(df_tmp, self.evs_dir, event, 'columnOfOnes')        
        
        
        ## parametric regressors
        
        # reward@outcome
        event = 'reward';
        df_tmp = self.events_df.loc[self.events_df['event'] == 'outcome']
        df_tmp['demeaned_reward'] = df_tmp['reward'] - df_tmp['reward'].mean()
        df_to_csv_3durations(df_tmp, self.evs_dir, event, 'demeaned_reward')        
         
        
task_name = 'simple2step'
nRuns = 12

root       = op.join('/','vols','Scratch','abaram','twoStep','BIDS')
full_paths_subj = sorted(glob(op.join(root,'sub-*')))
subj_names = [op.basename(x) for x in full_paths_subj]
subj_tags  = [x[-2:] for x in subj_names]
full_paths_func  = [op.join(x,'func') for x in full_paths_subj]
# list of strings with zero padded run numbers. 
run_tags = [str(x).zfill(2) for x in np.arange(nRuns)+1]
for subj_tag in subj_tags:
    subj_obj = subject(root,subj_tag)
    for run_tag in run_tags:    
        run_obj  = run_(subj_obj, run_tag)
        run_obj.generateEvs()
        




    