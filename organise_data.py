#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 12:55:54 2020

@author: abaram
"""

import os
import shutil
import os.path as op
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from io import StringIO
import zipfile
import json
import ipdb
# import seaborn as sns  # seaborn is the main plotting library for Pandas
# sns.set()
#%%

# original data from pedro
root_orig      = op.join('/','vols','Scratch','abaram','twoStep','original_data')
# target directory for BIDS organised data
root_target    = op.join('/','vols','Scratch','abaram','twoStep','BIDS')
if not op.exists(root_target):
    os.makedirs(root_target)
# get the dirs of the log files - these look like 6??behav/
orig_logs_dirpaths = glob(op.join(root_orig,'*behav'))
# sort by ascending subj ID number
orig_logs_dirpaths.sort()
# get first 3 characters of the last part of the path (normpath removes the slash and basepath gets the las part) - 6??
orig_subj_nums = [op.basename(op.normpath(path))[0:3] for path in orig_logs_dirpaths]
# number of subjects
nSubj = len(orig_subj_nums)
nRuns = 12
nTrials = 25 # trials in each run
nEventsInTrial = 6; # stim1 onset, choice1, stim2 onset, (forced) choice 2, outcome, ITI onset

# get new, BIDS subj numbers, starting from 1 and zero-padded.
new_subj_numbers = np.arange(len(orig_subj_nums)) + 1
new_subj_numbers = [str(x).zfill(2) for x in new_subj_numbers]
new_subj_names   = ['sub-' + x for x in new_subj_numbers]

#%% rename data paths without spaces and unzip dicom folders if needed

for iSubj, orig_subj_num in enumerate(orig_subj_nums):
    orig_logs_path = orig_logs_dirpaths[iSubj]         
    orig_data_path = glob(op.join(root_orig,'flywheel*','costa','Pedro*',orig_subj_num,'CostaPedro*'))[0]
    
    # if there are any saces in directory and filenames of the original data, replace 
    # them with underscore (this actually only happens once)
    if ' ' in orig_data_path:
        replace_spaces_in_all_subfolders_and_files(root_orig,'_')
        orig_data_path = glob(op.join(root_orig,'flywheel*','costa','Pedro*',orig_subj_num,'CostaPedro*'))[0]
        
        
    os.rename(orig_data_path,orig_data_path.replace("",""))
    
    ## if needed, unzip the zipped dicom folders that will be used. This will take a long time. 
    # check if unzipped folder with dcm files already exists, use run1  as example
    if not glob(op.join(orig_data_path,'run*','*','*.dcm')):
        paths_to_zip_files = glob(op.join(orig_data_path,'run*','*.zip'))
        for path_to_zip_file in paths_to_zip_files:
            print (f'extracting {path_to_zip_file}\n')
            with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
                zip_ref.extractall(path=op.dirname(path_to_zip_file))
                zip_ref.close()
    # same for structural scan
    if not glob(op.join(orig_data_path,'t1_mprage_sag_p2_1mm_iso','*','*.dcm')):
        path_to_zip_file = glob(op.join(orig_data_path,'t1_mprage_sag_p2_1mm_iso','*.zip'))
        print (f'extracting {path_to_zip_file}\n')
        with zipfile.ZipFile(path_to_zip_file[0], 'r') as zip_ref:
            zip_ref.extractall(path=op.dirname(path_to_zip_file[0]))
            zip_ref.close()
    # same for fieldmaps
    if not glob(op.join(orig_data_path,'gre_field_mapping_2mm*','*','*.dcm')):
        paths_to_zip_files = glob(op.join(orig_data_path,'gre_field_mapping_2mm*','*.zip'))
        for path_to_zip_file in paths_to_zip_files:
            print (f'extracting {path_to_zip_file}\n')
            with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
                zip_ref.extractall(path=op.dirname(path_to_zip_file))
                    

    
#%% create dcm2bids configuration file, then run it

## this was done by first running dcm2bids_helper
## on the dicom folder of one example subject, and looking at sidecar JSON files
## there. That's how I identified the "criteria" - how to know which sidecar 
## corresponds to which sequence. The echo times of the fieldmaps were also taken from there. 
bids_dict = {
   "descriptions": [
       {
          "dataType": "anat",
          "modalityLabel": "T1w",
          "criteria": {
             "SidecarFilename": "*t1_mprage*"
          }
       },
       {
          "dataType": "func",
          "modalityLabel": "bold",
          "customLabels": "task-simple2step_run-01",
          "criteria": {
             "SidecarFilename": "*run1_*"
          },
          "sidecarChanges": {
             "TaskName": "simple2step"
          }
       },
       {
          "dataType": "func",
          "modalityLabel": "bold",
          "customLabels": "task-simple2step_run-02",
          "criteria": {
             "SidecarFilename": "*run2*"
          },
          "sidecarChanges": {
             "TaskName": "simple2step"
          }
       },      
       {
          "dataType": "func",
          "modalityLabel": "bold",
          "customLabels": "task-simple2step_run-03",
          "criteria": {
             "SidecarFilename": "*run3*"
          },
          "sidecarChanges": {
             "TaskName": "simple2step"
          }
       },      
       {
          "dataType": "func",
          "modalityLabel": "bold",
          "customLabels": "task-simple2step_run-04",
          "criteria": {
             "SidecarFilename": "*run4*"
          },
          "sidecarChanges": {
             "TaskName": "simple2step"
          }
       },      
       {
          "dataType": "func",
          "modalityLabel": "bold",
          "customLabels": "task-simple2step_run-05",
          "criteria": {
             "SidecarFilename": "*run5*"
          },
          "sidecarChanges": {
             "TaskName": "simple2step"
          }
       },      
       {
          "dataType": "func",
          "modalityLabel": "bold",
          "customLabels": "task-simple2step_run-06",
          "criteria": {
             "SidecarFilename": "*run6*"
          },
          "sidecarChanges": {
             "TaskName": "simple2step"
          }
       },      
       {
          "dataType": "func",
          "modalityLabel": "bold",
          "customLabels": "task-simple2step_run-07",
          "criteria": {
             "SidecarFilename": "*run7*"
          },
          "sidecarChanges": {
             "TaskName": "simple2step"
          }
       },      
       {
          "dataType": "func",
          "modalityLabel": "bold",
          "customLabels": "task-simple2step_run-08",
          "criteria": {
             "SidecarFilename": "*run8*"
          },
          "sidecarChanges": {
             "TaskName": "simple2step"
          }
       },      
       {
          "dataType": "func",
          "modalityLabel": "bold",
          "customLabels": "task-simple2step_run-09",
          "criteria": {
             "SidecarFilename": "*run9*"
          },
          "sidecarChanges": {
             "TaskName": "simple2step"
          }
       },      
       {
          "dataType": "func",
          "modalityLabel": "bold",
          "customLabels": "task-simple2step_run-10",
          "criteria": {
             "SidecarFilename": "*run10*"
          },
          "sidecarChanges": {
             "TaskName": "simple2step"
          }
       },        
       {
          "dataType": "func",
          "modalityLabel": "bold",
          "customLabels": "task-simple2step_run-11",
          "criteria": {
             "SidecarFilename": "*run11*"
          },
          "sidecarChanges": {
             "TaskName": "simple2step"
          }
       },      
       {
          "dataType": "func",
          "modalityLabel": "bold",
          "customLabels": "task-simple2step_run-12",
          "criteria": {
             "SidecarFilename": "*run12*"
          },
          "sidecarChanges": {
             "TaskName": "simple2step"
          }
       },            
      {
         "dataType": "fmap",
         "modalityLabel": "magnitude1",
         "IntendedFor": [ 
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12
            ],           
         "criteria": {
            "SidecarFilename": "*gre_field_mapping*",
            "EchoTime": 0.00487,
            "ImageType": [
                "ORIGINAL",
                "PRIMARY",
                "M",
                "ND"
            ]         
         }
      },
      {
         "dataType": "fmap",
         "modalityLabel": "magnitude2",  
         "IntendedFor": [ 
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12
            ],          
         "criteria": {
            "SidecarFilename": "*gre_field_mapping*",
            "ImageType": [
                "ORIGINAL",
                "PRIMARY",
                "M",
                "ND"
            ],
            "EchoTime": 0.00733            
         }
      },      
      {
         "dataType": "fmap",
         "modalityLabel": "phasediff",
         "IntendedFor": [ 
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12
            ],           
         "criteria": {
            "SidecarFilename": "*gre_field_mapping*_ph*"
         },
         "sidecarChanges": {
            "EchoTime1": 0.00487,
            "EchoTime2": 0.00733
         }
      }
   ]
}
 
# change dir to new root
os.chdir(root_target)

# save dcm2bids config file
with open(op.join(root_target,'dcm2bids_configFile.json'), 'w') as f:
    json.dump(bids_dict, f)

# create BIDS scaffolding -
# os.system('dcm2bids_scaffold') 

# runs is what Pedro calls sessions - 1-12. Zero pad
runs = [str(x) for x in np.arange(nRuns)+1] 
runs = [str(x).zfill(2) for x in runs]

for iSubj, orig_subj_num in enumerate(orig_subj_nums):
    orig_logs_path = orig_logs_dirpaths[iSubj]         
    orig_data_path = glob(op.join(root_orig,'flywheel*','costa','Pedro*',orig_subj_num,'CostaPedro*'))[0]
    
    # new subj ID without "sub-"
    new_subj_number = new_subj_numbers[iSubj]
    # new subj name with "sub-"
    new_subj_name = new_subj_names[iSubj]
        
    # run dcm2bids
    os.system('dcm2bids -d ' + orig_data_path + ' -p ' + new_subj_number + ' -c ' + op.join(root_target,'dcm2bids_configFile.json'))
    print(new_subj_number)
    
    # copy dicom folders to sourcedata, with new subject IDs.     
    if not op.exists(op.join(root_target,'sourcedata',new_subj_name)):
        shutil.copytree(orig_data_path,op.join(root_target,'sourcedata',new_subj_name))    
    
    # create new_subj_func_path if it doesn't exist
    
    subj_func_path  = op.join(root_target,new_subj_names[iSubj],'func')

    print(f'save tsv log files to \n {subj_func_path} \n')
    for r in runs:
        ## log files ##
        # get the filename that doesn't have '_logs' at the end. 
        orig_fname = glob(op.join(orig_logs_path,orig_subj_num + '_session' + str(int(r)) + '_*' + '20??'))[0]    
        with open(orig_fname,'r') as f:
            # Tidy original files so that they'll look like standard CSV files
            csv_str = f.read().replace('   ',' ').replace(' ',',')
            logs_data = pd.read_csv(StringIO(csv_str), sep=',', skipfooter=2, engine='python')
        
            # create an empty table with all the columns we need. there are 5 events
            # each trial and 25 trials so 125 rows
            events_bids = pd.DataFrame(columns=['onset','duration','response_time','trial','event','state','reward'],index=np.arange(nEventsInTrial*nTrials))
            for iTrial in np.arange(nTrials):
                events_bids['trial'][iTrial*nEventsInTrial:iTrial*nEventsInTrial+nEventsInTrial] = iTrial + 1
                
                events_bids['event'][iTrial*nEventsInTrial] = 'stim1'
                events_bids['onset'][iTrial*nEventsInTrial] = logs_data['stim1ons_ms'][iTrial]
                events_bids['duration'][iTrial*nEventsInTrial] = logs_data['choice1ons_ms'][iTrial] - logs_data['stim1ons_ms'][iTrial]
                events_bids['response_time'][iTrial*nEventsInTrial] = logs_data['choice1ons_ms'][iTrial] - logs_data['stim1ons_ms'][iTrial]
                
                events_bids['event'][iTrial*nEventsInTrial + 1] = 'choice1'
                events_bids['onset'][iTrial*nEventsInTrial + 1] = logs_data['choice1ons_ms'][iTrial]
                events_bids['duration'][iTrial*nEventsInTrial + 1] = logs_data['jitter2_ms'][iTrial] - logs_data['choice1ons_ms'][iTrial]

                events_bids['event'][iTrial*nEventsInTrial + 2] = 'stim2'
                events_bids['onset'][iTrial*nEventsInTrial + 2] = logs_data['stim2ons_ms'][iTrial]
                events_bids['duration'][iTrial*nEventsInTrial + 2] = logs_data['choice2ons_ms'][iTrial] - logs_data['stim2ons_ms'][iTrial]
                events_bids['response_time'][iTrial*nEventsInTrial + 2] = logs_data['choice2ons_ms'][iTrial] - logs_data['stim2ons_ms'][iTrial]
                events_bids['state'][iTrial*nEventsInTrial + 2] = logs_data['state'][iTrial]
                
                events_bids['event'][iTrial*nEventsInTrial + 3] = 'choice2'
                events_bids['onset'][iTrial*nEventsInTrial + 3] = logs_data['choice2ons_ms'][iTrial]
                events_bids['duration'][iTrial*nEventsInTrial + 3] = logs_data['jitter3_ms'][iTrial] - logs_data['choice2ons_ms'][iTrial]
                events_bids['state'][iTrial*nEventsInTrial + 3] = logs_data['state'][iTrial]

                events_bids['event'][iTrial*nEventsInTrial + 4] = 'outcome'
                events_bids['onset'][iTrial*nEventsInTrial + 4] = logs_data['jitter3_ms'][iTrial] + logs_data['jitter3'][iTrial]
                events_bids['duration'][iTrial*nEventsInTrial + 4] = 1500 # there was no jitter here
                events_bids['reward'][iTrial*nEventsInTrial + 4] = logs_data['won'][iTrial]

                events_bids['event'][iTrial*nEventsInTrial + 5] = 'ITI_onset'
                events_bids['onset'][iTrial*nEventsInTrial + 5] = events_bids['onset'][iTrial*nEventsInTrial + 4] + 1500
            
            # get duration of ITI period by subtracting the stim1 onset of next trial from the end of outcome period (outcome_onset + 1500ms)
            for iTrial in np.arange(nTrials-1):                                            
                events_bids['duration'][iTrial*nEventsInTrial + 5] = events_bids['onset'][(iTrial+1)*nEventsInTrial] - (events_bids['onset'][iTrial*nEventsInTrial + 4] + 1500)
            events_bids['duration'][nEventsInTrial*nTrials -1] = 1000 # arbitrarily set ITI duration of last trial         
            
            # convert ms to seconds and subtract the first trigger (stim1 onset of trial 0)
            events_bids['onset'] = events_bids['onset'].div(1000) - (events_bids['onset'][0] / 1000)                   
            events_bids['duration'] = events_bids['duration'].div(1000)                    
            events_bids['response_time'] = events_bids['response_time'].div(1000)                    
            
            events_bids.fillna('n/a', inplace=True)
            
            
            new_fname = op.join(subj_func_path, new_subj_names[iSubj] + '_task-simple2step_run-' + r + '_events.tsv')
            events_bids.to_csv(new_fname, sep='\t', index=False)
        
#%% clean up - make sure all filenames conform to BIDS standard
if op.exists(op.join(root_target,'tmp_dcm2bids')):
    shutil.rmtree(op.join(root_target,'tmp_dcm2bids'))
    
os.remove(op.join(root_target,'dcm2bids_configFile.json'))    

# several sybjects had a run that started and was terminated quickly. Change
# the names of the file of the full run to conform to the BIDS standard and then
# delete the other empty runs
os.rename(op.join(root_target,'sub-09/func/sub-09_task-simple2step_run-01_run-02_bold.nii.gz'),op.join(root_target,'sub-09/func/sub-09_task-simple2step_run-01_bold.nii.gz'))
os.rename(op.join(root_target,'sub-09/func/sub-09_task-simple2step_run-01_run-02_bold.json'),op.join(root_target,'sub-09/func/sub-09_task-simple2step_run-01_bold.json'))
os.rename(op.join(root_target,'sub-11/func/sub-11_task-simple2step_run-01_run-02_bold.nii.gz'),op.join(root_target,'sub-11/func/sub-11_task-simple2step_run-01_bold.nii.gz'))
os.rename(op.join(root_target,'sub-11/func/sub-11_task-simple2step_run-01_run-02_bold.json'),op.join(root_target,'sub-11/func/sub-11_task-simple2step_run-01_bold.json'))
os.rename(op.join(root_target,'sub-13/func/sub-13_task-simple2step_run-01_run-03_bold.nii.gz'),op.join(root_target,'sub-13/func/sub-13_task-simple2step_run-01_bold.nii.gz'))
os.rename(op.join(root_target,'sub-13/func/sub-13_task-simple2step_run-01_run-03_bold.json'),op.join(root_target,'sub-13/func/sub-13_task-simple2step_run-01_bold.json'))
os.rename(op.join(root_target,'sub-16/func/sub-16_task-simple2step_run-01_run-04_bold.nii.gz'),op.join(root_target,'sub-16/func/sub-16_task-simple2step_run-01_bold.nii.gz'))
os.rename(op.join(root_target,'sub-16/func/sub-16_task-simple2step_run-01_run-04_bold.json'),op.join(root_target,'sub-16/func/sub-16_task-simple2step_run-01_bold.json'))
os.rename(op.join(root_target,'sub-24/func/sub-24_task-simple2step_run-09_run-02_bold.nii.gz'),op.join(root_target,'sub-24/func/sub-24_task-simple2step_run-09_bold.nii.gz'))
os.rename(op.join(root_target,'sub-24/func/sub-24_task-simple2step_run-09_run-02_bold.json'),op.join(root_target,'sub-24/func/sub-24_task-simple2step_run-09_bold.json'))
os.rename(op.join(root_target,'sub-26/func/sub-26_task-simple2step_run-01_run-04_bold.nii.gz'),op.join(root_target,'sub-26/func/sub-26_task-simple2step_run-01_bold.nii.gz'))
os.rename(op.join(root_target,'sub-26/func/sub-26_task-simple2step_run-01_run-04_bold.json'),op.join(root_target,'sub-26/func/sub-26_task-simple2step_run-01_bold.json'))   
for f in glob(op.join(root_target,'*','func','*run-*run*')):
    os.remove(f)




def replace_spaces_in_all_subfolders_and_files(parent,newChar):
    for path, folders, files in os.walk(parent):
        for f in files:
            os.rename(os.path.join(path, f), os.path.join(path, f.replace(' ', newChar)))
        for i in range(len(folders)):
            new_name = folders[i].replace(' ', newChar)
            os.rename(os.path.join(path, folders[i]), os.path.join(path, new_name))
            folders[i] = new_name

        
            