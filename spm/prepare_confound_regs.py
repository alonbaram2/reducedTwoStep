#!/usr/bin/env python3


# Use the output of fmriprep, and choose the confound regressors to add to GLMs. 

import pandas as pd
import os
import os.path as op
import numpy as np
import ipdb




task = 'simple2step'
nSubjs = 28
nRuns  = 12

subjects = [str(x) for x in np.arange(nSubjs)+1] 
subjects = ['sub-' + str(x).zfill(2) for x in subjects]
runs = [str(x) for x in np.arange(nRuns)+1] 

root = '/home/fs0/abaram/scratch/twoStep/BIDS/derivatives/'
for subj in subjects:
    confoundsDir = op.join(root,'spm',subj,'confound_regs')
    if not op.exists(confoundsDir):
        os.mkdir(confoundsDir)
    for run in runs:
        confound_data = pd.read_csv(op.join(root,'fmriprep',subj,'func',
                                            f"{subj}_task-{task}_run-{run}_desc-confounds_timeseries.tsv"), sep="\t")
        # Linear trend cosine regressors - betwen 3-5 for each run
        cosine_regs = confound_data.filter(like='cosine').columns.tolist()
        aCompCor_top5_regs = ['a_comp_cor_00', 'a_comp_cor_01', 'a_comp_cor_02', 'a_comp_cor_03', 'a_comp_cor_04']
        
        
        # key: denoising strategy number. Values: list of confound regressors.
        confounds_dict = {
            # 1: assuming ICA-AROMA was used, so no motion parameters or motion outliers. 
            '1': cosine_regs + aCompCor_top5_regs
                }
        

        for denoising_strategy in ['1']:
            # save files in the format run-01_1
            confound_data.to_csv(path_or_buf=op.join(confoundsDir,'run-' + str(run).zfill(2) + '_xnoise-' + denoising_strategy + '.csv'), 
                                 columns=confounds_dict[denoising_strategy], sep='\t', index = False)