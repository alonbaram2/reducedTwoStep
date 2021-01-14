rootData    = '/vols/Scratch/abaram/twoStep/BIDS/derivatives';

rootScripts = '/home/fs0/abaram/scripts/twoStep/spm';
addpath(genpath(rootScripts));
spmPath = '/vols/Scratch/abaram/MATLAB/spm12';
addpath(genpath(spmPath));

subj='sub-00';
glm='glm-000';
space='T1w';
denoising = '999';

% getEvs_glm001(rootData,subj)

runSpmGlmEst(rootData,subj,glm,space,denoising)

runSpmContrast_glm001(rootData,subj,space,denoising)