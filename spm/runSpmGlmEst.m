function runSpmGlmEst(rootData,sub,glm,space,denoising)
% rootData: /vols/Scratch/abaram/twoStep/BIDS
% sub: subject name, e.g. sub-01
% glm: GLM name, e.g. GLM1
% space: t1w or mni
% denoising strategy. e.g. '1'. Legend can be found in /home/fs0/abaram/scripts/twoStep/spm/prepare_confound_regs.py

% constants for this experiment
task = 'simple2step';
nRuns = 12;

rootSpm       = fullfile(rootData,'spm',sub);
outputDir     = fullfile(rootSpm,['glmDirs_' space],[glm '_xn-' denoising '_hrf8p5']);
evsDir        = fullfile(rootSpm,'evs',glm); % folder where model spec files (i.e. evs/regressors) are, created in wrapper_univariate.m
preprocDir    = fullfile(rootSpm,'preproc_fmriprep','func');
confoundsDir  = fullfile(rootSpm,'confound_regs'); % original fmriprep preproc dir, which includes the confound files
if ~exist(outputDir,'dir') % create a folder for 1st-level results
    mkdir(outputDir);
end

% acquisition parameters
TR      = 2;
nSlices = 72;

% mask for running GLMs and getting the right dimensions. from https://www.templateflow.org/usage/archive/
if strcmp (space,'MNI152NLin6Asym')
    mask = fullfile(rootData, 'spm', space, ['tpl-' space '_res-02_desc-brain_mask.nii']);
end

% depending on denoising strategy, use either AROMA cleaned data or just
% usual preproc dta
if str2num(denoising) < 10 
   aromaStr = 'smoothAROMAnonaggr';
else
    aromaStr = 'preproc';
end

%% set up SPM batch
% J is what is ususally called "fmri_spec" in SPM code
J.dir = {outputDir};
J.timing.units = 'secs';
J.timing.RT = TR;
J.timing.fmri_t = nSlices; % number of slices
J.timing.fmri_t0 = 1;        % reference slice
for iRun=1:nRuns 
    [filePathListThisRun] = spm_select('FPList',preprocDir,[sub '_task-' task '_run-' num2str(iRun,'%01.f') '_space-' space '_desc-' aromaStr '_bold_^*.*nii$']);
    filePathListThisRun = mat2cell(filePathListThisRun, ones(size(filePathListThisRun,1),1));%convert to cell arrays
    % scans
    J.sess(iRun).scans=filePathListThisRun;
    % set to empty
    J.sess(iRun).cond = struct('name', {}, 'onset', {}, 'duration', {}, 'tmod', {}, 'pmod', {});
    % multi cond files
    J.sess(iRun).multi = {fullfile(evsDir,['run-' num2str(iRun,'%02.f') '.mat'])}; 
    % set to empty
    J.sess(iRun).regress = struct('name', {}, 'val', {});
        
    % confound regressors 
    confounds = readtable(fullfile(confoundsDir,['run-' num2str(iRun,'%02.f') '_xnoise-' denoising '.csv']));
    R = table2array(confounds); % load motion correction parameters.
    names = confounds.Properties.VariableNames; % names of columns in motion parameters files of FSL. 
    save(fullfile(confoundsDir,['run-' num2str(iRun,'%02.f') '_xnoise-' denoising '.mat']),'names','R');    
    J.sess(iRun).multi_reg = {fullfile(confoundsDir,['run-' num2str(iRun,'%02.f') '_xnoise-' denoising '.mat'])};      
    % high pass filter
    J.sess(iRun).hpf = inf; % I already high-passed filtered during preprocessing using FSL, so no need to filter here. I will filter the design matrix manually later. 
end
J.fact = struct('name', {}, 'levels', {});
J.bases.hrf.derivs = [0 0];
J.volt = 1;
J.global = 'None';
J.mthresh = -Inf;  % we will use an explicit mask of the brain, so setting this to very low. 
J.mask = {[mask,',1']};
J.cvi = 'fast';   % autocorrelation correction
matlabbatch{1}.spm.stats.fmri_spec = J;
% model estimation
estName=fullfile(outputDir,'SPM.mat'); % SPM.mat for estimation
matlabbatch{2}.spm.stats.fmri_est.spmmat = {estName};
matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;
% run model specification and estation
spm_jobman('run',matlabbatch);