function getEvs_glm001(rootData,subj)

% rootData: /vols/Scratch/abaram/twoStep/BIDS
% subj: e.g. 'sub-01'

% constants for this experiment
task = 'simple2step';
nRuns = 12;

% Glm 001: reward vs no reward
glm = 'glm-001';

% preprocessed data: should be 3D files (i.e. if using FSL for preprocessing as
% I did, you'll need to use fslsplit to split the 4D files to 3D files).
% This is so that SPM can deal with the files.

% inputDir is the 'func' dir in the subject's BIDS dir. That's where the
% events and behavioural files are.
rootSpm = fullfile(rootData,'spm',subj);
outputDir = fullfile(rootSpm,'evs',glm);
mkdir(outputDir);

nOnsetEvs = 3; % stim1, stim 2, outcome
for iRun=1:nRuns
%     %% The commented out bit is not relevant if using pre-made 3-col EVs 
%     % it's only necessary for importing and organising the BIDS events files
%     %
%     % load events file
%     events = tdfread(fullfile(inputBidsDir,[subj '_task-' task '_run-' num2str(iRun,'%02.f') '_events.tsv']),'\t');
%     % format events structures as numbers and NaNs, and event names as a
%     % cell array
%     events = struct_char2num(events);
%     events.event = cellstr(events.event);
%     % find indeces of events in events cell array. Using Matlab2016a, so
%     % "contains" doesn't work here - need to write an ugly loop
%     events.event_names = unique(events.event);   
%     for iEvent = 1:length(events.event_names)
%         events.event_inds.(events.event_names{iEvent}) = find(strcmp(events.event,events.event_names{iEvent}));
%     end
    
    evsInputDir = fullfile(rootData,'evs',subj,['run-' num2str(iRun,'%02.f')]);
    
    names = cell(1,nOnsetEvs);
    onsets = cell(1,nOnsetEvs);
    durations = cell(1,nOnsetEvs);
    orth = cell(1,nOnsetEvs);
    
    % regressors
    iEv = 1;
    
    names{iEv}     = 'outcome_durStick';
    ev_3col        = dlmread(fullfile(evsInputDir,[names{iEv},'.tsv']));
    onsets{iEv}    = ev_3col(:,1);
    durations{iEv} = ev_3col(:,2);
    
    pmod(iEv).name{1}  = 'reward';
    ev_3col            = dlmread(fullfile(evsInputDir,'reward_durStick.tsv'));   
    pmod(iEv).param{1} = ev_3col(:,3);
    pmod(iEv).param{1} = pmod(iEv).param{1} - mean(pmod(iEv).param{1}); % demean 
    pmod(iEv).poly{1}  = 1;
    
    iEv = iEv + 1;
    
    % other onset regressors
    names{iEv}     = 'stim1_durStick';
    ev_3col        = dlmread(fullfile(evsInputDir,[names{iEv},'.tsv']));
    onsets{iEv}    = ev_3col(:,1);
    durations{iEv} = ev_3col(:,2);
    iEv = iEv + 1;
    
    names{iEv}     = 'stim2_durStick';
    ev_3col        = dlmread(fullfile(evsInputDir,[names{iEv},'.tsv']));
    onsets{iEv}    = ev_3col(:,1);
    durations{iEv} = ev_3col(:,2);
    iEv = iEv + 1;
    
    for iEv=1:length(names)
        orth{iEv}      = false; % orthogonalise parametric modulators?
    end
    save(fullfile(outputDir,['run-' num2str(iRun,'%02.f')]),'names','onsets','durations','pmod','orth');
    
end
