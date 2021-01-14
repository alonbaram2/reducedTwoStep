function runSpmContrast_glm001(rootData,subj,space,denoising)

glm = 'glm-001';
rootSpm       = fullfile(rootData,'spm',subj);
glmDir     = fullfile(rootSpm,['glmDirs_' space],[glm '_xn-' denoising '_hrf8p5']);

% # calculate contrasts
cd(glmDir);
load SPM
% delete preivious calculations of contrasts, if they exist
if isfield(SPM,'xCon')
    SPM=rmfield(SPM,'xCon');
end
delete('*spmT*')
delete('con*')
% contrasts names
cNames = {'reward'};
con=[];

% contrast 1: reward
c=zeros(1,size(SPM.xX.X,2));
for r=1:8 % blocks
    v = SPM.Sess(r).col; % all indeces of regressors of block
    c(v(2))=1; % outcome_durStick x reward
end
con = [con; c];

% caclulate contrast images
for iCon=1:numel(cNames)
    SPM.xCon(iCon)=spm_FcUtil('Set',sprintf('%s',cNames{iCon}), 'T', 'c',con(iCon,:)',SPM.xX.xKXs);
end
SPM=spm_contrasts(SPM,[1:length(SPM.xCon)]);
save SPM SPM;
