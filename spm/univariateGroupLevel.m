function univariateGroupLevel(rootData,subjects,glm,space,denoising)

rootSpm       = fullfile(rootData,'spm');
outputDir = fullfile(rootSpm,space, 'groupLevel',[glm '_xn-' denoising '_hrf8p5']);

standardMask = fullfile(rootSpm, space, ['tpl-' space '_res-02_desc-brain_mask.nii']);

% relevent fields in SPM.mat should be the same in all subjects (contrast name etc.), 
% so just loading one subject toget those fields

load(fullfile(rootSpm,'sub-01',['glmDirs_' space],[glm '_xn-' denoising '_hrf8p5'], 'SPM.mat'));

for iCon = 1:size(SPM.xCon,2) % number of contrasts
    % folders are named by the contrast name
    mkdir(fullfile(outputDir,SPM.xCon(iCon).name))
    for iSubj=1:length(subjects)
        % get all warped contrast images for subject and copy to 2ndLevel
        % folder
        originalFileName      = fullfile(rootSpm,subjects{iSubj},['glmDirs_' space],[glm '_xn-' denoising '_hrf8p5'],sprintf('con_%04d.nii',iCon));
        secondLevelFileName = fullfile(outputDir,SPM.xCon(iCon).name,sprintf('/%02s_con_%04d.nii',subjects{iSubj},iCon));
        
        copyfile(originalFileName,secondLevelFileName);               
    end
    
    %% group stats with SPM
    clear factorial_design matlabbatch
    
    factorial_design.dir{1} = fullfile(outputDir,SPM.xCon(iCon).name);
    cd(factorial_design.dir{1})
    
    contrast_images  = spm_select('List','*_con*');
    for epi = 1:size(contrast_images,1)
        factorial_design.des.t1.scans{epi,1} = [factorial_design.dir{1},'/',contrast_images(epi,:)];
    end
    
    factorial_design.cov = struct('c', {}, 'cname', {}, 'iCFI', {}, 'iCC', {});
    factorial_design.multi_cov = struct('files', {}, 'iCFI', {}, 'iCC', {});
    factorial_design.masking.tm.tm_none = 1;
    factorial_design.masking.im = 1;
    factorial_design.masking.em = {standardMask};
    factorial_design.globalc.g_omit = 1;
    factorial_design.globalm.gmsca.gmsca_no = 1;
    factorial_design.globalm.glonorm = 1;
    matlabbatch{1}.spm.stats.factorial_design = factorial_design;
    
    spm_jobman('run',matlabbatch);
    
    clear matlabbatch
    matlabbatch{1}.spm.stats.fmri_est.spmmat = {[factorial_design.dir{1} '/SPM.mat']};
    matlabbatch{1}.spm.stats.fmri_est.write_residuals = 0;
    matlabbatch{1}.spm.stats.fmri_est.method.Classical = 1;
    
    matlabbatch{2}.spm.stats.con.spmmat = {[factorial_design.dir{1} '/SPM.mat']};
    matlabbatch{2}.spm.stats.con.consess{1}.tcon.name = 'ttest';
    matlabbatch{2}.spm.stats.con.consess{1}.tcon.weights = 1;
    matlabbatch{2}.spm.stats.con.consess{1}.tcon.sessrep = 'none';
    matlabbatch{2}.spm.stats.con.delete = 0;
    
    spm_jobman('run',matlabbatch);
    
    % add contrast name to file name
    movefile([factorial_design.dir{1},'/spmT_0001.nii'],[factorial_design.dir{1},'/spmT_',SPM.xCon(iCon).name,'_',glm,'.nii'])    
end