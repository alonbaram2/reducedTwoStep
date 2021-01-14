#!/bin/sh
# subject tag, e.g. 01
subjectTag=$1

root=/home/fs0/abaram/scratch/twoStep/BIDS/derivatives
task=simple2step
nRuns=12

# location of fmriprep preprocessed outputs
preprocAnatDir=$root/fmriprep/sub-${subjectTag}/anat
preprocFuncDir=$root/fmriprep/sub-${subjectTag}/func

# output directories
outputAnatDir=$root/spm/sub-${subjectTag}/preproc_fmriprep/anat
outputFuncDir=$root/spm/sub-${subjectTag}/preproc_fmriprep/func

# make output directories
mkdir -p $outputAnatDir
mkdir -p $outputFuncDir

# move and gunzip anatomical files
cd $outputAnatDir
scp $preprocAnatDir/sub-${subjectTag}_desc-brain_mask.nii.gz ./
scp $preprocAnatDir/sub-${subjectTag}_desc-preproc_T1w.nii.gz ./
scp $preprocAnatDir/sub-${subjectTag}_space-MNI152NLin6Asym_desc-brain_mask.nii.gz ./
scp $preprocAnatDir/sub-${subjectTag}_space-MNI152NLin6Asym_desc-preproc_T1w.nii.gz ./
gunzip *.nii.gz

cd $outputFuncDir

for space in MNI152NLin6Asym T1w; do

  # create functional mask - intersection of masks from all runts
  fslmaths \
  $preprocFuncDir/sub-${subjectTag}_task-${task}_run-1_space-${space}_desc-brain_mask.nii.gz -mul \
  $preprocFuncDir/sub-${subjectTag}_task-${task}_run-2_space-${space}_desc-brain_mask.nii.gz -mul \
  $preprocFuncDir/sub-${subjectTag}_task-${task}_run-3_space-${space}_desc-brain_mask.nii.gz -mul \
  $preprocFuncDir/sub-${subjectTag}_task-${task}_run-4_space-${space}_desc-brain_mask.nii.gz -mul \
  $preprocFuncDir/sub-${subjectTag}_task-${task}_run-5_space-${space}_desc-brain_mask.nii.gz -mul \
  $preprocFuncDir/sub-${subjectTag}_task-${task}_run-6_space-${space}_desc-brain_mask.nii.gz -mul \
  $preprocFuncDir/sub-${subjectTag}_task-${task}_run-7_space-${space}_desc-brain_mask.nii.gz -mul \
  $preprocFuncDir/sub-${subjectTag}_task-${task}_run-8_space-${space}_desc-brain_mask.nii.gz -mul \
  $preprocFuncDir/sub-${subjectTag}_task-${task}_run-9_space-${space}_desc-brain_mask.nii.gz -mul \
  $preprocFuncDir/sub-${subjectTag}_task-${task}_run-10_space-${space}_desc-brain_mask.nii.gz -mul \
  $preprocFuncDir/sub-${subjectTag}_task-${task}_run-11_space-${space}_desc-brain_mask.nii.gz -mul \
  $preprocFuncDir/sub-${subjectTag}_task-${task}_run-12_space-${space}_desc-brain_mask.nii.gz \
  $outputFuncDir/sub-${subjectTag}_task-${task}_run-all_space-${space}_desc-brain_mask.nii.gz
done
  # split functional files. only use the preprocessed data in MNI152NLin6Asym (after NA-AROMA, for univariate analyses) or T1w space (for RSA toolbox)
for run in $(seq -f "%01g" 1 ${nRuns}); do
  echo split func files, sub-${subjectTag}, run $run
  fslsplit $preprocFuncDir/sub-${subjectTag}_task-${task}_run-${run}_space-MNI152NLin6Asym_desc-smoothAROMAnonaggr_bold.nii.gz sub-${subjectTag}_task-${task}_run-${run}_space-MNI152NLin6Asym_desc-smoothAROMAnonaggr_bold_ -t
  fslsplit $preprocFuncDir/sub-${subjectTag}_task-${task}_run-${run}_space-T1w_desc-preproc_bold.nii.gz sub-${subjectTag}_task-${task}_run-${run}_space-T1w_desc-preproc_bold_ -t
done

# gunzip all functional files
gunzip *.nii.gz
