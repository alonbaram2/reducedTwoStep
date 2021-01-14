#!/bin/sh

# archive preprocessed 4D files from fmriprep's func directory.
# this is to release space following the splitting and unzipping of these files
# for spm.

subjectTag=$1

root=/home/fs0/abaram/scratch/twoStep/BIDS/derivatives/fmriprep/sub-${subjectTag}/func
cd $root
mkdir $root/all_4D_bold_files
mv $root/*_bold.nii.gz $root/all_4D_bold_files/

archive all_4D_bold_files
