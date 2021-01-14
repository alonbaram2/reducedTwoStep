#!/bin/sh
# Run preprocessing of functional data

# Command line argument 1/1: subject tag
subjectTag=$1

echo Subject tag for this subject: $subjectTag.
# task name: what appears in the file names of the functional runs
taskName=simple2step
nRun=12
echo Task name is ${taskName}, total number of runs is ${nRun}
# Set scratch directory for execution on server
scratchDir=/home/fs0/abaram/scratch
# If this is not called on the server, but on a laptop connected to the server:
if [ ! -d $scratchDir ]; then
  scratchDir=/Volumes/Scratch_abaram
fi
# If this is not called on a laptop, but on a mac connected to the server:
if [ ! -d $scratchDir ]; then
  scratchDir=/Users/abaram/Documents/ServerHome/scratch
fi
# make scratchDir the project folder
scratchDir=$scratchDir/twoStep/BIDS
# Show what ended up being the scratch directory
echo Scratch directory is $scratchDir

# Set analysis directory for execution on server
homeDir=/home/fs0/abaram
# If this is not called on the server, but on a laptop connected to the server:
if [ ! -d $homeDir ]; then
  homeDir=/Volumes/abaram
fi
# If this is not called on a laptop, but on a mac connected to the server:
if [ ! -d $homeDir ]; then
  homeDir=/Users/abaram/Documents/ServerHome
fi
# Show what ended up being the home directory
echo Home directory is $homeDir
# Set scriptsDir
scriptsDir=$homeDir/scripts/twoStep/fsl
# Construct directory for raw data
rawDir=$scratchDir/sub-$subjectTag
# Construct directory for derived data
derivDir=$scratchDir/derivatives/fsl/sub-$subjectTag
# Construct func directory for derived file
funcDir=$derivDir/func
# And create directory for derived functional files
mkdir -p $funcDir

for run in $(seq -f "%02g" 1 ${nRun}); do
  echo run number $run;
  # Get number of volumes from fslinfo and some bash tricks
  numVols=$(fslval $rawDir/func/sub-${subjectTag}_task-${taskName}_run-${run}_bold.nii.gz dim4)

  # Take preprocessing template, replace subject id and number of volumes with current values and save to new file
  cat $scriptsDir/templates/preproc_smth0.fsf | sed "s/sub-01/sub-${subjectTag}/g" | sed "s/run-01/run-${run}/g"| sed "s/199/${numVols}/g" | sed "s:/vols/Scratch/abaram/twoStep/BIDS:${scratchDir}:g" > $funcDir/sub-${subjectTag}_run-${run}_design_preproc_smth0.fsf

  # Finally: run feat with these parameters
  feat $funcDir/sub-${subjectTag}_run-${run}_design_preproc_smth0.fsf
done
