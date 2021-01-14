subjectTag=$1
echo Subject tag for this subject: $subjectTag

# Set scratch directory for execution on server
scratchDir=/vols/Scratch/abaram/twoStep/BIDS

cd $scratchDir

singularity run --cleanenv -B /vols/Scratch/abaram:/vols/Scratch/abaram \
/vols/Scratch/abaram/my_images/fmriprep-20.2.1.simg /vols/Scratch/abaram/twoStep/BIDS \
/vols/Scratch/abaram/twoStep/BIDS/derivatives participant --participant-label $subjectTag \
--fs-license-file /vols/Scratch/abaram/fslicense.txt -w /vols/Scratch/abaram/twoStep/fmriPrep_workDirs/sub-${subjectTag} \
--output-spaces MNI152NLin6Asym T1w \
--use-aroma --error-on-aroma-warnings --skip_bids_validation --medial-surface-nan --mem 10 --nprocs 2

# 02            --mem 10 --nprocs 1     verylong
# 03, 11-28:    --mem 10 --nprocs 2     verylong, -s openmp,2  ??something different after sub-17? seems to be parallel - hav *.po*f files
# 04                     --nprocs 1              bigmem
# 05                     --mem 10                verylong
# 07-10                  --nprocs 1           bigmem

# -w /vols/Scratch/abaram/twoStep/fmriPrep_workDirs/sub-${subjectTag}
