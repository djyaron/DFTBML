#!/bin/bash

# This script runs analysis on the trained DFTBML model. Only execute this on a cluster
# with adequate memory AFTER the train_step.sh script 

# Copy this script into the DFTBML directory (one level above 20000_cc_reproduction)
# before running.

# Activate the virtual environment created during setup
conda activate DFTBML

# Set up the analysis
# NOTE: Make sure that the exec_path variable in analyze.py points to your installed 
# dftb+ executable!
rm -rf analysis_dir/results/*
rm -rf analysis_dir/analysis_files/*
cp -r benchtop_wdir/results/base_dset_expanded_20000_RESULT analysis_dir/results/

echo "Starting DFTB+ analysis..."
python analyze.py internal Y analysis_dir/results N

