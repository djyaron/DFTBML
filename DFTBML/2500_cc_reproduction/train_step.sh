#!/bin/bash

# This script runs through all the DFTBML model training. 
# Only execute this on a cluster with adequate memory AFTER 
# the precompute_step.sh script

# Copy this script into the DFTBML directory (one level above 20000_cc_reproduction)
# before running.

# Activate the virtual environment created during setup
conda activate DFTBML

# With the precomputed dataset, set up the benchtop_wdir directory. 

# First, some clean up
echo "Cleaning up benchtop_wdir..."
rm -rf benchtop_wdir/dsets/*
rm -rf benchtop_wdir/results/*
rm -rf benchtop_wdir/tmp/*

# Copy the settings file and the datasets over
echo "Transferring files over to benchtop_wdir..."
cp -r 2500_cc_reproduction/dset_2500_cc benchtop_wdir/dsets/base_dset
cp 2500_cc_reproduction/exp_config_files/exp6.json benchtop_wdir/settings_files/exp1.json

# Launch the training!
echo "Starting the training..."
python benchtop.py

