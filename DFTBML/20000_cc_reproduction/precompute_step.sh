#!/bin/bash

# This script runs through the precompute stage of the process. 
# Only execute this on a cluster with adequate memory. 

# Copy this script into the DFTBML directory (one level above 20000_cc_reproduction)
# before running.

# Activate the virtual environment created during setup
conda activate DFTBML

# Copy and execute the precomputation
echo "Starting precomputation..."
cp 20000_cc_reproduction/precompute_run_script.py .
python precompute_run_script.py


