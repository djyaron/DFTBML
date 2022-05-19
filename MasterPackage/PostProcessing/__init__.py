# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 22:55:09 2022

@author: fhu14

__init__.py file for the PostProcessing package which will be used to process analysis_dir
structures to generate tables, graphs, and figures.

An analysis_dir directory has the following structures:

analysis_dir
    -> analysis_files (dir, REQUIRED)
    -> pre_check.py
    -> cc_results (dir, REQUIRED)
        -> <results_directory>
            -> Split0 (dir, REQUIRED)
    -> wt_results (dir, REQUIRED)
        -> <results_directory>
            -> Split0 (dir, REQUIRED)
    -> test_set_cc.p 
    -> test_set_wt.p

The most important part of the analysis_dir is the analysis_files directory
which contains the different numerical results from the analysis. cc_results
and wt_results directories are necessary because the loss curve information
is stored in the results directories within the Split0 subdirectories. Everything
labeled with REQUIRED above should be present for PostProcessing to be 
successful.

The following modules are contained in the PostProcessing package:
    output_parser.py: parses output text files into a more manageable format
        (dictionary) for use later on

TODO: list other modules
"""
#%% Imports 
from .output_parser import save_to_json
from .table_ops import parse_out_table, generate_deltas_table, combine_frames
from .processing_manager import generate_master_table

