# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 10:45:08 2021

@author: fhu14
"""

"""
This is the top level script for calling modules written in the PaperPackage 
package. The way this works is running the script adds things into the 
PaperPackage directory, and all operations that require other modules 
(such as precomputing, training, etc.) are imported and called here.

This script will have implemented a simpler workflow implemented in separately
run-able cells.

Note to self: anything that invokes a multiprocess function later on must be wrapped
    in an if __name__ == "__main__" block.
    
    
TODO:
    1) Double-check and generate master dataset
    2) Finish code for running a training session
    3) Run training on generated master dataset ('cc' target, 300 epochs)
    4) Write + finish backend code for manipulating dataset into different forms (transfer, smaller, etc.)
    5) Run remaining runs
    6) Write + finish code for quantitative analysis of results
    7) Do analysis
    8) Write + finish code for graphical analysis of results
    9) Do analysis
"""
#%% Generate and precompute dataset (with and without reference)

from PaperPackage import create_datasets
from precompute_check import precompute_settings_check

if __name__ == "__main__":

    settings_filename = "PaperPackage/settings_refactor_tst.json"
    defaults_filename = "PaperPackage/refactor_default_tst.json"
    num_train_valid = 150
    mode = 'no_ref'
    ref_dir = None
    
    top_level_directory = "PaperPackage/master_dset"
    
    precompute_settings_check(settings_filename)
    
    create_datasets(settings_filename, defaults_filename, num_train_valid, mode, ref_dir)


#%% Do a training run using a certain dataset

from driver import run_training

settings_filename = ""
defaults_filename = ""
run_training(settings_filename, defaults_filename, skf_method = 'new')





#%% Analyze the results

