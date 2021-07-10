# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 13:47:35 2021

@author: fhu14

Testing for the functionalities implemented in DataPrecompute

NOTE: This test does NOT guarantee correctness, only that the code can run through
    the fold generation and precompute processes!
    
TODO: Incorporate gammas precompute into the precompute testing
"""
#%% Imports, definittions 
import os
from FoldManager import generate_folds, save_folds, compute_graphs_from_folds, precompute_gammas
from InputParser import parse_input_dictionaries, collapse_to_master_settings, inflate_to_dict
import shutil

#%% Code behind

def run_fold_gen_test():
    allowed_Zs = [1,6,7,8]
    heavy_atoms = [1,2,3,4,5,6,7,8]
    max_config = 5
    target = {'Etot' : 'cc',
            'dipole' : 'wb97x_dz.dipole',
            'charges' : 'wb97x_dz.cm5_charges'}
    data_path = os.path.join(os.getcwd(), "test_files", "ANI-1ccx_clean_fullentry.h5")
    exclude = ['O3', 'N2O1', 'H1N1O3', 'H2']
    lower_limit = 5
    num_folds = 6
    num_folds_lower = 3
    local_fold_molecs = "fold_molecs_internal"

    print("Testing fold generation")
    folds = generate_folds(allowed_Zs, heavy_atoms, max_config, target, data_path, exclude, 
                            lower_limit, num_folds, num_folds_lower)
    save_folds(folds, local_fold_molecs)

def run_precompute_test(clear_direc: bool = True):
    print("Testing out precompute...")
    settings_filename = "test_files/settings_refactor_tst.json"
    defaults_filename = "test_files/refactor_default_tst_precomp_only.json"
    resulting_settings_obj = parse_input_dictionaries(settings_filename, defaults_filename)
    opts = inflate_to_dict(resulting_settings_obj)
    final_settings = collapse_to_master_settings(resulting_settings_obj)
    #Do the graph computation
    compute_graphs_from_folds(final_settings, "fold_molecs_internal", True)
    #Now do the gammas computation
    precompute_gammas(opts, "fold_molecs_internal")
    print("Precompute executed successfully.")
    #Delete the directory at the end. 
    if clear_direc:
        shutil.rmtree("fold_molecs_internal")
    pass

def run_fold_precomp_tests(clear_direc: bool):
    run_fold_gen_test()
    run_precompute_test(clear_direc)

#%% Main block
if __name__ == "__main__":
    run_fold_precomp_tests()
    

