# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 13:47:35 2021

@author: fhu14

Testing for the functionalities implemented in DataPrecompute

NOTE: This test does NOT guarantee correctness, only that the code can run through
    the fold generation and precompute processes!
"""
#%% Imports, definittions 
import os
from DataPrecompute import generate_folds, save_folds, compute_graphs_from_folds
from InputParser import parse_input_dictionaries, collapse_to_master_settings

#%% Code behind

def run_fold_gen_test():
    allowed_Zs = [1,6,7,8]
    heavy_atoms = [1,2,3,4,5,6,7,8]
    max_config = 10
    target = {'Etot' : 'cc',
            'dipole' : 'wb97x_dz.dipole',
            'charges' : 'wb97x_dz.cm5_charges'}
    data_path = os.path.join("ANI-1ccx_clean_fullentry.h5")
    exclude = ['O3', 'N2O1', 'H1N1O3', 'H2']
    lower_limit = 5
    num_folds = 6
    num_folds_lower = 3
    local_fold_molecs = "fold_molecs"

    print("Testing fold generation")
    folds = generate_folds(allowed_Zs, heavy_atoms, max_config, target, data_path, exclude, 
                            lower_limit, num_folds, num_folds_lower)
    save_folds(folds, local_fold_molecs)

def run_precompute_test():
    print("Testing out precompute...")
    settings_filename = "test_files/settings_refactor_tst.json"
    defaults_filename = "test_files/refactor_default_tst.json"
    resulting_settings_obj = parse_input_dictionaries(settings_filename, defaults_filename)
    final_settings = collapse_to_master_settings(resulting_settings_obj)
    compute_graphs_from_folds(final_settings, "fold_molecs", True)
    print("Precompute executed successfully.")
    pass

def run_fold_precomp_tests():
    run_fold_gen_test()
    run_precompute_test()

#%% Main block
if __name__ == "__main__":
    run_fold_precomp_tests()
    

