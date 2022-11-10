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
import shutil

from FoldManager import (
    compute_graphs_from_folds,
    generate_folds,
    precompute_gammas,
    precompute_gammas_per_fold,
    save_folds,
)
from InputParser import (
    collapse_to_master_settings,
    inflate_to_dict,
    parse_input_dictionaries,
)

from .h5handler_test import run_safety_check
from .helpers import ani1_path, test_data_dir

#%% Code behind


def run_fold_gen_test():
    allowed_Zs = [1, 6, 7, 8]
    heavy_atoms = [1, 2, 3, 4, 5, 6, 7, 8]
    max_config = 5
    target = {"Etot": "cc", "dipole": "wb97x_dz.dipole", "charges": "wb97x_dz.cm5_charges"}
    exclude = ["O3", "N2O1", "H1N1O3", "H2"]
    lower_limit = 5
    num_folds = 6
    num_folds_lower = 3
    local_fold_molecs = "fold_molecs_internal"

    print("Testing fold generation")
    folds = generate_folds(
        allowed_Zs,
        heavy_atoms,
        max_config,
        target,
        ani1_path,
        exclude,
        lower_limit,
        num_folds,
        num_folds_lower,
    )
    save_folds(folds, local_fold_molecs)


def run_precompute_test(clear_direc: bool = True):
    print("Testing out precompute...")
    settings_filename = os.path.join(test_data_dir, "settings_refactor_tst.json")
    defaults_filename = os.path.join(test_data_dir, "refactor_default_tst_precomp_only.json")
    resulting_settings_obj = parse_input_dictionaries(settings_filename, defaults_filename)
    opts = inflate_to_dict(resulting_settings_obj)
    final_settings = collapse_to_master_settings(resulting_settings_obj)
    # Do the graph computation
    compute_graphs_from_folds(final_settings, "fold_molecs_internal", True)
    # Now do the gammas computation for the entire dataset
    precompute_gammas(opts, "fold_molecs_internal")
    # Now do the gammas computation for each fold
    precompute_gammas_per_fold(opts, "fold_molecs_internal")
    print("Precompute executed successfully.")
    run_safety_check("fold_molecs_internal", [0, 1, 2, 3, 4, 5])
    print("Safety check successfully executed for precomputed data")
    # Delete the directory at the end.
    if clear_direc:
        shutil.rmtree("fold_molecs_internal")
    pass


def run_fold_precomp_tests(clear_direc: bool):
    run_fold_gen_test()
    run_precompute_test(clear_direc)


#%% Main block
if __name__ == "__main__":
    run_fold_precomp_tests()
