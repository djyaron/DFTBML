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
    1) Double-check and generate master dataset (X) 
    2) Finish code for running a training session (X)
    3) Run training on generated master dataset ('cc' target, 300 epochs) (X)
    4) Write + finish backend code for manipulating dataset into different forms (transfer (X), smaller(X))
    5) Run transfer training
    5) Run smaller training (1000) (X)
    5) Run smaller training (300) (X)
    6) Write + finish code for quantitative analysis of results (X)
    7) Do analysis
    8) Write + finish code for graphical analysis of results
    9) Do analysis
"""
#%% Generate and precompute dataset (with and without reference)

from PaperPackage import create_datasets
from precompute_check import precompute_settings_check
from PaperPackage import split_to_comparative_dset, comparative_dset_check
from PaperPackage import precompute_comparative_datasets, expand_dataset

if __name__ == "__main__":

    settings_filename = "PaperPackage/dset_settings.json"
    defaults_filename = "PaperPackage/refactor_default_tst.json"
    # num_train_valid = 150
    # mode = 'no_ref'
    # ref_dir = None
    
    # top_level_directory = "PaperPackage/master_dset"
    
    # precompute_settings_check(settings_filename)
    
    # x = input("Did you check the necessary fields? (Y) ")
    
    # create_datasets(settings_filename, defaults_filename, num_train_valid, mode, ref_dir)
    
    expand_dataset(settings_filename, defaults_filename, "PaperPackage/master_dset_wt_ener_target")
    
    # split_to_comparative_dset("PaperPackage/master_dset_expanded_cc")
    
    # comparative_dset_check(
    #     ["PaperPackage/master_dset_expanded_cc_first_half",
    #       "PaperPackage/master_dset_expanded_cc_second_half"],
    #     "PaperPackage/master_dset_expanded_cc"
    #     )
    
    # location = "PaperPackage/master_dset_expanded_cc_second_half"
    # precompute_comparative_datasets(location, settings_filename,
    #                                 defaults_filename)
    
    
    #Add arguments here to precompute for the first and second half datasets!
    
    #Quick check of dset_settings.json top level fold path; makes sure the
    #   top_level_fold_path matches the directory the dset_settings.json file
    #   is contained in
    
    # import os, json
    
    # directories_to_check = ["master_dset_expanded_cc", "master_dset_expanded_cc_first_half", "master_dset_expanded_cc_second_half"]
    
    # for direc in directories_to_check:
    #     full_path = os.path.join(os.getcwd(), "PaperPackage", direc, "dset_settings.json")
    #     print("Checking", direc)
    #     with open(full_path, 'r') as jfile:
    #         jdict = json.load(jfile)
    #         assert(jdict['loaded_data_fields']['top_level_fold_path'] == f"PaperPackage/{direc}")

#%% Generate transfer dataset
# from PaperPackage import create_transfer_dataset
# from precompute_check import precompute_settings_check

# if __name__ == "__main__":

#     lower_limit = 5
#     parent_dir = "PaperPackage/master_dset"
#     settings_filename = "PaperPackage/master_dset/dset_settings.json"
#     precompute_settings_check(settings_filename)
#     x = input("Did you check the necessary fields? (Y) ")
#     create_transfer_dataset(lower_limit, parent_dir)

#%% Create smaller dataset
# from PaperPackage import create_smaller_dataset
# from precompute_check import precompute_settings_check

# if __name__ == "__main__":
    
#     size = 300
#     parent_dir = "PaperPackage/master_dset"
#     settings_filename = "PaperPackage/master_dset/dset_settings.json"
#     precompute_settings_check(settings_filename)
#     x = input("Did you check the necessary fields? (Y) ")
#     create_smaller_dataset(size, parent_dir)

#%% Dataset name non-overlap

#Ensures test set names do not overlap with training and validation sets. 

# import pickle, os

# dset_names = ["master_dset", "master_dset_reduced_300", "master_dset_reduced_1000", "master_dset_transfer_5",
#               "master_dset_wt_ener_target"]
# for name in dset_names:
#     full_path = os.path.join(os.getcwd(), "PaperPackage", name)
#     mols_0 = pickle.load(open(os.path.join(full_path, "Fold0_molecs.p"), "rb"))
#     mols_1 = pickle.load(open(os.path.join(full_path, "Fold1_molecs.p"), "rb"))
#     mols_t = pickle.load(open(os.path.join(full_path, "test_set.p"), "rb"))
#     train_valid_mols = mols_0 + mols_1
#     train_valid_names = [mol['name'] for mol in train_valid_mols]
#     test_names = [mol['name'] for mol in mols_t]
#     try:
#         assert(set(train_valid_names).intersection(set(test_names)) == set())
#         print(f"Passed name intersection testing for {name}")
#     except:
#         print(f"NAME INTERSECTION TESTING FAILED ON {name}")
    
    
#%% Dataset equivalence
#Ensures configurations are equivalent and in the same order between two datasets.
#datasets here refer to molecule pickle files. The target values can differ but the 
#coordinates should be the same. This check is very useful when trying to generate a 
#dataset with a reference dataset as a template

# import pickle
# import numpy as np

# dset_1_name = "PaperPackage/master_dset/test_set.p"
# dset_2_name = "PaperPackage/master_dset_300_epoch_run/test_set.p"

# dset1 = pickle.load(open(dset_1_name, 'rb'))
# dset2 = pickle.load(open(dset_2_name, 'rb'))

# assert(len(dset1) == len(dset2))
# for i, mol in enumerate(dset1):
#     assert(mol['name'] == dset2[i]['name'])
#     assert(all(mol['atomic_numbers'] == dset2[i]['atomic_numbers']))
#     assert(np.allclose(mol['coordinates'], dset2[i]['coordinates']))
#     # for target in mol['targets']:
#     #     assert(np.allclose(mol['targets'][target], dset2[i]['targets'][target]))

# print(f"Dataset equivalence check passed between {dset_1_name} and {dset_2_name}")

#%% Dataset equivalence non-ordered

#Ensures configurations and names are equivalent between two datasets, not necessarily order
# or targets (useful for testing against datasets generated with reference for alternative
# targets)

# import pickle
# import os
# import numpy as np

# dset_1_name = "PaperPackage/master_dset_transfer_5_wt_ener_target/Fold1_molecs.p"
# dset_2_name = "PaperPackage/master_dset_transfer_5/Fold1_molecs.p"

# dset1 = pickle.load(open(dset_1_name, 'rb'))
# dset2 = pickle.load(open(dset_2_name, 'rb'))

# assert(len(dset1) == len(dset2))

# name_confs_1 = [(mol['name'], mol['iconfig']) for mol in dset1]
# name_confs_2 = [(mol['name'], mol['iconfig']) for mol in dset2]

# assert(len(name_confs_1) == len(set(name_confs_1)))
# assert(len(name_confs_2) == len(set(name_confs_2)))

# mol_dict_1 = {(mol['name'], mol['iconfig']) : mol for mol in dset1}

# for mol in dset2:
#     mol_1 = mol_dict_1[(mol['name'], mol['iconfig'])]
#     assert(mol['name'] == mol_1['name'])
#     assert(mol['iconfig'] == mol_1['iconfig'])
#     assert(all(mol['atomic_numbers'] == mol_1['atomic_numbers']))
#     assert(np.allclose(mol['coordinates'], mol_1['coordinates']))

# #No target equivalence
# print(f"Dataset reference equivalence check passed between {dset_1_name} and {dset_2_name}")


#%% Do a training run using a certain dataset

# from driver import run_training
# from precompute_check import precompute_settings_check
# import json, os, shutil

# settings_filename = "PaperPackage/settings_refactor_tst.json"
# defaults_filename = "PaperPackage/refactor_default_tst.json"

# precompute_settings_check(settings_filename)
# x = input("Did you check the necessary fields? (Y) ")
# y = input("Did you check the run id? (Y) ")

# run_training(settings_filename, defaults_filename, skf_method = 'new')

# settings_full_path = os.path.join(os.getcwd(), settings_filename)
# with open(settings_full_path, 'r') as handle:
#     d = json.load(handle)

# result_dir = d['run_id']
# source = os.path.join(os.getcwd(), result_dir)
# destination = os.path.join(os.getcwd(), "PaperPackage", result_dir)
# if os.path.isdir(destination):
#     shutil.rmtree(destination)
# shutil.move(source, destination)

# dset_source = d['loaded_data_fields']['top_level_fold_path']

# #In copying split information, goes from top_level_fold_path/Split{num} --> run_id/Split{num}

# num_splits = len(d['training_settings']['split_mapping'])
# for i in range(num_splits):
#     src = os.path.join(dset_source, f"Split{i}")
#     dst = os.path.join(destination, f"Split{i}")
#     if os.path.exists(dst):
#         print("Removing existing directory")
#         shutil.rmtree(dst)
#     shutil.copytree(src, dst)

# print("Copying settings file")
# shutil.copy(settings_filename, os.path.join(destination, "settings_refactor_tst.json"))
# print("All information copied")

#%% Analyze the results
'''
All training now happens on cat1 which is faster than running DFTB+ calculations
locally
'''
# from DFTBPlus import add_dftb, compute_results_torch_newrep, compute_results_torch
# from Auorg_1_1 import ParDict
# import pickle, os, random
# from FoldManager import count_nheavy

# pardict = ParDict()
# exec_path = "C:\\Users\\fhu14\\Desktop\\DFTB17.1Windows\\DFTB17.1Windows-CygWin\\dftb+"
# test_set_path = "PaperPackage/master_dset_reduced_300/test_set.p"
# skf_dir = os.path.join(os.getcwd(), "PaperPackage", "master_dset_300_epoch_run")
# ref_param_path = skf_dir + "/ref_params.p"

# ref_params = pickle.load(open(ref_param_path, 'rb'))
# test_set = pickle.load(open(test_set_path, 'rb'))

# random.shuffle(test_set)

# test_set = test_set[:100] #For testing purposes


# add_dftb(test_set, skf_dir, exec_path, pardict, parse = "detailed")

# coef = ref_params['coef']
# intercept = ref_params['intercept'][0]
# atype_ordering = ref_params['atype_ordering']
# allowed_Zs = list(atype_ordering)
# target = "Etot"

# diffs, err = compute_results_torch_newrep(test_set, target, allowed_Zs, 
#                                   atype_ordering, coef, intercept, error_metric = "MAE") #Diffs are not absolute

# print(err * 627)

# # diffs, err = compute_results_torch(test_set, target, allowed_Zs)










