# -*- coding: utf-8 -*-
"""
Created on Mon May 30 16:23:37 2022

@author: fhu14
"""
'''
Performs a series of tests on the datsets before performing the 
precomputation on them. These tests are hard-coded and based on
expectations about the different datasets
'''
#%% Imports, definitions
import pickle
import os
import numpy as np
DESTINATION = 'ALL_EXPERIMENT_DSETS'

#The following are the experiment directories which should exist in 
#   ALL_EXPERIMENT_DSETS
experiment_names = [
    'base_dset',
    #'base_dset_wt',
    'base_dset_reduced_300',
    'base_dset_reduced_300_wt',
    'base_dset_reduced_1000',
    'base_dset_reduced_1000_wt',
    #'base_dset_expanded_5000',
    #'base_dset_expanded_5000_wt',
    'base_dset_expanded_10000',
    #'base_dset_expanded_10000_wt',
    #'dset_transfer_5',
    #'dset_transfer_5_wt',
    'base_dset_expanded_10000_first_half',
    'base_dset_expanded_10000_second_half',
    ]

#%% Code behind

#%% Create the datasets

from DatasetGeneration import generate_dset, create_larger_dataset, create_smaller_training_set,\
    create_comparative_datasets, create_transfer_dataset, copy_dset

# #Create the base dataset with random sampling
# dset_name = 'base_dset'
# num_train = 2500
# num_test = 10_000
# num_train_valid_forms = 250
# max_config = 15
# generate_dset(dset_name, num_train, num_test, num_train_valid_forms, mode = 'random',
#               max_config = max_config)
# print()

# #Create the larger dataset, 5_000, with random sampling
# num_train = 5_000
# max_config = 30
# create_larger_dataset(num_train, "ALL_EXPERIMENT_DSETS/base_dset", mode = 'random', max_config = max_config)
# print()
# num_train = 10_000
# max_config = 60
# create_larger_dataset(num_train, "ALL_EXPERIMENT_DSETS/base_dset", mode = 'random', max_config = max_config)
# print()
num_train = 20_000
max_config = 150
create_larger_dataset(num_train, "D:DFTBTORCH_DATA_REPOSITORY/main_results_dsets/new_version/base_dset", mode = 'random', max_config = max_config)

#Create the samller datasets cc, 1_000 and 300
# num_train = 1_000
# max_config = 5
# create_smaller_training_set(num_train, "ALL_EXPERIMENT_DSETS/base_dset", mode = 'random', max_config = max_config)
# print()
# num_train = 300
# max_config = 2
# create_smaller_training_set(num_train, "ALL_EXPERIMENT_DSETS/base_dset", mode = 'random', max_config = max_config)
# print()

#Copy the cc datasets into the wt version for 1000 and 300
# copy_dset("ALL_EXPERIMENT_DSETS/base_dset_reduced_300")
# print()
# copy_dset("ALL_EXPERIMENT_DSETS/base_dset_reduced_1000")
# print()

#Create the transfer datasets with cutoff number 5
# dset_name = 'dset_transfer_5'
# num_train = 2500
# num_test = 10_000
# max_config = 30
# create_transfer_dataset(dset_name, num_train, num_test, 5, mode = 'random', max_config = max_config)
# print()

#Create comparative dataset by splitting the 10_000 molecule dataset 
#   just for the cc version
# create_comparative_datasets("ALL_EXPERIMENT_DSETS/base_dset_expanded_10000", split_method = 'config')
# print()

#%% Testing the created datasets

#Use the functions implemented in the .util module of the DatasetGeneration
#   package! Check the following things:
#   1) Test set equivalence across datasets (excluding transfer)
#   2) Non-overlaps between test and train/val along with no config overlap
#       between train and val
#   3) Consistent appearance of empirical formulas (though some get cut off
#       because of internal list slicing)
#   4) Between every cc and wt set, there is strict equivalence between
#       the training, validation, and test sets (excluding energy target)
#   5) 80/20 length check tests: Check that the training and validation sets have the 
#       correct lengths
#   6) For the comparative datasets, check that they have non-overlapping
#       empirical formulas
#   7) For the transfer datasets, ensure that there is a correct partitioning
#       w.r.t the number of heavy atoms for train/valid versus test
#   8) Make sure that all the datasets are sufficiently randomized w.r.t the 
#       training and validation molecules. Test molecules is irrelevant

from DatasetGeneration import test_strict_molecule_set_equivalence

#Check that all the experiments exist
all_files = os.listdir(DESTINATION)
all_directories = list(filter(lambda x : os.path.isdir(os.path.join(DESTINATION, x)), all_files))
assert(sorted(all_directories) == sorted(experiment_names))

cc_directories = [directory for directory in all_directories if 'wt' not in directory]
wt_directories = [directory for directory in all_directories if 'wt' in directory]
assert(cc_directories != wt_directories)
assert(sorted(cc_directories + wt_directories) == sorted(experiment_names))

### TEST SET TESTS ###

#Check that all the datasets have the same test set except for the 
#   transfer datasets. Also take into account the differences between
#   'wt' and 'cc' variants of test sets. Exact equivalence with targets
#   when comparing 'cc' to 'cc' and same for 'wt', but equivalence excluding
#   targets when comparing 'cc' to 'wt'

#First doing the test set for cc to cc
sets_passed = 0

cc_base_set = os.path.join(DESTINATION, "base_dset", "test_set.p")
for cc_direc in cc_directories:
    #Will deal with the transfer datasets later
    if 'transfer' not in cc_direc:
        print(f"Testing {cc_direc}")
        test_set = os.path.join(DESTINATION, cc_direc, 'test_set.p')
        test_strict_molecule_set_equivalence(cc_base_set, test_set, include_targets = True)
        sets_passed += 1

#Now do the same with wt (UNCOMMENT LATER)
# wt_base_set = os.path.join(DESTINATION, "base_dset_wt", "test_set.p")
# for wt_direc in wt_directories:
#     if 'transfer' not in wt_direc:
#         print(f"Testing {wt_direc}")
#         test_set = os.path.join(DESTINATION, wt_direc, 'test_set.p')
#         test_strict_molecule_set_equivalence(wt_base_set, test_set, include_targets = True)
#         sets_passed += 1

#Lastly, check that the two sets for the transfer experiments are good
# cc_transfer = "dset_transfer_5"
# wt_transfer = cc_transfer + "_wt"

# cc_transfer_test_set = os.path.join(DESTINATION, cc_transfer, 'test_set.p')
# wt_transfer_test_set = os.path.join(DESTINATION, wt_transfer, 'test_set.p')
# test_strict_molecule_set_equivalence(cc_transfer_test_set, wt_transfer_test_set, 
#                                  include_targets = False)
# sets_passed += 2

assert(sets_passed == 6)
print()

sets_passed = 0
#Check across all directories excluding the transfer datasets. This check includes
#   the comprative datasets
cc_base_set = os.path.join(DESTINATION, 'base_dset', 'test_set.p')
for directory in all_directories:
    if 'transfer' not in directory:
        print(f"Testing {directory}")
        test_set_path = os.path.join(DESTINATION, directory, 'test_set.p')
        test_strict_molecule_set_equivalence(cc_base_set, test_set_path,
                                             include_targets = False)
        sets_passed += 1

assert(sets_passed == 8)

print("Test set check passed")
print()

### NON-OVERLAP TESTS ###

#Check that all the datasets have no overlaps in empirical formula
#   between the training + validation and test sets and
#   no overlap b/w training + validation w.r.t molecular configurations
from DatasetGeneration import name_nonoverlap, name_config_nonoverlap

sets_passed = 0
for directory in all_directories:
    print(f"Testing {directory}")
    #Checking no overlaps for the test set
    name_nonoverlap(os.path.join(DESTINATION, directory))
    #Checking no overlaps for the training + validation sets w.r.t configurations
    name_config_nonoverlap(os.path.join(DESTINATION, directory))
    sets_passed += 1

assert(sets_passed == 8)
print("Name non-overlap test passed")
print()

### EMPIRICAL FORMULAS TESTS ###

#Check that, excluding the transfer and comparative datasets, all the 
#   training and validation sets have the same set of empirical formulas.
#   This test is invalid under random sampling scheme because
#   you can get subsets of empirical formulas
sets_passed = 0
base_dset = 'base_dset'
base_training_mols = pickle.load(open(os.path.join(os.getcwd(), DESTINATION, base_dset, 
                                                   'Fold0_molecs.p'), 'rb'))
base_validation_mols = pickle.load(open(os.path.join(os.getcwd(), DESTINATION, base_dset, 
                                                   'Fold1_molecs.p'), 'rb'))
base_training_names = set([mol['name'] for mol in base_training_mols])
base_validation_names = set([mol['name'] for mol in base_validation_mols])
assert(base_validation_names.difference(base_training_names) == set())

for direc in all_directories:
    if ('transfer' not in direc) and ('half' not in direc):
        print(f"Testing {direc}")
        curr_train_mols = pickle.load(open(os.path.join(os.getcwd(), DESTINATION, 
                                                    direc, 'Fold0_molecs.p'), 'rb'))
        curr_valid_mols = pickle.load(open(os.path.join(os.getcwd(), DESTINATION, 
                                                    direc, 'Fold1_molecs.p'), 'rb'))
        curr_training_names = set([mol['name'] for mol in curr_train_mols])
        curr_valid_names = set([mol['name'] for mol in curr_valid_mols])
        if (len(curr_valid_mols) < len(curr_train_mols)) and (len(curr_train_mols) > 1000):
        #     #When we get too small, these checks can get iffy
        #     assert(curr_valid_names.difference(curr_training_names) == set())
            assert(curr_training_names == base_training_names)
        # assert(curr_valid_names == base_validation_names)
        sets_passed += 1

# cc_transfer = 'dset_transfer_5' UNCOMMENT LATER!
# wt_transfer = cc_transfer + '_wt'
# cc_transfer_train_set = os.path.join(DESTINATION, cc_transfer, 'Fold0_molecs.p')
# cc_transfer_valid_set = os.path.join(DESTINATION, cc_transfer, 'Fold1_molecs.p')
# wt_transfer_train_set = os.path.join(DESTINATION, wt_transfer, 'Fold0_molecs.p')
# wt_transfer_valid_set = os.path.join(DESTINATION, wt_transfer, 'Fold1_molecs.p')

# cc_transfer_train_mols = pickle.load(open(cc_transfer_train_set, 'rb'))
# cc_transfer_valid_mols = pickle.load(open(cc_transfer_valid_set, 'rb'))
# wt_transfer_train_mols = pickle.load(open(wt_transfer_train_set, 'rb'))
# wt_transfer_valid_mols = pickle.load(open(wt_transfer_valid_set, 'rb'))

# cc_transfer_train_names = set([mol['name'] for mol in cc_transfer_train_mols])
# cc_transfer_valid_names = set([mol['name'] for mol in cc_transfer_valid_mols])
# wt_transfer_train_names = set([mol['name'] for mol in wt_transfer_train_mols])
# wt_transfer_valid_names = set([mol['name'] for mol in wt_transfer_valid_mols])

# assert(cc_transfer_valid_names.difference(cc_transfer_train_names) == set())
# assert(wt_transfer_valid_names.difference(wt_transfer_train_names) == set())
# assert(cc_transfer_train_names == wt_transfer_train_names)
# assert(cc_transfer_valid_names == wt_transfer_valid_names)

# sets_passed += 2

assert(sets_passed == 6)
print("Empirical formula tests passed")
print()

### CC WT STRICT EQUIVALENCE TEST ###

#Check to make sure that between every cc and wt dataset, there is 
#   strict molecule equivalence between their training, validation, and 
#   test sets

assert(sorted(all_directories) == sorted(experiment_names))
filtered_datasets = [dset for dset in all_directories if 'half' not in dset]
print(f"Checking the following datasets {filtered_datasets}")

cc_dsets = [dset for dset in filtered_datasets if 'wt' not in dset]
sets_passed = 0

for cc_set in cc_dsets:
    wt_set = cc_set + "_wt"
    #Construct the cc paths
    cc_train = os.path.join(os.getcwd(), DESTINATION, cc_set, 'Fold0_molecs.p')
    cc_valid = os.path.join(os.getcwd(), DESTINATION, cc_set, 'Fold1_molecs.p')
    cc_test = os.path.join(os.getcwd(), DESTINATION, cc_set, 'test_set.p')
    #Construct the wt paths
    wt_train = os.path.join(os.getcwd(), DESTINATION, wt_set, 'Fold0_molecs.p')
    wt_valid = os.path.join(os.getcwd(), DESTINATION, wt_set, 'Fold1_molecs.p')
    wt_test = os.path.join(os.getcwd(), DESTINATION, wt_set, 'test_set.p')
    #Test molecular equivalence for each one
    test_strict_molecule_set_equivalence(cc_train, wt_train, False)
    test_strict_molecule_set_equivalence(cc_valid, wt_valid, False)
    test_strict_molecule_set_equivalence(cc_test, wt_test, False)
    #This test occurs in pairs so add 2 to sets_passed
    sets_passed += 2

assert(sets_passed == 12)
print("cc wt strict equivalence test passed")
print()

### 80/20 LENGTH CHECKS AND NHEAVY CHECKS###
from FoldManager import count_nheavy

#Make sure you have the correct number of molecules per dataset
#   for the training and validation sets
sets_passed = 0
#First check that all the test sets have a length of 10_000
for directory in all_directories:
    test_set_path = os.path.join(os.getcwd(), DESTINATION, directory, 'test_set.p')
    test_set_mols = pickle.load(open(test_set_path, 'rb'))
    assert(len(test_set_mols) == 10_000)

#Check the lengths based on the following manifest
experiment_set_sizes = [
    ('base_dset', 2500, 625),
    ('base_dset_wt', 2500, 625),
    ('base_dset_reduced_300', 300, 625),
    ('base_dset_reduced_300_wt', 300, 625),
    ('base_dset_reduced_1000', 1000, 625),
    ('base_dset_reduced_1000_wt', 1000, 625),
    ('base_dset_expanded_5000', 5000, 1250),
    ('base_dset_expanded_5000_wt', 5000, 1250),
    ('base_dset_expanded_10000', 10_000, 2500),
    ('base_dset_expanded_10000_wt', 10_000, 2500),
    ('dset_transfer_5', 2500, 625),
    ('dset_transfer_5_wt', 2500, 625),
    ('base_dset_expanded_10000_first_half', 4900, 2500),
    ('base_dset_expanded_10000_second_half', 4900, 2500)
    ]

#This test does not work when doing random sampling
for dset, t_len, v_len in experiment_set_sizes:
    print(f"Testing {dset}")
    t_mol = pickle.load(open(os.path.join(os.getcwd(), DESTINATION, dset, 'Fold0_molecs.p'), 'rb'))
    v_mol = pickle.load(open(os.path.join(os.getcwd(), DESTINATION, dset, 'Fold1_molecs.p'), 'rb'))
    test_mol = pickle.load(open(os.path.join(os.getcwd(), DESTINATION, dset, 'test_set.p'), 'rb'))
    assert(len(t_mol) == t_len)
    assert(len(v_mol) == v_len)
    #Test set should always have 1 -> 8, except for the transfer datasets
    if 'transfer' not in dset:
        test_nheavy = list(map(lambda x : count_nheavy(x), test_mol))
        assert(min(test_nheavy) == 1 and max(test_nheavy) == 8)
    if 'transfer' not in dset:
        #This test does not work when randomly sampling
        print(f"Additional testing for {dset} on nheavy count")
        t_nheavy = list(map(lambda x : count_nheavy(x), t_mol))
        v_nheavy = list(map(lambda x : count_nheavy(x), v_mol))
        assert(min(t_nheavy) == 1 and max(t_nheavy) == 8)
        assert(min(v_nheavy) == 1 and max(v_nheavy) == 8)
        
    
    sets_passed += 1
    
assert(sets_passed == 14)
print("Length checks passed")
print()

### COMAPRATIVE DSET RECONSTRUCTION ###

#Make sure that the comparative datasets have disjoint sets of empirical formulas
#   and that they can be used to reconstruct the names in the base dataset

total_training_path = os.path.join(os.getcwd(), DESTINATION, "base_dset_expanded_10000", "Fold0_molecs.p")
total_training_mols = pickle.load(open(total_training_path, 'rb'))
total_training_names = set([mol['name'] for mol in total_training_mols])

first_half_training_path = os.path.join(os.getcwd(), DESTINATION, "base_dset_expanded_10000_first_half", "Fold0_molecs.p")
first_half_training_mols = pickle.load(open(first_half_training_path, 'rb'))
first_half_training_names = set([mol['name'] for mol in first_half_training_mols])

second_half_training_path = os.path.join(os.getcwd(), DESTINATION, "base_dset_expanded_10000_second_half", "Fold0_molecs.p")
second_half_training_mols = pickle.load(open(second_half_training_path, 'rb'))
second_half_training_names = set([mol['name'] for mol in second_half_training_mols])

assert(first_half_training_names.union(second_half_training_names) == total_training_names)
assert(first_half_training_names.intersection(second_half_training_names) == set())
print("Comparative dataset check passed")
print()

### TRANSFER DATASET CHECK ###

#Makes sure that the transfer dataset is properly formatted. This is the final
#   check for now!

transfer_dsets = ['dset_transfer_5', 'dset_transfer_5_wt']
for dset in transfer_dsets:
    training_mols = pickle.load(open(os.path.join(os.getcwd(), DESTINATION, dset, 'Fold0_molecs.p'), 'rb'))
    validation_mols = pickle.load(open(os.path.join(os.getcwd(), DESTINATION, dset, 'Fold1_molecs.p'), 'rb'))
    test_mols = pickle.load(open(os.path.join(os.getcwd(), DESTINATION, dset, 'test_set.p'), 'rb'))
    train_valid_mols = training_mols + validation_mols
    train_val_nheavy = [count_nheavy(x) for x in train_valid_mols]
    test_nheavy = [count_nheavy(x) for x in test_mols]
    assert(min(train_val_nheavy) == 1)
    assert(max(train_val_nheavy) == 5)
    assert(min(test_nheavy) == 6)
    assert(max(test_nheavy) == 8)

print("Transfer dataset test completed")
print()

### CHECK FOR RANDOMIZATION ###

#This is a fairly minimal check for randomization

sets_passed = 0
for dset in all_directories:
    training_mol_path = os.path.join(os.getcwd(), DESTINATION, dset, 'Fold0_molecs.p')
    validation_mol_path = os.path.join(os.getcwd(), DESTINATION, dset, 'Fold1_molecs.p')
    
    train_mols = pickle.load(open(training_mol_path, 'rb'))
    validation_mols = pickle.load(open(validation_mol_path, 'rb'))
    
    tst_formula = train_mols[0]['name']
    
    train_indices = []
    for i, mol in enumerate(train_mols, 0):
        if mol['name'] == tst_formula:
            train_indices.append(i)
    
    if len(train_indices) > 1:
        assert(sorted(train_indices) != list(range(min(train_indices), max(train_indices) + 1)))
    
    valid_indices = []
    for i, mol in enumerate(validation_mols, 0):
        if mol['name'] == tst_formula:
            valid_indices.append(i)
    
    if len(valid_indices) > 1:
        assert(sorted(valid_indices) != list(range(min(valid_indices), max(valid_indices) + 1)))
    
    sets_passed += 1

assert(sets_passed == 14)

print("Randomization test passed")
print()


print("ALL TESTS PASSED, CONTINUE ONTO PRECOMPUTE")

#%% Precompute checks

from DatasetGeneration import populate_settings_files, check_dataset_paths,\
    precompute_datasets, perform_precompute_settings_check

#Four little commands, what could possibly go wrong?
populate_settings_files()
print()
check_dataset_paths()
print()
perform_precompute_settings_check()

#%% Precompute (manually per dset)

#The precompute process has to be done one dataset at a time because of 
#   the multiprocessing stuff in the backend

#SEE NEW FILE FOR THIS PRECOMPUTE STUFF

# from precompute_driver import precompute_folds
# from DatasetGeneration import process_settings_files

# current_default_file = "ALL_EXPERIMENT_DSETS/refactor_default_tst.json"
# current_settings_file = "ALL_EXPERIMENT_DSETS/base_dset/dset_settings.json"
# s_obj, opts = process_settings_files(current_settings_file, current_default_file)

# assert(s_obj.top_level_fold_path == "ALL_EXPERIMENT_DSETS/base_dset")
# assert(s_obj.spline_deg == 5)

# precompute_folds(s_obj, opts, s_obj.top_level_fold_path, True)