# -*- coding: utf-8 -*-
"""
Created on Sun May 29 13:20:53 2022

@author: fhu14
"""

"""
Code to create datasets based off the full_master_dset.p file which should 
exist.

It is generally assumed that the 'cc' version of a dataset is first, and the 
'wt' version comes second or is copied. 

For generating larger datasets, the 'cc' and 'wt' datasets are generated at the 
same time since it invokes the generate_dset() method. 

For generating smaller datasets, the 'cc' is generated first and the 'wt' is
copied over with the 'wt' energy target. The copy operation is based on the 
molecule name and configuration number
"""

#%% Imports, definitions
import os, pickle
import random
from typing import List, Dict, Tuple
from math import ceil
from copy import deepcopy
from .util import target_subdict_correction, save_dset_mols, shuffle_dict,\
    copy_molecule_set
from FoldManager import count_nheavy
from functools import reduce

DESTINATION = "ALL_EXPERIMENT_DSETS" #all datasets are saved here
#%% Code behind

def generate_dset(dset_name: str, num_train: int, num_test: int,
                  num_train_valid_forms: int, fix_O3: bool = True, 
                  reference_train_valid_forms: List[str] = None,
                  test_set_mol_path: str = None,
                  mode: str = 'random', max_config: int = None) -> None:
    r"""Generates a dataset with the specified parameters, and has the ability to
        generate for both 'cc' and 'wt' energy targets.
    
    Arguments:
        dset_name (str): The name of the dataset to generate with 'cc' energy target
        num_train (int): The nunmber of training molecules
        num_test (int): The number of testing molecules
        num_train_valid_form (int): The number of empirical formulas for 
            training and validation
        fix_O3 (bool): Whether or not to include O3 configurations in
            the training data. Defaults to True
        refrence_train_valid_forms (List[str]): The list of training and
            validation empirical formulas for generating larger datasets.
            Defaults to None
        test_set_mol_path (str): The path to the test set molecules to copy. 
            Defaults to None.
        mode (str): The approach used for generating the dataset. One of 
            'random' and 'uniform', defaults to 'random'
        max_config (int): The maximum number of configurations to use for each 
            empirical formula when doing random sampling
    
    Returns:
        train_mols (List[Dict]): The training molecules
        valid_mols: (List[Dict]): The validation molecules
        test_mols: (List[Dict]): The test molecules
    
    Notes: The empirical formulas used for training and validation are drawn 
        using a random sample without replacement from the set of all names 
        in the dataset dictionary. Only the number of training molecules is
        specified because of the 80/20 split rule, where the total number of 
        molecules is:
            
            num_tot = ceil(num_train / 0.8)
        
        And num_valid is 0.2 * num_tot. It is recommended that the value of 
        num_train is chosen so that both num_valid and num_train are divisible by 
        num_train_valid_forms because that ensures we get an even distribution
        of molecules across all the chosen empirical formulas while also reaching
        the target dataset size. Note that the training and validation sets
        are NOT separated by empirical formula but the test set IS completely
        separated from training + validation by empirical formula. 
        
        generate_dual should always be True to ensure that both the 'cc' and
        'wt' version of the dataset are generated at once. This ensures consistency
        between the two. 
        
        fix_O3 is an option because 'O3' was a formula that was problematic earlier
        on in dftbtorch development. That has since been addressed and including 
        O3 in the training data should not throw any errors.
        
        The mode is one of either 'random' or 'uniform'. In the uniform scheme, 
        the above approach is taken where approximately the same number of configurations
        from each empirical formula is included in the dataset to form the overall
        dataset. In the random scheme, the number of data points required is 
        randomly sampled from the set of all molecules with the appropriate 
        empirical formulas. Experimentally, it seems that the 'random' scheme
        yields the best outcome.
        
        Blind random sampling is problematic because the number of configurations
        for each empirical formula is unbalanced. This can lead to bad random 
        samples. In an attempt to equalize out the population of configurations,
        the max_config argument is given to cap out the number of configurations to
        include for each formula before random sampling.
    """
    #Pick out the names for the training/validation and test set
    dictionary_path = os.path.join(os.getcwd(), DESTINATION, "full_master_dset_dict.p")
    all_molecule_dictionary = pickle.load(open(dictionary_path, 'rb'))
    all_names = list(all_molecule_dictionary.keys())
    if reference_train_valid_forms is None:
        if fix_O3:
            tmp = [elem for elem in all_names if elem != 'O3']
            train_valid_forms = random.sample(tmp, num_train_valid_forms - 1)
            train_valid_forms += ["O3"]
        else:
            train_valid_forms = random.sample(all_names, num_train_valid_forms)
    else:
        train_valid_forms = reference_train_valid_forms
        assert(num_train_valid_forms == len(train_valid_forms))
    test_forms = [formula for formula in all_names if formula not in train_valid_forms]
    assert(len(test_forms) + len(train_valid_forms) == len(all_molecule_dictionary))
    #Ensure no empirical formula overlap
    assert(set(train_valid_forms).intersection(set(test_forms)) == set())
    
    #Do some arithmetic to figure out the number of configurations required
    num_valid = 0.2 * (num_train / 0.8)
    num_valid = int(num_valid)
    
    #Both code paths should generate the lists train_mols and valid mols. The test
    #   set is always generated uniformly across empirical formulas
    if mode == 'uniform':
        print("Generating using uniform scheme")
        num_train_config = ceil(num_train / num_train_valid_forms)
        num_valid_config = ceil(num_valid / num_train_valid_forms)
        
        #Now extract the training and validation molecules in the first pass
        train_mols, valid_mols = [], []
        good_tv_forms = 0 #Keep track of the formulas that still have configurations remaining
        for emp_formula in train_valid_forms:
            curr_mols = all_molecule_dictionary[emp_formula]
            train_mols.extend(curr_mols[:num_train_config])
            valid_mols.extend(curr_mols[num_train_config : num_train_config + num_valid_config])
            all_molecule_dictionary[emp_formula] = curr_mols[num_train_config + num_valid_config:]
            if len(all_molecule_dictionary[emp_formula]) > 0:
                good_tv_forms += 1
        
        #Fix the length if certain elements are missing
        if (len(train_mols) < num_train) or (len(valid_mols) < num_valid):
            #If the number of molecules is already greater than the required number we set the
            #   required number remaining at 0 using the max function
            remaining_train = max(num_train - len(train_mols), 0)
            remaining_val = max(num_valid - len(valid_mols), 0)
            remaining_train_configs = ceil(remaining_train / good_tv_forms)
            remaining_valid_configs = ceil(remaining_val / good_tv_forms)
            for formula in train_valid_forms:
                if len(all_molecule_dictionary[formula]) > 0:
                    curr_mols = all_molecule_dictionary[formula]
                    train_mols.extend(curr_mols[:remaining_train_configs])
                    valid_mols.extend(curr_mols[remaining_train_configs : remaining_train_configs + remaining_valid_configs])
                    all_molecule_dictionary[formula] = curr_mols[remaining_train_configs + remaining_valid_configs:]
    
    elif mode == 'random':
        print("Generating using random sampling scheme")
        assert(max_config is not None)
        print(f"Using {max_config} number of configurations per formula")
        #Safety to ensure that the populations are sufficiently large
        assert(max_config * len(train_valid_forms) >= num_train + num_valid)
        #Slice up to max_config number of configurations
        all_possible_mols = [all_molecule_dictionary[form][:max_config] for form in train_valid_forms]
        all_possible_mols_flat = list(reduce(lambda x, y : x + y, all_possible_mols))
        random.shuffle(all_possible_mols_flat)
        #first draw the training molecules
        train_mols = random.sample(all_possible_mols_flat, num_train)
        train_ncs = set((mol['name'], mol['iconfig']) for mol in train_mols)
        #construct remaining population for validation molecules
        potential_valid_mols = [mol for mol in all_possible_mols_flat if (mol['name'], mol['iconfig']) not in train_ncs]
        assert(len(potential_valid_mols) == len(all_possible_mols_flat) - num_train)
        valid_mols = random.sample(potential_valid_mols, num_valid)
    
    if test_set_mol_path is None:
        print("Generating test set uniformly")
        #Generate the test set
        #Extract the test molecules in the first pass
        num_test_config = ceil(num_test / len(test_forms))
        test_mols = []
        good_tst_forms = 0
        for emp_formula in test_forms:
            curr_mols = all_molecule_dictionary[emp_formula]
            test_mols.extend(curr_mols[:num_test_config])
            all_molecule_dictionary[emp_formula] = curr_mols[num_test_config:]
            if len(all_molecule_dictionary[emp_formula]) > 0:
                good_tst_forms += 1
        
        #Padding for test molecules
        if len(test_mols) < num_test:
            remaining_test = num_test - len(test_mols)
            remaining_test_configs = ceil(remaining_test / good_tst_forms)
            for formula in test_forms:
                if len(all_molecule_dictionary[formula]) > 0:
                    curr_mols = all_molecule_dictionary[formula]
                    test_mols.extend(curr_mols[:remaining_test_configs])
                    all_molecule_dictionary[formula] = curr_mols[remaining_test_configs:]
    
    else:
        print("Copying test set")
        #Load the test set in the case where the test set should be copied
        original_test_mols = pickle.load(open(test_set_mol_path, 'rb'))
        test_ncs = [(mol['name'], mol['iconfig']) for mol in original_test_mols]
        test_mols = []
        for name, iconfig in test_ncs:
            curr_mols = all_molecule_dictionary[name]
            for inner_mol in curr_mols:
                if inner_mol['iconfig'] == iconfig:
                    test_mols.append(inner_mol)
    
    #Quick length check
    if len(train_mols) > num_train: 
        train_mols = train_mols[:num_train]
    if len(valid_mols) > num_valid:
        valid_mols = valid_mols[:num_valid]
    if len(test_mols) > num_test:
        test_mols = test_mols[:num_test]
    
    #Shuffle the datasets
    random.shuffle(train_mols)
    random.shuffle(valid_mols)
    # No need to shuffle testing molecules
    # random.shuffle(test_mols)
    
    #The lengths should be correct
    assert(len(train_mols) == num_train)
    assert(len(valid_mols) == num_valid)
    assert(len(test_mols) == num_test)
    
    #Now copy, correct, and save
    train_wt, valid_wt, test_wt = deepcopy(train_mols), deepcopy(valid_mols), deepcopy(test_mols)
    #cc sets
    target_subdict_correction(train_mols, 'cc')
    target_subdict_correction(valid_mols, 'cc')
    target_subdict_correction(test_mols, 'cc')
    #wt sets
    target_subdict_correction(train_wt, 'wt')
    target_subdict_correction(valid_wt, 'wt')
    target_subdict_correction(test_wt, 'wt')
    
    save_dset_mols(dset_name, DESTINATION, train_mols, valid_mols, test_mols)
    save_dset_mols(dset_name + "_wt", DESTINATION, train_wt, valid_wt, test_wt)
    
    print("Base datasets generated for both cc and wt datasets")

def create_smaller_training_set(num_train: int, dset_dir: str, mode: str = 'random',
                                max_config: int = None) -> None:
    r"""Creates a smaller dataset with fewer training molecules. This method is used 
        to create the datasets for the reduced_300 and reduced_1000 experiments
    
    Arguments: 
        num_train (int): The number of training molecules to use
        dset_dir (str): The relative path to the dataset directory to shrink
        mode (str): The approach used for generating the dataset. One of 
            'random' and 'uniform', defaults to 'random'
        max_config (int): The maximum number of configurations to use for each 
            empirical formula when doing random sampling
    
    Returns:
        None
    
    Notes: There is an attempt to preserve the uniformity of the dataset by
        removing molecules systematically across all the emprical formulas.
        The only thing that we need to change is the training set.
        
        The post fix "_reduced_{num_train}" is attached onto the end of the 
        dset_dir to indicate from which dset it is derived from. 
        
        The implementation of 'random' mode in this case is just a random
        sample of the training molecules
    """
    training_mol_path = os.path.join(os.getcwd(), dset_dir, "Fold0_molecs.p")
    training_mols = pickle.load(open(training_mol_path, 'rb'))
    #Both code paths should create the variable reduced_training_mols
    if mode == 'uniform':
        print("Generating using uniform scheme")
        #Construct the dictionary
        mol_dict = {}
        for molecule in training_mols:
            curr_name = molecule['name']
            if curr_name in mol_dict:
                mol_dict[curr_name].append(molecule)
            else:
                mol_dict[curr_name] = [molecule]
        shuffle_dict(mol_dict)
        #Figure out the number of molecules needed for each empirical formula
        num_names = len(mol_dict.keys())
        num_configs = ceil(num_train / num_names)
        reduced_train_mols = []
        n_good_training_forms = 0
        for formula in mol_dict:
            curr_mols = mol_dict[formula]
            reduced_train_mols.extend(curr_mols[:num_configs])
            #Slice to ensure that the molecule dictionary is up to date for
            #   padding later on
            mol_dict[formula] = curr_mols[num_configs:]
            if len(mol_dict[formula]) > 0:
                n_good_training_forms += 1
        
        if len(reduced_train_mols) < num_train:
            num_training_remaining = num_train - len(reduced_train_mols)
            num_remaining_configs = ceil(num_training_remaining / n_good_training_forms)
            for formula in mol_dict:
                if len(mol_dict[formula]) > 0:
                    curr_mols = mol_dict[formula]
                    reduced_train_mols.extend(curr_mols[:num_remaining_configs])
    elif mode == 'random':
        print("Generating using random sampling scheme")
        assert(max_config is not None)
        #Construct the molecule dictionary
        mol_dict = {}
        for mol in training_mols:
            curr_name = mol['name']
            if curr_name in mol_dict:
                mol_dict[curr_name].append(mol)
            else:
                mol_dict[curr_name] = [mol]
        #Some checks
        mol_vals = list(mol_dict.values())
        mol_vals_flat = list(reduce(lambda x, y : x + y, mol_vals))
        assert(len(mol_vals_flat) == len(training_mols))
        
        #Do the sampling
        all_possible_mols = [mol_dict[form][:max_config] for form in mol_dict]
        all_possible_mols_flat = list(reduce(lambda x, y : x + y, all_possible_mols))
        assert(len(all_possible_mols_flat) > num_train)
        print(f"Sampling from {len(all_possible_mols_flat)}")
        random.shuffle(all_possible_mols_flat)
        reduced_train_mols = random.sample(training_mols, num_train)
    
    #Quick length check
    if len(reduced_train_mols) > num_train:
        reduced_train_mols = reduced_train_mols[:num_train]
    assert(len(reduced_train_mols) == num_train)
    #Be sure to shuffle the dataset
    random.shuffle(reduced_train_mols) 
    #Open and save the other elements to the dataset directory
    validation_mol_path = os.path.join(os.getcwd(), dset_dir, "Fold1_molecs.p")
    test_mol_path = os.path.join(os.getcwd(), dset_dir, "test_set.p")
    
    validation_mol = pickle.load(open(validation_mol_path, 'rb'))
    test_mol = pickle.load(open(test_mol_path, 'rb'))
    
    base_dset_dir = os.path.split(dset_dir)[-1]
    dset_name = base_dset_dir + f"_reduced_{num_train}"
    
    save_dset_mols(dset_name, DESTINATION, reduced_train_mols, validation_mol, test_mol)
    
    print(f"Reduced version of {dset_dir} generated with {num_train} training molecules")

def create_larger_dataset(num_train: int, dset_dir: str, mode: str = 'random',
                          max_config: int = None) -> None:
    r"""Generates a larger dataset by taking advantage of the generate_dset
        function
    
    Argument:
        num_train (int): The number of training molecules
        dset_dir (str): The path to the dataset directory this is based off
        mode (str): The approach used for generating the dataset. One of 
            'random' and 'uniform', defaults to 'random'
        max_config (int): The maximum number of configurations to use for each 
            empirical formula when doing random sampling
    
    Returns:
        None
    
    Notes: When creating larger datasets, the test set should be copied over
        so the path to the test set is passed as a parameter to generate_dset.
        That way the correct test set is copied over and reused
    """
    tst_mols = os.path.join(os.getcwd(), dset_dir, 'test_set.p')
    tst_path = tst_mols #save the test set path for later use
    train_mols = os.path.join(os.getcwd(), dset_dir, 'Fold0_molecs.p')
    valid_mols = os.path.join(os.getcwd(), dset_dir, 'Fold1_molecs.p')
    
    tst_mols = pickle.load(open(tst_mols, 'rb'))
    train_mols = pickle.load(open(train_mols, 'rb'))
    valid_mols = pickle.load(open(valid_mols, 'rb'))
    
    num_test = len(tst_mols)
    train_val_mols = train_mols + valid_mols
    train_valid_forms = list(set([mol['name'] for mol in train_val_mols]))
    num_train_valid_forms = len(train_valid_forms)
    
    dset_name = os.path.split(dset_dir)[-1]
    dset_name += f"_expanded_{num_train}"
    
    generate_dset(dset_name, num_train, num_test, num_train_valid_forms,
                  reference_train_valid_forms = train_valid_forms, test_set_mol_path = tst_path,
                  mode = mode, max_config = max_config)
    
def copy_dset(base_dset: str, energy_targ: str = 'wt') -> None:
    r"""Copies a 'cc' version dataset into a 'wt' version dataset based off the
        empirical formula and the configuration number as well as the 
        full_master_dset_dict.p dictionary
    
    Arguments:
        base_dset (str): The path of the dataset (cc version) to copy
        energy_targ (str): The energy target for the copied version of the dataset.
            defaults to 'wt'
    
    Returns:
        None
    
    Notes: This is used for copying 'cc' version datasets into their 'wt' counterparts, 
        and applies when creating smaller datasets because for those datasets,
        the 'cc' version is created first and the 'wt' version is copied over
    """
    all_molecule_dictionary = pickle.load(open(os.path.join(os.getcwd(), DESTINATION, "full_master_dset_dict.p"), 'rb'))
    training_mols = os.path.join(os.getcwd(), base_dset, "Fold0_molecs.p")
    validation_mols = os.path.join(os.getcwd(), base_dset, "Fold1_molecs.p")
    testing_mols = os.path.join(os.getcwd(), base_dset, "test_set.p")
    
    new_training_mols = copy_molecule_set(all_molecule_dictionary, training_mols, energy_targ)
    new_validation_mols = copy_molecule_set(all_molecule_dictionary, validation_mols, energy_targ)
    new_testing_mols = copy_molecule_set(all_molecule_dictionary, testing_mols, energy_targ)
    
    #Save the dataset
    dset_name = os.path.split(base_dset)[-1]
    dset_name = dset_name + f"_{energy_targ}"
    save_dset_mols(dset_name, DESTINATION, new_training_mols, new_validation_mols, new_testing_mols)
    
    print(f"Finished copying new dataset {dset_name}")

def create_transfer_dataset(dset_name: str, num_train: int, num_test: int,
                            cutoff_num: int, mode: str = 'random', max_config: int = None) -> None:
    r"""Generates a transfer dataset where the training and validation sets
        only have molecules with up to cutoff_num heavy atoms and the test
        set has molecules with strictly greater numbers of heavy atoms
    
    Arguments:
        dset_name (str): The name of the dataset to generate and save to
        num_train (int): The number of training molecules
        num_test (int): The number of testing molecules
        cutoff_num (int): The number of heavy atoms for the training and 
            validation molecules. Test molecules have more heavy atoms
        mode (str): The approach used for generating the dataset. One of 
            'random' and 'uniform', defaults to 'random'
        max_config (int): The maximum number of configurations to use for each 
            empirical formula when doing random sampling
    
    Returns:
        None
    
    Notes: The transfer dataset is constructed such that for all the molecules
        in the training and validation sets, the molecules have a number of 
        heavy atoms <= cutoff_num, but all the molecules in the test set 
        have a number of heavy atoms strictly greater than cutoff_num.
        
        This is also calling upon the generate_dset function
    """
    #For bookkeeping purposes, make sure that the cutoff number is included in the 
    #   dataset name
    assert(str(cutoff_num) in dset_name) 
    master_dict_path = os.path.join(os.getcwd(), DESTINATION, "full_master_dset_dict.p")
    master_molecule_dictionary = pickle.load(open(master_dict_path, 'rb'))
    light_formulas, heavy_formulas = [], []
    for formula in master_molecule_dictionary:
        #Test the heavy count based on the first molecule
        tst_molecule = master_molecule_dictionary[formula][0]
        num_heavy = count_nheavy(tst_molecule)
        if num_heavy <= cutoff_num:
            light_formulas.append(formula)
        else:
            heavy_formulas.append(formula)
    #Quick sanity check
    assert(len(light_formulas) + len(heavy_formulas) == len(master_molecule_dictionary.keys()))
    #Can call upon the generate_dset() function again
    num_train_valid_forms = len(light_formulas)
    generate_dset(dset_name, num_train, num_test, num_train_valid_forms, 
                  reference_train_valid_forms = light_formulas, mode = mode, max_config = max_config)

def create_comparative_datasets(base_dset: str, split_method: str = 'form') -> None:
    r"""Splits the training dataset of base_dset into halves based on the given split_method
    
    Arguments:
        base_dset (str): The path to the dataset to split into two comparable halves
        split_method (str): The method used to split the dataset into halves.
            One of 'form' and 'config'. Defaults to 'form'
        
    Returns:
        None
        
    Notes: The different split_methods have different behaviors. If you are
        splitting by the empirical formulas ('form'), then the training set
        is split in half with non-overlapping empirical formulas. To ensure that
        both datasets have the same size, they are scaled down so they match. 
        Both datasets are randomly shuffled and then scaled down to the size
        of the nearest dataset to the nearest hundred. 
    """
    training_mols_path = os.path.join(os.getcwd(), base_dset, "Fold0_molecs.p")
    training_mols = pickle.load(open(training_mols_path, 'rb'))
    assert(split_method in ['form', 'config'])
    
    #First handle the splitting by empirical formula
    if split_method == 'form':
        print("Splitting comparative datasets by empirical formula")
        all_training_names = list(set([mol['name'] for mol in training_mols]))
        first_half_names = all_training_names[:len(all_training_names) // 2]
        second_half_names = all_training_names[len(all_training_names) // 2 : ]
        #Some sanity checks
        assert(set(first_half_names).intersection(set(second_half_names)) == set())
        assert(len(first_half_names) + len(second_half_names) == len(all_training_names))
        #Separate the molecules
        first_half_mols, second_half_mols = [], []
        for mol in training_mols:
            if mol['name'] in first_half_names:
                first_half_mols.append(mol)
            else:
                second_half_mols.append(mol)
        assert(len(first_half_mols) + len(second_half_mols) == len(training_mols))
        #Round down to the nearest 100
        min_len = min(len(first_half_mols), len(second_half_mols))
        targ_len = min_len - (min_len % 100)
        random.shuffle(first_half_mols)
        random.shuffle(second_half_mols)
        #first_half_mols and second_half_mols are what are being saved
        first_half_mols = first_half_mols[:targ_len]
        second_half_mols = second_half_mols[:targ_len]
        assert(len(first_half_mols) == len(second_half_mols) == targ_len)
    
    #Also handle the configuration splitting possibility
    elif split_method == 'config':
        print("Splitting comparative datasets by configuration")
        random.shuffle(training_mols)
        first_half_mols = training_mols[:len(training_mols)//2]
        second_half_mols = training_mols[len(training_mols)//2:]
        assert(len(first_half_mols) == len(second_half_mols) == len(training_mols) // 2)
    
    #Load in the validation molecules and test sets to begin saving 
    #   everything
    validation_set = os.path.join(os.getcwd(), base_dset, "Fold1_molecs.p")
    test_set = os.path.join(os.getcwd(), base_dset, "test_set.p")
    
    validation_set = pickle.load(open(validation_set, 'rb'))
    test_set = pickle.load(open(test_set, 'rb'))
    #save the datasets
    dset_name = os.path.split(base_dset)[-1]
    save_dset_mols(dset_name + "_first_half", DESTINATION, first_half_mols, 
                   validation_set, test_set)
    save_dset_mols(dset_name + "_second_half", DESTINATION, second_half_mols, 
                   validation_set, test_set)
    print(f"Split datasets generated from base dataset {base_dset}")
    


#%% Main block