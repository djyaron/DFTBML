# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 15:16:24 2021

@author: fhu14
"""

"""
Dataset generation workflow for standardized experiments for DFTBML Organics Writeup

(Going to adhere to better coding style here compared to before)

The purpose of this module is to generate a standardized training, validation, 
and testing dataset for all the experiments used for benchmarking the 
quantitative performance of the DFTBML model. The only exception would be 
when the datasets need to be different sizes (e.g. verifying the comparative
performances between different dataset sizes), but in that case the composition
in terms of empirical formulas will be consistent. 

While the backend code is implemented to handle arbitrary combinations of 
individually saved folds, we will instead do an 80-20 split between the 
training and validation set. The test set will be a set of molecules with 
non-overlapping empirical formulas with both the training and validation sets.
The formation of the training and validation set will also be based on 
empirical formulas

Information (graph + feed info) needs to be precomputed for the training
and validation data, but no precomputation needs to be done for the test set;
these molecules can be saved directly in a pickle format.

We use the following method for generating the dataset: 
    1) Draw a large number of molecules from the ANI1-ccx dataset with 
        up to 8 heavy atoms containing C, H, N, O. Size of this initial set is 
        controlled by the max_config parameter which is passed into get_ani1data.
    2) Figure out all the names (empirical formulas) in this set.
    3) Randomly choose a subset of these empirical formulas to be contained
        within the training and validation set.
    4) The test set will contain the rest of the molecules from the original set
        whose empirical formulas are non-overlapping with the training + validation 
        set formulas. Test molecules are saved directly into a pickle file.
    5) The molecules from the original set whose empirical formulas are contained 
        in the training/validation set formulas are combined and shuffled.
    6) Precomputation is done for the training/validation set and saved in the
        standard format of a dataset. 

The first standardized dataset will be generated with the coupled cluster (cc)
energy target, but the same process can be done for other energy targets. To
keep things consistent, when a new dataset is generated with a different
energy target, the molecule names and configurations of interest are taken
from the original standardized dataset. 

For this workflow, all the correct information should be contained within the 
settings file, which will be used along with the defaults file to pass
the necessary information to different parts of the program.
"""

#%% Imports, definitions
from FoldManager import get_ani1data
from typing import List, Dict, Tuple
import os, random, pickle, shutil
from precompute_driver import precompute_folds
from InputParser import parse_input_dictionaries, collapse_to_master_settings, inflate_to_dict
PREFIX = "PaperPackage" #All datasets should be saved to this containing directory

#%% Code behind (general data generation scheme)

def generate_datasets(allowed_Z: List[int], heavy_atoms: List[int], max_config: int, 
                 target: Dict[str, str], data_path: str, exclude: List[str],
                 num_train_valid: int) -> Tuple[List[Dict]]:
    r"""Generates the training, validation, and test datasets.
    
    Arguments:
        allowed_Z (List[int]): The element numbers that molecules in the 
            dataset are allowed to contain.
        heavy_atoms (List[int]): The number of heavy (non-hydrogen) atoms the 
            molecules can contain.
        max_config (int): The maximum number of configurations for each 
            empirical formula allowed. 
        target (Dict): The target dictionary used for extracting information for 
            molecules from the ANI1-ccx dataset. 
        data_path (str): The path to the dataset h5 file.
        exclude (List[str]): The list of empirical formulas to exclude.
        num_train_valid (int): The number of empirical formulas used for 
            the training and validations.
    
    Returns:
        training_dset (List[Dict]): The training dataset
        validation_dset (List[Dict]): The validation dataset
        test_dset (List[Dict]): The test dataset
        
    Notes:
        Increasing max_config can increase the number of molecules in the 
        original set. 
        
        The complement of the set of empirical formulas for the training/validation
        dataset will be in the test dataset. 
    """
    dataset = get_ani1data(allowed_Z, heavy_atoms, max_config, target, data_path, exclude)
    full_names = list(set([mol['name'] for mol in dataset]))
    train_valid_names = random.sample(full_names, num_train_valid)
    #Should always pass because sampling occurs w/o replacement
    assert(len(set(train_valid_names)) == len(train_valid_names))
    train_valid_set = [mol for mol in dataset if mol['name'] in train_valid_names]
    test_set = [mol for mol in dataset if mol['name'] not in train_valid_names]
    assert(len(test_set) + len(train_valid_set) == len(dataset))
    random.shuffle(train_valid_set)
    num_train = int(len(train_valid_set) * 0.8)
    train_set = train_valid_set[:num_train]
    valid_set = train_valid_set[num_train:]
    assert(len(train_set) + len(valid_set) == len(train_valid_set))
    print(f"Num train: {len(train_set)}")
    print(f"Num valid: {len(valid_set)}")
    print(f"Num test: {len(test_set)}")
    return train_set, valid_set, test_set

def name_non_overlap_uniqueness_test(ref_dir: str, num_train_valid: int) -> None:
    r"""Checks that the molecules contained in Fold0, Fold1, and test are 
        unique in names and do not overlap with each other.
    
    Arguments:
        ref_dir (str): The dataset directory to check
        num_train_valid (int): The number of empirical formulas used for 
            the training and validations.
    
    Returns:
        None
    
    Raises: 
        AssertionError if any of the uniqueness/intersection checks fail
    """
    if PREFIX not in ref_dir:
        ref_dir = os.path.join(os.getcwd(), PREFIX, ref_dir)
    else:
        ref_dir = os.path.join(os.getcwd(), ref_dir)
    
    mols0_ref = pickle.load(open(os.path.join(ref_dir, "Fold0_molecs.p"), 'rb')) #Train molecules
    mols1_ref = pickle.load(open(os.path.join(ref_dir, "Fold1_molecs.p"), 'rb')) #Validation molecules
    test_ref = pickle.load(open(os.path.join(ref_dir, "test_set.p"), 'rb')) #Test molecules
    
    mols0_ref_nc = [mol['name'] for mol in mols0_ref]
    mols1_ref_nc = [mol['name'] for mol in mols1_ref]
    test_ref_nc = [mol['name'] for mol in test_ref]
    
    mols0_ref_nc_set = set(mols0_ref_nc)
    mols1_ref_nc_set = set(mols1_ref_nc)
    test_ref_nc_set = set(test_ref_nc)
    
    assert(mols0_ref_nc_set.intersection(test_ref_nc_set) == set())
    assert(mols1_ref_nc_set.intersection(test_ref_nc_set) == set())
    assert(len(mols0_ref_nc_set.intersection(mols1_ref_nc_set)) == min(len(mols0_ref_nc_set), len(mols1_ref_nc_set)))
    
    print("Uniqueness and overlap test passed")

def generate_datasets_with_ref(allowed_Z: List[int], heavy_atoms: List[int], max_config: int, 
                 target: Dict[str, str], data_path: str, exclude: List[str],
                 ref_dir: str) -> Tuple[List[Dict]]:
    r"""Generates a new dataset with the same molecules as a reference dataset.
    
    Arguments:
        allowed_Z (List[int]): The element numbers that molecules in the 
            dataset are allowed to contain.
        heavy_atoms (List[int]): The number of heavy (non-hydrogen) atoms the 
            molecules can contain.
        max_config (int): The maximum number of configurations for each 
            empirical formula allowed. 
        target (Dict): The target dictionary used for extracting information for 
            molecules from the ANI1-ccx dataset. 
        data_path (str): The path to the dataset h5 file.
        exclude (List[str]): The list of empirical formulas to exclude.
        ref_dir (str): The path to the directory containing the reference dataset.
        
    Returns:
        training_dset (List[Dict]): The training dataset
        validation_dset (List[Dict]): The validation dataset
        test_dset (List[Dict]): The test dataset
    
    Notes:
        To ensure that the proper number of configurations is included,
        the max_config parameter should be set to the same value as was used
        to generate the reference dataset. Differences in target will specify
        the new data to pull for the new set of molecules
    """
    # raise NotImplementedError("generate_datasets_with_ref not implemented!")
    
    dataset_raw = get_ani1data(allowed_Z, heavy_atoms, max_config, target, data_path, exclude)
    if PREFIX not in ref_dir:
        ref_dir = os.path.join(os.getcwd(), PREFIX, ref_dir)
    else:
        ref_dir = os.path.join(os.getcwd(), ref_dir)
    mols0_ref = pickle.load(open(os.path.join(ref_dir, "Fold0_molecs.p"), 'rb')) #Train molecules
    mols1_ref = pickle.load(open(os.path.join(ref_dir, "Fold1_molecs.p"), 'rb')) #Validation molecules
    test_ref = pickle.load(open(os.path.join(ref_dir, "test_set.p"), 'rb')) #Test molecules
    
    mols0_ref_nc = [(mol['name'], mol['iconfig']) for mol in mols0_ref]
    mols1_ref_nc = [(mol['name'], mol['iconfig']) for mol in mols1_ref]
    test_ref_nc = [(mol['name'], mol['iconfig']) for mol in test_ref]
    
    assert(len(set(mols0_ref_nc)) == len(mols0_ref_nc))
    assert(len(set(mols1_ref_nc)) == len(mols1_ref_nc))
    assert(len(set(test_ref_nc)) == len(test_ref_nc))
    
    mols0_ref_nc_set = set(mols0_ref_nc)
    mols1_ref_nc_set = set(mols1_ref_nc)
    test_ref_nc_set = set(test_ref_nc)
    
    assert(mols0_ref_nc_set.intersection(test_ref_nc_set) == set())
    assert(mols1_ref_nc_set.intersection(test_ref_nc_set) == set())
    
    train, test, valid = list(), list(), list()
    for mol in dataset_raw:
        if (mol['name'], mol['iconfig']) in mols0_ref_nc_set:
            train.append(mol)
        elif (mol['name'], mol['iconfig']) in mols1_ref_nc_set:
            valid.append(mol)
        elif (mol['name'], mol['iconfig']) in test_ref_nc_set:
            test.append(mol)
    
    return train, valid, test

def save_dataset(location: str, train_set: List[Dict], valid_set: List[Dict], 
                 test_set: List[Dict]) -> None:
    r"""Handles saving of the datasets to pickle files in the appropriate 
        locations.
    
    Arguments:
        location (str): The location (directory) to save the datasets to 
        train_set (List[Dict]): The training dataset obtained from calling
            generate_dataset().
        valid_set (List[Dict]): The validation dataset obtained from calling
            generate_dataset().
        test_set (List[Dict]): The testing dataset obtained from calling
            generate_dataset().
    
    Returns: 
        None
    
    Notes: 
        To ensure that everything is saved to the correct containing directory, 
        there will be a path check to see if PREFIX is contained within the 
        location. If not, then it will be prepended to the total path.
        
        At this stage, the datasets are saved as pickle files. The training, 
        validation, and test datasets are saved to the given location.
        
        Arbitrarily, training set molecules will be saved as Fold0_molecs.p and
        validation molecules will be saved as Fold1_molecs.p. Test set will be 
        saved to test_set.p.
    """
    train_valid_dir = None
    if PREFIX in location:
        train_valid_dir = os.path.join(os.getcwd(), location)
    else:
        train_valid_dir = os.path.join(os.getcwd(), PREFIX, location)
    if os.path.isdir(train_valid_dir):
        shutil.rmtree(train_valid_dir) #Overwrite old directories
    os.mkdir(train_valid_dir)
    train_full_path = os.path.join(train_valid_dir, "Fold0_molecs.p")
    valid_full_path = os.path.join(train_valid_dir, "Fold1_molecs.p")
    test_full_path = os.path.join(train_valid_dir, "test_set.p")
    with open(train_full_path, "wb") as handle:
        pickle.dump(train_set, handle)
    with open(valid_full_path, "wb") as handle:
        pickle.dump(valid_set, handle)
    with open(test_full_path, "wb") as handle:
        pickle.dump(test_set, handle)
    print(f"Training, validation, and test set successfully saved to {location}")

def copy_settings(settings_file_path: str, location: str) -> None:
    r"""Copies the given settings file into the location.
    
    Arguments:
        settings_file_path (str): The path to the settings file.
        location (str): The directory to copy the settings file into.
    
    Returns:
        None
    """
    dest = None
    if PREFIX in location:
        dest = os.path.join(os.getcwd(), location, "dset_settings.json")
    else:
        dest = os.path.join(os.getcwd(), PREFIX, location, "dset_settings.json")
    src = os.path.join(os.getcwd(), settings_file_path)
    shutil.copy(src, dest)
    print(f"Finished copying settings file for dataset to location {location}")
    
#Main method for creating a dataset
def create_datasets(settings_filename: str, defaults_filename: str, num_train_valid: int, mode: str,
                    ref_dir: str = None) -> None:
    r"""Creates and saves a dataset to the location specified in the settings file
    
    Arguments:
        settings_filename (str): The settings file filename
        defaults_filename (str): The default file filename
        num_train_valid (int): The number of empirical formulas used for 
            the training and validations.
        mode (str): Whether you are creating a new dataset independently (no_ref)
            or with a reference dataset (with_ref)
        ref_dir (str): The path to the reference directory if generating datasets with a 
            reference; defaults to None

    Returns:
        None
        
    Notes: 
        The important information for generating and precomputing the datasets
        will be included in the settings file, and this module will call upon
        the precompute_folds() method to link into the precompute workflow.
    """
    s_obj = parse_input_dictionaries(settings_filename, defaults_filename)
    opts = inflate_to_dict(s_obj) #opts is a dictionary for DFTBrepulsive to use only. 
    s_obj = collapse_to_master_settings(s_obj)
    
    if mode == 'no_ref':
        train, valid, test = generate_datasets(s_obj.allowed_Zs, s_obj.heavy_atoms, s_obj.max_config, s_obj.target,
                                               s_obj.data_path, s_obj.exclude, num_train_valid)
    elif mode == 'with_ref':
        assert(ref_dir is not None)
        train, valid, test = generate_datasets_with_ref(s_obj.allowed_Zs, s_obj.heavy_atoms, s_obj.max_config, s_obj.target, 
                                                        s_obj.data_path, s_obj.exclude, ref_dir)
    
    save_dataset(s_obj.top_level_fold_path, train, valid, test)
    copy_settings(settings_filename, s_obj.top_level_fold_path)
    name_non_overlap_uniqueness_test(s_obj.top_level_fold_path, num_train_valid)
    precompute_folds(s_obj, opts, s_obj.top_level_fold_path, True)
    print("Data generated and precomputed")
