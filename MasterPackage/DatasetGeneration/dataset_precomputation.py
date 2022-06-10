# -*- coding: utf-8 -*-
"""
Created on Mon May 30 17:01:07 2022

@author: fhu14
"""

"""
Module that handles the required precomputations for the different 
datasets that have been generated
"""

#%% Imports, definitions
from InputParser import parse_input_dictionaries, collapse_to_master_settings, inflate_to_dict
import os, json
from precompute_check import precompute_settings_check
from precompute_driver import precompute_folds

DESTINATION = "ALL_EXPERIMENT_DSETS" #Where all the datasets are located
#%% Code behind

def process_settings_files(settings_filename: str, defaults_filename: str):
    r"""Takes in a settings file and defaults file and processes them 
        to produce the settings object and opts dictionary necessary
    
    Arguments: 
        settings_filename (str): The name of the settings file to use for 
            the precomputation
        defaults_filename (str): The defaults filename to use for the
            precomputation
    
    Returns:
        s_obj (Settings): Settings object containing the hyperparameter settings
        opts (Dict): A dictionary containing the parameters relevant for the 
            DFTBrepulsiev code
    """
    s_obj = parse_input_dictionaries(settings_filename, defaults_filename)
    opts = inflate_to_dict(s_obj)
    s_obj = collapse_to_master_settings(s_obj)
    return s_obj, opts

def populate_settings_files() -> None:
    r"""Copies a computation settings file into each directory for the 
        precomputation to happen
    
    Arguments:
        None
    
    Returns:
        None
    
    Notes: This function assumes that there exists a ssettings and defaults
        json file in the DESTINATION directory. The settings file is copied
        and modified into each directory. At this point, there are no 
        differences in parameters between different experiments besides the top_level_fold_path.
        Also performs a precompute settings check for each file before
        saving
    """
    src = os.path.join(os.getcwd(), DESTINATION, "settings_refactor_tst.json")
    with open(src, 'r') as handle:
        src_jdict = json.load(handle)
    all_files = os.listdir(DESTINATION)
    all_directories = list(filter(lambda x : os.path.isdir(os.path.join(DESTINATION, x)), all_files))
    for dset_directory in all_directories:
        new_path = f"{DESTINATION}/{dset_directory}"
        src_jdict['loaded_data_fields']['top_level_fold_path'] = new_path
        src_jdict['run_id'] = 'precompute_ignore'
        full_settings_path = os.path.join(os.getcwd(), DESTINATION, dset_directory, "dset_settings.json")
        with open(full_settings_path, 'w+') as handle:
            json.dump(src_jdict, handle, indent = 4)
    print("Finished populating the settings files for precomputation")

def perform_precompute_settings_check() -> None:
    r"""Goes through and performs a check on each of the settings files before
        precomputing
    
    Arguments:
        None
    
    Returns:
        None
    
    Raises: AssertionError if the precompute check fails to pass any 
        of the required conditions
    
    Notes: The documentation for the precompute check is laid out in
        the precompute_check.py module
    """
    all_files = os.listdir(DESTINATION)
    all_directories = list(filter(lambda x : os.path.isdir(os.path.join(DESTINATION, x)), all_files))
    for dset_directory in all_directories:
        settings_path = os.path.join(os.getcwd(), DESTINATION, dset_directory, 'dset_settings.json')
        print(f"Checking {settings_path}")
        precompute_settings_check(settings_path)
        print("Check passed")
    print("Finished checking all dset_settings files")

def check_dataset_paths() -> None:
    r"""Makes sure that all the settings files are populated in the correct
        locations
    
    Arguments:
        None
    
    Returns:
        None
    
    Raises: AssertionError if the settings file is not correctly labeled
    """
    all_files = os.listdir(DESTINATION)
    all_directories = list(filter(lambda x : os.path.isdir(os.path.join(DESTINATION, x)), all_files))
    for dset_directory in all_directories:
        settings_path = os.path.join(os.getcwd(), DESTINATION, dset_directory, 'dset_settings.json')
        print(f"Checking {settings_path}")
        with open(settings_path, 'r') as handle:
            jdict = json.load(handle)
            print(dset_directory)
            print(jdict['loaded_data_fields']['top_level_fold_path'])
            assert(
                os.path.split(jdict['loaded_data_fields']['top_level_fold_path'])[-1] == dset_directory
                )
    print("Finished checking correct placement of dset_settings files")

def precompute_datasets() -> None:
    r"""Performs the precomputation for different datasets
        contained in the DESTINATION directory
    
    Arguments:
        None
    
    Returns:
        None
    
    Notes: Iterates through each dataset and performs the precomputation 
        necessary for running the data through the network
    """
    all_files = os.listdir(DESTINATION)
    all_directories = list(filter(lambda x : os.path.isdir(os.path.join(DESTINATION, x)), all_files))
    defaults_file = os.path.join(os.getcwd(), DESTINATION, "refactor_default_tst.json")
    for dset_directory in all_directories:
        settings_file = os.path.join(os.getcwd(), DESTINATION, dset_directory, 'dset_settings.json')
        print(f"Beginning precomputation using {settings_file}")
        s_obj, opts = process_settings_files(settings_file, defaults_file)
        #Always copy the molecules
        precompute_folds(s_obj, opts, s_obj.top_level_fold_path, True)
        print(f"Completed precomputation for {settings_file}")
    print("Finished performing the precomputations for all included datasets")