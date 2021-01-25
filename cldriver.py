# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 16:54:02 2021

@author: Frank

When running on PSC, the default will be to load data in rather than 
generating the data as usual. For this reason, 
"""

import argparse
import json
from typing import Dict, List, Union
from dftb_layer_splines_4 import load_data, dataset_sorting, graph_generation, model_loss_initialization,\
    feed_generation, saving_data, total_type_conversion, model_range_correction, get_ani1data, energy_correction
from dftbrep_fold import get_folds_cv_limited, extract_data_for_molecs
import importlib
import os, os.path
import random
from __future__ import print_function

def print(*args, **kwargs):
    if enable_print:
        return __builtins__.print(*args, **kwargs)

# Construct the parser
parser = argparse.ArgumentParser()
parser.add_argument("settings", help = "Name of the settings file for the current hyperparameter settings")
parser.add_argument("defaults", help = "Name of the default settings file for the hyperparameters")
parser.add_argument("--verbose", help = "increase output verbosity", action = "store_true")

class Settings:
    def __init__(self, settings_dict: Dict) -> None:
        r"""Generates a Settings object from the given dictionary
        
        Arguments:
            settings_dict (Dict): Dictionary containing key value pairs for the
                current hyperparmeter settings
        
        Returns:
            None
        
        Notes: Using an object rather than a dictionary is easier since you can
            just do settings.ZZZ rather than doing the bracket notation and the quotes.
        """
        for key in settings_dict:
            setattr(self, key, settings_dict[key])

def construct_final_settings_dict(settings_dict: Dict, default_dict: Dict) -> Dict:
    r"""Generates the final settings dictionary based on the input settings file and the
        defaults file
    
    Arguments:
        settings_dict (Dict): Dictionary of the user-defined hyperparameter settings. 
            Read from the settings json file
        default_dict (Dict): Dictionary of default hyperparameter settings. 
            Read from the defaults json file
    
    Returns:
        final_settings_dict (Dict): The final dictionary containing the settings
            to be used for the given run over all hyperparameters
    
    Notes: If something is not specified in the settings file, then the
        default value is pulled from the default_dict to fill in. For this reason,
        the default_dict contains the important keys and the settings file
        can contain a subset of these important keys. The settings file will include
        some information that is not found in the default dictionary, such as the 
        name given to the current run and the directory for saving the skf files at the end
        
        The settings_dict must contain the run_name key
    """
    final_dict = dict()
    for key in default_dict:
        if key not in settings_dict:
            final_dict[key] = default_dict[key]
        else:
            final_dict[key] = settings_dict[key]
    try:
        final_dict['run_id'] = settings_dict['run_id']
    except:
        raise KeyError("Settings file must include the 'run_id' field!")
    
    return final_dict

def dictionary_tuple_correction(input_dict: Dict) -> None:
    r"""Performs a correction on the input_dict to convert from string to tuple
    
    Arguments:
        input_dict (Dict): The dictionary that needs correction
    
    Returns:
        corrected_dict (Dict): Dictionary with the necessary corrections
            applied
    
    Notes: The two dictionaries that need correction are the cutoff dictionary and the
        range correction dictionary for model_range_dict. For the dictionary used to 
        correct model ranges, the keys are of the form "elem1,elem2" where elem1 and elem2 
        are the atomic numbers of the first and second element, respectively. These
        need to be converted to a tuple of the form (elem1, elem2).
        
        For the dictionary used to specify cutoffs (if one is provided), the format 
        of the keys is "oper,elem1,elem2" where oper is the operator of interest and
        elem1 and elem2 are again the atomic numbers of the elements of interest. This
        will be converted to a tuple of the form (oper, (elem1, elem2)). A check is 
        performed between these cases depending on the number of commas.
        
        The reason for this workaround is because JSON does not support tuples. 
        An alternative would have been to use a string representation of the tuple
        with the eval() method. 
    """
    num_commas = list(input_dict.keys())[0].count(",")
    new_dict = dict()
    #Assert key consistency in the dictionary
    for key in input_dict:
        assert(key.count(",") == num_commas)
        key_splt = key.split(",")
        if len(key_splt) == 2:
            elem1, elem2 = int(key_splt[0]), int(key_splt[1])
            new_dict[(elem1, elem2)] = input_dict[key]
        elif len(key_splt) == 3:
            oper, elem1, elem2 = key_splt[0], int(key_splt[1]), int(key_splt[2])
            new_dict[(oper(elem1, elem2))] = input_dict[key]
        else:
            raise ValueError("Given dictionary does not need tuple correction!")
    return new_dict


def get_graph_data_noCV(s: Settings, par_dict: Dict):
    r"""Handles the molecule grabbing and graph generating stages of pre-compute.
    
    Arguments:
        s (Settings): The settings object representing the current set of 
            hyperparameters
        par_dict (Dict): The relevant parameter dictionary (e.g. auorg-1-1)
        
    Returns:
        training_feeds (List[Dict]): List of training feed dictionaries
        validation_feeds (List[Dict]): List of validation feed dictionaries
        training_dftblsts (List[DFTBList]): List of DFTBList objects corresponding
            with training_feeds
        validation_dftblsts (List[DFTBList]): List of DFTBList objects corresponding
            with validation_feeds
        training_batches (List[List[Dict]]): List of lists of original molecule dictionaries 
            corresponding to the generated training_feeds
        validation_batches (List[List[Dict]]): List of listsof original molecule dictionaries
            corresponding to the generate validation_feeds
    
    Notes: If loading data, the validation and training batches are returned as empty lists
    """
    if not s.loaded_data:
        print("Getting training and validation molecules")
        dataset = get_ani1data(s.allowed_Zs, s.heavy_atoms, s.max_config, s.target, ani1_path = s.ani1_path, exclude = s.exclude)
        training_molecs, validation_molecs = dataset_sorting(dataset, s.prop_train, s.transfer_training, s.transfer_train_params, s.train_ener_per_heavy)
        print(f"number of training molecules: {len(training_molecs)}")
        print(f"number of validation molecules: {len(validation_molecs)}")
        
        print("Getting training graphs")
        config = {'opers_to_model' : s.opers_to_model}
        training_feeds, training_dftblsts, training_batches = graph_generation(training_molecs, config, s.allowed_Zs, par_dict, s.num_per_batch)
        print("Getting validation graphs")
        validation_feeds, validation_dftblsts, validation_batches = graph_generation(validation_molecs, config, s.allowed_Zs, par_dict, s.num_per_batch)
    else:
        print("Loading data")
        training_feeds, validation_feeds, training_dftblsts, validation_dftblsts = load_data(s.molec_file_names[0], s.batch_file_names[0],
                                                                                         s.molec_file_names[1], s.batch_file_names[1],
                                                                                         s.reference_data_names[0], s.reference_data_names[1],
                                                                                         s.dftblst_names[0], s.dftblst_names[1],
                                                                                         s.ragged_dipole, s.run_check)
        training_batches, validation_batches = [], []
    return training_feeds, training_dftblsts, training_batches, validation_feeds, validation_dftblsts, validation_batches

def get_graph_data_CV(s: Settings, par_dict: Dict, fold: tuple):
    r"""Handles the molecule grabbing and graph generating stages of pre-compute,
        but for running CV experiments
    
    Arguments:
        s (Settings): The settings object representing the current set of 
            hyperparameters
        par_dict (Dict): The relevant parameter dictionary (e.g. auorg-1-1)
        fold (tuple): The tuple representing the fold, (train, test)
    
    Returns:
        training_feeds (List[Dict]): List of training feed dictionaries
        validation_feeds (List[Dict]): List of validation feed dictionaries
        training_dftblsts (List[DFTBList]): List of DFTBList objects corresponding
            with training_feeds
        validation_dftblsts (List[DFTBList]): List of DFTBList objects corresponding
            with validation_feeds
        training_batches (List[List[Dict]]): List of lists of original molecule dictionaries 
            corresponding to the generated training_feeds
        validation_batches (List[List[Dict]]): List of listsof original molecule dictionaries
            corresponding to the generate validation_feeds
    
    TODO: Figure out the framework for loading data for the different folds rather than
        computing it directly!
    """
    print(f"Current parameter dictionary keys:")
    print(par_dict.keys())
    print("Getting validation, training molecules")
    #Getting the dataset is for debugging purposes only, remove later
    dataset = get_ani1data(s.allowed_Zs, s.heavy_atoms, s.max_config, s.target, ani1_path = s.ani1_path, exclude = s.exclude)
    training_molecs, validation_molecs = extract_data_for_molecs(fold, s.target, s.ani1_path)
    assert(len(training_molecs) + len(validation_molecs) == len(dataset))
    print(f"number of training molecules: {len(training_molecs)}")
    print(f"number of validation molecules: {len(validation_molecs)}")
    if s.train_ener_per_heavy:
        for molec in training_molecs:
            energy_correction(molec)
        for molec in validation_molecs:
            energy_correction(molec)
    random.shuffle(training_molecs)
    random.shuffle(validation_molecs)
    
    config = {'opers_to_model' : s.opers_to_model}
    print("Getting training graphs")
    training_feeds, training_dftblsts, training_batches = graph_generation(training_molecs, config, s.allowed_Zs, par_dict, s.num_per_batch)
    print("Getting validation graphs")
    validation_feeds, validation_dftblsts, validation_batches = graph_generation(validation_molecs, config, s.allowed_Zs, par_dict, s.num_per_batch)
    
    return training_feeds, training_dftblsts, training_batches, validation_feeds, validation_dftblsts, validation_batches



def run_method(settings_filename: str, defaults_filename: str) -> None:
    r"""The main method for running the cldriver
    
    Arguments:
        settings_filename (str): The filename for the settings json file
        defaults_filename (str): The filename for the default settings json file
    
    Returns:
        None
    
    Notes: The stages of the computation are as follows:
        1) Top level variable declaration
        2) Precompute
        3) Training loop
        4) Logging/results
    
        Each of these parts will be its own separate method, and the precompute stage
        part will vary slightly based on whether we are using folds or not
    """
    with open(settings_filename, "r") as read_file:
        input_settings_dict = json.load(read_file)
    with open(defaults_filename, "r") as read_file:
        default_settings_dict = json.load(read_file)
    final_settings = construct_final_settings_dict(input_settings_dict, default_settings_dict)
    #Convert settings to an object for easier handling
    settings = Settings(final_settings)
    if settings.par_dict_name == 'auorg_1_1':
        from auorg_1_1 import ParDict
        par_dict = ParDict()
    else:
        module = importlib.import_module(settings.par_dict_name)
        par_dict = module.ParDict()
        
    

if __name__ == "__main__":
    args = parser.parse_args()
    enable_print = 1 if args.verbose else 0
    
        
        
    

