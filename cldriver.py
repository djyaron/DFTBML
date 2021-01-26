# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 16:54:02 2021

@author: Frank

When running on PSC, the default will be to load data in rather than 
generating the data as usual.

TODO:
    1) Test and verify that pre-compute and loading data work
    2) Fill in code for the training loop
    3) Devise process for saving and loading data for folds
"""
from __future__ import print_function #__future__ imports must occur at the beginning of the file
import argparse
import json
from typing import Dict, List, Union
from dftb_layer_splines_4 import load_data, dataset_sorting, graph_generation, model_loss_initialization,\
    feed_generation, saving_data, total_type_conversion, model_range_correction, get_ani1data, energy_correction,\
        assemble_ops_for_charges, update_charges, Input_layer_pairwise_linear_joined, OffDiagModel2
from dftbrep_fold import get_folds_cv_limited, extract_data_for_molecs
import importlib
import os, os.path
import random
from batch import Model, RawData, DFTBList
from skfwriter import main
import pickle

#Trick for toggling print statements globally, code was found here:
# https://stackoverflow.com/questions/32487941/efficient-way-of-toggling-on-off-print-statements-in-python/32488016
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
        print(f"Getting training and validation molecules from {s.ani1_path}")
        print(f"Heavy atoms: {s.heavy_atoms}, allowed_Zs: {s.allowed_Zs}, max_config: {s.max_config}")
        print(f"The following targets are used: {s.target}")
        print(f"The following molecules are excluded: {s.exclude}")
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

def pre_compute_stage(s: Settings, par_dict: Dict, fold = None, established_models: Dict = None,
                      established_variables: Dict = None, established_range_dict = None):
    r"""Performs the precompute stage of the calculations
    
    Arguments:
        s (Settings): The settings object containing values for all the hyperparameters
        par_dict (Dict): Dictionary of skf parameters
        fold (Fold): The current fold object.
        established_models (Dict): The all_models dictionary from a previous fold
        established_variables (Dict): The model_variables dictionary from a previous fold
        established_range_dict (Dict): The model_range_dict from a previous fold
    
    Returns:
        all_models (Dict): Dictionary containing references to all the necessary models
        model_variables (Dict): Dictionary containing references to all the model_variables 
            that will be optimized during training
        training_feeds (List[Dict]): List of dictionaries containing the training feeds
        validation_feeds (List[Dict]): List of dictionaries containing the validation feeds
        training_dftblsts (List[DFTBList]): List of DFTBList objects for each of the training feeds
        validation_dftblsts (List[DFTBList]): List of DFTBList objects for each of the validation feeds
        losses (Dict): Dictionary of loss targets and their associated weights
        all_losses (Dict): Dictionary of the loss objects, mapped by their aliases (e.g. 'Etot' : TotalEnergyLoss())
        loss_tracker (Dict): A dictionary that will be populated with loss information
    
    Notes: The fold is an optional argument since it is only applicable in the case of
        computation involving CV. Similarly, the established_models, established_variables, and 
        established_range_dict arguments only apply in the case of the driver being in CV mode.
        
        All returned information here is necessary for the training loop stage of the computation
    """
    if s.driver_mode == 'non-CV':
        print("Using non-cv method for generating molecules and graphs")
        training_feeds, training_dftblsts, training_batches, validation_feeds, validation_dftblsts, validation_batches = get_graph_data_noCV(s, par_dict)
    elif s.driver_mode == 'CV':
        print("Using cv method for generating molecules and graphs")
        training_feeds, training_dftblsts, training_batches, validation_feeds, validation_dftblsts, validation_batches = get_graph_data_CV(s, par_dict)
    
    print("Creating loss dictionary")
    losses = dict()
    for loss in s.losses:
        #s.losses is a list of strings representing the different losses to factor into backpropagation
        if loss == 'Etot':
            losses[loss] = s.target_accuracy_energy
        elif loss == 'dipole':
            losses[loss] = s.target_accuracy_dipole
        elif loss == 'charges':
            losses[loss] = s.target_accuracy_charges
        elif loss == 'convex':
            losses[loss] = s.target_accuracy_convex
        elif loss == 'monotonic':
            losses[loss] = s.target_accuracy_monotonic
        else:
            raise ValueError("Unsupported loss type")
    
    print("Initializing models")
    all_models, model_variables, loss_tracker, all_losses, model_range_dict = model_loss_initialization(training_feeds, validation_feeds,
                                                                                                    s.allowed_Zs, losses, 
                                                                                                    ref_ener_start = s.reference_energy_starting_point)
    
    if (established_models is not None) and (established_variables is not None) and (established_range_dict is not None):
        print("Loading in previous models, variables, and ranges")
        all_models = established_models
        model_variables = established_variables
        model_range_dict = established_range_dict
    
    print("Performing model range correction")
    corrected_lowend_cutoff = dictionary_tuple_correction(s.low_end_correction_dict)
    model_range_dict = model_range_correction(model_range_dict, corrected_lowend_cutoff)
    
    print("Generating training feeds")
    feed_generation(training_feeds, training_batches, all_losses, all_models, model_variables, model_range_dict, par_dict, s.spline_mode, s.spline_deg, s.debug, s.loaded_data, 
                    s.num_knots, s.buffer, s.joined_cutoff, s.cutoff_dictionary, s.off_diag_opers, s.include_inflect)
    
    print("Generating validation feeds")
    feed_generation(validation_feeds, validation_batches, all_losses, all_models, model_variables, model_range_dict, par_dict, s.spline_mode, s.spline_deg, s.debug, s.loaded_data, 
                    s.num_knots, s.buffer, s.joined_cutoff, s.cutoff_dictionary, s.off_diag_opers, s.include_inflect)
    
    print("Performing type conversion to tensors")
    total_type_conversion(training_feeds, validation_feeds, ignore_keys = s.type_conversion_ignore_keys)
    
    print("Some information:")
    print(f"inflect mods: {[mod for mod in model_variables if mod != 'Eref' and mod.oper == 'S' and 'inflect' in mod.orb]}")
    print(f"s_mods: {[mod for mod in model_variables if mod != 'Eref' and mod.oper == 'S']}")
    print(f"len of s_mods: {len([mod for mod in model_variables if mod != 'Eref' and mod.oper == 'S'])}")
    print(f"len of s_mods in all_models: {len([mod for mod in all_models if mod != 'Eref' and mod.oper == 'S'])}")
    print("losses")
    print(losses)
    
    return all_models, model_variables, training_feeds, validation_feeds, training_dftblsts, validation_dftblsts, losses, all_losses, loss_tracker

def charge_update_subroutine(s: Settings, training_feeds: List[Dict], 
                             training_dftblsts: List[DFTBList],
                             validation_feeds: List[Dict],
                             validation_dftblsts: List[DFTBList], all_models: Dict,
                             epoch: int = -1) -> None:
    r"""Updates charges directly in each feed
    
    Arguments:
        s (Settings): Settings object containing all necessary hyperparameters
        training_feeds (List[Dict]): List of training feed dictionaries
        training_dftblsts (List[DFTBList]): List of training feed DFTBLists
        validation_feeds (list[Dict]): List of validation feed dictionaries
        validation_dftblsts (List[DFTBList]): List of validation feed DFTBLists
        all_models (Dict): Dictionary of all models
        epoch (int): The epoch indicating 
    
    Returns:
        None
    
    Notes: Charge updates are done for all training and validation feeds 
        and uses the H, G, and S operators (if all three are modeled; at the
        very least, the H operators is requires).
        
        Failed charge updates are reported, and the print statements there
        are set to always print.
    """
    print("Running training set charge update")
    for j in range(len(training_feeds)):
        # Charge update for training_feeds
        feed = training_feeds[j]
        dftb_list = training_dftblsts[j]
        op_dict = assemble_ops_for_charges(feed, all_models)
        try:
            update_charges(feed, op_dict, dftb_list, s.opers_to_model)
        except Exception as e:
            print(e, enable_print = 1)
            glabels = feed['glabels']
            basis_sizes = feed['basis_sizes']
            result_lst = []
            for bsize in basis_sizes:
                result_lst += list(zip(feed['names'][bsize], feed['iconfigs'][bsize]))
            print("Charge update failed for", enable_print = 1)
            print(result_lst, enable_print = 1)
    print("Training charge update done, doing validation set")
    for k in range(len(validation_feeds)):
        # Charge update for validation_feeds
        feed = validation_feeds[k]
        dftb_list = validation_dftblsts[k]
        op_dict = assemble_ops_for_charges(feed, all_models)
        try:
            update_charges(feed, op_dict, dftb_list, s.opers_to_model)
        except Exception as e:
            print(e, enable_print = 1)
            glabels = feed['glabels']
            basis_sizes = feed['basis_sizes']
            result_lst = []
            for bsize in basis_sizes:
                result_lst += list(zip(feed['names'][bsize], feed['iconfigs'][bsize]))
            print("Charge update failed for", enable_print = 1)
            print(result_lst, enable_print = 1)
    if epoch > -1:
        print(f"Charge updates done for epoch {epoch}")
    else:
        print("Charge updates done")
        
def paired_shuffle(lst_1: List, lst_2: List) -> (list, list):
    r"""Shuffles two lists while maintaining element-wise corresponding ordering
    
    Arguments:
        lst_1 (List): The first list to shuffle
        lst_2 (List): The second list to shuffle
    
    Returns:
        lst_1 (List): THe first list shuffled
        lst_2 (List): The second list shuffled
    """
    temp = list(zip(lst_1, lst_2))
    random.shuffle(temp)
    lst_1, lst_2 = zip(*temp)
    lst_1, lst_2 = list(lst_1), list(lst_2)
    return lst_1, lst_2

def training_loop(s: Settings, all_models: Dict, model_variables: Dict, 
                  training_feeds: List[Dict], validation_feeds: List[Dict], 
                  training_dftblsts: List[DFTBList], validation_dftblists: List[DFTBList],
                  losses: Dict, all_losses: Dict, loss_tracker: Dict):
    r"""Training loop portion of the calculation
    
    Arguments:
        s (Settings): Settings object containing all necessary hyperparameters
        all_models (Dict): The dictionary containing references to the
            spline models, mapped by model specs
        model_variables (Dict): Dictionary containing references to 
            all the variables that will be optimized by the model. Variables
            are stored as tensors with tracked gradients
        training_feeds (List[Dict]): List of feed dictionaries for the 
            training data
        validation_feeds (List[Dict]): List of feed dictionaries for the 
            validation data
        training_dftblsts (List[DFTBList]): List of DFTBList objects for the 
            charge updates on the training feeds
        validation_dftblists (List[DFTBList]): List of DFTBList objects for the
            charge updates on the validation feeds
        losses (Dict): Dictionary of target losses and their weights
        all_losses (Dict): Dictionary of target losses and their loss classes
        loss_tracker (Dict): Dictionary for keeping track of loss data during
            training
    
    Returns:
        ref_ener_params (List[float]): The current reference energy parameters
        loss_tracker (Dict): The final loss tracker after the training
        all_models (Dict): The dictionary of models after the training session
        model_variables (Dict): The dictionary of model variables after the
            training session
        model_range_dict (Dict): The dictionary of model ranges after the 
            training session

    Notes: The training loop consists of the main training as well as a 
        charge update subroutine.
    """
    pass


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
    
    # Testing code
    # This testing code assumes the following command line is used:
    # %python cldriver.py settings_default.json defaults.json --verbose
    # the --verbose flag enables all print statements globally
    
    print("Reading input json files...")
    
    with open(args.settings, 'r') as read_file:
        input_settings_dict = json.load(read_file)
    with open(args.defaults, 'r') as read_file:
        default_settings_dict = json.load(read_file)
        
    print("Finished reading json files.")
        
    # Check the final settings dictionary
    print("Generating final settings dictionary...")
    
    final_settings_dict = construct_final_settings_dict(input_settings_dict, default_settings_dict)
    for item in final_settings_dict:
        print(item, ":", final_settings_dict[item])
    
    print("Finished generating final settings dictionary.")
        
    # Make sure that the par_dict is loading the correct keys
    settings = Settings(final_settings_dict)
    
    print("Loading in skf parameter dictionary...")
    if settings.par_dict_name == 'auorg_1_1':
        from auorg_1_1 import ParDict
        par_dict = ParDict()
    else:
        module = importlib.import_module(settings.par_dict_name)
        par_dict = module.ParDict()

    print("SKF pardict keys:")
    print(par_dict.keys())
    print("Finished loading in skf parameter dictionary.")
    
    # Try generating some molecules from it
    # We know from previous experiments that for up to 5 heavy atoms, this
    # dataset should give 1134 molecules
    
    print("Generating feeds and batches for testing")
    training_feeds, training_dftblsts, training_batches, validation_feeds, validation_dftblsts, validation_batches = get_graph_data_noCV(settings, par_dict)
    print(f"Number of training feeds: {len(training_feeds)}")
    print(f"Number of validation feeds: {len(validation_feeds)}")
    
    # Cross-reference the molecules generated here with the ones generated from 
    # dftb_layer_driver.py:
    
    from functools import reduce # O(N^2) check in the number of molecules
    print("Flattening to get all molecules back")
    flattened_train_molecs = list(reduce(lambda x, y : x + y, training_batches))
    flattened_validation_molecs = list(reduce(lambda x, y : x + y, validation_batches))
    total_molecs = flattened_train_molecs + flattened_validation_molecs
    
    print("Checking total number of molecules")
    assert(len(total_molecs) == 1134) #We know this from previous experiments
    with open("molecule_test.p","rb") as handle:
        reference_molecs = pickle.load(handle)
    print("loading reference molecules and doing a direct comparison")
    assert(len(reference_molecs) == len(total_molecs))
    
    test_name_config = set([(x['name'], x['iconfig']) for x in total_molecs])
    ref_name_config = set([(y['name'], y['iconfig']) for y in reference_molecs])
    assert(test_name_config == ref_name_config)
    assert(len(test_name_config) == len(ref_name_config) == len(total_molecs) == 1134)
    print("Molecules are the same")
    
    print("Testing precompute stage, no CV")
    all_models, model_variables, training_feeds, validation_feeds, training_dftblsts, validation_dftblsts, losses, all_losses, loss_tracker = pre_compute_stage(settings, par_dict)
    
    print("Checking correct cutoffs, should be 3.0")
    for model in all_models:
        if hasattr(all_models[model], 'cutoff'):
            assert(all_models[model].cutoff == 3.0)
            assert(model.oper in ["S", "H", "R"])
    
    print("Checking length of B-spline coefficients, should be 49 (num_knots - 1)")
    for model in all_models:
        print(model)
        if isinstance(all_models[model], Input_layer_pairwise_linear_joined):
            assert(model.oper in ["S", "H", "R"])
            print(all_models[model].get_total())
            assert(len(all_models[model].get_total()) == 49)
        elif isinstance(all_models[model], OffDiagModel2):
            assert(model.oper == "G")
        else:
            print(all_models[model].get_variables()) #Eref and value case
            if model == 'Eref':
                assert(len(all_models[model].get_variables()) == 5)
            else:
                assert(len(all_models[model].get_variables()) == 1)
    
    print("Asserting that no inflection models made it into all_models")
    for model in all_models:
        if hasattr(model, 'orb'):
            assert('inflect' not in model.orb)
    
    print("Quick check on model_variables")
    
    for model in model_variables:
        if model == 'Eref':
            assert(len(model_variables[model]) == 5)
        elif model.oper == 'G':
            assert(len(model_variables[model]) == 1)
            assert(len(model.Zs) == 1)
        elif 'inflect' in model.orb:
            assert(len(model_variables[model]) == 1)
            assert(model.oper == 'S')
        else:
            print(len(model_variables[model]))
            
    
    
    
    
    
    
    
    
    
    
    
        
        
    

