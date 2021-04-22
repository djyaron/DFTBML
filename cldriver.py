# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 16:54:02 2021

@author: Frank

When running on PSC, the default will be to load data in rather than 
generating the data as usual.

TODO:
    1) Safety method for checking that the settings that are read in is valid.
"""
from __future__ import print_function #__future__ imports must occur at the beginning of the file
import argparse
import json
import torch
import time
import torch.optim as optim
from typing import Dict, List
from dftb_layer_splines_4 import load_data, dataset_sorting, graph_generation, model_loss_initialization,\
    feed_generation, saving_data, total_type_conversion, model_range_correction, get_ani1data, energy_correction,\
        assemble_ops_for_charges, update_charges, Input_layer_pairwise_linear_joined, OffDiagModel2, DFTB_Layer,\
                driver, repulsive_energy_2 #Need to change name of new repulsive energy model
from dftbrep_fold import get_folds_cv_limited, extract_data_for_molecs
import importlib
import os, os.path
import random
from batch import DFTBList
from skfwriter import main, atom_nums, atom_masses
from fold_generator import loading_fold, load_combined_fold
import pickle

#Trick for toggling print statements globally, code was found here:
# https://stackoverflow.com/questions/32487941/efficient-way-of-toggling-on-off-print-statements-in-python/32488016
# Apparently need to comment out this print when debugging in console??
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

def dictionary_tuple_correction(input_dict: Dict) -> Dict:
    r"""Performs a correction on the input_dict to convert from string to tuple
    
    Arguments:
        input_dict (Dict): The dictionary that needs correction
    
    Returns:
        new_dict (Dict): Dictionary with the necessary corrections
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
    if num_commas == 0:
        print("Dictionary does not need correction")
        print(input_dict)
        return input_dict #No correction needed
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
            new_dict[(oper, (elem1, elem2))] = input_dict[key]
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
        print(f"Getting training and validation molecules from {s.data_path}")
        print(f"Heavy atoms: {s.heavy_atoms}, allowed_Zs: {s.allowed_Zs}, max_config: {s.max_config}")
        print(f"The following targets are used: {s.target}")
        print(f"The following molecules are excluded: {s.exclude}")
        dataset = get_ani1data(s.allowed_Zs, s.heavy_atoms, s.max_config, s.target, ani1_path = s.data_path, exclude = s.exclude)
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
        print(f"Number of training feeds: {len(training_feeds)}")
        print(f"Number of validation feeds: {len(validation_feeds)}")
    return training_feeds, training_dftblsts, training_batches, validation_feeds, validation_dftblsts, validation_batches

def get_graph_data_CV(s: Settings, par_dict: Dict, fold: tuple, fold_num: int = -1, fold_mapping_dict: Dict = None):
    r"""Handles the molecule grabbing and graph generating stages of pre-compute,
        but for running CV experiments
    
    Arguments:
        s (Settings): The settings object representing the current set of 
            hyperparameters
        par_dict (Dict): The relevant parameter dictionary (e.g. auorg-1-1)
        fold (tuple): The tuple representing the fold, (train, test)
        fold_num (int): The number indicating the current fold of interest if we are 
            loading the pre-generated data for a fold. Defaults to -1
        fold_mapping_dict (Dict): The dictionary indicating how to combine individual folds for training and
            validation. The first element of the entry is the training fold numbers and the
            second element of the entry is the validation fold numbers. Defaults to None
    
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
    print("Current parameter dictionary keys:")
    print(par_dict.keys())
    print("Getting validation, training molecules")
    if not s.loaded_data:
        #Getting the dataset is for debugging purposes in ensuring that the correct number of molecules is extracted.
        print("Generating data rather than loading")
        dataset = get_ani1data(s.allowed_Zs, s.heavy_atoms, s.max_config, s.target, ani1_path = s.data_path, exclude = s.exclude)
        training_molecs, validation_molecs = extract_data_for_molecs(fold, s.target, s.data_path)
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
    elif s.loaded_data and fold_num > -1:
        print(f"loading data for fold {fold_num} rather than generating")
        top_fold_path = s.top_level_fold_path
        if s.fold_load_form == "combine_individual_folds":
            training_feeds, validation_feeds, training_dftblsts, validation_dftblsts = load_combined_fold(s, top_fold_path, fold_num, fold_mapping_dict)
        elif s.fold_load_form == "train_valid_per_fold":
            training_feeds, validation_feeds, training_dftblsts, validation_dftblsts = loading_fold(s, top_fold_path, fold_num)
        training_batches, validation_batches = list(), list()
        print(f"Number of training feeds: {len(training_feeds)}")
        print(f"Number of validation feeds: {len(validation_feeds)}")
    
    # import pdb; pdb.set_trace()
    
    return training_feeds, training_dftblsts, training_batches, validation_feeds, validation_dftblsts, validation_batches

def pre_compute_stage(s: Settings, par_dict: Dict, fold = None, fold_num: int = -1, fold_mapping_dict: Dict = None, established_models: Dict = None,
                      established_variables: Dict = None):
    r"""Performs the precompute stage of the calculations
    
    Arguments:
        s (Settings): The settings object containing values for all the hyperparameters
        par_dict (Dict): Dictionary of skf parameters
        fold (tuple): The current fold object.
        fold_num (int): The number of the current fold being used
        fold_mapping_dict (Dict): The dictionary indicating how to combine individual folds for training and
            validation. The first element of the entry is the training fold numbers and the
            second element of the entry is the validation fold numbers. Defaults to None
        established_models (Dict): The all_models dictionary from a previous fold
        established_variables (Dict): The model_variables dictionary from a previous fold
    
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
        training_feeds, training_dftblsts, training_batches, validation_feeds, validation_dftblsts, validation_batches = get_graph_data_CV(s, par_dict, fold, fold_num, fold_mapping_dict)
    
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
    
    if (established_models is not None) and (established_variables is not None):
        print("Loading in previous models, variables, and ranges")
        all_models = established_models
        model_variables = established_variables
    
    print("Performing model range correction")
    s.low_end_correction_dict = dictionary_tuple_correction(s.low_end_correction_dict)
    model_range_dict = model_range_correction(model_range_dict, s.low_end_correction_dict, universal_high = s.universal_high)
    
    #Change the tuples over if a cutoff dictionary is given
    if s.cutoff_dictionary is not None:
        s.cutoff_dictionary = dictionary_tuple_correction(s.cutoff_dictionary)
    
    print("Generating training feeds")
    feed_generation(training_feeds, training_batches, all_losses, all_models, model_variables, model_range_dict, par_dict, s.spline_mode, s.spline_deg, s.debug, s.loaded_data, 
                    s.num_knots, s.buffer, s.joined_cutoff, s.cutoff_dictionary, s.off_diag_opers, s.include_inflect)
    
    print("Generating validation feeds")
    feed_generation(validation_feeds, validation_batches, all_losses, all_models, model_variables, model_range_dict, par_dict, s.spline_mode, s.spline_deg, s.debug, s.loaded_data, 
                    s.num_knots, s.buffer, s.joined_cutoff, s.cutoff_dictionary, s.off_diag_opers, s.include_inflect)
    
    print("Performing type conversion to tensors")
    total_type_conversion(training_feeds, validation_feeds, ignore_keys = s.type_conversion_ignore_keys)
    
    
    print("Some information:")
    print(f"inflect mods: {[mod for mod in model_variables if (not isinstance(mod, str)) and mod.oper == 'S' and 'inflect' in mod.orb]}")
    print(f"s_mods: {[mod for mod in model_variables if (not isinstance(mod, str)) and mod.oper == 'S']}")
    print(f"len of s_mods: {len([mod for mod in model_variables if (not isinstance(mod, str)) and mod.oper == 'S'])}")
    print(f"len of s_mods in all_models: {len([mod for mod in all_models if (not isinstance(mod, str)) and mod.oper == 'S'])}")
    print("losses")
    print(losses)
    print(all_losses)
    
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
            print(e)
            glabels = feed['glabels']
            basis_sizes = feed['basis_sizes']
            result_lst = []
            for bsize in basis_sizes:
                result_lst += list(zip(feed['names'][bsize], feed['iconfigs'][bsize]))
            print("Charge update failed for")
            print(result_lst)
    print("Training charge update done, doing validation set")
    for k in range(len(validation_feeds)):
        # Charge update for validation_feeds
        feed = validation_feeds[k]
        dftb_list = validation_dftblsts[k]
        op_dict = assemble_ops_for_charges(feed, all_models)
        try:
            update_charges(feed, op_dict, dftb_list, s.opers_to_model)
        except Exception as e:
            print(e)
            glabels = feed['glabels']
            basis_sizes = feed['basis_sizes']
            result_lst = []
            for bsize in basis_sizes:
                result_lst += list(zip(feed['names'][bsize], feed['iconfigs'][bsize]))
            print("Charge update failed for")
            print(result_lst)
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

def exclude_R_backprop(model_variables: Dict) -> None:
    r"""Removes the R-spline mods from the backpropagation of the network
    
    Arguments:
        model_variables (Dict): Dictionary containing the model variables 
    
    Returns:
        None
    
    Notes: Removes the R models from the model_variables dictionary so they 
        are not optimized. This is only a necessity when using the new repulsive
        model.
    """
    bad_mods = [mod for mod in model_variables if (not isinstance(mod, str)) and (mod.oper == 'R')]
    for mod in bad_mods:
        del model_variables[mod]

def training_loop(s: Settings, all_models: Dict, model_variables: Dict, 
                  training_feeds: List[Dict], validation_feeds: List[Dict], 
                  training_dftblsts: List[DFTBList], validation_dftblsts: List[DFTBList],
                  losses: Dict, all_losses: Dict, loss_tracker: Dict,
                  init_repulsive: bool = False):
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
        validation_dftblsts (List[DFTBList]): List of DFTBList objects for the
            charge updates on the validation feeds
        losses (Dict): Dictionary of target losses and their weights
        all_losses (Dict): Dictionary of target losses and their loss classes
        loss_tracker (Dict): Dictionary for keeping track of loss data during
            training. The first list is validation, the second list is training.
        init_repulsive (bool): Whether or not to initialize the repulsive model.
            Defaults to False. Note that this parameter only has meaning if
            s.rep_setting == 'new' (this only works for new repulsive model)
    
    Returns:
        ref_ener_params (List[float]): The current reference energy parameters
        loss_tracker (Dict): The final loss tracker after the training
        all_models (Dict): The dictionary of models after the training session
        model_variables (Dict): The dictionary of model variables after the
            training session
        times_per_epoch (List): A list of the the amount of time taken by each epoch,
            reported in seconds

    Notes: The training loop consists of the main training as well as a 
        charge update subroutine.
    """
    #Instantiate the dftblayer, optimizer, and scheduler
    dftblayer = DFTB_Layer(device = None, dtype = torch.double, eig_method = s.eig_method, repulsive_method = s.rep_setting)
    learning_rate = s.learning_rate
    optimizer = optim.Adam(list(model_variables.values()), lr = learning_rate, amsgrad = s.ams_grad_enabled)
    #TODO: Experiment with alternative learning rate schedulers
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = s.scheduler_factor, 
                                                     patience = s.scheduler_patience, threshold = s.scheduler_threshold)
    
    validation_losses, training_losses = list(), list()
    
    times_per_epoch = list()
    print("Running initial charge update")
    charge_update_subroutine(s, training_feeds, training_dftblsts, validation_feeds, validation_dftblsts, all_models)
    if s.rep_setting == 'new':
        if init_repulsive:
            print("Initializing repulsive model")            
            all_models['rep'] = repulsive_energy_2(s, training_feeds, validation_feeds, all_models, dftblayer, torch.double)
        else:
            print("Updating existing repulsive model")
            all_models['rep'].update_model_crossover(s, training_feeds, validation_feeds, all_models, dftblayer, torch.double)
    
    nepochs = s.nepochs
    for i in range(nepochs):
        start = time.time()
        
        #Validation routine
        validation_loss = 0
        for feed in validation_feeds:
            with torch.no_grad():
                output = dftblayer(feed, all_models)
                #Add in the repulsive energies if using new repulsive model
                if s.rep_setting == 'new':
                    output['Erep'] = all_models['rep'].generate_repulsive_energies(feed, 'valid')
                tot_loss = 0
                for loss in all_losses:
                    if loss == 'Etot':
                        if s.train_ener_per_heavy:
                            val = losses[loss] * all_losses[loss].get_value(output, feed, True, s.rep_setting)
                        else:
                            val = losses[loss] * all_losses[loss].get_value(output, feed, False, s.rep_setting)
                        tot_loss += val
                        loss_tracker[loss][2] += val.item()
                    elif loss == 'dipole':
                        val = losses[loss] * all_losses[loss].get_value(output, feed, s.rep_setting)
                        loss_tracker[loss][2] += val.item()
                        if s.include_dipole_backprop:
                            tot_loss += val
                    else:
                        val = losses[loss] * all_losses[loss].get_value(output, feed, s.rep_setting)
                        tot_loss += val 
                        loss_tracker[loss][2] += val.item()
                validation_loss += tot_loss.item()
        
        
        if len(validation_feeds) > 0:
            #Print some information
            print("Validation loss:",i, (validation_loss/len(validation_feeds)))
            validation_losses.append((validation_loss/len(validation_feeds)))
            
            #Update loss_tracker 
            for loss in all_losses:
                loss_tracker[loss][0].append(loss_tracker[loss][2] / len(validation_feeds))
                #Reset the loss tracker after being done with all feeds
                loss_tracker[loss][2] = 0
        
            #Shuffle the validation data
            validation_feeds, validation_dftblsts = paired_shuffle(validation_feeds, validation_dftblsts)
        
        #Training routine
        epoch_loss = 0.0
        
        # import pdb; pdb.set_trace()
        
        for feed in training_feeds:
            optimizer.zero_grad()
            output = dftblayer(feed, all_models)
            if s.rep_setting == 'new':
                output['Erep'] = all_models['rep'].generate_repulsive_energies(feed, 'train')
            tot_loss = 0
            for loss in all_losses:
                if loss == 'Etot':
                    if s.train_ener_per_heavy:
                        val = losses[loss] * all_losses[loss].get_value(output, feed, True, s.rep_setting)
                    else:
                        val = losses[loss] * all_losses[loss].get_value(output, feed, False, s.rep_setting)
                    tot_loss += val
                    loss_tracker[loss][2] += val.item()
                elif loss == 'dipole':
                    val = losses[loss] * all_losses[loss].get_value(output, feed, s.rep_setting)
                    loss_tracker[loss][2] += val.item()
                    if s.include_dipole_backprop:
                        tot_loss += val
                else:
                    val = losses[loss] * all_losses[loss].get_value(output, feed, s.rep_setting)
                    tot_loss += val
                    loss_tracker[loss][2] += val.item()
    
            epoch_loss += tot_loss.item()
            tot_loss.backward()
            optimizer.step()
        #Train the repulsive model once per epoch
        #Training the repulsive model once per epoch does not give better results
        # all_models['rep'].update_model_training(s, training_feeds, all_models, dftblayer)
        scheduler.step(epoch_loss) #Step on the epoch loss
        
        #Print some information
        print("Training loss:", i, (epoch_loss/len(training_feeds)))
        training_losses.append((epoch_loss/len(training_feeds)))
        
        #Update the loss tracker
        for loss in all_losses:
            loss_tracker[loss][1].append(loss_tracker[loss][2] / len(training_feeds))
            loss_tracker[loss][2] = 0
        
        #Shuffle training data
        training_feeds, training_dftblsts = paired_shuffle(training_feeds, training_dftblsts)
            
        #Update charges every charge_update_epochs:
        if (i % s.charge_update_epochs == 0):
            charge_update_subroutine(s, training_feeds, training_dftblsts, validation_feeds, validation_dftblsts, all_models, epoch = i)
            #Move the repulsive training routine outside so it updates every epoch
            if s.rep_setting == 'new':
                print("Updating repulsive model")
                all_models['rep'].update_model_training(s, training_feeds, all_models, dftblayer)
    
        times_per_epoch.append(time.time() - start)
    
    print(f"Finished with {s.nepochs} epochs")
    
    print("Reference energy parameters:")
    reference_energy_params = list(model_variables['Eref'].detach().numpy())
    print(reference_energy_params)
    
    return reference_energy_params, loss_tracker, all_models, model_variables, times_per_epoch

def write_output_skf(s: Settings, all_models: Dict) -> None:
    r"""Writes the skf output files after done with training
    
    Arguments:
        s (Settings): The Settings object containing all the hyperparameters 
        all_models (Dict): The dictionary of trained models
    
    Returns:
        None
    """
    train_s_block = True if "S" in s.opers_to_model else False
    if train_s_block:
        print("Writing skf files with computed S")
    else:
        print("Writing skf files with copied S")
    if s.rep_setting == 'new':
        print("Writing skf files with new repulsive model")
    elif s.rep_setting == 'old':
        print("Writing skf files with old repulsive model")
    else:
        raise ValueError("Unrecognized repulsive setting")
    target_folder = os.path.join(s.skf_extension, s.run_id)
    if not os.path.isdir(target_folder):
        os.mkdir(target_folder)
    main(all_models, atom_nums, atom_masses, train_s_block, s.ref_direct, s.rep_setting, s.skf_strsep, 
         s.skf_ngrid, target_folder)

def write_output_lossinfo(s: Settings, loss_tracker: Dict, times_per_epoch: List[float], split_num: int,
                          split_mapping: Dict) -> None:
    r"""Function for outputting any loss information
    
    Arguments:
        s (Settings): Settings object containing hyperparameter settings
        loss_tracker (Dict): Dictionary for keeping track of loss data during
            training. The first list is validation, the second list is training
        times_per_epoch (List[float]): The amount of time taken per epoch, in
            seconds
        split_num (int): The number of the current split
        split_mapping (Dict): The dictionary indicating how to combine individual folds for training and
            validation. The first element of the entry is the training fold numbers and the
            second element of the entry is the validation fold numbers. Defaults to None
            
    Returns:
        None
    
    Notes: The loss tracker object is indexed by the type of loss. For example, for
        the 'Etot' loss, there are two list objects with the first being the validation
        losses and the second being the training losses:
            {'Etot' : [[valid_values], [train_values], value_holder]
             ...
                }
        The loss tracker is saved for each split, and it is saved in the same directory as the
        top_level_fold_path variable in the settings file.
    """
    target_dir = os.path.join(s.top_level_fold_path, f"Split{split_num}")
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)
    with open(os.path.join(target_dir, "loss_tracker.p"), "wb") as handle:
        pickle.dump(loss_tracker, handle)
    with open(os.path.join(target_dir, "times.p"), "wb") as handle:
        pickle.dump(times_per_epoch, handle)
    with open(os.path.join(target_dir, 'split_mapping.txt'), 'w+') as handle:
        train, valid = split_mapping[split_num]
        handle.write(f"Training fold numbers = {train}\n")
        handle.write(f"Validation fold numbers = {valid}\n")
        handle.close()
    
    print("All loss information saved")
    
def create_split_mapping(s: Settings) -> Dict:
    r"""Creates a fold mapping for the case where individual folds are
        combined to create total training/validation data
    
    Arguments:
        s (Settings): The Settings object with hyperparameter values
    
    Returns:
        mapping (Dict): The dictionary mapping current fold number to the 
            numbers of individual folds for train and validate. This only applies
            when you are combining individual folds. Each entry in the dictionary
            contains a list of two lists, the first inner list is the fold numbers 
            for training and the second inner list is the fold numbers for validation.
    
    Notes: Suppose we are training on five different folds / blocks of data numbered
        1 -> N. In the first training iteration in a CV driver mode, if the cv_mode is 
        'normal', we will train on the combined data of N - 1 folds together and test 
        on the remaining Nth fold. If the cv_mode is 'reverse', we will validate 
        on N - 1 folds while training on the remaining Nth fold. In previous iterations,
        each fold really contained a training set of validation set of feed dictionaries;
        now each fold means just one set of feed dictionaries, and we have to use all folds
        for every iteration of CV. 
    """
    num_directories = len(list(filter(lambda x : '.' not in x, os.listdir(s.top_level_fold_path))))
    num_folds = s.num_folds
    #num_folds should equal num_directories
    assert(num_folds == num_directories)
    cv_mode = s.cv_mode
    full_fold_nums = [i for i in range(num_folds)]
    mapping = dict()
    for i in range(num_folds):
        mapping[i] = [[],[]]
        if cv_mode == 'normal':
            mapping[i][1].append(i)
            mapping[i][0] = full_fold_nums[0 : i] + full_fold_nums[i + 1:]
        elif cv_mode == 'reverse':
            mapping[i][0].append(i)
            mapping[i][1] = full_fold_nums[0 : i] + full_fold_nums[i + 1:]
    return mapping

def convert_key_to_num(elem: Dict) -> Dict:
    return {int(k) : v for (k, v) in elem.items()}

def run_method(settings_filename: str, defaults_filename: str) -> None:
    r"""The main method for running the cldriver
    
    Arguments:
        settings_filename (str): The filename for the settings json file
        defaults_filename (str): The filename for the default settings json file
    
    Returns:
        None
    
    Notes: The stages of the computation are as follows:
        1) Precompute
        2) Training loop
        3) Resulting models, output, and logging
    
        Each of these parts will be its own separate method, and the precompute stage
        part will vary slightly based on whether we are using folds or not
    
    TODO: Right now, it is only set to handle non-fold CV as a testing step
    """
    #Read the input files and construct the settings dictionary
    with open(settings_filename, "r") as read_file:
        input_settings_dict = json.load(read_file)
    with open(defaults_filename, "r") as read_file:
        default_settings_dict = json.load(read_file)
    final_settings = construct_final_settings_dict(input_settings_dict, default_settings_dict)
    
    print(final_settings) 
    
    #Convert settings to an object for easier handling
    settings = Settings(final_settings)
    
    #Information on the run id:
    print(f"run id: {settings.run_id}")
    print(f"CV setting: {settings.driver_mode}")
        
    #Load the parameter dictionary
    if settings.par_dict_name == 'auorg_1_1':
        from auorg_1_1 import ParDict
        par_dict = ParDict()
    else:
        module = importlib.import_module(settings.par_dict_name)
        par_dict = module.ParDict()
    
    if settings.driver_mode == "non-CV":
        #Do the pre-compute stage
        all_models, model_variables, training_feeds, validation_feeds, training_dftblsts, validation_dftblsts, losses, all_losses, loss_tracker = pre_compute_stage(settings, par_dict)
        
        #Remove R models from backpropagation if dealing with new rep setting
        if settings.rep_setting == 'new':
            exclude_R_backprop(model_variables)
    
        #Do the training loop stage
        reference_energy_params, loss_tracker, all_models, model_variables, times_per_epoch = training_loop(settings, all_models, model_variables, training_feeds, validation_feeds,
                                                                                                        training_dftblsts, validation_dftblsts, losses, all_losses, loss_tracker)
        write_output_skf(settings, all_models)
        return reference_energy_params, loss_tracker, all_models, model_variables, times_per_epoch
    
    elif settings.driver_mode == "CV":
        #In CV, we need to keep track of models, variables, and ranges
        established_models = None
        established_variables = None
        
        #Get the folds regardless (very fast)
        print("Getting folds")
        folds_cv = get_folds_cv_limited(settings.allowed_Zs, settings.heavy_atoms, 
                                        settings.data_path, settings.num_folds, 
                                        settings.max_config, settings.exclude, tuple(settings.shuffle),
                                        reverse = False if settings.cv_mode == 'normal' else True)
        
        #If the fold_load_form is combine_individual_folds, then we need to create a mapping for the fold numbers:
        
        fold_mapping = convert_key_to_num(settings.split_mapping) if settings.split_mapping is not None else create_split_mapping(settings)
        print(fold_mapping)
        print("Done getting folds")
        
        dummy_folds = [None for i in range(len(fold_mapping))]
        
        # import pdb; pdb.set_trace()
        
        init_repulsive = True #Always initialize repulsive model at the beginning 
        
        for ind, fold in enumerate(folds_cv[:len(fold_mapping.keys())] if len(folds_cv) >= len(fold_mapping) else dummy_folds):
            #This is a HACK to constrain the number of iterations to the number of keys in fold_mapping. If a split_mapping
            # is provided, then only that many splits will be iterated over. If no split mapping is provided, then
            # the number iterations will be equal to num_folds. If split_mapping has more splits than folds, then 
            # the dummy folds will be used
            all_models, model_variables, training_feeds, validation_feeds, training_dftblsts, validation_dftblsts, losses, all_losses, loss_tracker = pre_compute_stage(settings, par_dict, fold, ind, fold_mapping, 
                                                                                                                                                                        established_models, established_variables)
            
            if settings.rep_setting == 'new':
                exclude_R_backprop(model_variables)
            
            reference_energy_params, loss_tracker, all_models, model_variables, times_per_epoch = training_loop(settings, all_models, model_variables, training_feeds, validation_feeds,
                                                                                                        training_dftblsts, validation_dftblsts, losses, all_losses, loss_tracker, init_repulsive)
            
            init_repulsive = False #No longer need to initialize repulsive model, just update it
            
            write_output_lossinfo(settings, loss_tracker, times_per_epoch, ind, fold_mapping)
            
            write_output_skf(settings, all_models) #Write the skf files each time just in case things crash on PSC
            
            if (established_models is not None) and (established_variables is not None):
                assert(all_models is established_models)
                assert(model_variables is established_variables)
                
            if ind == len(fold_mapping.keys()) - 1: #The should write after the final split
                write_output_skf(settings, all_models)
                return reference_energy_params, loss_tracker, all_models, model_variables, times_per_epoch
            
            established_models = all_models
            established_variables = model_variables
            
            assert(all_models is established_models)
            assert(model_variables is established_variables)

if __name__ == "__main__":
    args = parser.parse_args()
    enable_print = 1 if args.verbose else 0
    
    # Testing code
    # This testing code assumes the following command line is used:
    # %python cldriver.py settings_default.json defaults.json --verbose
    # the --verbose flag enables all print statements globally
    
    # print("Reading input json files...")
    
    # with open(args.settings, 'r') as read_file:
    #     input_settings_dict = json.load(read_file)
    # with open(args.defaults, 'r') as read_file:
    #     default_settings_dict = json.load(read_file)
        
    # print("Finished reading json files.")
        
    # # Check the final settings dictionary
    # print("Generating final settings dictionary...")
    
    # final_settings_dict = construct_final_settings_dict(input_settings_dict, default_settings_dict)
    # for item in final_settings_dict:
    #     print(item, ":", final_settings_dict[item])
    
    # print("Finished generating final settings dictionary.")
        
    # # Make sure that the par_dict is loading the correct keys
    # settings = Settings(final_settings_dict)
    
    # print("Loading in skf parameter dictionary...")
    # if settings.par_dict_name == 'auorg_1_1':
    #     from auorg_1_1 import ParDict
    #     par_dict = ParDict()
    # else:
    #     module = importlib.import_module(settings.par_dict_name)
    #     par_dict = module.ParDict()

    # print("SKF pardict keys:")
    # print(par_dict.keys())
    # print("Finished loading in skf parameter dictionary.")
    
    # # Try generating some molecules from it
    # # We know from previous experiments that for up to 5 heavy atoms, this
    # # dataset should give 1134 molecules
    
    # print("Generating feeds and batches for testing")
    # training_feeds, training_dftblsts, training_batches, validation_feeds, validation_dftblsts, validation_batches = get_graph_data_noCV(settings, par_dict)
    # print(f"Number of training feeds: {len(training_feeds)}")
    # print(f"Number of validation feeds: {len(validation_feeds)}")
    
    # # Cross-reference the molecules generated here with the ones generated from 
    # # dftb_layer_driver.py:
    
    # from functools import reduce # O(N^2) check in the number of molecules
    # print("Flattening to get all molecules back")
    # flattened_train_molecs = list(reduce(lambda x, y : x + y, training_batches))
    # flattened_validation_molecs = list(reduce(lambda x, y : x + y, validation_batches))
    # total_molecs = flattened_train_molecs + flattened_validation_molecs
    
    # print("Checking total number of molecules")
    # assert(len(total_molecs) == 1134) #We know this from previous experiments
    # with open("molecule_test.p","rb") as handle:
    #     reference_molecs = pickle.load(handle)
    # print("loading reference molecules and doing a direct comparison")
    # assert(len(reference_molecs) == len(total_molecs))
    
    # test_name_config = set([(x['name'], x['iconfig']) for x in total_molecs])
    # ref_name_config = set([(y['name'], y['iconfig']) for y in reference_molecs])
    # assert(test_name_config == ref_name_config)
    # assert(len(test_name_config) == len(ref_name_config) == len(total_molecs) == 1134)
    # print("Molecules are the same")
    
    # print("Testing precompute stage, no CV")
    # all_models, model_variables, training_feeds, validation_feeds, training_dftblsts, validation_dftblsts, losses, all_losses, loss_tracker = pre_compute_stage(settings, par_dict)
    
    # print("Keys for first training feed:")
    # print(training_feeds[0].keys())
    
    # print("Checking correct cutoffs, should be 3.0")
    # for model in all_models:
    #     if hasattr(all_models[model], 'cutoff'):
    #         assert(all_models[model].cutoff == 3.0)
    #         assert(model.oper in ["S", "H", "R"])
    
    # print("Checking length of B-spline coefficients, should be 49 (num_knots - 1)")
    # for model in all_models:
    #     print(model)
    #     if isinstance(all_models[model], Input_layer_pairwise_linear_joined):
    #         assert(model.oper in ["S", "H", "R"])
    #         print(all_models[model].get_total())
    #         assert(len(all_models[model].get_total()) == 49)
    #     elif isinstance(all_models[model], OffDiagModel2):
    #         assert(model.oper == "G")
    #     else:
    #         print(all_models[model].get_variables()) #Eref and value case
    #         if model == 'Eref':
    #             assert(len(all_models[model].get_variables()) == 5)
    #         else:
    #             assert(len(all_models[model].get_variables()) == 1)
    
    # print("Asserting that no inflection models made it into all_models")
    # for model in all_models:
    #     if hasattr(model, 'orb'):
    #         assert('inflect' not in model.orb)
    
    # print("Quick check on model_variables")
    
    # for model in model_variables:
    #     if model == 'Eref':
    #         assert(len(model_variables[model]) == 5)
    #     elif model.oper == 'G':
    #         assert(len(model_variables[model]) == 1)
    #         assert(len(model.Zs) == 1)
    #     elif 'inflect' in model.orb:
    #         assert(len(model_variables[model]) == 1)
    #         assert(model.oper == 'S')
    #     else:
    #         print(len(model_variables[model]))
    
    # #Full training loop testing (runs through all steps of the computation process)
    # # Calls the run_method which does all the computations in the non-CV case
    
    # print("Calling run method for full runthrough")
    # reference_energy_params, loss_tracker, all_models, model_variables, times_per_epoch = run_method(args.settings, args.defaults)
    
    # for loss in loss_tracker:
    #     print(f"Final {loss} train: {loss_tracker[loss][1][-1]}")
    #     print(f"Final {loss} valid: {loss_tracker[loss][0][-1]}")
    
    ## Testing for the CV case
    reference_energy_params, loss_tracker, all_models, model_variables, times_per_epoch = run_method(args.settings, args.defaults)
    print(loss_tracker)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
    

