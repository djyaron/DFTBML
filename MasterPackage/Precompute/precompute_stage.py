# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 18:02:59 2021

@author: fhu14

This should take care of loading the data in for the training loop
"""

#%% Imports, definitions
from DataManager import load_combined_fold
from typing import Dict
from DFTBLayer import model_loss_initialization, model_range_correction,\
    feed_generation, total_type_conversion
from Dispersion import LJ_Dispersion


#%% Code behind

def precompute_stage(s, par_dict: Dict, split_num: int, fold_mapping_dict: Dict, 
                     established_models: Dict = None, established_variables: Dict = None):
    r"""Performs precompute to get the feeds and other structures for the
        main training cycle
        
    Arguments:
        s (Settings): The settings object containing values for all the hyperparameters
        par_dict (Dict): Dictionary of skf parameters
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
        training_batches (List[List[Dict]]): The individual batches for all the training feeds
        validation_batches (list[List[Dict]]): The individual batches for all the validation feeds
    """
    #Load the information in for the current split
    print(f"Loading data for split {split_num}")
    top_fold_path = s.top_level_fold_path
    training_feeds, validation_feeds, training_dftblsts, validation_dftblsts, training_batches, validation_batches = load_combined_fold(s, top_fold_path, split_num, fold_mapping_dict)
    
    #Create the loss dictionary
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
    
    #Initialize models
    print("Initializing models")
    all_models, model_variables, loss_tracker, all_losses, model_range_dict = model_loss_initialization(training_feeds, validation_feeds,
                                                                                                    s.allowed_Zs, losses, s.tensor_device, s.tensor_dtype,
                                                                                                    ref_ener_start = s.reference_energy_starting_point)
    
    #Use established models and variables if they exist
    if (established_models is not None) and (established_variables is not None):
        print("Loading in previous models, variables, and ranges")
        all_models = established_models
        model_variables = established_variables
    
    #Perform range correction
    model_range_dict = model_range_correction(model_range_dict, s.low_end_correction_dict, s.cutoff_dictionary, s.joined_cutoff)
    
    #Generate feeds and perform type conversions
    feed_generation(training_feeds, training_batches, all_losses, all_models, model_variables, model_range_dict, par_dict, s.spline_mode, s.spline_deg,
                    s.tensor_device, s.tensor_dtype, s.debug, s.loaded_data, 
                    s.num_knots, s.buffer, s.joined_cutoff, s.cutoff_dictionary, s.off_diag_opers, s.include_inflect)
    
    print("Generating validation feeds")
    feed_generation(validation_feeds, validation_batches, all_losses, all_models, model_variables, model_range_dict, par_dict, s.spline_mode, s.spline_deg,
                    s.tensor_device, s.tensor_dtype, s.debug, s.loaded_data, 
                    s.num_knots, s.buffer, s.joined_cutoff, s.cutoff_dictionary, s.off_diag_opers, s.include_inflect)
    
    if s.dispersion_correction:
        print("Adding in dispersion correction")
        #For now, only going to use LJ dispersion, may expand to other dispersion 
        #   schemes later. 
        all_models['disp'] = LJ_Dispersion(s.tensor_device, s.tensor_dtype)
        #Add the dispersion parameters to the model variables dictionary. Add
        #   as individual tensors
        r_ij, d_ij = all_models['disp'].get_variables()
        for elems, val in r_ij.items():
            model_variables[f"disp_r_{elems}"] = val
        for elems, val in d_ij.items():
            model_variables[f"disp_d_{elems}"] = val
    
    print("Performing type conversion to tensors")
    total_type_conversion(training_feeds, validation_feeds, ignore_keys = s.type_conversion_ignore_keys,
                          device = s.tensor_device, dtype = s.tensor_dtype)
    
    print("Some information:")
    print(f"inflect mods: {[mod for mod in model_variables if (not isinstance(mod, str)) and mod.oper == 'S' and 'inflect' in mod.orb]}")
    print(f"s_mods: {[mod for mod in model_variables if (not isinstance(mod, str)) and mod.oper == 'S']}")
    print(f"len of s_mods: {len([mod for mod in model_variables if (not isinstance(mod, str)) and mod.oper == 'S'])}")
    print(f"len of s_mods in all_models: {len([mod for mod in all_models if (not isinstance(mod, str)) and mod.oper == 'S'])}")
    print("losses")
    print(losses)
    print(all_losses)
    
    return all_models, model_variables, training_feeds, validation_feeds, training_dftblsts, validation_dftblsts, losses, all_losses, loss_tracker, training_batches, validation_batches