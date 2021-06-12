# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 13:08:28 2021

@author: fhu14

TODO: Add in functionality for saving the batches from the precomputes
"""
#%% Imports, definitions
from typing import List, Dict
from DFTBLayer import DFTBList, graph_generation, model_loss_initialization,\
    model_range_correction, feed_generation
from DataManager import save_feed_h5
from .util import energy_correction
import os
import re
import pickle
import random

#%% Code behind

def single_fold_precompute(s, molecs: List[Dict], par_dict: Dict) -> (List[Dict], List[DFTBList]):
    r"""Runs through the precompute process for a single fold rather than a
        pair (train, validate)
    
    Arguments:
        s: The settings file containing all the hyperparameter settings
        molecs (List[Dict]): The list of molecules to generate the graphs and batches for
        par_dict (Dict): SKF parameter dictionary
        fold_num (int): The fold number to save the information under
    
    Returns:
        feeds (List[Dict]): The list of feed dictionaries
        dftblsts (list[DFTBList]): The list of DFTBList objects
        all_models (Dict): Dictionary mapping model specs to 
        model_variables (Dict): Dictionary of the variables that will be optimized
            by the training process
        losses (Dict): Dictionary keeping track of the weights for each target.
        all_losses (Dict): Dictionary mapping to the different loss objects.
        loss_tracker (Dict): Dictionary used to keep track of loss information
        batches (List[List[Dict]]): List of lists of molecule dictionaries; each inner
            list corresponds to each of the feeds in a 1-to-1 correspondence, 
            and the length of feeds and batches should be equal.
        
    Notes: Different from before, a fold now consists only of a single set of molecules.
        The "train" and "validate" folds are then chosen from among the general set 
        of folds.
    """
    config = {"opers_to_model" : s.opers_to_model}
    feeds, dftb_lsts, batches = graph_generation(molecs, config, s.allowed_Zs, par_dict, s.num_per_batch)
    
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
            
    all_models, model_variables, loss_tracker, all_losses, model_range_dict = model_loss_initialization(feeds, [],
                                                                               s.allowed_Zs, losses, s.tensor_device, s.tensor_dtype, ref_ener_start = s.reference_energy_starting_point)
    
    print("Performing model range correction")
    model_range_dict = model_range_correction(model_range_dict, s.low_end_correction_dict, universal_high = s.universal_high)
    
    feed_generation(feeds, batches, all_losses, all_models, model_variables, model_range_dict, par_dict, s.spline_mode, s.spline_deg, 
                    s.tensor_device, s.tensor_dtype, s.debug, False, 
                    s.num_knots, s.buffer, s.joined_cutoff, s.cutoff_dictionary, s.off_diag_opers, s.include_inflect)
    
    print(f"inflect mods: {[mod for mod in model_variables if mod != 'Eref' and mod.oper == 'S' and 'inflect' in mod.orb]}")
    print(f"s_mods: {[mod for mod in model_variables if mod != 'Eref' and mod.oper == 'S']}")
    print(f"len of s_mods: {len([mod for mod in model_variables if mod != 'Eref' and mod.oper == 'S'])}")
    print(f"len of s_mods in all_models: {len([mod for mod in all_models if mod != 'Eref' and mod.oper == 'S'])}")
    print("losses")
    print(losses)
    
    return feeds, dftb_lsts, all_models, model_variables, losses, all_losses, loss_tracker, batches

def compute_graphs_from_folds(s, top_level_molec_path: str, copy_molecs: bool) -> None:
    r"""Computes and saves the feed dictionaries for all the molecules in each fold
    
    Arguments:
        s (Settings): The settings object containing all the hyperparameter values
        top_level_molec_path (str): The relative path to the directory containing 
            the molecules of each fold
        copy_molecs (bool): Whether or not to duplicate the raw molecule pickle files
            in the directories with the saved h5 files (mostly for debugging purposes)
    
    Returns:
        None
        
    Notes: This function does two main things for each set of molecules found
        within the folder top_level_molec_path:
        1) Generate the feed dictionaries for each set of molecules
        2) Saves the feed dictionaries in h5 format
    Nothing is returned from this function
    """
    all_files = os.listdir(top_level_molec_path)
    pattern = r"Fold[0-9]+_molecs.p"
    fold_file_names = list(filter(lambda x : re.match(pattern, x), all_files))
    
    par_dict = s.par_dict_name
    
    #Now cycle through each fold and do the precompute on it
    for name in fold_file_names:
        total_path = os.path.join(top_level_molec_path, name)
        fold_num = name.split('_')[0][-1]
        with open(total_path, 'rb') as handle:
            molecs = pickle.load(handle)
            random.shuffle(molecs)
            if s.train_ener_per_heavy: #Only perform the energy correction if training per heavy atom
                for elem in molecs:
                    energy_correction(elem)
            feeds, dftb_lsts, _, _, _, _, _, batches = single_fold_precompute(s, molecs, par_dict)
            destination = os.path.join(top_level_molec_path, f"Fold{fold_num}")
            save_feed_h5(feeds, dftb_lsts, molecs, destination, batches, duplicate_data = copy_molecs)
            print(f"Data successfully saved for {name} molecules")
            
    print(f"All data successfully saved for molecules in {top_level_molec_path}")

