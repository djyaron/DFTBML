# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 13:08:28 2021

@author: fhu14

"""
#%% Imports, definitions
from typing import List, Dict
from DFTBLayer import DFTBList, graph_generation, model_loss_initialization,\
    model_range_correction, feed_generation
from DFTBrepulsive import compute_gammas
from InputLayer import generate_gammas_input
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
    
    The fields of s that matter are as follows:
        1) par_dict_name (Dict): The parameter dictionary for calculations
        2) train_ener_per_heavy (bool): Whether energies are trained per heavy atom
        3) opers_to_model (List[str]): The list of operators to model
        4) allowed_Zs (List[int]): The allowed elements in the molecules of the dataset
        5) num_per_batch (int): The number of molecules to have per batch
        6) losses (List[str]): The list of targets to include in the loss function
        7) target_accuracy_[loss] (float): The weight given to each loss contained
            in losses.
        8) tensor_device (torch.device): The device to use for generated tensors
        9) tensor_dtype (torch.dtype): The dtype to use for your generated tensors
        10) reference_energy_starting_point (List[float]): The reference energy
            parameters to use
        11) low_end_correction_dict (Dict): Dictionary containing low ends of ranges for
            atom pairs.
        12) universal_high (float): The maximal range for all atom pairs
        13) spline_mode (str): The mode of splines to use
        14) spline_deg (int): The degree of the spline to use
        15) debug (bool): Whether or not debugging mode is being used. Should 
            always be set to false.
        16) num_knots (int): The number of knots for the splines
        17) buffer (float): How much to extend the ends of the spline range by
        18) joined_cutoff (float): The cutoff point for splines 
        19) cutoff_dictionary (Dict): The dictionary indicating the cutoffs for
            different element pairs
        20) off_diag_opers (List[str]): The operators to be modeled differently
            than a normal univariate spline (i.e. hubbard G)
        21) include_inflect (bool): Whether the inflection point penalty should be 
            included for the S (overlap) operators.
    
    The values may not always be relevant, but be sure that these are 
    set appropriately.
    """
    all_files = os.listdir(top_level_molec_path)
    pattern = r"Fold[0-9]+_molecs.p"
    fold_file_names = list(filter(lambda x : re.match(pattern, x), all_files))
    
    par_dict = s.par_dict_name
    
    #Now cycle through each fold and do the precompute on it
    for name in fold_file_names:
        total_path = os.path.join(top_level_molec_path, name)
        fold_num = name.split('_')[0][4:]
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

def precompute_gammas(opts: Dict, top_level_molec_path: str) -> None:
    r"""Does the precompute process for the entire dataset to generate gammas. 
        Also saves the config_tracker in a pickle file for use with gammas.
    
    Arguments: 
        opts (Dict): The dictionary with hyperparameter settings for the
            DFTBrepulsive model. 
        top_level_molec_path (str): The relative path to the directory 
            containing the molecules stored in pickle files. 
    
    Returns: None
    
    Notes: This method saves the gammas and config tracker to a pickle format. 
        The config tracker is important because the gammas ordering is unique
        in terms of the configs of the molecules for a given empirical formula.
        
        The gammas should be saved with the dataset, so the path should be
        top_level_molec_path + gammas.p, and this path should be specified
        correctly in the opts dictionary!
        
        In specifying paths in the json configuration file, the 
        separator used in the path should be the forward slash, '/'. 
    """
    all_files = os.listdir(top_level_molec_path)
    pattern = r"Fold[0-9]+_molecs.p"
    fold_file_names = list(filter(lambda x : re.match(pattern, x), all_files))
    
    mols_2D = [pickle.load(open(os.path.join(top_level_molec_path, name), 'rb')) for name in fold_file_names]
    
    gammas_input, config_tracker = generate_gammas_input(mols_2D)
    
    gammas_path = opts['repulsive_settings']['gammas_path']
    gpath_splt = gammas_path.split("/")
    
    gammas = compute_gammas(gammas_input, opts, True)
    
    with open(os.path.join(top_level_molec_path, "config_tracker.p"), "wb") as handle:
        pickle.dump(config_tracker, handle)
    
    with open(os.path.join(top_level_molec_path, "gammas.p"), "wb") as handle:
        pickle.dump(gammas, handle)
    
    print("Gammas and config tracker have been saved")

def precompute_gammas_per_fold(opts: Dict, top_level_molec_path: str) -> None:
    r"""Similar functionality as precompute_gammas but the gammas are 
        computed for each fold and saved for each fold rather than for the entire
        dataset.
    """
    all_files = os.listdir(top_level_molec_path)
    pattern = r"Fold[0-9]+_molecs.p"
    fold_file_names = list(filter(lambda x : re.match(pattern, x), all_files))
    
    for name in fold_file_names:
        full_name = os.path.join(top_level_molec_path, name)
        molecs = pickle.load(open(full_name, 'rb'))
        #The input to generate_gammas_input has to be a 2D list due to internal operations,
        #   so wrap the 1D list in another list
        gammas_input, config_tracker = generate_gammas_input([molecs])
        gammas = compute_gammas(gammas_input, opts, return_gammas = True)
        fold_name = name.split("_")[0]
        gammas_name = os.path.join(top_level_molec_path,  f"{fold_name}_gammas.p")
        config_name = os.path.join(top_level_molec_path, f"{fold_name}_config_tracker.p")
        
        with open(gammas_name, 'wb') as handle:
            pickle.dump(gammas, handle)
        
        with open(config_name, 'wb') as handle:
            pickle.dump(config_tracker, handle)
        
        print(f"Gammas and config_tracker saved for {name}")
    