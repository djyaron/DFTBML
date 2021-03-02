# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 21:01:02 2021

@author: Frank

Module with various methods to help with loading batches one by one to conserve memory

DISCLAIMER: 
    This more memory efficient approach loads each batch in one at a time as they
    are passed through the training loop so that we do not have to load everything 
    into memory at once. Thus, it only works with "combine_individual_folds" and 
    requires a specification for the split_mapping

TODO:
    1) Change settings file to contain a field called "universal_high" which encodes the maximum distance
        for all splines to span (X)
    2) A TON OF DEBUGGING / TESTING
        Problem with mixing tensors into molec dicts during charge updates, causing failures
    
Uncertainties:
    1) The matrices A and b for each spline model (y = Ax + b) are repeatedly calculated and added into the 
        feeds over training since the feeds are loaded one at a time. Thus, A and b each time
        come from progressively more trained splines, which is different from when A and b were
        computed once at the start and added in, with those same matrices used throughout. The past approach
        only worked because all the feeds were together in memory at once, which is no longer the 
        case. The code in tfspline does not seem to have a dependence on the current spline coefficients, 
        for spline_new_xvals or spline_linear_model, so this should not be a problem. However, making a note of it here for
        future reference.
        
"""
import pickle, json
import time
import numpy as np
from h5handler import total_feed_combinator, per_molec_h5handler, get_model_from_string, compare_feeds
from dftb_layer_splines_4 import get_model_value_spline_2, OffDiagModel2, assemble_ops_for_charges,\
    update_charges, DFTB_Layer, recursive_type_conversion, Reference_energy
from batch import DFTBList
from loss_models import TotalEnergyLoss, FormPenaltyLoss, DipoleLoss, ChargeLoss, DipoleLoss2
from typing import List, Union, Dict
from skfwriter import main, atom_nums, atom_masses
import torch.optim as optim
import torch
from batch import Model
from functools import reduce
from fold_generator import Settings
import random
import importlib
import os, os.path
import sys
import h5py
import re

#%% General functions and methods

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

#%% Functions for handling single batches
def collect_all_models(batch_file: h5py.File) -> List[Model]:
    r"""Collects all the models needed for a batch file
    
    Arguments:
        batch_file (h5py.File): The h5 file that contains the batch information
    
    Returns:
        models (List[Model]): The list of all the model_spec objects that
            we need for the given batch
    """
    model_lst = list()
    for batch in batch_file:
        raw_mod_specs = batch_file[batch]['models'][()]
        model_lst.extend(list(map(lambda x : get_model_from_string(x), raw_mod_specs)))
    
    return list(set(model_lst))

def collect_all_models_all_batches(batch_file_lst: List[h5py.File]) -> List[Model]:
    r"""Grabs all the models needed for a series of batches
    
    Arguments:
        batch_file_lst (List[h5py.File]): List of h5py file pointers for batch h5 files
        
    Returns:
        The list of all the models needed for all batches
    """
    all_mods_lsts = list(map(lambda x : collect_all_models(x), batch_file_lst))
    flattened_all_mods_lst = list(reduce(lambda x, y : x + y, all_mods_lsts))
    unique_mod_specs = list(set(flattened_all_mods_lst))
    unique_mod_specs.sort(key = lambda x : len(x.Zs))
    return unique_mod_specs

def generate_losses_losstracker(s: Settings):
    r"""Function to generate losses and loss trackers based off the settings file
    
    Arguments:
        s (Settings): The Settings object containing all the hyperparameter settings
    
    Returns:
        losses (Dict): A dictionary of losses mapping loss name to the scaling factor
        all_losses (Dict): A dictionary mapping each target type to the loss model
            that handles that specific target (e.g. TotalEnergyLoss)
        loss_tracker (Dict): A dictionary where each value is a list with 3 elements:
            1) The list of validation losses for that specific target
            2) The list of training losses for that specific target
            3) A value for keeping track of averages
    """
    losses, all_losses, loss_tracker = dict(), dict(), dict()
    
    for loss in s.losses:
        #s.losses is a list of strings representing the different losses to factor into backpropagation
        if loss == 'Etot':
            losses[loss] = s.target_accuracy_energy
            all_losses[loss] = TotalEnergyLoss()
            loss_tracker[loss] = [list(), list(), 0]
        elif loss == 'dipole':
            losses[loss] = s.target_accuracy_dipole
            all_losses[loss] = DipoleLoss2()
            loss_tracker[loss] = [list(), list(), 0]
        elif loss == 'charges':
            losses[loss] = s.target_accuracy_charges
            all_losses[loss] = ChargeLoss()
            loss_tracker[loss] = [list(), list(), 0]
        elif loss == 'convex':
            losses[loss] = s.target_accuracy_convex
            all_losses[loss] = FormPenaltyLoss(loss)
            loss_tracker[loss] = [list(), list(), 0]
        elif loss == 'monotonic':
            losses[loss] = s.target_accuracy_monotonic
            all_losses[loss] = FormPenaltyLoss(loss)
            loss_tracker[loss] = [list(), list(), 0]
        else:
            raise ValueError("Unsupported loss type")
    
    return losses, all_losses, loss_tracker

def generate_spline_range_dictionary(s: Settings, all_model_specs: List[Model]) -> Dict:
    r"""Generates the spline range dict that is needed for creating the splines
    
    Arguments:
        s (Settings): The Settings object containing all the hyperparameter settings
        all_model_specs (List[Model]): The list of all the model_specs whose ranges 
            need to be found
    
    Returns:
        model_range_dict (Dict): Dictionary mapping each model_spec to a tuple of form
            (xlow, xhigh) where xlow and xhigh are in angstroms and together describe the total 
            distance of the spline
    
    Notes: Because we want spline ranges to be independent of the data, the new
        method is to fix some arbitrary maximum distance for all models regardless of model type.
        The arbitrary maximum distance, termed the "universal_high", is encoded 
        in the settings file
        
        For the purposes of this approach, xlow and xhigh must be known in advance for all models,
        and this information must be accessible in the settings file
    """
    s.low_end_correction_dict = dictionary_tuple_correction(s.low_end_correction_dict)
    model_range_dict = dict()
    for mod_spec in all_model_specs:
        if len(mod_spec.Zs) == 2: #Only two-body splines need to have their ranges known
            Zs, Zs_rev = mod_spec.Zs, (mod_spec.Zs[1], mod_spec.Zs[0])
            if Zs in s.low_end_correction_dict:
                model_range_dict[mod_spec] = (s.low_end_correction_dict[Zs], s.universal_high)
            elif Zs_rev in s.low_end_correction_dict:
                model_range_dict[mod_spec] = (s.low_end_correction_dict[Zs_rev], s.universal_high)
            else:
                raise ValueError(f"range of {mod_spec} cannot be determined")
                
    return model_range_dict

def pull_par_dict(s: Settings) -> Dict:
    r"""Imports the correct skf parameter dictionary
    
    Arguments:
        s (Settings): The Settings object containing all the hyperparameter settings
    
    Returns:
        pardict (Dict): A dictionary containing all the relevant SKF parameters
    """
    if s.par_dict_name == 'auorg_1_1':
        from auorg_1_1 import ParDict
        pardict = ParDict()
    else:
        module = importlib.import_module(s.par_dict_name)
        pardict = module.ParDict()
    
    return pardict

def generate_models_variables(s: Settings, all_model_specs: List[Model]) -> Dict:
    r"""Generates the all_models and model_variables dictionaries
    
    Arguments:
        s (Settings): Settings object containing all the hyperparameter settings
        all_model_specs (List[Model]): The list of all model specs to generate models for
    
    Returns:
        all_models (Dict): Dictionary of all models for all batches, completed
        model_variables (Dict): Dictionary of all model variables
    
    Notes: Because of the individual nature of fold loading in our new 
        memory efficient approach, it becomes necessary to have the model 
        generation process be independent of the feed generation process.
        
        Code copied from feed_generation in dftb_layer_splines_4.py
    """
    all_models, model_variables = dict(), dict()
    
    spline_range_dict = generate_spline_range_dictionary(s, all_model_specs)
    par_dict = pull_par_dict(s)
    
    s.cutoff_dictionary = dictionary_tuple_correction(s.cutoff_dictionary)
    
    for model_spec in all_model_specs:
        model, tag = get_model_value_spline_2(model_spec, model_variables, spline_range_dict,
                                              par_dict, s.num_knots, s.buffer, s.joined_cutoff, 
                                              s.cutoff_dictionary, s.spline_mode, s.spline_deg,
                                              s.off_diag_opers, s.include_inflect)
        all_models[model_spec] = model
        if tag != 'noopt' and not isinstance(model, OffDiagModel2):
            model_variables[model_spec] = all_models[model_spec].get_variables()
            if (hasattr(model, "inflection_point_var")) and (model.inflection_point_var is not None):
                old_oper, old_zs, old_orb = model_spec
                new_mod = Model(old_oper, old_zs, old_orb + '_inflect')
                model_variables[new_mod] = all_models[model_spec].get_inflection_pt()
        elif tag == 'noopt':
            #All tensors at this point
            all_models[model_spec].variables.requires_grad = False #Detach from computational graph
            
    allowed_Zs = s.allowed_Zs
    ref_ener_start = s.reference_energy_starting_point
    
    all_models['Eref'] = Reference_energy(allowed_Zs) if (ref_ener_start is None) else Reference_energy(allowed_Zs, prev_values = ref_ener_start)
    model_variables['Eref'] = all_models['Eref'].get_variables()
    
    return all_models, model_variables

def single_feed_pass_through(feed: Dict, all_losses: Dict, all_models: Dict, pardict: Dict, 
                             debug: bool = False) -> None:
    r"""Destructively modifies a given feed by adding in all losses feeds
    
    Arguments:
        feed (Dict): The feed dictionary to correct
        all_losses (Dict): Dictionary containing all the loss objects
        all_models (Dict): Dictionary containing all the models
        pardict (Dict): Dictionary containing the skf parameters
        debug (bool): Boolean indicating whether debug mode is enabled. Debug
            mode is deprecated, so defaults to False
    
    Returns:
        None
    
    Notes: Adds in the feed and loss information for each feed, code adapted 
        from feed_generation in dftb_layer_splines_4.py.
    """
    #Add in the feed information for each model
    for model_spec in feed['models']:
        model = all_models[model_spec]
        feed[model_spec] = model.get_feed(feed['mod_raw'][model_spec])
        
    #Add in the loss information for each loss
    for loss in all_losses:
        try:
            #In this case, you are only ever loading data, so the input batch is always []
            all_losses[loss].get_feed(feed, [], all_models, pardict, debug)
        except Exception as e:
            print(e)

def get_num_batches(batch_file_ptrs: List[h5py.File]) -> int:
    r"""Finds the number of batches in each fold
    
    Arguments:
        batch_file_ptrs (List[h5py.File]): List of open h5py file pointers
    
    Returns:
        num_batches (int): The number of batches encoded in each fold' batch h5
            file
    
    Notes: Each batch h5 file for each fold should have the same number of batches
        as they should have been constructed from the same number of underlying
        molecules
    """
    if len(batch_file_ptrs) == 0:
        return 0 #Return 0 if there are no open h5py file pointers
    num_0 = len(batch_file_ptrs[0].keys())
    for ptr in batch_file_ptrs:
        try:
            assert(len(ptr.keys()) == num_0)
        except:
            print("Not all batch h5 files have the same number of batches!")
    return num_0
    
def get_relative_batch_index(index: int, num_batches: int) -> int:
    r"""Simple math to figure out which batch file a batch came from, and 
        which molecule dictionary it needs
    
    Arguments:
        index (int): The index that the batch has when considered in conjunction
            with all other batches.
        num_batches (int): The number of batches that are contained in each 
            batch h5 file.
    
    Returns:
        file_num (int): Which batch file it belongs to
        true_index (int): The true index of the batch within that batch file
    
    Notes: When working with a collection of batch files together, the batches
        from each are combined together and labeled 1 - N. However, there needs
        to be an accounting system that maps each index in the combined vector to 
        the true index in its respective batch file. This is especially critical when
        shuffling the indices during training. This approach assumes that pointers
        to the open h5 files for batches are stored in arrays that have a 1:1
        correspondence with another array which stores the loaded molecule dictionaries.
        
        Say that each batch file contains x batches, and our current index in the
        combined batch array is y. Then, the index of the batch file that that batch
        uses is given as y // x and the true index within that batch is y % x.
    """
    file_num = index // num_batches
    true_index = index % num_batches
    return file_num, true_index

def create_fold_file_mappings(s: Settings) -> Dict:
    r"""Creates a dictionary mapping the fold names to the batch filenames and
        the molecule filenames
    
    Arguments:
        s (Settings): Settings object containing all the hyperparameter settings
    
    Returns:
        fold_file_map (Dict): A dictionary mapping the fold name to the batch filename, 
            molecule filename, and the dftblist filename
    
    Notes: This dictionary is mostly for efficiency considerations. The format is as 
        follows:
            {0 : {"foldname" : "Fold0",
                      "molec" : molec filename or dict,
                      "batch" : batch filename or pointer, 
                      "dftblst" : List of dftblist objects},
             ...}
        As each fold is used, its "molec" key gets replaced with its master molecule dictionary,
        its "batch" key gets replaced by the file pointer, and "dftblist" is changed with the
        loaded dftblsts from the pickle file. This avoids the overhead from
        reloading data that's already loaded. The top level keys are numbers
        for indexing convenience.
    """
    pattern = "Fold[0-9]+"
    all_files = os.listdir(s.top_level_fold_path)
    fold_names_only = list(filter(lambda x : (re.match(pattern, x) and "." not in x), all_files))
    
    fold_file_map = dict()
    for fold_name in fold_names_only:
        fold_num = int(re.findall(r"\d+", fold_name)[0])
        fold_file_map[fold_num] = {
            "molec" : os.path.join(s.top_level_fold_path, fold_name, "molecs.h5"),
            "batch" : os.path.join(s.top_level_fold_path, fold_name, "batches.h5"),
            "foldname" : fold_name,
            "dftblst" : os.path.join(s.top_level_fold_path, fold_name, "dftblsts.p")
            }
    
    return fold_file_map

def assemble_segment(fold_file_map: Dict, fold_indices: List[int]):
    r"""Creates the molec dict list, batch file pointer list, dftblst list, 
        and other constants for the given fold indices
    
    Arguments:
        fold_file_map (Dict): Dictionary mapping fold number to the filenames
        fold_indices (List[int]): The indices of the fold to use for this segment
    
    Returns:
        batch_file_ptrs (List[h5py.File]): The List of h5py file pointers to use
            when reading the batch information from for this segment
        num_batches (int): The number of batches in each h5 file contained in batch_file_ptrs
        molec_dict_lst (List[Dict]): The list of molecule dictionaries indexed by the 
            empirical formula and the configuration number
        dftblst_lst (List[DFTBList]): List of dftblsts to go along with the segment
        starting_indices (List[int]): The list of starting indices for the batches
    
    Notes: There is a 1:1 correspondence in position between the molecule dictionaries
        in molec_dict_lst and batch_file_ptrs where the ith molecule dictionary in
        molec_dict_lst works for the ith batch file in batch_file_ptrs.
    """
    batch_file_ptrs = list()
    molec_dict_lst = list()
    dftblst_lst = list()
    
    for fold_num in fold_indices:
        
        if isinstance(fold_file_map[fold_num]["molec"], str):
            molec_filename = fold_file_map[fold_num]["molec"]
            fold_file_map[fold_num]["molec"] = per_molec_h5handler.extract_molec_feeds_h5(molec_filename)
            molec_dict_lst.append(fold_file_map[fold_num]["molec"])
        # Need to reload the molecule dictionary because charge updates affect the values of these dictionaries
        #   and so need to make sure that original values are used
        # elif isinstance(fold_file_map[fold_num]["molec"], dict):
        #     molec_dict_lst.append(fold_file_map[fold_num]["molec"])
            
        if isinstance(fold_file_map[fold_num]["batch"], str):
            batch_filename = fold_file_map[fold_num]["batch"]
            fold_file_map[fold_num]["batch"] = h5py.File(batch_filename, "r")
            batch_file_ptrs.append(fold_file_map[fold_num]["batch"])
        elif isinstance(fold_file_map[fold_num]["batch"], h5py.File):
            batch_file_ptrs.append(fold_file_map[fold_num]["batch"])
        
        if isinstance(fold_file_map[fold_num]["dftblst"], str):
            dftblst_filename = fold_file_map[fold_num]["dftblst"]
            fold_file_map[fold_num]["dftblst"] = pickle.load(open(dftblst_filename, 'rb'))
            dftblst_lst.append(fold_file_map[fold_num]['dftblst'])
        elif isinstance(fold_file_map[fold_num]['dftblst'], list):
            dftblst_lst.append(fold_file_map[fold_num]['dftblst'])
    
    num_batches = get_num_batches(batch_file_ptrs) #number of batches in each batch file pointer
    starting_indices = [i for i in range(num_batches * len(batch_file_ptrs))]
    dftblst_lst = list(reduce(lambda x, y : x + y, dftblst_lst)) if len(dftblst_lst) > 0 else [] #Flattens the dftblst into a 1D list
    
    return batch_file_ptrs, num_batches, molec_dict_lst, dftblst_lst, starting_indices

def transform_seg_to_dict(batch_file_ptrs: List[h5py.File], num_batches: int, molec_dict_lst: List[Dict], 
                          dftblst_lst: List[DFTBList], starting_indices: List[int]) -> Dict:
    r"""Generates a dictionary for the current segment, takes as input the output of
        assemble_segment
    
    Arguments:
        batch_file_ptrs (List[h5py.File]): List of h5py File pointers
        num_batches (int): The number of batches contained in each h5py file referenced
            in bach_file_ptrs
        molec_dict_lst (List[Dict]): The list of molecule dictionaries indexed by empirical
            formula and configuration number
        dftblst_lst (List[DFTBList]): The list of the DFTBList objects to be used for the 
            charge updates
        starting_indices (List[int]): The index of the batches for the segment, numbered
            continuously from 1 to N.
    
    Return:
        seg_info (Dict): All the above information organized into a dictionary with
            intuitive keys:
                'file_ptrs' : batch_file_ptrs,
                'num_batches' : num_batches,
                'molec_dicts' : molec_dict_lst,
                'dftblst_lsts' : dftblst_lst,
                'indices' : starting_indices
    """
    return {'file_ptrs' : batch_file_ptrs,
            'num_batches' : num_batches,
            'molec_dicts' : molec_dict_lst,
            'dftblst_lsts' : dftblst_lst,
            'indices' : starting_indices}

#%% Training code

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

def compute_loss(s: Settings, all_losses: Dict, losses: Dict, loss_tracker: Dict, feed_in: Dict, output: Dict):
    r"""Computes the loss from a given output against the corresponding input
    
    Arguments:
        s (Settings): Settings object containing values for the different hyperparameter 
            settings
        all_losses (Dict): Dictionary mapping loss target to loss_model implementation. Each 
            value is an object implemented in loss_models.py
        losses (Dict): The dictionary mapping loss target to the weighting for the loss. Each 
            value is a float
        loss_tracker (Dict): The dictionary to keep track of loss information per 
            target. This dictionary is updated directly throughout the loss computation
        feed_in (Dict): The input dictionary
        output (Dict): The output dictionary
    
    Returns:
        tot_loss (torch.float): A total loss value with full backward
            differentiability
    """
    tot_loss = 0
    for loss in all_losses:
        if loss == 'Etot':
            if s.train_ener_per_heavy:
                val = losses[loss] * all_losses[loss].get_value(output, feed_in, True)
            else:
                val = losses[loss] * all_losses[loss].get_value(output, feed_in, False)
            tot_loss += val
            loss_tracker[loss][2] += val.item()
        elif loss == 'dipole':
            val = losses[loss] * all_losses[loss].get_value(output, feed_in)
            loss_tracker[loss][2] += val.item()
            if s.include_dipole_backprop:
                tot_loss += val
        else:
            val = losses[loss] * all_losses[loss].get_value(output, feed_in)
            tot_loss += val 
            loss_tracker[loss][2] += val.item()
            
    return tot_loss

def update_dQ_rhomask(molec_dict: Dict, updated_feed: Dict) -> None:
    r"""Updates the dQ and occ_rho_mask values in the molec_dict with the
        updated values in updated_feed
    
    Arguments:
        molec_dict (Dict): Dictionary indexed by empirical formula and then 
            configuration number
        updated_feed (Dict): The updated feed dictionary to pull the new dQ
            and occ_rho_mask values from
    
    Returns:
        None
    
    Notes: Because dQ and occ_rho_mask are saved on a per-molecule basis, 
        we can update the state of the molecule dictionaries which are
        persistent in memory
    """
    all_bsizes = updated_feed['basis_sizes']
    for bsize in all_bsizes:
        current_occ_rhos = updated_feed['occ_rho_mask'][bsize]
        current_dQs = updated_feed['dQ'][bsize]
        current_names = updated_feed['names'][bsize]
        current_iconfigs = updated_feed['iconfigs'][bsize]
        assert(len(current_occ_rhos) == len(current_names) == len(current_iconfigs) == len(current_dQs))
        for i in range(len(current_names)):
            #Assert same shape
            assert(molec_dict[current_names[i]][current_iconfigs[i]]['occ_rho_mask'].shape == current_occ_rhos[i].shape)
            assert(molec_dict[current_names[i]][current_iconfigs[i]]['dQ'].shape == current_dQs[i].shape)
            #Update the values in molec_dict
            #Convert to numpy arrays because only one element tensors (not lists of tensors) can be converted.
            #   this wasn't a problem in the past becuase all info was loaded before type conversion, and 
            #   no loading happened during training (no longer the case now)
            molec_dict[current_names[i]][current_iconfigs[i]]['occ_rho_mask'] = current_occ_rhos[i].numpy()
            molec_dict[current_names[i]][current_iconfigs[i]]['dQ'] = current_dQs[i].numpy()
            

def charge_update_subroutine(seg: Dict, all_models: Dict, all_losses: Dict,
                             par_dict: Dict, tag: str, s: Settings, epoch: int) -> None:
    r"""Does the charge update within the single-load infrastructure on a give segment
    
    Arguments:
        seg (Dict): The dictionary containing the information for the segment to 
            perform the charge updates on
        all_models (Dict): Dictionary containing references to each model object
        all_losses (Dict): Dictionary containing each target and the corresponding
            loss_model object
        par_dict (Dict): The dictionary containing the SKF parameter files
        tag (str): One of 'train', 'validate', or whatever, indicates the identity
            of the segment where the charge updates are being performed
        s (Settings): Settings object containing all the hyperparameter settings
        epoch (int): The number of the current epoch
    
    Returns: 
        None
    
    Notes: The dQ information is stored per molecule, so we need to update dQ and
        occ_rho_mask. Luckily, both of these are stored in the molecule dictionary,
        so we do not need to overwrite h5 file contents.
        
        Differing from past frameworks, this only performs the charge update for
        one segment, i.e. only train or only validation. Does not do both at once;
        done to reduce code duplication.
    """
    #Do the training feeds first
    print(f"Running {tag} set charge update for epoch {epoch}")
    batches, nperbatch, molec_dicts, dftblsts, indices = seg['file_ptrs'],\
        seg['num_batches'], seg['molec_dicts'], seg['dftblst_lsts'],\
            seg['indices']
    for index in indices:
        file_num, true_index = get_relative_batch_index(index, nperbatch)
        batch_file_ptr = batches[file_num]
        molec_dict = molec_dicts[file_num]
        dftb_lst = dftblsts[index] #The dftblsts are enumerated the same length as indices (1:1 correspondence with batches)
        current_feed = total_feed_combinator.create_single_feed(batch_file_ptr, molec_dict, true_index, True)
        single_feed_pass_through(current_feed, all_losses, all_models, par_dict)
        recursive_type_conversion(current_feed, ignore_keys = s.type_conversion_ignore_keys)
        op_dict = assemble_ops_for_charges(current_feed, all_models)
        try:
            update_charges(current_feed, op_dict, dftb_lst, s.opers_to_model)
        except Exception as e:
            print(e)
            glabels = current_feed['glabels']
            basis_sizes = current_feed['basis_sizes']
            result_lst = []
            for bsize in basis_sizes:
                result_lst += list(zip(current_feed['names'][bsize], current_feed['iconfigs'][bsize]))
            print("Charge update failed for")
            print(result_lst)
        update_dQ_rhomask(molec_dict, current_feed) #Update dQ and occ_rho_mask in molecule dictionary
    
    print(f"{tag} charge update done for epoch {epoch}")
    
def training_loop(s: Settings, training_seg: Dict, validation_seg: Dict, all_models: Dict, 
                  model_variables: Dict, losses: Dict, all_losses: Dict, loss_tracker: Dict, par_dict: Dict) -> None:
    r"""Main training loop subroutine
    
    Arguments:
        s (Settings): The settings object containing all the hyperparameter settings
        training_seg (Dict): The dictionary containing information for the 
            training data (see output of transform_seg_to_dict)
        validation_seg (Dict): The dictionary containing the information for the
            validation data (see output of transform_seg_to_dict)
        all_models (Dict): The dictionary containing references to the
            spline models, mapped by model specs
        model_variables (Dict): Dictionary containing references to 
            all the variables that will be optimized by the model. Variables
            are stored as tensors with tracked gradients
        losses (Dict): Dictionary of target losses and their weights
        all_losses (Dict): Dictionary of target losses and their loss classes
        loss_tracker (Dict): Dictionary for keeping track of loss data during
            training. The first list is validation, the second list is training.
        par_dict (Dict): parameter dictionary containing SKF values
    
    Returns: 
        ref_ener_params (List[float]): The current reference energy parameters
        loss_tracker (Dict): The final loss tracker after the training
        all_models (Dict): The dictionary of models after the training session
        model_variables (Dict): The dictionary of model variables after the
            training session
        times_per_epoch (List): A list of the the amount of time taken by each epoch,
            reported in seconds
            
    TODO: Implement me!
    """
    #Instantiate the dftblayer, optimizer, and scheduler
    dftblayer = DFTB_Layer(device = None, dtype = torch.double, eig_method = s.eig_method)
    learning_rate = s.learning_rate
    optimizer = optim.Adam(list(model_variables.values()), lr = learning_rate, amsgrad = s.ams_grad_enabled)
    #TODO: Experiment with alternative learning rate schedulers
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = s.scheduler_factor, 
                                                     patience = s.scheduler_patience, threshold = s.scheduler_threshold)
    
    validation_losses, training_losses = list(), list()
    times_per_epoch = list()
    print("Running charge updates")
    charge_update_subroutine(training_seg, all_models, all_losses, par_dict, "training", s, -1)
    charge_update_subroutine(validation_seg, all_models, all_losses, par_dict, "validating", s, -1)
    print("Moving to true training")
    
    #Unpack the segment for training and validation
    v_batches, v_nperbatch, v_molec_dicts, v_dftblsts, v_indices = validation_seg['file_ptrs'],\
        validation_seg['num_batches'], validation_seg['molec_dicts'], validation_seg['dftblst_lsts'],\
            validation_seg['indices']
            
    t_batches, t_nperbatch, t_molec_dicts, t_dftblsts, t_indices = training_seg['file_ptrs'],\
        training_seg['num_batches'], training_seg['molec_dicts'], training_seg['dftblst_lsts'],\
            training_seg['indices']
    
    nepochs = s.nepochs
    for i in range(nepochs):
        start = time.time()
        
        #Validation routine
        validation_loss = 0
        
        with torch.no_grad(): #No gradient backpropagation in validation
            for index in v_indices:
                file_num, true_index = get_relative_batch_index(index, v_nperbatch)
                batch_file_ptr = v_batches[file_num]
                molec_dict = v_molec_dicts[file_num]
                feed = total_feed_combinator.create_single_feed(batch_file_ptr, molec_dict, true_index, True)
                single_feed_pass_through(feed, all_losses, all_models, par_dict) #Add feed stuff to feed
                recursive_type_conversion(feed, ignore_keys = s.type_conversion_ignore_keys)
                output = dftblayer(feed, all_models)
                loss_for_feed = compute_loss(s, all_losses, losses, loss_tracker, feed, output)
                validation_loss += loss_for_feed.item()
            
        if len(v_indices) > 0:
            #Print validation information
            print("Validation loss:", i, (validation_loss / len(v_indices)))
            validation_losses.append(validation_loss / len(v_indices))
            
            #Update loss tracker
            for loss in all_losses:
                loss_tracker[loss][0].append(loss_tracker[loss][2] / len(v_indices))
                #Reset the loss tracker after being done with all feeds
                loss_tracker[loss][2] = 0
                
            #Shuffle the validation data; here, we are only shuffling the v_indices and 
            v_indices, v_dftblsts = paired_shuffle(v_indices, v_dftblsts)
        
        #Training routine
        epoch_loss = 0
        
        for index in t_indices:
            optimizer.zero_grad()
            file_num, true_index = get_relative_batch_index(index, t_nperbatch)
            batch_file_ptr = t_batches[file_num]
            molec_dict = t_molec_dicts[file_num]
            feed = total_feed_combinator.create_single_feed(batch_file_ptr, molec_dict, true_index, True)
            single_feed_pass_through(feed, all_losses, all_models, par_dict)
            recursive_type_conversion(feed, ignore_keys = s.type_conversion_ignore_keys)
            output = dftblayer(feed, all_models)
            loss_for_feed = compute_loss(s, all_losses, losses, loss_tracker, feed, output)
            epoch_loss += loss_for_feed.item()
            loss_for_feed.backward()
            optimizer.step()
        
        scheduler.step(epoch_loss)
        
        print("Training loss:", i, epoch_loss / len(t_indices))
        training_losses.append(epoch_loss / len(t_indices))
        
        for loss in all_losses:
            loss_tracker[loss][1].append(loss_tracker[loss][2] / len(t_indices))
            loss_tracker[loss][2] = 0
        
        t_indices, t_dftblsts = paired_shuffle(t_indices, t_dftblsts)
        
        #Update charges based on the parameter in the settings file
        if (i % s.charge_update_epochs == 0):
            #Changing the values in training_seg should carry over to the molecule
            # dictionaries through aliasing
            charge_update_subroutine(training_seg, all_models, all_losses, par_dict, "training", s, i)
            charge_update_subroutine(validation_seg, all_models, all_losses, par_dict, "validating", s, i)
        
        times_per_epoch.append(time.time() - start)
        
    print(f"Finished with {s.nepochs} epochs")
    print("Reference energy parameters:")
    reference_energy_params = list(model_variables['Eref'].detach().numpy())
    print(reference_energy_params)
    
    return reference_energy_params, loss_tracker, all_models, model_variables, times_per_epoch

#%% output code
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
    target_folder = os.path.join(s.skf_extension, s.run_id)
    if not os.path.isdir(target_folder):
        os.mkdir(target_folder)
    main(all_models, atom_nums, atom_masses, train_s_block, s.ref_direct, s.skf_strsep, 
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

#%% Driver code
def convert_key_to_num(elem: Dict) -> Dict:
    return {int(k) : v for (k, v) in elem.items()}

def run_method(settings_filename: str, defaults_filename: str) -> None:
    r"""The main method for running the memdriver
    
    Arguments:
        settings_filename (str): The filename for the settings json file
        defaults_filename (str): The filename for the default settings json file
    
    Returns:
        None
    """
    with open(settings_filename, "r") as read_file:
        input_settings_dict = json.load(read_file)
    with open(defaults_filename, "r") as read_file:
        default_settings_dict = json.load(read_file)
    final_settings = construct_final_settings_dict(input_settings_dict, default_settings_dict)
    
    settings = Settings(final_settings)
    par_dict = pull_par_dict(settings)
    
    print(f"run id: {settings.run_id}")
    print(f"CV setting: {settings.driver_mode}")
    
    fold_file_mapping = create_fold_file_mappings(settings)
    all_ptrs = list()
    for fold in fold_file_mapping:
        all_ptrs.append(h5py.File(fold_file_mapping[fold]['batch'], 'r'))
    all_mod_specs = collect_all_models_all_batches(all_ptrs)
    losses, all_losses, loss_tracker = generate_losses_losstracker(settings)
    all_models, model_variables = generate_models_variables(settings, all_mod_specs)
    
    split_mapping = settings.split_mapping #split_mapping must be defined within settings file
    split_mapping = convert_key_to_num(split_mapping)
    
    established_models, established_variables = None, None
    
    for split in split_mapping:
        print(f"Doing work for split {split}")
        training_fold_inds, validation_fold_inds = split_mapping[split]
        t_batch_file_ptrs, t_num_batches, t_molec_dict_lst, t_dftblst_lst, t_starting_indices = assemble_segment(fold_file_mapping, training_fold_inds)
        v_batch_file_ptrs, v_num_batches, v_molec_dict_lst, v_dftblst_lst, v_starting_indices = assemble_segment(fold_file_mapping, validation_fold_inds)
        training_seg = transform_seg_to_dict(t_batch_file_ptrs, t_num_batches, t_molec_dict_lst, t_dftblst_lst, t_starting_indices)
        validation_seg = transform_seg_to_dict(v_batch_file_ptrs, v_num_batches, v_molec_dict_lst, v_dftblst_lst, v_starting_indices)
        
        reference_energy_params, loss_tracker, all_models, model_variables, times_per_epoch = \
            training_loop(settings, training_seg, validation_seg, all_models, model_variables, 
                          losses, all_losses, loss_tracker, par_dict)
        
        write_output_lossinfo(settings, loss_tracker, times_per_epoch, split, split_mapping)
        write_output_skf(settings, all_models)
        
        if (established_models is not None) and (established_variables is not None):
            assert(all_models is established_models)
            assert(model_variables is established_variables)
                
        if split == len(split_mapping.keys()) - 1: #The should write after the final split
            write_output_skf(settings, all_models)
            return reference_energy_params, loss_tracker, all_models, model_variables, times_per_epoch
            
        established_models = all_models
        established_variables = model_variables
        
        assert(all_models is established_models)
        assert(model_variables is established_variables)


#%% Testing
def test_everything():
    top_level_fold_path = os.path.join("fold_molecs_test", "Fold0")
    batch_file = os.path.join(top_level_fold_path, "batches.h5")
    molec_filename = os.path.join(top_level_fold_path, "molecs.h5")
    reference_filename = os.path.join(top_level_fold_path,"reference_data.p")
    dftblist_name = os.path.join(top_level_fold_path, "dftblsts.p")
    
    master_molec_dict = per_molec_h5handler.extract_molec_feeds_h5(molec_filename)
    
    master_molec_dict_mem = sys.getsizeof(master_molec_dict)
    total_mem = 0
    pointer = h5py.File(batch_file, 'r')
    
    all_feeds = list()
    
    for i in range(20):
        
        resulting_feed = total_feed_combinator.create_single_feed(pointer, master_molec_dict, i, True)
        all_feeds.append(resulting_feed)
        total_mem += sys.getsizeof(resulting_feed)
    
    #Run a safety check to make sure each individual fold is being loaded correctly
    compare_feeds(reference_filename, all_feeds)
    
    #Testing the fold_file_map generating function
    with open("settings_default.json", "r") as handle:
        settings_obj = Settings(json.load(handle))
    
    settings_obj.top_level_fold_path = "fold_molecs_test"
    fold_file_mapping = create_fold_file_mappings(settings_obj)
    
    indices = [0, 2, 4, 5]
    batch_file_ptrs, num_batches, molec_dict_lst, dftblst_lst, starting_indices = assemble_segment(fold_file_mapping, indices)
    seg_dict = transform_seg_to_dict(batch_file_ptrs, num_batches, molec_dict_lst, dftblst_lst, starting_indices)
    assert(len(batch_file_ptrs) == len(molec_dict_lst))
    assert(len(dftblst_lst) == len(starting_indices))
    
    #Test single paththrough
    losses, all_losses, loss_tracker = generate_losses_losstracker(settings_obj)
    all_mod_specs = collect_all_models_all_batches([pointer])
    all_models, model_variables = generate_models_variables(settings_obj, all_mod_specs)
    single_feed_pass_through(all_feeds[0], all_losses, all_models, pull_par_dict(settings_obj))
    
    
    
    #Check correspondence 
    for i in range(len(seg_dict['dftblst_lsts'])):
        assert(seg_dict['dftblst_lsts'][i] is dftblst_lst[i])
    
    for j in range(len(dftblst_lst)):
        assert(dftblst_lst[j] is fold_file_mapping[indices[j//20]]['dftblst'][j%20])
    
    print("Initial correspondence passes")
    
    #Check correspondence after shuffling
    test_indexes = [5, 13, 50, 25]
    for test_index in test_indexes:
        seg_dict['indices'], seg_dict['dftblst_lsts'] = paired_shuffle(seg_dict['indices'], seg_dict['dftblst_lsts'])
        test_index_location = seg_dict['indices'].index(test_index) #test_index_location contains the element that was originally at test_index in the unshuffled list
        print(seg_dict['indices'])
        
        assert(seg_dict['dftblst_lsts'][test_index_location] is dftblst_lst[test_index])
        
        for j in range(len(dftblst_lst)):
            assert(dftblst_lst[j] is fold_file_mapping[indices[j//20]]['dftblst'][j%20])
        
        assert(seg_dict['dftblst_lsts'][test_index_location] is fold_file_mapping[indices[test_index // 20]]['dftblst'][test_index % 20])
        
        print("Shuffled correspondence passed")
    
    print("All tests passed!")

#%% Main
if __name__ == '__main__':
    # Execute test cases
    # test_everything() 
    
    # Try runnning through an actual run
    settings_file_name = "settings_default.json"
    defaults_file_name = "defaults.json"
    run_method(settings_file_name, defaults_file_name)
    
    
    
    
    pass
