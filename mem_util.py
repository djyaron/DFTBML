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
        for all splines to span
    2) 

"""
import pickle, json
import numpy as np
from h5handler import total_feed_combinator, per_molec_h5handler, get_model_from_string, compare_feeds
from dftb_layer_splines_4 import get_model_value_spline_2, OffDiagModel2
from loss_models import TotalEnergyLoss, FormPenaltyLoss, DipoleLoss, ChargeLoss, DipoleLoss2
from typing import List, Union, Dict
from batch import Model
from functools import reduce
from fold_generator import Settings
import importlib
import os, os.path
import sys
import h5py
import re

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
        batch_file_lst (List[h5py.File]): List of h5py file pointers
        
    Returns:
        The list of all the models needed for all batches
    """
    all_mods_lsts = list(map(lambda x : collect_all_models(x), batch_file_lst))
    flattened_all_mods_lst = list(reduce(lambda x, y : x + y, all_mods_lsts))
    return list(set(flattened_all_mods_lst))

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
            all_losses[loss] = FormPenaltyLoss()
            loss_tracker[loss] = [list(), list(), 0]
        elif loss == 'monotonic':
            losses[loss] = s.target_accuracy_monotonic
            all_losses[loss] = FormPenaltyLoss()
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
    low_end_cutoffs = dictionary_tuple_correction(s.low_end_correction_dict)
    model_range_dict = dict()
    for mod_spec in all_model_specs:
        if len(mod_spec.Zs) == 2: #Only two-body splines need to have their ranges known
            Zs, Zs_rev = mod_spec.Zs, (mod_spec.Zs[1], mod_spec.Zs[0])
            if Zs in low_end_cutoffs:
                model_range_dict[mod_spec] = (low_end_cutoffs[Zs], s.univeral_high)
            elif Zs_rev in low_end_cutoffs:
                model_range_dict[mod_spec] = (low_end_cutoffs[Zs_rev], s.universal_high)
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
        molec_dict_lst (List[Dict]): The list of molecule dictionaries 
        dftblst_lst (List[DFTBList]): List of dftblsts to go along with the segment
        starting_indices (List[int]): The list of starting indices for the batches
    
    TODO: Update this docstring
    """
    batch_file_ptrs = list()
    molec_dict_lst = list()
    dftblst_lst = list()
    
    for fold_num in fold_indices:
        
        if isinstance(fold_file_map[fold_num]["molec"], str):
            molec_filename = fold_file_map[fold_num]["molec"]
            fold_file_map[fold_num]["molec"] = per_molec_h5handler.extract_molec_feeds_h5(molec_filename)
            molec_dict_lst.append(fold_file_map[fold_num]["molec"])
        elif isinstance(fold_file_map[fold_num]["molec"], dict):
            molec_dict_lst.append(fold_file_map[fold_num]["molec"])
            
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
    dftblst_lst = list(reduce(lambda x, y : x + y, dftblst_lst))
    
    return batch_file_ptrs, num_batches, molec_dict_lst, dftblst_lst, starting_indices
    

if __name__ == '__main__':
    # Some starting memory tests
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
    
    print("hello")
    pass
