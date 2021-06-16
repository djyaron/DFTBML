# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 18:34:07 2021

@author: fhu14
"""
#%% Imports, definitions
from typing import List, Dict
import random
from DFTBLayer import DFTBList, assemble_ops_for_charges, update_charges
import os, pickle
from SKF import main
from MasterConstants import atom_nums, atom_masses

#%% Code behind

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

def paired_shuffle_triple(lst_1: List, lst_2: List, lst_3: List) -> (list, list, list):
    r"""Same as paired_shuffle but for three lists
    """
    temp = list(zip(lst_1, lst_2, lst_3))
    random.shuffle(temp)
    lst_1, lst_2, lst_3 = zip(*temp)
    lst_1, lst_2, lst_3 = list(lst_1), list(lst_2), list(lst_3)
    return lst_1, lst_2, lst_3

def charge_update_subroutine(s, training_feeds: List[Dict], 
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
        op_dict = assemble_ops_for_charges(feed, all_models, s.tensor_device, s.tensor_dtype)
        try:
            update_charges(feed, op_dict, dftb_list, s.tensor_device, s.tensor_dtype, s.opers_to_model)
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
        op_dict = assemble_ops_for_charges(feed, all_models, s.tensor_device, s.tensor_dtype)
        try:
            update_charges(feed, op_dict, dftb_list, s.tensor_device, s.tensor_dtype, s.opers_to_model)
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
        
def write_output_skf(s, all_models: Dict) -> None:
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

def write_output_lossinfo(s, loss_tracker: Dict, times_per_epoch: List[float], split_num: int,
                          split_mapping: Dict, all_models: Dict) -> None:
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
        all_models (Dict): Dictionary of the trained models to be saved at the
            end of training
            
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
    with open(os.path.join(target_dir, "saved_models.p"), "wb") as handle:
        pickle.dump(all_models, handle)
    
    print("All loss information saved")