# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 17:41:58 2021

@author: Frank

Module containing methods for generating and saving the folds in the correct file format.
To accomplish this, each fold is saved in its own directory under a mastery directory:
TotalData:
    Fold0:
        Train_molecs.h5
        Train_batches.h5
        Valid_molecs.h5
        Valid_batches.h5
        Train_dftblsts.p
        Valid_dftblsts.p
        Train_ref.p
        Valid_ref.p
        Train_fold_molecs.p
        Valid_fold_molecs.p
    Fold1:
        ...
    ...

The file names are fixed so that everything can be properly read from the fold 
without any additional work in figuring out the file names. Only the top level directory
containing all the fold subdirectories can be alternately named by the user
"""

from dftb_layer_splines_4 import load_data, saving_data, graph_generation, feed_generation, model_loss_initialization
from dftbrep_fold import get_folds_cv_limited, extract_data_for_molecs
from typing import List, Union, Dict
from batch import DFTBList
import collections
import os, os.path
from h5handler import per_molec_h5handler, per_batch_h5handler, total_feed_combinator, compare_feeds
import pickle

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

def energy_correction(molec: Dict) -> None:
    r"""Performs in-place total energy correction for the given molecule by dividing Etot/nheavy
    
    Arguments:
        molec (Dict): The dictionary in need of correction
    
    Returns:
        None
    """
    zcount = collections.Counter(molec['atomic_numbers'])
    ztypes = list(zcount.keys())
    heavy_counts = [zcount[x] for x in ztypes if x > 1]
    num_heavy = sum(heavy_counts)
    molec['targets']['Etot'] = molec['targets']['Etot'] / num_heavy

def generate_fold_molecs(s: Settings) -> List[(List[Dict], List[Dict])]:
    r"""Generates the molecules in each fold
    
    Arguments:
        s (Settings): Settings object containing all the necessary values for the
            hyperparameters
    
    Returns:
        fold_molecs (List[(List[Dict], List[Dict])]): A list of tuples of molecule dictionary
            lists where the first list is the training molecules and the second list is the 
            validation molecules
    """
    folds_cv = get_folds_cv_limited(s.allowed_Zs, s.heavy_atoms, s.ani1_path, s.num_folds, s.max_config, s.exclude, shuffle = tuple(s.shuffle), 
                                    reverse = False if s.cv_mode == 'normal' else True)
    fold_molecs = list()
    for fold in folds_cv:
        training_molecs, validation_molecs = extract_data_for_molecs(fold, s.target, s.data_path)
        if s.train_ener_per_heavy:
            for molec in training_molecs:
                energy_correction(molec)
            for molec in validation_molecs:
                energy_correction(molec)
        fold_molecs.append((training_molecs, validation_molecs))
    return fold_molecs

def fold_precompute(s: Settings, par_dict: Dict, training_molecs: List[Dict], 
                    validation_molecs: List[Dict]):
    r"""Generates the computational feed dictionaries for each fold
    
    Arguments:
        s (Settings): The settings object containing all the hyperparameter settings
        par_dict (Dict): Dictionary of skf parameters
        training_molecs (List[Dict]): List of molecule dictionaries for the training molecules
        validation_molecs (List[Dict]): List of molecule dictionaries for the validation molecules
        
    
    Returns:
        

    """
    config = {"opers_to_model" : s.opers_to_model}
    training_feeds, training_dftblsts, training_batches = graph_generation(training_molecs, config, s.allowed_Zs, par_dict, s.num_per_batch)
    validation_feeds, validation_dftblsts, validation_batches = graph_generation(validation_molecs, config, s.allowed_Zs, par_dict, s.num_per_batch)
    
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
    
    all_models, model_variables, loss_tracker, all_losses, model_range_dict = model_loss_initialization(training_feeds, validation_feeds,
                                                                               s.allowed_Zs, losses, ref_ener_start = s.reference_energy_starting_point)
    
    feed_generation(training_feeds, training_batches, all_losses, all_models, model_variables, model_range_dict, par_dict, s.spline_mode, s.spline_deg, s.debug, s.loaded_data, 
                    s.num_knots, s.buffer, s.joined_cutoff, s.cutoff_dictionary, s.off_diag_opers, s.include_inflect)
    feed_generation(validation_feeds, validation_batches, all_losses, all_models, model_variables, model_range_dict, par_dict, s.spline_mode, s.spline_deg, s.debug, s.loaded_data, 
                    s.num_knots, s.buffer, s.joined_cutoff, s.cutoff_dictionary, s.off_diag_opers, s.include_inflect)
    print(f"inflect mods: {[mod for mod in model_variables if mod != 'Eref' and mod.oper == 'S' and 'inflect' in mod.orb]}")
    print(f"s_mods: {[mod for mod in model_variables if mod != 'Eref' and mod.oper == 'S']}")
    print(f"len of s_mods: {len([mod for mod in model_variables if mod != 'Eref' and mod.oper == 'S'])}")
    print(f"len of s_mods in all_models: {len([mod for mod in all_models if mod != 'Eref' and mod.oper == 'S'])}")
    print("losses")
    print(losses)
    return training_feeds, validation_feeds, training_dftblsts, validation_dftblsts

def saving_fold(s: Settings, training_feeds: List[Dict], validation_feeds: List[Dict],
                 training_dftblsts: List[DFTBList], validation_dftblsts: List[DFTBList], top_fold_path: str,
                 fold_num: int) -> None:
    r"""Method for saving the feeds using the h5 handler methods
    
    Arguments:
        s (Settings): The Settings object containing all the relevant hyperparameters
        training_feeds (List[Dict]): The list of training feed dictionaries
        validation_feeds (List[Dict]): The list of validation feed dictionaries
        training_dfbtlsts (List[DFTBList]): The list of training DFTBList objects to go along
            with training_feeds
        validation_dftblsts (List[DFTBList]): The list of validation DFTBList objects to go
            along with validation_feeds
        top_fold_path (str): The file path to the top level directory containing the folds
        fold_num (int): The number of the current fold
        
    Returns:
        None
    
    Notes: This method saves the information for the current set of training and validation feeds, which
        together represent the information of one fold
    """
    current_folder_name = f"Fold{fold_num}"
    total_folder_path = os.path.join(top_fold_path, current_folder_name)
    if not os.path.isdir(total_folder_path):
        os.mkdir(total_folder_path)
    train_molec_filename = os.path.join(total_folder_path, 'train_molecs.h5')
    valid_molec_filename = os.path.join(total_folder_path, 'valid_molecs.h5')
    train_batch_filename = os.path.join(total_folder_path, 'train_batches.h5')
    valid_batch_filename = os.path.join(total_folder_path, 'valid_batches.h5')
    train_reference_filename = os.path.join(total_folder_path, 'train_reference.p')
    valid_reference_filename = os.path.join(total_folder_path, 'valid_reference.p')
    train_dftblst_filename = os.path.join(total_folder_path, 'train_dftblsts.p')
    valid_dftblst_filename = os.path.join(total_folder_path, 'valid_dftblsts.p')
    
    #Save the training information
    per_molec_h5handler.save_all_molec_feeds_h5(training_feeds, train_molec_filename)
    per_batch_h5handler.save_multiple_batches_h5(training_feeds, train_batch_filename)
    
    #Save the validation information
    per_molec_h5handler.save_all_molec_feeds_h5(validation_feeds, valid_molec_filename)
    per_batch_h5handler.save_multiple_batches_h5(validation_feeds, valid_batch_filename)
    
    with open(train_reference_filename, 'wb') as handle:
        pickle.dump(training_feeds, handle)
    
    with open(valid_reference_filename, 'wb') as handle:
        pickle.dump(validation_feeds, handle)
    
    with open(train_dftblst_filename, 'wb') as handle:
        pickle.dump(training_dftblsts, handle)
    
    with open(valid_dftblst_filename, 'wb') as handle:
        pickle.dump(validation_dftblsts, handle)
    
    print("All information successfully saved")

def loading_fold(s: Settings, top_fold_path: str, fold_num: int):
    r"""Loads the data from a given fold
    
    Arguments:
        s (Settings): The settings object containing hyperparameter settings
        top_folder_path (str): The path to the top level directory containing all
            the fold subdirectories
        target_fold_num (int): The fold number that should be loaded
    
    Returns:
        training_feeds (List[Dict]): List of feed dictionaries for training
        validation_feeds (List[Dict]): List of feed dictionaries for validation
        training_dftblsts (List[DFTBList]): List of DFTBList objects for training feeds
        validation_dftblsts (List[DFTBList]): List of DFTBList objects for validation feeds
        
    Notes: The check will only be performed if required by the value in s
    """
    current_folder_name = f"Fold{fold_num}"
    total_folder_path = os.path.join(top_fold_path, current_folder_name)
    if not os.path.isdir(total_folder_path):
        raise ValueError("Data for fold does not exist")
    train_molec_filename = os.path.join(total_folder_path, 'train_molecs.h5')
    valid_molec_filename = os.path.join(total_folder_path, 'valid_molecs.h5')
    train_batch_filename = os.path.join(total_folder_path, 'train_batches.h5')
    valid_batch_filename = os.path.join(total_folder_path, 'valid_batches.h5')
    train_reference_filename = os.path.join(total_folder_path, 'train_reference.p')
    valid_reference_filename = os.path.join(total_folder_path, 'valid_reference.p')
    train_dftblst_filename = os.path.join(total_folder_path, 'train_dftblsts.p')
    valid_dftblst_filename = os.path.join(total_folder_path, 'valid_dftblsts.p')
    
    training_feeds = total_feed_combinator.create_all_feeds(train_batch_filename, train_molec_filename, True)
    validation_feeds = total_feed_combinator.create_all_feeds(valid_batch_filename, valid_molec_filename, True)
    
    if s.run_check:
        compare_feeds(train_reference_filename, training_feeds)
        compare_feeds(valid_reference_filename, validation_feeds)
    
    training_dftblsts = pickle.load(open(train_dftblst_filename, "rb"))
    validation_dftblsts = pickle.load(open(valid_dftblst_filename, "rb"))
    
    return training_feeds, validation_feeds, training_dftblsts, validation_dftblsts


    
    
    
    
    
    
    
    
        

