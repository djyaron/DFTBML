# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 16:22:49 2021

@author: fhu14

Contains methods for loading feeds in for training
"""
#%% Imports, definitions
import os
from .h5handler import total_feed_combinator, compare_feeds
import pickle
from typing import Dict

#%% Code behind

def load_single_fold(s, top_level_fold_path: str, fold_num: int):
    r"""Loads a single fold using the new fold format based on heavy atoms
    
    Arguments:
        s (Settings): Settings object containing values for hyperparameters
        top_level_fold_path (str): The relative path to the directory containing all the folds
        fold_num (int): The fold number to load
    
    Returns:
        feeds (List[Dict]): The input feeds for the DFTB layer
        dftb_lsts (List[DFTBList]): The DFTBList objects to use for the training
        batches (List[List[Dict]]): The molecules originally used to generate the 
            feeds; the ith batch matches the ith feed.
    
    Notes: The s.run_check flag indicates if a safety check is conducted on the
        loaded data. DFTBML saves unprocessed versions of the training data, and the 
        safety check consists of checking the loaded data against the unprocessed
        data to ensure that the total_feed_combinator object is reconstituting the
        saved data correctly. For efficiency sake, this check is neglected and 
        was primarily a development tool.
    """
    total_fold_path = os.path.join(top_level_fold_path, f"Fold{fold_num}")
    print(f"Loading from {total_fold_path}")
    batch_info_name = os.path.join(total_fold_path, 'batches.h5')
    molec_info_name = os.path.join(total_fold_path, 'molecs.h5')
    dftb_lst_name = os.path.join(total_fold_path, 'dftblsts.p')
    reference_data_name = os.path.join(total_fold_path, 'reference_data.p')
    batch_original_name = os.path.join(total_fold_path, 'batch_original.p')
    
    feeds = total_feed_combinator.create_all_feeds(batch_info_name, molec_info_name, s.ragged_dipole)
    dftb_lsts = pickle.load(open(dftb_lst_name, 'rb'))
    batches = pickle.load(open(batch_original_name, 'rb'))
    
    if s.run_check:
        print("Running safety check")
        compare_feeds(reference_data_name, feeds)
    
    return feeds, dftb_lsts, batches

def load_combined_fold(s, top_level_fold_path: str, split_num: int, fold_mapping: Dict):
    r"""Generates the training and validation feeds through combining individual folds
    
    Arguments:
        s (Settings): The Settings object containing all the hyperparameters
        top_level_fold_path (str): The relative path to the directory containing the 
            individual folds.
        fold_num (int): The fold number used to index into fold_mapping
        fold_mapping (Dict): The dictionary mapping fold_nums to the numbers for
            the training data and the validation data, stored as 2D list with the 
            first list containing training fold numbers and the second list containing
            the validation fold numbers
    
    Returns:
        training_feeds (List[Dict]): List of training feed dictionaries
        validation_feeds (List[Dict]): List of validation feed dictionaries
        training_dftblsts (List[DFTBList]): List of DFTBList objects for training
        validation_dftblsts (List[DFTBList]): List of DFTBList objects for validation
        training_batches (list[List[Dict]]): The batches for the training feeds
        validation_batches (list[List[Dict]]): The batches for the validation feeds
    
    Notes: Here, the fold_num just indicates which split we're doing. The fold_mapping
        maps the split number to the numbers of the individual segments of data that
        needs to be combined for training and validation.
        
    """
    current_train_folds, current_valid_folds = fold_mapping[split_num]
    #Now we need to load the data for each fold number. Load the training folds first
    training_feeds, training_dftblsts, training_batches = list(), list(), list()
    validation_feeds, validation_dftblsts, validation_batches = list(), list(), list()
    
    #Get the training information
    for num in current_train_folds:
        feeds, dftblsts, batches = load_single_fold(s, top_level_fold_path, num)
        training_feeds.extend(feeds)
        training_dftblsts.extend(dftblsts)
        training_batches.extend(batches)
    
    #Get the validation information
    for num in current_valid_folds:
        feeds, dftblsts, batches = load_single_fold(s, top_level_fold_path, num)
        validation_feeds.extend(feeds)
        validation_dftblsts.extend(dftblsts)
        validation_batches.extend(batches)
    
    return training_feeds, validation_feeds, training_dftblsts, validation_dftblsts,\
        training_batches, validation_batches

def loading_fold(s, top_fold_path: str, fold_num: int):
    r"""Loads the data from a single given fold
    
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
        
    Notes: The check will only be performed if required by the value in s. This is 
        not typically used but exists if one wishes to examine the data for a 
        single fold.
    """
    current_folder_name = f"Fold{fold_num}"
    total_folder_path = os.path.join(top_fold_path, current_folder_name)
    if not os.path.isdir(total_folder_path):
        raise ValueError("Data for fold does not exist")
    print(f"loading data from {total_folder_path}")
    train_molec_filename = os.path.join(total_folder_path, 'train_molecs.h5')
    valid_molec_filename = os.path.join(total_folder_path, 'valid_molecs.h5')
    train_batch_filename = os.path.join(total_folder_path, 'train_batches.h5')
    valid_batch_filename = os.path.join(total_folder_path, 'valid_batches.h5')
    train_reference_filename = os.path.join(total_folder_path, 'train_reference.p')
    valid_reference_filename = os.path.join(total_folder_path, 'valid_reference.p')
    train_dftblst_filename = os.path.join(total_folder_path, 'train_dftblsts.p')
    valid_dftblst_filename = os.path.join(total_folder_path, 'valid_dftblsts.p')
    
    training_feeds = total_feed_combinator.create_all_feeds(train_batch_filename, train_molec_filename, s.ragged_dipole)
    validation_feeds = total_feed_combinator.create_all_feeds(valid_batch_filename, valid_molec_filename, s.ragged_dipole)
    
    if s.run_check:
        print("Running safety check")
        compare_feeds(train_reference_filename, training_feeds)
        compare_feeds(valid_reference_filename, validation_feeds)
    
    training_dftblsts = pickle.load(open(train_dftblst_filename, "rb"))
    validation_dftblsts = pickle.load(open(valid_dftblst_filename, "rb"))
    
    return training_feeds, validation_feeds, training_dftblsts, validation_dftblsts