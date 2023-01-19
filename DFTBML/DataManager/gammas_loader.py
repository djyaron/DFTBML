# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 17:34:13 2021

@author: fhu14

Module used to load in gammas that were precomputed per fold.
"""
#%% Imports, definitions
import os, pickle, re
from typing import List

#%% Code behind

def load_elems_per_fold(data_path: str, pattern: str) -> List:
    r"""Generic loading of elements that were saved per fold
    
    Arguments:
        data_path (str): Path to the dataset
        pattern (str): The regular expression pattern used for picking out
            the files of interest in the directory. 
    
    Returns:
        elem_lst (List): A list where the element at index i corresponds to 
        Fold{i}_molecs.p
    
    Notes: This algorithm requires that the files are numbered sequentially,
        going from Fold0... to FoldN...
    """
    all_files = os.listdir(data_path)
    fold_file_names = list(filter(lambda x : re.match(pattern, x), all_files))
    elem_lst = [None] * len(fold_file_names)
    
    for name in fold_file_names:
        fold_num = int(name.split("_")[0][4:])
        full_name = os.path.join(data_path, name)
        with open(full_name, 'rb') as handle:
            elem = pickle.load(handle)
        elem_lst[fold_num] = elem
    
    assert(None not in elem_lst)
    
    return elem_lst

def load_gammas_per_fold(data_path: str) -> List:
    r"""Loads the gammas generated per fold from the dataset stored at data_path
    
    Arguments:
        data_path (str): The path to the dataset.
    
    Returns:
        gam_lst (list): The list of per-fold gammas loaded in.
    """
    pattern = r"Fold[0-9]+_gammas.p"
    gam_lst = load_elems_per_fold(data_path, pattern)
    return gam_lst

def load_config_tracker_per_fold(data_path: str) -> List:
    r"""Loads the configuration trackers generated per fold from the dataset
        stored at data_path
    
    Arguments:
        data_path (str): The path to the dataset
    
    Returns:
        c_track_lst (List): List of configuration trackers, where the index 
            position in the list indicates which fold the configuration tracker
            belongs to.
    """
    pattern = r"Fold[0-9]+_config_tracker.p"
    c_track_lst = load_elems_per_fold(data_path, pattern)
    return c_track_lst

