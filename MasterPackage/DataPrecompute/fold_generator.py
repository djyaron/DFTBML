# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 19:46:00 2021

@author: fhu14

Module containing logic used to generate folds from molecules

TODO: Add fancy logic for distributions, randomizations, etc.
"""
#%% Imports, definitions
from typing import List, Dict
from .util import count_nheavy
import random
from .ani1_interface import get_ani1data
import pickle, os

#%% Code behind

def get_folds_from_molecs(num_molecs: int, num_folds_lower: int, num_folds_higher: int, 
                          lower_molecs: List[Dict], higher_molecs: List[Dict]) -> List[List[Dict]]:
    r"""Generates the folds given the number of molecules per fold and the molecules
    
    Arguments:
        num_molecs (int): The number of molecules to have for each fold
        num_folds_lower (int): The number of folds generated from lower_molecs
        num_folds_higher (int): The number of folds generated from higher_molecs
        lower_molecs (List[Dict]): The molecule dictionaries with molecules
            only having up to some nheavy limit
        higher_molecs (List[Dict]): The molecule dictionaries with molecules
            containing more heavy atoms than those in lower_molecs
    
    Returns:
        folds (List[List[Dict]]): The molecules for each fold
    """
    folds = list()
    for i in range(num_folds_lower):
        start, end = i * num_molecs, (i + 1) * num_molecs
        folds.append(lower_molecs[start : end])
    for j in range(num_folds_higher):
        start, end = j * num_molecs, (j + 1) * num_molecs
        folds.append(higher_molecs[start : end])
    return folds

def generate_folds(allowed_Zs: List[int], heavy_atoms: List[int], max_config: int, 
                   target: Dict[str, str], data_path: str, exclude: List[str], lower_limit: int, 
                   num_folds: int, num_folds_lower: int) -> List[List[Dict]]:
    r"""Generates folds based on the number of heavy atoms by dividing up the molecules
    
    Arguments:
        allowed_Zs (List[int]): The allowed elements in the dataset
        heavy_atoms (List[int]): The allowed heavy (non-hydrogen) atoms
        max_config (int): The maximum number of configurations
        target (Dict): Dictionary mapping the target names (e.g. 'Etot') to the 
            ani1 target names (e.g. 'cc')
        data_path (str): The relative path to the dataset from which to pull molecules
        exclude (List[str]): The molecular formulas to exclude when pulling the dataset
        lower_limit (int): The number of heavy atoms to include up to for the folds containing
            lower heavy elements (e.g. folds up to 5)
        num_folds (int): The total number of folds
        num_folds_lower (int): The number of folds that contain molecules with heavy atoms
            only up to lower_limit
    
    Returns:
        fold_molecs (List[List[Dict]]): A list of list of dictionaries where each inner list 
            is a set of molecules for the fold
    
    Notes: This approach does not use the Fold class and instead segments the data
        based on the number of heavy atoms, with num_folds_lower folds containing 
        only molecules with up to lower_limit number of heavy atoms. If done right, this 
        only has to be done once which is why it is not dependent on the settings file.
    """
    #Grab the dataset using get_ani1_data
    assert(num_folds_lower <= num_folds)
    dataset = get_ani1data(allowed_Zs, heavy_atoms, max_config, target, data_path, exclude)
    print(f"Number of molecules: {len(dataset)}")
    heavy_mapped = list(map(lambda x : (x, count_nheavy(x)), dataset))
    lower_molecs = [elem[0] for elem in heavy_mapped if elem[1] <= lower_limit]
    higher_molecs = [elem[0] for elem in heavy_mapped if elem[1] > lower_limit]
    assert(len(lower_molecs) + len(higher_molecs) == len(dataset))
    num_folds_higher = num_folds - num_folds_lower
    random.shuffle(lower_molecs)
    random.shuffle(higher_molecs)
    
    #Figure out the limiting factor between the lower and higher molecules in generating the folds
    num_molec_per_fold_lower = int(len(lower_molecs) / num_folds_lower)
    num_molec_per_fold_higher = int(len(higher_molecs) / num_folds_higher)
    
    lower_criterion = len(higher_molecs) >= num_folds_higher * num_molec_per_fold_lower
    higher_criterion = len(lower_molecs) >= num_folds_lower * num_molec_per_fold_higher
    
    if lower_criterion: #Give precedence to the lower molecules
        num_molecs = num_molec_per_fold_lower
    elif higher_criterion:
        num_molecs = num_molec_per_fold_higher
    folds = get_folds_from_molecs(num_molecs, num_folds_lower, num_folds_higher,
                                      lower_molecs, higher_molecs)
    return folds

def save_folds(folds: List[List[Dict]], dest: str) -> None:
    r"""Saves all the generated folds to the designated destination.
    
    Arguments:
        folds (List[List[Dict]]): A list of list of dictionaries where each inner list 
            is a set of molecules for the fold
        dest (str): The relative or absolute path to the destination directory
    
    Returns:
        None
    
    Notes: The naming conventions for the folds are that they are pickled
        and each fold is saved as "Fold{fold_num}_molecs.p"
    """
    #Check if directory exists and make it if it doesn't exist
    if (not os.path.isdir(dest)):
        os.mkdir(dest)
    
    for i, fold in enumerate(folds):
        full_path = os.path.join(dest, f"Fold{i}_molecs.p")
        with open(full_path, 'wb') as handle:
            pickle.dump(fold, handle)
    print("Fold molecules saved successfully")
        
