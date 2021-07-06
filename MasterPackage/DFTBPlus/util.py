# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 12:48:03 2021

@author: fhu14

Additional functions for DFTBPlus package
"""

#%% Imports, definitions
import pickle
from functools import reduce
from typing import List, Dict
import os, re


#%% Code behind

def find_all_used_configs(dataset_path: str) -> List[tuple]:
    r"""Goes through the pickled molecules stored at dataset_path
        to find all unique (name, iconfig) pairs
    
    Arguments:
        dataset_path (str): The relative path to the dataset
        
    Returns:
        mols (List[tuple]): A list of (name, iconfig) pairs that encompass
            all unique molecules in the dataset
    """
    pattern = r"Fold[0-9]+_molecs.p"
    valid_names = list(filter(lambda x : re.match(pattern, x), os.listdir(dataset_path)))
    all_molec_lsts = [pickle.load(open( os.path.join(dataset_path, name) , 'rb')) for name in valid_names]
    molecs = list(reduce(lambda x, y : x + y, all_molec_lsts))
    name_conf_pairs = [(molec['name'], molec['iconfig']) for molec in molecs]
    assert(len(name_conf_pairs) == len(molecs))
    assert(len(list(set(name_conf_pairs))) == len(molecs))
    return name_conf_pairs

def filter_dataset(dataset: List[Dict], name_conf_pairs: List[tuple]) -> List[Dict]:
    r"""Takes the used configurations and removes elements of dataset that 
        are found in name_conf_pairs
    
    Arguments:
        dataset (List[Dict]): The list of molecule dictionaries to whittle
            down
        name_conf_pairs (list[tuple]): The list of pairs of names and 
            configuration numbers that should be removed.
    
    Returns:
        cleaned_set (List[Dict]): List of molecule dictionaries where 
            every molecule included in name_conf_pairs is excluded.
    """
    cleaned_dataset = []
    name_conf_pairs_set = set(name_conf_pairs)
    for molecule in dataset:
        if (molecule['name'], molecule['iconfig']) not in name_conf_pairs_set:
            cleaned_dataset.append(molecule)
    return cleaned_dataset