# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 12:48:03 2021

@author: fhu14

Additional functions for DFTBPlus package
"""

#%% Imports, definitions
import pickle
from functools import reduce
from typing import List, Dict, Union
import os, re
import numpy as np
from statistics import mean, stdev

Array = np.ndarray

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

def filter_dataset(dataset: List[Dict], name_conf_pairs: List[tuple], mode: str = "form_conf") -> List[Dict]:
    r"""Takes the used configurations and removes elements of dataset that 
        are found in name_conf_pairs
    
    Arguments:
        dataset (List[Dict]): The list of molecule dictionaries to whittle
            down
        name_conf_pairs (list[tuple]): The list of pairs of names and 
            configuration numbers that should be removed.
        mode (str): The mode to use for molecule exclusion when choosing molecules for
            the test dataset that are not found in the training or validation datasets. 
            Should be one of "form_conf" or "form", where "form" excludes based only on
            empirical formula and "form_conf" excludes based on empirical formula and 
            configuration number. Defaults to "form_conf"
    
    Returns:
        cleaned_set (List[Dict]): List of molecule dictionaries where 
            every molecule included in name_conf_pairs is excluded.
    """
    cleaned_dataset = []
    name_conf_pairs_set = set(name_conf_pairs)
    if mode == "form":
        all_names = [pair[0] for pair in name_conf_pairs]
        name_set = set(all_names)
    for molecule in dataset:
        if mode == "form_conf":
            if (molecule['name'], molecule['iconfig']) not in name_conf_pairs_set:
                cleaned_dataset.append(molecule)
        elif mode == "form":
            if (molecule['name'] not in name_set):
                cleaned_dataset.append(molecule)
    return cleaned_dataset

def sequential_outlier_exclusion(data: List, threshold: Union[int, float] = 20) -> Array:
    r"""Performs sequential outlier exclusion on the data using a threshold 
        value for standard deviations
    
    Arguments:
        data (List): The data to perform the outlier exclusion for
        threshold (Union[int, float]): The number of standard deviations to use
            for outlier exclusion. Defaults to 20 standard deviations.
    
    Returns:
        None
    
    Notes: The sequential outlier exclusion method is as follows:
        1) Compute the mean and standard deviation
        2) All values that are greater than or equal to 20 standard deviations above the mean are removed
        3) A new mean and standard deviation are calculated
        4) Process repeats until the data is left with no values greater than or equal to threshold 
            standard deviations above the mean
    """
    if not isinstance(data, list):
        data = list(data)
    
    while ((max(data) - mean(data)) / stdev(data)) >= threshold:
        data.pop(data.index(max(data)))
    
    print(f"Outlier exclusion finished with threshold of {threshold}")
    return np.array(data)