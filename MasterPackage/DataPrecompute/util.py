# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 13:17:20 2021

@author: fhu14
"""
#%% Imports, definitions
from typing import Dict
import collections

#%% Code behind

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
            raise ValueError("Given dictionary cannot go through tuple correction!")
    return new_dict

def count_nheavy(molec: Dict) -> int:
    r"""Counts the number of heavy atoms in a molecule
    
    Arguments:
        molec (Dict): Dictionary representation of a molecule
    
    Returns:
        n_heavy (int): The number of heavy molecules
    """
    n_heavy = 0
    for elem in molec['atomic_numbers']:
        if elem > 1:
            n_heavy += 1
    return n_heavy

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