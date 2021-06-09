# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 13:17:20 2021

@author: fhu14
"""
#%% Imports, definitions
from typing import Dict
import collections

#%% Code behind

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