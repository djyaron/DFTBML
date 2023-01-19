# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 13:17:20 2021

@author: fhu14
"""
#%% Imports, definitions
from typing import Dict, List
import collections
import numpy as np
Array = np.ndarray
import random

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

def init_ref_ener_params(molecs: List[Dict], method1_target: str, method2_target: str, 
                         allowed_Zs: List[int]) -> Array:
    r"""Solves linear least squares to get reference energy parameters
    
    Arguments:
        molecs (List[Dict]): The molecules to use in the least squares fit
        method1_target (str): The first method
        method2_target (str): The second method that the first is being
            corrected to
        allowed_Zs (list[int]): The list of atomic numbers allowed in the
            molecules of the dataset
    
    Returns:
        coefs (Array): The least square coefficients for the reference
            energy.
    
    Notes: 
    
    The reference energy is a correction between two methods, so we could solve for 
    the reference energy parameters by pulling two sets of equivalent data with alternate
    targets for energy and using that difference in energy to solve a least squares 
    equation in the number of atoms of each type in the molecule. This is
    the approach used in dftbplus.py
    
    The 'dt' energy target is what we are emulating since we are doing dftb, so 
    using that as the starting point. Thus, we find Eref as 
    
    E_method2 = E_dftb + Eref => Eref = E_method2 - E_dftb
    
    and Eref = N @ C where N is a matrix of the number of atoms and C is the coefficients
    we are trying to solve for. Then, the equation is
    
    N @ C = E_method2 - E_dftb, and we can solve this in least squares.
    
    For every molecule contained in molecs, the molecule dictionary must have the 
    'targets' key which is accessed by the code.
    """

    iZ = {x : i for i, x in enumerate(allowed_Zs)}

    
    random.shuffle(molecs)
    nmol = len(molecs)
    XX = np.zeros([nmol,len(allowed_Zs)+1])
    method1_mat = np.zeros([nmol])
    method2_mat = np.zeros([nmol])
    
    for imol,mol in enumerate(molecs):
        Zc = collections.Counter(mol['atomic_numbers'])
        for Z,count in Zc.items():
            XX[imol, iZ[Z]] = count
            XX[imol, len(allowed_Zs)] = 1.0
        method1_mat[imol] = mol['targets'][method1_target]
        method2_mat[imol] = mol['targets'][method2_target]
    
    yy = method2_mat - method1_mat
    lsq_res = np.linalg.lstsq(XX, yy, rcond = None)
    coefs = lsq_res[0]
    print(f"Least squares energies are {coefs}")
    return coefs