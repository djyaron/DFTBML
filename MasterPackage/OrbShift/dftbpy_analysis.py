# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 13:33:19 2022

@author: fhu14

Script for running code through DFTBpy and comparing the results. The steps 
of running a DFTBpy calculation are taken from the run_dftbplus.py file in
the DFTPlus package.
"""

#%% Imports, definitions
from DFTBpy import DFTB
import pickle, os
from PlottingUtil import generate_pardict
import numpy as np
from typing import Dict, List
import scipy
Array = np.ndarray


#%% Code behind

def cartesian_conversion(mol: Dict) -> Array:
    r"""Converts the molecular coordinates into the correct input format for
        dftbpy
    
    Arguments:
        mol (Dict): The molecule dictionary
    
    Returns:
        cart (Array): The coordinates for input into DFTBpy.
    
    Notes:
        The conversion is taken from run_dftbplus.py.
    """
    Zs = mol['atomic_numbers']
    rcart = mol['coordinates']
    
    # Input format for python dftb.py
    natom = len(Zs)
    cart = np.zeros([natom,4])
    cart[:,0] = Zs
    for ix in range(3):
        cart[:,ix+1] = rcart[:,ix]
    return cart

def run_dftbpy(pardict: Dict, cart: Array, charge: int = 0, mult: int = 1) -> Dict:
    r"""Runs a dftbpy calculation. Steps are taken from run_dftbplus.py
    
    Arguments:
        pardict (Dict): The parameter dictionary to use
        cart (Array): The array of atomic coordinates specifying the molecule
        charge (int): The charge of the species. Defaults to 0 (neutral species)
        mult (int): Spin multiplicity. Defaults to 1 (singlet multiplicity)
    
    Returns:
        res (Dict): The result dictionary
    
    Notes: There are no finite-temperature effects considered.
    """
    res = dict()
    try:
        dftb_us = DFTB(pardict, cart, charge, mult)
        res['e'],focklist,_ = dftb_us.SCF()
        eorbs, _ = scipy.linalg.eigh(a=focklist[0], b = dftb_us.GetOverlap())
        homo = eorbs[ dftb_us.GetNumElecAB()[1] - 1]
        lumo = eorbs[ dftb_us.GetNumElecAB()[1]]
        res['gap'] = (lumo - homo) * 27.211
        res['r'] = dftb_us.repulsion
        res['t'] = res['e'] + res['r']
        res['conv'] = True
    except Exception:
        res['conv'] = False
    return res

def dftbpy_compare_skf_sets(set1_base: str, set2_shifted: str, test_set: str) -> List[Dict]:
    r"""Runs dftbpy on a given test set of molecules using two different skf sets.
    
    Arguments:
        set1_base (str): The path to the set of molecules with calculations 
            obtained from the non-shifted SKFs
        set2_shifted (str): The path to the set of molecules with calculations
            obtained from the shifted SKFs
        test_set (str): Path to the test set to use. 
    
    Returns:
        test_set (List[Dict]): The list of molecule dictionaries from the 
            test set with the results of the dftbpy calculation added. 
    """
    elements = [1,6,7,8] #Only H, C, N, O
    set1_base_pardict = generate_pardict(set1_base, elements)
    set2_shifted_pardict = generate_pardict(set2_shifted, elements)
    test_set = pickle.load(open(test_set, 'rb'))
    for i, mol in enumerate(test_set):
        print(f"Starting {i} {mol['name']}")
        cart = cartesian_conversion(mol)
        base_result = run_dftbpy(set1_base_pardict, cart)
        shifted_result = run_dftbpy(set2_shifted_pardict, cart)
        mol['dzero_base'] = base_result
        mol['dzero_shifted'] = shifted_result
    return test_set
    