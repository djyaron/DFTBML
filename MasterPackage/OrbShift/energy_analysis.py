# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 16:27:04 2022

@author: fhu14

Quick functions for energy analysis
"""
#%% Imports, definitions
import os, pickle
from typing import List, Dict
from DatasetGeneration import test_strict_molecule_set_equivalence



#%% Code behind

def determine_nelectrons(mol: Dict) -> int:
    r"""Determines the number of bonding electrons in the molecule 
        based on the atomic numbers (atom composition)
    
    Arguments:
        mol (Dict): The molecule dictionary
    
    Returns:
        n_electron (int): The number of electrons
    
    Notes: In DFTB/DFTB+, the number of electrons in a molecular system is 
        equal to the sum of all the valence electrons across the 
        constituent atoms. This is convenient because it is just the sum of the 
        atomic numbers. For heavy atoms, the number of valence electrons is
        the atomic number minus 2 (2 electrons for the 1s orbital)
    """
    atom_nums = mol['atomic_numbers']
    n_electron = 0
    for num in atom_nums:
        if num > 1:
            n_electron += (num - 2)
        else:
            n_electron += num
    return n_electron

def analyze_ener_diffs(mols_base: str, mols_shifted: str) -> List[float]:
    r"""Analyzes the energy difference between a set of molecules computed
        using a non-shifted and shifted version of the skf files.
    
    Arguments:
        mols_base (str): The path to the set of molecules computed using the 
            non-shifted skf files
        mols_shifted (str): The path to the set of molecules computed using the 
            shifted version of the skf files
        
    Returns:
        electron_corrected_diffs (List[flaot]): The list of energy differences
            corrected by the number of electrons
    
    Notes:
        Theoretically, the difference between the shifted and non-shifted 
        energies for each molecule should be equal to the number of electrons
        multiplied by some constant where the constant is the initial value
        used to shift the skf files.
    """
    #Begin with a molecule set test including targets
    test_strict_molecule_set_equivalence(mols_base, mols_shifted, True)
    with open(mols_base, 'rb') as handle:
        base_mols = pickle.load(handle)
    with open(mols_shifted, 'rb') as handle:
        shifted_mols = pickle.load(handle)
    assert(len(base_mols) == len(shifted_mols))
    electron_corrected_diffs = []
    for i in range(len(base_mols)):
        current_base_mol = base_mols[i]
        current_shifted_mol = shifted_mols[i]
        num_electrons_base = determine_nelectrons(current_base_mol)
        num_electrons_shifted = determine_nelectrons(current_shifted_mol)
        assert(num_electrons_base == num_electrons_shifted)
        #Switch from total to electronic only
        ener_difference = current_shifted_mol['pzero']['t'] - current_base_mol['pzero']['t']
        corrected_ediff = ener_difference / num_electrons_base
        electron_corrected_diffs.append(corrected_ediff)
    return electron_corrected_diffs
    

