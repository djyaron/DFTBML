# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 12:19:10 2021

@author: fhu14

This test is to ensure that the ordering of critical datafields (e.g. atomic numbers)
obeys two conditions:
    1) Invariant to permutations in the input arguments to get_ani1data (or equivalent retrival
       function)
    2) Every molecule dictionary retrieved follows the same order, e.g. C, H, N, O or 6, 1, 7, 8.
       It appears that the ordering is based on electronegativity, but should be verified just in case.
       
This ordering invariant is critical because certain pieces of code (e.g. code for generating
pairwise distance distribution histograms) depends on this assumption, and the results would be 
invalid if this invariant is not true. It seems like everything's fine so far, but need to double
check with this formalized test.
"""
#%% Imports, definitions
from FoldManager import get_ani1data
import numpy as np
import random


#%% Code behind
def test_ani1_Zs_ordering():
    r"""Tests that the ordering of the atomic numbers for the
        ANI-1ccx_clean_fullentry.h5 dataset is fixed for every molecule obtained
        from the h5 file through the ani1 interface utilities
    """
    ordering = [6, 1, 7, 8] #C, H, N, O
    
    allowed_Zs = [1,6,7,8]
    heavy_atoms = [i + 1 for i in range(8)]
    max_config = 8
    target = {"Etot" : "cc",
               "dipole" : "wb97x_dz.dipole",
               "charges" : "wb97x_dz.cm5_charges"}
    
    ani1_path = "ANI-1ccx_clean_fullentry.h5"
    
    all_mols_1 = get_ani1data(allowed_Zs, heavy_atoms, max_config, target, ani1_path)
    
    for mol in all_mols_1:
        atomic_nums = mol['atomic_numbers']
        test_arr = [i for i in range(len(atomic_nums))]
        #Obtain the indices for each element in ordering in the given order
        indices = [np.where(atomic_nums == elem)[0] for elem in ordering]
        combined_indices = np.hstack(tuple(indices))
        assert(len(combined_indices) == len(atomic_nums))
        assert(all([x == y for x, y in zip(combined_indices, test_arr)]))
    
    #Permuting the order of the inputs should not have any effect on the
    #   ordering outcome, i.e. ordering of the atomic numbers should be 
    #   intrinsic to the dataset and consistent across all configurations
    
    random.shuffle(allowed_Zs)
    random.shuffle(heavy_atoms)
    
    all_mols_2 = get_ani1data(allowed_Zs, heavy_atoms, max_config, target, ani1_path)
    
    for mol in all_mols_2:
        atomic_nums = mol['atomic_numbers']
        test_arr = [i for i in range(len(atomic_nums))]
        #Obtain the indices for each element in ordering in the given order
        indices = [np.where(atomic_nums == elem)[0] for elem in ordering]
        combined_indices = np.hstack(tuple(indices))
        assert(len(combined_indices) == len(atomic_nums))
        assert(all([x == y for x, y in zip(combined_indices, test_arr)]))
    
    print("Atomic number ordering test for ANI-1 dataset passed successfully")
    
def run_invariant_tests():
    test_ani1_Zs_ordering()
        

#%% Main block

if __name__ == "__main__":
    run_invariant_tests()