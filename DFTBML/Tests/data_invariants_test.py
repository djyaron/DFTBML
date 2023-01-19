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

import random
from typing import Dict

import numpy as np
import pytest
from FoldManager import get_ani1data

from .helpers import ani1_path

#%% Code behind


@pytest.fixture
def test_mol_equiv():
    def _test_mol_equiv(mol1: Dict, mol2: Dict) -> bool:
        r"""Checks if two molecules are equivalent across all datafields."""
        assert mol1["name"] == mol2["name"]
        assert mol1["iconfig"] == mol2["iconfig"]
        assert np.allclose(mol1["atomic_numbers"], mol2["atomic_numbers"])
        assert np.allclose(mol1["coordinates"], mol2["coordinates"])

        for t in mol1["targets"]:
            if t == "Etot":
                assert mol1["targets"][t] == mol2["targets"][t]
            else:
                assert np.allclose(mol1["targets"][t], mol2["targets"][t])

        return True

    return _test_mol_equiv


def test_ani1_Zs_ordering():
    r"""Tests that the ordering of the atomic numbers for the
    ANI-1ccx_clean_fullentry.h5 dataset is fixed for every molecule obtained
    from the h5 file through the ani1 interface utilities
    """
    ordering = [6, 1, 7, 8]  # C, H, N, O

    allowed_Zs = [1, 6, 7, 8]
    heavy_atoms = [i + 1 for i in range(8)]
    max_config = 8
    target = {"Etot": "cc", "dipole": "wb97x_dz.dipole", "charges": "wb97x_dz.cm5_charges"}

    all_mols_1 = get_ani1data(allowed_Zs, heavy_atoms, max_config, target, ani1_path)

    for mol in all_mols_1:
        atomic_nums = mol["atomic_numbers"]
        test_arr = [i for i in range(len(atomic_nums))]
        # Obtain the indices for each element in ordering in the given order
        indices = [np.where(atomic_nums == elem)[0] for elem in ordering]
        combined_indices = np.hstack(tuple(indices))
        assert len(combined_indices) == len(atomic_nums)
        assert all([x == y for x, y in zip(combined_indices, test_arr)])

    # Permuting the order of the inputs should not have any effect on the
    #   ordering outcome, i.e. ordering of the atomic numbers should be
    #   intrinsic to the dataset and consistent across all configurations

    random.shuffle(allowed_Zs)
    random.shuffle(heavy_atoms)

    all_mols_2 = get_ani1data(allowed_Zs, heavy_atoms, max_config, target, ani1_path)

    for mol in all_mols_2:
        atomic_nums = mol["atomic_numbers"]
        test_arr = [i for i in range(len(atomic_nums))]
        # Obtain the indices for each element in ordering in the given order
        indices = [np.where(atomic_nums == elem)[0] for elem in ordering]
        combined_indices = np.hstack(tuple(indices))
        assert len(combined_indices) == len(atomic_nums)
        assert all([x == y for x, y in zip(combined_indices, test_arr)])

    print("Atomic number ordering test for ANI-1 dataset passed successfully")


def test_ani1_extraction(test_mol_equiv):
    r"""Testing to make sure that extracting data from ANI1 is a non-random
    process and that changing the max_config number while keeping other
    settings the same ensures that smaller datasets are an ordered
    subset of larger datasets.
    """
    allowed_Zs = [1, 6, 7, 8]
    heavy_atoms = [i + 1 for i in range(8)]
    max_config = 8
    target = {"Etot": "cc", "dipole": "wb97x_dz.dipole", "charges": "wb97x_dz.cm5_charges"}

    small_set = get_ani1data(allowed_Zs, heavy_atoms, max_config, target, ani1_path)

    # Change max_config and redraw
    max_config = 13
    large_set = get_ani1data(allowed_Zs, heavy_atoms, max_config, target, ani1_path)

    # Check the set of names is consistent
    small_names = set([mol["name"] for mol in small_set])
    large_names = set([mol["name"] for mol in large_set])
    assert small_names == large_names

    small_set_dict = dict()
    large_set_dict = dict()

    for mol in small_set:
        name = mol["name"]
        if name in small_set_dict:
            small_set_dict[name].append(mol)
        else:
            small_set_dict[name] = [mol]

    for mol in large_set:
        name = mol["name"]
        if name in large_set_dict:
            large_set_dict[name].append(mol)
        else:
            large_set_dict[name] = [mol]

    # Things should be properly ordered

    assert (
        set(small_set_dict.keys())
        == set(large_set_dict.keys())
        == set(small_names)
        == set(large_names)
    )

    for name in list(small_names):
        min_len = min(len(small_set_dict[name]), len(large_set_dict[name]))
        for i in range(min_len):
            assert test_mol_equiv(small_set_dict[name][i], large_set_dict[name][i])

    print("Ordering test passed")


def run_invariant_tests():
    test_ani1_Zs_ordering()
    test_ani1_extraction()


#%% Main block

if __name__ == "__main__":
    run_invariant_tests()
