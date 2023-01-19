# -*- coding: utf-8 -*-
"""
Created on Fri May 20 13:33:21 2022

@author: fhu14
"""

'''
Some generic utility functions for the datasets generated
'''
#%% Imports, definitions
import os, pickle
from typing import List, Dict
import numpy as np
import re
from functools import reduce

#%% Code behind

def test_strict_molecule_set_equivalence(set1_path: str, set2_path: str, include_targets: bool = True) -> None:
    r"""Tests if two molecule sets are equivalent. A molecular configuration is
        defined using a dictionary for each configuration
    
    Arguments:
        set1_path (str): The path to the first set
        set2_path (str): The path to the second set
        include_targets (bool): Whether to include the
            targets in the equivalence check
    
    Returns:
        None
    
    Raises:
        AssertionError: If any of the conditions of equality are violated
    
    Notes: Each molecule configuration dictionary contains the following keys:
        'name' (str) : empirical formula of the molecule
        'iconfig' (int) : the configuration number of the molecule
        'atomic_numbers' (array) : the atomic numbers of the elements in that molecule
        'coordinates' (array): The (N, 3) matrix of coordinates for that molecule
        'targets' (dict): Dictionary with the following keys:
                'Etot'(float): the total molecular energy
                'dipole' (array): the molecular dipole
                'charges' (array): array of atomic charge partitions
        The length of the molecule list read in from set1_path and set2_path, 
        respectively, should have the same length. Here, order matters
    """
    set1_mols = pickle.load(open(set1_path, 'rb'))
    set2_mols = pickle.load(open(set2_path, 'rb'))
    #The molecule lists should be the same length
    assert(len(set1_mols) == len(set2_mols))
    for i in range(len(set1_mols)):
        curr_mol1, curr_mol2 = set1_mols[i], set2_mols[i]
        assert(curr_mol1['name'] == curr_mol2['name'])
        assert(curr_mol1['iconfig'] == curr_mol2['iconfig'])
        #Test for equivalence taken from 
        #   https://stackoverflow.com/questions/10580676/comparing-two-numpy-arrays-for-equality-element-wise
        assert((curr_mol1['atomic_numbers'] == curr_mol2['atomic_numbers']).all())
        assert((curr_mol1['coordinates'] == curr_mol2['coordinates']).all())
        if include_targets:
            assert(curr_mol1['targets']['Etot'] == curr_mol2['targets']['Etot'])
            assert((curr_mol1['targets']['dipole'] == curr_mol2['targets']['dipole']).all())
            assert((curr_mol1['targets']['charges'] == curr_mol2['targets']['charges']).all())
    print(f"{set1_path} and {set2_path} are equivalent molecule wise")

def test_molecule_name_configuration_equivalence(set1_path: str, set2_path: str) -> bool:
    r"""Tests that two molecule sets have the same empirical formulas and 
        configuration numbers
    
    Arguments:
        set1_path (str): The path to the first set
        set2_path (str): The path to the second set
    
    Returns:
        (bool): Whether the two molecule sets have the same empirical formulas and
            configurations. True for the same configurations and names, False otherwise
    
    Notes: To test whether two sets have the same elements, the two differences 
        A - B and B - A should both yield the empty set for the sets to be equal.
    """
    set1_mols = pickle.load(open(set1_path, 'rb'))
    set2_mols = pickle.load(open(set2_path, 'rb'))
    set1_names_configs = [(mol['name'], mol['iconfig']) for mol in set1_mols]
    set2_names_configs = [(mol['name'], mol['iconfig']) for mol in set2_mols]
    set1_names_configs = set(set1_names_configs)
    set2_names_configs = set(set2_names_configs)
    first_empty = set1_names_configs.difference(set2_names_configs) == set()
    second_empty = set2_names_configs.difference(set1_names_configs) == set()
    return first_empty and second_empty

def test_molecule_set_equivalence_unordered(dset_1_name: str, dset_2_name: str) -> None:
    r"""Tests the equivalence between two datasets regardless of order
    
    Arguments:
        dset_1_path (str): The path to the first set
        dset_2_path (str): The path to the second set
    
    Returns:
        None
    
    Raises:
        AssertionError: If any of the conditions required for equivalence are violated
    """
    dset1 = pickle.load(open(dset_1_name, 'rb'))
    dset2 = pickle.load(open(dset_2_name, 'rb'))

    assert(len(dset1) == len(dset2))

    name_confs_1 = [(mol['name'], mol['iconfig']) for mol in dset1]
    name_confs_2 = [(mol['name'], mol['iconfig']) for mol in dset2]

    assert(len(name_confs_1) == len(set(name_confs_1)))
    assert(len(name_confs_2) == len(set(name_confs_2)))

    mol_dict_1 = {(mol['name'], mol['iconfig']) : mol for mol in dset1}

    for mol in dset2:
        mol_1 = mol_dict_1[(mol['name'], mol['iconfig'])]
        assert(mol['name'] == mol_1['name'])
        assert(mol['iconfig'] == mol_1['iconfig'])
        assert(all(mol['atomic_numbers'] == mol_1['atomic_numbers']))
        assert(np.allclose(mol['coordinates'], mol_1['coordinates']))

    #No target equivalence
    print(f"Dataset reference equivalence check passed between {dset_1_name} and {dset_2_name}")

def extract_all_train_valid_forms(dset_dir: str) -> List[str]:
    r"""Extracts all the empirical formulas from the training and validation molecules
        stored in Fold0_molecs.p and onward
        
    Arguments:
        dset_dir (str): the path to the dataset directory

    Returns:
        forms (List[str]): the list of empirical formulas
    """
    pattern = r"Fold[0-9]+_molecs.p"
    valid_names = list(filter(lambda x : re.match(pattern, x), os.listdir(dset_dir)))
    all_mols = [pickle.load(open(os.path.join(dset_dir, name), 'rb')) for name in valid_names]
    all_mols = list(reduce(lambda x, y : x + y, all_mols))
    forms = [mol['name'] for mol in all_mols]
    return forms

def non_overlap_with_test(dset_dir: str) -> None:
    r"""Checks that the the empirical formulas for the training and validation sets
        do not overlap with the test set
    
    Arguments:
        dset_dir (str): path to the directory to check
    
    Returns:
        None
    
    Raises: 
        AssertionError: if there is an overlap detected beween the training
            and validation sets and the test set empirical formulas
    """
    test_set = pickle.load(open(os.path.join(dset_dir, 'test_set.p'), 'rb'))
    train_valid_forms = set(extract_all_train_valid_forms(dset_dir))
    test_forms = set([mol['name'] for mol in test_set])
    assert(train_valid_forms.intersection(test_forms) == set())
    print(f"No overlap between training/validation formulas and test formulas in dataset {dset_dir}")

def check_dset_inheritance(parent_dset_dir: str, child_dset_dir: str, criterion: str,
                           check_test_set_equivalence: bool = True) -> None:
    r"""Checks to ensure that the child dataset and parent dataset correctly inherited
        using a specific criterion
    
    Arguments:
        parent_dset_dir (str): The path to the parent dataset
        child_dset_dir (str): The path to the child dataset
        criterion (str): The criterion to use for checking inheritance. The options are:
            1) same_emp_forms: checks that the empirical formulas are consistent for the 
                training and validation sets for each (e.g., makes sure that expanded datasets
                are correctly built from their reference datasets)
            2) molec_equivalence: checks if the molecules are equivalent (notwithstanding targets). 
                This is useful for checking if datasets of alternate targets are 
                correctly inherited, i.e. 'wt' target dsets are correctly inherited from 
                their 'cc' counterparts
        check_test_set_equivalence (bool): Whether or not to check if the test sets
            have strict equivalence. Defaults to True, in which case the 
            test is performed
        
    Raises:
        AssertionError: If any of the conditions for correct inheritance are violated, 
            an assertion error is raised

    Returns:
        None
    """
    #First, the test sets are always inherited so they should be equivalent
    if check_test_set_equivalence:
        parent_test_set_path = os.path.join(parent_dset_dir, 'test_set.p')
        child_test_set_path = os.path.join(child_dset_dir, 'test_set.p')
        test_strict_molecule_set_equivalence(parent_test_set_path, child_test_set_path,
                                             include_targets = True)
    
    #behavior changes depending on the criterion
    assert(criterion in ['same_emp_forms', 'molec_equivalence'])
    #case of same empirical formulas for the training and validation sets
    if criterion == 'same_emp_forms':
        parent_train_valid_forms = extract_all_train_valid_forms(parent_dset_dir)
        child_train_valid_forms = extract_all_train_valid_forms(child_dset_dir)
        assert(set(parent_train_valid_forms) == set(child_train_valid_forms))
        print(f"Successfuly inheritance between {parent_dset_dir} and {child_dset_dir} with criterion {criterion}")
    elif criterion == 'molec_equivalence':
        pattern = r"Fold[0-9]+_molecs.p"
        parent_valid_names = list(filter(lambda x : re.match(pattern, x), os.listdir(parent_dset_dir)))
        child_valid_names = list(filter(lambda x : re.match(pattern, x), os.listdir(child_dset_dir)))
        assert(set(parent_valid_names) == set(child_valid_names))
        for name in parent_valid_names:
            parent_path = os.path.join(parent_dset_dir, name)
            child_path = os.path.join(child_dset_dir, name)
            test_strict_molecule_set_equivalence(parent_path, child_path, include_targets = False)
        print(f"Successfuly inheritance between {parent_dset_dir} and {child_dset_dir} with criterion {criterion}")
    #Also check that there is no overlap in the empirical formulas between the training and validation molecules
    #   and the test set
    non_overlap_with_test(parent_dset_dir)
    print(f"{parent_dset_dir} no overlap between train/valid and test")
    non_overlap_with_test(child_dset_dir)
    print(f"{child_dset_dir} no overlap between train/valid and test")
    