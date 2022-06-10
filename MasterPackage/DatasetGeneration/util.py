# -*- coding: utf-8 -*-
"""
Created on Sun May 29 14:04:15 2022

@author: fhu14
"""
#%% Imports, definitions
from typing import Dict, List
import random
import os, shutil, pickle
import re

#%% Code behind

def shuffle_dict(dictionary: Dict) -> None:
    r"""Shuffles the molecule list for each empirical formula in the 
        dictionary
    
    Arguments:
        dictionary (Dict): The dictionary to perform the shuffle operation on
    
    Returns:
        None
    
    Notes: Keep in mind that the random.shuffle operation is an in-place operation
        and does not return a new list
    """
    for formula in dictionary:
        random.shuffle(dictionary[formula])
    print("Finished shuffling dictionary")

def save_dset_mols(dset_name: str, destination: str, train_mols: List[Dict], valid_mols: List[Dict],
                   test_mols: List[Dict]) -> None:
    r"""Saves the dataset with the given dset_name to the given destination
    
    Arguments:
        dset_name (str): The name of the dset directory
        destination (str): The location to save the dset directory to
        train_mols (List[Dict]): The training molecules
        valid_mols (List[Dict]): The validation molecules
        test_mols (List[Dict]): The testing molecules
    
    Returns:
        None
    
    Notes: The training molecules are saved as Fold0_molecs.p, the validation
        molecules are saved as Fold1_molecs.p, and the test molecules are 
        saved as test_set.p
    """
    dir_path = os.path.join(os.getcwd(), destination, dset_name)
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
        os.mkdir(dir_path)
    else:
        os.mkdir(dir_path)
        
    train_name = os.path.join(dir_path, "Fold0_molecs.p")
    valid_name = os.path.join(dir_path, "Fold1_molecs.p")
    test_name = os.path.join(dir_path, "test_set.p")
    with open(train_name, 'wb') as handle:
        pickle.dump(train_mols, handle)
    with open(valid_name, 'wb') as handle:
        pickle.dump(valid_mols, handle)
    with open(test_name, 'wb') as handle:
        pickle.dump(test_mols, handle)
    
    print(f"Training, validation, and test sets successfully saved to {dir_path}")

def name_config_nonoverlap(dset_dir: str) -> None:
    r"""Asserts that the training and validation sets do not have 
        repeat molecular configurations. This works mostly between the 
        training and validation sets
    
    Arguments:
        dset_dir (str): The path to the dataset directory to check
    
    Returns:
        None
    
    Notes: Because the training and validation sets are not strictly overlapping 
        with regards to empirical formula, they should at least have distinct
        and non-overlapping configurations of these molecules
    """
    mols0 = pickle.load(open(os.path.join(os.getcwd(), dset_dir, "Fold0_molecs.p"), 'rb'))
    mols1 = pickle.load(open(os.path.join(os.getcwd(), dset_dir, "Fold1_molecs.p"), 'rb'))
    assert(len(mols1) != len(mols0))
    mols0_ncs = [(mol['name'], mol['iconfig']) for mol in mols0]
    mols1_ncs = [(mol['name'], mol['iconfig']) for mol in mols1]
    #Just checking for no duplicated
    assert(len(set(mols0_ncs)) == len(mols0_ncs))
    assert(len(set(mols1_ncs)) == len(mols1_ncs))
    assert(set(mols0_ncs).intersection(set(mols1_ncs)) == set())
    print(f"Configurations are non-overlapping for train/valid in dataset {dset_dir}")

def name_nonoverlap(dset_dir: str) -> None:
    r"""Asserts that there is no overlap in empirical formulas between the 
        test set and the train + validation sets. These should be distinct
    
    Arguments:
        dset_dir (str): The path to the dataset directory to check
    
    Returns:
        None
    
    Notes: The test set should have no empirical formula overlaps with the 
        training + validation sets
    """
    mols0 = pickle.load(open(os.path.join(os.getcwd(), dset_dir, "Fold0_molecs.p"), 'rb'))
    mols1 = pickle.load(open(os.path.join(os.getcwd(), dset_dir, "Fold1_molecs.p"), 'rb'))
    test_mols = pickle.load(open(os.path.join(os.getcwd(), dset_dir, "test_set.p"), 'rb'))
    assert(len(mols1) != len(mols0))
    all_train_valid_mols = mols0 + mols1
    all_train_valid_names = set([mol['name'] for mol in all_train_valid_mols])
    all_test_names = set([mol['name'] for mol in test_mols])
    assert(all_test_names.intersection(all_train_valid_names) == set())
    print(f"Empirical formulas are non-overlapping for train/valid and test in dataset {dset_dir}")

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
        assert((curr_mol1['targets']['dipole'] == curr_mol2['targets']['dipole']).all())
        assert((curr_mol1['targets']['charges'] == curr_mol2['targets']['charges']).all())
        if include_targets: #Really only need to exclude energy
            assert(curr_mol1['targets']['Etot'] == curr_mol2['targets']['Etot'])
            assert((curr_mol1['targets']['dipole'] == curr_mol2['targets']['dipole']).all())
            assert((curr_mol1['targets']['charges'] == curr_mol2['targets']['charges']).all())
    print(f"{set1_path} and {set2_path} are equivalent molecule wise")
    
def test_strict_molecule_nonequivalence(dset_dir: str, len_check: int = None) -> None:
    r"""Checks that the molecules are strictly non-equivalent in terms of 
        coordinates and physical target values
    
    Arguments:
        dset_dir (str): The dataset directory to check to ensure that all
            molecules are strictly different (no duplicates). Relative path 
            here!
        len_check (int): The total number of molecules that should be present.
            Optional parameter to pass in to double-check the number of molecules
    
    Returns:
        None
    
    Raises:
        AssertionError if any condition is violated
    
    Notes: 
        The pair of (name, iconfig) should be enough to ensure uniqueness but
        just in case, this check exists. This has an O(n^2) runtime where
        n is the number of molecules, so best not to use it frequently since it's
        slow
        
        Certain molecules such as N2 have the same charges for each 
        configuration so those are excluded for the charge check
    """
    training = pickle.load(open(os.path.join(dset_dir, "Fold0_molecs.p"), 'rb'))
    validation = pickle.load(open(os.path.join(dset_dir, "Fold1_molecs.p"), 'rb'))
    testing = pickle.load(open(os.path.join(dset_dir, "test_set.p"), 'rb'))
    total_mols = training + validation + testing
    if len_check is not None:
        assert(len(total_mols) == len_check)
    for i in range(len(total_mols)):
        first_mol = total_mols[i]
        for j in range(i + 1, len(total_mols)):
            second_mol = total_mols[j]
            if first_mol['coordinates'].shape == second_mol['coordinates'].shape:
                assert(not (first_mol['coordinates'] == second_mol['coordinates']).all())
            if first_mol['targets']['dipole'].shape == second_mol['targets']['dipole'].shape:
                assert(not (first_mol['targets']['dipole'] == second_mol['targets']['dipole']).all())
            if (first_mol['targets']['charges'].shape == second_mol['targets']['charges'].shape)\
                and ((first_mol['name'], second_mol['name']) != ("N2", "N2")):
                assert(not (first_mol['targets']['charges'] == second_mol['targets']['charges']).all())
            assert(not (first_mol['targets']['Etot'] == second_mol['targets']['Etot']))
    print(f"For {dset_dir}, no two molecules are the same")
    

def target_subdict_correction(molecs: List[Dict], ener_targ: str) -> None:
    r"""Configures the dictionaries in-place to remove one of the energy targets
        while allowing the other to remain.
    
    Arguments:
        molecs (List[Dict]): The list of dictionaries to correct
        ener_targ (str): The energy target to keep 

    Returns:
        None
    
    Notes: There are only two possible values for the ener_targ argument, either
        'cc' or 'wt'
    """
    for mol_dict in molecs:
        if ener_targ == 'cc':
            keep_energy = mol_dict['targets']['cc']
        elif ener_targ == 'wt':
            keep_energy = mol_dict['targets']['wt']
        mol_dict['targets']['Etot'] = keep_energy
        del(mol_dict['targets']['cc'])
        del(mol_dict['targets']['wt'])

def write_dset_metadata():
    r"""Every single dataset should write some kind of metadata to the results
    """
    raise NotImplementedError()

def copy_molecule_set(master_dictionary: Dict, original_set: str, energy_targ: str = 'wt') -> List[Dict]:
    r"""Copies over molecules from one set to another with a different energy 
        target. Uses the given dictionary to find the correct configurations
    
    Arguments:
        master_dictionary (Dict): The dictionary containing information about 
            all of the molecules
        original_set (str): Path to the original molecule set
        energy_targ (str): The energy target to keep. Defaults to 'wt' since
            that is the most common use case
    
    Returns:
        new_molecs (List[Dict]): The copied over version of the molecules with 
            the energy_targ for the 'Etot' key in the 'targets' subdictionary
    
    Notes: This is not the most efficient implementation since it searches
        through each inner list of values for each empirical formula, but 
        it works well enough given the size of the data
    """
    #Set of 'cc' energy target molecules
    original_molecs = pickle.load(open(original_set, 'rb'))
    molecs_ncs = [(mol['name'], mol['iconfig']) for mol in original_molecs]
    assert(len(molecs_ncs) == len(set(molecs_ncs)))
    new_molecs = []
    for (name, iconfig) in molecs_ncs:
        curr_mols = master_dictionary[name]
        for mol in curr_mols:
            if (mol['name'] == name) and (mol['iconfig'] == iconfig):
                new_molecs.append(mol)
    assert(len(new_molecs) == len(original_molecs))
    #Correct these molecules with the correct energy target
    target_subdict_correction(new_molecs, energy_targ)
    return new_molecs

def count_nheavy_empirical_formula(formula: str) -> int:
    r"""Counts the number of heavy atoms (non-hydrogen atoms) based only on
        the empirical formula
    
    Arguments:
        formula (str): The empirical formula given
    
    Returns:
        n_heavy (int): The number of heavy atoms
    """
    pattern = '[A-Z][a-z]?|[0-9]+'
    lst = re.findall(pattern, formula)
    assert(len(lst) % 2 == 0)
    n_heavy = 0
    for i in range(0, len(lst), 2):
        if lst[i].isalpha() and lst[i] != 'H':
            n_heavy += int(lst[i+1])
    return n_heavy