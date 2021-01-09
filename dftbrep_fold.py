import numpy as np
import pandas as pd
import pickle as pkl
from collections.abc import MutableMapping
from dftbrep_consts import *
from h5py import File
from random import Random
from dftbrep_util import get_dataset_type
from typing import Union, List, Optional, Dict, Any, Literal
Array = np.ndarray
import re
from dftb_layer_splines_ani1ccx import get_data_type

atom_syms = {
    1 : 'H',
    6 : 'C',
    7 : 'N',
    8 : 'O'
    }

class Fold(MutableMapping):
    def __init__(self, *args, **kwargs):
        self.__dict__.update(*args, **kwargs)

    def __setitem__(self, mol, conf_arr):
        self.__dict__[mol] = conf_arr

    def __getitem__(self, mol):
        return self.__dict__[mol]

    def __delitem__(self, mol):
        del self.__dict__[mol]

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return f"Fold({self.__dict__})"

    def __add__(self, other):
        new_fold = self.copy()
        new_fold.update(other)
        # check if repeated molecules exists
        rep_mols = set(self.mols()) & set(other.mols())
        if rep_mols:
            # combine the configs of repeated molecules
            rep_confs = {mol: np.array(sorted(set(self[mol]) | set(other[mol]))) for mol in rep_mols}
            new_fold.update(rep_confs)
        return Fold(new_fold)

    def __sub__(self, other):
        new_fold = self.copy()
        for mol, conf_arr in other.items():
            try:
                new_fold[mol] = np.setdiff1d(new_fold[mol], conf_arr)
                if len(new_fold[mol]) == 0:
                    del new_fold[mol]
            except KeyError:
                continue
        return Fold(new_fold)

    def copy(self):
        return Fold(self)

    def mols(self):
        return list(self.keys())

    def mol_confs(self):
        return self.__dict__

    def nmols(self):
        return len(self.__dict__)

    def nconfs(self):
        return sum(list(len(conf_arr) for conf_arr in self.__dict__.values()))

    def shuffle(self, rs_mol=None, rs_conf=None):
        mols = self.mols()
        mol_confs = self.mol_confs().copy()
        if rs_mol is not None:
            Random(rs_mol).shuffle(mols)
        if rs_conf is not None:
            for mol in mols:
                Random(rs_conf).shuffle(mol_confs[mol])
        self.__dict__ = {mol: mol_confs[mol] for mol in mols}

    def split(self, n_splits, show=False):
        folds = list()
        mols = self.mols()
        mols_splits = np.array_split(mols, n_splits)
        for mols_split in mols_splits:
            mol_confs_split = {mol: np.sort(self[mol]) for mol in mols_split}
            folds.append(Fold(mol_confs_split))
        if show:
            print(f"======================= Fold Split =======================")
            for ifold, fold in enumerate(folds):
                print(f"Fold {ifold}: {fold.nmols()} molecules, {fold.nconfs()} configurations")
            print(f"==========================================================")
        return folds

    @classmethod
    def sum(cls, folds):
        new_fold = Fold({})
        for fold in folds:
            new_fold += fold
        return new_fold

    @classmethod
    def from_dataset(cls, dataset):
        return Fold({mol: np.arange(len(data['coordinates'])) for mol, data in dataset.items()})


class FoldCVGenerator:
    def __init__(self, n_splits=5, reverse=False):
        self.reverse = reverse
        self.n_splits = n_splits

    def get_n_splits(self):
        return self.n_splits

    def split(self, fold):
        folds = fold.split(self.n_splits)
        for i_fold, fold in enumerate(folds):
            folds_tmp = folds.copy()
            if self.reverse:
                train_fold = fold
                folds_tmp.pop(i_fold)
                test_fold = Fold.sum(folds_tmp)
            else:
                test_fold = fold
                folds_tmp.pop(i_fold)
                train_fold = Fold.sum(folds_tmp)
            yield train_fold, test_fold


def get_folds_cv(dataset_path, cv, shuffle, reverse):
    dataset_type = get_dataset_type(dataset_path)
    if dataset_type == 'h5':
        with File(dataset_path, 'r') as dataset:
            total_set = Fold.from_dataset(dataset)
    else:
        with open(dataset_path, 'rb') as f:
            dataset = pkl.load(f)
            total_set = Fold.from_dataset(dataset)
    if shuffle:
        total_set.shuffle(shuffle[0], shuffle[1])
    folds_cv = list(FoldCVGenerator(cv, reverse).split(total_set))
    return folds_cv

def get_nheavy_elems(molec_name: str):
    r"""Takes the molecular formula as a string and returns the elements and n_heavy
    
    Arguments:
        molec_name (str): Name of the molecule to analyze
    
    Returns:
        (elems, nheavy) (set, int): The elements in a set and the number of heavy
            elements (non-hydrogen)
    """
    pattern = '[A-Z][a-z]?|[0-9]+'
    elem_lst = re.findall(pattern, molec_name)
    assert(len(elem_lst) % 2 == 0)
    elem_set = set()
    n_heavy = 0
    for i in range(0, len(elem_lst), 2):
        if elem_lst[i].isalpha():
            elem_set.add(elem_lst[i])
            if elem_lst[i] != 'H':
                n_heavy += int(elem_lst[i + 1])
    return (elem_set, n_heavy)

def get_folds_cv_limited(allowed_Zs: List[int], n_heavies: List[int], dataset_path: str,
                         cv: int, max_config: int,  exclude: List[str], shuffle: (int, int) = (None, None), reverse: bool = False) -> List:
    r"""Generates the folds using the fold method in DFTBRepulsive
    
    Arguments:
        allowed_Zs (List[int]): The list of allowed elements
        n_heavies (List[int]): List of allowed number of heavy atoms
        dataset_path (str): Path to the dataset to use
        cv (int): Number of folds to use
        max_config (int): The maximum number of configurations to use for each molecule
        exclude (List[str]): List of chemical formulas to exclude
        shuffle (int, int): Pair of seeds for shuffling. Defaults to (None, None)
        reverse (bool): Whether or not to do reverse cv. Defaults to False.
    
    Returns:
        folds_cv (List): A list of folds
    
    Notes: The difference between this function and get_folds_cv is that this
        function performs a filtering on the total set so only molecules that have
        the allowed elements and the allowed number of heavy atoms are put into the folds
    """
    dataset_type = get_dataset_type(dataset_path)
    if dataset_type == 'h5':
        with File(dataset_path, 'r') as dataset:
            total_set = Fold.from_dataset(dataset)
    else:
        with open(dataset_path, 'rb') as f:
            dataset = pkl.load(f)
            total_set = Fold.from_dataset(dataset)
    #First, filter out all the molecules that do not conform to the 
    # allowed_Zs constraint and the n_heavies constraint
    allowed_Zs_set = set(map(lambda x : atom_syms[x], allowed_Zs))
    bad_molecs = list()
    for molecule in total_set.keys():
        elem_set, n_heavy = get_nheavy_elems(molecule)
        if (not elem_set.issubset(allowed_Zs_set)) or\
            (n_heavy not in n_heavies) or (molecule in exclude):
            bad_molecs.append(molecule)
    for molec in bad_molecs:
        del total_set[molec]
    #total_set contians only molecules that are of the proper constraints. Now need to 
    # Get the maximum number of configs down
    for molecule in total_set:
        total_set[molecule] = total_set[molecule][: max_config]
    if shuffle:
        total_set.shuffle(shuffle[0], shuffle[1])
    folds_cv = list(FoldCVGenerator(cv, reverse).split(total_set))
    return folds_cv

def extract_data_for_molecs(folds, targets: Dict[str, str], dataset_path: str) -> (List[Dict], List[Dict]):
    r"""Takes the molecs and the configuration arrays and pulls out the information from the dataset
    
    Arguments:
        folds (tuple): A tuple of folds representing each fold, (train, test)
        targets (Dict[str, str]): The targets for each molecule, with the alias mapped to the ani identifier.
            For example, 'Etot' : 'cc'
        dataset_path (str): Path to the dataset
    
    Returns:
        training_molecs (List[Dict]): List of molecule dictionaries for the training molecules
        validation_molecs (List[Dict]): List of molecule dictionaries for the validation molecules
    
    Notes: Must go through the dataset and index things out
    """
    training_molecs, validation_molecs = list(), list()
    training_fold, validation_fold = folds
    target_alias, h5keys = zip(*targets.items())
    target_alias, h5keys = list(target_alias), list(h5keys)
    h5keys = get_data_type(h5keys)
    with File(dataset_path, 'r') as dataset:
        for molecule in training_fold:
            for configuration_num in training_fold[molecule]:
                molec_dict = dict()
                molec_dict['name'] = molecule
                molec_dict['iconfig'] = configuration_num
                molec_dict['atomic_numbers'] = dataset[molecule]['atomic_numbers'][()]
                molec_dict['coordinates'] = dataset[molecule]['coordinates'][configuration_num][()]
                molec_dict['targets'] = dict()
                for index, target in enumerate(target_alias):
                    molec_dict['targets'][target] = dataset[molecule][h5keys[index]][configuration_num][()]
                training_molecs.append(molec_dict)
        for molecule in validation_fold:
            for configuration_num in validation_fold[molecule]:
                molec_dict = dict()
                molec_dict['name'] = molecule
                molec_dict['iconfig'] = configuration_num
                molec_dict['atomic_numbers'] = dataset[molecule]['atomic_numbers'][()]
                molec_dict['coordinates'] = dataset[molecule]['coordinates'][configuration_num][()]
                molec_dict['targets'] = dict()
                for index, target in enumerate(target_alias):
                    molec_dict['targets'][target] = dataset[molecule][h5keys[index]][configuration_num][()]
                validation_molecs.append(molec_dict)
        return training_molecs, validation_molecs

def flattened_dataset_from_fold(dataset, fold: Fold, target_types: list):
    # The flattened dataset have columns of:
    # [count of atom_type_1, count of atom_type_2, ..., values of target_type_1, values of target_type_2, ...]
    # Record atom types (Xis) existing in the fold
    Xis = set()
    for mol in fold.mols():
        Xis.update(dataset[mol]['atomic_numbers'][()])
    Xis = sorted(Xis)
    # Flatten given fold
    flatset = list()
    for mol, conf_arr in fold.mol_confs().items():
        atomic_numbers = list(dataset[mol]['atomic_numbers'][()])
        moldata = list()
        # count the number of atoms of each atom type
        for atom in Xis:
            atom_count = np.ones(len(conf_arr))
            atom_count *= atomic_numbers.count(atom)
            atom_count = pd.Series(atom_count, name=str(atom))
            moldata.append(atom_count)
        # slice the values of given target types of selected conformations
        for tt in target_types:
            target = pd.Series(dataset[mol][TARGETS[tt]][conf_arr], name=TARGETS[tt])
            moldata.append(target)
        moldata = pd.concat(moldata, axis=1)
        flatset.append(moldata)
    flatset = pd.concat(flatset, axis=0)
    return flatset


if __name__ == "__main__":
    # # Test Fold.__add__(), Fold.__sub__() and Fold.sum()
    # print(f"============= __add__(), __sub__(), sum() =================")
    # a = Fold({'C10H10': np.array([0, 1, 2]), 'O3': np.array([0, 1, 2])})
    # b = Fold({'C6H6': np.array([0, 1, 2]), 'N2': np.array([0, 1, 2])})
    # c = Fold({'C10H10': np.array([1, 2, 3]), 'O3': np.array([3, 4]), 'N2': np.array([2, 3])})
    # print(f"a = {a}")
    # print(f"b = {b}")
    # print(f"c = {c}")
    # print(f"a + b = {a + b}")
    # print(f"a + c = {a + c}")
    # print(f"b + c = {b + c}")
    # print(f"a - b = {a - b}")
    # print(f"a - c = {a - c}")
    # print(f"b - c = {b - c}")
    # print(f"a + b + c = {Fold.sum([a, b, c])}")

    # Test Fold.from_dataset(), Fold.shuffle and Fold.split
    print(f"================== from_dataset() ========================")
    import os
    if os.getenv("USER") == "yaron":
        ani1_path = 'data/ANI-1ccx_clean_shifted.h5'
        gammas_path = 'data/gammas_50_5_extend.h5'
    elif os.getenv("USER") == "francishe":
        ani1_path = "/home/francishe/Downloads/ANI-1ccx_clean_shifted.h5"
        gammas_path = "/home/francishe/Downloads/gammas_50_5_extend.h5"
    else:
        ani1_path = os.path.join('data', "ANI-1ccx_clean_fullentry.h5")
        gammas_path = "gammas_50_5_extend.h5"

    with File(ani1_path, 'r') as dataset:
        total_set = Fold.from_dataset(dataset)
    print("mols in total_set:")
    print(total_set.mols())

    print(f"===================== shuffle() ==========================")
    total_set.shuffle(1, None)
    print("mols in total_set after shuffling")
    print(total_set.mols())
    print("conf_arr of total_set['C10H10'] after shuffling")
    print(total_set["C10H10"])

    # print(f"======================= split() ==========================")
    # folds = total_set.split(5, show=True)

    print(f"=================== FoldCVGenerator ======================")
    print("Reversed CV")
    fold_cvgen = FoldCVGenerator(n_splits=5, reverse=True)
    for train_fold, test_fold in fold_cvgen.split(total_set):
        print(f"Train: {train_fold.nmols()} mols, {train_fold.nconfs()} confs. "
              f"Test: {test_fold.nmols()} mols, {test_fold.nconfs()} confs.")
    
    allowed_Zs = [1,6,7,8]
    n_heavies = [1,2,3,4,5,6,7,8]
    max_config = 10
    targets = {'Etot' : 'cc',
           'dipole' : 'wb97x_dz.dipole',
           'charges' : 'wb97x_dz.cm5_charges'}
    dataset_path = os.path.join("data", "ANI-1ccx_clean_fullentry.h5")
    #Should have 1134 molecules for n_heavy up to 5 and 4623 molecules for n_heavy up to 8
    res2 = get_folds_cv_limited(allowed_Zs, n_heavies, dataset_path, 5, max_config, ["O3", "N2O1", "H1N1O3", "H2"], shuffle = (1, 1)) 
    tst_fold = res2[0]
    training_molecs, validation_molecs = extract_data_for_molecs(tst_fold, targets, dataset_path)
