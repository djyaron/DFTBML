from __future__ import annotations

import copy
from collections.abc import MutableMapping
from random import Random
from typing import ItemsView, Iterator, List, Union

import h5py
import numpy as np


class Fold(MutableMapping):
    def __init__(self, *args, **kwargs) -> None:
        r"""Ordered-dictionary-like interface to molecule names and conformation indices in ANI-like datasets.

        For a fold.Fold object, its keys are expected to be strings of molecular
        formulas, whereas its values are expected to be 1-D np.ndarray, which are
        the indices of the molecule specified by the keys. Type checking is not
        built-in for efficiency and flexibility.

        Examples:
            >>> from fold import Fold
            >>> mol_confs = {"CH4": np.array([0, 1, 2]), "NH3": np.array([0, 3])}
            >>> a = Fold(mol_confs)
            >>> print(a)
            Fold({'CH4': array([0, 1, 2]), 'NH3': array([0, 3])})
            >>> for mol, conf_arr in a:
            ...     print(mol, conf_arr)
            CH4 [0, 1, 2]
            NH3 [0, 3]
            >>> a_copy = Fold(a)
            >>> print(a_copy)
            Fold({'CH4': array([0, 1, 2]), 'NH3': array([0, 3])})
        """
        self.__dict__.update(*args, **kwargs)

    def __setitem__(self, mol: str, conf_arr: np.ndarray) -> None:
        self.__dict__[mol] = conf_arr

    def __getitem__(self, mol: str) -> np.ndarray:
        return self.__dict__[mol]

    def __delitem__(self, mol: str) -> None:
        del self.__dict__[mol]

    def __iter__(self) -> Iterator:
        return iter(self.__dict__)

    def __len__(self) -> int:
        return len(self.__dict__)

    def __add__(self, other) -> Fold:
        r"""Addition between Fold objects

        Examples:
            >>> from fold import Fold
            >>> a = Fold({'C10H10': np.array([0, 1, 2]), 'O3': np.array([0, 1, 2])})
            >>> b = Fold({'C6H6': np.array([0, 1, 2]), 'N2': np.array([0, 1, 2])})
            >>> c = Fold({'C10H10': np.array([1, 2, 3]), 'O3': np.array([3, 4]), \
            ...           'N2': np.array([2, 3])})
            >>> print(a + b)
            Fold({'C10H10': array([0, 1, 2]), 'O3': array([0, 1, 2]), 'C6H6': array([0, 1, 2]), 'N2': array([0, 1, 2])})
            >>> print(a + c)
            Fold({'C10H10': array([0, 1, 2, 3]), 'O3': array([0, 1, 2, 3, 4]), 'N2': array([2, 3])})
            >>> print(b + c)
            Fold({'C6H6': array([0, 1, 2]), 'N2': array([0, 1, 2, 3]), 'C10H10': array([1, 2, 3]), 'O3': array([3, 4])})
            >>> print(a + b + c)
            Fold({'C10H10': array([0, 1, 2, 3]), 'O3': array([0, 1, 2, 3, 4]), 'C6H6': array([0, 1, 2]), 'N2': array([0, 1, 2, 3])})
        """
        new_fold = self.copy()
        new_fold.update(other)
        # check if repeated molecules exists
        rep_mols = set(self.mols()) & set(other.mols())
        if rep_mols:
            # combine the configs of repeated molecules
            rep_confs = {mol: np.array(sorted(set(self[mol]) | set(other[mol]))) for mol in rep_mols}
            new_fold.update(rep_confs)
        return Fold(new_fold)

    def __sub__(self, other) -> Fold:
        r"""Subtract a Fold from another

        Examples:
            >>> from fold import Fold
            >>> a = Fold({'C10H10': np.array([0, 1, 2]), 'O3': np.array([0, 1, 2])})
            >>> b = Fold({'C6H6': np.array([0, 1, 2]), 'N2': np.array([0, 1, 2])})
            >>> c = Fold({'C10H10': np.array([1, 2, 3]), 'O3': np.array([3, 4]), \
            ...           'N2': np.array([2, 3])})
            >>> print(a - b)
            Fold({'C10H10': array([0, 1, 2]), 'O3': array([0, 1, 2])})
            >>> print(a - c)
            Fold({'C10H10': array([0]), 'O3': array([0, 1, 2])})
            >>> print(b - c)
            Fold({'C6H6': array([0, 1, 2]), 'N2': array([0, 1])})
            >>> print(a - a)
            Fold({})
            >>> print(a - a - b)
            Fold({})
        """
        new_fold = self.copy()
        for mol, conf_arr in other.items():
            try:
                new_fold[mol] = np.setdiff1d(new_fold[mol], conf_arr)
                if len(new_fold[mol]) == 0:
                    del new_fold[mol]
            except KeyError:
                continue
        return Fold(new_fold)

    def copy(self) -> Fold:
        return Fold(self.__dict__)

    def deepcopy(self) -> Fold:
        return Fold(copy.deepcopy(self.__dict__))

    def confs(self) -> List[np.ndarray]:
        r"""List the conformation indices in current Fold

        Returns:
            List[np.ndarray]
        """
        return list(self.__dict__.values())

    def mols(self) -> List[str]:
        r"""List the molecular formulas in current Fold

        Returns:
              List[str]
        """
        return list(self.__dict__.keys())

    def mol_confs(self) -> ItemsView:
        r"""View the items in current Fold, same as dict.items() method

        Returns:
            ItemsView
        """
        return self.__dict__.items()

    def nmols(self) -> int:
        r"""Count the number of molecular formulas

        Returns:
            int
        """
        return self.__len__()

    def nconfs(self) -> int:
        r"""Count the number of conformations

        Returns:
            int
        """
        return sum(list(len(conf_arr) for conf_arr in self.confs()))

    def shuffle(self, rs_mol: Union[int, None] = False,
                rs_conf: Union[int, None] = False) -> None:
        r""" Shuffle the molecular formulas and conformation indices in place

        Args:
            rs_mol (Union[int, None]): Random state for shuffling molecular formulas,
                set to False to disable shuffling
            rs_conf (Union[int, None]): Random state for shuffling conformation indices,
                set to False to disable shuffling

        Returns:
            None

        Examples:
            >>> from fold import Fold
            >>> a = Fold({'C10H10': np.arange(3), 'O3': np.arange(4), 'N2': np.arange(5)})
            >>> print(a)
            Fold({'C10H10': array([0, 1, 2]), 'O3': array([0, 1, 2, 3]), 'N2': array([0, 1, 2, 3, 4])})
            >>> b = a.deepcopy()
            >>> b.shuffle(0, False)  #: shuffle molecular formulas only
            >>> print(b)
            Fold({'C10H10': array([0, 1, 2]), 'N2': array([0, 1, 2, 3, 4]), 'O3': array([0, 1, 2, 3])})
            >>> c = a.deepcopy()
            >>> c.shuffle(False, 0)  #: shuffle conformation indices only
            >>> print(c)
            Fold({'C10H10': array([0, 2, 1]), 'O3': array([2, 0, 1, 3]), 'N2': array([2, 1, 0, 4, 3])})
            >>> d = a.deepcopy()
            >>> d.shuffle(0, 0)
            >>> print(d)
            Fold({'C10H10': array([0, 2, 1]), 'N2': array([2, 1, 0, 4, 3]), 'O3': array([2, 0, 1, 3])})
        """
        mols = self.mols()
        mol_confs = self.__dict__
        if rs_mol is not False:
            Random(rs_mol).shuffle(mols)
        if rs_conf is not False:
            for mol in mols:
                Random(rs_conf).shuffle(mol_confs[mol])
        self.__dict__ = {mol: mol_confs[mol] for mol in mols}

    def split(self, n_splits: int, show: bool = False) -> List[Fold]:
        r"""Generate a list of n equal splits of current Fold.

        The number of molecular formulas in each split is almost equal,
        with conformation indices unchanged. For datasets with small number
        of molecular formulas, the number of conformations in each split
        can vary greatly.

        Args:
            n_splits (int): Number of splits
            show (bool): Show a summary of the splits, including the number
                of molecules and the number of conformations

        Returns:
            List[Fold]

        Examples:
            >>> from fold import Fold
            >>> a = Fold({'H2': np.arange(2), 'N2': np.arange(3),\
            ...           'O2': np.arange(4), 'F2': np.arange(5),\
            ...           'Cl2': np.arange(6)})
            >>> a_splits = a.split(3, show=True)
            ======================= Fold Split =======================
            Fold 0: 2 molecules, 5 conformations
            Fold 1: 2 molecules, 9 conformations
            Fold 2: 1 molecules, 6 conformations
            ==========================================================
            >>> for s in a_splits:
            ...     print(s)
            Fold({'H2': array([0, 1]), 'N2': array([0, 1, 2])})
            Fold({'O2': array([0, 1, 2, 3]), 'F2': array([0, 1, 2, 3, 4])})
            Fold({'Cl2': array([0, 1, 2, 3, 4, 5])})
        """
        folds = list()
        mols = self.mols()
        mols_splits = np.array_split(mols, n_splits)
        for mols_split in mols_splits:
            mol_confs_split = {mol: np.sort(self[mol]) for mol in mols_split}
            folds.append(Fold(mol_confs_split))
        if show:
            print(f"======================= Fold Split =======================")
            for ifold, fold in enumerate(folds):
                print(f"Fold {ifold}: {fold.nmols()} molecules, {fold.nconfs()} conformations")
            print(f"==========================================================")
        return folds

    @classmethod
    def from_dataset(cls, dset: Union[dict, h5py.File], conf_entry: str = 'coordinates') -> Fold:
        r"""Generate a Fold from an ANI-like dataset

        Args:
            dset (Union[dict, h5py.File]):
                ANI-like dataset, either a dictionary or a HDF5 file handler
            conf_entry (str):
                Entry of which the length will be used to determine the conformation indices

        Returns:
            Fold

        Examples:
            >>> from fold import Fold
            >>> from h5py import File
            >>> dataset_path = "HDF5_DATASET_PATH"
            >>> with File (dataset_path, 'r') as dataset:
            ...     fold = Fold.from_dataset(dataset)
        """
        mol_confs = {mol: np.arange(len(moldata[conf_entry])) for mol, moldata in dset.items()}
        return Fold(mol_confs)

    @classmethod
    def sum(cls, *folds: Fold) -> Fold:
        r"""Summation of a list of Fold objects

        Args:
            *folds (Fold)

        Returns:
            Fold

        Examples:
            >>> from fold import Fold
            >>> a = Fold({'C10H10': np.array([0, 1, 2]), 'O3': np.array([0, 1, 2])})
            >>> b = Fold({'C6H6': np.array([0, 1, 2]), 'N2': np.array([0, 1, 2])})
            >>> c = Fold({'C10H10': np.array([1, 2, 3]), 'O3': np.array([3, 4]), \
            ...           'N2': np.array([2, 3])})
            >>> folds = [a, b, c]
            >>> Fold.sum(a, b, c)
            Fold({'C10H10': array([0, 1, 2, 3]), 'O3': array([0, 1, 2, 3, 4]), 'C6H6': array([0, 1, 2]), 'N2': array([0, 1, 2, 3])})
            >>> Fold.sum(*folds)
            Fold({'C10H10': array([0, 1, 2, 3]), 'O3': array([0, 1, 2, 3, 4]), 'C6H6': array([0, 1, 2]), 'N2': array([0, 1, 2, 3])})
        """
        new_fold = Fold({})
        for fold in folds:
            new_fold += fold
        return new_fold
