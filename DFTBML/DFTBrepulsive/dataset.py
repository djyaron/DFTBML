from __future__ import annotations

import copy
import pickle as pkl
from itertools import combinations
from typing import ItemsView, Iterable, Iterator, KeysView, List, Union, ValuesView, Dict

import h5py
import numpy as np
import pandas as pd
import torch
from h5py import File
from scipy.spatial.distance import pdist

from .consts import ATOM2SYM, TARGET2ALIAS
from .fold import Fold
from .target import Target
from .util import formatZ, count_n_heavy_atoms

Tensor = torch.Tensor


# TODO: implement Dataset.shuffle and Dataset.split when necessary
# TODO: concatenate datasets (with same entries)?
# TODO: Flatten dataset with coordinates?

class Dataset:
    def __init__(self, dset: Union[dict, h5py.File],
                 conf_entry: str = 'coordinates',
                 fixed_entry: Iterable = ('atomic_numbers',)) -> None:
        r"""Load a dataset into memory

        Args:
            dset (Union[dict, h5py.File]):
                ANI-like dataset, either a dictionary or a HDF5 file handler
            conf_entry (str):
                Entry of which the length will be used to determine the conformation indices by Fold
            fixed_entry (Union[list, tuple, np.ndarray]):
                Non-sliceable entries, i.e. entries dependent on molecules instead of conformations,
                e.g. atomic numbers
        """

        # Load dataset
        self.dset = {}
        if isinstance(dset, File):
            for mol, moldata in dset.items():
                _moldata = {entry: data[()] for entry, data in moldata.items()}
                self.dset[mol] = _moldata
        else:
            self.dset.update(dset)
        self.fixed_entry = tuple(fixed_entry)
        self.conf_entry = conf_entry
        self._Zs = None
        self._atypes = None
        self.fold = Fold.from_dataset(self.dset, self.conf_entry)

    def __setitem__(self, mol: str, moldata: dict) -> None:
        self.dset[mol] = moldata

    def __getitem__(self, mol: str) -> dict:
        return self.dset[mol]

    def __delitem__(self, mol: str) -> None:
        del self.dset[mol]

    def __iter__(self) -> Iterator:
        return iter(self.dset.keys())

    def __len__(self) -> int:
        return len(self.dset)

    def atypes(self) -> tuple:
        if self._atypes is not None:
            return self._atypes
        _atypes = set()
        try:
            for moldata in self.dset.values():
                _atypes.update(moldata['atomic_numbers'])
        except KeyError:
            print("Atomic numbers not specified")
        self._atypes = tuple(sorted(_atypes))
        return self._atypes

    def confs(self) -> List[np.ndarray]:
        return self.fold.confs()

    def copy(self) -> Dataset:
        return Dataset(self.dset, self.conf_entry, self.fixed_entry)

    def deepcopy(self) -> Dataset:
        return Dataset(copy.deepcopy(self.dset), self.conf_entry, self.fixed_entry)

    def distance(self) -> Dict[tuple, np.ndarray]:
        dists = {}
        for moldata in self.values():
            coords = moldata['coordinates']
            atoms = moldata['atomic_numbers']
            dist = self._dist_mol(coords, atoms)
            for z, d in dist.items():  # z: atom pair tuple
                try:
                    dists[z].extend(d)
                except KeyError:
                    dists[z] = list(d)
        return {z: np.array(d) for z, d in dists.items()}

    def entries(self) -> List[str]:
        # assuming molecules have the same entries
        _mol = self.mols()[0]
        return sorted(self[_mol].keys())

    def extract(self, target: str,
                entries: Iterable = ('atomic_numbers',),
                conf_entry: str = 'target',
                fixed_entry: Iterable = ('atomic_numbers',)) -> Dataset:
        r"""Extract a target (single target or an expression) from the dataset

        Args:
            target: str
                Alias of an entry or an expression of entries
                E.g. 'fm', 'fm-pf', 'fm - pf', 'fm-pf+pr', 'fm - pf + pr' are valid targets
            entries: Iterable
                Entries to be included other than the target
            conf_entry:
            fixed_entry:

        Returns:

        """
        alias_disabled = target in self.entries()
        tg = target if alias_disabled else Target(target)
        res = {}
        for mol, moldata in self.items():
            res[mol] = {}
            for entry in entries:
                try:
                    res[mol].update({entry: moldata[entry]})
                except KeyError:
                    continue
            if alias_disabled:
                data = moldata[tg]
            else:
                data = None
                for tt, op in tg.items():
                    if data is None:
                        try:
                            data = op * moldata[tt]
                        except KeyError:
                            data = op * moldata[TARGET2ALIAS[tt]]
                    else:
                        try:
                            data += op * moldata[tt]
                        except KeyError:
                            data += op * moldata[TARGET2ALIAS[tt]]
            res[mol].update({target: data})

        _fixed_entry = fixed_entry
        _conf_entry = target if conf_entry == 'target' else conf_entry
        return Dataset(res, _conf_entry, _fixed_entry)

    # TODO: support entries with more than one dimension (coordinates, forces, etc.)
    def flatten(self, atypes: Iterable = 'infer') -> pd.DataFrame:
        r"""Flatten the dataset (currently only supports energies)"""
        _atypes = atypes if atypes != 'infer' else self.atypes()
        fset = {'molecule': [], 'conformation': []}  # "flattened set" -> "fset"
        if _atypes is not None:
            fset.update({ATOM2SYM[atype]: [] for atype in _atypes})
        fset.update({entry: [] for entry in self.entries()
                     # Skip fixed entries and entries with more than 1-D (for now),
                     # e.g. coordinates and forces
                     if entry not in self.fixed_entry
                     and entry != 'coordinates'
                     and 'forces' not in entry})
        for mol, moldata in self.items():
            nconf = len(moldata[self.conf_entry])
            fset['molecule'].extend([mol] * nconf)
            fset['conformation'].extend(range(nconf))
            if _atypes is not None:
                anumbers = list(moldata['atomic_numbers'])
                # count atoms of each type
                for atype in _atypes:
                    count = anumbers.count(atype)
                    fset[ATOM2SYM[atype]].extend([count] * nconf)
            for entry, data in moldata.items():
                if entry not in self.fixed_entry \
                        and entry != 'coordinates' \
                        and 'forces' not in entry:
                    fset[entry].extend(data)
        fset = pd.DataFrame(fset)
        return fset

    def items(self) -> ItemsView:
        return self.dset.items()

    def keys(self) -> KeysView:
        return self.dset.keys()

    def mols(self) -> List[str]:
        return self.fold.mols()

    def mol_confs(self) -> ItemsView:
        return self.fold.mol_confs()

    def nconfs(self) -> int:
        return self.fold.nconfs()

    def nmols(self) -> int:
        return self.fold.nmols()

    # TODO: implement when necessary
    def shuffle(self):
        raise NotImplementedError

    def slice(self, fd: Fold) -> Dataset:
        r"""Slice the dataset given a fold"""
        res = {}
        for mol, conf_arr in fd.mol_confs():
            moldata = {}
            for entry, data in self.dset[mol].items():
                if entry in self.fixed_entry:
                    moldata[entry] = np.array(data)
                else:
                    moldata[entry] = data[conf_arr, ...]
            res[mol] = moldata
        return Dataset(res, self.conf_entry, self.fixed_entry)

    # TODO: implement when necessary
    def split(self):
        raise NotImplementedError

    def to_hdf5(self, path):
        with File(path, 'w') as des:
            for mol, moldata in self.items():
                g_mol = des.create_group(mol)
                for entry, data in moldata.items():
                    g_mol.create_dataset(entry, data=data)

    def to_pickle(self, path):
        pkl.dump(self, open(path, 'wb'))

    def values(self) -> ValuesView:
        return self.dset.values()

    def Zs(self) -> tuple:
        if self._Zs is not None:
            return self._Zs
        _Zs = set()
        try:
            for moldata in self.values():
                _Zs.update(combinations(moldata['atomic_numbers'], 2))
        except KeyError:
            print("Atomic numbers not specified")
        self._Zs = formatZ(_Zs, unique=True)
        return self._Zs

    @classmethod
    def merge(cls, *dsets: Dataset) -> Dataset:
        r"""Merge multiple datasets with the same number of molecules

        Args:
            *dsets: Dataset
                Datasets with identical molecules and conformations

        Returns:
            Dataset

        Notes:
            This method well always keep the data from the last dataset
            if there are overlapping entries among the datasets
        """
        rset = {}  # "result set" -> "rset"
        fixed_entry, conf_entry, entries = set(), set(), set()
        for dset in dsets:
            fixed_entry.update(dset.fixed_entry)
            conf_entry.add(dset.conf_entry)
            entries.update(dset.entries())
            if len(rset) == 0:
                rset.update(dset.dset)
            else:
                for (mol, moldata_rset), (mol_dset, moldata_dset) in zip(rset.items(), dset.dset.items()):
                    assert mol == mol_dset, "Molecular formulas do not match across datasets"
                    # WARNING: Only keeps the data from the last dataset if there are overlapping entries
                    moldata_rset.update({entry: data for entry, data in moldata_dset.items()})
        fixed_entry = sorted(fixed_entry)
        conf_entry = sorted(conf_entry)[0]  # conf_entry is a unique entry to determine the number of conformations
        rset = Dataset(rset, conf_entry, fixed_entry)
        return rset

    @classmethod
    def compare(cls, dset1: Dataset, entry1: str,
                dset2: Dataset, entry2: str,
                metric: str = 'mae', scale: bool = False):
        # Check if the molecules and conformations of the datasets are identical
        try:
            for (mol1, conf_arr1), (mol2, conf_arr2) in zip(dset1.mol_confs(), dset2.mol_confs()):
                assert mol1 == mol2
                assert (np.array(conf_arr1) == np.array(conf_arr2)).all()
        except AssertionError:
            raise ValueError('Molecules and/or conformations of the two datasets do not match')
        # Compute the difference between two entries
        diff = []
        _d1 = dset1.extract(entry1)
        _d2 = dset2.extract(entry2)
        if scale:
            for moldata1, moldata2 in zip(_d1.values(), _d2.values()):
                nHA = count_n_heavy_atoms(moldata1['atomic_numbers'])
                diff.extend((moldata1[entry1] - moldata2[entry2]) / nHA)
        else:
            for moldata1, moldata2 in zip(_d1.values(), _d2.values()):
                diff.extend(moldata1[entry1] - moldata2[entry2])
        # Compute the different in given metric
        if metric in ('mae', 'MAE', 'mean absolute error', 'mean_absolute_error'):
            return np.average(np.abs(diff))
        elif metric in ('rms', 'RMS', 'rmse', 'RMSE', 'root mean square error', 'root_mean_square_error'):
            return np.sqrt(np.average(np.square(diff)))
        elif metric in ('err', 'error'):
            return np.array(diff)
        else:
            raise ValueError("Metric type not recognized")

    @staticmethod
    def _dist_mol(coords_mol: np.ndarray, atoms: np.ndarray) -> dict:
        r"""Generate pairwise distances of conformations of a molecular formula

        Args:
            coords_mol (np.ndarray): coordinate tensor of a molecular formula
                with shape (#conformations, #atoms, 3), where "3" corresponds
                to Cartesian coordinates
            atoms (np.ndarray): a 1-D array of atomic numbers corresponding
                to the atoms in the coordinate tensor

        Returns:
            dict: whose keys are atom pairs,
                and values are a 1-D np.ndarray of pairwise distances

        Examples:
            >>> import numpy as np
            >>> from dataset import Dataset
            >>> np.random.seed(0)  #: used to generate reproducible random numbers
            >>> coords_mol = np.random.rand(5, 4, 3)
            >>> atoms = np.arange(4)
            >>> res = Dataset._dist_mol(coords_mol, atoms)
            >>> for k, v in res.items():
            ...     print(f"{k}: {np.around(v, 3)}")
            (0, 1): [0.295 1.277 0.878 0.573 0.79 ]
            (0, 2): [0.417 0.933 0.371 0.628 0.243]
            (0, 3): [0.197 0.879 0.902 0.912 0.347]
            (1, 2): [0.576 1.105 0.727 0.378 1.02 ]
            (1, 3): [0.419 0.839 0.66  0.342 0.572]
            (2, 3): [0.449 0.455 0.83  0.559 0.512]
        """
        atom_pairs = tuple(combinations(atoms, 2))  #: atom pairs, used to keep track of the output of pdist
        zs = sorted(set(combinations(sorted(atoms), 2)))  #: sorted and deduplicated atom pairs

        dist_tensor = np.stack([pdist(coords) for coords in coords_mol], axis=0)

        res_tmp = {z: [] for z in zs}
        for i, atom_pair in enumerate(atom_pairs):
            res_tmp[tuple(sorted(atom_pair))].extend(dist_tensor[:, i])

        res = {Z: np.array(dist) for Z, dist in res_tmp.items()}
        return res


# noinspection PyAbstractClass
class Gammas(Dataset):
    def __init__(self, dset: Union[dict, h5py.File],
                 conf_entry: str = 'gammas',
                 fixed_entry: Iterable = ('atomic_numbers',)):
        super().__init__(dset, conf_entry, fixed_entry)
        self.opts = None
        self.loc = None

    def nvars(self) -> int:
        mol = self.mols()[0]
        return self[mol]['gammas'].shape[1]

    def __mul__(self, coef: Union[np.ndarray, Tensor]) -> Dataset:
        r"""Multiple gammas by a coefficient vector to give model predictions"""
        rset = {mol: {'atomic_numbers': moldata['atomic_numbers'],
                      'prediction': moldata['gammas'].dot(coef) if isinstance(coef, np.ndarray)
                      else torch.matmul(torch.tensor(moldata['gammas'], dtype=torch.double), coef)}
                for mol, moldata in self.items()}
        return Dataset(rset, conf_entry='prediction', fixed_entry=('atomic_numbers',))

# if __name__ == '__main__':
#     h5set_path = '/home/francishe/Documents/DFTBrepulsive/Datasets/aed_1K.h5'
#     with File(h5set_path, 'r') as h5set:
#         dset = Dataset(h5set)
#
#     eset = dset.extract('fhi_aims_md.mermin_energy', ('atomic_numbers', 'coordinates'))
#     print(eset.entries())
#     eset = eset.extract('fhi_aims_md.mermin_energy', ('atomic_numbers', 'coordinates'))
#     print(eset.entries())
#     eset = eset.extract('fm', ('atomic_numbers', 'coordinates'))
#     print(eset.entries())
#     eset = eset.extract('fm', ('atomic_numbers', 'coordinates'))
#     print(eset.entries())
#     eset = eset.extract('fm', ('atomic_numbers', 'coordinates', 'non_existing_entry'))
#     print(eset.entries())
#
#     fset = dset.flatten()
#     print(fset.head())
#
#     err = Dataset.compare(dset, 'fm', dset, 'pt')
#     print(f"{err: .3f} Hartrees")
#
#     a = dset.extract('fm')
#     b = dset.extract('coordinates')
#     c = dset.extract('pf')
#     rset = Dataset.merge(a, b, c)
#     print(rset.entries())
