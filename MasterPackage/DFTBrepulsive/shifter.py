from collections import Counter

import numpy as np
from sklearn.linear_model import LinearRegression

from .consts import SYM2ATOM
from .dataset import Dataset


class Shifter:
    def __init__(self, fit_intercept=True) -> None:
        self.fit_intercept = fit_intercept
        self.shifter = LinearRegression(fit_intercept=fit_intercept)
        self.atypes = None
        self.coef = None
        self.intercept = None
        self.target = None

    def fit(self, dset: Dataset, target: str) -> None:
        r"""Fit the shifter to a target

        Args:
            dset:
            target: str
                A single entry or an expression of entries that the shifter is fitted to.
        """
        self.target = target
        # Columns of fset: 'molecule', 'conformation', atom counts ('H', 'C', 'N', etc.), targets
        fset = dset.extract(self.target).flatten()
        atom_counts = fset.iloc[:, -len(dset.atypes())-1:-1]
        self.atypes = tuple(SYM2ATOM[_atype] for _atype in atom_counts.columns)
        targets = fset.iloc[:, -1:]
        self.shifter.fit(X=atom_counts, y=targets)
        self.coef = self.shifter.coef_
        self.intercept = self.shifter.intercept_

    # WARNING: Hardcoded output dataset entries: coordinates, atomic_numbers and the shifted entry
    def shift(self, dset: Dataset, target: str, reverse=False) -> Dataset:
        r"""Shift a target using pre-fitted shifter

        Args:
            dset: Dataset
            target: str
            reverse: bool

        Returns:
            sset: Dataset
                Shifted dataset

        """
        sset = {}
        _dset = dset.extract(target, entries=('atomic_numbers', 'coordinates'))
        for mol, moldata in _dset.items():
            data = {}
            for entry in ('atomic_numbers', 'coordinates'):
                try:
                    data.update({entry: moldata[entry]})
                except KeyError:
                    continue
            atom_counts = Counter(moldata['atomic_numbers'])
            atom_counts = np.reshape([atom_counts[_atype] for _atype in self.atypes], (1, -1))
            shifter_pred = self.shifter.predict(atom_counts).flatten()
            if reverse:
                data.update({target: moldata[target] + shifter_pred})
            else:
                data.update({target: moldata[target] - shifter_pred})
            sset[mol] = data
        rset = Dataset(sset, dset.conf_entry, dset.fixed_entry)
        return rset


if __name__ == '__main__':
    from h5py import File
    from dataset import Dataset

    h5set_path = '/home/francishe/Documents/DFTBrepulsive/Datasets/aed_1K.h5'
    with File(h5set_path, 'r') as h5set:
        dset = Dataset(h5set)
    shifter = Shifter()
    shifter.fit(dset, 'fm-pf+pr')
    sset = shifter.shift(dset, 'fm-pf+pr')
    print(sset.entries())
    print(sset['s01_0_2Au'])
    fset = sset.flatten()
    print(fset.head(5))
