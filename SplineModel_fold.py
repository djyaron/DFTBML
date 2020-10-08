import numpy as np
from collections.abc import MutableMapping
from random import Random


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
        ani1_path = input("Please enter ani1_path")
        gammas_path = input("Please enter gammas_path")

    from h5py import File
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
