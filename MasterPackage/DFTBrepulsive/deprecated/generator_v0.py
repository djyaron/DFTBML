from collections import Counter
from itertools import combinations
from multiprocessing import cpu_count, Pool

import numpy as np
from h5py import File
from scipy.spatial.distance import pdist

from consts import *
from tfspline import spline_linear_model
from util import path_check


# TODO: Gammas class?

class Generator:
    # TODO: pack model hyperparameters in an Options object
    # TODO: infer Zs
    def __init__(self, dataset, Nknots=50, Degrees=3, R_cutoffs='full', Bconds='vanishing',
                 gammas_path='./results/gammas/gammas.h5', in_memory=False, n_worker=1):
        """
        Originated from SplineModel_v2. Only core functions were kept to generate gammas

        Parameters
        ----------
        dataset
        Nknots: int
        Degrees: dict or int
        R_cutoffs: dict or str
        Bconds: dict or str
        gammas_path: str
        in_memory: bool
        """
        self.dataset = dataset
        self.Nknots = Nknots
        self.Degrees = Degrees
        self.R_cutoffs = R_cutoffs
        self.Bconds = Bconds
        self.gammas_path = gammas_path
        self.Zs = None
        self.Xis = None
        self.Xknots = None
        self.Xgrid = None
        self.loc = None
        self._gamma = None
        self.gammas = None
        self.nvars = None
        self.in_memory = in_memory
        self.n_worker = n_worker

        self.sanity_check()
        path_check(self.gammas_path)

    def init_model(self):
        self.get_ZXis()
        self.params_convert()
        self.get_Xknots()
        self.init_LABG()
        self.get_gammas()

    def get_gammas(self):
        gammas_mol = list()

        if self.in_memory:
            opts = {'Zs': self.Zs,
                    'Xis': self.Xis,
                    'Bconds': self.Bconds,
                    'Nknots': self.Nknots,
                    'Degrees': self.Degrees,
                    'R_cutoffs': self.R_cutoffs,
                    'Xknots': self.Xknots,
                    'nvars': self.nvars,
                    'loc': self.loc}
            self.gammas = compute_gammas(self.dataset, opts, self.n_worker)

        else:
            with File(self.gammas_path, 'w') as gammas_file:
                for mol in self.dataset.keys():
                    coordinates = self.dataset[mol]['coordinates'][()]
                    atomic_numbers = self.dataset[mol]['atomic_numbers'][()]
                    n_heavy_atoms = self.count_heavy_atoms(atomic_numbers)

                    for coords in coordinates:
                        self._gamma[self.loc['const']] = 1
                        for Xi in self.Xis:
                            self._gamma[self.loc[Xi]] = list(atomic_numbers).count(Xi)
                        Rs_config = self.pairwise_dist(coords, atomic_numbers)

                        for Z in self.Zs:
                            try:
                                spl = spline_linear_model(xknots=self.Xknots[Z], xeval=Rs_config[Z], xyfit=None,
                                                          bconds=self.Bconds[Z], max_der=0, deg=self.Degrees[Z])
                            except KeyError:
                                continue  # Skip interactions not existing in the molecule
                            self._gamma[self.loc[Z]] = np.einsum('ij->j', spl['X'][0])

                        gammas_mol.append(self._gamma.copy())
                        self._gamma.fill(0)

                    group_mol = gammas_file.create_group(mol)
                    group_mol.create_dataset('n_heavy_atoms', data=n_heavy_atoms)
                    group_mol.create_dataset('gammas', data=np.stack(gammas_mol, axis=0))
                    gammas_mol.clear()

                # Write dataset and model info
                info_dict = {'Zs': np.array([list(Z) for Z in self.Zs]),
                             'Xis': np.array(self.Xis),
                             'Nknots': np.array([[n] for n in self.Nknots.values()]),
                             'Degrees': np.array([[d] for d in self.Degrees.values()]),
                             'R_cutoffs': np.array([list(r) for r in self.R_cutoffs.values()]),
                             'Xknots': np.array([x for x in self.Xknots.values()])}
                group_info = gammas_file.create_group('_INFO')
                for param, info in info_dict.items():
                    group_info.create_dataset(param, data=info)

    def get_Xknots(self):
        self.Xknots = dict()
        for Z, cutoff_z in self.R_cutoffs.items():
            self.Xknots[Z] = np.linspace(cutoff_z[0], cutoff_z[1], self.Nknots[Z])

    def get_ZXis(self):
        self.Zs = set()
        self.Xis = set()
        for mol in self.dataset.keys():
            atomic_numbers = self.dataset[mol]['atomic_numbers'][()]
            self.Zs.update(combinations(sorted(atomic_numbers), 2))
            self.Xis.update(atomic_numbers)
        self.Zs = sorted(self.Zs)
        self.Xis = sorted(self.Xis)

    def init_LABG(self):
        self.loc = dict()
        idx_spl_b = 0
        idx_running = idx_spl_b

        # locations of coefficients of splines in gamma
        for Z in self.Zs:
            xknots_dummy = np.linspace(0.0, 1.0, self.Nknots[Z])
            xeval_dummy = [0]
            spl = spline_linear_model(xknots=xknots_dummy, xeval=xeval_dummy, xyfit=None,
                                      bconds=self.Bconds[Z], max_der=0, deg=self.Degrees[Z])
            nvars_Z = spl['X'][0].shape[1]
            self.loc[Z] = np.arange(idx_running, idx_running + nvars_Z)
            idx_running += nvars_Z

        idx_ref_b = idx_spl_e = idx_running

        # locations of coefficients of reference energies in gamma
        self.loc['const'] = np.array([idx_running])
        for iXi, Xi in enumerate(self.Xis):
            self.loc[Xi] = np.array([idx_running + 1 + iXi])

        idx_ref_e = idx_running + 1 + len(self.Xis)
        self.nvars = idx_ref_e
        self.loc['spline'] = np.arange(idx_spl_b, idx_spl_e)
        self.loc['ref'] = np.arange(idx_ref_b, idx_ref_e)

        # initialize _gamma
        self._gamma = np.zeros(self.nvars)

    def pairwise_dist(self, coordinates, atomic_numbers):
        zs = list(combinations(atomic_numbers, 2))
        zs_unique = set(combinations(sorted(atomic_numbers), 2))
        tmp = dict((z, list()) for z in zs_unique)
        dists = pdist(coordinates)
        for iz, z in enumerate(zs):
            cutoff_z = self.R_cutoffs[tuple(sorted(z))]
            if min(cutoff_z) <= dists[iz] <= max(cutoff_z):
                tmp[tuple(sorted(z))].append(dists[iz])
        res = dict((z, np.sort(dist)) for z, dist in tmp.items() if dist)
        return res

    def params_convert(self):
        # Convert self.Nknots into a dictionary
        if isinstance(self.Nknots, int):
            self.Nknots = dict().fromkeys(self.Zs, self.Nknots)
        else:  # self.Nknots is a dictionary
            # sort the keys
            self.Nknots = dict((tuple(sorted(k)), v) for k, v in self.Nknots.items())
            if sorted(self.Nknots.keys()) != self.Zs:
                raise KeyError("Nknots: Keys not matching Zs")

        # Convert self.Bconds into a dictionary
        if self.Bconds == 'vanishing':
            self.Bconds = dict().fromkeys(self.Zs, [Bcond(0, 2, 0.0), Bcond(-1, 0, 0.0), Bcond(-1, 1, 0.0)])
        elif self.Bconds == 'natural':
            self.Bconds = dict().fromkeys(self.Zs, [Bcond(0, 2, 0.0), Bcond(-1, 2, 0.0)])
        else:  # self.Bconds is a dictionary
            # sort the keys
            self.Bconds = dict((tuple(sorted(k)), v) for k, v in self.Bconds.items())
            if sorted(self.Bconds.keys()) != self.Zs:
                raise KeyError("Bconds: Keys not matching Zs")

        # Convert self.Degrees into a dictionary
        if isinstance(self.Degrees, int):
            if not 2 <= self.Degrees <= 5:
                raise ValueError("Degrees: must between 2 and 5")
            else:
                self.Degrees = dict().fromkeys(self.Zs, self.Degrees)
        else:  # self.Degrees is a dictionary
            # sort the keys
            self.Degrees = dict((tuple(sorted(k)), v) for k, v in self.Degrees.items())
            if sorted(self.Degrees.keys()) != self.Zs:
                raise KeyError("Degrees: Keys not matching Zs")

        # Convert self.R_cutoffs into a dictionary
        if isinstance(self.R_cutoffs, str):
            # CUTOFFS is defined in consts.py
            self.R_cutoffs = CUTOFFS[self.R_cutoffs]
        elif isinstance(self.R_cutoffs, float):
            self.R_cutoffs = dict().fromkeys(self.Zs, (0, self.R_cutoffs))
        elif isinstance(self.R_cutoffs, dict):
            # sort the atom pairs
            self.R_cutoffs = dict((tuple(sorted(k)), v) for k, v in self.R_cutoffs.items())
            if sorted(self.R_cutoffs.keys()) != self.Zs:
                raise KeyError("R_cutoffs: Keys not matching Zs")
        else:
            raise NotImplementedError

    def sanity_check(self):
        if not isinstance(self.Nknots, (int, dict)):
            raise TypeError("Nknots: Invalid data type")
        if not isinstance(self.Degrees, (int, dict)):
            raise TypeError("Degrees: Invalid data type")
        if not isinstance(self.R_cutoffs, (str, float, dict)):
            raise TypeError("R_cutoffs: Invalid data type")
        if not isinstance(self.Bconds, (list, dict)) and self.Bconds not in ('natural', 'vanishing'):
            raise TypeError("Bconds: Invalid data type")
        if not isinstance(self.gammas_path, str):
            raise TypeError("gammas_path: Invalid data type")

    def save_gammas(self, gammas_path=None):
        g_path = gammas_path if gammas_path else self.gammas_path
        if self.in_memory:
            with File(g_path, 'w') as gammas:
                for mol, moldata in self.gammas.items():
                    g = gammas.create_group(mol)
                    for entry, data in moldata.items():
                        g.create_dataset(entry, data=data)
        else:
            pass


    @staticmethod
    def count_heavy_atoms(atomic_numbers):
        return sum(list(v for k, v in Counter(atomic_numbers).items() if k != 1))


def compute_gammas(dataset, opts, n_worker=1) -> dict:
    r"""Compute gammas in parallel

    Args:
        dataset:
        opts:
        n_worker:

    Returns:
        gammas (dict): gammas dictionary with data structure of
            {mol_0: {'n_heavy_atoms': int, 'gammas': },
             mol_1: {},
             ...,
             mol_-1: {},
             __INFO: info_dict}
    """
    Zs = opts['Zs']
    Xis = opts['Xis']
    Nknots = opts['Nknots']
    Degrees = opts['Degrees']
    R_cutoffs = opts['R_cutoffs']
    Xknots = opts['Xknots']

    if isinstance(dataset, File):
        args = [(mol, moldata['coordinates'][()], moldata['atomic_numbers'][()], opts)
                for mol, moldata in dataset.items()]
    else:
        args = [(mol, moldata['coordinates'], moldata['atomic_numbers'], opts)
                for mol, moldata in dataset.items()]

    pool = Pool(n_worker)
    out = pool.starmap_async(compute_gammas_single_mol, args)
    out.wait()
    res = out.get()

    gammas = {}
    for g_mol in res:
        gammas.update(g_mol)

    # Write dataset and model info
    info_dict = {'Zs': np.array([list(Z) for Z in Zs]),
                 'Xis': np.array(Xis),
                 'Nknots': np.array([[n] for n in Nknots.values()]),
                 'Degrees': np.array([[d] for d in Degrees.values()]),
                 'R_cutoffs': np.array([list(r) for r in R_cutoffs.values()]),
                 'Xknots': np.array([x for x in Xknots.values()])}
    gammas.update({'_INFO': info_dict})
    return gammas


def compute_gammas_single_mol(mol, coordinates, atomic_numbers, opts):
    nvars = opts['nvars']
    loc = opts['loc']
    Xis = opts['Xis']
    Zs = opts['Zs']
    Xknots = opts['Xknots']
    Bconds = opts['Bconds']
    Degrees = opts['Degrees']
    R_cutoffs = opts['R_cutoffs']

    _gammas = []

    for coords in coordinates:
        _gamma = np.zeros(nvars)
        _gamma[loc['const']] = 1

        Xi_counts = Counter(atomic_numbers)
        for Xi in Xis:
            _gamma[loc[Xi]] = Xi_counts[Xi]

        Rs_config = pairwise_dist(coords, atomic_numbers, R_cutoffs)

        for Z in Zs:
            try:
                spl = spline_linear_model(xknots=Xknots[Z], xeval=Rs_config[Z], xyfit=None,
                                          bconds=Bconds[Z], max_der=0, deg=Degrees[Z])
            except KeyError:
                continue  # Skip interactions not existing in the molecule
            _gamma[loc[Z]] = np.einsum('ij->j', spl['X'][0])

        _gammas.append(_gamma)

    gammas = {mol: {'atomic_numbers': atomic_numbers,
                    'gammas': np.vstack(_gammas)}}
    return gammas


def pairwise_dist(coordinates, atomic_numbers, R_cutoffs):
        zs = list(combinations(atomic_numbers, 2))
        zs_unique = set(combinations(sorted(atomic_numbers), 2))
        tmp = dict((z, list()) for z in zs_unique)
        dists = pdist(coordinates)
        for iz, z in enumerate(zs):
            cutoff_z = R_cutoffs[tuple(sorted(z))]
            if min(cutoff_z) <= dists[iz] <= max(cutoff_z):
                tmp[tuple(sorted(z))].append(dists[iz])
        res = dict((z, np.sort(dist)) for z, dist in tmp.items() if dist)
        return res


def count_heavy_atoms(atomic_numbers):
    return sum(list(v for k, v in Counter(atomic_numbers).items() if k != 1))


if __name__ == '__main__':
    from util import Timer

    # dataset_path = '/home/francishe/Documents/DFTBrepulsive/Datasets/Au_energy_clean_dispersion.h5'
    # gammas_path = '/home/francishe/Documents/DFTBrepulsive/Datasets/gammas_50_5_full_aed.h5'
    # dataset_path = '/home/francishe/Documents/DFTBrepulsive/Datasets/aed_1K.h5'
    # gammas_path = '/home/francishe/Documents/DFTBrepulsive/Datasets/gammas_50_5_full_a1k.h5'
    dataset_path = '/home/francishe/Documents/DFTBrepulsive/Datasets/ANI-1ccx_clean_fullentry.h5'
    gammas_path = '/home/francishe/Documents/DFTBrepulsive/Datasets/gammas_50_5_full_cf.h5'

    Au = False

    Nknots = 50
    Degrees = 5
    Bconds = 'vanishing'
    R_cutoffs = "au_full" if Au else "full"

    with Timer("generating gammas"):
        with File(dataset_path, 'r') as dataset:
            mod = Generator(dataset, Nknots, Degrees, R_cutoffs, Bconds, gammas_path,
                            in_memory=True, n_worker=cpu_count())
            mod.init_model()
            mod.save_gammas()
