from collections import Counter
from itertools import combinations
from multiprocessing import Pool

import numpy as np
from scipy.spatial.distance import pdist

from dataset import Dataset, Gammas
from options import Options
from tfspline import spline_linear_model


# TODO: compatibility with different models
class Generator:
    def __init__(self, dset: Dataset, opts: Options):
        self.dset = dset
        self.opts = opts
        self.Zs = opts.Zs
        self._gammas = None
        self._loc = None
        self._nvars = None

    def gammas(self, n_worker: int = 1) -> Gammas:
        if self._gammas is not None:
            return self._gammas
        else:
            # Precompute loc to avoid repetitive calculation during parallelization
            loc = self.loc()
            # Compute gammas of each molecule in parallel
            # joblib.Parallel is not used, since dset contains large objects
            pool = Pool(n_worker)
            out = pool.starmap_async(self.gammas_mol, self.dset.items())
            out.wait()
            res_gammas = out.get()
            # Combine gammas of molecules
            gammas = {}
            for gammas_mol in res_gammas:
                gammas.update(gammas_mol)
            self._gammas = Gammas(gammas)
            self._gammas.opts = self.opts
            self._gammas.loc = loc
            return self._gammas

    def gammas_mol(self, mol: str, moldata: dict) -> dict:
        nvars = self.nvars()
        loc = self.loc()
        _gammas_mol = []

        # Compute spline bases for each conformation
        for coords in moldata['coordinates']:
            _gammas_conf = np.zeros(nvars)
            ## Compute intramolecular pairwise distances
            rs = self.pairwise_dist(coords, moldata['atomic_numbers'], self.opts.cutoff)
            ## Compute spline bases for each pairwise interaction (Zs)
            for Z in self.opts.Zs:
                try:
                    spl = spline_linear_model(xknots=self.opts.xknots[Z], xeval=rs[Z], xyfit=None,
                                              bconds=self.opts.bconds[Z], deg=self.opts.deg[Z],
                                              max_der=self.opts.maxder)
                except KeyError:
                    ### Skip interactions not existing in current conformation
                    continue
                _gammas_conf[loc[Z]] = np.einsum('ij->j', spl['X'][0])
            _gammas_mol.append(_gammas_conf)

        # Stacking gammas (1-D array) of conformations gives gammas of current molecule (nconfs x nvars matrix)
        _gammas_mol = np.vstack(_gammas_mol)

        # Fill in reference energy components
        try:
            _gammas_mol[:, loc['const']] = 1
        except KeyError:
            pass
        atom_counts = Counter(moldata['atomic_numbers'])
        for entry, idx in loc.items():
            try:
                # atom counts in loc should be integers (int, np.int, ...)
                _gammas_mol[:, idx] = atom_counts[int(entry)]
            except (TypeError, ValueError):
                continue

        # Pack atomic numbers and gammas
        res_gammas = {mol: {'atomic_numbers': moldata['atomic_numbers'],
                            'gammas': _gammas_mol}}
        return res_gammas

    def loc(self) -> dict:
        r"""
        Returns: self._loc (dict)
            Structure: {# ---- Locations of spline bases of pairwise interactions ----
                        Z1: 1-D array
                        Z2: 1-D array
                        ...
                        'spline': 1-D array, concatenation of arrays of Zs

                        # ---- Locations of reference energy components ----
                        'const': 1x1 array. Exist only if opts.ref is set to 'full'
                        ## --- Locations of atom counts ---
                        ## --- Exists only if opts.ref is NOT set to 'none' ---
                        atype1: 1x1 array.
                        atype2: 1x1 array
                        ...
                        'ref': 1-D array, concatenation of arrays of reference energy
                               components. Exists only if opts.ref is NOT set to 'none'}

        """
        if self._loc is not None:
            return self._loc
        else:
            self._loc = dict()
            idx_spl_begin = 0
            idx_running = idx_spl_begin
            # Locations of spline bases
            for Z in self.Zs:
                spl = spline_linear_model(xknots=np.linspace(0.0, 1.0, self.opts.nknots[Z]),
                                          xeval=[0], xyfit=None, bconds=self.opts.bconds[Z],
                                          max_der=self.opts.maxder, deg=self.opts.deg[Z])
                nvars_Z = spl['X'][0].shape[1]
                self._loc[Z] = np.arange(idx_running, idx_running + nvars_Z)
                idx_running += nvars_Z
            idx_spl_end = idx_running
            self._loc['spline'] = np.arange(idx_spl_begin, idx_spl_end)
            # Locations of reference energy components
            if self.opts.ref != 'none':
                idx_ref_begin = idx_running
                if self.opts.ref == 'full':
                    self._loc['const'] = np.array([idx_running])
                    idx_running += 1
                ## Locations of atom counts
                for i, atype in enumerate(self.opts.atypes):
                    self._loc[atype] = np.array([idx_running + i])
                idx_running += len(self.opts.atypes)
                idx_ref_end = idx_running
                self._loc['ref'] = np.arange(idx_ref_begin, idx_ref_end)
            else:
                self._loc['ref'] = np.array([])
            return self._loc

    def nvars(self) -> int:
        if self._nvars is not None:
            return self._nvars
        else:
            return 1 + max([max(idx) for idx in self.loc().values()])

    @staticmethod
    def pairwise_dist(coords, atomic_numbers, cutoff) -> dict:
        zs = combinations(atomic_numbers, 2)
        zs_unique = set(combinations(sorted(atomic_numbers), 2))
        res = {z: [] for z in zs_unique}
        dists = pdist(coords)
        for iz, z in enumerate(zs):
            cutoff_z = cutoff[tuple(sorted(z))]
            if min(cutoff_z) <= dists[iz] <= max(cutoff_z):
                res[tuple(sorted(z))].append(dists[iz])
        res = {z: np.sort(dist) for z, dist in res.items() if dist}
        return res


if __name__ == '__main__':
    from h5py import File

    h5set_path = '/home/francishe/Documents/DFTBrepulsive/Datasets/aed_1K.h5'
    with File(h5set_path, 'r') as h5set:
        dset = Dataset(h5set)

    opts = Options({'nknots': 25,
                    'cutoff': 'au_short',
                    'deg': 3,
                    'bconds': 'vanishing',
                    }).convert(dset.Zs())

    gen = Generator(dset, opts)
    gammas = gen.gammas(8)