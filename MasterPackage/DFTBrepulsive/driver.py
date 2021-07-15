import numpy as np

import pickle as pkl

from scipy.interpolate import CubicSpline

from .consts import CUTOFFS, ANGSTROM2BOHR, EXPCOEFS
from .dataset import Dataset
from .generator import Generator

from .model import RepulsiveModel
from .options import Options
from .util import Z2A, Zs_from_opts, count_n_heavy_atoms


def compute_gammas(dset: dict, opts: dict) -> None:
    r"""Precomputation of gammas
    Args:
        dset: dict
            ANI-like dataset (nested dictionary). No targets needed for this function
            ================================= Data structure =================================
            {mol_0 (str):
                {'atomic_numbers': np.ndarray, shape=(n_atoms,),
                 'coordinates': np.ndarray, shape=(n_confs, n_atoms, 3),
             mol_1: {...},
             mol_2: {...},
             ...}
            ==================================================================================
        opts: dict
            Highest level dictionary of options and hyperparameters

    Note: Call this function in precomputation over the entire dataset
    """
    n_worker = opts['training_settings']['n_worker']
    gammas_path = opts['repulsive_settings']['gammas_path']
    _dset = Dataset(dset, conf_entry='coordinates', fixed_entry=('atomic_numbers',))
    _opts = Options(opts['repulsive_settings']['opts'])
    _Zs = Zs_from_opts(opts)

    _gen = Generator(_dset, _opts.convert(_Zs))
    gammas = _gen.gammas(n_worker=n_worker)
    gammas.to_pickle(gammas_path)


def train_repulsive_model(dset: dict, opts: dict):
    r"""API for dftbtorch

    Args:
        dset: dict
            ANI-like data fold (nested dictionary)
            ================================= Data structure =================================
            {mol_0 (str):
                {'atomic_numbers': np.ndarray, shape=(n_atoms,),
                 'coordinates': np.ndarray, shape=(n_confs, n_atoms, 3),
                 'target': np.ndarray, shape=(n_confs,)
             mol_1: {...},
             mol_2: {...},
             ...}
            ==================================================================================
        opts: dict
            Highest level dictionary of options and hyperparameters.
            The following entries are required:
            {"model_settings": {"low_end_correction_dict": dict},
             # The keys in low_end_correction_dict are used as Zs in DFTBrepulsive
             # (after sorting and formatting)

             "skf_settings": {"spl_ngrid": int},
             # The number of points of the dense grid on which the splines are evaluated
             # during generating the spline block

             "training_settings": {"n_worker": int},
             # The number of CPUs for DFTBrepulsive to run in parallel

             "repulsive_settings": {"opts": {"nknots": int,
                                             # The number of knots of the splines
                                             # of each pairwise potential

                                             "cutoff": str,
                                             # The alias of the cutoff radii dictionary
                                             # specified in consts.py

                                             "deg": int,
                                             # The order of the splines (3 to 5)

                                             "bconds": str,
                                             # The alias of the boundary conditions of the splines
                                             # Examples
                                             #     "vanishing": 2nd deriv -> 0 at the first knot
                                             #                  0th and 1st deriv -> 0 at the last knot
                                             #     "natural": 2nd deriv -> 0 at the first and the last knot

                                             "constr": str
                                             # The constraints on the derivatives of the splines
                                             # "+" or "-" symbol indicates positive/negative constraints,
                                             # with the following number indicating the order of derivative
                                             # (no more than 5-th order)
                                             # Multiple constraints can be applied simultaneously
                                             # Examples:
                                             #     "-1": negative first derivative (monotonic decreasing)
                                             #     "+2": positive second derivative (convex)
                                             #     "+2-1": convex and monotonic decreasing

                                    "gammas_path": "/export/home/hanqingh/Documents/DFTBrepulsive/gammas_a1k.pkl"}
                                    # The file path to save gammas}

    Returns:
        mod: model.RepulsiveModel
        preds: dict
            ANI-like data fold (nested dictionary)
            ================================= Data structure =================================
            {mol_0 (str):
                {'atomic_numbers': np.ndarray, shape=(n_atoms,),
                 'prediction': np.ndarray, shape=(n_confs,)
             mol_1: {...},
             mol_2: {...},
             ...}
            ==================================================================================

    Note: target = ground_truth - dftbtorch_elec
          LOSS = (ground_truth - dftbtorch_elec)/n_heavy_atom
          for i in epochs:
            for j in data_batch:
                gradient_descent
          Update_charges
          DFTBrepulsive_rep

    """

    _dset = {mol: {'atomic_numbers': moldata['atomic_numbers'],
                   'coordinates': moldata['coordinates'],
                   # Scale up the target
                   'target': moldata['target'] * count_n_heavy_atoms(moldata['atomic_numbers'])}
             for mol, moldata in dset.items()}
    n_worker = opts['training_settings']['n_worker']
    gammas_path = opts['repulsive_settings']['gammas_path']
    _dset = Dataset(_dset, conf_entry='coordinates', fixed_entry=('atomic_numbers',))
    _opts = Options(opts['repulsive_settings']['opts'])
    _Zs = Zs_from_opts(opts)

    _gammas = pkl.load(open(gammas_path, 'rb'))

    mod = RepulsiveModel(_opts.convert(_Zs))
    mod.fit(_dset, target='target', gammas=_gammas, shift=True, n_worker=n_worker)
    _preds = mod.predict(_dset, gammas=_gammas, n_worker=n_worker)

    preds = {mol: {'atomic_numbers': moldata['atomic_numbers'],
                   'prediction': moldata['prediction'] / count_n_heavy_atoms(moldata['atomic_numbers'])}
             for mol, moldata in _preds.items()}

    return mod, preds


def get_spline_block(mod: RepulsiveModel, opts: dict) -> dict:
    r"""Generate a spline block from pretrained model

    Args:
        mod: RepulsiveModel
        opts: dict

    Returns:
        spl_blocks: dict
            Keys: Zs, values: spline block in lines

    """
    # Use exponential coefficients in mio-0-1 or auorg-1-1
    Au = 'au' in opts['repulsive_settings']['opts']['cutoff']
    exp_coefs = EXPCOEFS['auorg-1-1'] if Au else EXPCOEFS['mio-0-1']

    Zs = mod.Zs
    spl_ngrid = opts['skf_settings']['spl_ngrid']
    cutoff = CUTOFFS[opts['repulsive_settings']['opts']['cutoff']]

    spl_grids = {Z: np.linspace(c[0], c[1], spl_ngrid) for Z, c in cutoff.items()}
    spl_vals = mod(spl_grids)

    spl_blocks = {}
    for Z in Zs:
        spl_grid = spl_grids[Z] * ANGSTROM2BOHR
        spl_val = spl_vals[Z]
        spl_coef = CubicSpline(spl_grid, spl_val).c
        spl_ints = np.array([spl_grid[:-1], spl_grid[1:]]).T
        exp_coef = exp_coefs[Z]

        line0 = ['Spline']  # title
        line1 = [spl_ngrid - 1, cutoff[Z][-1]]  # nInt, cutoff
        line2 = list(exp_coef)  # exponential coefficients
        lines = [[*spl_int, *spl_coef[::-1, i]] for i, spl_int in enumerate(spl_ints)]
        lines[-1].extend([0, 0])  # zero-pad cubic spline coefs into 5-th order
        spl_blocks[Z] = [line0, line1, line2, *lines]

    return spl_blocks


if __name__ == '__main__':
    from h5py import File

    h5set_path = '/export/home/hanqingh/Documents/DFTBrepulsive/Datasets/aed_1k.h5'
    dset = Dataset(File(h5set_path, 'r'))

    eset = dset.extract('coordinates').dset
    target = 'fm-pf+pr'
    _tset = dset.extract(target, entries=('atomic_numbers', 'coordinates'))
    tset = {mol: {'atomic_numbers': moldata['atomic_numbers'],
                  'coordinates': moldata['coordinates'],
                  'target': moldata[target] / count_n_heavy_atoms(moldata['atomic_numbers'])}
            for mol, moldata in _tset.items()}

    opts = {"model_settings": {"low_end_correction_dict": {(1, 1) : 1.00,
                                                           (6, 6) : 1.04,
                                                           (1, 6) : 0.602,
                                                           (7, 7) : 0.986,
                                                           (6, 7) : 0.948,
                                                           (1, 7) : 0.573,
                                                           (1, 8) : 0.599,
                                                           (6, 8) : 1.005,
                                                           (7, 8) : 0.933,
                                                           (8, 8) : 1.062,
                                                           (1, 79): 0.000,
                                                           (6, 79): 0.000,
                                                           (7, 79): 0.000,
                                                           (8, 79): 0.000,
                                                           (79, 79): 0.000}},
            "skf_settings": {"spl_ngrid": 500},
            "training_settings": {"n_worker": 24},
            "repulsive_settings": {"opts": {"nknots": 25,
                                            "cutoff": "au_full",
                                            "deg": 3,
                                            "bconds": "vanishing",
                                            "constr": "+2"},
                                   "gammas_path": "/export/home/hanqingh/Documents/DFTBrepulsive/gammas_a1k.pkl"}}

    compute_gammas(eset, opts)
    mod, c, loc, preds = train_repulsive_model(tset, opts)
    spl_block = get_spline_block(mod, opts)

