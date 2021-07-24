import pickle as pkl

from .dataset import Dataset
from .generator import Generator
from .model import RepulsiveModel
from .options import Options
from .util import Zs_from_opts, count_n_heavy_atoms


def compute_gammas(dset: dict, opts: dict, return_gammas: bool = False):
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
        return_gammas (bool): Whether gammas should be returned or not. If False,
            the gammas are saved to pickle format by invoking the to_pickle() method.
            otherwise, the gammas are returned for the user to handle explicitly.

    Note: Call this function in precomputation over the entire dataset
    """
    n_worker = opts['training_settings']['n_worker']
    gammas_path = opts['repulsive_settings']['gammas_path']
    _dset = Dataset(dset, conf_entry='coordinates', fixed_entry=('atomic_numbers',))
    _opts = Options(opts['repulsive_settings']['opts'])
    _Zs = Zs_from_opts(opts)

    _gen = Generator(_dset, _opts.convert(_Zs))
    gammas = _gen.gammas(n_worker=n_worker)
    if (not return_gammas):
        gammas.to_pickle(gammas_path)
    elif return_gammas:
        return gammas


def train_repulsive_model(dset: dict, opts: dict, gammas = None) -> RepulsiveModel:
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
        gammas: Gammas (optional)
            Gammas object being loaded in. Defaults to None.

    Returns:
        mod: model.RepulsiveModel

    Notes:
        The input target energies will be scaled up by the number of heavy atoms before being fed to RepulsiveModel

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
    if gammas is None:
        _gammas = pkl.load(open(gammas_path, 'rb'))
    else:
        _gammas = gammas
    mod = RepulsiveModel(_opts.convert(_Zs))
    mod.fit(_dset, target='target', gammas=_gammas, shift=True, n_worker=n_worker)
    return mod


def repulsive_model_predict(mod: RepulsiveModel, dset: dict, opts: dict, gammas = None) -> dict:
    r"""Make predictions using trained repulsive model

    Args:
        mod: RepulsiveModel
            Trained repulsive model
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
            Highest level dictionary of options and hyperparameters
        gammas: Gammas (optional)
            Gammas object being loaded in. Defaults to None.

    Returns:
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

    Notes: The prediction in the output is scaled down by the number of heavy atoms
           target = ground_truth - dftbtorch_elec
           LOSS = (ground_truth - dftbtorch_elec)/n_heavy_atom
           for i in epochs:
             for j in data_batch:
                 gradient_descent
           Update_charges
           DFTBrepulsive_rep
    """
    n_worker = opts['training_settings']['n_worker']
    gammas_path = opts['repulsive_settings']['gammas_path']
    # Note: Here the energies are not scaled down as they are not used during predicting
    _dset = Dataset(dset, conf_entry='coordinates', fixed_entry=('atomic_numbers',))
    if gammas is None:
        _gammas = pkl.load(open(gammas_path, 'rb'))
    else:
        _gammas = gammas
    _preds = mod.predict(_dset, gammas=_gammas, n_worker=n_worker)
    preds = {mol: {'atomic_numbers': moldata['atomic_numbers'],
                   'prediction': moldata['prediction'] / count_n_heavy_atoms(moldata['atomic_numbers'])}
             for mol, moldata in _preds.items()}
    return preds


if __name__ == '__main__':
    from h5py import File

    h5set_path = '/export/home/hanqingh/Documents/DFTBrepulsive/Datasets/aed_1k.h5'
    with File(h5set_path, 'r') as h5set:
        dset = Dataset(h5set)

    eset = dset.extract('coordinates').dset
    target = 'fm-pf+pr'
    _tset = dset.extract(target, entries=('atomic_numbers', 'coordinates'))
    tset = {mol: {'atomic_numbers': moldata['atomic_numbers'],
                  'coordinates': moldata['coordinates'],
                  'target': moldata[target] / count_n_heavy_atoms(moldata['atomic_numbers'])}
            for mol, moldata in _tset.items()}

    opts = {"model_settings": {"low_end_correction_dict": {(1, 1): 1.00,
                                                           (6, 6): 1.04,
                                                           (1, 6): 0.602,
                                                           (7, 7): 0.986,
                                                           (6, 7): 0.948,
                                                           (1, 7): 0.573,
                                                           (1, 8): 0.599,
                                                           (6, 8): 1.005,
                                                           (7, 8): 0.933,
                                                           (8, 8): 1.062,
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
    mod = train_repulsive_model(tset, opts)
    preds = repulsive_model_predict(mod, tset, opts)
    # See skf.py for the instructions of creating SKFs using model xydata
