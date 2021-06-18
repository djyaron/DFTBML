from .deprecated import *
from .fold import *
from sklearn.linear_model import LinearRegression
from .consts import CUTOFFS
from .util import count_n_heavy_atoms


def train_repulsive_model(dset, opts):
    r"""API for dftbtorch

    Args:
        dset: dict
            ANI-like data fold (nested dictionary)
            ================================= Data structure =================================
            {mol_name_0 (str):
                {'atomic_numbers': np.ndarray, shape=(n_atoms,),
                 'coordinates': np.ndarray, shape=(n_confs, n_atoms, 3),
                 'baseline': np.ndarray, shape=(n_confs,). E.g. 'dftb.elec_energy',
                 'target': np.ndarray, shape=(n_confs,). E.g. 'ccsd(t)_cbs.energy'},
             mol_name_1: {...},
             mol_name_2: {...},
             ...}
            ==================================================================================
        opts: dict
            'nknots': number of knots of the sparse model
            'deg': degree of splines of the sparse model
            'rmax': cut-off radii of the sparse model, e.g. 'short', 'medium', etc.
            'bconds': recommended value: 'vanishing'
            'shift': set to True to shift the targets using a linear model (linear shifter)
            'scale': scale the energies in data by the number of heavy atoms
            'atom_types': set to 'infer' to automatically determine the atom types from input
                          data fold. Or can be set to a SORTED array of atomic numbers
            'map_grid': recommended value: 500
            'constraint': 'monotonic' or 'convex'
            'pen_grid': recommended value: 500
            'n_worker': number of CPUs to use in parallel

    Returns:
        c: np.ndarray
            coefficients of the repulsive model
        loc: dict(str, np.ndarray)
            dictionary describing the locations of the content in c, whose keys include
              atom pair tuples ((1, 1), (1, 6), ...) and 'ref'
            Usage:
            c[loc[(1, 1)]] -> coefficients of H-H splines
            c[loc['ref']] -> coefficients of reference energies
        gammas: dict(str, dict)
            spline bases and atom counts of the repulsive model
            ================================= Data structure =================================
            {mol_name_0 (str):
                {'gammas': np.ndarray, shape=(n_confs, len(c))
                 'n_heavy_atoms': np.int64}
             mol_name_1: {...},
             mol_name_2: {...},
             ...,
             '_INFO': dict
                The last entry of gammas dictionary is always '_INFO', which contains the
                parameters used to generate gammas and to build corresponding dense model.
                Refer to generator_v0.py for more details.
                {'Zs': np.array
                 'Xis': np.array
                 'Nknots': np.array
                 'Degrees': np.array
                 'R_cutoffs': np.array
                 'Xknots': np.array}
            }
            ==================================================================================
            Usage:
            gammas[mol].dot(c) -> estimated repulsive energies of all the conformations
                                  of a molecule
            gammas[mol][conf_id].dot(c) -> estimated repulsive energy of a conformation
            gammas[mol][conf_id][loc[Z]].dot(c[loc[Z]]) -> estimated repulsive energy of
                                                           pairwise interaction Z of a
                                                           conformation
    """

    # Scale and shift energies if needed
    _dset = dset
    c_shifter = np.zeros(len(opts['atom_types']) + 1)
    if opts['scale']:
        _dset = scale_dset(_dset)
    if opts['shift']:
        _dset, c_shifter = shift_dset(_dset, opts['atom_types'])

    # Train repulsive model
    gammas = compute_gammas(_dset, opts)  # 50 knots, 5-th order dense model
    c, loc = train(_dset, gammas, opts)
    c[loc['ref']] += c_shifter

    # # Test: print training error
    # preds = get_predictions_from_dense(c, gammas=gammas, in_memory=True)
    # targets = get_targets_from_dataset(data=scale_dset(dset), in_memory=True)
    # errdict, err = compare_target_pred(targets, preds)
    # print(errdict['mae'] * HARTREE)

    return c, loc, gammas


def scale_dset(dset):
    sset = {}
    for mol, moldata in dset.items():
        _moldata = {}
        n_heavy_atoms = count_n_heavy_atoms(moldata['atomic_numbers'])
        for entry, data in moldata.items():
            if entry == 'target':
                _data = data * n_heavy_atoms
            else:
                _data = data
            _moldata[entry] = _data
        sset[mol] = _moldata
    return sset


def count_atomic_numbers(atomic_numbers, atom_types='infer'):
    r"""Count the number of atoms of each type in an array of atomic numbers"""
    _atypes = atom_types if atom_types != 'infer' else sorted(set(atomic_numbers))
    _anumbers = list(atomic_numbers)
    counts = [_anumbers.count(_atype) for _atype in _atypes]
    return _atypes, counts


def flatten_dset(dset, atom_types='infer'):
    r"""Create a flattened dataset including only atom counts and energies"""
    # Infer atom types from the input dataset
    _atypes = infer_atom_types(dset) if atom_types == 'infer' else atom_types

    # Flatten dataset
    fset = {_atype: [] for _atype in _atypes}
    fset.update({'baseline': [], 'target': []})

    for mol, moldata in dset.items():
        nconfs = len(moldata['target'])
        # Columns of atom types
        _, counts = count_atomic_numbers(moldata['atomic_numbers'], _atypes)
        for _atype, count in zip(_atypes, counts):
            fset[_atype].extend([count] * nconfs)
        # Columns of energies
        fset['baseline'].extend(moldata['baseline'])
        fset['target'].extend(moldata['target'])

    return pd.DataFrame(fset)


def shift_dset(dset, atom_types='infer'):
    r"""Shift baselines and targets in ANI-like data fold"""
    # Infer atom types from the input dataset
    _atypes = infer_atom_types(dset) if atom_types == 'infer' else atom_types

    # Fit a linear model to the difference between target and baseline
    fset = flatten_dset(dset, _atypes)
    X = fset[_atypes]
    y = fset['target'] - fset['baseline']

    shifter = LinearRegression()
    shifter.fit(X, y)

    # Create a shifted dataset
    sset = {}
    for mol, moldata in dset.items():
        sset[mol] = {'atomic_numbers': moldata['atomic_numbers'],
                     'coordinates': moldata['coordinates'],
                     'baseline': moldata['baseline']}
        # Shift targets
        _, counts = count_atomic_numbers(moldata['atomic_numbers'], _atypes)
        X = np.reshape(counts, (1, -1))
        pred = shifter.predict(X)
        sset[mol]['target'] = moldata['target'] - pred

    # concatenate intercept and coefficients of the shifter into c_shifter
    c_shifter = np.insert(shifter.coef_, 0, shifter.intercept_)

    return sset, c_shifter


def compute_gammas(dset, opts):
    # Generate gammas using data (50 knots, 5-th order dense model by default)
    Au = 'au' in opts['rmax']
    n_worker = opts.get('n_worker', 1)

    dense_nknots = 50
    dense_deg = 5
    dense_bconds = 'vanishing'
    dense_rmax = 'au_full' if Au else 'full'
    
    # import pdb; pdb.set_trace()

    g_mod = Generator(dset, dense_nknots, dense_deg, dense_rmax,
                      dense_bconds, in_memory=True, n_worker=n_worker)
    g_mod.init_model()
    return g_mod.gammas


def train(dset, gammas, opts):
    map_grid = opts.get('map_grid', 500)
    ptype = opts.get('constraint')
    pen_grid = opts.get('pen_grid', 500)
    # Create dense model
    dense_model_Z, dense_loss_func = create_dense(data=dset, gammas=gammas, in_memory=True)
    # Create sparse model
    Zs = dense_model_Z.Zs()
    nknots_Z = {Z: opts['nknots'] for Z in Zs}
    rmax_Z = CUTOFFS[opts['rmax']]
    xknots_Z = {Z: np.linspace(rmin, rmax, n)
                for (rmin, rmax), (Z, n) in zip(rmax_Z.values(), nknots_Z.items())}
    config_Z = {Z: {'xknots': xknots,
                    'deg': opts['deg'],
                    'bconds': opts['bconds']}
                for Z, xknots in xknots_Z.items()}
    sparse_models_only = {Z: SplineModel(config)
                          for Z, config in config_Z.items()}
    map_models, sparse_model_Z = map_linear_models_Z(sparse_models_only, dense_model_Z, map_grid)
    sparse_loss_func = sparse_loss_from_dense(map_models, dense_loss_func)
    solver = Solver(sparse_model_Z, sparse_loss_func, {ptype: None}, pen_grid)
    sparse_coef = solver.solve()
    c = dense_coefs_from_sparse(map_models, sparse_coef)
    loc = dense_model_Z.locs
    return c, loc


def infer_atom_types(dset):
    _atypes = set()
    for moldata in dset.values():
        _atypes.update(moldata['atomic_numbers'])
    _atypes = sorted(_atypes)
    return _atypes


if __name__ == '__main__':
    dset_path = '/home/francishe/Documents/DFTBrepulsive/example_dict.p'
    with open(dset_path, 'rb') as f:
        dset = pkl.load(f)

    # # Save dset as h5
    # sample_path = '/home/francishe/Documents/DFTBrepulsive/sample.h5'
    # with File(sample_path, 'w') as des:
    #     dset = scale_dset(dset)
    #     for mol, moldata in dset.items():
    #         g = des.create_group(mol)
    #         g.create_dataset('atomic_numbers', data=moldata['atomic_numbers'])
    #         g.create_dataset('coordinates', data=moldata['coordinates'])
    #         g.create_dataset('ccsd(t)_cbs.energy', data=moldata['target'])
    #         g.create_dataset('dftb_plus.elec_energy', data=moldata['baseline'])

    # # Load aed_1K dataset
    # src_path = '/home/francishe/Documents/DFTBrepulsive/Datasets/aed_1K.h5'
    # with File(src_path, 'r') as src:
    #     dset = {mol: {'atomic_numbers': moldata['atomic_numbers'][()],
    #                   'coordinates': moldata['coordinates'][()],
    #                   'target': moldata[TARGETS['fm']][()],
    #                   # See tests/driver__precision_test.py for details
    #                   'baseline': moldata[TARGETS['pf']][()] - moldata[TARGETS['pr']][()]}
    #             for mol, moldata in src.items()}

    opts = {'nknots': 50,
            'deg': 3,
            'rmax': 'short',
            'bconds': 'vanishing',
            'shift': True,
            'scale': True,  # scale 'target' by number of heavy atoms
            'atom_types': [1, 6, 7, 8],
            'map_grid': 500,
            'constraint': 'convex',
            'pen_grid': 500,
            'n_worker': 8}

    from util import Timer
    with Timer('driver'):
        c, loc, gammas = train_repulsive_model(dset, opts)
