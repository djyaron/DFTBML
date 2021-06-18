import os
from multiprocessing import cpu_count

import numpy as np
from h5py import File

from consts import CUTOFFS, HARTREE
from dataset import Dataset
from dftb_calc import dftb_calc
from dftb_parse import dftb_parse
from model import RepulsiveModel
from options import Options
from shifter import Shifter
from skfwriter import main as skfwriter_main
from util import Timer, path_check
import pickle as pkl

n_cpu = cpu_count()
os.environ['OMP_NUM_THREADS'] = '1'  # Single thread for each calculation

if __name__ == '__main__':

    # Model training and testing
    # h5set_path = '/home/francishe/Documents/DFTBrepulsive/Datasets/aed_1K.h5'
    h5set_path = '/home/francishe/Documents/DFTBrepulsive/Datasets/Au_energy_clean_dispersion.h5'

    target = 'fm-pf+pr'

    opts = Options({'nknots': 25,
                    'cutoff': 'au_full',
                    'deg': 3,
                    'bconds': 'vanishing',
                    'constr': '+2',
                    'solver': 'cvxopt',
                    'ref': 'full'})

    dset = Dataset(File(h5set_path, 'r'))
    mod = RepulsiveModel(opts.convert(dset.Zs()))
    with Timer('RepulsiveModel training'):
        mod.fit(dset, target='fm-pf+pr', gammas=None, shift=True, n_worker=n_cpu)
    with Timer('RepulsiveModel predicting'):
        pred = mod.predict(dset, mod.gammas, n_worker=n_cpu)
    mae = Dataset.compare(mod.shifter.shift(dset, target), target,
                          pred, 'prediction',
                          metric='mae')
    print(f"Training error by RepulsiveModel: {mae * HARTREE:.3f} kcal/mol")

    # Plot splines and save spline values to generate SKFs
    ngrid = 500
    res_path = f"/home/francishe/Documents/DFTBrepulsive/xydata.pkl"
    save_dir = f"/home/francishe/Documents/DFTBrepulsive/SKF/{opts['cutoff']}/"
    path_check(save_dir)

    grid = {Z: np.linspace(c[0], c[1], ngrid) for Z, c in CUTOFFS[opts['cutoff']].items()}
    res = mod.plot(grid)
    # WARNING: outdated res structure for compatibility with skfwriter
    res = {'all_xydata': {'sparse_xydata': [[[res]]]}}
    assert isinstance(res['all_xydata']['sparse_xydata'][0][0][0], dict)
    pkl.dump(res, open(res_path, 'wb'))
    skfwriter_main(res_path, save_dir)

    # Run DFTB+ with generated SKFs and parse the output
    calc_opts = {'n_worker': n_cpu,
                 'FermiTemp': 0.1,  # in electronvolts (0.1 eV ~ 1100 K)
                 'ShellResolvedSCC': True,
                 'dftb_dir': '/home/francishe/opt/dftb+/',

                 # 'dataset_path': '/home/francishe/Documents/DFTBrepulsive/Datasets/Au_energy_clean_dispersion.h5',
                 'dataset_path': h5set_path,
                 # 'dataset_path': '/home/francishe/Documents/DFTBrepulsive/Datasets/n2.h5',

                 'save_dir': '/home/francishe/Documents/DFTBrepulsive/dftb+_res/',
                 # 'save_dir': '/home/francishe/Documents/DFTBrepulsive/dftb+_res_1K/',
                 # 'save_dir': '/home/francishe/Documents/DFTBrepulsive/res_n2/',

                 # 'skf_dir': '/home/francishe/opt/dftb+/slakos/auorg-1-1/',
                 # 'skf_dir': '/home/francishe/Documents/DFTBrepulsive/SKF/aed_convex/',
                 'skf_dir': '/home/francishe/Documents/DFTBrepulsive/SKF/a1k/',
                 }
    parse_opts = {'n_worker': n_cpu,
                  'res_dir': calc_opts['save_dir'],

                  # 'save_path_h5': '/home/francishe/Documents/DFTBrepulsive/Datasets/aec_dftb+.h5',
                  'save_path_h5': '/home/francishe/Documents/DFTBrepulsive/Datasets/a1k_convex.h5',
                  # 'save_path_h5': '/home/francishe/Documents/DFTBrepulsive/Datasets/n2_tb.h5',

                  # 'save_path_pkl': '/home/francishe/Documents/DFTBrepulsive/Datasets/aec_dftb+.pkl',
                  'save_path_pkl': '/home/francishe/Documents/DFTBrepulsive/Datasets/a1k_convex.pkl',
                  # 'save_path_pkl': '/home/francishe/Documents/DFTBrepulsive/Datasets/n2_tb.pkl',
                  }

    with Timer("DFTB+ calculation"):
        dftb_calc(calc_opts)
    with Timer("DFTB+ parsing"):
        res = dftb_parse(parse_opts)

    # Examining DFTB+ output
    rset = Dataset(File(parse_opts['save_path_h5'], 'r'), conf_entry='dftb_plus.rep_energy', fixed_entry=())
    tset = dset.extract('coordinates', entries=('atomic_numbers',))
    rset = Dataset.merge(rset, tset)
    rset.conf_entry = 'coordinates'
    rset.fixed_entry = ('atomic_numbers',)

    _dset = mod.shifter.shift(dset, 'fm')
    # WARNING: Hard-coded ref location (const may not exist)
    ref_shifter = Shifter()
    ref_shifter.shifter.coef_ = mod.coef[mod.loc['ref']][1:]
    ref_shifter.shifter.intercept_ = mod.coef[mod.loc['const']]
    ref_shifter.atypes = mod.atypes
    _dset = ref_shifter.shift(_dset, 'fm')
    mae = Dataset.compare(_dset, 'fm',
                          rset, 'pf',
                          metric='mae')
    print(f"Training error by DFTB+: {mae * HARTREE:.3f} kcal/mol")
