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
    h5set_path = '/export/home/hanqingh/Documents/DFTBrepulsive/Datasets/aed_1k.h5'
    # h5set_path = '/export/home/hanqingh/Documents/DFTBrepulsive/Datasets/Au_energy_clean_dispersion.h5'

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
        mod.fit(dset, target=target, gammas=None, shift=True, n_worker=n_cpu)
    with Timer('RepulsiveModel predicting'):
        pred = mod.predict(dset, mod.gammas, n_worker=n_cpu)
    # MAE
    err = Dataset.compare(dset, target,
                          pred, 'prediction',
                          metric='mae', scale=False)
    print(f"Training MAE by RepulsiveModel: {err * HARTREE:.3f} kcal/mol")
    ## Error per heavy atom
    err = Dataset.compare(dset, target,
                          pred, 'prediction',
                          metric='mae', scale=True)
    print(f"Training MAE per heavy atom by RepulsiveModel: {err * HARTREE:.3f} kcal/mol")
    # RMS
    err = Dataset.compare(dset, target,
                          pred, 'prediction',
                          metric='rms', scale=False)
    print(f"Training RMS by RepulsiveModel: {err * HARTREE:.3f} kcal/mol")
    ## Error per heavy atom
    err = Dataset.compare(dset, target,
                          pred, 'prediction',
                          metric='rms', scale=True)
    print(f"Training RMS per heavy atom by RepulsiveModel: {err * HARTREE:.3f} kcal/mol")

    # Plot splines and save spline values to generate SKFs
    ngrid = 500
    res_path = f"/export/home/hanqingh/Documents/DFTBrepulsive/xydata.pkl"
    save_dir = f"/export/home/hanqingh/Documents/DFTBrepulsive/SKF/{opts['cutoff']}/"
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
                 'dftb_dir': '/export/home/hanqingh/opt/dftb+/',
                 'dataset_path': h5set_path,

                 # 'save_dir': '/scratch/dftb+_res/',
                 'save_dir': '/scratch/dftb+_res_1K/',

                 'skf_dir': '/export/home/hanqingh/opt/dftb+/slakos/auorg-1-1/',
                 # 'skf_dir': '/export/home/hanqingh/Documents/DFTBrepulsive/SKF/a1k/',
                 }
    parse_opts = {'n_worker': n_cpu,
                  'res_dir': calc_opts['save_dir'],
                  'save_path_h5': '/export/home/hanqingh/Documents/DFTBrepulsive/a1k_auorg.h5',
                  # 'save_path_h5': '/export/home/hanqingh/Documents/DFTBrepulsive/a1k_+2.h5',
                  'save_path_pkl': '/export/home/hanqingh/Documents/DFTBrepulsive/a1k_auorg.pkl',
                  # 'save_path_h5': '/export/home/hanqingh/Documents/DFTBrepulsive/a1k_+2.pkl',
                  }

    with Timer("DFTB+ calculation"):
        dftb_calc(calc_opts)
    with Timer("DFTB+ parsing"):
        res = dftb_parse(parse_opts)

    # Examine DFTB+ output
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
