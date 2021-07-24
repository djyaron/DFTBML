import os
import re
import time
from multiprocessing import cpu_count

import numpy as np
from h5py import File

from consts import CUTOFFS, HARTREE
from dataset import Dataset
from dftb_calc import dftb_calc
from dftb_parse import dftb_parse
from model import RepulsiveModel
from options import Options
from skf import SKFSet, SKFBlockCreator
from util import Timer

n_cpu = cpu_count()
os.environ['OMP_NUM_THREADS'] = '1'  # Single thread for each calculation
timestamp = time.time()  # use a timestamp to create unique paths and filenames


def compare_multiple_metrics(dset, target, pred):
    # MAE
    err = Dataset.compare(dset, target,
                          pred, 'prediction',
                          metric='mae', scale=False)
    print(f"Training MAE: {err * HARTREE:.3f} kcal/mol")
    # MAE per heavy atom
    err = Dataset.compare(dset, target,
                          pred, 'prediction',
                          metric='mae', scale=True)
    print(f"Training MAE per heavy atom: {err * HARTREE:.3f} kcal/mol")
    # RMS
    err = Dataset.compare(dset, target,
                          pred, 'prediction',
                          metric='rms', scale=False)
    print(f"Training RMS: {err * HARTREE:.3f} kcal/mol")
    # RMS per heavy atom
    err = Dataset.compare(dset, target,
                          pred, 'prediction',
                          metric='rms', scale=True)
    print(f"Training RMS per heavy atoms: {err * HARTREE:.3f} kcal/mol")


def Au_judge(h5set_path):
    r"""Is this an Au dataset?"""
    filename = os.path.splitext(os.path.split(h5set_path)[-1])[0]
    if re.match(r'(?i)^au', filename):
        return True
    elif re.match(r'(?i)^ani', filename):
        return False
    elif filename.startswith('a'):
        return True
    elif filename.startswith('c'):
        return False
    else:
        raise ValueError("Dataset type not recognized")


if __name__ == '__main__':

    # Model training and testing
    h5set_path = '/export/home/hanqingh/Documents/DFTBrepulsive/Datasets/aed_1k.h5'
    # h5set_path = '/export/home/hanqingh/Documents/DFTBrepulsive/Datasets/ccf_2k.h5'
    # h5set_path = '/export/home/hanqingh/Documents/DFTBrepulsive/Datasets/Au_energy_clean_dispersion.h5'
    # h5set_path = '/export/home/hanqingh/Documents/DFTBrepulsive/Datasets/ANI-1ccx_clean_fullentry.h5'

    Au = Au_judge(h5set_path)

    target = 'fm-pf+pr' if Au else 'cc-de'  # WARNING: targets are hardcoded in "Examine DFTB+ results" section

    opts = Options({'nknots': 25,
                    'cutoff': 'au_full' if Au else 'full',
                    'deg': 3,
                    'bconds': 'vanishing',
                    'constr': '+2',
                    'solver': 'cvxopt',
                    'ref': 'full'})

    with File(h5set_path, 'r') as h5set:
        dset = Dataset(h5set)
    mod = RepulsiveModel(opts.convert(dset.Zs()))
    with Timer('RepulsiveModel training'):
        mod.fit(dset, target=target, gammas=None, shift=True, n_worker=n_cpu)
    with Timer('RepulsiveModel predicting'):
        pred = mod.predict(dset, mod.gammas, n_worker=n_cpu)

    # Print errors in multiple metrics
    compare_multiple_metrics(dset, target, pred)

    # Generate SKFs
    skf_dir = f"/export/home/hanqingh/Documents/DFTBrepulsive/SKF/SKF_{opts['cutoff']}_{timestamp}/"
    ref_skf_root = f'./slakos/'
    ref_skf_set = 'auorg-1-1' if Au else 'mio-1-1'

    # Create spline block
    creator = SKFBlockCreator()
    # Generate model xydata
    spl_ngrid = 500
    spl_grid = {Z: np.linspace(c[0], c[1], spl_ngrid) for Z, c in CUTOFFS[opts['cutoff']].items()}
    xydata = mod.create_xydata(spl_grid, expand=True)

    # Replace the spline block in auorg-1-1 or mio-1-1 but keep the exponential coefficients
    skfset = SKFSet.from_dir(os.path.join(ref_skf_root, ref_skf_set))
    for Z in skfset.Zs():
        # Remove SKFs for interactions not included in xydata
        if Z not in xydata.keys():
            del skfset[Z]
            continue
        # Incorporate splines from xydata but keep the exponential coefficients
        exp_coef = skfset[Z].exp_coef.values.flatten()
        spline_block = creator.create_spline_block(exp_coef, xydata[Z])
        skfset[Z]['spline'] = spline_block
        # Remove the documentation
        skfset[Z]['doc'] = {'doc': []}
    # Save SKFSet to file
    skfset.to_file(skf_dir)

    # Run DFTB+ with generated SKFs and parse the output
    calc_opts = {'dftb_dir': '/export/home/hanqingh/opt/dftb+/',
                 'skf_dir': skf_dir,
                 'save_dir': f'/scratch/dftb+_res_{timestamp}/',
                 'dataset_path': h5set_path,

                 'FermiTemp': 0.1,  # in electronvolts (0.1 eV ~ 1100 K)
                 'ShellResolvedSCC': True,

                 'n_worker': n_cpu}

    parse_opts = {'res_dir': calc_opts['save_dir'],
                  'save_path': f'/export/home/hanqingh/Documents/DFTBrepulsive/dftb+_res_{timestamp}.h5',

                  'n_worker': n_cpu}

    with Timer("DFTB+ calculation"):
        dftb_calc(calc_opts)
    with Timer("DFTB+ parsing"):
        res = dftb_parse(parse_opts)

    # Examine DFTB+ results
    # Load DFTB+ result dataset and integrate with corresponding atomic numbers and coordinates
    with File(parse_opts['save_path'], 'r') as res_h5set:
        rset = Dataset(res_h5set, conf_entry='dftb_plus.rep_energy', fixed_entry=())
    rset = Dataset.merge(rset, dset.extract('coordinates', entries=('atomic_numbers',)))
    rset.conf_entry = 'coordinates'
    rset.fixed_entry = ('atomic_numbers',)
    # Shift the result dataset with (linear) shifters trained to reference energies
    rset = mod.ref_shift(rset, 'pf' if Au else 'pe+pr', mode='+')

    # Compared the shifted result dataset with the target
    mae = Dataset.compare(dset, 'fm' if Au else 'cc',
                          rset, 'pf' if Au else 'pe+pr',
                          metric='mae')
    print(f"Training MAE by DFTB+: {mae * HARTREE:.3f} kcal/mol")
