import re
from multiprocessing import Pool
from os import listdir
from os.path import join, exists

import numpy as np
import pandas as pd
from h5py import File


def dftb_parse(opts):
    res_dir = opts['res_dir']
    save_path = opts['save_path']
    n_worker = opts.get('n_worker', 1)

    # Parse in parallel
    mol_conf_names = sorted(listdir(res_dir))
    out_dirs = [join(res_dir, d + '/') for d in mol_conf_names]
    pool = Pool(processes=n_worker)
    out = pool.map_async(parse_single_conf, out_dirs)
    out.wait()

    # Collect parsed results in a DataFrame
    res = pd.DataFrame(out.get())
    res.sort_values(by=['mol', 'i_conf'], inplace=True)

    # Structure the DataFrame into ANI-like hierarchical dataset
    mols = res['mol'].unique()
    with File(save_path, 'w') as des:
        for mol in mols:
            g = des.create_group(mol)
            moldata = res.loc[res['mol'] == mol].iloc[:, 2:]
            for entry, data in moldata.items():
                g.create_dataset(entry, data=data)

    return res


def parse_single_conf(out_dir):
    mol, i_conf = out_dir.split('/')[-2].split('__')
    res = {'mol': mol,
           'i_conf': int(i_conf),
           'dftb_plus.elec_energy': np.nan,
           # The non-SCC energy plus other contributions to
           # the electronic energy (SCC, spin, ...)
           'dftb_plus.rep_energy': np.nan,
           # The pairwise contribution to the total energy
           'dftb_plus.disp_energy': np.nan,
           'dftb_plus.total_energy': np.nan,
           # Sum of electronic energy (elec + rep + disp)
           'dftb_plus.0K_energy': np.nan,
           # Estimated zero temperature energy if
           # at finite temperatures
           'dftb_plus.mermin_energy': np.nan,
           # U − TS, relevant free energy at finite temperatures
           'dftb_plus.force_related_energy': np.nan,
           # Free energy relevant to forces in the system
           'dftb_plus.scc_wall_time': np.nan,
           'dftb_plus.diag_wall_time': np.nan,
           'dftb_plus.total_wall_time': np.nan}
    STARTS = {'dftb_plus.elec_energy': 'Total Electronic energy',
              'dftb_plus.rep_energy': 'Repulsive energy',
              'dftb_plus.disp_energy': 'Dispersion energy',
              'dftb_plus.total_energy': 'Total energy',
              'dftb_plus.0K_energy': 'Extrapolated to 0',
              'dftb_plus.mermin_energy': 'Total Mermin free energy',
              'dftb_plus.force_related_energy': 'Force related energy',
              'dftb_plus.scc_wall_time': "SCC     ",
              'dftb_plus.diag_wall_time': "  Diagonalisation     ",
              'dftb_plus.total_wall_time': "Total     "}

    if exists(join(out_dir, 'failed')):
        return res

    # Parse detailed.out for energies
    with open(join(out_dir, 'detailed.out'), 'r') as out:
        for line in out.readlines():
            for entry, start in STARTS.items():
                if 'time' in entry:
                    continue
                if line.startswith(start):
                    try:
                        res[entry] = float(line.split()[-4])
                    except ValueError:
                        res[entry] = np.nan

    # Parse dftb.stdout for time
    with open(join(out_dir, 'dftb.stdout'), 'r') as out:
        for line in out.readlines():
            for entry, start in STARTS.items():
                if 'energy' in entry:
                    continue
                if line.startswith(start):
                    try:
                        split = re.findall("\[[^]]*]|\([^)]*\)|\"[^\"]*\"|\S+", line)
                        res[entry] = float(split[-2])
                    except ValueError:
                        res[entry] = np.nan

    return res
