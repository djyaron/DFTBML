from h5py import File
import h5py
import numpy as np
from os import walk
from os.path import join



if __name__ == "__main__":

    raw_dir = '/home/francishe/Documents/DFTBrepulsive/Datasets/raw/cutdown'
    raw_2 = '/home/francishe/Documents/DFTBrepulsive/Datasets/raw/Molecule_Au2_cut_down.hdf5'
    combined_path = '/home/francishe/Documents/DFTBrepulsive/Datasets/Au_combined.h5'
    af_path = '/home/francishe/Documents/DFTBrepulsive/Datasets/Au_full.h5'
    ae_path = '/home/francishe/Documents/DFTBrepulsive/Datasets/Au_energy.h5'
    aec_path = '/home/francishe/Documents/DFTBrepulsive/Datasets/Au_energy_clean.h5'
    aed_path = '/home/francishe/Documents/DFTBrepulsive/Datasets/Au_energy_clean_dispersion.h5'
    sub_path = '/home/francishe/Documents/DFTBrepulsive/Datasets/aed_1K.h5'

    # Combine all the datasets
    fns = tuple(tuple(walk(raw_dir))[0][2])  # file names of Au4 ~ Au10 datasets
    fps = [raw_2]  # file path of Au2 dataset
    fps.extend([join(raw_dir, fn) for fn in fns])  # file paths of Au4 ~ Au10 datasets

    with File(combined_path, 'w') as des:
        for fp in fps:
            with File(fp, 'r') as src:
                for mol in src.keys():
                    src.copy(mol, des)

    # Extract all the entries and combine each entry by molecule
    with File(combined_path, 'r') as src, File(af_path, 'w') as des:
        mols = tuple(sorted(src.keys()))
        for mol in mols:
            molgroup = src[mol]
            ## Sort conformations by their id (starting from 1)
            i_confs = sorted(int(mol_conf.rsplit('_', 1)[-1]) for mol_conf in molgroup.keys())
            confs = tuple(f"{mol}_{i_conf}" for i_conf in i_confs)
            moldata = {'E_mermin': [],
                       'E_total': [],
                       'HOMO': [],
                       'LUMO': [],
                       'Occ_mulliken_l': [],
                       'Q_mulliken': [],
                       'atomic_numbers': None,
                       'Edftb_elec': [],
                       'Edftb_rep': [],
                       'dipole_vector': [],
                       'epsilons': [],
                       'positions': []}
            for conf in confs:
                confdata = molgroup[conf]
                ## Miscellaneous data
                moldata['E_mermin'].append(confdata['E_mermin'][()])
                moldata['E_total'].append(confdata['E_total'][()])
                moldata['HOMO'].append(confdata['HOMO'][()])
                moldata['LUMO'].append(confdata['LUMO'][()])
                moldata['Q_mulliken'].append(confdata['Q_mulliken'][()])
                moldata['Edftb_elec'].append(confdata['dftbd']['Edftb_elec'][()])
                moldata['Edftb_rep'].append(confdata['dftbd']['Edftb_rep'][()])
                moldata['dipole_vector'].append(confdata['dipole_vector'][()])
                moldata['epsilons'].append(confdata['epsilons'][()])
                moldata['positions'].append(confdata['positions'][()])
                ## Atomic numbers
                if moldata['atomic_numbers'] is None:
                    moldata['atomic_numbers'] = confdata['atomic_numbers'][()]
                elif 0 in moldata['atomic_numbers']:
                    # Update positions in moldata if positions in confdata is valid
                    if 0 not in confdata['atomic_numbers'][()]:
                        moldata['atomic_numbers'] = confdata['atomic_numbers'][()]
                ## Mulliken occupancies
                occ = confdata['Occ_mulliken_l']
                ### Sort occupancies by the id of corresponding atom
                i_atoms = sorted(int(atoms.rsplit('_', 1)[-1]) for atoms in occ.keys())
                atoms = tuple(f"atom_{i_atom}" for i_atom in i_atoms)
                ### Pack Mulliken occupancies in an np.ndarray with shape (n_atoms, 3)
                ### where 3 corresponds to s, p and d orbitals
                occdata = np.full((len(atoms), 3), fill_value=np.nan, dtype='float')
                for i_atom, atom in enumerate(atoms):
                    n_orbitals = len(occ[atom])
                    occdata[i_atom, :n_orbitals] = occ[atom][()]
                moldata['Occ_mulliken_l'].append(occdata)
            ## Stack the data in each entry and write to file
            g = des.create_group(mol)
            for entry, data in moldata.items():
                if entry == 'atomic_numbers':
                    g.create_dataset(entry, data=data)
                elif entry in ('E_mermin', 'E_total', 'HOMO', 'LUMO', 'Edftb_elec', 'E_dftb_rep'):
                    g.create_dataset(entry, data=np.array(data))
                else:
                    g.create_dataset(entry, data=np.stack(data, axis=0))

    # Extract energy entries from Au_full and remove molecules with invalid atomic numbers
    with File(af_path, 'r') as src, File(ae_path, 'w') as des:
        for mol, moldata in src.items():
            if 0 in moldata['atomic_numbers']:
                continue
            g = des.create_group(mol)
            g.create_dataset('atomic_numbers', data=moldata['atomic_numbers'])
            g.create_dataset('coordinates', data=moldata['positions'])
            g.create_dataset('fhi_aims_md.mermin_energy', data=moldata['E_mermin'])
            g.create_dataset('fhi_aims_md.total_energy', data=moldata['E_total'])
            g.create_dataset('fhi_aims_md.homo_energy', data=moldata['HOMO'])
            g.create_dataset('fhi_aims_md.lumo_energy', data=moldata['LUMO'])
            ## DFTBd energies
            elec = moldata['Edftb_elec'][()]
            rep = moldata['Edftb_rep'][()]
            g.create_dataset('dftb.elec_mermin_energy', data=elec)
            g.create_dataset('dftb.rep_energy', data=rep)
            g.create_dataset('dftb.mermin_free_energy', data=elec+rep)

    # Clean Au_energy
    # There are still invalid (non-negative) entries in dftb.total_mermin_energy
    # But we will re-compute energies with DFTB+ and clean the dataset later
    with File(ae_path, 'r') as src, File(aec_path, 'w') as des:
        for mol, moldata in src.items():
            ## Create a mask to filter out NANs
            masks = [~np.isnan(data) for entry, data in moldata.items()
                     if entry not in ('atomic_numbers', 'coordinates')]
            mask = np.logical_and.reduce(masks)
            ### Skip current molecule if none of its conformations is valid
            if mask.sum() == 0:
                continue
            ## Copy valid conformations into Au_energy_clean
            g = des.create_group(mol)
            for entry, data in moldata.items():
                if entry == 'atomic_numbers':
                    g.create_dataset(entry, data=data)
                elif entry == 'coordinates':
                    g.create_dataset(entry, data=data[mask, :, :])
                else:  ### Energy entries are 1-D
                    g.create_dataset(entry, data=data[mask])

    # Run DFTB+ on Au_energy_clean and integrate the results into Au_energy_clean_dispersion
    ## Run DFTB+ and parse output
    from dftb_calc import dftb_calc
    from dftb_parse import dftb_parse
    from multiprocessing import cpu_count
    from os import environ
    from util import Timer

    opts = {'dataset_path': '/home/francishe/Documents/DFTBrepulsive/Datasets/Au_energy_clean.h5',
            'save_dir': '/home/francishe/Documents/DFTBrepulsive/dftb+_res/',
            'dftb_dir': '/home/francishe/opt/dftb+/',
            'skf_dir': '/home/francishe/opt/dftb+/slakos/auorg-1-1/',
            'n_worker': cpu_count(),
            'FermiTemp': 0.1,  # in electronvolts (0.1 eV ~ 1100 K)
            'ShellResolvedSCC': True,
            'res_dir': '/home/francishe/Documents/DFTBrepulsive/dftb+_res/',
            'save_path_h5': '/home/francishe/Documents/DFTBrepulsive/Datasets/aec_dftb+.h5',
            'save_path_pkl': '/home/francishe/Documents/DFTBrepulsive/Datasets/aec_dftb+.pkl'}
    environ['OMP_NUM_THREADS'] = '1'  # Single thread for each calculation

    with Timer("DFTB+"):
        dftb_calc(opts)
        dftb_parse(opts)

    ## Integrate results into Au_energy_clean_dispersion
    with File(aec_path, 'r') as src, File(aed_path, 'w') as des:
        for mol in src.keys():
            src.copy(mol, des)
    with File(opts['save_path_h5'], 'r') as src, File(aed_path, 'r+') as des:
        for mol, moldata in src.items():
            g = des[mol]
            for entry, data in moldata.items():
                if 'energy' in entry:
                    g.create_dataset(entry, data=data)

    ## Clean Au_energy_clean_dispersion
    with File(aed_path, 'r+') as des:
        invalid_mols = []
        for mol, moldata in des.items():
            ## Create a mask to filter out NANs
            masks = [~np.isnan(data) for entry, data in moldata.items()
                     if entry not in ('atomic_numbers', 'coordinates')]
            # ## Filter out non-negative energies (?)
            # masks.extend([data[()] <= 0 for entry, data in moldata.items()
            #               if entry not in ('atomic_numbers', 'coordinates')])
            mask = np.logical_and.reduce(masks)
            ### Skip current molecule if none of its conformations is valid
            if mask.sum() == 0:
                invalid_mols.append(mol)
                continue
            ## Filter out invalid conformations
            g = des[mol]
            for entry in g.keys():
                if entry == 'atomic_numbers':
                    continue
                elif entry == 'coordinates':
                    data = g[entry][mask, :, :]
                    del g[entry]
                    g.create_dataset(entry, data=data)
                else:
                    data = g[entry][mask]
                    del g[entry]
                    g.create_dataset(entry, data=data)
        for mol in invalid_mols:
            del des[mol]

    # Check each dataset
    from fold import Fold
    for fp in (ae_path, aec_path, aed_path):
        with File(fp, 'r') as dataset:
            fd = Fold.from_dataset(dataset)
            print(f"{fd.nmols()} molecules, {fd.nconfs()} conformations")

    # Check which conformations were removed
    with File(aec_path, 'r') as src, File(aed_path, 'r') as des:
        fd_src = Fold.from_dataset(src)
        fd_des = Fold.from_dataset(des)
        fd_diff = fd_src - fd_des
        print(f"{fd_diff.nmols()} molecules, {fd_diff.nconfs()} conformations were removed")

    # Sample a subset with ~1000 conformations
    with File(aed_path, 'r') as src, File(sub_path, 'w') as des:
        for mol, moldata in src.items():
            g = des.create_group(mol)
            n_confs = len(moldata['coordinates'])
            choices = np.sort(np.random.choice(n_confs, 2, replace=False))
            for entry, data in moldata.items():
                if entry == 'atomic_numbers':
                    g.create_dataset(entry, data=data)
                elif entry == 'coordinates':
                    g.create_dataset(entry, data=data[choices, :, :])
                else:
                    g.create_dataset(entry, data=data[choices])
