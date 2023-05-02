# -*- coding: utf-8 -*-
"""
Created on Sun May 29 12:58:27 2022

@author: fhu14

A master dataset is generated as a list of molecule dictionaries containing 
the energy targets for both coupled-cluster (cc) and DFT (wt). The max_config
parameter is set to int(10E6) to ensure that we get all possible configurations
from the h5 dataset. 

The target dictionary is going to be ALL_EXPERIMENT_DSETS which is where 
everything will be saved. The target dictionaries and other constants are also going
to be fixed within this file so that we can generate the exact dataset that we want
"""

#%% Imports, definitions
from FoldManager import get_ani1data
import os, pickle
from .util import shuffle_dict

#These are the constants we are going to use to generate the master dataset
#   every time. Both the coupled-cluster and DFT energies are included for
#   convenience
allowed_Z = [1,6,7,8]
heavy_atoms = [1,2,3,4,5,6,7,8]
max_config = int(10E6)
# target = {"cc" : "cc", "wt" : "wt",
#        "dipole" : "wb97x_dz.dipole",
#        "charges" : "wb97x_dz.cm5_charges"}
#Target to use for comparing different energy targets only
#   this saves to the file full_energy_master_dset.p
target = {'dt': 'dftb.energy',  # Dftb Total
            'de': 'dftb.elec_energy',  # Dftb Electronic
            'dr': 'dftb.rep_energy',  # Dftb Repulsive
            'pt': 'dftb_plus.energy',  # dftb Plus Total
            'pe': 'dftb_plus.elec_energy',  # dftb Plus Electronic
            'pr': 'dftb_plus.rep_energy',  # dftb Plus Repulsive
            'hd': 'hf_dz.energy',  # Hf Dz
            'ht': 'hf_tz.energy',
            'hq': 'hf_qz.energy',
            'wd': 'wb97x_dz.energy',  # Wb97x Dz
            'wt': 'wb97x_tz.energy',
            'md': 'mp2_dz.energy',  # Mp2 Dz
            'mt': 'mp2_tz.energy',
            'mq': 'mp2_qz.energy',
            'td': 'tpno_ccsd(t)_dz.energy',  # Tpno Dz
            'nd': 'npno_ccsd(t)_dz.energy',  # Npno Dz
            'nt': 'npno_ccsd(t)_tz.energy',
            'cc': 'ccsd(t)_cbs.energy',
            'frc_tz' : 'wb97x_tz.forces',    #DFT TZ forces
            'frc_dz' : 'wb97x_dz.forces'}    #DFT DZ forces

ani1_path = "ANI-1ccx_clean_fullentry.h5"
exclude = []
DESTINATION = "ALL_EXPERIMENT_DSETS"

#%% Code behind

def generate_master_dset() -> None:
    r"""Generates the master dataset with both 'cc' and 'wt' energy targets
        that are then manipulated for later use
    """
    dataset = get_ani1data(allowed_Z, heavy_atoms, max_config, target, ani1_path,
                           exclude)
    full_path = os.path.join(os.getcwd(), DESTINATION, "full_energy_master_dset.p")
    with open(full_path, 'wb') as handle:
        pickle.dump(dataset, handle)
    print("Master dataset saved with both cc and wt energy targets")
    
def convert_dataset_to_dictionary() -> None:
    r"""Converts the full_master_dset.p file into a dictionary that maps
        each empirical formula to a list of all the molecules that contain 
        that empirical formula
    
    Notes: The dictionary looks something like this:
        {emp_form_1: [mol1, mol2, mol3, ...],
         emp_form_2 : [mol1, mol2, mol3, ...], ...}
        This representation will allow for more precise control over the 
        size of the dataset as well as the uniformity of the dataset's 
        construction
    """
    full_path = os.path.join(os.getcwd(), DESTINATION, "full_master_dset.p")
    all_molecules = pickle.load(open(full_path, 'rb'))
    master_dictionary = {}
    all_names = list(set([mol['name'] for mol in all_molecules]))
    #Initialize the dictionary with empty lists
    for name in all_names:
        master_dictionary[name] = []
    for molecule in all_molecules:
        curr_name = molecule['name']
        master_dictionary[curr_name].append(molecule)
    #Now check some invariants
    tot_mol_count = 0
    for formula in master_dictionary:
        tot_mol_count += len(master_dictionary[formula])
    assert(tot_mol_count == len(all_molecules))
    #Now shuffle and save the dictionary
    shuffle_dict(master_dictionary)
    save_path = os.path.join(os.getcwd(), DESTINATION, "full_master_dset_dict.p")
    with open(save_path, 'wb') as handle:
        pickle.dump(master_dictionary, handle)
    print("Master dset converted to dictionary form and saved")
    
