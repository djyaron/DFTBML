# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 14:49:03 2020

@author: Frank

Create the dataset from "ANI-1ccx_clean_shifted.h5" file stored in data folder. Fetch the target energies using the 
get_targets_from_h5_file in SplineModel_v3.py. For use with dftb_layer_splines_1.py.

WIP
"""
import numpy as np
import random
import torch
torch.set_printoptions(precision = 10)
from geometry import Geometry
import pickle
from h5py import File
import os, os.path

def get_data_type(specs):
    if not isinstance(specs,list):
        specs = [specs]
    ANI1TYPES = {'dt': 'dftb.energy',  # Dftb Total
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
                'cc': 'ccsd(t)_cbs.energy'}
    res = []
    for spec in specs:
        if spec in ANI1TYPES.keys():
            res.append( ANI1TYPES[spec] )
        elif spec in ANI1TYPES.values():
            res.append( spec )
        else:
            res.append(spec) #To handle additional things
    return res

def get_targets_from_h5file(data_specs, ani1_path, exclude = None):
    # if target_type is a list of length 2, the target is the difference
    #   target = target_type[0] - target_type[1]
    if exclude == None:
        exclude = dict()
    dtypes = get_data_type(data_specs)
    target_molecs = dict()
    with File(ani1_path, 'r') as ani1data:
        for mol, gdata in ani1data.items():
            moldata = [gdata[x][()] for x in dtypes]
            if mol in exclude.keys():
                if exclude[mol] == 'all':
                    continue
                moldata = [np.delete(x, exclude[mol],0) for x in moldata]     
            if len(moldata) == 1:
                target_molecs[mol] = moldata[0]
            else:
                target_molecs[mol] = [moldata[0] - moldata[1]]
                target_molecs[mol] += moldata[2:]
    return target_molecs

# x, y = None, None
# with File(data_file_path, 'r') as ani1data_shifted:
#     x = [mol for mol, _ in ani1data_shifted.items()]
# with File(test_file_path2, 'r') as ani1data_full:
#     y = [mol for mol, _ in ani1data_full.items()]

# print(f"The arrays are equal: {x == y}")

# molec_energy_targets = get_targets_from_dataset(energy_targets, data_file_path)
# geom_targets = get_targets_from_dataset(geometry_targets, geom_file_path, subtract = False)

def choose_molecule_subset(full_dataset, num_in_subset):
    '''
    Choose a subset of molecules to work with from the total dataset. Every entry in the dataset should have
    an array of energies, a 3D array of the coordinates, and a final array of the atomic numbers.
    The number of geometries should match with the length of the energy array
    and the dimensions of each 2D array which describes a geometry should match with the
    atomic numbers
    
    Returns both the subset dictionary and the molecules contained (keys)
    '''
    keys = list(full_dataset.keys())
    chosen = random.sample(keys, num_in_subset)
    subset_dictionary = dict()
    for chosen_key in chosen:
        subset_dictionary[chosen_key] = full_dataset[chosen_key]
    return subset_dictionary, chosen

def choose_data_subset(molecule_data_subset, num_per_molecule, molecs_to_use = None, 
                       safety_check = True):
    '''
    Picks a series of points from each molecule (if there is enough data for each molecule). If not,
    all datapoints for that molecule are chosen. Entries for each molecule consists of energy array,
    coordinates array, and atomic numbers. Choosing is done for coordinates and energy.
    '''
    if molecs_to_use is None:
        molecs_to_use = list(molecule_data_subset.keys())
    reduced_dict = dict()
    for molecule in molecs_to_use:
        curr_energy, curr_coords, curr_nums = molecule_data_subset[molecule]
        if safety_check:
            assert(len(curr_energy) == len(curr_coords))
        if len(curr_energy) <= num_per_molecule:
            reduced_dict[molecule] = [curr_energy, curr_coords, curr_nums, 
                                      [i for i in range(len(curr_energy))]]
        else:
            possible_indices = [i for i in range(len(curr_energy))]
            chosen_indices = random.sample(possible_indices, num_per_molecule)
            #Gather based on randomly chosen sub_indices
            new_energy, new_coords = curr_energy[chosen_indices], curr_coords[chosen_indices]
            reduced_dict[molecule] = [new_energy, new_coords, curr_nums, chosen_indices] 
            # Add in the configuration numbers too
    return reduced_dict

def geometry_correction(final_subset_dictionary):
    '''
    Goes through and does a simple corection for each coordinate, where the shape for
    the geometry object has to be (3, natom)
    
    A transpose should do the trick...check back on this
    '''
    for _, data in final_subset_dictionary.items():
        coordinates = data[1]
        new_coord_array = []
        for matrix in coordinates:
            new_coord_array.append(matrix.T)
        data[1] = np.array(new_coord_array)

def reformat_data(final_subset_dictionary):
    '''
    Takes the final_subset_dictionary and converts it into a list of where each dictionary 
    corresponds to one molecule, containing its geometry information and target information.
    In this case, we are only interested in energies so the only target will be 'Etot'
    '''
    final_molecule_list = list()
    for _, data in final_subset_dictionary.items():
        #data here is a list of three arrays
        energies, rcarts, Zs , configs = data[0], data[1], data[2], data[3]
        for i in range(len(energies)):
            curr_energy, curr_rcart, curr_config = energies[i], rcarts[i], configs[i]
            molecule_dict = dict()
            molecule_dict['geom'] = vars(Geometry(Zs, curr_rcart))
            molecule_dict['targets'] = {'Etot' : curr_energy}
            molecule_dict['config_num'] = curr_config
            final_molecule_list.append(molecule_dict)
    return final_molecule_list

def save_subset_data(subset_dictionary_list, num_molecs, num_per_molecule):
    '''
    Uses pickling to save the subset of the dataset as a list of correctly formatted dictionaries
    that was generated. This is more efficient than regenerating the whole dataset again every time.
    
    TODO: Continue from here
    '''
    filename = f"dataset_{num_molecs}_molecs_{num_per_molecule}_permolec.p"
    with open(filename, 'wb') as handle:
        pickle.dump(subset_dictionary_list, handle)
        print("Finished saving data subset")
    pass

def generate_subset_data(ani_path, energy_targets, geometry_targets, num_molecules, num_data_per_molec,
                         save_data = True):
    '''
    Main driver function for generating the data so that modules that use this can
    just import this function.
    '''
    data_file_path = os.path.join(os.getcwd(), 'data', ani_path)
    full_dataset = get_targets_from_h5file(energy_targets + geometry_targets, data_file_path)
    molecule_subset, chosen_molecs = choose_molecule_subset(full_dataset, num_molecules)
    true_subset = choose_data_subset(molecule_subset, num_data_per_molec, chosen_molecs)
    geometry_correction(true_subset)
    final_result = reformat_data(true_subset)
    save_subset_data(final_result, num_molecules, num_data_per_molec)
    
    
if __name__ == "__main__":
    # ani_path = "ANI-1ccx_clean_shifted.h5"
    # ani_path_2 = "ANI-1ccx_clean.h5"
    # energy_targets = ['cc','pe']
    # geometry_targets = ['coordinates', 'atomic_numbers'] 
    # data_file_path = os.path.join(os.getcwd(), "data", ani_path)
    # geom_file_path = os.path.join(os.getcwd(), "data", ani_path_2)
    # num_molecules = 40
    # num_data_per_molec = 5
    
    # full_dataset = get_targets_from_dataset(energy_targets + geometry_targets, data_file_path)
    
    # molecule_subset, chosen_molecs = choose_molecule_subset(full_dataset, num_molecules)
    
    # true_subset = choose_data_subset(molecule_subset, num_data_per_molec, chosen_molecs)
    # geometry_correction(true_subset)
    # final_result = reformat_data(true_subset)
    generate_subset_data("ANI-1ccx_clean_shifted.h5", ['cc', 'pe'], ['coordinates', 'atomic_numbers'], 40, 5)
    
    
    pass