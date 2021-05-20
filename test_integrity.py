# -*- coding: utf-8 -*-
"""
Created on Thu May 20 10:50:54 2021

@author: fhu14
"""

"""
This module is designed to test the agreement between the DFTB layer implementation
in dftb_layer_splines_4.py and actual DFTB. This is done by pulling molecules from the 
ani1 dataset and running those molecules through the layer as well as through
DFTB+.exe. The results for total molecular energy are then compared to 
some tolerance.

Currently, only runs on molecules from ani1, i.e. organics, and a single fold is generated
that is run through both DFTB+ and the DFTB_Layer

By default, we are using the auorg_1_1 pardict for testing, and the target will
just be the 'dt' energy, i.e. DFTB+ energy
"""
import torch
import numpy as np
from dftb_layer_splines_4 import DFTB_Layer, get_ani1data, model_loss_initialization,\
    feed_generation, total_type_conversion, repulsive_energy_2
from fold_generator import single_fold_precompute
import os, os.path
import pickle, json
from statistics import mean

from run_dftbplus import add_dftb #Method for running the dftb executable

from typing import Union, List, Dict
Tensor = torch.Tensor
Array = np.ndarray
from auorg_1_1 import ParDict

class Settings:
    def __init__(self, settings_dict: Dict) -> None:
        r"""Generates a Settings object from the given dictionary
        
        Arguments:
            settings_dict (Dict): Dictionary containing key value pairs for the
                current hyperparmeter settings
        
        Returns:
            None
        
        Notes: Using an object rather than a dictionary is easier since you can
            just do settings.ZZZ rather than doing the bracket notation and the quotes.
        """
        for key in settings_dict:
            setattr(self, key, settings_dict[key])
            
def update_pytorch_arguments(settings: Settings) -> None:
    r"""Updates the arguments in the settings object to the corresponding 
        PyTorch types
        
    Arguments:
        settings (Settings): The settings object representing the current set of 
            hyperparameters
    
    Returns:
        None
        
    Notes: First checks if a CUDA-capable GPU is available. If not, it will
        default to using CPU only.
        
        
    TODO: Need to add tensor_dtype, tensor_device, and device_index as fields in
        the settings files
    """
    if settings.tensor_dtype == 'single':
        print("Tensor datatype set as single precision (float 32)")
        settings.tensor_dtype = torch.float
    elif settings.tensor_dtype == 'double':
        print("Tensor datatype set as double precision (float 64)")
        settings.tensor_dtype = torch.double
    else:
        raise ValueError("Unrecognized tensor datatype")
        
    num_gpus = torch.cuda.device_count()
    if settings.tensor_device == 'cpu':
        print("Tensor device set as cpu")
        settings.tensor_device = 'cpu'
    elif num_gpus == 0 or (not (torch.cuda.is_available())):
        print("Tensor device set as cpu because no GPUs are available")
        settings.tensor_device = 'cpu'
    else:
        gpu_index = int(settings.device_index)
        if gpu_index >= num_gpus:
            print("Invalid GPU index, defaulting to CPU")
            settings.tensor_device = 'cpu'
        else:
            print("Valid GPU index, setting tensor device to GPU")
            #I think the generic way is to do "cuda:{device index}", but not sure about this
            settings.tensor_device = f"cuda:{gpu_index}"
            print(f"Used GPU name: {torch.cuda.get_device_name(settings.tensor_device)}")

def load_ani1(ani_path: str, max_config: int, maxheavy: int = 8, allowed_Zs: List[int] = [1, 6, 7, 8]):
    r"""Pulls data from the ani h5 datafile
    
    Arguments:
        ani_path (str): The path to the ani dataset as an h5 file
        max_config (int): The maximum number of conformations to include 
            for each empirical formula
        maxheavy (int): The maximum number of heavy atoms that is allowed
        allowed_Zs (List[int]): The list containing atomic numbers for the
            allowed elements
    
    Returns:
        dataset (List[Dict]): The return from get_ani1data(), contains an 
            entry for each molecule as a dictionary
    """
    target = {'Etot': 'dt', 'dr': 'dr', 'pt': 'pt', 'pe': 'pe', 'pr': 'pr',
              'cc': 'cc', 'ht': 'ht',
              'dipole': 'wb97x_dz.dipole',
              'charges': 'wb97x_dz.cm5_charges'}
    exclude = ['O3', 'N2O1', 'H1N1O3', 'H2']

    heavy_atoms = [x for x in range(1, maxheavy + 1)]

    dataset = get_ani1data(allowed_Zs, heavy_atoms, max_config, target, exclude=exclude)
    return dataset


def get_total_energy_from_output(output: Dict, feed: Dict, molec_energies: Dict) -> None:
    r"""Extracts the total energy from the output and organizes it per feed
    
    Arguments:
        output (Dict): Output from the DFTB_Layer
        feed (Dict): The feed dictionary passed through the DFTB_Layer to generate output
        molec_energies (Dict): A dictionary mapping (Name, iconfig) : Etot, where the 
            total energy is the sum of the electronic and repulsive components
        
    Returns:
        None
            
    Notes: Here, Etot = Eelec + Erep because Eref = 0 given that we are comparing
        dt to dt, so there should be no reference energy contribution if the two
        methods are the same.
        
        The dictionary molec_energies starts off as an empty dictionary and is updated as 
        each new output is generated. Each value is a list of three elements consisting of 
        the total energy, electronic energy, and repulsive energy in that order.
    """
    for bsize in feed['basis_sizes']:
        curr_Etots = output['Eelec'][bsize] + output['Erep'][bsize]
        curr_names, curr_iconfigs = feed['names'][bsize], feed['iconfigs'][bsize]
        assert(len(curr_names) == len(curr_iconfigs) == len(curr_Etots))
        for i in range(len(curr_names)):
            key = (curr_names[i], curr_iconfigs[i])
            if key in molec_energies:
                return
            molec_energies[key] = [curr_Etots[i].item(), output['Eelec'][bsize][i].item(), output['Erep'][bsize][i].item()]
            
def test_agreement(s: Settings, tolerance: float, skf_dir: str, ani1_path: str, par_dict: Dict) -> bool:
    r"""Main driver function for testing agreement b/w DFTB_Layer and DFTB+
    
    Arguments:
        s (Settings): Settings object that contains values for the hyperparameters
        integ_tolerance (float): The tolerance for differences b/w the DFTB_Layer and
            DFTB+. 
        skf_dir (str): Relative path to the directory containing the skf files used by
            DFTB+
        ani1_path (str): Relative path to the h5 ani1 data file
        par_dict (Dict): Dictionary of operator values from skf files
        
    Returns:
        passed (bool): Indicates whether the test passed
    
    Notes:
        The integrity is tested by first computing the energy from the DFTB_Layer
            and DFTB+, computing their difference, and then testing that difference 
            against the tolerance.
    """
    dataset = get_ani1data(s.allowed_Zs, s.heavy_atoms, s.max_config, s.target, ani1_path, s.exclude)
    layer = DFTB_Layer(s.tensor_device, s.tensor_dtype, s.eig_method, s.rep_setting)
    feeds, dftb_lsts, all_models, _, _, _, _ = single_fold_precompute(s, dataset, par_dict)
    total_type_conversion(feeds, [], s.type_conversion_ignore_keys, device = s.tensor_device, dtype = s.tensor_dtype)
    
    if s.rep_setting == 'new':
        all_models['rep'] = repulsive_energy_2(s, feeds, [], all_models, layer, s.tensor_dtype, s.tensor_device)
        #Zero out the reference energy
        num_ref_ener_coeffs = len(s.reference_energy_starting_point)
        c_len = len(all_models['rep'].c_sparse)
        all_models['rep'].c_sparse[c_len - num_ref_ener_coeffs : c_len] = 0
    
    molec_energies = dict()
    for feed in feeds:
        output = layer(feed, all_models)
        if s.rep_setting == 'new':
            output['Erep'] = all_models['rep'].generate_repulsive_energies(feed, 'train')
        get_total_energy_from_output(output, feed, molec_energies)
        
    #add results from real dftb
    add_dftb(dataset, skf_dir, do_our_dftb = True, do_dftbplus = True, fermi_temp = None)
    
    #Compare the results b/w our DFTB and DFTB+ and the DFTB_Layer results and DFTB+
    our_dftb_vs_dftbplus = [abs(molec['pzero']['t'] - molec['dzero']['t']) for molec in dataset]
    dftb_layer_vs_dftbplus = [abs(molec['pzero']['t'] - molec_energies[(molec['name'], molec['iconfig'])][0]) for molec in dataset]
    dftb_layer_vs_our_dftb = [abs(molec['dzero']['t'] - molec_energies[(molec['name'], molec['iconfig'])][0]) for molec in dataset]
    dftb_layer_elec_vs_our_dftb_elec = [abs(molec['dzero']['e'] - molec_energies[(molec['name'], molec['iconfig'])][1]) for molec in dataset]
    dftb_layer_rep_vs_our_dftb_rep = [abs(molec['dzero']['r'] - molec_energies[(molec['name'], molec['iconfig'])][2]) for molec in dataset]
    
    print(f"Average disagreement between our dftb and dftb+ on total energy (kcal/mol): {mean(our_dftb_vs_dftbplus) * 627.0}")
    print(f"Average disagreement between dftb layer and dftb+ on total energy (kcal/mol): {mean(dftb_layer_vs_dftbplus) * 627}")
    print(f"Average disagreement between dftb layer and our dftb on total energy (kcal/mol): {mean(dftb_layer_vs_our_dftb) * 627}")
    print(f"Average disagreement between dftb layer and our dftb on electronic energy (kcal/mol): {mean(dftb_layer_elec_vs_our_dftb_elec) * 627}")
    print(f"Average disagreement between dftb layer and our dftb on repulsive energy (kcal/mol): {mean(dftb_layer_rep_vs_our_dftb_rep) * 627}")

if __name__ == "__main__":
    #File names
    settings_file_name = 'settings_default.json'
    ani1_path = os.path.join(os.getcwd(), "data", "ANI-1ccx_clean_fullentry.h5")
    skf_path = os.path.join(os.getcwd(), "auorg-1-1")
    
    
    with open(settings_file_name, 'r') as handle:
        settings = json.load(handle)
    
    settings_obj = Settings(settings)
    update_pytorch_arguments(settings_obj)
    par_dict = ParDict()
    
    test_agreement(settings_obj, 0, skf_path, ani1_path, par_dict)
    
    # #Load the data and pardict
    # ani_path = os.path.join(os.getcwd(), "data", "ANI-1ccx_clean_fullentry.h5")
    # dataset = load_ani1(ani_path, 1, settings_obj.heavy_atoms[-1], settings_obj.allowed_Zs)

    # #Fix the old repulsive method to use MIO repulsive spline block
    # layer = DFTB_Layer(settings_obj.tensor_device, settings_obj.tensor_dtype, eig_method = 'old', repulsive_method = 'old')
    
    # #Call single precompute on the data and convert to tensor data type
    # feeds, dftb_lsts, all_models, _, _, _, _ = single_fold_precompute(settings_obj, dataset, par_dict)
    # total_type_conversion(feeds, [], settings_obj.type_conversion_ignore_keys, device = settings_obj.tensor_device, dtype = settings_obj.tensor_dtype)
    
    # #Pass the feeds and all_models through the DFTB_Layer
    # molec_energies = dict()
    # for feed in feeds:
    #     output = layer(feed, all_models)
    #     get_total_energy_from_output(output, feed, molec_energies)
    
    # for molec in molec_energies:
    #     print(molec, molec_energies[molec])
        
    # #add results from real dftb
    # skf_dir = os.path.join(os.getcwd(), "auorg-1-1")
    # add_dftb(dataset, skf_dir, do_our_dftb = True, do_dftbplus = True, fermi_temp = None)
    
    
    
    
    



