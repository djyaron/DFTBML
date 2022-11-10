# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 11:50:21 2021

@author: fhu14


This module is designed to test the agreement between the DFTB layer implementation
in dftb_layer_splines_4.py and actual DFTB. This is done by pulling molecules from the 
ani1 dataset and running those molecules through the layer as well as through
DFTB+.exe. The results for total molecular energy are then compared to 
some tolerance.

Currently, only runs on molecules from ani1, i.e. organics, and a single fold is generated
that is run through both DFTB+ and the DFTB_Layer

By default, we are using the auorg_1_1 pardict for testing, and the target will
just be the 'dt' energy, i.e. DFTB+ energy. The spline mode should be set to
'debugging' in the settings file since this is used to test dftb.py against
the dftb_layer against DFTB+

TODO: THE CODE IS CURRENTLY BROKEN, NOT GOING TO RUN
"""
#%% Imports, definitions

import os
import os.path
from statistics import mean
from typing import Dict, List

import Auorg_1_1
import MIO_0_1
import numpy as np
import TestSKF
import torch
from DFTBLayer import DFTB_Layer, total_type_conversion
from DFTBPlus import add_dftb
from DFTBpy import _Gamma12
from FoldManager import get_ani1data, single_fold_precompute
from InputLayer import Input_layer_hubbard, repulsive_energy
from InputParser import collapse_to_master_settings, parse_input_dictionaries

from .helpers import (
    auorg_dir,
    get_dftbplus_executable,
    mio_dir,
    test_data_dir,
    test_skf_dir,
)

Tensor = torch.Tensor
Array = np.ndarray

#%% Code behind

def apx_equal(x, y, tol=1E-12):
    return abs(x - y) < tol

def get_total_energy_from_output(output: Dict, feed: Dict, molec_energies: Dict) -> None:
    r"""Extracts the total energy from the output and organizes it per feed
    
    Arguments:
        output (Dict): Output from the DFTB_Layer
        feed (Dict): The feed dictionary passed through the DFTB_Layer to generate output
        molec_energies (Dict): A dictionary mapping (Name, iconfig) : Etot, where the 
            total energy is the sum of the electronic and repulsive components
        rep_setting (str): The repulsive setting being used (one of 'old', 'new')
        
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
                print("Early return")
                return
            molec_energies[key] = [curr_Etots[i].item(), output['Eelec'][bsize][i].item(), output['Erep'][bsize][i].item()]
            
def test_agreement(s, tolerance: float, skf_dir: str, exec_path: str, ani1_path: str, par_dict: Dict, dataset_input: List[Dict] = None,
                   saved_precompute_data: List = None) -> (bool, List[float]):
    r"""Main driver function for testing agreement b/w DFTB_Layer and DFTB+
    
    Arguments:
        s (Settings): Settings object that contains values for the hyperparameters
        tolerance (float): The tolerance for differences b/w the DFTB_Layer, dftbpy, and
            dftb+
        skf_dir (str): Relative path to the directory containing the skf files used by
            DFTB+
        exec_path (str): Path to the DFTB+ executable to run
        ani1_path (str): Relative path to the h5 ani1 data file
        par_dict (Dict): Dictionary of operator values from skf files
        dataset_input (List[Dict]): Passed in data used for the check. Defaults to None.
        saved_precompute_data (List): The feeds, dftb_lsts, and all_models
            saved from a single_fold_precompute. Stored in a three-element list
            and passed in to cut down on recomputation time. The order must be
            feeds, dftb_lsts, all_models. Defaults to None.
        
    Returns:
        List[float]: The values computed for the test.
        bool: whether the test passed (all computed values less than tolerance)
    """
    dataset = get_ani1data(s.allowed_Zs, s.heavy_atoms, s.max_config, s.target, ani1_path, s.exclude) if dataset_input is None else dataset_input
    layer = DFTB_Layer(s.tensor_device, s.tensor_dtype, s.eig_method, s.rep_setting)
    feeds, dftb_lsts, all_models, _, _, _, _, _ = single_fold_precompute(s, dataset, par_dict) if saved_precompute_data is None else saved_precompute_data
    total_type_conversion(feeds, [], s.type_conversion_ignore_keys, device = s.tensor_device, dtype = s.tensor_dtype)
    
    if s.rep_setting == 'new':
        all_models['rep'] = repulsive_energy(s, feeds, [], all_models, layer, s.tensor_dtype, s.tensor_device)
        #Zero out the reference energy
        num_ref_ener_coeffs = len(s.reference_energy_starting_point)
        c_len = len(all_models['rep'].c_sparse)
        all_models['rep'].c_sparse[c_len - num_ref_ener_coeffs : c_len] = 0
    
    molec_energies = dict()
    for feed in feeds:
        output = layer.forward(feed, all_models)
        if s.rep_setting == 'new':
            output['Erep'] = all_models['rep'].generate_repulsive_energies(feed, 'train')
        get_total_energy_from_output(output, feed, molec_energies)
    
    assert(len(molec_energies) == len(dataset))
    d_name_conf = [(mol['name'], mol['iconfig']) for mol in dataset]
    assert(list(molec_energies.keys()) == d_name_conf)
    
    #add results from real dftb
    add_dftb(dataset, skf_dir, exec_path, par_dict, do_our_dftb = True, do_dftbplus = True, fermi_temp = None)
    
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
    
    all_elems = [ mean(our_dftb_vs_dftbplus) * 627, 
                 mean(dftb_layer_vs_dftbplus) * 627, 
                 mean(dftb_layer_vs_our_dftb) * 627, 
                 mean(dftb_layer_elec_vs_our_dftb_elec) * 627, 
                 mean(dftb_layer_rep_vs_our_dftb_rep) * 627]
    passed = all([elem < tolerance for elem in all_elems])
    return passed, all_elems

def test_G_agreement(s, tolerance: float, skf_dir: str, ani1_path: str, par_dict: Dict, dataset_input: List[Dict] = None, 
                     saved_precompute_data: List = None) -> (bool, float):
    r"""Method for testing disagreements in the coulomb matrix G b/w the dftb_layer and
        our pythonic DFTB
        
    Arguments:
        s (Settings): Settings object that contains values for the hyperparameters
        tolerance (float): The tolerance for differences b/w the DFTB_Layer, dftbpy, and
            dftb+
        skf_dir (str): Relative path to the directory containing the skf files used by
            DFTB+
        ani1_path (str): Relative path to the h5 ani1 data file
        par_dict (Dict): Dictionary of operator values from skf files
        dataset_input (List[Dict]): Passed in data used for the check. Defaults to None.
        saved_precompute_data (List): The feeds, dftb_lsts, and all_models
            saved from a single_fold_precompute. Stored in a three-element list
            and passed in to cut down on recomputation time. The order must be
            feeds, dftb_lsts, all_models. Defaults to None.
        
    Returns:
        passed (bool): Indicates if test passed
        float: The average G disagreements
        
    Notes:
        Compares the outputted Gamma from the dftb_layer to the gamma from dftb.py, 
        with the dftb.py Gamma converted to a full basis by the function ShellToFullBasis
    """
    dataset = get_ani1data(s.allowed_Zs, s.heavy_atoms, s.max_config, s.target, ani1_path, s.exclude) if dataset_input is None else dataset_input
    layer = DFTB_Layer(s.tensor_device, s.tensor_dtype, s.eig_method, s.rep_setting)
    feeds, dftb_lsts, all_models, _, _, _, _, _ = single_fold_precompute(s, dataset, par_dict) if saved_precompute_data is None else saved_precompute_data
    total_type_conversion(feeds, [], s.type_conversion_ignore_keys, device = s.tensor_device, dtype = s.tensor_dtype)
    
    g_disagreements = []
    
    for index, feed in enumerate(feeds):
        output = layer.forward(feed, all_models)
        curr_dftbs = dftb_lsts[index].dftbs_by_bsize
        all_gammas = output['G']
        for bsize in feed['basis_sizes']:
            curr_gammas = all_gammas[bsize]
            curr_gammas_dftb = curr_dftbs[bsize]
            for i in range(len(curr_gammas_dftb)):
                curr_gam = curr_gammas[i].detach().cpu().numpy()
                curr_dftb = curr_gammas_dftb[i]
                dftb_gam = curr_dftb.ShellToFullBasis(curr_dftb.GetGamma())
                assert(dftb_gam.shape == curr_gam.shape)
                g_disagreements.append(np.sum(np.abs(dftb_gam - curr_gam)))
    
    mean_disagreement = mean(g_disagreements)
    print(f"Average disagreement between dftb layer gamma and our dftb gamma, average sum of elemment-wise differences per molecule: {mean_disagreement}")
    return mean_disagreement < tolerance, mean_disagreement

def test_G_diag_agreement(s, tolerance: float, skf_dir: str, ani1_path: str, par_dict: Dict, dataset_input: List[Dict] = None, 
                          saved_precompute_data: List = None) -> bool:
    r"""Method for testing disagreements in the coulomb matrix G b/w the dftb_layer and our pythonic DFTB,
        but only in terms of the diagonal elements.
        
    Arguments:
        s (Settings): Settings object that contains values for the hyperparameters
        tolerance (float): The tolerance for differences b/w the DFTB_Layer, dftbpy, and
            dftb+
        skf_dir (str): Relative path to the directory containing the skf files used by
            DFTB+
        ani1_path (str): Relative path to the h5 ani1 data file
        par_dict (Dict): Dictionary of operator values from skf files
        dataset_input (List[Dict]): Passed in data used for the check. Defaults to None.
        saved_precompute_data (List): The feeds, dftb_lsts, and all_models
            saved from a single_fold_precompute. Stored in a three-element list
            and passed in to cut down on recomputation time. The order must be
            feeds, dftb_lsts, all_models. Defaults to None.
        
    Returns:
        passed (bool): Indicates if test passed
        
    Notes:
        Compares the outputted Gamma from the dftb_layer to the gamma from dftb.py, 
        with the dftb.py Gamma converted to a full basis by the function ShellToFullBasis.
        In theory, the diagonal elements should be fine, but the off-diagonal elements 
        should disagree.
    """
    dataset = get_ani1data(s.allowed_Zs, s.heavy_atoms, s.max_config, s.target, ani1_path, s.exclude) if dataset_input is None else dataset_input
    layer = DFTB_Layer(s.tensor_device, s.tensor_dtype, s.eig_method, s.rep_setting)
    feeds, dftb_lsts, all_models, _, _, _, _, _ = single_fold_precompute(s, dataset, par_dict) if saved_precompute_data is None else saved_precompute_data
    total_type_conversion(feeds, [], s.type_conversion_ignore_keys, device = s.tensor_device, dtype = s.tensor_dtype)
    
    g_disagreements = []
    
    for index, feed in enumerate(feeds):
        output = layer.forward(feed, all_models)
        curr_dftbs = dftb_lsts[index].dftbs_by_bsize
        all_gammas = output['G']
        for bsize in feed['basis_sizes']:
            curr_gammas = all_gammas[bsize]
            curr_gammas_dftb = curr_dftbs[bsize]
            for i in range(len(curr_gammas_dftb)):
                curr_gam = curr_gammas[i].detach().cpu().numpy()
                curr_dftb = curr_gammas_dftb[i]
                dftb_gam = curr_dftb.ShellToFullBasis(curr_dftb.GetGamma())
                assert(dftb_gam.shape == curr_gam.shape)
                disagreement = np.sum(np.abs(np.diag(dftb_gam - curr_gam)))
                g_disagreements.append(disagreement)
    
    mean_disagreement = mean(g_disagreements)
    print(f"Average disagreement between dftb layer gamma and our dftb gamma, average sum of elemment-wise differences per molecule ALONG THE DIAGONAL: {mean_disagreement}")
    return mean_disagreement < tolerance, mean_disagreement

def test_G_get_values(s, skf_dir: str, ani1_path: str, par_dict: Dict, dataset_input: List[Dict] = None, 
                      saved_precompute_data: List = None) -> bool:
    r"""Method for testing disagreements in the coulomb matrix G b/w the dftb_layer and our pythonic DFTB,
        but only in terms of the diagonal elements.
        
    Arguments:
        s (Settings): Settings object that contains values for the hyperparameters
        skf_dir (str): Relative path to the directory containing the skf files used by
            DFTB+
        ani1_path (str): Relative path to the h5 ani1 data file
        par_dict (Dict): Dictionary of operator values from skf files
        dataset_input (List[Dict]): Passed in data used for the check. Defaults to None.
        saved_precompute_data (List): The feeds, dftb_lsts, and all_models
            saved from a single_fold_precompute. Stored in a three-element list
            and passed in to cut down on recomputation time. The order must be
            feeds, dftb_lsts, all_models. Defaults to None.
        
    Returns:
        passed (bool): Indicates if test passed
        
    Notes:
        This tests the values obtained from the G models used in the dftb layer against
        the output of the _Gamma12 function. This serves as a test to ensure correct
        implementation of _Gamma12 operations in dftb_layer_splines_4.py
    """
    dataset = get_ani1data(s.allowed_Zs, s.heavy_atoms, s.max_config, s.target, ani1_path, s.exclude) if dataset_input is None else dataset_input
    feeds, dftb_lsts, all_models, _, _, _, _, _ = single_fold_precompute(s, dataset, par_dict) if saved_precompute_data is None else saved_precompute_data
    total_type_conversion(feeds, [], s.type_conversion_ignore_keys, device = s.tensor_device, dtype = s.tensor_dtype)
    
    for feed in feeds:  
        g_double_mods = [mod for mod in feed if (not isinstance(mod, str)) and (mod.oper == 'G') and (len(mod.Zs) == 2)]
        for g_mod in g_double_mods:
            print(f"Evaluating for {g_mod}")
            curr_g_mod = all_models[g_mod]
            assert(isinstance(curr_g_mod, Input_layer_hubbard))
            curr_vals = curr_g_mod.get_values(feed[g_mod])
            curr_dists = feed[g_mod]['xeval']
            vals_dists = list(zip(curr_vals, curr_dists))
            hub1, hub2 = curr_g_mod.get_variables()
            hub1, hub2 = hub1.item(), hub2.item()
            for pair in vals_dists:
                gamma_result = _Gamma12(pair[1].item(), hub1, hub2)
                print(pair[0].item(), gamma_result)
                assert(apx_equal(pair[0].item(), _Gamma12(pair[1].item(), hub1, hub2)))
    
    print("Input_layer_hubbard computation agrees with _Gamma12 method")
    return True

def run_layer_tests():
    r"""This is the main driver method for the tests. Note that to run the test,
        the spline mode shoule be set to debugging and the auorg pardict
        should be used. 
    """
    print("Running all DFTB Layer tests...")
    
    tol_G = 1E-12
    tol_Val = 3E-5
    settings_filename = os.path.join(test_data_dir, "dftb_layer_tst_settings.json")
    defaults_filename = os.path.join(test_data_dir, "refactor_default_tst.json")
    ani1_path = os.path.join(test_data_dir, "ANI-1ccx_clean_fullentry.h5")
    skf_path_au = os.path.join(auorg_dir, "auorg-1-1")
    skf_path_mio = os.path.join(mio_dir, "mio-0-1")
    skf_path_home = os.path.join(test_skf_dir, "skf_8020_no_trained_S")
    exec_path = get_dftbplus_executable()
    
    s_obj = parse_input_dictionaries(settings_filename, defaults_filename)
    s_obj = collapse_to_master_settings(s_obj)
    par_dict_au = Auorg_1_1.ParDict()
    par_dict_mio = MIO_0_1.ParDict()
    par_dict_home = TestSKF.ParDict()
    
    # print("With Home skfs...")
    
    # passed_G ,mean_G_total = test_G_agreement(s_obj, tol_G, skf_path_home, ani1_path, par_dict_home)
    # passed_G_diag, mean_G_diag = test_G_diag_agreement(s_obj, tol_G, skf_path_home, ani1_path, par_dict_home)
    # passed, vals = test_agreement(s_obj, tol_Val, skf_path_home, exec_path, ani1_path, par_dict_home)
    # our_dftb_vs_dftbplus, dftb_layer_vs_dftbplus, dftb_layer_vs_our_dftb, dftb_layer_elec_vs_our_dftb_elec, dftb_layer_rep_vs_our_dftb_rep = vals
    
    # print()
    # print()
    # print()
    
    # print(f"Average disagreement between our dftb and dftb+ on total energy (kcal/mol): {our_dftb_vs_dftbplus}")
    # print(f"Average disagreement between dftb layer and dftb+ on total energy (kcal/mol): {dftb_layer_vs_dftbplus}")
    # print(f"Average disagreement between dftb layer and our dftb on total energy (kcal/mol): {dftb_layer_vs_our_dftb}")
    # print(f"Average disagreement between dftb layer and our dftb on electronic energy (kcal/mol): {dftb_layer_elec_vs_our_dftb_elec}")
    # print(f"Average disagreement between dftb layer and our dftb on repulsive energy (kcal/mol): {dftb_layer_rep_vs_our_dftb_rep}")
    
    # print(f"Average disagreement between dftb layer gamma and our dftb gamma, average sum of elemment-wise differences per molecule: {mean_G_total}")
    
    # print(f"Average disagreement between dftb layer gamma and our dftb gamma, average sum of elemment-wise differences per molecule ALONG THE DIAGONAL: {mean_G_diag}")
    

    # assert(passed_G)
    # print("Passed test_G_agreement")
    # assert(passed_G_diag)
    # print("Passed test_G_diag_agreement")
    # assert(passed)
    # print("Passed test_agreement")
    
    # print("Passed with Home SKFS")
    
    print("With Auorg skfs...")
    
    passed_G ,mean_G_total = test_G_agreement(s_obj, tol_G, skf_path_au, ani1_path, par_dict_au)
    passed_G_diag, mean_G_diag = test_G_diag_agreement(s_obj, tol_G, skf_path_au, ani1_path, par_dict_au)
    passed, vals = test_agreement(s_obj, tol_Val, skf_path_au, exec_path, ani1_path, par_dict_au)
    our_dftb_vs_dftbplus, dftb_layer_vs_dftbplus, dftb_layer_vs_our_dftb, dftb_layer_elec_vs_our_dftb_elec, dftb_layer_rep_vs_our_dftb_rep = vals
    
    print()
    print()
    print()
    
    print(f"Average disagreement between our dftb and dftb+ on total energy (kcal/mol): {our_dftb_vs_dftbplus}")
    print(f"Average disagreement between dftb layer and dftb+ on total energy (kcal/mol): {dftb_layer_vs_dftbplus}")
    print(f"Average disagreement between dftb layer and our dftb on total energy (kcal/mol): {dftb_layer_vs_our_dftb}")
    print(f"Average disagreement between dftb layer and our dftb on electronic energy (kcal/mol): {dftb_layer_elec_vs_our_dftb_elec}")
    print(f"Average disagreement between dftb layer and our dftb on repulsive energy (kcal/mol): {dftb_layer_rep_vs_our_dftb_rep}")
    
    print(f"Average disagreement between dftb layer gamma and our dftb gamma, average sum of elemment-wise differences per molecule: {mean_G_total}")
    
    print(f"Average disagreement between dftb layer gamma and our dftb gamma, average sum of elemment-wise differences per molecule ALONG THE DIAGONAL: {mean_G_diag}")
    

    assert(passed_G)
    print("Passed test_G_agreement")
    assert(passed_G_diag)
    print("Passed test_G_diag_agreement")
    assert(passed)
    print("Passed test_agreement")
    
    print("Passed with Au SKFS")
    
    print("With Mio skfs...")
    
    passed_G ,mean_G_total = test_G_agreement(s_obj, tol_G, skf_path_mio, ani1_path, par_dict_mio)
    passed_G_diag, mean_G_diag = test_G_diag_agreement(s_obj, tol_G, skf_path_mio, ani1_path, par_dict_mio)
    passed, vals = test_agreement(s_obj, tol_Val, skf_path_mio, exec_path, ani1_path, par_dict_mio)
    our_dftb_vs_dftbplus, dftb_layer_vs_dftbplus, dftb_layer_vs_our_dftb, dftb_layer_elec_vs_our_dftb_elec, dftb_layer_rep_vs_our_dftb_rep = vals
    
    print()
    print()
    print()
    
    print(f"Average disagreement between our dftb and dftb+ on total energy (kcal/mol): {our_dftb_vs_dftbplus}")
    print(f"Average disagreement between dftb layer and dftb+ on total energy (kcal/mol): {dftb_layer_vs_dftbplus}")
    print(f"Average disagreement between dftb layer and our dftb on total energy (kcal/mol): {dftb_layer_vs_our_dftb}")
    print(f"Average disagreement between dftb layer and our dftb on electronic energy (kcal/mol): {dftb_layer_elec_vs_our_dftb_elec}")
    print(f"Average disagreement between dftb layer and our dftb on repulsive energy (kcal/mol): {dftb_layer_rep_vs_our_dftb_rep}")
    
    print(f"Average disagreement between dftb layer gamma and our dftb gamma, average sum of elemment-wise differences per molecule: {mean_G_total}")
    
    print(f"Average disagreement between dftb layer gamma and our dftb gamma, average sum of elemment-wise differences per molecule ALONG THE DIAGONAL: {mean_G_diag}")
    
    assert(passed_G)
    print("Passed test_G_agreement")
    assert(passed_G_diag)
    print("Passed test_G_diag_agreement")
    assert(passed)
    print("Passed test_agreement")
    
    print("Passed with Mio SKFS")
    
    print("DFTB Layer tests passed")

if __name__ == "__main__":
    run_layer_tests()
    
    



