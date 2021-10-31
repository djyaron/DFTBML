#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 18:19:17 2021

@author: yaron

(Frank): Made some changes to add_dftb and load_ani1 (documentation and path fixing)

TODO: More options in the code to eventually handle non-zero Fermi temperatures. 
"""

#%% Imports, definitions
import os
import shutil
from subprocess import call
import numpy as np
Array = np.ndarray
from .dftbplus import write_dftb_infile, read_dftb_out, read_detailed_out, parse_charges,\
        compute_ESP_dipole
from FoldManager import get_ani1data
from h5py import File
from collections import Counter
import scipy
from DFTBpy import DFTB
from MasterConstants import valence_dict

from typing import List, Dict

from time import time

#%% Code behind

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
    target = {'dt': 'dt', 'dr': 'dr', 'pt': 'pt', 'pe': 'pe', 'pr': 'pr',
              'cc': 'cc', 'ht': 'ht', 'wt' : 'wt',
              'dipole': 'wb97x_dz.dipole',
              'charges': 'wb97x_dz.cm5_charges'}
    exclude = ['O3', 'N2O1', 'H1N1O3', 'H2']

    heavy_atoms = [x for x in range(1, maxheavy + 1)]

    dataset = get_ani1data(allowed_Zs, heavy_atoms, max_config, target, ani1_path = ani_path, exclude=exclude)
    return dataset

def load_au(au_type, max_mol, max_config, include_au2):
    '''
    input:
        au_type = 'full' for all or 'diff' for those with large differences
        max_mol = maximum number of different molecules (by name)
        max_config = for each mol, max number of configurations
        include_au2 = bool, False to exclude all molecules with just 2 Au atoms
        
    output:
        dataset: list of all configurations
          target field keys start with:
                  a: Adam   d: dftb.py    p: dftb+
             then:
                 e: elec   r: repulsive  t: total
            or 
             dft
    '''
    if os.getenv("USER") == "yaron":
        Au_data_path = "/home/yaron/code/dftbtorch/data/Au_energy_clean.h5"
        Au_large_diff_path = "/home/yaron/code/dftbtorch/data/Au_large_diff.h5"
    else:
        Au_data_path = "/home/francishe/Downloads/Datasets/Au_energy_clean.h5"
        Au_large_diff_path = "/home/francishe/Downloads/Datasets/Au_large_diff.h5"
    dataset = []
    if au_type == 'full':
        file_path = Au_data_path
        # targets starting with a refer to the value sent by Adam McSloy, generating
        # using or dftb code
        targets = {'ae': 'dftb_plus.elec_energy',
                   'ar': 'dftb_plus.rep_energy',
                   'at': 'dftb_plus.energy',
                   'dft': 'wb97x_dz.energy'}
    else:
        file_path = Au_large_diff_path
        targets = {'de': 'dftb.elec_energy',
                   'dr': 'dftb.rep_energy',
                   'dt': 'dftb.energy',
                   'pt': 'dftb_plus.energy',
                   'dft': 'wb97x_dz.energy'}
        
    
    with File(file_path, 'r') as h5data:
        for mol_name in h5data.keys():
            Zs = h5data[mol_name]['atomic_numbers'][()]
            emp_formula = Counter(Zs)
            if not include_au2 and emp_formula[79] <= 2:
                continue
            if len(dataset) >= max_mol:
                continue
            coords = h5data[mol_name]['coordinates'][()]
            nconfig = coords.shape[0]
            targets_nconfig = {k: h5data[mol_name][v][()] for k, v in targets.items()}
            for iconfig in range(min(max_config, nconfig)):
                curr = dict()
                curr['name'] = mol_name + '_'+str(iconfig)
                curr['atomic_numbers'] = Zs
                curr['coordinates'] = coords[iconfig, :, :]
                curr['targets'] = {k: targets_nconfig[k][iconfig] for k in targets}
                curr['targets']['conv'] = True
                dataset.append(curr)

    return dataset

def add_dftb(dataset: List[Dict], skf_dir: str, exec_path: str, pardict: Dict, do_our_dftb: bool = True, 
             do_dftbplus: bool = True, fermi_temp: float = None, parse: str = 'dftb',
             dispersion: bool = False) -> None:
    r"""Runs and parses out the DFTB+ results and adds them to the molecule
        dictionaries for a given dataset. 
    
    Arguments:
        dataset (List[Dict]): The dataset to run DFTB+ on
        skf_dir (str): The path to the directory containing all the 
            skf files to be used by DFTB+
        exec_path (str): The path to the DFTB+ executable
        pardict (Dict): The parameter dictionary to use
        do_our_dftb (bool): Whether to perform DFTBpy calculations on the 
            data. Defaults to True.
        do_dftbplus (bool): Whether to perform DFTB+ calculations on the 
            data. Defaults to True.
        fermi_temp (float): The fermi temperature to use for finite temperature
            smearing. Defaults to None.
        parse (str): Which file to parse. One of 'dftb' or 'detailed'. If 'detailed' 
            is chosen, then a breakdown of total energy into electronic and
            repulsive contributions is obtainable.
        dispersion (bool): Whether or not to include the dispersion block
            in the DFTB in file.
    
    Returns:
        None
    
    Notes: The options included in the DFTB_infile include whether the 
        DFTB+ calculation should be ShellResolvedSCC. This option defaults to True, 
        and should be left as True since both MIO and Au_org use resolved SCC. 
    """
    
    dftb_exec = exec_path

    DFTBoptions = {'ShellResolvedSCC': True}
    scratch_dir = "dftbscratch"
    if (not os.path.isdir(scratch_dir)):
        os.mkdir(scratch_dir)
    savefile_dir_path = os.path.join(scratch_dir, "save_files")
    if not os.path.isdir(savefile_dir_path):
        os.mkdir(savefile_dir_path)
        print(f"Created directory at {savefile_dir_path}")
    
    if fermi_temp is not None:
        DFTBoptions['FermiTemp'] = fermi_temp
        
    for imol,mol in enumerate(dataset):
        print('starting', imol, mol['name'])
        Zs = mol['atomic_numbers']
        rcart = mol['coordinates']
        
        # Input format for python dftb.py
        natom = len(Zs)
        cart = np.zeros([natom,4])
        cart[:,0] = Zs
        for ix in range(3):
            cart[:,ix+1] = rcart[:,ix]
        charge = 0
        mult = 1

        
        if do_our_dftb:
            start_time = time()
            res1 = dict()
            try:
                if fermi_temp is None:
                    dftb_us = DFTB(pardict, cart, charge, mult)
                else:
                    smearing = {'scheme': 'fermi',
                                'width' : 3.16679e-6 * fermi_temp}                 
                    dftb_us = DFTB(pardict, cart, charge, mult,
                                   smearing = smearing)
                res1['e'],focklist,_ = dftb_us.SCF()
                eorbs, _ = scipy.linalg.eigh(a=focklist[0], b = dftb_us.GetOverlap())
                homo = eorbs[ dftb_us.GetNumElecAB()[1] - 1]
                lumo = eorbs[ dftb_us.GetNumElecAB()[1]]
                res1['gap'] = (lumo - homo) * 27.211
                res1['r'] = dftb_us.repulsion
                res1['t'] = res1['e'] + res1['r']
                res1['conv'] = True
            except Exception:
                res1['conv'] = False
            end_time = time()
            res1['time'] = end_time-start_time
            if fermi_temp is None:
                res_key = 'dzero'
            else:
                res_key = 'd300'
            mol[res_key] = res1
            
        if do_dftbplus:
            dftb_infile = os.path.join(scratch_dir,'dftb_in.hsd')
            if parse == 'dftb':
                dftb_outfile = os.path.join(scratch_dir,'dftb.out')
            elif parse == 'detailed':
                dftb_outfile = os.path.join(scratch_dir,'detailed.out')
            charge_filename = os.path.join(scratch_dir, 'charges.dat')
            write_dftb_infile(Zs, rcart, dftb_infile, skf_dir, dispersion, DFTBoptions)
            start_time = time()
            with open(dftb_outfile,'w') as f:
                res2 = dict()
                try:
                    res = call(dftb_exec,cwd=scratch_dir,stdout = f, shell=False)
                    if parse == 'dftb':
                        dftb_res = read_dftb_out(dftb_outfile)
                        res2['t'] = dftb_res['Ehartree']
                    elif parse == 'detailed':
                        dftb_res = read_detailed_out(dftb_outfile)
                        res2['t'] = dftb_res['t']
                        #Should always be guaranteed the total energy
                        for key in ['e', 'r', 'disp']:
                            try:
                                res2[key] = dftb_res[key]
                            except:
                                pass
                        #Also see if dipoles and charges are present
                        charges = parse_charges(charge_filename, rcart, Zs, val_dict = valence_dict)
                        dipole = compute_ESP_dipole(charges, rcart)
                        res2['charges'] = charges
                        res2['dipole'] = dipole

                    res2['conv'] = True
                    if fermi_temp is None:
                        dftb_savefile = os.path.join(savefile_dir_path,
                                                     mol['name']+'_zero.out')
                    else:
                        dftb_savefile = os.path.join(scratch_dir,'auout',
                                                     mol['name']+'_300.out')
                        
                    shutil.copy(os.path.join(scratch_dir,'detailed.out'), 
                                dftb_savefile)
                except Exception:
                    if fermi_temp is None:
                        dftb_savefile = os.path.join(scratch_dir,'auout',
                                                     'err_' + mol['name']+'_zero.out')
                    else:
                        dftb_savefile = os.path.join(scratch_dir,'auout',
                                                     'err_' + mol['name']+'_300.out')
                    shutil.copy(os.path.join(scratch_dir,'detailed.out'), 
                                dftb_savefile)
                    res2['conv'] = False
            end_time = time()
            res2['time'] = end_time-start_time
            if fermi_temp is None:
                res_key = 'pzero'
            else:
                res_key = 'p300'
            mol[res_key] = res2

        if 'dconv' in mol[res_key] and 'pconv' in mol[res_key]:
            mol1 = mol[res_key]
            if mol1['dconv'] and mol1['pconv']:
                print(f"{mol['name']} us elec {mol1['de']:7.4f} rep {mol1['dr']:7.4f} sum {mol1['dt']:7.4f}" \
                      f" diff DFTB+ {np.abs(mol1['dt']-mol1['pt']):7.4e}") 
            elif mol1['dconv']:
                print(f"{mol['name']} DFTB+ failed")
            elif mol1['pconv']:
                print(f"{mol['name']} our dftb failed")
            else:
                print(f"{mol['name']} both our dftb and DFTB+ failed")
            #ts = mol['targets']
            #print(f"H5 elec {ts['pe']} rep {ts['pr']} sum {ts['pe'] +ts['pr']}" \
            #      f"tot {ts['pt']}  diff {ts['pt'] - ts['pe'] -ts['pr']} ")
            #print(f"{skf_type} on mol {imol} E(H) = {Ehartree:7.3f} " \
            #  #f" diff resolved(kcal/mol) {np.abs(Ehartree-mol['targets']['dt'])*627.0:7.3e}" \
            #  f" not resolved {np.abs(Ehartree-mol['targets']['pt'])*627.0:7.3e}" )

# def compare_convergence(dataset):
#     failed = dict()
#     failed['our dftb'] = [x['name'] for x in dataset if not x['dconv'] and x['pconv']]
#     failed['dftb+'] = [x['name'] for x in dataset if not x['pconv'] and x['dconv']]
#     failed['both'] = [x['name'] for x in dataset if not x['pconv'] and not x['dconv']]
#     for ftype,names in failed.items():
#         print(f"{len(names)} molecules failed {ftype}")
#         #print(names)

def compare_results(dataset, type1, field1, type2, field2):
    failed = dict()
    failed[type1] = [x['name'] for x in dataset if not x[type1]['conv'] and     x[type2]['conv']]
    failed[type2] = [x['name'] for x in dataset if     x[type1]['conv'] and not x[type2]['conv']]
    failed['both'] = [x['name'] for x in dataset if not x[type1]['conv'] and not x[type2]['conv']]
    for ftype,names in failed.items():
        print(f"{len(names)} molecules failed {ftype}")
    conv = [x for x in dataset if x[type1]['conv'] and x[type2]['conv']]
    if len(conv) == 0:
        print('no results to compare')
        return
    diff = np.array([x[type1][field1] - x[type2][field2] for x in conv]) * 627.0
    print(f"rms diff  {np.sqrt(np.average(np.square(diff)))} kcal/mol")
    print(f"mae diff  {np.average(np.abs(diff))} kcal/mol")
    print(f"max diff {np.max(np.abs(diff))} kcal/mol")
    
def fit_linear_ref_ener(dataset: List[Dict], target: str, allowed_Zs: List[int]) -> Array:
    r"""Fits a linear reference energy model between the DFTB+ method and some
        energy target
        
    Arguments:
        dataset (List[Dict]): The list of molecule dictionaries that have had the
            DFTB+ results added to them.
        target (str): The energy target that is being aimed for
        allowed_Zs (List[int]): The allowed atoms in the molecules
    
    Returns:
        coefs (Array): The coefficients of the reference energy
        XX (Array): 2D matrix in the number of atoms
        
    Notes: The reference energy corrects the magnitude between two methods
        in the following way: 
            
        E_2 = E_1 + sum_z N_z * C_z + C_0
        
        where N_z is the number of times atom z occurs within the molecule and 
        C_z is the coefficient for the given molecule. This is accomplished by solving
        a least squares problem.
    """
    nmol = len(dataset)
    XX = np.zeros([nmol, len(allowed_Zs) + 1])
    method1_mat = np.zeros([nmol])
    method2_mat = np.zeros([nmol])
    iZ = {x : i for i, x in enumerate(allowed_Zs)}
    
    for imol, molecule in enumerate(dataset):
        Zc = Counter(molecule['atomic_numbers'])
        for Z, count in Zc.items():
            XX[imol, iZ[Z]] = count
            XX[imol, len(allowed_Zs)] = 1.0
        method1_mat[imol] = molecule['pzero']['t'] #Start with the predicted energy from DFTB+ and correct upward to cc
        method2_mat[imol] = molecule['targets'][target]
    
    yy = method2_mat - method1_mat
    lsq_res = np.linalg.lstsq(XX, yy, rcond = None)
    coefs = lsq_res[0]
    return coefs, XX

def generate_linear_ref_mat(dataset: List[Dict], atypes: tuple) -> Array:
    r"""Generates the matrix required for the linear reference energy term 
        according to the ordering in atypes.
    
    Arguments:
        dataset (List[Dict]): The dataset being used, where each molecule/conformation
            is represented as an individual dictionary.
        atypes: The ordering of the elements used in the dataset.
    
    Returns: 
        reference_ener_mat (Array): The matrix used in computing the reference energy.
    
    Notes: This method is included because of the new repulsive model used in 
        DFTBrepuslive. The form of the reference energy has not chnaged, but since
        the intercept and the coefficients are pre-computed by the DFTBrepulsive
        backend, only the matrix is required. The form of the reference energy is
        as follows:
            
            E_ref = sum_z N_z * C_z + C_0
            
        Let X be the matrix generated by generate_linear_ref_mat. Then:
            
            E_ref = X * coef + intercept
        
        Where coef and intercept are computed from the DFTBrepulsive backend.
    """
    nmol = len(dataset)
    XX = np.zeros([nmol, len(atypes)])
    iZ = {x : i for i, x in enumerate(atypes)}
    
    for imol, molecule in enumerate(dataset):
        Zc = Counter(molecule['atomic_numbers'])
        for Z, count in Zc.items():
            XX[imol, iZ[Z]] = count
    
    return XX

def compute_results_torch(dataset: List[Dict], target: str, allowed_Zs: List[int],
                         error_metric: str = "RMS", compare_model_skf: bool = False) -> float:
    r"""Computes the results for the new skf files in predicting energies
    
    Arguments:
        dataset (List[Dict]): The list of molecule dictionaries that have had the
            DFTB+ results added to them.
        target (str): The energy target that is being aimed for
        allowed_Zs (List[int]): The allowed atoms in the molecules
        error_metric (str): The error metric used for computing the 
            deviations
        compare_model_skf (bool): Whether comparing ani1 target against 
            DFTB+ prediction (False) or comparing DFTBLayer prediction
            against DFTB+ prediction, i.e. saved models vs skf (True)
    
    Returns:
        
    Notes: The predicted energy from DFTB+ with the new skf files is stored 
        in the 'pzero' key. A linear reference energy term is fit between the 
        'pzero' 't' energy and the energy referred to by 'target'.
    """
    reference_ener_coefs, ref_ener_mat = fit_linear_ref_ener(dataset, target, allowed_Zs)
    #The predicted energy is in 'pzero'
    predicted_dt = np.array([molec['pzero']['t'] for molec in dataset])
    predicted_target = predicted_dt + np.dot(ref_ener_mat, reference_ener_coefs)
    if (not compare_model_skf):
        true_target = np.array([molec['targets'][target] for molec in dataset])
    else:
        true_target = np.array([molec['predictions'][target] for molec in dataset])
    diff = true_target - predicted_target
    if error_metric == "RMS":
        return diff, np.sqrt(np.mean(np.square(diff))) 
    elif error_metric == "MAE":
        return diff, np.mean(np.abs(diff))

def compute_results_torch_newrep(dataset: List[Dict], target: str, allowed_Zs: List[int], 
                                 atypes: tuple, coefs: Array, intercept: float, error_metric: str = "MAE") -> float:
    r"""Similar to compute_results_torch but with precomputed coefficients and intercepts
    
    Arguments:
        dataset (list[Dict]): The list of molecule dictionaries that have had the
            DFTB+ results added to them.
        target (str): The energy target that is being aimed for
        allowed_Zs (List[int]): The allowed atoms in the molecules
        atypes (tuple): The ordering of the elements to be used for constructing 
            the reference energy matrix
        coefs (Array): The reference energy coefficients
        intercept (Array): The reference energy intercept
        error_metric (str): "MAE" or "RMS"
    
    Returns:
        float: The error computed according to the method specified by error_metric
    """
    XX = generate_linear_ref_mat(dataset, atypes)
    predicted_dt = np.array([molec['pzero']['t'] for molec in dataset])
    predicted_target = predicted_dt + (np.dot(XX, coefs) + intercept)
    true_target = np.array([molec['targets'][target] for molec in dataset])
    if error_metric == "RMS":
        return true_target - predicted_target, np.sqrt(np.mean(np.square(true_target - predicted_target)))
    elif error_metric == "MAE":
        return true_target - predicted_target, np.mean(np.abs(true_target - predicted_target))
    
def compute_results_alt_targets(dataset: List[Dict], targets: List[str], error_metric: str = "MAE") -> Dict[str, float]:
    r"""Computes the error for other physical targets beside total energy
    
    Arguments:
        dataset (list[Dict]): The dataset with the DFTB+ results added to it
        targets (List[str]): The targets you want to compute the error for
        error_metric (str): The error metric to use. Defaults to MAE, meaning
            mean absolute error
    
    Returns:
        losses (Dict): The dictionary mapping the loss target to the 
            calculated error
    
    Notes: 
        The returned Losses dictionary also contains a key indicating the error
        metric that was used for computing that set of errors. For example,
        if the error_metric used is "MAE", then the dictionary key 
        "metric" will map to the string "MAE".
        
        The units for the losses will be standard atomic units, e.g. the unit
        of atomic charge (e) for atomic charges. For dipoles, if it is based 
        on EPS dipoles, it will be e * Angstrom.
        
        It is assumed that the physical targets are represented as vector 
        quantities rather than single values (e.g. dipoles, charges)
    """
    losses = {k : 0 for k in targets}
    for t in targets:
        #Restricts the series of values added based on if it's contained for the predictions dictionary
        pred_t = [mol['pzero'][t] for mol in dataset if t in mol['pzero']]
        true_t = [mol['targets'][t] for mol in dataset if t in mol['pzero']]
        assert(len(pred_t) == len(true_t))
        pred_t = list(map(lambda x : np.array(x), pred_t))
        true_t = list(map(lambda x : np.array(x), true_t))
        pred_flat = np.concatenate(pred_t)
        true_flat = np.concatenate(true_t)
        if error_metric == "MAE":
            losses[t] = np.mean(np.abs(true_flat - pred_flat))
        elif error_metric == "RMS":
            losses[t] = np.sqrt(np.mean(np.square(true_flat - pred_flat)))
    losses['metric'] = error_metric
    return losses
        
                
