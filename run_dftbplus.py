#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 18:19:17 2021

@author: yaron

(Frank): Made some changes to add_dftb and load_ani1 (documentation and path fixing)
"""
import os
import shutil
from subprocess import call
import numpy as np
import scipy
import pickle as pkl
from h5py import File
from collections import Counter
from matplotlib import pyplot as plt
from dftbplus import write_dftb_infile, read_dftb_out
from dftb_layer_splines_4 import get_ani1data

from typing import List, Union, Dict
Array = np.ndarray
import collections

from time import time
from auorg_1_1 import ParDict
# from mio_0_1 import ParDict
from dftb import DFTB
pardict = ParDict()

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
              'cc': 'cc', 'ht': 'ht',
              'dipole': 'wb97x_dz.dipole',
              'charges': 'wb97x_dz.cm5_charges'}
    exclude = ['O3', 'N2O1', 'H1N1O3', 'H2']

    heavy_atoms = [x for x in range(1, maxheavy + 1)]

    dataset = get_ani1data(allowed_Zs, heavy_atoms, max_config, target, exclude=exclude)
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

def add_dftb(dataset, skf_dir, do_our_dftb = True, do_dftbplus = True, fermi_temp = None):
    if os.getenv("USER") == "yaron":
        dftb_exec = "/home/yaron/code/dftbplusexe/dftbplus-20.2.1/bin/dftb+"
    else:
        dftb_exec = "C:\\Users\\fhu14\\Desktop\\DFTB17.1Windows\\DFTB17.1Windows-CygWin\\dftb+"

    DFTBoptions = {'ShellResolvedSCC': True}
    scratch_dir = "dftbscratch"
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
            dftb_outfile = os.path.join(scratch_dir,'dftb.out')
            write_dftb_infile(Zs, rcart, dftb_infile, skf_dir, DFTBoptions)
            start_time = time()
            with open(dftb_outfile,'w') as f:
                res2 = dict()
                try:
                    res = call(dftb_exec,cwd=scratch_dir,stdout = f, shell=False)
                    dftb_res = read_dftb_out(dftb_outfile)
                    res2['t'] = dftb_res['Ehartree']
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
        Zc = collections.Counter(molecule['atomic_numbers'])
        for Z, count in Zc.items():
            XX[imol, iZ[Z]] = count
            XX[imol, len(allowed_Zs)] = 1.0
        method1_mat[imol] = molecule['pzero']['t'] #Start with the predicted energy from DFTB+ and correct upward to cc
        method2_mat[imol] = molecule['targets'][target]
    
    yy = method2_mat - method1_mat
    lsq_res = np.linalg.lstsq(XX, yy, rcond = None)
    coefs = lsq_res[0]
    return coefs, XX

def compute_results_ANI1(dataset: List[Dict], target: str, allowed_Zs: List[int],
                         error_metric: str = "RMS", compare_model_skf: bool = False):
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
        in the 'pzero' key.
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
        return np.sqrt(np.mean(np.square(diff))) 
    elif error_metric == "MAE":
        return np.mean(np.abs(diff))
    
#%% ANI Testing (dftbtorch, electronic and organic only)
if __name__ == "__main__":
    
    allowed_Zs = [1,6,7,8]
    target = 'cc'
    data_path = os.path.join(os.getcwd(), "data", "ANI-1ccx_clean_fullentry.h5")
    skf_dir_base = os.path.join(os.getcwd(), "dftbscratch", "au") #0.0006212656602691023
    skf_dir_mio = os.path.join(os.getcwd(), "mio-0-1") #
    skf_dir_mio_1_1 = os.path.join(os.getcwd(), "mio-1-1")
    skf_dir_diff = os.path.join(os.getcwd(), "second_run") #1653.8940664244737
    skf_dir_psc = os.path.join(os.getcwd(), "pscskf", "run27") #9.591953499658114
    skf_dir_old_rep = os.path.join(os.getcwd(), "old_rep_setting_run") #0.0007158606052278207
    skf_dir_small_set = os.path.join(os.getcwd(), "fmt8020_skf")
    skf_dir_small_set_2 = os.path.join(os.getcwd(), "fold_molecs_test_8020_internal")
    dataset = load_ani1(data_path, 1)
    # dataset = [dataset[453]]
    print(f"The number of molecules in the dataset is {len(dataset)}")
    add_dftb(dataset, skf_dir_small_set_2)
    # diff = np.array([molec['pzero']['t'] - molec['dzero']['t'] for molec in dataset])
    # print(f"Simple error is {np.mean(np.abs(diff)) * 627} in kcal/mol, MAE")
    RMS = compute_results_ANI1(dataset, target, allowed_Zs, "MAE")
    print(f"Mean square error is {RMS} in Ha")
    print(f"Mean square error is {RMS * 627} in kcal/mol")
    pass
    #skf_dir_base using auorg_1_1 is 14.556713352104877 kcal/mol, RMS error.
    #skf_dir_psc using run 25 is 8.642467707661924 in kcal/mol, RMS error.
    #skf_dir_psc using run 26 is 14.459295342708376 in kcal/mol, RMS error.
    #skf_dir_psc using run 27 is 10.047199443654845 in kcal/mol, RMS error.
    #skf_dir_small_set is 6.13989050763862 in kcal/mol, RMS error.
    #skf_dir_mio is 15.3868130304943 in kcal/mol, RMS error.
    #skf_dir_small_set_2 is 6.1886413563640605 in kcal/mol, RMS error.
    
    #skf_dir_base using auorg_1_1 10.902432164852327 in kcal/mol, MAE error.
    #skf_dir_mio using mio_0_1 is 11.498048267608118 in kcal/mol, MAE error. 
    #skf_dir_small_set is 4.754482065178127 in kcal/mol, MAE error.
    #skf_dir_small_set_2 is 4.877564199475571 in kcal/mol, MAE error.
    
#%% Comparing SKF to saved models
    
    import pickle
    from functools import reduce
    import random
    
    training_predictions = "MasterPackage/predicted_train.p"
    validation_predictions = "MasterPackage/predicted_validation.p"
    target = "Etot"
    allowed_Zs = [1,6,7,8]
    
    train = pickle.load(open(training_predictions, 'rb'))
    valid = pickle.load(open(validation_predictions, 'rb'))
    
    train_molecs = list(reduce(lambda x, y : x + y, train))
    valid_molecs = list(reduce(lambda x, y : x + y, valid))
    
    dataset = train_molecs + valid_molecs
    dataset = random.sample(dataset, 500) #Randomly choose 500 molecules
    skf_dir = os.path.join(os.getcwd(), "MasterPackage", "old_rep_setting_run")
    
    add_dftb(dataset, skf_dir)
    diff = np.array([molec['pzero']['t'] - molec['predictions']['Etot'] for molec in dataset])
    print(f"Error in MAE is {np.mean(np.abs(diff))}")
    # error = compute_results_ANI1(dataset, target, allowed_Zs, "MAE", True)
    
    
    
    
    
    #%%
    # There are 154275 conformations without Au2, of which 104 have diffs
    #  82 configs of s03_19_Au6
    #  13 configs of s03_19_Au10
    #   7 configs of s02_12_Au10
    #   1 config of  s03_15_Au10_0
    #   1 config of  s04_Au10_0
    # and 37121 conformations with Au2, of which 5695 have diffs
    max_mol = 2000000
    max_config = 1
    include_au2 = False
    dataset = load_au('full',max_mol,max_config,include_au2)
    print(f"loaded {len(dataset)} molecular configurations")
    #%%
    do_our_dftb = True
    do_dftbplus = True
    fermi_temp = None
    add_dftb(dataset, 'au', do_our_dftb, do_dftbplus, fermi_temp)
    fermi_temp = 300.0
    add_dftb(dataset, 'au', do_our_dftb, do_dftbplus, fermi_temp)
    
    #%%
    #pkl.dump(dataset, open('rundftb_normal.pkl', 'wb'))
    #%% print results
    #print('**** new dftbd versus target dftb')
    #compare_results(dataset,'dzero','t','targets','dt')
    print('**** new dftb versus dftb+ at 300K')
    compare_results(dataset,'d300','t','p300','t')
    print('**** dftbd 0K to dftb+ 300K')
    compare_results(dataset,'dzero', 't', 'p300','t')
    #print('*** gaps at 0K versus 300 K')
    #compare_results(dataset,'dzero', 'gap', 'd300','gap')
    print('*** dftb+ at 300K versus at')
    compare_results(dataset,'d300', 't', 'targets','at')
    
    #%%
    gaps = []
    diffs = []
    for x in dataset:
        if x['dzero']['conv'] and x['p300']['conv']:
            gaps.append(x['dzero']['gap'])
            diffs.append(x['p300']['t'] - x['dzero']['t'])
    plt.figure(1)
    plt.plot(gaps, np.abs(diffs)*627.0, 'r.')
    plt.xlabel('HOMO-LUMO gap from 0 K calculation (eV)')
    plt.ylabel('Etot (DFTB+ at 300 K) - Etot(dftbd at 0 K)')
    #%%
    gaps = []
    diffs = []
    for x in dataset:
        if x['d300']['conv'] and x['p300']['conv']:
            gaps.append(x['d300']['gap'])
            diffs.append(x['p300']['t'] - x['d300']['t'])
    plt.figure(2)
    plt.plot(gaps, np.abs(diffs)*627.0, 'r.')
    plt.xlabel('HOMO-LUMO gap from dftbd 300 K calculation (eV)')
    plt.ylabel('Etot (DFTB+ at 300 K) - Etot(dftbd at 300 K)')
    #%%
    gaps = []
    diffs = []
    for x in dataset:
        if x['dzero']['conv'] and x['pzero']['conv'] and x['p300']['conv']:
            gaps.append(x['dzero']['gap'])
            diffs.append(x['p300']['t'] - x['pzero']['t'])
    plt.figure(3)
    plt.plot(gaps, np.abs(diffs)*627.0, 'r.')
    plt.xlabel('HOMO-LUMO gap from dftbd 0 K calculation (eV)')
    plt.ylabel('Etot (DFTB+ at 300 K) - Etot(DFTB+ at 0 K)')
    
    
    
    #%% Less old, commented out results
    # if False:    
    #     diff = np.array([x['dt'] - x['targets']['at'] for x in conv]) * 627.0
    #     print(f"rms diff us and Adam {np.sqrt(np.average(np.square(diff)))} kcal/mol")
    #     print(f"max diff us and Adam {np.max(np.abs(diff))} kcal/mol")
        
    #     our_time = np.abs([x['dtime'] for x in dataset])
    #     plus_time = np.abs([x['ptime'] for x in dataset])
    #     print(f"our time {np.sum(our_time)}  plus time {np.sum(plus_time)}")
    
    #%% OLD, commented out, CODE FROM HERE DOWN
    
    # allowed_Zs = [1,6,7,8]
    # maxheavy = 2
    # skf_test = 'ANI1rep1'
    # pkl_file = os.path.join('dftbscratch',skf_test,'_heavy' + str(maxheavy) + '.pk')
    
    # if not os.path.exists(pkl_file):    
    #     heavy_atoms = [x for x in range(1,maxheavy+1)]
    #     #Still some problems with oxygen, molecules like HNO3 are problematic due to degeneracies
    #     max_config = 10 
    #     # target = 'dt'
    #     target = {'dt' : 'dt', 'dr': 'dr', 'pt' : 'pt', 'pe' : 'pe', 'pr' : 'pr',
    #               'cc' : 'cc', 'ht': 'ht',
    #                'dipole' : 'wb97x_dz.dipole',
    #                'charges' : 'wb97x_dz.cm5_charges'}
    #     exclude = ['O3', 'N2O1', 'H1N1O3', 'H2']
        
        
    #     dataset = get_ani1data(allowed_Zs, heavy_atoms, max_config, target, exclude=exclude)
        
    #     #Proportion for training and validation
    #     # prop_train = 0.8
    #     # transfer_training = False
    #     # transfer_train_params = {
    #     #     'test_set' : 'pure',
    #     #     'impure_ratio' : 0.2,
    #     #     'lower_limit' : 4
    #     #     }
    #     # train_ener_per_heavy = False
        
    #     # training_molecs, validation_molecs = dataset_sorting(dataset, prop_train, 
    #     #             transfer_training, transfer_train_params, train_ener_per_heavy)
        
    #     #mol = pkl.load(open('mtest.pk','rb'))
    #     all_mol = dataset
    #     print('generating data for', len(all_mol),'molecules')
        
    #     for skf_type in ['mio',skf_test]:
    #         copyskf(os.path.join(scratch_dir,skf_type), os.path.join(scratch_dir,'skf'))
    
    #         for imol,mol in enumerate(all_mol):
    #             Zs = mol['atomic_numbers']
    #             rcart = mol['coordinates']
    #             write_dftb_in_hsd(Zs, rcart, scratch_dir)
    #             with open(os.path.join(scratch_dir,'dftb.out'),'w') as f:       
    #                 res = call(dftb_exec,cwd=scratch_dir,stdout = f, shell=False)
    #             Ehartree = read_dftb_out(scratch_dir)
    #             print(skf_type, imol,Ehartree,np.abs(Ehartree-mol['targets']['dt']))
    #             mol[skf_type] = Ehartree
        
    #     pkl.dump(all_mol, open(pkl_file,'wb'))
    # else:
    #     all_mol = pkl.load(open(pkl_file,'rb'))
    
    # #%%
    # # fit reference energy
    # iZ = {x:i for i,x in enumerate(allowed_Zs)}
    
    # nmol = len(all_mol)
    # XX = np.zeros([nmol,len(allowed_Zs)+1])
    # mio = np.zeros([nmol])
    # cc  = np.zeros([nmol])
    # ht = np.zeros([nmol])
    # ml  = np.zeros([nmol])
    # pt  = np.zeros([nmol])
    # for imol,mol in enumerate(all_mol):
    #     Zc = collections.Counter(mol['atomic_numbers'])
    #     for Z,count in Zc.items():
    #         XX[imol, iZ[Z]] = count
    #         XX[imol, len(allowed_Zs)] = 1.0
    #     cc[imol] = mol['targets']['cc']
    #     ht[imol] = mol['targets']['ht']
    #     mio[imol] = mol['mio']
    #     ml[imol] = mol[skf_test]
    #     pt[imol] = mol['targets']['pt']
    
    # yy = ht - mio
    # lsq_res = np.linalg.lstsq(XX, yy, rcond=None)
    # coefs = lsq_res[0]
    # pred = mio + np.dot(XX,coefs)
    
    # print(len(pred),'molecules')
    # mae_mio = np.mean(np.abs(pred - ht)) * 627.509
    # print('mio: mae of preds',mae_mio)
    
    # yy = ht - pt
    # lsq_res = np.linalg.lstsq(XX, yy, rcond=None)
    # coefs = lsq_res[0]
    # pred = pt + np.dot(XX,coefs)
    # mae_pt = np.mean(np.abs(pred - ht)) * 627.509
    # print('pt: mae of preds',mae_pt)
    
    # yy = ht - ml
    # lsq_res = np.linalg.lstsq(XX, yy, rcond=None)
    # coefs = lsq_res[0]
    # pred = ml + np.dot(XX,coefs)
    # mae_ml = np.mean(np.abs(pred - ht)) * 627.509
    # print('ml: mae of preds',mae_ml)
    
    # #%%
    # if False:
    #     pardict = ParDict()
    #     cart = np.hstack([Zs.reshape([-1,1]), rcart])
    #     dftb = DFTB(pardict, cart)
    #     Eelec, Fs, rhos = dftb.SCF()
    #     Erep = dftb.repulsion
    #     Eus = Eelec + Erep
        
    #     diff = (Ehartree - Eus) * 627
    
    #     print('E from our code Hartree', Eus, 'Ediff  kcal/mol', diff)
    
    
    # dftb_infile = os.path.join(scratch_dir, 'dftb_in.hsd')
    # dftb_outfile = os.path.join(scratch_dir, 'dftb.out')
    # DFTBoptions = {'ShellResolvedSCC': True}
    # pardict = ParDict()
    # charge = 0
    # mult = 1
    
    
    # with File(Au_data_path, 'r') as src, File(des_path, 'w') as des,\
    #      open(dftb_outfile, 'w') as f:
    
    #     for mol, moldata in src.items():
    #         Zs = moldata['atomic_numbers'][()]
    #         coords = moldata['coordinates'][()]
    #         des_mol = {'dftb.elec_energy': [],
    #                    'dftb.rep_energy': [],
    #                    'dftb.energy': [],
    #                    'dconv': [],
    #                    'dftb_plus.energy': [],
    #                    'pconv': []}
    
    #         for iconf, coord in enumerate(coords):
    #             # DFTB calculation
    #             cart = np.block([Zs.reshape(-1, 1), coord])
    #             # noinspection PyBroadException
    #             try:
    #                 dftb = DFTB(pardict, cart, charge, mult)
    #                 de = dftb.SCF()[0]
    #                 dr = dftb.repulsion
    #                 dt = de + dr
    #                 des_mol['dftb.elec_energy'].append(de)
    #                 des_mol['dftb.rep_energy'].append(dr)
    #                 des_mol['dftb.energy'].append(dt)
    #                 des_mol['dconv'].append(True)
    #             except Exception:
    #                 des_mol['dftb.elec_energy'].append(np.nan)
    #                 des_mol['dftb.rep_energy'].append(np.nan)
    #                 des_mol['dftb.energy'].append(np.nan)
    #                 des_mol['dconv'].append(False)
    #                 print(f"DFTB divergence: {mol}, conf #{iconf}")
    
    #             # DFTB+ calculation
    #             write_dftb_infile(Zs, coord, dftb_infile, skf_dir, DFTBoptions)
    #             # noinspection PyBroadException
    #             try:
    #                 call(dftb_exec, cwd=scratch_dir, stdout=f, shell=False)
    #                 dftb_res = read_dftb_out(dftb_outfile)
    #                 pe = dftb_res['Ehartree']
    #                 des_mol['dftb_plus.energy'].append(pe)
    #                 des_mol['pconv'].append(True)
    #             except Exception:
    #                 des_mol['dftb_plus.energy'].append(np.nan)
    #                 des_mol['pconv'].append(False)
    #                 print(f"DFTB+ divergence: {mol}, conf #{iconf}")
    
    #         # Save results to des
    #         g = des.create_group(mol)
    #         g.create_dataset('atomic_numbers', data=Zs)
    #         g.create_dataset('coordinates', data=coords)
    #         for entry, data in des_mol.items():
    #             g.create_dataset(name=entry, data=data)
    
    # for imol, mol in enumerate(dataset):
    #     Zs = mol['atomic_numbers']
    #     rcart = mol['coordinates']
    #
    #     natom = len(Zs)
    #     cart = np.zeros([natom, 4])
    #     cart[:, 0] = Zs
    #     for ix in range(3):
    #         cart[:, ix + 1] = rcart[:, ix]
    #     charge = 0
    #     mult = 1
    #     try:
    #         dftb_us = DFTB(pardict, cart, charge, mult)
    #         mol['de'], _, _ = dftb_us.SCF()
    #         mol['dr'] = dftb_us.repulsion
    #         mol['dt'] = mol['de'] + mol['dr']
    #         mol['dconv'] = True
    #     except Exception:
    #         mol['dconv'] = False
    #
    #     dftb_infile = os.path.join(scratch_dir, 'dftb_in.hsd')
    #     dftb_outfile = os.path.join(scratch_dir, 'dftb.out')
    #     write_dftb_infile(Zs, rcart, dftb_infile, skf_dir, DFTBoptions)
    #     with open(dftb_outfile, 'w') as f:
    #         try:
    #             res = call(dftb_exec, cwd=scratch_dir, stdout=f, shell=False)
    #             dftb_res = read_dftb_out(dftb_outfile)
    #             mol['pt'] = dftb_res['Ehartree']
    #             mol['pconv'] = True
    #         except Exception:
    #             mol['pconv'] = False
    #
    #     if mol['dconv'] and mol['pconv']:
    #         print(f"{mol['name']} us elec {mol['de']:7.4f} rep {mol['dr']:7.4f} sum {mol['dt']:7.4f}" \
    #               f" diff DFTB+ {np.abs(mol['dt'] - mol['pt']):7.4e}")
    #     elif mol['dconv']:
    #         print(f"{mol['name']} DFTB+ failed")
    #     elif mol['pconv']:
    #         print(f"{mol['name']} our dftb failed")
    #     else:
    #         print(f"{mol['name']} both our dftb and DFTB+ failed")
    #     # ts = mol['targets']
    #     # print(f"H5 elec {ts['pe']} rep {ts['pr']} sum {ts['pe'] +ts['pr']}" \
    #     #      f"tot {ts['pt']}  diff {ts['pt'] - ts['pe'] -ts['pr']} ")
    #     # print(f"{skf_type} on mol {imol} E(H) = {Ehartree:7.3f} " \
    #     #  #f" diff resolved(kcal/mol) {np.abs(Ehartree-mol['targets']['dt'])*627.0:7.3e}" \
    #     #  f" not resolved {np.abs(Ehartree-mol['targets']['pt'])*627.0:7.3e}" )
    #
    # # print summary of Au results
    # failed = dict()
    # failed['our dftb'] = [x['name'] for x in dataset if not x['dconv'] and x['pconv']]
    # failed['dftb+'] = [x['name'] for x in dataset if not x['pconv'] and x['dconv']]
    # failed['both'] = [x['name'] for x in dataset if not x['pconv'] and not x['dconv']]
    #
    # for ftype, names in failed.items():
    #     print(f"{len(names)} molecules failed {ftype}")
    #     print(names)
    #
    # # allowed_Zs = [1,6,7,8]
    # # maxheavy = 2
    # # skf_test = 'ANI1rep1'
    # # pkl_file = os.path.join('dftbscratch',skf_test,'_heavy' + str(maxheavy) + '.pk')
    #
    # # if not os.path.exists(pkl_file):
    # #     heavy_atoms = [x for x in range(1,maxheavy+1)]
    # #     #Still some problems with oxygen, molecules like HNO3 are problematic due to degeneracies
    # #     max_config = 10
    # #     # target = 'dt'
    # #     target = {'dt' : 'dt', 'dr': 'dr', 'pt' : 'pt', 'pe' : 'pe', 'pr' : 'pr',
    # #               'cc' : 'cc', 'ht': 'ht',
    # #                'dipole' : 'wb97x_dz.dipole',
    # #                'charges' : 'wb97x_dz.cm5_charges'}
    # #     exclude = ['O3', 'N2O1', 'H1N1O3', 'H2']
    #
    #
    # #     dataset = get_ani1data(allowed_Zs, heavy_atoms, max_config, target, exclude=exclude)
    #
    # #     #Proportion for training and validation
    # #     # prop_train = 0.8
    # #     # transfer_training = False
    # #     # transfer_train_params = {
    # #     #     'test_set' : 'pure',
    # #     #     'impure_ratio' : 0.2,
    # #     #     'lower_limit' : 4
    # #     #     }
    # #     # train_ener_per_heavy = False
    #
    # #     # training_molecs, validation_molecs = dataset_sorting(dataset, prop_train,
    # #     #             transfer_training, transfer_train_params, train_ener_per_heavy)
    #
    # #     #mol = pkl.load(open('mtest.pk','rb'))
    # #     all_mol = dataset
    # #     print('generating data for', len(all_mol),'molecules')
    #
    # #     for skf_type in ['mio',skf_test]:
    # #         copyskf(os.path.join(scratch_dir,skf_type), os.path.join(scratch_dir,'skf'))
    #
    # #         for imol,mol in enumerate(all_mol):
    # #             Zs = mol['atomic_numbers']
    # #             rcart = mol['coordinates']
    # #             write_dftb_in_hsd(Zs, rcart, scratch_dir)
    # #             with open(os.path.join(scratch_dir,'dftb.out'),'w') as f:
    # #                 res = call(dftb_exec,cwd=scratch_dir,stdout = f, shell=False)
    # #             Ehartree = read_dftb_out(scratch_dir)
    # #             print(skf_type, imol,Ehartree,np.abs(Ehartree-mol['targets']['dt']))
    # #             mol[skf_type] = Ehartree
    #
    # #     pkl.dump(all_mol, open(pkl_file,'wb'))
    # # else:
    # #     all_mol = pkl.load(open(pkl_file,'rb'))
    #
    # # #%%
    # # # fit reference energy
    # # iZ = {x:i for i,x in enumerate(allowed_Zs)}
    #
    # # nmol = len(all_mol)
    # # XX = np.zeros([nmol,len(allowed_Zs)+1])
    # # mio = np.zeros([nmol])
    # # cc  = np.zeros([nmol])
    # # ht = np.zeros([nmol])
    # # ml  = np.zeros([nmol])
    # # pt  = np.zeros([nmol])
    # # for imol,mol in enumerate(all_mol):
    # #     Zc = collections.Counter(mol['atomic_numbers'])
    # #     for Z,count in Zc.items():
    # #         XX[imol, iZ[Z]] = count
    # #         XX[imol, len(allowed_Zs)] = 1.0
    # #     cc[imol] = mol['targets']['cc']
    # #     ht[imol] = mol['targets']['ht']
    # #     mio[imol] = mol['mio']
    # #     ml[imol] = mol[skf_test]
    # #     pt[imol] = mol['targets']['pt']
    #
    # # yy = ht - mio
    # # lsq_res = np.linalg.lstsq(XX, yy, rcond=None)
    # # coefs = lsq_res[0]
    # # pred = mio + np.dot(XX,coefs)
    #
    # # print(len(pred),'molecules')
    # # mae_mio = np.mean(np.abs(pred - ht)) * 627.509
    # # print('mio: mae of preds',mae_mio)
    #
    # # yy = ht - pt
    # # lsq_res = np.linalg.lstsq(XX, yy, rcond=None)
    # # coefs = lsq_res[0]
    # # pred = pt + np.dot(XX,coefs)
    # # mae_pt = np.mean(np.abs(pred - ht)) * 627.509
    # # print('pt: mae of preds',mae_pt)
    #
    # # yy = ht - ml
    # # lsq_res = np.linalg.lstsq(XX, yy, rcond=None)
    # # coefs = lsq_res[0]
    # # pred = ml + np.dot(XX,coefs)
    # # mae_ml = np.mean(np.abs(pred - ht)) * 627.509
    # # print('ml: mae of preds',mae_ml)
    #
    # # #%%
    # # if False:
    # #     pardict = ParDict()
    # #     cart = np.hstack([Zs.reshape([-1,1]), rcart])
    # #     dftb = DFTB(pardict, cart)
    # #     Eelec, Fs, rhos = dftb.SCF()
    # #     Erep = dftb.repulsion
    # #     Eus = Eelec + Erep
    #
    # #     diff = (Ehartree - Eus) * 627
    #
    # #     print('E from our code Hartree', Eus, 'Ediff  kcal/mol', diff)
