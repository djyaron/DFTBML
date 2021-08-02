# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 18:21:45 2021

@author: fhu14

Series of tests for the Dispersion energy correction. Currently, only a 
Lennard-Jones style potential is implemented for dispersion corrections.

Will also have a few test cases for the geometric mean
"""
#%% Imports, definitions
from Dispersion import LJ_Dispersion, torch_geom_mean, np_geom_mean
from DataManager import load_combined_fold
from DFTBLayer import DFTB_Layer
from Dispersion import LJ_Dispersion
from DFTBPlus import add_dftb
import torch
import pickle, os
Tensor = torch.Tensor
import numpy as np
Array = np.ndarray
from functools import reduce
import Auorg_1_1

#%% Code behind
class dummy:
    def __init__(self):
        pass

def test_torch_geom_mean():
    r"""Only concerned with 2-term geometric means in application, but
        will include additional terms here for testing purposes.
    """
    print("Testing geometric mean...")
    
    x, y = torch.tensor(0.), torch.tensor(100.)
    assert(torch_geom_mean([x, y]).item() == 0)
    x, y, z = torch.tensor(0.), torch.tensor(100.), torch.tensor(20.)
    assert(torch_geom_mean([x, y, z]) == 0)
    x, y = torch.tensor(1.), torch.tensor(2.)
    assert(torch_geom_mean([x, y]) == (2)**0.5)
    x, y = torch.tensor(3.), torch.tensor(4.)
    assert(torch_geom_mean([x, y]) == (12) ** 0.5)
    # tst_tens = torch.abs(torch.randn(2))
    # assert(torch_geom_mean([tst_tens[0], tst_tens[1]]) == (tst_tens[0].item() * tst_tens[1].item()) ** 0.5)
    x, y = torch.tensor(3., requires_grad = True), torch.tensor(4., requires_grad = True)
    assert(torch_geom_mean([x, y]).grad_fn != None) #Make sure it's differentiable
    
    print("Geometric mean tests passed")
    
    
def test_np_geom_mean():
    r"""Only concerned with 2-term geometric means in application, but
        will include additional terms here for testing purposes.
    """
    print("Testing geometric mean...")
    
    x, y = 0, 100
    assert(np_geom_mean([x, y]) == 0)
    x, y, z = 0, 100, 20
    assert(np_geom_mean([x, y, z]) == 0)
    x, y = 1, 2
    assert(abs(np_geom_mean([x, y]) - (2)**0.5) < 1E-12)
    x, y = 3, 4
    assert(abs(np_geom_mean([x, y]) - (12) ** 0.5) < 1E-12)
    
    print("Geometric mean tests passed")
    
def test_lj_dispersion():
    r"""Test to see if the Dispersion model functions properly
    """
    print("Testing dispersion functionality...")
    
    tst_feed_path = "test_files/tst_feed.p"
    device, dtype = None, torch.double
    dispersion = LJ_Dispersion(device, dtype)
    
    tst_feed = pickle.load(open(tst_feed_path, 'rb'))
    variables = dispersion.get_variables()
    disp_dict = dispersion.get_disp_energy(tst_feed)
    
    print("Dispersion energy dictionary was successfully generated")
    
    print(f"Dispersion variables: {variables}")
    
    assert(set(disp_dict.keys()) == set(tst_feed['basis_sizes']))
    
    print("Dispersion keys match")
    
    for basis_size in tst_feed['basis_sizes']:
        curr_names = tst_feed['names'][basis_size]
        curr_disp_eners = disp_dict[basis_size]
        print(f"For current molecules: {curr_names}")
        print(f"Dispersion energies are: {curr_disp_eners}")
    
    print("Dispersion tests finished")

def test_lj_dispersion_against_dftbplus():
    r"""Tests the dispersion model predictions of dispersion energy against
        the DFTB+ dispersion energies
    """
    print("Testing dispersion against DFTB+ predictions")
    s = dummy()
    s.ragged_dipole = True
    s.run_check = False
    fold_mapping = {0 : [[0],[1]]}
    fold_num = 0
    top_level_fold_path = "test_files/fold_molecs_test_8020"
    skf_dir = os.path.join(os.getcwd(), "Auorg_1_1", "auorg-1-1")
    exec_path = "C:\\Users\\fhu14\\Desktop\\DFTB17.1Windows\\DFTB17.1Windows-CygWin\\dftb+"
    pardict = Auorg_1_1.ParDict()
    
    #Load in the data
    training_feeds, validation_feeds, training_dftblsts, validation_dftblsts,\
        training_batches, validation_batches = load_combined_fold(s, top_level_fold_path, fold_num, fold_mapping)
    
    all_feeds = training_feeds + validation_feeds
    all_batches = training_batches + validation_batches
    
    #Initialize the dispersion model
    disp_mod = LJ_Dispersion(None, torch.double)
    
    for i, feed in enumerate(all_feeds):
        disp_dict = disp_mod.get_disp_energy(feed)
        curr_batch = all_batches[i]
        for bsize in feed['basis_sizes']:
            curr_glabels = feed['glabels'][bsize]
            for i, glabel in enumerate(curr_glabels):
                curr_batch[glabel]['dispersion'] = disp_dict[bsize][i].item()
    
    all_mols = list(reduce(lambda x, y : x + y, all_batches))
    all_mols_trunc = all_mols[:500]
    
    #The SKF files to use for this run through DFTB+ is irrelevant, just want
    #   to see what the dispersion energies are in the output
    add_dftb(all_mols_trunc, skf_dir, exec_path, pardict, parse = 'detailed', dispersion = True)
    
    disagreements = []
    for mol in all_mols_trunc:
        if ('dispersion' in mol) and ('disp' in mol['pzero']):
            disagreements.append(abs(mol['dispersion'] - mol['pzero']['disp']))
    
    all_calc_disp = [abs(mol['dispersion']) for mol in all_mols_trunc]
    all_dftbp_disp = [abs(mol['pzero']['disp']) for mol in all_mols_trunc if 'disp' in mol['pzero']]
    average_abs_disp_calc = sum(all_calc_disp) / len(all_calc_disp)
    average_abs_disp_dftbp = sum(all_dftbp_disp) / len(all_dftbp_disp)
    
    print(f"Average disagreement on dispersion energies (in MAE): {sum(disagreements) / len(disagreements)}")
    print(f"Average absolute calculated dispersion: {average_abs_disp_calc}")
    print(f"Average absolute DFTB+ dispersion: {average_abs_disp_dftbp}")
    
    print("DFTB+ dispersion information added")
    
    
    
    
    
    
    
    
        
        
        
    
    
    

def run_dispersion_tests():
    test_torch_geom_mean()
    test_np_geom_mean()
    test_lj_dispersion()
    test_lj_dispersion_against_dftbplus()

if __name__ == "__main__":
    run_dispersion_tests()
