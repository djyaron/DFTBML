# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 18:21:45 2021

@author: fhu14

Series of tests for the Dispersion energy correction. Currently, only a 
Lennard-Jones style potential is implemented for dispersion corrections.

Will also have a few test cases for the geometric mean
"""
#%% Imports, definitions
from Dispersion import LJ_Dispersion, torch_geom_mean
import torch
import pickle
Tensor = torch.Tensor
import numpy as np
Array = np.ndarray

#%% Code behind

def test_geom_mean():
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
    
def test_lj_dispersion():
    r"""Test to see if the Dispersion model functions properly
    """
    print("Testing dispersion functionality...")
    
    tst_feed_path = "test_files/tst_feed.p"
    device, dtype = None, torch.double
    cutoff = 2.0 #Dummy cutoff value for testing purposes only
    dispersion = LJ_Dispersion(cutoff, device, dtype)
    
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

def run_dispersion_tests():
    test_geom_mean()
    test_lj_dispersion()

if __name__ == "__main__":
    run_dispersion_tests()
