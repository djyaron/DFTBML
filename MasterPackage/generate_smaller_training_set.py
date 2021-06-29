# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 13:37:54 2021

@author: fhu14

Creates a smaller training set for testing out the model
"""
#Import block
import pickle
import os
from FoldManager import get_ani1data
import random

#Code block
    
#%% Generate smaller training set (80/20 split) from ANI-1ccx_clean_fullentry.h5
if __name__ == "__main__":
    
    source_file = "ANI-1ccx_clean_fullentry.h5"
    source_dir = "fold_molecs_test_8020"
    prop_train = 0.8
    prop_valid = 0.2
    
    allowed_Z = [1, 6, 7, 8]
    heavy_atoms = [1, 2, 3, 4, 5, 6, 7, 8]
    max_config = 5
    target = {"Etot" : "cc",
           "dipole" : "wb97x_dz.dipole",
           "charges" : "wb97x_dz.cm5_charges"}
    exclude = ["O3", "N2O1", "H1N1O3", "H2"]
    
    dataset = get_ani1data(allowed_Z, heavy_atoms, max_config, target, source_file, exclude)
    random.shuffle(dataset)
    
    num_molecs_train = int(prop_train * len(dataset))
    num_molecs_valid = len(dataset) - num_molecs_train
    
    print(f"{len(dataset)}")
    
    with open(os.path.join(source_dir, "Fold0_molecs.p"), 'wb') as handle:
        pickle.dump(dataset[:num_molecs_train], handle)
    
    with open(os.path.join(source_dir, "Fold1_molecs.p"), 'wb') as handle:
        pickle.dump(dataset[num_molecs_train:], handle)
        
    print("Data saved successfully")
    
#%% Generate a test training set of gold molecules from Au_energy_clean.h5

    source_file = "Au_energy_clean.h5"
    source_dir = "fold_molecs_au_clean"
    
    if (not os.path.isdir(source_dir)):
        os.mkdir(source_dir)
    
    prop_train = 0.8
    prop_valid = 0.2
    
    allowed_Z = [1, 6, 7, 8, 79]
    heavy_atoms = [1, 2, 3, 4, 5, 6, 7, 8]
    max_config = 5
    target = {
        "Etot" : 'wb97x_dz.energy',
        "Etot_DFTB+" : 'dftb_plus.energy'
        }
    exclude = ["O3", "N2O1", "H1N1O3", "H2"]
    
    dataset = get_ani1data(allowed_Z, heavy_atoms, max_config, target, source_file, exclude)
    random.shuffle(dataset)
    
    num_molecs_train = int(prop_train * len(dataset))
    num_molecs_valid = len(dataset) - num_molecs_train
    
    print(f"{len(dataset)}")
    
    with open(os.path.join(source_dir, "Fold0_molecs.p"), 'wb') as handle:
        pickle.dump(dataset[:num_molecs_train], handle)
    
    with open(os.path.join(source_dir, "Fold1_molecs.p"), 'wb') as handle:
        pickle.dump(dataset[num_molecs_train:], handle)
        
    print("Data saved successfully")
    
    
    