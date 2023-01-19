# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 16:26:50 2021

@author: fhu14

This is a dry run of the total machine learning model. The loss numbers
should agree well with established benchmarks. This will be using the 
same settings as was used for the skf_8020_100knot run. 
"""
#%% Imports, definitions
import os
import pickle
import shutil

import numpy as np
from driver import run_training

from .helpers import test_data_dir


#%% Code behind
def test_total_model():
    settings = os.path.join(test_data_dir, "skf_8020_100knot/settings_refactor_tst.json")
    defaults = os.path.join(test_data_dir, "skf_8020_100knot/default_refactor_tst.json")
    reference_energy_params, loss_tracker, all_models, model_variables, times_per_epoch = run_training(settings, defaults)
    
    #Compare losses to benchmarks
    benchmark_loss_path = os.path.join(test_data_dir, "skf_8020_100knot/Split0/loss_tracker.p")
    true_lt = pickle.load(open(benchmark_loss_path, 'rb'))
    assert(set(true_lt.keys()) == set(loss_tracker.keys()))
    
    loss_disagreements = dict()
    
    for loss in loss_tracker:
        loss_val, loss_train = loss_tracker[loss][0], loss_tracker[loss][1]
        assert(len(loss_val) == 10)
        assert(len(loss_train) == 10)
        true_loss_val, true_loss_train = true_lt[loss][0], true_lt[loss][1]
        true_loss_val, true_loss_train = true_loss_val[:10], true_loss_train[:10]
        
        loss_val, loss_train = np.array(loss_val), np.array(loss_train)
        true_loss_val, true_loss_train = np.array(true_loss_val), np.array(true_loss_train)
        
        print(f"For {loss} loss:")
        print("Benchmark val and train:")
        print(true_loss_val, true_loss_train)
        print("Obtained val and train:")
        print(loss_val, loss_train)
        
        loss_disagreements[loss] = dict()
        loss_disagreements[loss]['valid'] = np.mean(np.abs(true_loss_val - loss_val))
        loss_disagreements[loss]['train'] = np.mean(np.abs(true_loss_train - loss_train))
    
    tolerance = 0.1
    for loss in loss_disagreements:
        for key in ['valid', 'train']:
            print(f"{loss}, {key}, {loss_disagreements[loss][key]}")
            assert(loss_disagreements[loss][key] < tolerance)
    
    print("Total model tests passed")
    
    shutil.rmtree(os.path.join(test_data_dir, "test_ignore"))
        
    
def run_total_model_tests():
    test_total_model()

if __name__ == "__main__":
    run_total_model_tests()