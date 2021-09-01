# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 19:15:47 2021

@author: fhu14

Test module for testing construction of gammas from gammas saved per fold
"""
#%% Imports, definitions
from InputLayer import combine_gammas_ctrackers
from Training import sort_gammas_ctracks
import pickle, os
import numpy as np
from DataManager import load_gammas_per_fold, load_config_tracker_per_fold

#%% Code behind
def test_gammas_combination():
    r"""Testing reading and fusing of gammas and configuration trackers
    """
    split_mapping = {0 : [[0],[1]]}
    path = "test_files/gam_con_files"
    # path = "multiprocess_compute_test"
    
    gam_lst = load_gammas_per_fold(path)
    conf_lst = load_config_tracker_per_fold(path)
    assert(len(gam_lst) == 2 and len(conf_lst) == 2)
    assert(gam_lst[0].keys() == conf_lst[0].keys())
    assert(gam_lst[1].keys() == conf_lst[1].keys())
    
    for mol in gam_lst[0]:
        assert(gam_lst[0][mol]['gammas'].shape[0] == len(conf_lst[0][mol]))
    
    for mol in gam_lst[1]:
        assert(gam_lst[1][mol]['gammas'].shape[0] == len(conf_lst[1][mol]))
    
    tot_mol_0 = 0
    for mol, moldata in gam_lst[0].items():
        tot_mol_0 += moldata['gammas'].shape[0]
    tot_mol_1 = 0
    for mol, moldata in gam_lst[1].items():
        tot_mol_1 += moldata['gammas'].shape[0]
    assert(tot_mol_0 == 1863 and tot_mol_1 == 466)
    
    res = sort_gammas_ctracks(split_mapping, 0, gam_lst, conf_lst)
    assert(len(res) == 4)
    
    for mol in res[1][0]:
        if mol in res[3][0]:
            assert(set(res[1][0][mol]).isdisjoint(set(res[3][0][mol])))
    
    #Adding the train + valid gammas and train + valid config trackers should give back the 
    #   total config tracker
    total_gammas, total_config_tracker = combine_gammas_ctrackers(res[0] + res[2], res[1] + res[3])
    
    for key in total_config_tracker:
        assert(len(set(total_config_tracker[key])) == len(total_config_tracker[key]))
    
    assert(total_gammas.keys() == total_config_tracker.keys())
    
    for key in total_gammas:
        assert(total_gammas[key]['gammas'].shape[0] == len(total_config_tracker[key]))
    
    tru_config = pickle.load(open(path + "/config_tracker.p", "rb"))
    tru_gam = pickle.load(open(path + "/gammas.p", "rb"))
    
    assert(tru_config == total_config_tracker)
    assert(tru_gam.keys() == total_gammas.keys())
    
    for key in tru_gam:
        assert(np.allclose(tru_gam[key]['gammas'], total_gammas[key]['gammas']))
    
    print("Gammas and config tracker construction tests passed")

def run_gammas_tests():
    test_gammas_combination()

if __name__ == "__main__":
    run_gammas_tests()
    

