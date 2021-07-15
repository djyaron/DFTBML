# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 16:23:37 2021

@author: fhu14

Testing file for playing around with the new repulsive model

predicted_train_elec_only.p and predicted_validation_elec_only.p are 
pickle files that contain the Electronic energy targets predicted from
trained models, 150 epochs with lj dispersion correction. The impact of 
the lj dispersion for long range is negligible. 

This test is designed to assess the compute_gammas and the 
train_repulsive_model function, which is called in the framework of 
DFTBRepulsiveModel. 
"""

#%% Imports, definitions
from DFTBrepulsive import compute_gammas
from InputLayer import generate_gammas_input, DFTBRepulsiveModel
import pickle
from InputParser import parse_input_dictionaries, inflate_to_dict
from functools import reduce

#%% Code behind
def test_repulsive():
    r"""Goes through the computation of gammas and the prediction of 
        repulsive and reference energies. The MAE error (in kcal/mol)
        should be less than 1.0 kcal/mol.
    """
    print("Testing repulsive...")
    
    tolerance = 0.002
    #Dummy options dictionary
    settings_filename = "test_files/settings_refactor_tst_new_rep.json"
    defaults_filename = "test_files/refactor_default_tst_new_rep.json"
    final_s = parse_input_dictionaries(settings_filename, defaults_filename)
    opts = inflate_to_dict(final_s)
    
    #Try to generate gammas using dummy data
    batches_train = pickle.load(open("test_files/predicted_train_elec_only.p", "rb"))
    batches_valid = pickle.load(open("test_files/predicted_validation_elec_only.p", "rb"))
    all_train = list(reduce(lambda x, y : x + y, batches_train))
    all_valid = list(reduce(lambda x, y : x + y, batches_valid))
    gammas_input, config_tracker = generate_gammas_input(batches_train + batches_valid)
    compute_gammas(gammas_input, opts)
    
    #Try generating dummy repulsive energies
    model = DFTBRepulsiveModel(config_tracker)
    model.compute_repulsive_energies(batches_train + batches_valid, opts)
    
    #The predictions returned are Erep + Eref from DFTBrepulsive, so do a sanity
    #   check: mol['predictions']['Etot'] + model.pred... = mol['targets']['Etot']
    #Do the test on all_train first
    disagreements = []
    
    for molecule in all_train + all_valid:
        name, config = molecule['name'], molecule['iconfig']
        prediction_index = config_tracker[name].index(config)
        pred_rep_ref = model.pred[name]['prediction'][prediction_index]
        tot_ener = pred_rep_ref + molecule['predictions']['Etot']
        disagreements.append(abs(molecule['targets']['Etot'] - tot_ener))
    
    avg_disagreement = sum(disagreements) / len(disagreements)
    assert(avg_disagreement < tolerance)
    
    print(f"Repulsive test passed with tolerance of {tolerance} with a disagreement of {avg_disagreement}")

def run_repulsive_tests():
    test_repulsive()

if __name__ == "__main__":
    run_repulsive_tests()
    
    
    
    

