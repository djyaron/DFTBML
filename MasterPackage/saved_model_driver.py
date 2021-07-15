# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 18:00:19 2021

@author: fhu14

Alternative driver method where the models are saved and reloaded
rather than being generated from the beginning. Predictions from the layer
are saved to the molecule dictionaries using the functionality of 
the PredictionHandler package.
"""
#%% Imports, definitions
from InputParser import parse_input_dictionaries, collapse_to_master_settings
from DFTBLayer import DFTB_Layer
from PredictionHandler import organize_predictions
import argparse
from Precompute import precompute_stage
from Training import training_loop, exclude_R_backprop, write_output_skf, \
    write_output_lossinfo
import time, pickle
import torch

#%% Code behind

def pass_feeds_through(settings_filename: str, defaults_filename: str,
                       all_models_filename: str) -> None:
    r"""Passes feeds through the dftb layer and generates predictions
        from the DFTBlayer.
    
    Arguments:
        settings_filename (str): The name of the settings filename
        defaults_filename (str): The name of the defaults filename
        all_models_filename (str): The filename of the file
            containing the saved models
    
    Returns: 
        None
    """
    s = parse_input_dictionaries(settings_filename, defaults_filename)
    s = collapse_to_master_settings(s)
    
    #Established models and variables
    saved_models = pickle.load(open(all_models_filename, "rb"))
    print("Loaded in saved_models")
    
    layer = DFTB_Layer(device = s.tensor_device, dtype = s.tensor_dtype, eig_method = s.eig_method, repulsive_method = s.rep_setting)
    
    model_save, variable_save = None, None
    
    all_models, model_variables, training_feeds, validation_feeds, training_dftblsts, validation_dftblsts, losses, all_losses, loss_tracker, training_batches, validation_batches = precompute_stage(s, s.par_dict_name, 0, s.split_mapping, model_save, variable_save)
            
    
    for i, feed in enumerate(validation_feeds):
        with torch.no_grad():
            output = layer.forward(feed, saved_models)
            #Add in the repulsive energies if using new repulsive model
            for loss in all_losses:
                if loss == 'Etot':
                    if s.train_ener_per_heavy:
                        res = all_losses[loss].get_value(output, feed, True, s.rep_setting)
                    else:
                        res = all_losses[loss].get_value(output, feed, False, s.rep_setting)
                    #Add in the prediction
                    feed['predicted_Etot'] = res[1]
                elif loss == 'dipole':
                    res = all_losses[loss].get_value(output, feed, s.rep_setting)
                    #Add in the prediction 
                    feed['predicted_dipole'] = res[1]
                else:
                    res = all_losses[loss].get_value(output, feed, s.rep_setting)
                    if isinstance(res, tuple):
                        feed[f"predicted_{loss}"] = res[1]
        organize_predictions(feed, validation_batches[i], losses, ['Eelec'], s.train_ener_per_heavy)
        
    for i, feed in enumerate(training_feeds):
        output = layer.forward(feed, saved_models)
        for loss in all_losses:
            if loss == 'Etot':
                if s.train_ener_per_heavy:
                    res = all_losses[loss].get_value(output, feed, True, s.rep_setting)
                else:
                    res = all_losses[loss].get_value(output, feed, False, s.rep_setting)
                #Add in the prediction
                feed['predicted_Etot'] = res[1]
            elif loss == 'dipole':
                res = all_losses[loss].get_value(output, feed, s.rep_setting)
                #Add in the prediction 
                feed['predicted_dipole'] = res[1]
            else:
                res = all_losses[loss].get_value(output, feed, s.rep_setting)
                if isinstance(res, tuple):
                    feed[f"predicted_{loss}"] = res[1]
        organize_predictions(feed, training_batches[i], losses, ['Eelec'], s.train_ener_per_heavy)
    
    MAE_val = []
    MAE_train = []
    
    for batch in validation_batches:
        for molec in batch:
            MAE_val.append(abs(molec['targets']['Etot'] - molec['predictions']['Etot']))
    
    for batch in training_batches:
        for molec in batch:
            MAE_train.append(abs(molec['targets']['Etot'] - molec['predictions']['Etot']))
    
    print(f"MAE diff of validation is {sum(MAE_val) / len(MAE_val)}")
    print(f"MAE diff of train is {sum(MAE_train) / len(MAE_train)}")
                           
    
    with open("predicted_train.p", "wb") as handle:
        pickle.dump(training_batches, handle)
    
    with open("predicted_validation.p", "wb") as handle:
        pickle.dump(validation_batches, handle)
    
    print("Predictions generated and saved for batches")

if __name__ == "__main__":
    settings = "test_files/settings_refactor_tst_pred.json"
    default = "test_files/refactor_default_tst_pred.json"
    all_models = "fold_molecs_test_8020/Split0/saved_models.p"
    pass_feeds_through(settings, default, all_models)
    
    

