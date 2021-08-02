# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 19:31:05 2021

@author: fhu14

Main driver file
"""
#%% Imports, definitions
from InputParser import parse_input_dictionaries, collapse_to_master_settings,\
    inflate_to_dict
import argparse
from DataManager import load_gammas_per_fold, load_config_tracker_per_fold
from Precompute import precompute_stage
from Training import training_loop, exclude_R_backprop, write_output_skf, \
    write_output_lossinfo, check_split_mapping_disjoint, sort_gammas_ctracks
import time, pickle

#%% Code behind
def run_training(settings_filename: str, defaults_filename: str, skf_method: str = 'old'):
    r"""Main driver method
    
    Arguments:
        settings_filename (str): The name of the settings json file
        defaults_filename (str): The name of the defaults json file
        skf_method (str): The method to use when writing skfs. 'old' uses
            the old skfwriter, 'new' uses the new write_skfs() function.
    """
    s_obj = parse_input_dictionaries(settings_filename, defaults_filename)
    opts = inflate_to_dict(s_obj) #opts is a dictionary for DFTBrepulsive to use only. 
    s_obj = collapse_to_master_settings(s_obj)
    
    #Established models and variables
    model_save, variable_save = None, None
    
    init_repulsive = True #Need to initialize the repulsive model
    
    #Analyze the split mapping
    num_splits = len(s_obj.split_mapping.keys())
    check_split_mapping_disjoint(s_obj.split_mapping)
    
    for i in range(num_splits):
        #Do the precompute stage
        all_models, model_variables, training_feeds, validation_feeds, training_dftblsts, validation_dftblsts, losses, all_losses, loss_tracker, training_batches, validation_batches = precompute_stage(s_obj, s_obj.par_dict_name, i, s_obj.split_mapping, model_save, variable_save)
        
        train_gammas, train_c_trackers, valid_gammas, valid_c_trackers = None, None, None, None
        #Exclude the R models if new rep setting
        if s_obj.rep_setting == 'new':
            exclude_R_backprop(model_variables)
            gammas = load_gammas_per_fold(s_obj.top_level_fold_path)
            c_trackers = load_config_tracker_per_fold(s_obj.top_level_fold_path)
            train_gammas, train_c_trackers, valid_gammas, valid_c_trackers = sort_gammas_ctracks(s_obj.split_mapping, i, gammas, c_trackers)
            
        #Run the training loop
        reference_energy_params, loss_tracker, all_models, model_variables, times_per_epoch = training_loop(s_obj, all_models, model_variables, training_feeds, validation_feeds,
                                                                                                        training_dftblsts, validation_dftblsts, training_batches, validation_batches, losses, all_losses, loss_tracker, opts, init_repulsive,
                                                                                                        train_gammas, train_c_trackers, valid_gammas, valid_c_trackers)
        
        init_repulsive = False #Repulsive model should be initialized
        
        write_output_lossinfo(s_obj, loss_tracker, times_per_epoch, i, s_obj.split_mapping, all_models)

        write_output_skf(s_obj, all_models, opts, method = skf_method)
        
        if (model_save is not None) and (variable_save is not None):
            assert(model_save is all_models)
            assert(variable_save is model_variables)
        
        if i == num_splits - 1:
            write_output_skf(s_obj, all_models, opts, method = skf_method)
            return reference_energy_params, loss_tracker, all_models, model_variables, times_per_epoch
        
        model_save = all_models
        variable_save = model_variables
        
        assert(all_models is model_save)
        assert(model_variables is variable_save)


#%% Main block
if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("settings", help = "Name of the settings file for the current hyperparameter settings")
    # parser.add_argument("defaults", help = "Name of the default settings file for the hyperparameters")
    # parser.add_argument("--verbose", help = "increase output verbosity", action = "store_true")
    
    # args = parser.parse_args()
    
    settings = "settings_refactor_tst.json"
    defaults = "refactor_default_tst.json"
    
    start = time.process_time()
    reference_energy_params, loss_tracker, all_models, model_variables, times_per_epoch = run_training(settings, defaults, skf_method = 'new')
    # import pdb; pdb.set_trace()
    end = time.process_time()
    
    print(f"Run took {end - start} seconds")
    print(loss_tracker)
