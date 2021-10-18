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
from MasterConstants import Model #For debugging
from Spline import get_dftb_vals #For debugging
import torch #For debugging
from InputLayer import Input_layer_pairwise_linear
import os, shutil

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
        
        print(f"Number of training feeds: {len(training_feeds)}")
        print(f"Number of validation feeds: {len(validation_feeds)}")
        
        ### Debugging code for testing out proper formulation of the on-atom G models ( e.g. Model("G", (7,), 'ps') )
        g_mods = [mod for mod in all_models if (not isinstance(mod, str)) and (len(mod.Zs) == 1) and (mod.oper == "G")]
        match = [mod for mod in g_mods if mod.orb[0] == mod.orb[1]]
        non_match = [mod for mod in g_mods if mod.orb[0] != mod.orb[1]]
        assert(len(match) + len(non_match) == len(g_mods))
        assert(len(g_mods) == 10 and len(match) == 7)
        for mod_spec in match:
            assert(mod_spec in model_variables)
            assert(mod_spec in all_models)
        for mod_spec in non_match:
            assert(mod_spec not in model_variables)
            assert(mod_spec in all_models)
        for mod_spec in non_match:
            orb1, orb2 = mod_spec.orb[0], mod_spec.orb[1]
            orb1, orb2 = orb1 * 2, orb2 * 2
            m1 = Model("G", mod_spec.Zs, orb1)
            m2 = Model("G", mod_spec.Zs, orb2)
            variables = all_models[mod_spec].get_variables()
            assert(variables[0] is model_variables[m1])
            assert(variables[1] is model_variables[m2])
        
        tst_feed_dict = {
            "zero_indices" : torch.tensor([0,1]),
            "nonzero_indices" : torch.tensor([]),
            "nonzero_distances" : torch.tensor([])
            }
        for mod_spec in non_match:
            mod_pred_value = all_models[mod_spec].get_values(tst_feed_dict)[0].item()
            dftb_val = get_dftb_vals(mod_spec, s_obj.par_dict_name)[0]
            assert(mod_pred_value == dftb_val)
        
        all_models_keys = set(all_models.keys())
        model_variables_keys = set(model_variables.keys())
        for elem in all_models_keys - model_variables_keys:
            assert(elem.oper == "G" and (len(elem.Zs) == 2 or (elem.orb[0] != elem.orb[-1])))
        for elem in model_variables_keys - all_models_keys:
            assert(elem.oper == "S" and 'inflect' in elem.orb)
        assert(len(all_models_keys - model_variables_keys) == 31)
        assert(len(model_variables_keys - all_models_keys) == 34)
        
        for model in all_models:
            if isinstance(all_models[model], Input_layer_pairwise_linear):
                if model.oper == "R":
                    assert(all_models[model].cutoff == 2.2)
                    assert(all_models[model].pairwise_linear_model.xknots[-1] == 2.2)
                else:
                    assert(all_models[model].cutoff == 4.5)
                    assert(all_models[model].pairwise_linear_model.xknots[-1] == 4.5)
        
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
    
    print("Copying and saving split information")
    
    #Some book keeping code
    s_obj = parse_input_dictionaries(settings, defaults)
    s_obj = collapse_to_master_settings(s_obj)
    
    dset_source = s_obj.top_level_fold_path
    results_dest = s_obj.run_id
    
    num_splits = len(s_obj.split_mapping)
    for i in range(num_splits):
        src = os.path.join(dset_source, f"Split{i}")
        dst = os.path.join(results_dest, f"Split{i}")
        #TODO: Fix problem of existing directory
        if os.path.exists(dst):
            print("Removing existing directory")
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    
    print("Copying settings file")
    
    shutil.copy("settings_refactor_tst.json", os.path.join(results_dest, "settings_refactor_tst.json"))
    
    print("All information copied")
    
    
    
    
        
        
