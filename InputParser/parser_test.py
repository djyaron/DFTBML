# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 16:35:35 2021

@author: fhu14
"""

from input_parser import construct_final_settings_dict,\
    parse_input_dictionaries
import json

if __name__ == "__main__":
    
    #Test exception handling with missing 'run_id' key in settings
    with open("test_files/settings_empty.json", "r") as handle:
        settings = json.load(handle)
    
    with open("test_files/refactor_default_tst.json", "r") as handle:
        defaults = json.load(handle)
    
    print("Testing KeyError exception...")
    
    try:
        
        final_dictionary = construct_final_settings_dict(settings, defaults)
        raise ValueError("Call should not have succeeded!")
        
    except KeyError:
        
        print("KeyError test passed, exception was thrown")
    
    #Test total reconstruction
    with open("test_files/refactor_default_tst.json", "r") as handle:
        defaults = json.load(handle)
    
    with open("test_files/settings_empty_runid.json", "r") as handle:
        settings = json.load(handle)
    
    print("Testing total reconstruction...")
    
    final_dict = construct_final_settings_dict(settings, defaults)
    
    for top_key in final_dict:
        if top_key != 'run_id':
            assert(final_dict[top_key] == defaults[top_key])
    
    print("Total reconstruction test passed")
    
    #Test partial reconstruction
    test_keys = ["loaded_data_fields", "model_settings"]
    
    with open("test_files/refactor_default_tst.json", "r") as handle:
        defaults = json.load(handle)
    
    with open("test_files/settings_refactor_partial.json", "r") as handle:
        settings = json.load(handle)
        
    print("Testing partial reconstruction...")
        
    final_dict = construct_final_settings_dict(settings, defaults)
    
    assert(final_dict["loaded_data_fields"] == defaults["loaded_data_fields"])
    assert(final_dict["model_settings"] == defaults["model_settings"])
    
    print("Partial reconstruction test passed")
    
    #Testing proper translation of inner values
    '''
    The test keys here will be as follows, where every value that should
    equal the default value is missing from the settings file:
        batch_data_fields -> allowed_Zs = [1, 6] (settings)
        batch_data_fields -> num_per_batch = 1 (default) 
        batch_data_fields -> prop_train = 2.5 (settings)
        batch_data_fields -> shuffle = [1, 1] (default) 
        
        loaded_data_fields -> loaded_data = true (default) 
        loaded_data_fields -> run_check = true (settings)
        
        model_settings -> num_knots = 30 (settings)
        model_settings -> spline_deg = 3 (default)
        model_settings -> universal_high = 15.0 (settings)
        model_settings -> include_inflect = true (default)
        
        training_settings -> target_accuracy_energy = 7000 (settings)
        training_settings -> target_accuracy_monotonic = 1000 (default)
        training_settings -> target_accuracy_charges = 50 (settings)
        training_settings -> ragged_dipole = false (settings)
        training_settings -> debug = false (default)
    '''
    with open("test_files/refactor_default_tst.json", "r") as handle:
        defaults = json.load(handle)
    
    with open("test_files/settings_refactor_differ.json", "r") as handle:
        settings = json.load(handle)
    
    print("Testing inner value translation")
    
    final_dict = construct_final_settings_dict(settings, defaults)
    
    assert(final_dict["batch_data_fields"]["allowed_Zs"] == [1, 6])
    assert(final_dict["batch_data_fields"]["num_per_batch"] == 1)
    assert(final_dict["batch_data_fields"]["prop_train"] == 2.5)
    assert(final_dict["batch_data_fields"]["shuffle"] == [1,1])
    
    assert(final_dict["loaded_data_fields"]["loaded_data"])
    assert(final_dict["loaded_data_fields"]["run_check"])
    
    assert(final_dict["model_settings"]["num_knots"] == 30)
    assert(final_dict["model_settings"]["spline_deg"] == 3)
    assert(final_dict["model_settings"]["universal_high"] == 15.0)
    assert(final_dict["model_settings"]["include_inflect"])
    
    assert(final_dict["training_settings"]["target_accuracy_energy"] == 7000)
    assert(final_dict["training_settings"]["target_accuracy_monotonic"] == 1000)
    assert(final_dict["training_settings"]["target_accuracy_charges"] == 50)
    assert(final_dict["training_settings"]["ragged_dipole"] == False)
    assert(final_dict["training_settings"]["debug"] == False)
    
    assert(final_dict["optim_sched_settings"] == defaults["optim_sched_settings"])
    assert(final_dict["tensor_settings"] == defaults["tensor_settings"])
    assert(final_dict["skf_settings"] == defaults["skf_settings"])
    assert(final_dict["repulsive_settings"] == defaults["repulsive_settings"])
    assert(final_dict["run_id"] == "old_rep_setting_run")
    
    print("Inner value translation test passed")
    
    #Using these same settings and default dictionaries, test the resulting settings object
    #Remember to command out the par_dict parsing portion before running this next test!
    final_settings_obj = parse_input_dictionaries("test_files/settings_refactor_differ.json",
                                                  "test_files/refactor_default_tst.json")
    
    print("Testing the conversion from dictionary to object...")
    
    assert(final_settings_obj.batch_data_fields.allowed_Zs == [1, 6])
    assert(final_settings_obj.batch_data_fields.num_per_batch == 1)
    assert(final_settings_obj.batch_data_fields.prop_train == 2.5)
    assert(final_settings_obj.batch_data_fields.shuffle == [1,1])
    
    assert(final_settings_obj.loaded_data_fields.loaded_data)
    assert(final_settings_obj.loaded_data_fields.run_check)
    
    assert(final_settings_obj.model_settings.num_knots == 30)
    assert(final_settings_obj.model_settings.spline_deg == 3)
    assert(final_settings_obj.model_settings.universal_high == 15.0)
    assert(final_settings_obj.model_settings.include_inflect)
    
    assert(final_settings_obj.training_settings.target_accuracy_energy == 7000)
    assert(final_settings_obj.training_settings.target_accuracy_monotonic == 1000)
    assert(final_settings_obj.training_settings.target_accuracy_charges == 50)
    assert(final_settings_obj.training_settings.ragged_dipole == False)
    assert(final_settings_obj.training_settings.debug == False)
    
    print("Object conversion test passed")
    
    
    
    
    
    
    
            
        
        