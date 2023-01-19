# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 16:35:35 2021

@author: fhu14
"""
import json
import os

from InputParser import (collapse_to_master_settings,
                         construct_final_settings_dict,
                         parse_input_dictionaries)

from .helpers import test_data_dir


def test_exception_handling():
    #Test exception handling with missing 'run_id' key in settings
    with open(os.path.join(test_data_dir, "settings_empty.json"), "r") as handle:
        settings = json.load(handle)
    
    with open(os.path.join(test_data_dir, "refactor_default_tst.json"), "r") as handle:
        defaults = json.load(handle)
    
    print("Testing KeyError exception...")
    
    try:
        
        final_dictionary = construct_final_settings_dict(settings, defaults)
        raise ValueError("Call should not have succeeded!")
        
    except KeyError:
        
        print("KeyError test passed, exception was thrown")
        
def test_total_reconstruction():
    #Test total reconstruction
    with open(os.path.join(test_data_dir, "refactor_default_tst.json"), "r") as handle:
        defaults = json.load(handle)
    
    with open(os.path.join(test_data_dir, "settings_empty_runid.json"), "r") as handle:
        settings = json.load(handle)
    
    print("Testing total reconstruction...")
    
    final_dict = construct_final_settings_dict(settings, defaults)
    
    for top_key in final_dict:
        if top_key != 'run_id':
            assert(final_dict[top_key] == defaults[top_key])
    
    print("Total reconstruction test passed")
    
def test_partial_reconstruction():
    #Test partial reconstruction
    test_keys = ["loaded_data_fields", "model_settings"]
    
    with open(os.path.join(test_data_dir, "refactor_default_tst.json"), "r") as handle:
        defaults = json.load(handle)
    
    with open(os.path.join(test_data_dir, "settings_refactor_partial.json"), "r") as handle:
        settings = json.load(handle)
        
    print("Testing partial reconstruction...")
        
    final_dict = construct_final_settings_dict(settings, defaults)
    
    assert(final_dict["loaded_data_fields"] == defaults["loaded_data_fields"])
    assert(final_dict["model_settings"] == defaults["model_settings"])
    
    print("Partial reconstruction test passed")
    
def test_inner_translation():
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
    with open(os.path.join(test_data_dir, "refactor_default_tst.json"), "r") as handle:
        defaults = json.load(handle)
    
    with open(os.path.join(test_data_dir, "settings_refactor_differ.json"), "r") as handle:
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
    
def test_object_conversion():
    #Using these same settings and default dictionaries, test the resulting settings object
    final_settings_obj = parse_input_dictionaries(os.path.join(test_data_dir, "settings_refactor_differ.json"),
                                                  os.path.join(test_data_dir, "refactor_default_tst.json"))
    
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

def test_object_addition():
    #Checking that operator overload is functioning correctly.
    final_settings_obj = parse_input_dictionaries(os.path.join(test_data_dir, "settings_refactor_differ.json"),
                                                  os.path.join(test_data_dir, "refactor_default_tst.json"))
    
    print("Testing the addition operator overload for settings objects...")
    
    starting_obj = final_settings_obj.batch_data_fields
    starting_obj += final_settings_obj.loaded_data_fields
    
    assert(starting_obj.allowed_Zs == [1, 6])
    assert(starting_obj.num_per_batch == 1)
    assert(starting_obj.prop_train == 2.5)
    assert(starting_obj.shuffle == [1,1])
    
    assert(starting_obj.loaded_data)
    assert(starting_obj.run_check)
    
    starting_obj_keys = set(starting_obj.__dict__.keys())
    batch_data_field_keys = set(final_settings_obj.batch_data_fields.__dict__.keys())
    loaded_data_keys = set(final_settings_obj.loaded_data_fields.__dict__.keys())
    total_keys = batch_data_field_keys.union(loaded_data_keys)
    assert(starting_obj_keys.difference(total_keys) == set() and total_keys.difference(starting_obj_keys) == set())
    
    print("Operator overload test passed.")
    
def test_collapse_master():
    #Checking that operator overload is functioning correctly.
    print("Testing the addition operator overload and run_id...")
    final_settings_obj = parse_input_dictionaries(os.path.join(test_data_dir, "settings_refactor_differ.json"),
                                                  os.path.join(test_data_dir, "refactor_default_tst.json"))
    
    final_settings_obj = collapse_to_master_settings(final_settings_obj)
    assert('run_id' in final_settings_obj.__dict__)
    print(final_settings_obj.__dict__.keys())
    print(final_settings_obj.__dict__['run_id'])
    print("run_id presence test passed.")
    
    
def run_parser_tests():
    test_exception_handling()
    test_total_reconstruction()
    test_partial_reconstruction()
    test_inner_translation()
    test_object_conversion()
    test_object_addition()
    test_collapse_master()

if __name__ == "__main__":
    run_parser_tests()
    
    
    
    
    
    
            
        
        