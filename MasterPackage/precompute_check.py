# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 21:50:08 2021

@author: fhu14
"""


import json, os
from typing import Dict

def assert_unique_keys(d: Dict) -> bool:
    r"""Checks that for a settings file, all the dictionary keys are unique under
        each section (no duplicates).
    
    Arguments:
        d (Dict): The dictionary to check
    
    Returns:
        (bool): Whether the dictionary passes the duplicate key check. True f
            there are no duplicate keys, False otherwise
    """
    all_inner_keys = []
    for key in d:
        if isinstance(d[key], dict):
            for inner_key in d[key]:
                all_inner_keys.append(inner_key)
    return len(set(all_inner_keys)) == len(all_inner_keys)

def precompute_settings_check(settings_filename: str) -> None:
    r"""Performs a quick check on the values of the settings file to ensure that
        everything is correct and ready to go.
    
    Arguments:
        settings_filename (str): The name of the settings json file used
            for dataset precomputation
    
    Returns:
        None
    
    Raises:
        AssertionError if certain values are not properly assigned
    
    Notes:
        Certain fields without a easily checkable valuue (e.g. low-end cutoff dictionary)
        need to be checked by hand, but this takes care of all other fields
        with simple values.
    """
    full_path = os.path.join(os.getcwd(), settings_filename)
    with open(full_path, 'r') as handle:
        d = json.load(handle)
    
    assert(assert_unique_keys(d))
    print("Passed key uniqueness check!")
    
    assert(d['training_settings']['par_dict_name'] == "Auorg_1_1")
    assert(d['training_settings']['train_ener_per_heavy'] == True)
    assert(d['training_settings']['opers_to_model'] == ["H", "R", "G", "S"])
    
    assert(d['batch_data_fields']['allowed_Zs'] == [1,6,7,8])
    assert(d['batch_data_fields']['num_per_batch'] == 10)
    
    assert(d['training_settings']['losses'] == ["Etot", "dipole", "charges", "convex", "smooth"])
    assert('convex' in d['training_settings']['losses'])
    assert(d['training_settings']['target_accuracy_energy'] == 6270)
    assert(d['training_settings']['target_accuracy_dipole'] == 100)
    assert(d['training_settings']['target_accuracy_charges'] == 1)
    #assert(d['training_settings']['target_accuracy_charges'] == d['training_settings']['target_accuracy_dipole'])
    assert(d['training_settings']['target_accuracy_convex'] == 1000)
    assert(d['training_settings']['target_accuracy_monotonic'] == 1000)
    assert(d['training_settings']['target_accuracy_smooth'] == 10)
    assert(d['training_settings']['nepochs'] == 2500)
    
    assert(d['tensor_settings']['tensor_device'] == 'cpu')
    assert(d['tensor_settings']['tensor_dtype'] == 'double')
    assert(d['training_settings']['reference_energy_starting_point'] == [-2.30475824e-01, -3.63327215e+01, -5.23253002e+01, -7.18450781e+01,
  1.27026973e-03]) #Need to hand-check
    
    assert(d['model_settings']['low_end_correction_dict'] == {
    "1,1" : 0.500,
    "6,6" : 1.04,
    "1,6" : 0.602,
    "7,7" : 0.986,
    "6,7" : 0.948,
    "1,7" : 0.573,
    "1,8" : 0.599,
    "6,8" : 1.005,
    "7,8" : 0.933,
    "8,8" : 1.062
    }) #Need to hand-check
    assert(d['model_settings']['universal_high'] == 10.0)
    assert(d['model_settings']['spline_mode'] == 'non-joined')
    assert(d['model_settings']['spline_deg'] == 5)
    assert(d['training_settings']['debug'] == False)
    assert(d['model_settings']['num_knots'] == 100)
    assert(d['model_settings']['buffer'] == 0.0)
    assert(d['model_settings']['joined_cutoff'] == 4.5)
    assert(d['model_settings']['cutoff_dictionary'] is not None)
    assert(d['model_settings']['off_diag_opers'] == ["G"])
    assert(d['model_settings']['include_inflect'] == True) 
    
    assert(d['repulsive_settings']['opts'] == {
        "nknots" : 50, #Try running with 50 knots over the short cutoff
        "cutoff" : "short", #Try running with modified short cutoff
        "deg" : 3,
        "bconds" : "vanishing",
        "constr" : "+2"
    })

    assert(d['repulsive_settings']['rep_setting'] == 'new')
    assert(d['repulsive_settings']['rep_integration'] == 'external')
    
    assert(d['skf_settings'] == {

    "skf_extension" : "",
    "skf_ngrid" : 50,
    "skf_strsep" : "  ",
    "spl_ngrid" : 500

    })
    
    print("Need to check reference_energy_starting_point, low_end_correction_dict, and cutoff_dictionary. Everything else is good!")
    
