# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 13:39:19 2021

@author: fhu14

This module is used for taking in the settings file and performing the 
necessary corrections on the different blocks of the file. This includes
correction of dictionary keys and values for the torch tensor data type and 
device settings. This module will be a collection of global functions
rather than a class, and is the base module that contains the settings object.

The final settings object will be one where the top-level keys (e.g. batch_data_fields,
loaded_data_fields) map to groups of settings for different purposes. So 
loaded_data_fields would correspond to values relevant for loading in data for 
training the network. 
"""

#%% Imports
from typing import Dict, List, Union
import torch
import json
import importlib

#%% Code
class Settings:
    def __init__(self, settings_dict: Dict) -> None:
        r"""Generates a Settings object from the given dictionary
        
        Arguments:
            settings_dict (Dict): Dictionary containing key value pairs for the
                current hyperparmeter settings
        
        Returns:
            None
        
        Notes: Using an object rather than a dictionary is easier since you can
            just do settings.ZZZ rather than doing the bracket notation and the quotes.
        """
        for key in settings_dict:
            setattr(self, key, settings_dict[key])
    
    def __add__(self, other_settings_obj):
        r"""Operator override for addition of two settings objects, creates a 
            new object that contains the attributes of the constituting objects.
        """
        assert(isinstance(other_settings_obj, Settings))
        for key in other_settings_obj.__dict__:
            setattr(self, key, other_settings_obj.__dict__[key])
        return self
    
def convert_key_to_num(elem: Dict) -> Dict:
    return {int(k) : v for (k, v) in elem.items()}

def update_pytorch_arguments(settings: Settings) -> None:
    r"""Updates the arguments in the settings object to the corresponding 
        PyTorch types
        
    Arguments:
        settings (Settings): The settings object representing the current set of 
            hyperparameters
    
    Returns:
        None
        
    Notes: First checks if a CUDA-capable GPU is available. If not, it will
        default to using CPU only.
    """
    if settings.tensor_dtype == 'single':
        print("Tensor datatype set as single precision (float 32)")
        settings.tensor_dtype = torch.float
    elif settings.tensor_dtype == 'double':
        print("Tensor datatype set as double precision (float 64)")
        settings.tensor_dtype = torch.double
    else:
        raise ValueError("Unrecognized tensor datatype")
        
    num_gpus = torch.cuda.device_count()
    if settings.tensor_device == 'cpu':
        print("Tensor device set as cpu")
        settings.tensor_device = 'cpu'
    elif num_gpus == 0 or (not (torch.cuda.is_available())):
        print("Tensor device set as cpu because no GPUs are available")
        settings.tensor_device = 'cpu'
    else:
        gpu_index = int(settings.device_index)
        if gpu_index >= num_gpus:
            print("Invalid GPU index, defaulting to CPU")
            settings.tensor_device = 'cpu'
        else:
            print("Valid GPU index, setting tensor device to GPU")
            #I think the generic way is to do "cuda:{device index}", but not sure about this
            settings.tensor_device = f"cuda:{gpu_index}"
            print(f"Used GPU name: {torch.cuda.get_device_name(settings.tensor_device)}")

def dictionary_tuple_correction(input_dict: Dict) -> Dict:
    r"""Performs a correction on the input_dict to convert from string to tuple
    
    Arguments:
        input_dict (Dict): The dictionary that needs correction
    
    Returns:
        new_dict (Dict): Dictionary with the necessary corrections
            applied
    
    Notes: The two dictionaries that need correction are the cutoff dictionary and the
        range correction dictionary for model_range_dict. For the dictionary used to 
        correct model ranges, the keys are of the form "elem1,elem2" where elem1 and elem2 
        are the atomic numbers of the first and second element, respectively. These
        need to be converted to a tuple of the form (elem1, elem2).
        
        For the dictionary used to specify cutoffs (if one is provided), the format 
        of the keys is "oper,elem1,elem2" where oper is the operator of interest and
        elem1 and elem2 are again the atomic numbers of the elements of interest. This
        will be converted to a tuple of the form (oper, (elem1, elem2)). A check is 
        performed between these cases depending on the number of commas.
        
        The reason for this workaround is because JSON does not support tuples. 
        An alternative would have been to use a string representation of the tuple
        with the eval() method. 
    """
    num_commas = list(input_dict.keys())[0].count(",")
    if num_commas == 0:
        print("Dictionary does not need correction")
        print(input_dict)
        return input_dict #No correction needed
    new_dict = dict()
    #Assert key consistency in the dictionary
    for key in input_dict:
        assert(key.count(",") == num_commas)
        key_splt = key.split(",")
        if len(key_splt) == 2:
            elem1, elem2 = int(key_splt[0]), int(key_splt[1])
            new_dict[(elem1, elem2)] = input_dict[key]
        elif len(key_splt) == 3:
            oper, elem1, elem2 = key_splt[0], int(key_splt[1]), int(key_splt[2])
            new_dict[(oper, (elem1, elem2))] = input_dict[key]
        else:
            raise ValueError("Given dictionary cannot go through tuple correction!")
    return new_dict

def construct_final_settings_dict(settings_dict: Dict, default_dict: Dict) -> Dict:
    r"""Generates the final settings dictionary based on the input settings file and the
        defaults file
    
    Arguments:
        settings_dict (Dict): Dictionary of the user-defined hyperparameter settings. 
            Read from the settings json file
        default_dict (Dict): Dictionary of default hyperparameter settings. 
            Read from the defaults json file
    
    Returns:
        final_settings_dict (Dict): The final dictionary containing the settings
            to be used for the given run over all hyperparameters
    
    Notes: If something is not specified in the settings file, then the
        default value is pulled from the default_dict to fill in. For this reason,
        the default_dict contains the important keys and the settings file
        can contain a subset of these important keys. The settings file will include
        some information that is not found in the default dictionary, such as the 
        name given to the current run and the directory for saving the skf files at the end
        
        The settings_dict must contain the run_id key. With the refactored format
        of the settings file, another leve of looping is necessary for each 
        key.
    """
    final_dict = dict()
    for top_key in default_dict:
        if (top_key not in settings_dict):
            final_dict[top_key] = default_dict[top_key]
        else:
            final_dict[top_key] = dict()
            for inner_key in default_dict[top_key]:
                if (inner_key not in settings_dict[top_key]):
                    final_dict[top_key][inner_key] = default_dict[top_key][inner_key]
                else:
                    final_dict[top_key][inner_key] = settings_dict[top_key][inner_key]
    if ('run_id' not in settings_dict):
        raise KeyError("Settings dictionary must contain run_id!")
    else:
        final_dict['run_id'] = settings_dict['run_id']
    return final_dict

def parse_input_dictionaries(settings_filename: str, default_filename: str) -> Settings:
    r"""Parses and assembles the final settings object to be used throughout the 
        rest of the code.
    
    Arguments:
        settings_filename (str): The filename corresponding to the settings file.
        default_filename (str): The filename corresponding to the defaults file.
    
    Returns:
        final_settings (Settings): The combined final settings object.
    
    Notes: None
    """
    with open(settings_filename, 'r') as handle:
        settings_dict = json.load(handle)
    
    with open(default_filename, 'r') as handle:
        default_dict = json.load(handle)
    
    final_dictionary = construct_final_settings_dict(settings_dict, default_dict)
    #Parse out the parameter dictionary using import lib
    par_dict_name = final_dictionary['training_settings']['par_dict_name']
    module = importlib.import_module(par_dict_name)
    #It is assumed that every parameter dictionary loader has a method included
    #   called ParDict()
    final_dictionary['training_settings']['par_dict_name'] = module.ParDict()
    
    #Perform the dictionary tuple corrections
    if final_dictionary['model_settings']['cutoff_dictionary'] is not None:
        final_dictionary['model_settings']['cutoff_dictionary'] = \
            dictionary_tuple_correction(final_dictionary['model_settings']['cutoff_dictionary'])
    if final_dictionary['model_settings']['low_end_correction_dict'] is not None:
        final_dictionary['model_settings']['low_end_correction_dict'] = \
            dictionary_tuple_correction(final_dictionary['model_settings']['low_end_correction_dict'])
    if final_dictionary["training_settings"]["split_mapping"] is not None:
        final_dictionary["training_settings"]["split_mapping"] = \
            convert_key_to_num(final_dictionary["training_settings"]["split_mapping"])
    
    
    #Convert to a settings object
    for key in final_dictionary:
        if (isinstance(final_dictionary[key], dict)):
            final_dictionary[key] = Settings(final_dictionary[key])
    
    #Update pytorch arguments
    update_pytorch_arguments(final_dictionary['tensor_settings'])
    
    #Final settings object
    final_settings = Settings(final_dictionary)
    
    return final_settings

def collapse_to_master_settings(final_settings: Settings) -> Settings:
    r"""Combines the individual settings objects into a master settings object
        with all keys exposed at top level.
    
    Arguments:
        final_settings (Settings): The settings object generated from
            parse_input_dictionaries
    
    Returns:
        collapsed_settings (Settings): The combined settings object.
    """
    collapsed_settings = None
    for sub_field in final_settings.__dict__:
        if sub_field != 'run_id':
            if collapsed_settings == None:
                collapsed_settings = final_settings.__dict__[sub_field]
            else:
                collapsed_settings += final_settings.__dict__[sub_field]
    collapsed_settings.run_id = final_settings.__dict__['run_id']
    return collapsed_settings

#%% Main block
if __name__ == "__main__":
    parse_input_dictionaries('settings_refactor.json', 'refactor_default.json')