# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 23:00:31 2022

@author: fhu14

Module for parsing output text files into dictionaries for use by other
modules. The structure of output text files should remain constant
"""
#%% Imports, definitions
import os, path, json
from typing import Dict, Union

#%% Code behind

def sanitize_keys(key: str) -> str:
    r"""Santiizes the keys and replaces all whitespace and tab characters with
        underscores
    
    Arguments:
        key (str): The key to sanitize
    
    Returns:
        clean_key (str): The cleaned key
    
    Notes: 
        Sanitizing here consists of removing all white space characters (such as tabs and spaces)
        with underscore characters
    """
    key = key.replace(".", "") #Remove periods as they might become problematic later
    key = key.replace("\t", " ")
    key = key.replace(" ", "_")
    return key

def numerical_type_correction(val: str) -> Union[int, float, None]:
    r"""Converts a value from a string into a numerical value
    
    Arguments:
        val (str): The value to convert to a numerical value
    
    Returns:
        Union[int, float]: Converted value
    
    Raises:
        Exception: If the value cannot be converted to numerical. In this case
            None is returned instead
    """
    try:
        return float(val)
    except:
        return None

def output_file_dict_conversion(file_path: str) -> Dict:
    r"""Takes in an output file form the analysis code and converts it to a json 
        dictionary format for easy parsing later on
    
    Arguments: 
        file_path (str): The file that needs to be converted to a json file format,
            should be a .txt file
    
    Returns:
        new_dict (Dict): Dictionary containing the parsed information contained in
            the file specified at file_path
    
    Notes:
        A json format is chosen because it is a very easy to parse json files
        using the default python utilities
    """
    with open(file_path, 'r') as handle:
        content = handle.readlines()
        new_dict = {}
        curr_dict = None
        good_elems = [elem for elem in content if ":" in elem]
        for line in good_elems:
            #.strip() removes newline characters that may occur at the end of a line
            split_line = line.strip().split(":")
            key, value = split_line[0], split_line[1]
            key = sanitize_keys(key)
            #Handle the experiment label case
            if key.strip() == "Experiment_label":
                new_dict[key] = value
            #Case of a section header
            elif value == "": 
                new_dict[key] = {}
                curr_dict = new_dict[key]
            else:
                key, value = key.strip(), value.strip()
                converted_value = numerical_type_correction(value)
                value = converted_value if (converted_value is not None) else value
                curr_dict[key] = value
        return new_dict

def save_to_json(file_path: str, save_location: str = None) -> None:
    r"""Saves a file_path to a json (nested dictionary) format using the built-in
        Python utilities
    
    Arguments:
        file_path (str): The path to the current file to deduce
        save_location (str): Parameter to specify the location to save the json
            file. Defaults to None, in which case the new json file is saved to the 
            same location as the original file. If this is passed in, then it should
            contain the full path along with the filename (e.g. ...\head\newfile.json)
    
    Returns:
        None
    
    Notes:
        This function will save the contents of the given file, as a json file,
        in the same location as the original file. The given file_path is
        analyzed to find the save location.
        
        Some caution when using os.path.split() and os.path.join(): different 
        operating systems (Linux vs windows) have different path delimiting 
        characters ('/' vs '\'). Python is smart enough to treat '/' and '\' as the 
        same, but this behavior is not guaranteed in all cases. For consistency, 
        run all this code on the same operating system.
    """
    if save_location is None:
        head, tail = os.path.split(file_path)
        original_filename = tail.split(".")[0]
        new_filename = original_filename + ".json"
        save_location = os.path.join(head, new_filename)
    parsed_dictionary = output_file_dict_conversion(file_path)
    with open(save_location, 'w+') as handle:
        json.dump(parsed_dictionary, handle, indent = 4)
    print(f"{file_path} file successfully converted to json and saved to {save_location}")


#%% Main block
if __name__ == "__main__":
    pass


