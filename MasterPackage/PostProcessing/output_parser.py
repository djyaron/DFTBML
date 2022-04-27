# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 23:00:31 2022

@author: fhu14

Module for parsing output text files into dictionaries for use by other
modules. The structure of output text files should remain constant
"""
#%% Imports, definitions
import os, path
from typing import Dict, Union



#%% Code behind

def parse_output_file(file_path: str) -> Dict:
    r"""Parses an output file from analysis into a dictionary with appropriate
        key value pairs.
        
    Arguments:
        file_path (str): The path to the filename to parse into a dictionary
    
    Returns:
        file_dict (Dict): The dictionary containing the parsed results from the 
            file given in file_path
    
    Notes:
        The format of the dictionary will follow that of json files with
            nested dictionaries. All data will be extracted and stored.
    """
    raise NotImplementedError("Implement parse_output_files()")

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
        print("Numerical type conversion failed!")
        return None




#%% Main block
if __name__ == "__main__":
    pass


