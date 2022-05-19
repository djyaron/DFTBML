# -*- coding: utf-8 -*-
"""
Created on Thu May 19 13:39:59 2022

@author: fhu14
"""

'''
This module handles high level operations related to parsing through and analyzing a large
number of results. This includes os and shutil level operations
'''

#%% Imports, definitions
import os, shutil, json
from .table_ops import write_tables_to_csv, parse_out_table, combine_frames
from .output_parser import save_to_json




#%% Code behind

def generate_master_table(directory_path: str) -> None:
    r"""Given a directory of text file outputs, converts all of them to 
        json dictionaries and creates a master table
    
    Arguments:
        directory_path (str): The path to the directory containing all of the
            output text files
    
    Returns:
        None
    
    Notes:
        The conversion involves two steps:
            1) Convert all text files to json and save those json files
            2) Load in all the json dictionaries
            3) Generate a property table for energy
            4) Generate a property table for the physical targets
            5) Save everything to a csv table
        Specific tables will be generated from different functions in this package
    """
    text_files = os.listdir(directory_path)
    for file in text_files:
        if '.txt' in file:
            save_to_json(os.path.join(directory_path, file))
    json_files = list(filter(lambda x : '.json' in x, os.listdir(directory_path)))
    jdicts = list(map(lambda x : json.load(open(os.path.join(directory_path, x), 'r')), json_files))
    energy_table = parse_out_table(jdicts, "energy")
    property_table = parse_out_table(jdicts, "property")
    write_tables_to_csv([energy_table, property_table], ['master_energy_table.csv', 'master_property_table.csv'],
                        directory_path)
    total_table = combine_frames([energy_table, property_table])
    write_tables_to_csv([total_table], ['energy_property_table.csv'], directory_path)
    print("Finished generating master table")