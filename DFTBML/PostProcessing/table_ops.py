# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 23:30:09 2022

@author: fhu14

Module for handling table-level operations for computing deltas and other
elements. This will take advantage of pandas functionality. Dictionaries 
will be generated from the json files that output files are parsed into
"""
#%% Imports, definitions

import pandas as pd
import os, shutil, json
from typing import Dict, List, Union
DataFrame = pd.DataFrame

#%% Code behind

def parse_out_experiment_name(exp_label: str) -> str:
    r"""Parses out the experiment name from an experiment label string
    
    Arguments:
        exp_label (str): The experiment label to parse out
    
    Returns:
        experiment_name (str): The name of the experiment
    """
    return os.path.split(exp_label)[-1]

def turn_keys_readable(key: str) -> str:
    r"""Replaces underscores with spaces so that keys are human readable
    """
    return key.replace("_", " ")

def parse_out_table(json_dicts: List[Dict], target: str) -> DataFrame:
    r"""Parses out the energy portion of the json dictionary and converts it into 
        a dataframe
    
    Arguments:
        json_dicts (List[Dict]): The list of json dictionaries to construct the 
            energy table from
        target (str): The target of interest, either energy or the physical properties.
            One of 'energy' or 'property'
    
    Returns:
        energy_frame (DataFrame): The dataframe containing the information for the 
            target values
    """
    tst_dict = json_dicts[0]
    if target == 'energy':
        target_keys = tst_dict['Energy_error'].keys()
    elif target == 'property':
        target_keys = tst_dict['Error_for_other_physical_targets'].keys()
    master_dict = {}
    master_dict['Experiment Name'] = []
    for key in target_keys:
        master_dict[turn_keys_readable(key)] = []
    for jdict in json_dicts:
        experiment_name = parse_out_experiment_name(jdict['Experiment_label'])
        master_dict['Experiment Name'].append(experiment_name)
        if target == 'energy':
            original_jdict_section = jdict['Energy_error'].items()
        elif target == 'property':
            original_jdict_section = jdict['Error_for_other_physical_targets'].items()
        for key, value in original_jdict_section:
            master_dict[turn_keys_readable(key)].append(value)
    frame = pd.DataFrame.from_dict(master_dict)
    return frame

def combine_frames(frames: List[DataFrame]) -> DataFrame:
    r"""Combines the list of given frames into one cohesive data table
    
    Arguments:
        frames (List[DataFrame]): The list of frames to combine
    
    Returns:
        total_frame (DataFrame): The combined dataframe formed from the constituent
            frames
    
    Notes:
        The frames are stacked together horizontally in the order that they are given in 
            the list. All frames should have the same column for experiment names!
    """
    first_frame_tst = frames[0]
    for other_frame in frames:
        assert(all(other_frame['Experiment Name'] == first_frame_tst['Experiment Name']))
    #Remove the Experiment Name column from each subsequent frame after the first one
    for i in range(len(frames)):
        #Specify axis = 1 for columns according to documentation https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html
        if i > 0:
            frames[i] = frames[i].drop(['Experiment Name'], axis = 1)
    total_frame = pd.concat(frames, axis = 1)
    # #Solution for removing duplicate columns taken from https://stackoverflow.com/questions/14984119/python-pandas-remove-duplicate-columns
    # total_frame = temp_frame.loc[:,~temp_frame.columns.duplicated()]
    return total_frame

def write_tables_to_csv(frames: List[DataFrame], labels: List[str], save_location: str) -> None:
    r"""Writes the output table to a csv file at the specified save_location
    
    Arguments:
        frames (List[DataFrame]): The series of frames to save to csv files
        labels (List[str]): The name for each of the tables to be saved, should have 
            .csv extension in it
        save_location (str): The directory/location to save the tables to
    
    Returns:
        None
    
    Notes:
        The frames and labels list should be the same length and there should be
        a 1-to-1 correspondence between the elements of the two lists. The save location
        should be a directory (not a full path with the file names at the end)
    """
    assert(len(frames) == len(labels))
    assert(all('.csv' in x for x in labels))
    for i in range(len(frames)):
        curr_frame, curr_label = frames[i], labels[i]
        filename = os.path.join(save_location, curr_label)
        print(f"Currently saving csv {filename}")
        #Omit the index column
        curr_frame.to_csv(filename, index = False)
    print("All tables are successfully saved to csv format")

def generate_deltas_table(table1: DataFrame, table2: DataFrame) -> DataFrame:
    r"""Generates a comparison between two tables by subtracting/calculating differences
        between the values 
    
    Arguments:
        table1 (DataFrame): The first table
        table2 (DataFrame): The second table
    
    Returns:
        diff_table (DataFrame): The dataframe with values generated by computing 
            the difference between the two tables. This is computed as table2 - table1.
        
    Notes:
        Subtraction is a binary operation and only works between numerical values. 
        For this reason, values with non-numerical types are ommitted from the 
        final table. Both tables should have the same set of column headers; 
        otherwise, the subtraction will not work. Also, the experiment names
        should all be consistent
    """
    assert(all(table1.columns == table2.columns))
    assert(all(table1['Experiment Name'] == table2['Experiment Name']))
    good_columns = []
    for col, dtype in table1.dtypes.iteritems():
        #Pull out all the columns that are not objects (i.e., numerical)
        if str(dtype) != 'object':
            good_columns.append(col)
    new_table1 = table1[good_columns].copy()
    new_table2 = table2[good_columns].copy()
    new_frame = new_table2 - new_table1
    #Add back in the Experiment Name column at the beginning
    new_frame.insert(0, "Experiment Name", table1['Experiment Name'].tolist())
    return new_frame

#%% Main block
if __name__ == "__main__":
    pass

