# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 14:20:24 2022

@author: fhu14
"""

#%% Imports, definitions
import pickle
import os
from typing import List
from functools import reduce



#%% Code behind

def infer_skf_elements(direc: str) -> List[str]: 
    r"""Finds out the elements of an skf set
    
    Arguments:
        direct (str): The path to the directory to infer the skf set from
    
    Returns:
        elems (List[str]): The list of elements contained within an skf set
    """
    all_files = os.listdir(direc)
    skf_files_names = list(filter(lambda x : '.skf' in x, all_files))
    skf_elem_pairs = list(map(lambda x : x.split(".")[0].split("-"), skf_files_names))
    flat_elems = reduce(lambda x, y : x + y, skf_elem_pairs)
    return list(set(flat_elems))

def convert_skf_file_to_data(skf_file_name: str) -> List[str]:
    r"""Converts an skf file into a 2D list format for easy handling
    
    Arguments:
        skf_file_name (str): Path to the skf file to read in
    
    Returns:
        skf_data (List[List[str]]): The data contained in the skf file formatted
            into a 1D list of strings where each string represents a line.
    """
    with open(skf_file_name, 'r') as handle:
        content = handle.readlines()
    return content

def shift_second_line(skf_file_data: List[str], const: float,
                      no_d_check: bool = False, num_spaces: int = 4) -> None:
    r"""Performs the arbitrary shift to the atomic on-site energies by the
        specified const
    
    Arguments:
        skf_file_data (List[str]): The skf file data represented as a list of 
            strings
        const (float): The floating point value to correct the skf_file_data
            by
        no_d_check (bool): If True, a check is performed to ensure that the 
            Ed on-site energy is 0. Defaults to False. 
        num_spaces (int): The number of spaces used to separate out elements.
            Defaults to 4, the tab width.
        
    Returns:
        None
    
    Notes: The on-site atomic energies (line 2) only appears in the homonuclear
        case (see slakoformat file form more information). The elements of the 
        second line are in the following order:
            Ed Ep Es SPE Ud Up Us fd fp fs
        Where Ed, Ep, and Es are the on-site atomic energies of the d, p, and
        s orbitals, respectively. 
    """
    second_row_elems = skf_file_data[1].split()
    assert(len(second_row_elems) == 10)
    numerical_elems = list(map(lambda x : float(x), second_row_elems))
    if no_d_check:
        assert(numerical_elems[0] == 0)
        assert(numerical_elems[4] == 0)
        print("Both Ed and Ud passed d check with 0")
    #Shift for Ep and Es only
    if numerical_elems[1] != 0:
        numerical_elems[1] += const
    if numerical_elems[2] != 0:
        numerical_elems[2] += const
    # if numerical_elems[5] != 0:
    #     numerical_elems[5] += const
    # if numerical_elems[6] != 0:
    #     numerical_elems[6] += const
    string_elems = list(map(lambda x : str(x), numerical_elems))
    joined_string = (' ' * num_spaces).join(string_elems) + "\n"
    skf_file_data[1] = joined_string

def shift_set_eners(skf_set_dir: str, const: float, no_d_check: bool = False, num_spaces: int = 4) -> None:
    r"""Shifts the on-site energies for a set of skf files by a constant amount
    
    Arguments:
        skf_set_dir (str): The path to the skf set location
        const (float): The constant factor to shift up by
        no_d_check (bool): If True, checks Ed = 0. Defaults to False.
        num_spaces (int): The number of spaces to separate elemments by. Defaults to 4. 
    
    Returns:
        None
    
    Notes: This performs and overwrites the homonuclear SKf files (e.g. C-C, H-H, etc.)
        and only the homonuclear SKFs. No changes should be observed between
        heteronuclear SKFs and no changes beyond the second line should be observed
        for homonuclear SKFs.
    """
    elements = infer_skf_elements(skf_set_dir)
    file_names = [f"{x}-{x}.skf" for x in elements]
    assert(len(file_names) == 4)
    skf_data = [convert_skf_file_to_data(os.path.join(skf_set_dir, skf_name)) for skf_name in file_names]
    for _, data in enumerate(skf_data):
        shift_second_line(data, const, no_d_check, num_spaces)
    for i, name in enumerate(file_names):
        with open(os.path.join(skf_set_dir, name), 'w+') as handle:
            for line in skf_data[i]:
                handle.write(line)
        print(f"Finished overwriting {name}")
    print("Overwriting completed")