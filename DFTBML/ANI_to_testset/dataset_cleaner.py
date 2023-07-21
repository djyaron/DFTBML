"""
This file generates the clean version of the ANI-1ccx dataset used for our calculations as of July 10th, 2023
"""
# %%
from h5_converter import h5_handler
from pathlib import Path
import numpy as np

# %%
def load_dset_to_dict():
    r"""Uses the h5 handler to load the entire h5 dataset directly into a dictionary
    
    Arguments:
    None

    Returns:
    dict: The full ani1x-release dataset as a dictionary.
    """
    dict = h5_handler.h5_to_dict('ani1x-release.h5')
    return dict

# %%
def count_total_configs(dic: dict):
    r"""Loops through the inputted dictionary to count how many different iconfigs there are.
    
    Arguments:
    dic: the dictionary to scan for iconfigs.

    Returns:
    None
    """
    count = 0
    for mol in dic.keys():
        iconfig = 0
        while iconfig < len(dic[mol]['coordinates']):
            iconfig+=1
            count+=1
    print(count)

# %%

def recursive_read(object):
    r"""Recursively scans an inputted array for NaN entries.
    
    Arguments:
    array: An array to scan for NaN entries.

    Returns:
    isnan: A Boolean value. True if there is a NaN value in the array and False if not.
    """
    isnan = False
    i = 0
    while i < len(object):
        if isinstance(object[i], np.ndarray):
            isnan = recursive_read(object[i])
        else:
            if np.isnan(object[i]):
                isnan = True
                return isnan
        i += 1
        if isnan:
            break
    return isnan

# %%
def scan(source_dset: dict):
    r"""The main function for sorting the dataset into complete and incomplete entries. Looks through all molecules and iconfigs. If the iconfig 
    is complete, this saves the unique identifier as a tuple to the complete_entries list.
    
    Arguments:
    source_dset: The dictionary that has the source dataset contents.

    Returns:
    complete_entries: A list of Tuples (molecule name, iconfig number in the source dataset).
    needed_keys: The list of all keys to examine for NaN values.
    ccsd_full_entries: A list (like complete_entries) of all entries that were not complete but had complete entries for ccsd energy.
    """

    needed_keys = list(source_dset['C10H10'].keys()) # ! this currently requires all keys. Can change this if we want to specify later.
    if 'atomic_numbers' in needed_keys:
        needed_keys.remove('atomic_numbers')  #atomic numbers doesn't have NaN values

    complete_entries = []
    ccsd_full_entries = []

    totconfigs = 0
    totmolecs = 0
    nan_entries = 0
    ccsd_nan_entries = 0
    
    with open('cleaned_dsets/logfile.txt', 'w') as log:
        print('LOG:', file=log)
    
    molecs = source_dset.keys()
    for molec in molecs:
        totmolecs+=1
        if molec == 'O2':
            with open('cleaned_dsets/logfile.txt', 'a') as log: 
                print('Skipped O2 molecule', file=log)
        else:
            iconfig = 0
            numconfigs = len(source_dset[molec]['coordinates'])
            while iconfig < numconfigs:
                for key in needed_keys:
                    containsnan = False
                    if isinstance(source_dset[molec][key][iconfig], np.ndarray):
                        containsnan = recursive_read(source_dset[molec][key][iconfig])
                    else:
                        if np.isnan(source_dset[molec][key][iconfig]):
                            containsnan = True
                    if containsnan:
                        if key == 'ccsd(t)_cbs.energy':
                            ccsd_nan_entries += 1
                        else:
                            ccsd_full_entries.append((molec, iconfig))
                        nan_entries+=1
                        break

                if not containsnan:
                    complete_entries.append((molec, iconfig))
                    
                if iconfig == numconfigs-1:
                    with open('cleaned_dsets/logfile.txt', 'a') as log:
                        print(f"all {iconfig+1} configs of {numconfigs} {molec} configs completed", file=log)

                iconfig+=1
                totconfigs+=1

    with open('cleaned_dsets/logfile.txt', 'a') as log:
        print(complete_entries, file=log)
        print(len(complete_entries), file=log)
        print(f"All molecules: {totmolecs}", file=log)
        print(f"All configs: {totconfigs}", file=log)
        print(f"Total NaN configs: {nan_entries} with {ccsd_nan_entries} due to ccx (verify number: {totconfigs - len(complete_entries)})", file=log)

    return complete_entries, needed_keys, ccsd_full_entries
    #4,956,005

# %%
def complete_entries_to_clean_dset(source_dset: dict, complete_entries: list, needed_keys):
    r"""Uses the results from the scan function to only copy over the entries in the source dataset that are complete to the new dataset.
    
    Arguments:
    source_dset: The dictionary that has the source dataset contents.
    complete_entries: A list of Tuples (molecule name, iconfig number in the source dataset).
    needed_keys: The list of all keys to copy into the new dataset.

    Returns:
    None.
    """
    print(f"Turning {len(complete_entries)} to dset")
    dic = {}
    for (mol, iconfig) in complete_entries:
        if mol in dic.keys():
            # ! iconfig (See note below)
            dic[mol]['iconfig'].append(iconfig)
            for key in needed_keys:
                if key != 'atomic_numbers':
                    dic[mol][key].append(source_dset[mol][key][iconfig])
        else:
            dic[mol]={}
            # ! iconfig IS NOT usually stored in the h5 datasets. This is to preserve a reference to the original ANI1ccx dataset. Not sure how it may conflict with existing code.
            dic[mol]['atomic_numbers'] = source_dset[mol]['atomic_numbers']
            dic[mol]['iconfig'] = [iconfig]
            for key in needed_keys:
                if key == 'atomic_numbers':
                    dic[mol][key] = source_dset[mol][key]
                else:
                    dic[mol][key] = [source_dset[mol][key][iconfig]]
    with open('cleaned_dsets/logfile.txt', 'a') as log:
        print(f"Saving dict...", file=log)
        h5_handler.save_dict_new_h5(dic, 'cleaned_dsets/new_ANI1ccx_clean.h5')
        print(f"Dict Saved.", file=log)

# %%

def nan_search_all_keys(sourcedict: dict, searchlist: list, needed_keys):
    r"""Searches an given list of molecules within the source dataset to determine the number of occurances of NaN values in each key. (Used for a better understanding of the ANI1x dataset).
    
    Arguments:
    sourcedict: The dictionary that has the source dataset contents.
    ccsd_full_entries: A list (like complete_entries) of all entries that were not complete but had complete entries for ccsd energy.
    needed_keys: The list of all keys to examine for NaN values.
    
    Returns: 
    None.
    """
    with open('cleaned_dsets/logfile.txt', 'a') as log:
        print(f"Scanning {len(searchlist)} full ccsd entries for nan values", file=log)
    occurance_counter = {}
    for key in needed_keys:
        occurance_counter[key] = 0
        for (mol, iconfig) in searchlist:
            containsnan = False
            if key != 'atomic_numbers':
                if isinstance(sourcedict[mol][key][iconfig], np.ndarray):
                    containsnan = recursive_read(sourcedict[mol][key][iconfig])
                else:
                    if np.isnan(sourcedict[mol][key][iconfig]):
                        containsnan = True
            if containsnan:
                occurance_counter[key]+=1
        with open('cleaned_dsets/logfile.txt', 'a') as log:
            print(f"{key} occurances: {occurance_counter[key]}", file=log) 


# %%
if __name__ == '__main__':
    this_file = Path(__file__).resolve()
    clean_dir = this_file.parent.parent / "cleaned_dsets"
    clean_dir.mkdir(exist_ok=True)
    
    source_dict = load_dset_to_dict()
    #count_total_configs(source_dict)
    complete_entries, needed_keys, ccsd_full_list = scan(source_dict)
    complete_entries_to_clean_dset(source_dict, complete_entries, needed_keys)

    # additional check for what keys are missing in molecules with complete ccsd entries.
    nan_search_all_keys(source_dict, ccsd_full_list, needed_keys)

