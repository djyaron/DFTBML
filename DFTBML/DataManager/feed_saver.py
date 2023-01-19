# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 13:40:21 2021

@author: fhu14
"""
#%% Imports, definitions 
from typing import List, Dict
from DFTBLayer import DFTBList
import os
from .h5handler import per_molec_h5handler, per_batch_h5handler
import pickle

#%% Code behind

def save_feed_h5(feeds: List[Dict], dftb_lsts: List[DFTBList], 
                 molecs: List[Dict], dest: str, batches: List[List[Dict]], duplicate_data: bool = False) -> None:
    r"""Saves some feeds and dftb_lsts to the appropriate location (dest) in h5 file format
    
    Arguments:
        feeds (List[Dict]): The list of feed dictionaries to save
        dftb_lsts (List[DFTBList]): The list of DFTBList objects to save
        molecs (List[Dict]): The list of molecule dictionaries to save
        dest (str): Relative path to the folder to save all these things in
        batches (List[List[Dict]]): The original list of batches, where the 
            ith inner list of molecules were used to generate the ith batch.
        duplicate_data (bool): Whether to duplicate the raw molecule pickle files 
            in the same directories as the h5 files. Defaults to False.
    
    Returns:
        None
        
    Notes: Uses the h5handler interface functions similar to what's done in 
        saving_fold
    """
    if not os.path.isdir(dest):
        os.mkdir(dest)
    
    molec_filename = os.path.join(dest, "raw_molecules.p")
    molec_h5_filename = os.path.join(dest, "molecs.h5")
    batch_h5_filename = os.path.join(dest, "batches.h5")
    dftb_lst_filename = os.path.join(dest, "dftblsts.p")
    reference_data_name = os.path.join(dest, "reference_data.p")
    batch_original_filename = os.path.join(dest, "batch_original.p")
    
    #Save the per_molec and per_batch info
    per_molec_h5handler.save_all_molec_feeds_h5(feeds, molec_h5_filename)
    per_batch_h5handler.save_multiple_batches_h5(feeds, batch_h5_filename)
    
    #Save the DFTBlists
    with open(dftb_lst_filename, 'wb') as handle:
        pickle.dump(dftb_lsts, handle)
    
    #Save the reference data
    with open(reference_data_name, 'wb') as handle:
        pickle.dump(feeds, handle)
        
    #Save the batches
    with open(batch_original_filename, 'wb') as handle:
        pickle.dump(batches, handle)
    
    if duplicate_data:
        with open(molec_filename, 'wb') as handle:
            pickle.dump(molecs, handle)
    
    print("All data saved successfully")