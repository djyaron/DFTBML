# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 12:46:20 2021

@author: fhu14

Module to run organic molecules from ANI-1 through DFTB+ with new skf files. 
"""
#%% Imports, definitions
from .run_dftbplus import compute_results_torch, add_dftb, load_ani1, compute_results_torch_newrep
from .util import find_all_used_configs, filter_dataset
from typing import List, Dict

#%% Code behind

def run_organics(data_path: str, max_config: int, maxheavy: int, allowed_Zs: List[int], 
                 target: str, skf_dir: str, exec_path: str, pardict: Dict,
                 rep_setting: str, dftbrep_ref_params: Dict = None, num_to_use: int = None,
                 do_dftbpy: bool = True, do_dftbplus: bool = True, fermi_temp: float = None,
                 error_metric: str = "MAE", filter_test: bool = False,
                 filter_dir: str = None) -> float:
    r"""Computes the error from running DFTB+ using a set of skf files, with 
        error_metric as the method of computing the error. Error is
        computed on total energy.
    
    Arguments:
        data_path (str): The path to the dataset file
        max_config (int): The maximum number of configurations 
            for a given empirical formula
        maxheavy (int): The maximum number of heavy (non-hydrogen) atoms 
            allowed
        allowed_Zs (List[int]): The list of elements allowed in the molecules
            calculations are done for
        target (str): The energy target to train to
        skf_dir (str): The path to the skf files used by DFTB+
        exec_path (str): The path to the DFTB+ executable
        pardict (Dict): The dictionary containing skf parameters for different
            two-body integrals
        rep_setting (str): The repulsive setting being used. Determines which methods
            to use for computing error. One of 'new' or 'old'
        dftbrep_ref_params (Dict): The dictionary containing the reference energy 
            parameters derived from the DFTBrepulsive backend.
        num_to_use (int): The number of molecules to use from the dataset.
            Defaults to None, in which case all molecules extracted are used.
            Note that num_to_use extracts the first num_to_use molecules from the
            front of the dataset. 
        do_dftbpy (bool): Whether to perform DFTBpy calculations on the 
            data. Defaults to True.
        do_dftbplus (bool): Whether to perform DFTB+ calculations on the 
            data. Defaults to True.
        fermi_temp (float): The fermi temperature to use for finite temperature
            smearing. Defaults to None.
        error_metric (str): The method used for computing the error. One of
            "MAE" or "RMS" for mean absolute error and root mean square error, 
            respectively.
        filter_test (bool): Whether to test skf files on molecules 
            not used during the training. Defaults to False.
        filter_dir (str): The path to the directory containing the molecule
            pickle files to indicate which molecules to exclude from the 
            dataset. 
    
    Returns:
        error_Ha (float): The error computed between the true target value and
            the value predicted by DFTB+ in Ha
        error_Kcal (float): error_Ha * 627
    
    Notes: The important parameter here is which set of skf files you load into
        the code. If you are loading skf files from the trained model, then
        you are effectively testing out skfs from the trained models.
        
        In filtering, the skfs are tested on molecules that were not used in the
        training to generate the skfs. To let the code know which molecules to
        remove from the dataset, the filter_dir parameter is used. The directory
        that the filter_dir path points to should contain the molecules used
        in the training stored in pickle files as lists of dictionaries. 
        
        The dftbrep_ref_params dict should contain the following keys:
            'coef', 'intercept', and 'atype_ordering' (to indicate ordering for 
            atomic nunmbers)
    """
    
    dataset = load_ani1(data_path, max_config, maxheavy, allowed_Zs)
    print(f"Length of dataset: {len(dataset)}")
    
    if (filter_test and filter_dir != None):
        print("Filtering dataset")
        dataset = filter_dataset(dataset, find_all_used_configs(filter_dir))
        print(f"The number of molecules in the filtered dataset is {len(dataset)}")
    elif (filter_test and filter_dir == None):
        print("Cannot filter dataset, no directory path provided for filtering")
    
    if not (num_to_use is None): 
        dataset = dataset[:num_to_use]
    
    add_dftb(dataset, skf_dir, exec_path, pardict, do_dftbpy, do_dftbplus, fermi_temp)
    #Case based on what kind of repulsive model is being used
    if rep_setting == 'old':
        error = compute_results_torch(dataset, target, allowed_Zs, error_metric)
    elif rep_setting == 'new':
        coefs, intercept, atypes = dftbrep_ref_params['coef'], dftbrep_ref_params['intercept'], \
            dftbrep_ref_params['atype_ordering']
        error = compute_results_torch_newrep(dataset, target, allowed_Zs, atypes, coefs, intercept, error_metric)
    error_Ha, error_Kcal = error, error * 627
    print(f"Using error metric {error_metric}, the error is:")
    print(f"{error_Ha} in Hartrees")
    print(f"{error_Kcal} in Kcal/mol")
    return error_Ha, error_Kcal


