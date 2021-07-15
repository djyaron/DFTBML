# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 12:50:30 2021

@author: fhu14
"""

#%% Imports, definitions
import os
from DFTBPlus import find_all_used_configs, filter_dataset, run_organics, read_detailed_out
import pickle

#%% Code behind

def apx_equal(x: float, y: float, tol: float = 1E-12) -> bool:
    return abs(x - y) < tol

def test_molecule_exclusion() -> None:
    r"""Tests the correct functionality of the molecule exclusion when running
        DFTB+ on a dataset of unseen molecules.
    """
    print("Testing molecule exclusion...")
    dataset_path = os.path.join(os.getcwd(), "test_files", "molec_exclusion")
    name_conf_pairs = find_all_used_configs(dataset_path)
    dataset = pickle.load(open(os.path.join(dataset_path, "Fold0_molecs.p"), 'rb')) + \
        pickle.load(open(os.path.join(dataset_path, "Fold1_molecs.p"), 'rb'))
    print(f"Length of the dataset: {len(dataset)}")
    cleaned_dataset = filter_dataset(dataset, name_conf_pairs)
    #If the filter directory contains all the same molecules as the dataset,
    #   then the dataset should be filtered to 0 molecules.
    assert(len(cleaned_dataset) == 0)
    
    #General test for correct exclusion.
    dataset_path = os.path.join(os.getcwd(), "test_files", "molec_exclusion")
    dataset_ref_path = os.path.join(os.getcwd(), "test_files", "molec_exclusion", "molec_exclusion_single0")
    name_conf_pairs = find_all_used_configs(dataset_ref_path)
    dataset = pickle.load(open(os.path.join(dataset_path, "Fold0_molecs.p"), 'rb')) + \
        pickle.load(open(os.path.join(dataset_path, "Fold1_molecs.p"), 'rb'))
    print(f"Length of the dataset: {len(dataset)}")
    assert(len(dataset) == 2329)
    assert(len(name_conf_pairs) == 1863)
    cleaned_dataset = filter_dataset(dataset, name_conf_pairs)
    clean_dataset_names = [(molec['name'], molec['iconfig']) for molec in cleaned_dataset]
    clean_name_set = set(clean_dataset_names)
    nc_pair_set = set(name_conf_pairs)
    #No overlapping elements
    assert(clean_name_set.difference(nc_pair_set) == clean_name_set)
    assert(nc_pair_set.difference(clean_name_set) == nc_pair_set)
    
    dataset_ref_path = os.path.join(os.getcwd(), "test_files", "molec_exclusion", "molec_exclusion_single1")
    name_conf_pairs = find_all_used_configs(dataset_ref_path)
    assert(len(name_conf_pairs) == 466)
    cleaned_dataset = filter_dataset(dataset, name_conf_pairs)
    clean_dataset_names = [(molec['name'], molec['iconfig']) for molec in cleaned_dataset]
    clean_name_set = set(clean_dataset_names)
    nc_pair_set = set(name_conf_pairs)
    #No overlapping elements
    assert(clean_name_set.difference(nc_pair_set) == clean_name_set)
    assert(nc_pair_set.difference(clean_name_set) == nc_pair_set)
    
    print("Molecule exclusion test passed")

def test_dftbplus_organics():
    r"""Runs organic molecules through DFTB+ and computes the resulting error
        to previously obtained values. Since doing float comparisons, using 
        apx_equal for testing. 
    """
    print("Testing DFTB+ run on organics...")
    data_path = os.path.join(os.getcwd(), "test_files", "ANI-1ccx_clean_fullentry.h5")
    max_config = 1
    maxheavy = 8
    allowed_Zs = [1,6,7,8]
    target = 'cc'
    skf_dir = os.path.join(os.getcwd(), "Auorg_1_1", "auorg-1-1")
    exec_path = "C:\\Users\\fhu14\\Desktop\\DFTB17.1Windows\\DFTB17.1Windows-CygWin\\dftb+"
    from Auorg_1_1 import ParDict
    pardict = ParDict()
    #Not going to do filter testing, so values are not required for those 
    #   parameters. Also going to stick with mean abs error for testing purposes.
    
    skf_dir_au = os.path.join(os.getcwd(), "Auorg_1_1", "auorg-1-1")
    skf_dir_mio = os.path.join(os.getcwd(), "MIO_0_1", "mio-0-1")
    skf_dir_refac_joined = os.path.join(os.getcwd(), "test_files", "refacted_joined_spline_run")
    skf_dir_100knot = os.path.join(os.getcwd(), "test_files", "skf_8020_100knot")
    
    skf_names = [skf_dir_au, skf_dir_mio, skf_dir_refac_joined, skf_dir_100knot]
    accepted_vals_kcal = [10.902432164852327, 11.498048267608118, 4.30573249547126,
                          4.36884195335689]
    
    for i, skf_dir_name in enumerate(skf_names):
        #Non-specified argument default values are good for testing.
        error_Ha, error_Kcal = run_organics(data_path, max_config, maxheavy, 
                                            allowed_Zs, target, skf_dir_name, 
                                            exec_path, pardict, 'old')
        assert(apx_equal(error_Kcal, accepted_vals_kcal[i]))
    
    print("DFTB+ run on organics passed")

def test_dftbplus_detailed_out():
    r"""Tests out the function to parse and read a detailed.out file.
    """
    print("Testing detailed.out parsing...")
    
    file_path = "test_files/detailed.out"
    result = read_detailed_out(file_path)
    assert(result['t'] == -4.8133377035)
    assert(result['e'] == -5.1997524973)
    assert(result['r'] == 0.3864147938)
    
    print("detailed.out parsing successful")
    
def run_dftbplus_tests():
    test_molecule_exclusion()
    test_dftbplus_organics()
    
if __name__ == "__main__":
    run_dftbplus_tests()

