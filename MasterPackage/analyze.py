# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 21:49:06 2021

@author: fhu14
"""

"""
Updated Cat1 DFTB+ analysis script with master functions written in
run_dftbplus.py

Analysis pipeline:
    1) get test dataset
    2) run add_dftb to compute and add targets based on trained SKFs
    3) feed resulting completed dataset to calc_format_targets_report
    4) read about results in the analysis.txt output file
"""

#%% Imports, code behind
from DFTBPlus import add_dftb, calc_format_target_reports
import os, argparse, pickle
import shutil
from typing import List, Dict, Union
from Auorg_1_1 import ParDict

#%% Parameter space

#Rather than designing a new system for configuration files for analysis,
#   just set values for variables here. This isn't the best style but it's 
#   easier this way

exec_path = os.path.join(os.getcwd(), "../../../dftbp/dftbplus-21.1.x86_64-linux/bin/dftb+")
parse = "detailed"
#pardict = ParDict()
charge_form = "gross" #This should always be set to gross
dipole_unit = "Debye" #Used for both parsing out from DFTB+ and for input into calc_format_target_reports
target_dipole_unit = "eA"
error_metric = "MAE"
tot_ener_targ = "Etot"
pairwise_targs = [['dipole', 'dipole'],
        ['charges', 'charges'],
        ['dipole_ESP', 'dipole'],
       ]
allowed_Zs = [1,6,7,8]
exclusion_threshold = 20 #Number of standard deviations for excluding outliers
dest = "analysis_dir" #The destination for the analysis.txt output file

#%% Code behind

def analysis_pipeline(dset_path: str, skf_dir: str, fit_ref: bool = False) -> None:
    r"""Executes the analysis pipeline
    
    Arguments:
        dset_path (str): The path to the test set of molecules to analyze, 
            saved as a pickle file
        skf_dir (str): Path to the skf directory to use for analysis. The 
            skf directory should also contain a pickle file with the 
            saved reference energy parameters
        parse (str): Mode for parsing. Default is 'detailed'
        fit_ref (bool): Whether or not to do a fresh reference energy fit. 
            Defaults to False because generally, you are testing with
            trained reference energy parameters
    
    Returns:
        None
    
    Notes: The analysis_pipeline() function calls on two other functions, those being
        add_dftb() and calc_format_target_reports(), both defined in run_dftbplus.py. 
        The goal of this is to simplify the workflow for analyzing results from 
        running trained SKFs on a set of test molecules. 
        
        The important thing to remember is that a number of the parameters for 
        the functions are defined globally in the "Parameter space" block. 
        Be sure that those are set properly before running the analysis. 
        
        There are a few bookkeeping things that are done in the body of the 
        analysis_pipeline() function before calling on the other two functions,
        such as deducing the location of the reference parameter file. 
    """
    test_molecules = pickle.load(open(dset_path, 'rb'))
    #Join with master path so DFTB+ does not freak out
    skf_dir = os.path.join(os.getcwd(), skf_dir)
    #Pass empty dictionary since not doing DFTBPy calculation
    add_dftb(test_molecules, skf_dir, exec_path, {}, do_our_dftb = False, do_dftbplus = True, parse = parse,
             charge_form = charge_form, dipole_unit = dipole_unit)
    ref_params = None
    #import pdb; pdb.set_trace()
    if not fit_ref:
        ref_param_path = os.path.join(os.getcwd(), skf_dir, "ref_params.p")
        ref_params = pickle.load(open(ref_param_path, 'rb'))
    calc_format_target_reports(skf_dir, dest, test_molecules, error_metric, tot_ener_targ, pairwise_targs, fit_ref, 
                               allowed_Zs, ref_params = ref_params, dipole_conversion = True, 
                               prediction_dipole_unit = dipole_unit, target_dipole_unit = target_dipole_unit,
                               exclusion_threshold = exclusion_threshold)
    
    src = os.path.join(os.getcwd(), dest, "analysis.txt")
    skf_dir_splt = os.path.split(skf_dir)
    if skf_dir_splt[-1] == "":
        skf_dir_splt = os.path.split(skf_dir_splt[0])
    simple_skf_dir = skf_dir_splt[-1]
    dst = os.path.join(os.getcwd(), dest, "analysis_files", f"analysis_{simple_skf_dir}.txt")
    shutil.copy(src, dst)

    
    print("Analysis is done, files copied")
    
    
    # add_dftb(dataset: List[Dict], skf_dir: str, exec_path: str, pardict: Dict, do_our_dftb: bool = True, 
    #          do_dftbplus: bool = True, fermi_temp: float = None, parse: str = 'detailed', charge_form: str = "gross",
    #          dipole_unit: str = "Debye",
    #          dispersion: bool = False) -> None:
    # calc_format_target_reports(exp_label: str, dest: str, dataset: List[Dict], error_metric: str,
    #                       tot_ener_targ: str, pairwise_targs: List[List[str]],
    #                       fit_fresh_ref_ener: bool, allowed_Zs: List[int], ref_params: Dict = None, 
    #                       dipole_conversion: bool = True, prediction_dipole_unit: str = "Debye",
    #                       target_dipole_unit: str = "eA", exclusion_threshold: Union[int, float] = 20) -> None:


def mass_analyze(test_set_path: str, master_directory: str, fit_ref: bool = False)->None:
    r"""Analyzes a series of results using the pipeline defined above.

    Arguments:
        master_directory (str): The relative path to the directory containing all the results subdirectories
            that require analysis
        test_set_path (str): The relative path to the test dataset (usually saved as a pickle file)
        fit_ref (bool): Whether or not to do a fresh reference energy fit. 
            Defaults to False because generally, you are testing with
            trained reference energy parameters

    Returns:
        None

    Raises: 
        ValueError: If the test_set_path does not exist
        ValueError: If the master_directory does not exist

    Notes: 
        Because the analysis needs to run DFTB+ calculations and those cannot be done concurrently,
        it's going to be done sequentially. Calling analysis should be a blocking call and
        the script should not be run on multiple servers. By relative path, it means the path to the 
        dataset from the level of this script, e.g. analysis_dir/{dset}
    """
    try:
        assert(os.path.exists(os.path.join(os.getcwd(), master_directory)))
        assert(os.path.exists(os.path.join(os.getcwd(), test_set_path)))
    except Exception as e:
        raise ValueError("Necessary datapaths were found to not exist!")
    all_results_directories = sorted(os.listdir(master_directory))
    for skf_dir in all_results_directories:
        relative_skf_dir = os.path.join(master_directory, skf_dir)
        analysis_pipeline(test_set_path, relative_skf_dir, fit_ref)
    print("Mass analysis done")

def mass_analyze_infer_test_set(master_directory: str, fit_ref: bool = False) -> None:
    r"""Performs a mass analysis with the test set contained within the 
        results directory.
    
    Arguments:
        master_directory (str): The relative path to the directory containing all the results subdirectories
            that require analysis
        fit_ref (bool): Whether or not to do a fresh reference energy fit. 
            Defaults to False because generally, you are testing with
            trained reference energy parameters
    
    Returns:
        None
    
    Raises: 
        ValueError: If the test set does not exist
        ValueError: If the master_directory does not exist
    
    Notes:
        Refer to notes for mass_analyze. This function is triggered by setting the 
        dser argument to 'internal'
    """
    try:
        assert(os.path.exists(os.path.join(os.getcwd(), master_directory)))
    except Exception as e:
        raise ValueError("Master directory does not exist!")
    all_result_directories = sorted(os.listdir(master_directory))
    for skf_dir in all_result_directories:
        relative_skf_dir = os.path.join(master_directory, skf_dir)
        test_set_path = os.path.join(master_directory, skf_dir, 'test_set.p')
        try:
            assert(os.path.exists(test_set_path))
        except Exception as e:
            raise ValueError("Test set does not exist!")
        analysis_pipeline(test_set_path, relative_skf_dir, fit_ref)
    print("Mass analysis with internal test sets completed")

#%% Main block

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #If this is set to 'internal', then the internal test sets for each result
    #   directory are used
    parser.add_argument("dset", help = "Name of the test set to use")
    parser.add_argument("batch", help = "Whether you are analyzing a batch or not")
    #Note that if batch is true, then this is not the single skf_dir but the relative path to the directory containing
    #   all skf_dirs to analyze.
    parser.add_argument("skf_dir", help = "Name of the skf file directory to use. Also where results are saved")
    parser.add_argument("fit_fresh_ref", help = "Whether a fresh reference energy fit should be done, e.g. in the case of auorg-1-1")
    
    args = parser.parse_args()
    dset_path, batch, skf_dir, fit_ref = args.dset, args.batch, args.skf_dir, args.fit_fresh_ref
    
    assert(fit_ref.upper() in ["Y", "N"])
    assert(batch.upper() in ["Y", "N"])
    assert(fit_ref.upper() == "N")
    assert(batch.upper() == "Y")
    if dset_path.lower() != 'internal':
        assert((("cc" in dset_path) and ("cc" in skf_dir)) or (("wt" in dset_path) and ("wt" in skf_dir)))
    
    fit_ref = True if fit_ref.upper() == "Y" else False
    batch = True if batch.upper() == "Y" else False
    
    if not batch:
        print(f"Performing single result directory analysis with the given test set {dset_path}")
        analysis_pipeline(dset_path, skf_dir, fit_ref = fit_ref)
    else:
        if dset_path != 'internal':
            print(f"Performing mass analysis with given test set {dset_path}")
            mass_analyze(dset_path, skf_dir, fit_ref = fit_ref)
        else:
            print("Performing mass analysis with internal test sets")
            mass_analyze_infer_test_set(skf_dir, fit_ref = fit_ref)
