# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 19:23:53 2022

@author: fhu14
"""

"""
Final script for running multiple experiments consecutively with numerous
settings files.

This script should be applied to a scratch directory that has the following
structure:

benchtop_wdir:
    - settings_files (starts off containing only the default settings file)
    - results (empty)
    - dsets (empty)
    - tmp (empty)

The user uploads a series of datasets (one or more) to this scratch directory
and the code executes all the experiments where each experiment is specified 
by a settings file contained in the settings_files subdirectory. 
Also has safety checks built in to ensure that the dataset the user is 
trying to access during an experiment exists. 

This simply loops over the settings files, performs the existence check on the 
datasets, and calls on the driver functionality to do perform the 
run. A log file is written out detailing each successful execution. The log file
is deleted and recreated at the beginning of each run. 

This script should always be at the same directory level relative to 
the benchtop_wdir. 

The workflow for running experiments is as follows:
    1) Write script to generate datasets (e.g. PaperPackage/dataset_gen_script.py)
        (writing novel dataset extractions gives greater control over correctness
         of data format and included data)
    2) Upload datasets all to benchtop_wdir/dsets
    3) Generate and upload all settings files to benchtop_wdir/settings_files along 
        with a default settings file
    4) Execute the script
    5) Analyze results afterwards
"""

"""
Note to self: remember that path-searching is all relative in Python. Since
this is the top level module, everything must be done relative to the top level
path of .../MasterPackage

This module will be written in such a way that it can take advantage of the 
multiple cat machines available. This is accomplished through a temp file being
written and deleted as a placeholder for each experiment. Because multiple 
cat servers will be executing at the same time, the safety and existence
checks should only be executed once at the very beginning.
"""

#%% Imports, definitions
import numpy as np
from driver import run_training
import os, shutil, pickle, json
from precompute_check import precompute_settings_check
from typing import List, Union, Dict
import time
from datetime import datetime
from functools import reduce
from Tests import run_safety_check #comes from h5handler_test.py
import re

#Save the prefix and important paths to prevent repeated computation
LOG_FILE_PATH = os.path.join(os.getcwd(), "benchtop_wdir", "EXP_LOG.txt")
WDIR_PREFIX = os.path.join(os.getcwd(), "benchtop_wdir")
SETTINGS_DIR_PATH = os.path.join(os.getcwd(), "benchtop_wdir", "settings_files")
RESULTS_DIR_PATH = os.path.join(os.getcwd(), "benchtop_wdir", "results")
DSETS_DIR_PATH = os.path.join(os.getcwd(), "benchtop_wdir", "dsets")
TMP_DIR_PATH = os.path.join(os.getcwd(), "benchtop_wdir", "tmp")


#%% Code behind

def reset_log_file() -> None:
    if os.path.isfile(LOG_FILE_PATH):
        os.remove(LOG_FILE_PATH)

def write_to_log(message: str) -> None:
    r"""Writes the given message to the log file
    
    Arguments:
        message (str): The message to write
    
    Returns:
        None
    """
    timestamp = str(datetime.now())
    with open(LOG_FILE_PATH, 'a') as handle:
        handle.write(f"[{timestamp}]: {message}\n")

def existence_checks() -> None:
    r"""Checks whether all the specified datasets are present for each of the
        settings files contained in the settings_files directory
    
    Arguments:
        None
    
    Returns:
        None
    
    Raises:
        ValueError: If the required dataset for an experiment does not exist,
            information is logged, a ValueError is raised, and execution is cancelled.
        ValueError: If the directory does not only contain json files, information 
            is logged, a ValueError is raised, and execution is cancelled.
        ValueError: If the top_level_fold_path path is not properly formatted
            with the necessary prefxes, information is logged and an error is raised.
    
    Notes: In the json settings files, the dataset to use is specified by the
        top_level_fold_path parameter. The top level fold path should have the 
        appropriate prefixes so that the directory can be accessed from the 
        top level, where this script exists. Thus, every path should be
        of the form "benchtop_wdir/dsets/{dset_name}"
    """
    settings_files_names = os.listdir(SETTINGS_DIR_PATH)
    #Do an assert to ensure only json files are present in the directory
    try:
        assert(
            reduce(lambda x, y : x and y, 
                   map(lambda x : x.split('.')[1] == 'json', settings_files_names))
            )
    except:
        write_to_log("Non-json format file detected in settings_files directory!")
        raise ValueError("Non-json format file detected in settings_files directory!")
    #Next, loop through the settings files and ensure that for each of the experiments,
    #   the necessary dataset is there
    for filename in settings_files_names:
        if "default" not in filename:
            filename = os.path.join(SETTINGS_DIR_PATH, filename)
            with open(filename, 'r') as handle:
                settings_dict = json.load(handle)
                dset_path = settings_dict['loaded_data_fields']['top_level_fold_path']
                try:
                    dset_path_splt = dset_path.split("/")
                    assert(dset_path_splt[0] == "benchtop_wdir" and dset_path_splt[1] == "dsets")
                except:
                    write_to_log(f"Dataset path for {filename} does not contain necessary prefixes")
                    raise ValueError(f"Dataset path for {filename} does not contain necessary prefixes")
                if not os.path.exists(dset_path):
                    write_to_log(f"Dataset {dset_path} is missing for experiment based on file {filename}")
                    raise ValueError(f"Dataset {dset_path} is missing for experiment based on file {filename}")
    
    print("Existence checks completed")
    write_to_log("Existence checks completed")

def run_dataset_safety_checks() -> None:
    r"""Runs a series of safety checks on each of the datasets contained in the 
        datasets directory. Uses the run_safety_check function implemented in 
        the h5handler_test.py module.
        
    Arguments:
        None
    
    Returns:
        None
    
    Raises:
        ValueError: If one of the datasets does not pass, information is logged
            and a value error is raised.
    """
    #To get exact regular expression matches: https://stackoverflow.com/questions/45244813/python-regular-expression-exact-match-only
    pattern = r"^Fold[0-9]+$"
    names = os.listdir(DSETS_DIR_PATH)
    for dataset in names:
        full_dset_path = os.path.join(DSETS_DIR_PATH, dataset)
        all_dset_filenames = os.listdir(full_dset_path)
        num_folds = len(list(filter(lambda x : re.match(pattern, x), all_dset_filenames)))
        try:
            run_safety_check(full_dset_path, [i for i in range(num_folds)])
            write_to_log(f"Safety check passed for dataset {dataset}")
        except:
            write_to_log(f"Safety check failed on dataset {dataset}")
            raise ValueError(f"Safety check failed on dataset {dataset}")
    print("All safety checks passed for included datasets")
    write_to_log("All safety checks passed for included datasets")

def construct_tmp_filename(settings_filename: str) -> str:
    r"""Constructs the full path for a given temp file given the associated
        settings_filename
    
    Arguments:
        settings_filename (str): The settings_filename whose temp file name needs
            to be constructed. The settings filename can have the extension
            (e.g. 'file_name.json') or not (e.g. just 'filename'). 
            The filename should not have any path prefixes on it.
    
    Returns:
        tmp_path (str): The full temp file path for the associated setting_filename.
            This tmp_path is good to check directly using os.path.exists() utility.
    """
    if '.' in settings_filename:
        settings_filename = settings_filename.split('.')[0]
    tmp_filename = settings_filename + "_TMP.txt"
    tmp_path = os.path.join(TMP_DIR_PATH, tmp_filename)
    return tmp_path

def write_tmp_file(settings_filename: str) -> None:
    r"""Writes out a temporary placeholder file for a settings file to let
        all the other cat servers know that the experiment is being handled.
    
    Arguments:
        settings_filename (str): The settings filename indicating the experiment
            currently being run. The settings filename can have the extension 
            (e.g. 'file_name.json') or not (e.g. just 'filename').
            The filename should not have any path prefixes on it.
    
    Returns:
        None
    
    Notes: temp files are stored in the tmp subdirectory and are used as 
        placeholders to ensure that multiple cat servers are not trying
        to run the same experiments.
    """
    tmp_filename = construct_tmp_filename(settings_filename)
    timestamp = str(datetime.now())
    with open(tmp_filename, 'a') as handle:
        handle.write(f"[{timestamp}]: Experiment {settings_filename} in progress")

def check_tmp_file_existence(settings_filename: str) -> bool:
    r"""Checks if an experiment is currently underway by checking if the 
        temp file associated with the given settings_filename exists
    
    Arguments:
        settings_filename (str): The settings filename indicating the experiment
            currently being run. The settings filename can have the extension 
            (e.g. 'file_name.json') or not (e.g. just 'filename').
            The filename should not have any path prefixes on it.
    
    Returns: 
        file_exists (bool): True if the tmp file exists, false otherwise
    
    Notes: Checking for a tmp_file's existence is how the script executing 
        on several servers at once does not overlap on the experiment being done.
    """
    tmp_filename = construct_tmp_filename(settings_filename)
    return os.path.exists(tmp_filename)
        
def delete_tmp_file(settings_filename: str) -> None:
    r"""Removes a temporary placeholder file from the tmp directory.
    
    Arguments:
        settings_filename (str): The settings file name, either with an extension
            or without one, (e.g. 'file_name.json') or (e.g. just 'filename').
            The filename should not have any path prefixes on it.
    
    Returns:
        None
    
    Raises:
        ValueError: If the temporary file cannot be found in the tmp directory
            (although this should not happen)
    """
    tmp_filename = construct_tmp_filename(settings_filename)
    if not os.path.exists(tmp_filename):
        write_to_log(f"Error detected, placeholder file {tmp_filename} does not exist!")
        raise ValueError(f"Error detected, placeholder file {tmp_filename} does not exist!")
    else:
        os.remove(tmp_filename)
        write_to_log(f"Successfully removed temporary file {tmp_filename}")
        print(f"Successfully removed temporary file {tmp_filename}")

def clear_directory(path_to_direc: str) -> None:
    r"""Generic file management utility for clearing directories. 
        Inspiration taken from https://stackoverflow.com/questions/185936/how-to-delete-the-contents-of-a-folder
    
    Arguments:
        path_to_direc (str): The path to the directory that needs to be cleared
    
    Returns:
        None
    
    Raises:
        Exception: Raises a generic exception if something cannot be cleared 
            or deleted
    
    Notes: DO NOT REMOVE THINGS or clear directories when running a series of
        experiments as the presence and absence of temporary files are
        necessary for the multi-server approach to work properly. Also, the 
        path_to_direc should be the FULL PATH, i.e. it can be used directory
        with os module utilities.
    """
    for elem in os.listdir(path_to_direc):
        full_elem_path = os.path.join(path_to_direc, elem)
        try:
            if os.path.isfile(full_elem_path) or os.path.islink(full_elem_path):
                os.remove(full_elem_path)
            elif os.path.isdir(full_elem_path):
                shutil.rmtree(full_elem_path)
        except Exception as e:
            write_to_log(f"Could not remove file {full_elem_path} because {e}")
            raise Exception(f"Could not remove file {full_elem_path} because {e}")
    
    print(f"Directory {path_to_direc} fully cleared")
    write_to_log(f"Directory {path_to_direc} fully cleared")

def copy_results_files(directory_name: str) -> None:
    r"""Because the results of an experiment are written to a directory
        at the same level as the master module, it needs to be moved into the 
        results directory and all relevant information needs to be copied over.
    
    Arguments:
        directory_name (str): The name of the directory that needs to be copied
            over into the results directory.
    
    Returns:
        None
    
    Raises:
        Exception: If shutil canont successfully move the directory_name into the
            results directory.
    
    Notes: Because the results directory will be created at the same level as
        this script, the directory_name should just be the name and not the full
        path (since Python automatically searches at the directory level of the
        master module). This also simplifies a lot of issues since the directory
        name can be used directly for searching and filesystem manipulations.
    """
    #A lot of this logic has already bee implemented in the paper_package_top_level.py 
    #   module and can be translated over from there with minimal modifications.,
    destination = os.path.join(RESULTS_DIR_PATH, directory_name)
    if os.path.isdir(destination):
        shutil.remove(destination)
        rm_msg = f"Cleared pre-existing results directory at {destination}"
        print(rm_msg)
        write_to_log(rm_msg)
    try:
        shutil.move(directory_name, destination)
        msg = f"Successfully moved result directory {directory_name} to {destination}"
        print(msg)
        write_to_log(msg)
    except Exception as e:
        write_to_log(f"Could not move results directory {directory_name} to {destination} because of {e}")
        raise Exception(f"Could not move results directory {directory_name} to {destination} because of {e}")
    
def run_experiments() -> None:
    r"""Master function for running all the specified experiments and 
        performing the necessary safety checks.
    
    Arguments:
        None
    
    Returns:
        None
    
    Notes: There are a few intricacies to how this works since this script 
        can be independently executed on mutliple servers that share the same
        files.
        
        First, the sequence of checks (existence, correctness for formatting)
        for the datasets are only executed if the temporary directory is empty.
        This makes sense because if the directoy is not empty, then it means 
        that another server is already operating and has already performed those
        checks. If it is empty, one of two things has happened: either the current
        call to the script is the first call, OR there is somethind wrong and a
        previous server failed to start executing in which case this current 
        call will also fail. One can verify this by checking the log file 
        which resides at the top level of the directory.
        
        Second, to figure out which experiment to execute, the code will loop
        through each experiment file, check if the corresponding temporary 
        file exists in the tmp directory, and execute the first settings file
        for which a corresponding TMP file does not exist. Because experiments
        can take anywhere from 30 minutes to an hour to run, the likelihood of 
        collisions between different servers is low. 
        
        Third, because this is not multithreading but rather multiple instances
        of the same code being executed on different servers, there is no utility
        for clearing all the temporary files. A function is included for doing so,
        but UNDER NO CIRCUMSTANCES SHOULD IT BE CALLED DURING A RUN. This will
        mess everyting up as other instances will not be able to determine 
        which experiments to excecute based on the presence or absence of TMP files.
        Manually clear tmp files before the start of a series of runs. This approach
        is optimal because the cat servers cannot talk to each other. 
    
        Fourth, because of the way that settings files are read in and 
        interpreted by the backend code, a default setings files is included 
        in the settings file directory.
    """
    #Try executing the safety checks now
    if len(os.listdir(TMP_DIR_PATH)) == 0:
        reset_log_file()
        existence_checks()
        run_dataset_safety_checks()
    
    #Start looping through the settings files and executing experiments
    #The experiment is the name of the settings file, the experiment_path is the full path to the file
    defaults_path = os.path.join(SETTINGS_DIR_PATH, "refactor_default_tst.json")
    for experiment in os.listdir(SETTINGS_DIR_PATH):
        if "default" not in experiment:
            experiment_path = os.path.join(SETTINGS_DIR_PATH, experiment)
            file_executing = check_tmp_file_existence(experiment)
            if not file_executing:
                precompute_settings_check(experiment_path)
                x = input("Did you check the necessary fields? (Y) ")
                y = input("Did you check the run id? (Y) ")
                #Write placeholder file to tell other servers this experiment is being done
                write_tmp_file(experiment)
                write_to_log(f"Beginning experiment {experiment}")
                print(f"Beginning experiment {experiment}")
                try:
                    #Forgot to add the correct skf method
                    run_training(experiment_path, defaults_path, skf_method = 'new')
                except Exception as e:
                    print(f"Experiment {experiment} failed because {e}")
                    write_to_log(f"Experiment {experiment} failed because {e}")
                    return #Just going to early return instead of raising another exception
                with open(experiment_path, 'r') as handle:
                    json_dict = json.load(handle)
                    results_directory_name = json_dict['run_id']
                copy_results_files(results_directory_name)
                delete_tmp_file(experiment)
                write_to_log(f"Completed experiment{experiment}")
                print(f"Completed experiment{experiment}")
    
    print("All experiments completed/already in progress")
            
#%% Main block
if __name__ == "__main__":
    run_experiments()
    pass

