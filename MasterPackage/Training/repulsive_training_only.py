# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 13:44:36 2022

@author: fhu14

Training process for the repulsive model only. The electronic aspects of the
SKF files are maintained as constant (not trained).

The workflow looks like this:
    1) Add DFTB+ predictions to the molecules using the existing analysis code
    2) Treat the DFTB+ electronic predictions as the true electronic target 
        (NOTE: DON'T FORGET TO DIVIDE BY N_HEAVY since the batches will also have energies per heavy atom)
    3) Create the inputs and push through the predictions
    4) At some point, need to join the electronic parts of auorg with the repulsive parts 
"""

#%% Imports, definitions
import os, pickle
from InputLayer import DFTBRepulsiveModel, generate_gammas_input, combine_gammas_ctrackers
import numpy as np
from typing import List, Dict
from FoldManager import count_nheavy
from DataManager import load_gammas_per_fold, load_config_tracker_per_fold
from Training import sort_gammas_ctracks
from InputParser import parse_input_dictionaries,\
    inflate_to_dict
from functools import reduce
from DFTBrepulsive import SKFBlockCreator
from PlottingUtil import read_skf_set
from MasterConstants import atom_nums
import re, shutil

#%% Code behind

def reformat_molecule_data(mol: Dict) -> None:
    r"""Reformats the data so that it is per heavy atom and contained at the 
        correct path in the dictionary
    
    Arguments:
        mol (Dict): The molecule dictionary to correct
    
    Returns:
        None
    
    Notes: Two corrections are performed: the predictions of the electronic
        energy are divided by the number of heavy atoms and the total energy prediction
        is set at the correct location in the dictionary
    """
    nheavy = count_nheavy(mol)
    #Only looking at the electronic energy!
    nheavy_corrected_energy = mol['pzero']['e'] / nheavy
    mol['predictions'] = {}
    mol['predictions']['Etot'] = {}
    mol['predictions']['Etot']['Etot'] = nheavy_corrected_energy

def train_repulsive_model(dset_dir: str, settings_filename: str, defaults_filename: str) -> DFTBRepulsiveModel:
    r"""Trains the repulsive model using a series of training molecules where
        the DFTB+ predictions have been added in. 
    
    Arguments:
        dset_dir (str): The path to the dataset to train on
        settings_filename (str): The path to the settings json file
        defaults_filename (str): The path to the defaults json file
    
    Returns:
        rep_mod (DFTBRepulsiveModel): The trained DFTBrepulsive model using the given
            training data.
    
    Notes: The predictions from DFTB+ are generated such that the electronic and repulsive
        aspects are separated. From the DFTBRepulsive/driver.py docs, the target is 
            target = ground_truth - dftbtorch_elec
        We are just replacing dftbtorch_elec with DFTB+_elec and training on the 
        difference (so removing the electronic aspect of the model from consideration)
        
        The training portion of the model scales the target up by the number of 
        heavy atoms, and both ground_truth and dftbtorch_elec in the dftbtorch
        datasets are divided by n_heavy. This should also be the case for DFTB+_elec so that
            raw_target = (ground_truth - DFTB+_elec) / n_heavy
        and then in DFTBRepulsive:
            target = raw_target * n_heavy = ground_truth - DFTB+_elec
        as intended. The predictions generated are divided by n_heavy, but that is
        irrelevant. The resulting model will be written to an SKF file and combined
        with the Auorg electronic sections.
        
        Unlike with the dftbtorch model, the information contained in settings_file does not
        matter except for the repulsive model settings. 
    """
    s_obj = parse_input_dictionaries(settings_filename, defaults_filename)
    opts = inflate_to_dict(s_obj) #opts is a dictionary for DFTBrepulsive to use only. 
    #First, load the gammas and c_trackers of the data (driver.py)
    gammas = load_gammas_per_fold(dset_dir)
    c_trackers = load_config_tracker_per_fold(dset_dir)
    split_mapping = {0 : [[0], [1]]}
    train_gammas, train_c_trackers, valid_gammas, valid_c_trackers = sort_gammas_ctracks(split_mapping, 0, gammas, c_trackers)
    #Load in the batched data with the DFTB+ informationa added in
    T_batch_name = os.path.join(os.getcwd(), dset_dir, 'Fold0', 'processed_batches.p')
    V_batch_name = os.path.join(os.getcwd(), dset_dir, 'Fold1', 'processed_batches.p')
    training_batches = pickle.load(open(T_batch_name, 'rb'))
    validation_batches = pickle.load(open(V_batch_name, 'rb'))
    # assert(len(training_batches) == 250)
    assert(len(training_batches) != len(validation_batches))
    # assert(len(validation_batches) == 63)
    #Correct the data
    for batch in training_batches: 
        for mol in batch:
            reformat_molecule_data(mol)
    for batch in validation_batches:
        for mol in batch:
            reformat_molecule_data(mol)
    #Additional steps in training_loop.py
    gamma_T, c_tracker_T = combine_gammas_ctrackers(train_gammas, train_c_trackers)
    gamma_V, c_tracker_V = combine_gammas_ctrackers(valid_gammas, valid_c_trackers)
    #Now to initialize the repulsive model
    rep_mod = DFTBRepulsiveModel([c_tracker_T, c_tracker_V],
                                 [gamma_T, gamma_V])
    #Go directly to training the model
    rep_mod.train_update_model(training_batches, validation_batches, opts)
    return rep_mod

def flatten_batches(batches: List[List[Dict]]) -> List[Dict]:
    r"""Flattens a 2D list of molecules into a 1D list
    """
    return list(reduce(lambda x, y : x + y, batches))

def reconstruct_batches(flattened_batch: List[Dict], batch_size: int) -> None:
    r"""Re-batches flattened data. Logic is taken from data_loader module of DataManager package
    """
    batches = list()
    for i in range(0, len(flattened_batch), batch_size):
        current_batch = flattened_batch[i : i + batch_size]
        batches.append(current_batch)
    return batches

def auto_flatten_batches(dset_dir: str) -> None:
    r"""Goes through a dataset and flattens all the batches
    
    Arguments:
        dset_dir (str): The dataset directory to flatten the batches for
    
    Returns:
        None
    
    Notes: Goes through the directory and flattens the batch_original.p files in 
        each of the Fold{num} directories. Saves the file as batch_flat.p in that directory
    """
    all_files = os.listdir(dset_dir)
    for elem in all_files:
        #Right now there are only two folds we care about 
        if elem  in ['Fold0', 'Fold1']:
            num = int(elem[-1])
            print(f"Flattening for {elem}")
            full_direc_path = os.path.join(os.getcwd(), dset_dir, elem)
            batch_path = os.path.join(full_direc_path, 'batch_original.p')
            with open(batch_path, 'rb') as handle:
                batches = pickle.load(handle)
            flattened_batches = flatten_batches(batches)
            with open(os.path.join(full_direc_path, f'batch_flat_{num}.p'), 'wb') as handle:
                pickle.dump(flattened_batches, handle)
    print(f"All batches flattened for {dset_dir}")

def write_out_repulsive_model(trained_rep_mod: DFTBRepulsiveModel, ref_set: str, dest: str,
                              spl_ngrid: int = 500) -> None:
    r"""Write out the repulsive model into a series of SKF files using the reference
        set, effectively splicing the new repulsive model together with the 
        original electronic.
    
    Arguments:
        trained_rep_mod (DFTBRepulsiveModel): The trained repulsive model.
        ref_set (str): The path to the reference set of SKF files to use for everything but
            the repulsive spline.
        dest (str): The destination to save the newly written SKF files to.
        spl_ngrid (int): The number of grid points used in writing out the spline repulsive 
            for the SKF files. Defaults to 500.
        
    Returns:
        None
    
    Notes: Writing out the repulsive model involves using the electronic portion 
        and exponential coefficients of the original SKF file and just replacing the 
        spline block. The ref_set used in this method should be the same as the 
        SKF set used to generate the initial electronic predictions used to train 
        the repulsive model. 
        
        Also need to write out the repulsive parameters
    """
    ref_skset = read_skf_set(ref_set)
    creator = SKFBlockCreator()
    #Get the model xy data
    grid_dict = {elems : np.linspace(v[0], v[1], spl_ngrid) for elems, v in trained_rep_mod.mod.opts.cutoff.items()}
    #Setting expand = True includes the reflexives of hte atomic pairs
    xy_data = trained_rep_mod.mod.create_xydata(grid_dict, expand = True)
    all_Zs = xy_data.keys()
    for Zs in all_Zs:
        print(f"Currently writing SKF for {Zs}")
        curr_ref_skf = ref_skset[Zs]
        spline_block = creator.create_spline_block(curr_ref_skf['exp_coef'].to_numpy()[0], xy_data[Zs])
        #Re-assign the spline block to create a new SKF
        curr_ref_skf['spline'] = spline_block
        name = f"{atom_nums[Zs[0]]}-{atom_nums[Zs[1]]}.skf"
        curr_ref_skf.to_file(os.path.join(dest, name))
    #Also write out the reference energy parameters
    ref_info = trained_rep_mod.get_ref_ener_info(False)
    with open(os.path.join(dest, "ref_params.p"), 'wb') as handle:
        pickle.dump(ref_info, handle)

def copy_molecules_from_analysis_dir(dset_dir: str, batch_size: int = 10) -> None:
    r"""Copies the calcultion results from analysis_dir/analysis_files
        into the corresponding Fold directory.
    
    Arguments:
        dset_dir (str): The dataset directory to perform the copy operation for
    
    Returns:
        None
    
    Notes: Also reformats to produce the batches again.
    """
    all_analysis_files = os.path.join(os.getcwd(), dset_dir, 'analysis_dir', 'analysis_files')
    pattern = r'Fold[0-9]+'
    for file in os.listdir(all_analysis_files):
        #Only care about the pickle files containing the calculation results
        if file.split(".")[-1] == 'p':
            filename_splt = file.split('.')[0].split("_")
            #Find the corresponding fold directory
            fold_dir = list(filter(lambda x : re.match(pattern, x), filename_splt))[0]
            src = os.path.join(all_analysis_files, file)
            with open(src, 'rb') as handle:
                calc_mols = pickle.load(handle)
            calc_batches = reconstruct_batches(calc_mols, batch_size)
            dst = os.path.join(os.getcwd(), dset_dir, fold_dir, 'processed_batches.p')
            with open(dst, 'wb') as handle:
                pickle.dump(calc_batches, handle)
            print(f"Shifted {src} to {dst}")

def train_all_repulsive_models(dset_dirs: List[str], batch_size: int, settings_file: str, defaults_file: str,
                               ref_set: str) -> None:
    r"""Reorganizes the data and trains the repulsive model on each of the datasets. 
        It then writes out the SKF files for each of the trained models.
    
    Arguments:
        dset_dirs (List[str]): The list of dataset directories to train on
        batch_size (int): The number of molecules per batch (used for re-organization)
        settings_file (str): Settings json file used for training the repulsive model
        defaults_file (str): The defaults json file for training the repulsive model
        ref_set (str): Path to the reference set of SKF files used to perform the 
            initial DFTB+ calculations
    
    Returns:
        None
    
    Notes: This method performs a re-organization as well as the training calculation
        and writing output. 
    """
    for dset in dset_dirs:
        print(f"Performing training for {dset}")
        #First step is re-organize the data through copying
        copy_molecules_from_analysis_dir(dset, batch_size)
        #Perform the training
        rep_mod = train_repulsive_model(dset, settings_file, defaults_file)
        #Write the output SKF files
        dset_name = os.path.split(dset)[-1]
        destination = os.path.join(os.getcwd(), dset, f"{dset_name}_SKFs")
        if (not os.path.isdir(destination)):
            os.mkdir(destination)
        write_out_repulsive_model(rep_mod, ref_set, destination)
        #Last step: copy the test set
        src = os.path.join(os.getcwd(), dset, 'test_set.p')
        shutil.copy(src, destination)
        