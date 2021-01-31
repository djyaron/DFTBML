# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 17:41:58 2021

@author: Frank

Module containing methods for generating and saving the folds in the correct file format.
To accomplish this, each fold is saved in its own directory under a mastery directory:
TotalData:
    Fold0:
        Train_molecs.h5
        Train_batches.h5
        Valid_molecs.h5
        Valid_batches.h5
        Train_dftblsts.p
        Valid_dftblsts.p
        Train_ref.p
        Valid_ref.p
        Train_fold_molecs.p (pickle file of the raw molecule dictionaries)
        Valid_fold_molecs.p (pickle file of the raw molecule dictionaries)
    Fold1:
        ...
    ...

The file names are fixed so that everything can be properly read from the fold 
without any additional work in figuring out the file names. Only the top level directory
containing all the fold subdirectories can be alternately named by the user
"""

from dftb_layer_splines_4 import load_data, saving_data, graph_generation, feed_generation, model_loss_initialization
from dftbrep_fold import get_folds_cv_limited, extract_data_for_molecs
from typing import List, Union, Dict
from batch import DFTBList
import collections
import os, os.path
from h5handler import per_molec_h5handler, per_batch_h5handler, total_feed_combinator, compare_feeds
import pickle, json
import importlib
import random
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.spatial.distance import pdist
from matplotlib.ticker import AutoMinorLocator
import re

#%% General functions and constants
class Settings:
    def __init__(self, settings_dict: Dict) -> None:
        r"""Generates a Settings object from the given dictionary
        
        Arguments:
            settings_dict (Dict): Dictionary containing key value pairs for the
                current hyperparmeter settings
        
        Returns:
            None
        
        Notes: Using an object rather than a dictionary is easier since you can
            just do settings.ZZZ rather than doing the bracket notation and the quotes.
        """
        for key in settings_dict:
            setattr(self, key, settings_dict[key])

def energy_correction(molec: Dict) -> None:
    r"""Performs in-place total energy correction for the given molecule by dividing Etot/nheavy
    
    Arguments:
        molec (Dict): The dictionary in need of correction
    
    Returns:
        None
    """
    zcount = collections.Counter(molec['atomic_numbers'])
    ztypes = list(zcount.keys())
    heavy_counts = [zcount[x] for x in ztypes if x > 1]
    num_heavy = sum(heavy_counts)
    molec['targets']['Etot'] = molec['targets']['Etot'] / num_heavy

atom_nums = {
    6 : "C",
    1 : "H",
    7 : "N",
    8 : "O"
    }
#%% Methods for generating and saving folds
def generate_fold_molecs(s: Settings):
    r"""Generates the molecules in each fold
    
    Arguments:
        s (Settings): Settings object containing all the necessary values for the
            hyperparameters
    
    Returns:
        fold_molecs (List[(List[Dict], List[Dict])]): A list of tuples of molecule dictionary
            lists where the first list is the training molecules and the second list is the 
            validation molecules
    """
    folds_cv = get_folds_cv_limited(s.allowed_Zs, s.heavy_atoms, s.data_path, s.num_folds, s.max_config, s.exclude, shuffle = tuple(s.shuffle), 
                                    reverse = False if s.cv_mode == 'normal' else True)
    fold_molecs = list()
    for fold in folds_cv:
        training_molecs, validation_molecs = extract_data_for_molecs(fold, s.target, s.data_path)
        if s.train_ener_per_heavy:
            for molec in training_molecs:
                energy_correction(molec)
            for molec in validation_molecs:
                energy_correction(molec)
        random.shuffle(training_molecs)
        random.shuffle(validation_molecs)
        fold_molecs.append((training_molecs, validation_molecs))
    return fold_molecs

def fold_precompute(s: Settings, par_dict: Dict, training_molecs: List[Dict], 
                    validation_molecs: List[Dict]):
    r"""Generates the computational feed dictionaries for each fold
    
    Arguments:
        s (Settings): The settings object containing all the hyperparameter settings
        par_dict (Dict): Dictionary of skf parameters
        training_molecs (List[Dict]): List of molecule dictionaries for the training molecules
        validation_molecs (List[Dict]): List of molecule dictionaries for the validation molecules
        
    
    Returns:
        training_feeds (List[Dict]): The generated training feed dictionaries
        validation_feeds (List[Dict]): The generated validation feed dictionaries
        training_dftblsts (List[DFTBList]): The DFTBList objects to go along with the 
            training feeds
        validation_dftblsts (List[DFTBList]): The DFTBList objects to go along with the validation feeds

    """
    config = {"opers_to_model" : s.opers_to_model}
    training_feeds, training_dftblsts, training_batches = graph_generation(training_molecs, config, s.allowed_Zs, par_dict, s.num_per_batch)
    validation_feeds, validation_dftblsts, validation_batches = graph_generation(validation_molecs, config, s.allowed_Zs, par_dict, s.num_per_batch)
    
    losses = dict()
    for loss in s.losses:
        #s.losses is a list of strings representing the different losses to factor into backpropagation
        if loss == 'Etot':
            losses[loss] = s.target_accuracy_energy
        elif loss == 'dipole':
            losses[loss] = s.target_accuracy_dipole
        elif loss == 'charges':
            losses[loss] = s.target_accuracy_charges
        elif loss == 'convex':
            losses[loss] = s.target_accuracy_convex
        elif loss == 'monotonic':
            losses[loss] = s.target_accuracy_monotonic
        else:
            raise ValueError("Unsupported loss type")
    
    all_models, model_variables, loss_tracker, all_losses, model_range_dict = model_loss_initialization(training_feeds, validation_feeds,
                                                                               s.allowed_Zs, losses, ref_ener_start = s.reference_energy_starting_point)
    
    #the loaded_data argument is hard set to False since we are not going to be loading data when trying to generate the folds
    feed_generation(training_feeds, training_batches, all_losses, all_models, model_variables, model_range_dict, par_dict, s.spline_mode, s.spline_deg, s.debug, False, 
                    s.num_knots, s.buffer, s.joined_cutoff, s.cutoff_dictionary, s.off_diag_opers, s.include_inflect)
    feed_generation(validation_feeds, validation_batches, all_losses, all_models, model_variables, model_range_dict, par_dict, s.spline_mode, s.spline_deg, s.debug, False, 
                    s.num_knots, s.buffer, s.joined_cutoff, s.cutoff_dictionary, s.off_diag_opers, s.include_inflect)
    print(f"inflect mods: {[mod for mod in model_variables if mod != 'Eref' and mod.oper == 'S' and 'inflect' in mod.orb]}")
    print(f"s_mods: {[mod for mod in model_variables if mod != 'Eref' and mod.oper == 'S']}")
    print(f"len of s_mods: {len([mod for mod in model_variables if mod != 'Eref' and mod.oper == 'S'])}")
    print(f"len of s_mods in all_models: {len([mod for mod in all_models if mod != 'Eref' and mod.oper == 'S'])}")
    print("losses")
    print(losses)
    print(training_feeds[0].keys())
    return training_feeds, validation_feeds, training_dftblsts, validation_dftblsts

def saving_fold(s: Settings, training_feeds: List[Dict], validation_feeds: List[Dict],
                 training_dftblsts: List[DFTBList], validation_dftblsts: List[DFTBList],
                 training_molecs: List[Dict], validation_molecs: List[Dict], top_fold_path: str,
                 fold_num: int) -> None:
    r"""Method for saving the feeds using the h5 handler methods
    
    Arguments:
        s (Settings): The Settings object containing all the relevant hyperparameters
        training_feeds (List[Dict]): The list of training feed dictionaries
        validation_feeds (List[Dict]): The list of validation feed dictionaries
        training_dfbtlsts (List[DFTBList]): The list of training DFTBList objects to go along
            with training_feeds
        validation_dftblsts (List[DFTBList]): The list of validation DFTBList objects to go
            along with validation_feeds
        training_molecs (List[Dict]): List of molecule dictionaries for the training molecules
        validation_molecs (List[Dict]): List of molecule dictionaries for the validation molecules
        top_fold_path (str): The file path to the top level directory containing the folds
        fold_num (int): The number of the current fold
        
    Returns:
        None
    
    Notes: This method saves the information for the current set of training and validation feeds, which
        together represent the information of one fold
    """
    current_folder_name = f"Fold{fold_num}"
    total_folder_path = os.path.join(top_fold_path, current_folder_name)
    if not os.path.isdir(total_folder_path):
        os.mkdir(total_folder_path)
    train_molec_filename = os.path.join(total_folder_path, 'train_molecs.h5')
    valid_molec_filename = os.path.join(total_folder_path, 'valid_molecs.h5')
    train_batch_filename = os.path.join(total_folder_path, 'train_batches.h5')
    valid_batch_filename = os.path.join(total_folder_path, 'valid_batches.h5')
    train_reference_filename = os.path.join(total_folder_path, 'train_reference.p')
    valid_reference_filename = os.path.join(total_folder_path, 'valid_reference.p')
    train_dftblst_filename = os.path.join(total_folder_path, 'train_dftblsts.p')
    valid_dftblst_filename = os.path.join(total_folder_path, 'valid_dftblsts.p')
    train_raw_molecs_filename = os.path.join(total_folder_path, 'train_total_molecs.p')
    valid_raw_molecs_filename = os.path.join(total_folder_path, 'valid_total_molecs.p')
    
    #Save the training information
    per_molec_h5handler.save_all_molec_feeds_h5(training_feeds, train_molec_filename)
    per_batch_h5handler.save_multiple_batches_h5(training_feeds, train_batch_filename)
    
    #Save the validation information
    per_molec_h5handler.save_all_molec_feeds_h5(validation_feeds, valid_molec_filename)
    per_batch_h5handler.save_multiple_batches_h5(validation_feeds, valid_batch_filename)
    
    with open(train_reference_filename, 'wb') as handle:
        pickle.dump(training_feeds, handle)
    
    with open(valid_reference_filename, 'wb') as handle:
        pickle.dump(validation_feeds, handle)
    
    with open(train_dftblst_filename, 'wb') as handle:
        pickle.dump(training_dftblsts, handle)
    
    with open(valid_dftblst_filename, 'wb') as handle:
        pickle.dump(validation_dftblsts, handle)
        
    with open(train_raw_molecs_filename, 'wb') as handle:
        pickle.dump(training_molecs, handle)
    
    with open(valid_raw_molecs_filename, 'wb') as handle:
        pickle.dump(validation_molecs, handle)
    
    print("All information successfully saved")

def loading_fold(s: Settings, top_fold_path: str, fold_num: int):
    r"""Loads the data from a given fold
    
    Arguments:
        s (Settings): The settings object containing hyperparameter settings
        top_folder_path (str): The path to the top level directory containing all
            the fold subdirectories
        target_fold_num (int): The fold number that should be loaded
    
    Returns:
        training_feeds (List[Dict]): List of feed dictionaries for training
        validation_feeds (List[Dict]): List of feed dictionaries for validation
        training_dftblsts (List[DFTBList]): List of DFTBList objects for training feeds
        validation_dftblsts (List[DFTBList]): List of DFTBList objects for validation feeds
        
    Notes: The check will only be performed if required by the value in s
    """
    current_folder_name = f"Fold{fold_num}"
    total_folder_path = os.path.join(top_fold_path, current_folder_name)
    if not os.path.isdir(total_folder_path):
        raise ValueError("Data for fold does not exist")
    print(f"loading data from {total_folder_path}")
    train_molec_filename = os.path.join(total_folder_path, 'train_molecs.h5')
    valid_molec_filename = os.path.join(total_folder_path, 'valid_molecs.h5')
    train_batch_filename = os.path.join(total_folder_path, 'train_batches.h5')
    valid_batch_filename = os.path.join(total_folder_path, 'valid_batches.h5')
    train_reference_filename = os.path.join(total_folder_path, 'train_reference.p')
    valid_reference_filename = os.path.join(total_folder_path, 'valid_reference.p')
    train_dftblst_filename = os.path.join(total_folder_path, 'train_dftblsts.p')
    valid_dftblst_filename = os.path.join(total_folder_path, 'valid_dftblsts.p')
    
    training_feeds = total_feed_combinator.create_all_feeds(train_batch_filename, train_molec_filename, s.ragged_dipole)
    validation_feeds = total_feed_combinator.create_all_feeds(valid_batch_filename, valid_molec_filename, s.ragged_dipole)
    
    if s.run_check:
        print("Running safety check")
        compare_feeds(train_reference_filename, training_feeds)
        compare_feeds(valid_reference_filename, validation_feeds)
    
    training_dftblsts = pickle.load(open(train_dftblst_filename, "rb"))
    validation_dftblsts = pickle.load(open(valid_dftblst_filename, "rb"))
    
    return training_feeds, validation_feeds, training_dftblsts, validation_dftblsts

def generate_save_folds(settings_path: str) -> None:
    r"""Generates and saves folds based on the configurations presented in
        the settings_path json file
    
    Arguments:
        settings_path (str): The path to the file that specifies the parameters 
            for generating the folds
    
    Returns:
        None
    
    Notes: The idea behind making the fold generater and cldriver depend on the 
        same settings file JSON format is for convenience
    """
    with open(settings_path, 'r') as read_file:
        input_settings_dict = json.load(read_file)
    settings = Settings(input_settings_dict)
    
    top_fold_path = settings.top_level_fold_path
    if not os.path.isdir(top_fold_path):
        os.mkdir(top_fold_path)
        
    par_dict_path = settings.par_dict_name
    if par_dict_path == 'auorg_1_1':
        from auorg_1_1 import ParDict
        par_dict = ParDict()
    else:
        module = importlib.import_module(par_dict_path)
        par_dict = module.ParDict()
        
    
    #Generate the folds in terms of the molecules in the train and validation sets
    fold_molecs = generate_fold_molecs(settings)
    
    for ind, (train_molecs, valid_molecs) in enumerate(fold_molecs):
        training_feeds, validation_feeds, training_dftblsts, validation_dftblsts = fold_precompute(settings, par_dict, 
                                                                                                   train_molecs, valid_molecs)
        saving_fold(settings, training_feeds, validation_feeds, training_dftblsts, validation_dftblsts,
                    train_molecs, valid_molecs, top_fold_path, ind)
        print(f"Fold {ind} saved")
    
    print("All folds saved successfully")

#%% Methods for analyzing fold data distance distributions
def get_dist_dict(molec: Dict) -> dict:
    r"""Generate pairwise distances of conformations of a molecular formula
    Args:
        molec (Dict): The dictionary representation of a molecule
        
    Returns:
        result_dict (Dict): whose keys are atom pairs,
            and values are a 1-D np.ndarray of pairwise distances
    """
    atoms = molec['atomic_numbers']
    coords_mol = molec['coordinates'] #(Natom x 3)
    
    atom_pairs = tuple(combinations(atoms, 2))  #: atom pairs, used to keep track of the output of pdist
    zs = sorted(set(combinations(sorted(atoms), 2)))  #: sorted and deduplicated atom pairs
    distances = pdist(coords_mol)
    
    result_dict = {z : [] for z in zs}
    for i, pair in enumerate(atom_pairs):
        pair_rev = (pair[1], pair[0])
        if pair in result_dict:
            result_dict[pair].append(distances[i])
        elif pair_rev in result_dict:
            result_dict[pair_rev].append(distances[i])
    
    return result_dict

def correct_tuple_to_str(elem_nums: tuple) -> str:
    r"""Takes a tuple of atomic numbers and converts it to a string of atom 
        symbols
    
    Arguments:
        elem_nums (tuple): Tuple of atomic numbers to convert
    
    Returns:
        str: String of the two element symbols separated by a dash
    
    Example: (6, 6) => "C-C"
    """
    sym1, sym2 = atom_nums[elem_nums[0]], atom_nums[elem_nums[1]]
    return f"{sym1}-{sym2}"

def analyze_molec_set(molecs: List[Dict]) -> Dict:
    r"""Generates a dictionary of all the pairwise distances seen in a set of molecules
    
    Arguments:
        molecs (List[Dict]): The list of molecule dictionaries representing the current 
            molecule set
    
    Returns:
        total_distances (Dict): A dictionary indexed by atom symbol pair containing
            all the distances seen for that atom pair
    """
    total_distances = dict()
    for elem in molecs:
        molecule_result = get_dist_dict(elem)
        for pair in molecule_result:
            corrected_pair = correct_tuple_to_str(pair)
            if corrected_pair not in total_distances:
                total_distances[corrected_pair] = molecule_result[pair]
            else:
                total_distances[corrected_pair] += molecule_result[pair]

    return total_distances

def plot_distributions(total_distances: Dict, dest: str, set_label: str = 'train',
                       x_min: float = 0.05, x_max: float = 10.0, n_bins: int = 150,
                       minor_locator_factor: int = 10) -> None:
    r"""Method to plot the distribution of distances as histograms, and to save those
        figures
    
    Arguments:
        total_distances (Dict): Dictionary of total distances indexed by a string of 
            atom elemets, e.g. 'C-C' : [dist_1, dist_2, dist_3, ..., dist_n] output from
            analyze_molec_set
        dest (str): The relative path to the destination where the figure will be saved. 
            The plot name is appended onto this destination string to get the total save path
        set_label (str): Whether this is the train or validation set of molecules
        x_min (float): The minimum value for the x-axis
        x_max (float): The maximum value for the x-axis
        n_bins (int): The number of bins to use for each histogram
        minor_locator_factor (int): The number of minor tick marks assigned by the
            AutoMinorLocator. Defaults to 10
    
    Returns:
        None
    
    Notes: Fixing the ranges on the x-axis makes it easier to compare the histograms.
        The x_min value must be strictly smaller than the x_max value
    """
    assert(x_max > x_min)
    target_directory = os.path.join(dest, f"{set_label}_distributions")
    if not os.path.isdir(target_directory):
        os.mkdir(target_directory)
    for pair in total_distances:
        fig, axs = plt.subplots()
        axs.hist(total_distances[pair], bins = n_bins)
        axs.set_xlim(left = x_min, right = x_max)
        axs.set_xlabel("Distance (Angstroms)")
        axs.set_ylabel("Frequency")
        axs.set_title(pair)
        axs.xaxis.set_minor_locator(AutoMinorLocator(minor_locator_factor))
        axs.yaxis.set_minor_locator(AutoMinorLocator(minor_locator_factor))
        fig.savefig(os.path.join(target_directory, pair + "_dist.png"))
    #Also save the total distances information for later use
    with open(os.path.join(target_directory, "dist_info.p"), 'wb') as handle:
        pickle.dump(total_distances, handle)
    print(f"Destination: {dest}")
    print(f"Target directory {target_directory}")
    print("Finished generating distributions")

def analyze_all_folds(top_level_fold_path: str) -> None:
    r"""Method to go through all the saved folds and use the saved molecule information
        for plotting distributions
    
    Arguments:
        top_level_fold_path (str): The path to the top level directory containing all 
            fold subdirectories
    
    Retuns:
        None
    """
    if not os.path.isdir(top_level_fold_path):
        raise ValueError("Data for the folds does not exist")
    sub_direcs = os.listdir(top_level_fold_path)
    pattern = r"Fold[0-9]+"
    filtered_subdirecs = list(filter(lambda x : re.match(pattern, x) is not None, sub_direcs))
    for sub_direc in filtered_subdirecs:
        train_total_path = os.path.join(top_level_fold_path, sub_direc, 'train_total_molecs.p')
        valid_total_path = os.path.join(top_level_fold_path, sub_direc, 'valid_total_molecs.p')
        train_molecs = pickle.load(open(train_total_path, 'rb'))
        valid_molecs = pickle.load(open(valid_total_path, 'rb'))
        total_train_dists = analyze_molec_set(train_molecs)
        total_valid_dists = analyze_molec_set(valid_molecs)
        plot_distributions(total_train_dists, os.path.join(top_level_fold_path, sub_direc), 
                           'train')
        plot_distributions(total_valid_dists, os.path.join(top_level_fold_path, sub_direc),
                           'valid')

if __name__ == "__main__":
    
    pass
    
    # #Testing saving features
    # settings_json = "settings_default.json"
    # generate_save_folds(settings_json)
    
    # with open(settings_json, 'r') as read_file:
    #     settings = Settings(json.load(read_file))
    
    # #Testing loading data for the folds
    # top_fold_path = "test_fold"
    # training_feeds, validation_feeds, training_dftblsts, validation_dftblsts = loading_fold(settings, top_fold_path, 0)
    
    #Testing plotting features for the histograms
    test_molec_path = os.path.join("test_fold", "Fold0", "train_total_molecs.p")
    with open(test_molec_path, 'rb') as handle:
        data = pickle.load(handle)
    test_molec = data[0]
    print(test_molec)
    result_dict = get_dist_dict(test_molec)
    print(result_dict)
    total_distances = analyze_molec_set(data)
    # print(total_distances)
    dest = "test_distr"
    plot_distributions(total_distances, dest)
    
    #Run through all the subdirectories in test_fold
    top_level_path = 'test_fold'
    analyze_all_folds(top_level_path)
    
    
    
        
        
    
    
    
    
    
        

