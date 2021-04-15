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

TODO:
    1) Finish single distribution plotting function
    2) Start testing stuff out
"""

from dftb_layer_splines_4 import graph_generation, feed_generation, model_loss_initialization,\
    get_ani1data
from dftbrep_fold import get_folds_cv_limited, extract_data_for_molecs
from typing import List, Dict
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
from scipy.stats import ks_2samp, iqr, mannwhitneyu

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
        
#%% Generate folds based on nheavy
def count_nheavy(molec: Dict) -> int:
    r"""Counts the number of heavy atoms in a molecule
    
    Arguments:
        molec (Dict): Dictionary representation of a molecule
    
    Returns:
        n_heavy (int): The number of heavy molecules
    """
    n_heavy = 0
    for elem in molec['atomic_numbers']:
        if elem > 1:
            n_heavy += 1
    return n_heavy

def get_folds_from_molecs(num_molecs: int, num_folds_lower: int, num_folds_higher: int, 
                          lower_molecs: List[Dict], higher_molecs: List[Dict]) -> List[List[Dict]]:
    r"""Generates the folds given the number of molecules per fold and the molecules
    
    Arguments:
        num_molecs (int): The number of molecules to have for each fold
        num_folds_lower (int): The number of folds generated from lower_molecs
        num_folds_higher (int): The number of folds generated from higher_molecs
        lower_molecs (List[Dict]): The molecule dictionaries with molecules
            only having up to some nheavy limit
        higher_molecs (List[Dict]): The molecule dictionaries with molecules
            containing more heavy atoms than those in lower_molecs
    
    Returns:
        folds (List[List[Dict]]): The molecules for each fold
    """
    folds = list()
    for i in range(num_folds_lower):
        start, end = i * num_molecs, (i + 1) * num_molecs
        folds.append(lower_molecs[start : end])
    for j in range(num_folds_higher):
        start, end = j * num_molecs, (j + 1) * num_molecs
        folds.append(higher_molecs[start : end])
    return folds
    
def generate_folds(allowed_Zs: List[int], heavy_atoms: List[int], max_config: int, 
                   target: Dict[str, str], data_path: str, exclude: List[str], lower_limit: int, 
                   num_folds: int, num_folds_lower: int) -> List[List[Dict]]:
    r"""Generates folds based on the number of heavy atoms by dividing up the molecules
    
    Arguments:
        allowed_Zs (List[int]): The allowed elements in the dataset
        heavy_atoms (List[int]): The allowed heavy (non-hydrogen) atoms
        max_config (int): The maximum number of configurations
        target (Dict): Dictionary mapping the target names (e.g. 'Etot') to the 
            ani1 target names (e.g. 'cc')
        data_path (str): The relative path to the dataset from which to pull molecules
        exclude (List[str]): The molecular formulas to exclude when pulling the dataset
        lower_limit (int): The number of heavy atoms to include up to for the folds containing
            lower heavy elements (e.g. folds up to 5)
        num_folds (int): The total number of folds
        num_folds_lower (int): The number of folds that contain molecules with heavy atoms
            only up to lower_limit
    
    Returns:
        fold_molecs (List[List[Dict]]): A list of list of dictionaries where each inner list 
            is a set of molecules for the fold
    
    Notes: This approach does not use the Fold class and instead segments the data
        based on the number of heavy atoms, with num_folds_lower folds containing 
        only molecules with up to lower_limit number of heavy atoms. If done right, this 
        only has to be done once which is why it is not dependent on the settings file.
    """
    #Grab the dataset using get_ani1_data
    assert(num_folds_lower <= num_folds)
    dataset = get_ani1data(allowed_Zs, heavy_atoms, max_config, target, data_path, exclude)
    print(f"Number of molecules: {len(dataset)}")
    heavy_mapped = list(map(lambda x : (x, count_nheavy(x)), dataset))
    lower_molecs = [elem[0] for elem in heavy_mapped if elem[1] <= lower_limit]
    higher_molecs = [elem[0] for elem in heavy_mapped if elem[1] > lower_limit]
    assert(len(lower_molecs) + len(higher_molecs) == len(dataset))
    num_folds_higher = num_folds - num_folds_lower
    random.shuffle(lower_molecs)
    random.shuffle(higher_molecs)
    
    #Figure out the limiting factor between the lower and higher molecules in generating the folds
    num_molec_per_fold_lower = int(len(lower_molecs) / num_folds_lower)
    num_molec_per_fold_higher = int(len(higher_molecs) / num_folds_higher)
    
    lower_criterion = len(higher_molecs) >= num_folds_higher * num_molec_per_fold_lower
    higher_criterion = len(lower_molecs) >= num_folds_lower * num_molec_per_fold_higher
    
    if lower_criterion: #Give precedence to the lower molecules
        num_molecs = num_molec_per_fold_lower
    elif higher_criterion:
        num_molecs = num_molec_per_fold_higher
    folds = get_folds_from_molecs(num_molecs, num_folds_lower, num_folds_higher,
                                      lower_molecs, higher_molecs)
    return folds

def perform_KS_test(distances_1: Dict, distances_2: Dict, p_threshold: float,
                    statistic_threshold: float = None) -> bool:
    r"""Runs the KS test across each category of interatomic distance
    
    Arguments:
        distances_1 (Dict): First dictionary of distances
        distances_2 (Dict): Second dictionary of distances
        p_threshold (float): The minimum value that the p value has to exceed
            for the distributions to be acceptably similar. Default is 0.05 (5%)
        statistic_threshold (float): The maximum value that the Kolmogorov-Smirnov test
            statistic can have for the distributions to be considered reasonably similar.
            This is optional as the p-value alone should be sufficient, so the 
            value defaults to None.
    
    Returns:
        similar (bool): True if the two dictionaries have similar distributions
            for all interatomic distances, False otherwise
    
    Notes: The distances dictionaries will have the following format:
        distances_1: {'C-C' : [dist_1, dist_2, ..., dist_n],
                      'C-O' : [dist_1, dist_2, ..., dist_n],
                      'H-H' : [dist_1, dist_2, ..., dist_n],
                      ...}
        The function only returns true if the KS test returns that the two distributions
        are similar for each atom pair across both dictionaries. It is expected that 
        distances_1 and distances_2 have the same keys.
        
        In the case that the KS test fails, a histogram is plotted for visual inspection
    """
    assert(set(distances_1.keys()) == set(distances_2.keys()))
    for key in distances_1:
        first_dict_distr = distances_1[key]
        second_dict_distr = distances_2[key]
        test_result = ks_2samp(first_dict_distr, second_dict_distr)
        statistic, p_val = test_result.statistic, test_result.pvalue
        p_passed = p_val >= p_threshold
        if not p_passed:
            print(f"KS test failed for {key} on p_value")
            print(statistic, p_val)
            total_data = first_dict_distr + second_dict_distr
            total_min, total_max = min(total_data), max(total_data)
            plot_single_distribution(first_dict_distr, key, total_min, total_max)
            plot_single_distribution(second_dict_distr, key, total_min, total_max)
            return False
        statistic_passed = statistic <= statistic_threshold if statistic_threshold is not None else True
        if not statistic_passed:
            print(f"KS test failed for {key} on statistic")
            print(statistic, p_val)
            total_data = first_dict_distr + second_dict_distr
            total_min, total_max = min(total_data), max(total_data)
            plot_single_distribution(first_dict_distr, key, total_min, total_max)
            plot_single_distribution(second_dict_distr, key, total_min, total_max)
            return False
    return True

def perform_MWU_test(distances_1: Dict, distances_2: Dict, p_threshold: float,
                    statistic_threshold: float = None, dist_1_marker: str = "", dist_2_marker: str = "") -> bool:
    r"""Runs the mann_whitney_U test across each category of interatomic distance
    
    Arguments:
        distances_1 (Dict): First dictionary of distances
        distances_2 (Dict): Second dictionary of distances
        p_threshold (float): The minimum value that the p value has to exceed
            for the distributions to be acceptably similar. Default is 0.05 (5%)
        statistic_threshold (float): The maximum value that the Kolmogorov-Smirnov test
            statistic can have for the distributions to be considered reasonably similar.
            This is optional as the p-value alone should be sufficient, so the 
            value defaults to None.
        dist_1_marker (str): Identifier for where the first distance dictionary comes from
        dist_2_marker (str): Identifier for where the second distance dictionary comes from
    
    Returns:
        similar (bool): True if the two dictionaries have similar distributions
            for all interatomic distances, False otherwise
    
    Notes: The distances dictionaries will have the following format:
        distances_1: {'C-C' : [dist_1, dist_2, ..., dist_n],
                      'C-O' : [dist_1, dist_2, ..., dist_n],
                      'H-H' : [dist_1, dist_2, ..., dist_n],
                      ...}
        The function only returns true if the MWU test returns that the two distributions
        are similar for each atom pair across both dictionaries. It is expected that 
        distances_1 and distances_2 have the same keys.
        
        In the case that the MWU test fails, a histogram is plotted for visual inspection
    """
    assert(set(distances_1.keys()) == set(distances_2.keys()))
    for key in distances_1:
        if key != 'H-H':
            first_dict_distr = distances_1[key]
            second_dict_distr = distances_2[key]
            test_result = mannwhitneyu(first_dict_distr, second_dict_distr)
            statistic, p_val = test_result.statistic, test_result.pvalue
            p_passed = p_val >= p_threshold
            if not p_passed:
                print(f"MWU test failed for {key} on p_value")
                print(statistic, p_val)
                total_data = first_dict_distr + second_dict_distr
                total_min, total_max = min(total_data), max(total_data)
                plot_single_distribution(first_dict_distr, key + dist_1_marker, total_min, total_max)
                plot_single_distribution(second_dict_distr, key + dist_2_marker, total_min, total_max)
                return False
            statistic_passed = statistic <= statistic_threshold if statistic_threshold is not None else True
            if not statistic_passed:
                print(f"MWU test failed for {key} on statistic")
                print(statistic, p_val)
                total_data = first_dict_distr + second_dict_distr
                total_min, total_max = min(total_data), max(total_data)
                plot_single_distribution(first_dict_distr, key, total_min, total_max)
                plot_single_distribution(second_dict_distr, key, total_min, total_max)
                return False
    return True

def perform_statistic_test(distances_1: Dict, distances_2: Dict, p_threshold: float, statistic_test: str = 'KS',
                    statistic_threshold: float = None, dist_1_marker: str = "", dist_2_marker: str = "",
                    exclude: List[str] = []) -> bool:
    r"""Runs either the Mann-Whitney U test or Kolmogorov Smirnov test across all interatomic 
        distance categories
    
    Arguments:
        distances_1 (Dict): First dictionary of distances
        distances_2 (Dict): Second dictionary of distances
        p_threshold (float): The minimum value that the p value has to exceed
            for the distributions to be acceptably similar. Default is 0.05 (5%)
        statistic_test (str): The statistic test to use. Defaults to Kolmogorov-Smirnov (KS)
        statistic_threshold (float): The maximum value that the Kolmogorov-Smirnov test
            statistic can have for the distributions to be considered reasonably similar.
            This is optional as the p-value alone should be sufficient, so the 
            value defaults to None.
        dist_1_marker (str): Identifier for where the first distance dictionary comes from
        dist_2_marker (str): Identifier for where the second distance dictionary comes from
        exclude (Listr[str]): The list of element pairs (e.g. ['H-H', 'H-C']) that should
            be excluded from analysis
    
    Returns:
        similar (bool): True if the two dictionaries have similar distributions
            for all interatomic distances, False otherwise
    
    Notes: The distances dictionaries will have the following format:
        distances_1: {'C-C' : [dist_1, dist_2, ..., dist_n],
                      'C-O' : [dist_1, dist_2, ..., dist_n],
                      'H-H' : [dist_1, dist_2, ..., dist_n],
                      ...}
        The function only returns true if the  test returns that the two distributions
        are similar for each atom pair across both dictionaries. It is expected that 
        distances_1 and distances_2 have the same keys.
        
        In the case that the test fails, a histogram is plotted for visual inspection
    """
    assert(set(distances_1.keys()) == set(distances_2.keys()))
    for key in distances_1:
        if key not in exclude:
            first_dict_distr = distances_1[key]
            second_dict_distr = distances_2[key]
            if statistic_test == 'MWU':
                test_result = mannwhitneyu(first_dict_distr, second_dict_distr)
            elif statistic_test == 'KS':
                test_result = ks_2samp(first_dict_distr, second_dict_distr)
            statistic, p_val = test_result.statistic, test_result.pvalue
            p_passed = p_val >= p_threshold
            if not p_passed:
                print(f"{statistic_test} test failed for {key} on p_value")
                print(statistic, p_val)
                total_data = first_dict_distr + second_dict_distr
                total_min, total_max = min(total_data), max(total_data)
                plot_single_distribution(first_dict_distr, key + dist_1_marker, total_min, total_max)
                plot_single_distribution(second_dict_distr, key + dist_2_marker, total_min, total_max)
                return False
            statistic_passed = statistic <= statistic_threshold if statistic_threshold is not None else True
            if not statistic_passed:
                print(f"{statistic_test} test failed for {key} on statistic")
                print(statistic, p_val)
                total_data = first_dict_distr + second_dict_distr
                total_min, total_max = min(total_data), max(total_data)
                plot_single_distribution(first_dict_distr, key, total_min, total_max)
                plot_single_distribution(second_dict_distr, key, total_min, total_max)
                return False
    return True

def use_freedman_diaconis(data: List, x_min: float, x_max: float, return_as: str = 'num_bins'):
    r"""Implements the freedman diaconis rule.
    
    Arguments:
        data (List): The data to bin
        x_min (float): The minimum x_value on the histogram
        x_max (float): The maximum x_value on the histogram
        return_as (str): Whether to return the number of bins ('num_bins') or 
            the width of each bin ('bin_width')
    
    Returns: 
        num_bins (int) or bin_width (float)
    """
    iqr_val = iqr(data)
    n = len(data)
    bin_width = 2 * iqr_val / (n**(1/3))
    if return_as == 'bin_width':
        return bin_width
    elif return_as == 'num_bins':
        return int((x_max - x_min) / bin_width) + 1

def plot_single_distribution(data: List, title: str, x_min: float, x_max: float) -> None:
    r"""Generates a histogram of the given data.
    
    Arguments:
        data (List): The array of data points to use
        title (str): The title for the histogram
        x_min (float): The minimum value for the histogram
        x_max (float): The maximum value for the histogram
    
    Returns:
        None
    
    Notes: The number of bins is determined using the freedman diaconis rule.
        The minimum and maximum value for the histogram x-axis are fed into the function
        as parameters.
    """
    assert(x_max > x_min)
    num_bins = use_freedman_diaconis(data, x_min, x_max) * 2
    fig, axs = plt.subplots()
    axs.hist(data, bins = num_bins)
    axs.set_xlim(left = x_min, right = x_max)
    axs.set_xlabel("Distance (Angstroms)")
    axs.set_ylabel("Frequency")
    axs.set_title(title)
    axs.xaxis.set_minor_locator(AutoMinorLocator(10))
    axs.yaxis.set_minor_locator(AutoMinorLocator(10))
    plt.show()
    pass

def compare_distributions_distance_metric(distances_1: Dict, distances_2: Dict, tolerance: float, num_bins: int = 100, 
                                          exclude: List[str] = []) -> bool:
    r"""Does a comparison across folds and across inter-atomic pairs based on
        distances
    
    Arguments:
        distances_1: The dictionary of the first distances
        distances_2: The dictionary of the second distances
        tolerance (float): How many non-matching samples are allowed. This is 
            defined as the difference in the number of datapoints between a 
            zero and non-zero bin divided by the total number of datapoints. 
        num_bins (int): The number of bins to use for this approach. Defaults to 100.
        exclude (List[str]): The interatomic pairs to ignore from the analysis. 
            Defaults to []
        
    Returns:
        test_passed (bool): True if the test is passed and the distributions are sufficiently similar
            based on the distance ranges spanned, False otherwise
    
    Notes: In comparing the distances spanned, we first bin both sets of data using the same 
        number of bins. Then, we compare the ith bin for the first distribution with the ith bin
        in the second distribution, and as long as they both have non-zero length (or both have zero length)
        then the test will pass. In the case that one bin has data and the
        other one does not, as long as the difference is not greater than the tolerance,
        which is expressed as the number of disagreement points divided by the 
        total number of points in the non-zero bin's distribution, then the test
        will still pass. Thus, two distributions are considered similar if they
        span the same distances within a tolerance.
    """
    assert(set(distances_1.keys()) == set(distances_2.keys()))
    for key in distances_1:
        if key not in exclude:
            first_dist_distr = distances_1[key]
            second_dist_distr = distances_2[key]
            total_distances = first_dist_distr + second_dist_distr
            total_min, total_max = min(total_distances), max(total_distances)
            bin_width = (total_max - total_min) / num_bins
            first_bins_lst = [0 for i in range(num_bins + 1)] #Numerical stability issue
            second_bins_lst = [0 for j in range(num_bins + 1)]
            
            first_array = np.array(first_dist_distr)
            second_array = np.array(second_dist_distr)
            
            first_indices = ((first_array - total_min) // bin_width).astype(np.int64)
            second_indices = ((second_array - total_min) // bin_width).astype(np.int64)
            assert(min(first_indices) >= 0 and max(first_indices) <= num_bins)
            assert(min(second_indices) >= 0 and max(second_indices) <= num_bins)
            
            
            for index in first_indices:
                first_bins_lst[index] += 1
            for index in second_indices:
                second_bins_lst[index] += 1
            
            total_first = sum(first_bins_lst)
            total_second = sum(second_bins_lst)
            
            #Track the cumulative error for each disagreement
            cumulative_first_error = 0
            total_first_mismatch = 0
            cumulative_second_error = 0
            total_second_mismatch = 0
            #Now scan over both lists
            for i in range(len(first_bins_lst)):
                first_elem = first_bins_lst[i]
                second_elem = second_bins_lst[i]
                if first_elem == 0 and second_elem > 0:
                    if second_elem / total_second > tolerance:
                        print(f"Failed on {key}, second distr")
                        return False
                    else:
                        cumulative_second_error += (second_elem / total_second)
                        total_second_mismatch += 1
                elif first_elem > 0 and second_elem == 0:
                    if first_elem / total_first > tolerance:
                        print(f"Failed on {key}, first distr")
                        return False
                    else: 
                        cumulative_first_error += (first_elem / total_first)
                        total_first_mismatch += 1
            
            if cumulative_first_error > tolerance:
                print(f"Cumulative error for first distribution failed with value {cumulative_first_error}")
                print(key)
                return False
            if cumulative_second_error > tolerance:
                print(f"Cumulative error for second distribution failed with value {cumulative_second_error}")
                print(key)
                return False
    return True

def compare_distributions(fold_molecs: List[List[Dict]],
                          p_threshold: float = 0.05, statistic_threshold: float = None,
                          metric: str = 'KS', exclude: List[str] = []) -> bool:
    r"""Analyzes the distance distributions of the interatomic distances
        in each molecule to see if they are similar across folds
    
    Arguments: 
        fold_molecs (List[List[Dict]]): The molecules for each fold
        p_threshold (float): The minimum value that the p value has to exceed
            for the distributions to be acceptably similar. Default is 0.05 (5%)
        statistic_threshold (float): The maximum value that the Kolmogorov-Smirnov test
            statistic can have for the distributions to be considered reasonably similar.
            This is optional as the p-value alone should be sufficient, and so it defaults to 
            None.
        metric (str): Which test to use for comparing distance distributions. One of
            'KS' and 'MWU', defaults to 'KS'.
        exclude (List[str]): The list of atomic pairs to ignore when performing the analysis. 
            Defaults to [].
    
    Returns:
        test_passed (bool): True if all the same pairwise distances are seen 
            across each fold and the distribution of values for the pairwise distances
            are close enough
    
    Notes: To statistically determine if two distributions are close enough, we 
        employ either the 2 sample Kolmogorov-Smirnov test for comparing distributions or
        the Mann-Whitney U test. 
        
        There are two metrics returned by both the KS and MWU test in a tuple. The
        first value is the statistic and the second value is the p value. Generally,
        comparison of the p_value against the threshold is sufficient to 
        formally disprove the null hypothesis that the two distributions are similar. 
    """
    fold_dictionaries = [analyze_molec_set(molecs) for molecs in fold_molecs]
    key_sets = list(map(lambda x : set(x.keys()), fold_dictionaries))
    #Ensures every set of molecules has the same set of interactions represented
    if key_sets.count(key_sets[0]) != len(key_sets): 
        return False
    
    #Now use the KS test for checking 
    # Given a set of distributions [S1, S2, ..., SN], if S1 is shown to be similar to S2
    # and S2 is shown similar to S3, this does not imply that S1 is similar to S3 since 
    # there is no strong equivalence with KS statistics
    for i in range(len(fold_dictionaries)):
        current_fold_dict = fold_dictionaries[i]
        for j in range(i + 1, len(fold_dictionaries)):
            next_fold_dict = fold_dictionaries[j]
            result = perform_statistic_test(current_fold_dict, next_fold_dict, p_threshold, metric, statistic_threshold, str(i), str(j), exclude)
            if not result:
                return False
    return True

def compare_distribution_distances(fold_molecs: List[List[Dict]], tolerance: float, num_bins: int,
                                  exclude: List[str] = []) -> bool:
    r"""Does a comparison across folds and across inter-atomic pairs based on
        distances
    
    Arguments:
        fold_molecs (List[List[Dict]]): The molecules for each fold
        tolerance (float): How many non-matching samples are allowed. This is 
            defined as the difference in the number of datapoints between a 
            zero and non-zero bin divided by the total number of datapoints. 
        num_bins (int): The number of bins to use for the binning approach of comparing 
            distance distributions
        exclude (List[str]): The interatomic pairs to exclude from the analysis
        
    Returns:
        test_passed (bool): True if the test is passed and the distributions are sufficiently similar
            based on the distance ranges spanned, False otherwise
    
    Notes: In comparing the distances spanned, we first bin both sets of data using the same 
        number of bins. Then, we compare the ith bin for the first distribution with the ith bin
        in the second distribution, and as long as they both have non-zero length (or both have zero length)
        then the test will pass. In the case that one bin has data and the
        other one does not, as long as the difference is not greater than the tolerance,
        which is expressed as the number of disagreement points divided by the 
        total number of points in the non-zero bin's distribution, then the test
        will still pass. Thus, two distributions are considered similar if they
        span the same distances within a tolerance.
    """
    fold_dictionaries = [analyze_molec_set(molecs) for molecs in fold_molecs]
    key_sets = list(map(lambda x : set(x.keys()), fold_dictionaries))
    #Ensures every set of molecules has the same set of interactions represented
    if key_sets.count(key_sets[0]) != len(key_sets): 
        return False
    
    for i in range(len(fold_dictionaries)):
        current_fold_dictionary = fold_dictionaries[i]
        for j in range(i + 1, len(fold_dictionaries)):
            next_fold_dict = fold_dictionaries[j]
            result = compare_distributions_distance_metric(current_fold_dictionary, next_fold_dict, tolerance, num_bins, exclude)
            if not result:
                return False
    return True

def single_fold_precompute(s: Settings, molecs: List[Dict], par_dict: Dict) -> (List[Dict], List[DFTBList]):
    r"""Runs through the precompute process for a single fold rather than a
        pair (train, validate)
    
    Arguments:
        s (Settings): The settings file containing all the hyperparameter settings
        molecs (List[Dict]): The list of molecules to generate the graphs and batches for
        par_dict (Dict): SKF parameter dictionary
        fold_num (int): The fold number to save the information under
    
    Returns:
        feeds (List[Dict]): The list of feed dictionaries
        feed_dftblsts (list[DFTBList]): The list of DFTBList objects
    
    Notes: Different from before, a fold now consists only of a single set of molecules.
        The "train" and "validate" folds are then chosen from among the general set 
        of folds.
    """
    config = {"opers_to_model" : s.opers_to_model}
    feeds, dftb_lsts, batches = graph_generation(molecs, config, s.allowed_Zs, par_dict, s.num_per_batch)
    
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
            
    all_models, model_variables, loss_tracker, all_losses, model_range_dict = model_loss_initialization(feeds, [],
                                                                               s.allowed_Zs, losses, ref_ener_start = s.reference_energy_starting_point)
    feed_generation(feeds, batches, all_losses, all_models, model_variables, model_range_dict, par_dict, s.spline_mode, s.spline_deg, s.debug, False, 
                    s.num_knots, s.buffer, s.joined_cutoff, s.cutoff_dictionary, s.off_diag_opers, s.include_inflect)
    
    print(f"inflect mods: {[mod for mod in model_variables if mod != 'Eref' and mod.oper == 'S' and 'inflect' in mod.orb]}")
    print(f"s_mods: {[mod for mod in model_variables if mod != 'Eref' and mod.oper == 'S']}")
    print(f"len of s_mods: {len([mod for mod in model_variables if mod != 'Eref' and mod.oper == 'S'])}")
    print(f"len of s_mods in all_models: {len([mod for mod in all_models if mod != 'Eref' and mod.oper == 'S'])}")
    print("losses")
    print(losses)
    
    return feeds, dftb_lsts

def save_feed_h5(s: Settings, feeds: List[Dict], dftb_lsts: List[DFTBList], 
                 molecs: List[Dict], dest: str, duplicate_data: bool = False) -> None:
    r"""Saves some feeds and dftb_lsts to the appropriate location (dest) in h5 file format
    
    Arguments:
        s (Settings): The settings object containing all the hyperparameters
        feeds (List[Dict]): The list of feed dictionaries to save
        dftb_lsts (List[DFTBList]): The list of DFTBList objects to save
        molecs (List[Dict]): The list of molecule dictionaries to save
        dest (str): Relative path to the folder to save all these things in
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
    reference_data_name = os.path.join(dest, 'reference_data.p')
    
    #Save the per_molec and per_batch info
    per_molec_h5handler.save_all_molec_feeds_h5(feeds, molec_h5_filename)
    per_batch_h5handler.save_multiple_batches_h5(feeds, batch_h5_filename)
    
    #Save the DFTBlists
    with open(dftb_lst_filename, 'wb') as handle:
        pickle.dump(dftb_lsts, handle)
    
    #Save the reference data
    with open(reference_data_name, 'wb') as handle:
        pickle.dump(feeds, handle)
    
    if duplicate_data:
        with open(molec_filename, 'wb') as handle:
            pickle.dump(molecs, handle)
    
    print("All data saved successfully")

def compute_graphs_from_folds(s: Settings, top_level_molec_path: str, copy_molecs: bool) -> None:
    r"""Computes and saves the feed dictionaries for all the molecules in each fold
    
    Arguments:
        s (Settings): The settings object containing all the hyperparameter values
        top_level_molec_path (str): The relative path to the directory containing 
            the molecules of each fold
        copy_molecs (bool): Whether or not to duplicate the raw molecule pickle files
            in the directories with the saved h5 files (mostly for debugging purposes)
    
    Returns:
        None
        
    Notes: This function does two main things for each set of molecules found
        within the folder top_level_molec_path:
        1) Generate the feed dictionaries for each set of molecules
        2) Saves the feed dictionaries in h5 format
    Nothing is returned from this function
    """
    all_files = os.listdir(top_level_molec_path)
    pattern = r"Fold[0-9]+_molecs.p"
    fold_file_names = list(filter(lambda x : re.match(pattern, x), all_files))
    
    #Load the parameter dictionary specified in settings
    par_dict_path = s.par_dict_name
    if par_dict_path == 'auorg_1_1':
        from auorg_1_1 import ParDict
        par_dict = ParDict()
    else:
        module = importlib.import_module(par_dict_path)
        par_dict = module.ParDict()
    
    #Now cycle through each fold and do the precompute on it
    for name in fold_file_names:
        total_path = os.path.join(top_level_molec_path, name)
        fold_num = name.split('_')[0][-1]
        with open(total_path, 'rb') as handle:
            molecs = pickle.load(handle)
            random.shuffle(molecs)
            if s.train_ener_per_heavy: #Only perform the energy correction if training per heavy atom
                for elem in molecs:
                    energy_correction(elem)
            feeds, dftb_lsts = single_fold_precompute(s, molecs, par_dict)
            destination = os.path.join(top_level_molec_path, f"Fold{fold_num}")
            save_feed_h5(s, feeds, dftb_lsts, molecs, dest = destination, duplicate_data = copy_molecs)
            print(f"Data successfully saved for {name} molecules")
            
    print(f"All data successfully saved for molecules in {top_level_molec_path}")
    
def load_single_fold(s: Settings, top_level_fold_path: str, fold_num: int):
    r"""Loads a single fold using the new fold format based on heavy atoms
    
    Arguments:
        s (Settings): Settings object containing values for hyperparameters
        top_level_fold_path (str): The relative path to the directory containing all the folds
        fold_num (int): The fold number to load
    
    Returns:
        feeds (List[Dict]): The input feeds for the DFTB layer
        dftb_lsts (List[DFTBList]): The DFTBList objects to use for the training
    """
    total_fold_path = os.path.join(top_level_fold_path, f"Fold{fold_num}")
    print(f"Loading from {total_fold_path}")
    batch_info_name = os.path.join(total_fold_path, 'batches.h5')
    molec_info_name = os.path.join(total_fold_path, 'molecs.h5')
    dftb_lst_name = os.path.join(total_fold_path, 'dftblsts.p')
    reference_data_name = os.path.join(total_fold_path, 'reference_data.p')
    
    feeds = total_feed_combinator.create_all_feeds(batch_info_name, molec_info_name, s.ragged_dipole)
    dftb_lsts = pickle.load(open(dftb_lst_name, 'rb'))
    
    if s.run_check:
        print("Running safety check")
        compare_feeds(reference_data_name, feeds)
    
    return feeds, dftb_lsts

def load_combined_fold(s: Settings, top_level_fold_path: str, fold_num: int, fold_mapping: Dict):
    r"""Generates the training and validation feeds through combining individual folds
    
    Arguments:
        s (Settings): The Settings object containing all the hyperparameters
        top_level_fold_path (str): The relative path to the directory containing the 
            individual folds.
        fold_num (int): The fold number used to index into fold_mapping
        fold_mapping (Dict): The dictionary mapping fold_nums to the numbers for
            the training data and the validation data, stored as 2D list with the 
            first list containing training fold numbers and the second list containing
            the validation fold numbers
    
    Returns:
        training_feeds (List[Dict]): List of training feed dictionaries
        validation_feeds (List[Dict]): List of validation feed dictionaries
        training_dftblsts (List[DFTBList]): List of DFTBList objects for training
        validation_dftblsts (List[DFTBList]): List of DFTBList objects for validation
    
    Notes: Here, the fold_num just indicates which split we're doing. The fold_mapping
        maps the split number to the numbers of the individual segments of data that
        needs to be combined for training and validation.
        
    """
    current_train_folds, current_valid_folds = fold_mapping[fold_num]
    #Now we need to load the data for each fold number. Load the training folds first
    training_feeds, training_dftblsts = list(), list()
    validation_feeds, validation_dftblsts = list(), list()
    
    #Get the training information
    for num in current_train_folds:
        feeds, dftblsts = load_single_fold(s, top_level_fold_path, num)
        training_feeds.extend(feeds)
        training_dftblsts.extend(dftblsts)
    
    #Get the validation information
    for num in current_valid_folds:
        feeds, dftblsts = load_single_fold(s, top_level_fold_path, num)
        validation_feeds.extend(feeds)
        validation_dftblsts.extend(dftblsts)
    
    return training_feeds, validation_feeds, training_dftblsts, validation_dftblsts

if __name__ == "__main__":
    
    pass
    
    ## Testing saving features
    # settings_json = "settings_default.json"
    # generate_save_folds(settings_json)
    
    # with open(settings_json, 'r') as read_file:
    #     settings = Settings(json.load(read_file))
    
    # #Testing loading data for the folds
    # top_fold_path = "test_fold"
    # training_feeds, validation_feeds, training_dftblsts, validation_dftblsts = loading_fold(settings, top_fold_path, 0)
    
    ## Testing plotting features for the histograms
    # test_molec_path = os.path.join("test_fold", "Fold0", "train_total_molecs.p")
    # with open(test_molec_path, 'rb') as handle:
    #     data = pickle.load(handle)
    # test_molec = data[0]
    # print(test_molec)
    # result_dict = get_dist_dict(test_molec)
    # print(result_dict)
    # total_distances = analyze_molec_set(data)
    # # print(total_distances)
    # dest = "test_distr"
    # plot_distributions(total_distances, dest)
    
    # #Run through all the subdirectories in test_fold
    # top_level_path = 'test_fold'
    # analyze_all_folds(top_level_path)
    
    ## Testing features for generating nheavy separated folds
    # allowed_Zs = [1,6,7,8]
    # heavy_atoms = [1,2,3,4,5,6,7,8]
    # max_config = 160
    # target = {'Etot' : 'cc',
    #        'dipole' : 'wb97x_dz.dipole',
    #        'charges' : 'wb97x_dz.cm5_charges'}
    # data_path = os.path.join("data", "ANI-1ccx_clean_fullentry.h5")
    # exclude = ['O3', 'N2O1', 'H1N1O3', 'H2']
    # lower_limit = 5
    # num_folds = 6
    # num_folds_lower = 3
    # tolerance = 0.05
    # num_bins = 200
    # local_fold_molecs = "fold_molecs"
    
    # i = 0
    # while True:
    #     print(f"Testing fold generation {i}")
    #     folds = generate_folds(allowed_Zs, heavy_atoms, max_config, target, data_path, exclude, 
    #                            lower_limit, num_folds, num_folds_lower)
    
    #     new_exclude = []
    #     result = compare_distribution_distances(folds, tolerance, num_bins, exclude = new_exclude)
    #     if result:
    #         print("Folds are similar enough")
    #         if not os.path.isdir(local_fold_molecs):
    #             os.mkdir(local_fold_molecs)
    #         for j in range(len(folds)):
    #             curr_fold_path = os.path.join(local_fold_molecs, f"Fold{j}_molecs.p")
    #             with open(curr_fold_path, 'wb') as handle:
    #                 pickle.dump(folds[j], handle)
    #         with open(os.path.join(local_fold_molecs, "settings.txt"), "w+") as handle:
    #             handle.write("allowed Zs: " + str(allowed_Zs) + "\n")
    #             handle.write("heavy atoms: " + str(heavy_atoms) + "\n")
    #             handle.write("Maximum configurations: " + str(max_config) + "\n")
    #             handle.write("target: " + str(target) + "\n")
    #             handle.write("data path: " + str(data_path) + "\n")
    #             handle.write("exclude: " + str(exclude) + "\n")
    #             handle.write("lower limit: " + str(lower_limit) + "\n")
    #             handle.write("num folds: " + str(num_folds) + "\n")
    #             handle.write("num folds lower: " + str(num_folds_lower) + "\n")
    #             handle.write("tolerance: " + str(tolerance) + "\n")
    #             handle.write("num bins: " + str(num_bins) + "\n")
    #         break
    #     else:
    #         print("Folds are not similar enough")
    #         i += 1
    
    ## Testing pre-computes and saves for smaller sets of molecules from nheavy based folds
    test_fold = 'fold_molecs_test'
    if not os.path.isdir(test_fold):
        os.mkdir(test_fold)
        
    pattern = r"Fold[0-9]+_molecs.p"
    valid_names = list(filter(lambda x : re.match(pattern, x), os.listdir('fold_molecs')))
    
    #First copy over a small set of molecules into the test folder
    for name in valid_names:
        full_name = os.path.join("fold_molecs", name)
        molecs = pickle.load(open(full_name, 'rb'))
        new_molecs = molecs[:200]
        dest_name = os.path.join(test_fold, name)
        with open(dest_name, 'wb') as handle:
            pickle.dump(new_molecs, handle)
            
    #Now do the precompute using the settings
    with open("settings_default.json", "r") as handle:
        settings = json.load(handle)
    
    settings_obj = Settings(settings)
    compute_graphs_from_folds(settings_obj, "fold_molecs_test", copy_molecs = True)
    
    ## Testing loading for single folds after saving
    test_fold = 'fold_molecs_test'
    assert(os.path.isdir(test_fold))
    pattern = r"Fold[0-9]+"
    valid_names = list(filter(lambda x : re.match(pattern, x), os.listdir('fold_molecs')))
    
    with open("settings_default.json", "r") as handle:
        settings = json.load(handle)
    
    settings_obj = Settings(settings)
    
    for i in range(len(valid_names)):
        _, _ = load_single_fold(settings_obj, top_level_fold_path = test_fold, fold_num = i)
        
    
    print("Hello")
    
    
    
    
    
    
    
    
    pass
    
    
    
        
        
    
    
    
    
    
        

