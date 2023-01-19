# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 17:18:18 2021

@author: fhu14

TODO: Add more functionalities that you might want for manipulating datasets
"""
#%% Imports, definitions
import os, re, pickle, random, math
from functools import reduce

#%% Code behind

def randomize_existing_set(src_dir: str, dest_dir: str) -> None:
    r"""Takes an existing set of folds in the pickle form
        and randomizes the molecules contained within.
    
    Arguments:
        src_dir (str): The relative path to the source directory to generate
            the new dataset off of the existing folds contained in src_dir
        dest_dir (str): Relative path to destination directory for saving
        
    Returns:
        None
        
    Notes:
        This method takes an existing set of molecules and randomizes the molecules
        across all the pickle files. It then divides the molecules into folds 
        of the original sizes and saves them in pickle format. Precomputation to 
        generate the graphs and feeds will have to be done on PSC since the datasets
        are too big for laptops.
    """
    if (not os.path.isdir(dest_dir)):
        os.mkdir(dest_dir)
    pattern = r"Fold[0-9]+_molecs.p"
    #Should contain the names of all valid molecule pickle files (e.g. Fold0_molecs.p)
    valid_names = list(filter(lambda x : re.match(pattern, x), os.listdir(src_dir)))
    total_molecs = [pickle.load(open(os.path.join(src_dir, filename), 'rb')) for filename in valid_names]
    fold_sizes = [len(x) for x in total_molecs]
    total_molecs = list(reduce(lambda x, y : x + y, total_molecs))
    #Random shuffling of molecules without bias to permutations
    random.shuffle(total_molecs)
    splitting_indices = [0] + [fold_sizes[i] + sum(fold_sizes[:i]) if i > 0 else fold_sizes[i] for i in range(len(fold_sizes))]
    #Split back into the original folds and save the molecules into a pickle
    #   file based on the original fold sizes
    for i in range(len(splitting_indices) - 1):
        curr_molecs = total_molecs[splitting_indices[i] : splitting_indices[i + 1]]
        save_path = os.path.join(dest_dir, f"Fold{i}_molecs.p")
        with open(save_path, 'wb') as handle:
            pickle.dump(curr_molecs, handle)
    
    print("New folds generated")
    
def split_existing_set(src_dir: str, dest_dir: str, prop_train: float, randomize: bool) -> None:
    r"""Takes an existing set and generates a split into two folds.
    
    Arguments: 
        src_dir (str): The source directory
        dest_dir (str): The destination directory
        prop_train (float): What proportion of molecules should be used for
            the training set
        randomize (bool): Whether to randomize the molecules before splitting
            into the training and test set.
    
    Returns:
        None
    
    Notes: This method takes all the N molecules found in src_dir and divides
        them into a training and test set, where prop_train * N molecules
        are used in the training set and (1 - prop_train) * N molecules are
        used in the validation set. They are saved as two separate folds, and 
        Fold 0 is the training fold and Fold 1 is the validation fold.
    """
    if (not os.path.isdir(dest_dir)):
        os.mkdir(dest_dir)
    pattern = r"Fold[0-9]+_molecs.p"
    #Should contain the names of all valid molecule pickle files (e.g. Fold0_molecs.p)
    valid_names = list(filter(lambda x : re.match(pattern, x), os.listdir(src_dir)))
    total_molecs = [pickle.load(open(os.path.join(src_dir, filename), 'rb')) for filename in valid_names]
    if randomize:
        random.shuffle(total_molecs)
    #Now split the total_molecules 
    num_train = math.ceil(prop_train * len(total_molecs))
    training_molecs = total_molecs[: num_train]
    validation_molecs = total_molecs[num_train : ]
    train_filename = "Fold0_molecs.p"
    valid_filename = "Fold1_molecs.p"
    full_train_path = os.path.join(dest_dir, train_filename)
    full_valid_path = os.path.join(dest_dir, valid_filename)
    
    with open(full_train_path, 'wb') as handle:
        pickle.dump(training_molecs, handle)
    
    with open(full_valid_path, 'wb') as handle:
        pickle.dump(validation_molecs, handle)
    
    print("Alternative split saved")
    
    
    