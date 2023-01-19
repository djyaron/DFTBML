# -*- coding: utf-8 -*-
"""
Created on Sun May 29 16:17:56 2022

@author: fhu14
"""
'''
Some diagnostic utilities for visualizing and understanding statistics about 
different datasets. This needs some work and is the bare bones right now
'''
#%% Imports, definitions
import matplotlib.pyplot as plt
from typing import List, Tuple
import pickle, os


#%% Code behind

def obtain_formula_distribution(mol_set: str) -> Tuple[List[str], List[int]]:
    r"""Obtains the distribution of the occurrences of different 
        empirical formulas in a dataset
    
    Arguments:
        mol_set (str): The path to the molecule to visualize the distributions for
    
    Returns:
        names (List[str]): The list of formula strings
        counts (List[int]): The list of frequencies of occurrence for each 
            corresponding empirical formula
    """
    mols = pickle.load(open(os.path.join(os.getcwd(), mol_set), 'rb'))
    name_counts = {}
    for mol in mols:
        curr_name = mol['name']
        if curr_name not in name_counts:
            name_counts[curr_name] = 1
        else:
            name_counts[curr_name] += 1
    names, counts = [], []
    for name, count in name_counts.items():
        names.append(name)
        counts.append(count)
    
    return names, counts

def plot_distribution(counts: List[int], dest: str = None, name: str = None) -> None:
    r"""Plots the distribution as a histogram
    
    Arguments:
        counts (List[int]): The list of frequencies of occurrence for each 
            corresponding empirical formula
        dest (str): The location to save the figure. Defaults to None, in which 
            case it will not be saved
        name (str): If dest is not None, this is the name to use for
            saving the distribution. Also used to title the plots
        
    Returns:
        None
    """
    fig, axs = plt.subplots()
    axs.hist(counts)
    axs.set_xlabel("Frequency of occurrence per formula")
    axs.set_ylabel("Frequency")
    if name is not None:
        axs.set_title(f"Frequency distribution for {name}")
    else:
        axs.set_title("Frequency distribution")
    plt.show()
    if dest is not None:
        fig.savefig(os.path.join(os.getcwd(), dest, f"{name}.png"))
    
def plot_distribution_per_dataset(dset_path: str, dest: str = None) -> None:
    r"""Plots out the distributions for the training, validation, and
        test sets for a given dataset
    
    Arguments:
        dset_path (str): The path to the current dataset
        dest (str): The location to save it. Defaults to None, so the 
            plot is not saved
    
    Returns:
        None
    """
    training_mol_path = os.path.join(os.getcwd(), dset_path, 'Fold0_molecs.p')
    validation_mol_path = os.path.join(os.getcwd(), dset_path, 'Fold1_molecs.p')
    testing_mol_path = os.path.join(os.getcwd(), dset_path, 'test_set.p')
    
    training_distr = obtain_formula_distribution(training_mol_path)
    validation_distr = obtain_formula_distribution(validation_mol_path)
    testing_distr = obtain_formula_distribution(testing_mol_path)
    
    dset_name = os.path.split(dset_path)[-1]
    plot_distribution(training_distr[-1], dest, dset_name + " training set")
    plot_distribution(validation_distr[-1], dest, dset_name + " validation set")
    plot_distribution(testing_distr[-1], dest, dset_name + " testing set")


#%% Main block
