# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 15:14:45 2021

@author: fhu14

File used to analyze the differences between the repulsive splines returned by
DFTBrepulsive and the repulsive splines contained in MIO
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from typing import List, Dict, Union
from batch import Model
from modelspline import get_dftb_vals
from auorg_1_1 import ParDict
import os.path

atom_dict = {
    1 : 'H',
    6 : 'C',
    7 : 'N',
    8 : 'O',
    79 : 'Au'
    }

def load_rep_results(result_filename: str) -> dict:
    r"""Loads the repulsive results from the pickle file
    
    Arguments:
        result_filename (str): The name of the repulsive results
    
    Returns:
        results (dict): The results
    """
    return pickle.load(open(result_filename, 'rb'))

def unpack_rep_results(results: dict) -> (dict, list):
    r"""Takes the results dictionary and unpacks it to data dictionary mapping
        element pairs to values
        
    Arguments:
        results (dict): Dictionary of raw results contained in the pickle file
    
    Returns:
        cleaned_result (dict): Maps the element pair to the corresponding 
            xydata
        elem_pairs (list): The list of element pairs that we are interested in
            
    Notes: We are interested in third order splines, so we are using the sparse
        data
    """
    curr_results = results['all_xydata']['sparse_xydata'][0][0][0]
    return curr_results, list(curr_results.keys())

def generate_repulsive_model(elem_pair: tuple) -> Model:
    r"""Generates the repulsive model for a given element pair
    
    Arguments: 
        elem_pair (tuple): Tuple of the atomic numbers
    
    Returns:
        rep_mod (Model): The repulsive model
    """
    return Model('R', elem_pair, 'ss')

def obtain_results_skf(elem_pairs: list, par_dict: dict, results: dict) -> dict:
    r"""Pulls repulsive information from the reference skf files contained
        in the skf_direc
    
    Arguments:
        elem_pairs (list): List of tuples containing the element pairs
        par_dict (dict): The parameter dictionary generated from reference
            skf files
        results (dict): The unpacked results from unpack_rep_results
    
    Returns:
        ref_results (dict): The reference results for the corresponding element
            pairs in elem_pairs
    """
    ref_results = dict()
    for pair in elem_pairs:
        rep_mod = generate_repulsive_model(pair)
        rgrid = results[pair][0]
        y_vals = get_dftb_vals(rep_mod, par_dict, rgrid)
        ref_results[pair] = (rgrid, y_vals)
    return ref_results

def plot_results(ref_results: dict, calc_results: dict, mode: str = 'overlay') -> None:
    r"""Plots the repulsive splines obtained from the reference files and
        the calculated results
        
    Arguments:
        ref_results (dict): The repulsive spline results from the reference
            skf files
        calc_results (dict): The computed repulsive spline results
        mode (str): The plotting mode, one of 'overlay' or 'single', where
            'overlay' plots the calculated and reference together
        
    Returns:
        None
    """
    assert(ref_results.keys() == calc_results.keys())
    for pair in ref_results:
        if mode == 'overlay':
            fig, axs = plt.subplots()
            axs.plot(*calc_results[pair], 'bo', label = 'calculated')
            axs.plot(*ref_results[pair], 'r+', label = 'reference')
            axs.legend()
            axs.set_xlabel('Angstroms')
            axs.set_ylabel('Hartrees')
            axs.set_title(str(pair))
            plt.show()
        elif mode == 'single':
            for result_dict in [calc_results, ref_results]:
                fig, axs = plt.subplots()
                axs.plot(*result_dict[pair])
                axs.set_xlabel('Angstroms')
                axs.set_ylabel('Hartrees')
                axs.set_title(str(pair))
                plt.show()
    
def analyze(result_filename: str, par_dict: dict, mode: str = 'overlay') -> None:
    r"""Graphical analysis between calculated repulsive splines and 
        reference MIO repulsive splines
    
    Arguments:
        result_filename (str): The name of the repulsive results
        par_dict (dict): The parameter dictionary generated from reference
            skf files
        mode (str): The plotting mode, one of 'overlay' or 'single', where
            'overlay' plots the calculated and reference together
    
    Returns:
        None
    """
    rep_results, keys = unpack_rep_results(load_rep_results(result_filename))
    ref_results = obtain_results_skf(keys, par_dict, rep_results)
    plot_results(ref_results, rep_results, mode)

if __name__ == "__main__":
    analyze(os.path.join('ANI_cv', 'ANI_cv', 'cv (cvxopt, nknots=50, deg=3, rmax=short~short, ptype=convex)', 'cv_rmax.pkl'), ParDict())



