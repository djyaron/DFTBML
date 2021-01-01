# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 16:00:42 2021

@author: Frank

Module for determining the optimal range of the spline models. This is
accomplished by reading in the entire skf file and finding the inflection point,
as the usable region will extend from the inflection point onward

The inflection points are determined through the reference skf files
"""
import numpy as np
from auorg_1_1 import ParDict
from modelspline import get_dftb_vals
from dftb import ANGSTROM2BOHR
from typing import Union, List, Optional, Dict, Any, Literal
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from batch import Model

start = 0.02 / ANGSTROM2BOHR # angstroms
end = 10 # angstroms
rgrid = np.linspace(start, end, 1000)

def plot_skf_values(feeds: List[Dict], par_dict: Dict) -> None:
    r"""Plot the dftbvalues for all the models used in the feed to visually determine the 
    best range
    
    Arguments:
        feeds (List[Dict]): List of all feeds, training and validation
        par_dict (Dict): Dictionary of skf parameters

    Returns: None
    """
    all_models = []
    for feed in feeds:
        all_models += feed['models']
    all_models = list(set(all_models))
    
    for model in all_models:
        if len(model.Zs) == 2:
            y_vals = get_dftb_vals(model, par_dict, rgrid)
            fig, ax = plt.subplots()
            ax.scatter(rgrid, y_vals)
            ax.set_title(model)
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            plt.show()
            

def model_range_dict():
    r"""Hardcoded ranges for the spline models. All ranges are represented as 
    a tuple of (x_low, x_high) in angstroms
    """
    model_range_dict = dict()
    model_range_dict[Model('H', (7, 7), 'sp')] = (0.8, end)
    model_range_dict[Model('R', (7, 6), 'ss')] = (start, end)
    model_range_dict[Model('G', (8, 7), 'sp')] = (0.5, end)
    model_range_dict[Model('G', (6, 6), 'ps')] = (0.5, end)
    model_range_dict[Model('H', (7, 6), 'sp')] = (0.8, end)

        
        
        
        
        
        
    




