# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 20:32:36 2021

@author: fhu14
"""

#%% Imports, definitions
from typing import List, Dict
import numpy as np
from MasterConstants import ANGSTROM2BOHR, Model
from Spline import get_dftb_vals
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import os

#%% Code behind

def plot_skf_values(models: List[Model], par_dict: Dict, dest_dir: str) -> None:
    r"""Plot the dftbvalues for all the models used in the feed to visually determine the 
    best range
    
    Arguments:
        models (List[Model]): List of model specs defined as a list of 
            named tuples.
        par_dict (Dict): Dictionary of skf parameters
        dest_dir (str): Where to save the plots generated from the skf files.

    Returns: None
    
    Notes: The values are drawn from the skf files of the corresponding ParDict
    """
    start = 0.02 / ANGSTROM2BOHR # angstroms
    end = 10 # angstroms
    rgrid = np.linspace(start, end, 1000)
    
    for model in models:
        if len(model.Zs) == 2:
            y_vals = get_dftb_vals(model, par_dict, rgrid)
            fig, ax = plt.subplots()
            ax.scatter(rgrid, y_vals)
            ax.set_title(model)
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            fig.savefig(os.path.join(dest_dir, f"{model}_skf.png"))
            plt.show()