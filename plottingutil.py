# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 13:14:56 2021

@author: Frank

Collection of functions for plotting and analyzing splines
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List, Optional, Dict, Any, Literal
from batch import Model
from modelspline import get_dftb_vals
from dftb import ANGSTROM2BOHR
from matplotlib.ticker import AutoMinorLocator
import os, os.path

Array = np.ndarray

def plot_spline(spline_model, ngrid: int = 500, same_plot: bool = False, mode: str = 'scatter') -> None:
    r"""Takes an instance of input_pairwise_linear and plots using present variable vector
    
    Arguments:
        spline_model (input_pairwise_linear): Instance of input_pairwise_linear for plotting
        ngrid (int): Number of grid points for evaluation. Defaults to 500
        same_plot (bool): Plots the 0th, 1st, and 2nd derivative on the same plot. 
            Defaults to True
        mode (str): Either 'scatter' or 'plot'. Defaults to 'scatter'
    
    Returns:
        None
    
    Notes: Method now works with joined splines. Also, option to plot the zeroeth, first, 
        and second derivative on the same plot
    """
    rlow, rhigh = spline_model.pairwise_linear_model.r_range()
    rgrid = np.linspace(rlow, rhigh, ngrid)
    dgrids_consts = [spline_model.pairwise_linear_model.linear_model(rgrid, 0),
                     spline_model.pairwise_linear_model.linear_model(rgrid, 1),
                     spline_model.pairwise_linear_model.linear_model(rgrid, 2)]
    model_variables = spline_model.get_variables().detach().numpy()
    print(len(model_variables))
    if hasattr(spline_model, "joined"):
        fixed_coefs = spline_model.get_fixed().detach().numpy()
        model_variables = np.concatenate((model_variables, fixed_coefs))
    model = spline_model.model
    if not same_plot:
        for i in range(3):
            fig, ax = plt.subplots()
            y_vals = np.dot(dgrids_consts[i][0], model_variables) + dgrids_consts[i][1]
            if mode == 'scatter':
                ax.scatter(rgrid, y_vals)
            elif mode == 'plot':
                ax.plot(rgrid, y_vals)
            ax.set_title(f"{model} deriv {i}")
            plt.show()
    else:
        fig, (ax0, ax1, ax2) = plt.subplots(3)
        y_vals_0 = np.dot(dgrids_consts[0][0], model_variables) + dgrids_consts[0][1]
        y_vals_1 = np.dot(dgrids_consts[1][0], model_variables) + dgrids_consts[1][1]
        y_vals_2 = np.dot(dgrids_consts[2][0], model_variables) + dgrids_consts[2][1]
        if mode == 'scatter':
            ax0.scatter(rgrid, y_vals_0)
            ax1.scatter(rgrid, y_vals_1)
            ax2.scatter(rgrid, y_vals_2)
        elif mode == 'plot':
            ax0.plot(rgrid, y_vals_0)
            ax1.plot(rgrid, y_vals_1)
            ax2.plot(rgrid, y_vals_2)
        ax0.set_title(f"{model} deriv {0}")
        ax1.set_title(f"{model} deriv {1}")
        ax2.set_title(f"{model} deriv {2}")
        fig.tight_layout()
        plt.show()
            
            
            
def get_x_y_vals (spline_model, ngrid: int) -> (Array, Array, str):
    r"""Obtains the x and y values for plotting the spline
    
    Arguments:
        spline_model (input_pairwise_linear or joined): Model to get values for
        ngrid (int): Number of grid points to use for evaluation
    
    Returns:
        (rgrid, y_vals, title) (Array, Array, str): Data used for plotting the spline
            where rgrid is on x-axis, y_vals on y-axis, with the given title
            
    Notes: Works for joined and non-joined splines.
    """
    rlow, rhigh = spline_model.pairwise_linear_model.r_range()
    rgrid = np.linspace(rlow, rhigh, ngrid)
    dgrids_consts = spline_model.pairwise_linear_model.linear_model(rgrid, 0)
    model_variables = spline_model.get_variables().detach().numpy()
    if hasattr(spline_model, "joined"):
        fixed_coefs = spline_model.get_fixed().detach().numpy()
        model_variables = np.concatenate((model_variables, fixed_coefs))
    model = spline_model.model
    y_vals = np.dot(dgrids_consts[0], model_variables) + dgrids_consts[1]
    oper, Zs, orb = model
    title = f"{oper, Zs, orb}"
    return (rgrid, y_vals, title)

def plot_multi_splines(target_models: List[Model], all_models: Dict, ngrid: int = 500, max_per_plot: int = 4,
                       method: str = 'plot') -> None:
    r"""Plots all the splines whose specs are listed in target_models
    
    Arguments:
        target_models (List[Model]): List of model specs to plot
        all_models (Dict): Dictionary referencing all the models used
        ngrid (int): Number of grid points to use for evaluation. Defaults to 500
        max_per_plot (int): Maximum number of splines for each plot
    
    Returns:
        None
    
    Notes: Works for both joined and non-joined splines.
    """
    total_mods = len(target_models)
    total_figs_needed = total_mods // max_per_plot if total_mods % max_per_plot == 0 else (total_mods // max_per_plot) + 1
    # max per plot should always be a square number
    num_row = num_col = int(max_per_plot**0.5)
    fig_subsections = list()
    for i in range(0, len(target_models), max_per_plot):
        fig_subsections.append(target_models[i : i + max_per_plot])
    assert(len(fig_subsections) == total_figs_needed)
    for i in range(total_figs_needed):
        curr_subsection = fig_subsections[i]
        curr_pos = 0
        fig, axs = plt.subplots(num_row, num_col)
        for row in range(num_row):
            for col in range(num_col):
                if curr_pos != len(curr_subsection):
                    x_vals, y_vals, title = get_x_y_vals(all_models[curr_subsection[curr_pos]], ngrid)
                    if method == 'plot':
                        axs[row, col].plot(x_vals, y_vals)
                    elif method == 'scatter':
                        axs[row, col].scatter(x_vals, y_vals)
                    axs[row, col].set_title(title)
                    curr_pos += 1
        fig.tight_layout()
        fig.savefig(os.path.join(os.getcwd(), "Splines", f"SplineGraph{i}.png"))
        plt.show()

def plot_skf_values(feeds: List[Dict], par_dict: Dict) -> None:
    r"""Plot the dftbvalues for all the models used in the feed to visually determine the 
    best range
    
    Arguments:
        feeds (List[Dict]): List of all feeds, training and validation
        par_dict (Dict): Dictionary of skf parameters

    Returns: None
    
    Notes: The values are drawn from the skf files of the corresponding ParDict
    """
    start = 0.02 / ANGSTROM2BOHR # angstroms
    end = 10 # angstroms
    rgrid = np.linspace(start, end, 1000)
    
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




