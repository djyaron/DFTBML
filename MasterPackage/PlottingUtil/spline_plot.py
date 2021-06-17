# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 15:32:47 2021

@author: fhu14
"""
#%% Imports, definitions
import numpy as np
import matplotlib.pyplot as plt
import os
from MasterConstants import Model
from InputLayer import Input_layer_pairwise_linear_joined, Input_layer_pairwise_linear
import pickle

#%% Code behind

def plot_spline(spline_model, dest_dir: str, max_ider: int = 0, ngrid: int = 500, 
                same_plot: bool = False, mode: str = 'scatter') -> None:
    r"""Takes an instance of input_pairwise_linear and plots using present variable vector
    
    Arguments:
        spline_model (input_pairwise_linear OR input_pairwise_linear_joined): 
            Instance of input_pairwise_linear for plotting
        dest_dir (str): The destination directory to save the plots
        max_ider (int): The maximum derivative to plot the spline up to.
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
    dgrids_consts = [spline_model.pairwise_linear_model.linear_model(rgrid, i) for i in range(max_ider + 1)]
    model_variables = spline_model.get_variables().detach().cpu().numpy()
    print(f"Number of coefficients: {len(model_variables)}")
    if hasattr(spline_model, "joined"):
        fixed_coefs = spline_model.get_fixed().detach().cpu().numpy()
        model_variables = np.concatenate((model_variables, fixed_coefs))
    model = spline_model.model
    if not same_plot:
        for i in range(max_ider + 1):
            fig, ax = plt.subplots()
            y_vals = np.dot(dgrids_consts[i][0], model_variables) + dgrids_consts[i][1]
            if mode == 'scatter':
                ax.scatter(rgrid, y_vals)
            elif mode == 'plot':
                ax.plot(rgrid, y_vals)
            ax.set_title(f"{model} deriv {i}")
            ax.set_xlabel("Angstroms")
            ax.set_ylabel("Hartrees")
            fig.savefig(os.path.join(dest_dir, f"{model}_d_{i}.png"))
            plt.show()
    else:
        fig, axs = plt.subplots(max_ider + 1)
        for i in range(max_ider + 1):
            y_vals = np.dot(dgrids_consts[i][0], model_variables) + dgrids_consts[i][1]
            if mode == "scatter":
                axs[i].scatter(rgrid, y_vals)
                axs[i].set_title(f"{model} deriv {i}")
        plt.xlabel("Angstroms")
        plt.ylabel("Hartrees")
        fig.tight_layout()
        fig.savefig(os.path.join(dest_dir, f"{model}.png"))
        plt.show()
    
    print(f"Plots finished for {model}")

def plot_all_splines(model_file: str, dest_dir: str, max_ider: int = 0, ngrid: int = 500, 
                same_plot: bool = False, mode: str = 'scatter') -> None:
    r"""Wrapper method for plot_spline that takes in a pickle file of saved models.
    
    Arguments:
        model_file (str): The path to the pickle file containing the 
            saved models.
        dest_dir (str): The destination directory to save the plots
        max_ider (int): The maximum derivative to plot the spline up to.
        ngrid (int): Number of grid points for evaluation. Defaults to 500
        same_plot (bool): Plots the 0th, 1st, and 2nd derivative on the same plot. 
            Defaults to True
        mode (str): Either 'scatter' or 'plot'. Defaults to 'scatter'
    
    Returns:
        None
    """
    with open(model_file, 'rb') as handle:
        all_models = pickle.load(handle)
    
    if (not os.path.isdir(dest_dir)):
        os.mkdir(dest_dir)
    
    for model_spec in all_models:
        if (isinstance(model_spec, Model)) and (len(model_spec.Zs) == 2)\
            and (isinstance(all_models[model_spec], Input_layer_pairwise_linear_joined) or isinstance(all_models[model_spec], Input_layer_pairwise_linear)):
            plot_spline(all_models[model_spec], dest_dir, max_ider, ngrid,
                        same_plot, mode)