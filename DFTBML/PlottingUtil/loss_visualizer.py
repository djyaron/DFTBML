# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 15:13:25 2021

@author: fhu14

Module for visualizing information stored in the loss tracker framework
"""
#%% Imports, definitions
import pickle 
import numpy as np
Array = np.ndarray
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from typing import Union

loss_conversion_units = {
    "Etot" : 10,
    "dipole" : 100,
    'charges' : 1
    }

loss_yaxs_labels = {
    "Etot" : r"Energy (kcal mol$^{-1}$ atom$^{-1}$)",
    "dipole" : r"Dipole ($e\AA$)",
    "charges" : r"Atomic charge ($e$)"
    }

#%% Code behind
def exceeds_tick_limit(loss: Array, increment: Union[int, float], limit: int = 1000) -> bool:
    r"""Checks that using the proposed minor axis incrementation, the 
        number of tick marks does not exceed a certain limit.
    
    Arguments:
        loss (Array): The array of loss values
        increment (Uniont[int, float]): The incrementation for the minor axis
        limit (int): The limit for an acceptable number of tick marks. Defaults
            to 1000
    
    Returns:
        bool: Whether the proposed number of ticks exceeds the tick limit
    
    Notes:
        It is recommended to do this calculation based on the 
            minor axis incrementation.
    """
    max_val, min_val = max(loss), min(loss)
    diff = max_val - min_val
    return diff / increment > limit

def visualize_loss_tracker(lt_filename: str, dest_dir: str, mode: str = 'plot', 
                           scale: str = 'normal', x_axis_mode: str = 'epochs',
                           n_batch: int = None, y_major: Union[int, float] = 1,
                           y_minor: Union[int, float] = 0.1) -> None:
    r"""Reads in a loss tracker from a pickle file and generates graphs of
        the losses
        
    Arguments:
        lt_filename (str): The filename/path of the loss tracker
        dest_dir (str): The path to the destination directory where the 
            plots are saved. If set to None, then the plots are not saved.
        mode (str): The mode to use when plotting out the losses. One of 'plot'
            or 'scatter' for plotting a line plot or plotting a scatter plot, 
            respectively. Defaults to 'plot'.
        scale (str): The scaling to use for the y-axis of the loss. One of
            'normal' or 'log', where 'normal' does not transform the values in
            any way but 'log' transforms the values by taking the base 10 logarithm
            of the loss. Defaults to 'normal'
        x_axis_mode (str): The units of the x-axis. One of 'epochs' or 'grad_desc'
        n_batch (int): The number of batches. Defaults to None, but should be a 
            positive numerical value if x_axis_mode is set to 'grad_desc'
        y_major (Union[int, float]): The incrementation for the major tick marks
            on the y-axis. Defaults to 1.
        y_minor (Union[int, float]): The incrementation for the minor tick marks
            on the y-axis. Defaults to 0.1.
    
    Returns:
        None
    
    Notes: This method visualizes the losss trackers into learning curves for the 
        different physical targets that are learned by DFTBML. This includes
        total energy, dipoles, and charges. The number of batches is equal to the 
        number of gradient descent steps since one gradient descent step is taken
        for every batch seen. 
    """
    if (dest_dir is not None) and (not os.path.isdir(dest_dir)):
        os.mkdir(dest_dir)
    
    with open(lt_filename, 'rb') as handle:
        loss_tracker = pickle.load(handle)
    
    for loss in loss_tracker:
        fig, axs = plt.subplots()
        unit = loss_conversion_units[loss] if loss in loss_conversion_units else 1
        validation_loss = np.array(loss_tracker[loss][0]) / unit
        training_loss = np.array(loss_tracker[loss][1]) / unit
        if scale == 'log':
            validation_loss = np.log(validation_loss)
            training_loss = np.log(training_loss)
        if x_axis_mode == 'epochs':
            x_vals = [i + 1 for i in range(len(validation_loss))]
        elif x_axis_mode == 'grad_desc':
            assert(n_batch is not None)
            x_vals = [(i + 1) * n_batch for i in range(len(validation_loss))]
        axs.plot(x_vals, validation_loss, label = "Validation loss")
        axs.plot(x_vals, training_loss, label = "Training loss")
        axs.set_title(f"{loss} loss")
        if scale == 'log':
            axs.set_ylabel('log loss')
        elif scale == 'normal':
            axs.set_ylabel(loss_yaxs_labels[loss] if loss in loss_yaxs_labels else "Loss")
        if x_axis_mode == 'epochs':
            axs.set_xlabel("Epochs")
        elif x_axis_mode == 'grad_desc':
            axs.set_xlabel("Number of gradient descent steps")
        if exceeds_tick_limit(validation_loss, y_minor) or\
            exceeds_tick_limit(training_loss, y_minor):
                y_minor_temp = y_minor * 10
                y_major_temp = y_major * 10
                axs.yaxis.set_minor_locator(MultipleLocator(y_minor_temp))
                axs.yaxis.set_major_locator(MultipleLocator(y_major_temp))
        else:
            axs.yaxis.set_minor_locator(AutoMinorLocator())
            # axs.yaxis.set_major_locator(AutoMinorLocator())
        axs.xaxis.set_minor_locator(AutoMinorLocator())
        axs.legend()
        if dest_dir is not None:
            fig.savefig(os.path.join(dest_dir, f"{loss}_loss.png"), dpi = 2000)
        plt.show()
    
    total_val_loss = np.zeros(len(loss_tracker['Etot'][0]))
    total_train_loss = np.zeros(len(loss_tracker['Etot'][1]))
    for loss in loss_tracker:
        total_val_loss += np.array(loss_tracker[loss][0])
        total_train_loss += np.array(loss_tracker[loss][1])
    
    fig, axs = plt.subplots()
    axs.plot(total_val_loss, label = 'Validation loss')
    axs.plot(total_train_loss, label = 'Training loss')
    axs.set_title('Total loss')
    axs.set_xlabel('Epoch')
    axs.set_ylabel('Average Epoch Loss (unitless)')
    axs.yaxis.set_minor_locator(AutoMinorLocator())
    axs.xaxis.set_minor_locator(AutoMinorLocator())
    axs.legend()
    if dest_dir is not None:
        fig.savefig(os.path.join(dest_dir, "Total_loss.png"), dpi = 2000)
    plt.show()
    
    if dest_dir is not None:
        print("All loss graphs saved")


    
        
        
        
        
    
    
    
    