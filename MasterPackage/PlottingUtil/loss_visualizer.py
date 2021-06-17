# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 15:13:25 2021

@author: fhu14

Module for visualizing information stored in the loss tracker framework
"""
#%% Imports, definitions
import pickle 
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

#%% Code behind

def visualize_loss_tracker(lt_filename: str, dest_dir: str) -> None:
    r"""Reads in a loss tracker from a pickle file and generates graphs of
        the losses
        
    Arguments:
        lt_filename (str): The filename/path of the loss tracker
        dest_dir (str): The path to the destination directory where the 
            plots are saved.
    
    Returns:
        None
    """
    if (not os.path.isdir(dest_dir)):
        os.mkdir(dest_dir)
    
    with open(lt_filename, 'rb') as handle:
        loss_tracker = pickle.load(handle)
    
    for loss in loss_tracker:
        fig, axs = plt.subplots()
        validation_loss = loss_tracker[loss][0]
        training_loss = loss_tracker[loss][1]
        axs.plot(validation_loss, label = "Validation loss")
        axs.plot(training_loss, label = "Training loss")
        axs.set_title(f"{loss} loss")
        axs.set_ylabel("Average Epoch Loss (unitless)")
        axs.set_xlabel("Epoch")
        axs.yaxis.set_minor_locator(AutoMinorLocator())
        axs.xaxis.set_minor_locator(AutoMinorLocator())
        axs.legend()
        fig.savefig(os.path.join(dest_dir, f"{loss}_loss.png"))
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
    fig.savefid(os.path.join(dest_dir, "Total_loss.png"))
    plt.show()
    
    print("All loss graphs saved")


    
        
        
        
        
    
    
    
    