# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 01:18:34 2020

@author: Frank
"""
'''
Visualizing the loss
'''
import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)


data_file = "losses.p"
hartree_to_kcal = 627.0


if __name__ == "__main__":
    train_losses, validation_losses = None, None
    config_info = None
    with open(data_file, "rb") as handle:
        train_losses = pickle.load(handle)
        validation_losses = pickle.load(handle)
    xs_for_train = [i for i in range(len(train_losses))]
    xs_for_valid = [i for i in range(len(validation_losses))]
    
    fig, axs = plt.subplots()
    axs.plot(xs_for_train, train_losses, label = 'Training loss')
    axs.plot(xs_for_valid, validation_losses, label = 'Validation loss')
    axs.set_title("Train and validation losses, Elems = 1, 6, 7, 8, Num heavy = 1 - 8")
    axs.set_xlabel("Epochs")
    axs.set_ylabel("Average Epoch Loss (kcal / mol)")
    axs.yaxis.set_minor_locator(AutoMinorLocator())
    axs.legend()
    plt.show()
    
    
    
    