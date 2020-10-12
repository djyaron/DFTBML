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


data_file = "losses.p"
num_outer = 100
num_inner = 1000
num_inner_printed = 25

if __name__ == "__main__":
    train_losses, first_epoch_losses = None, None
    config_info = None
    with open(data_file, "rb") as handle:
        train_losses = pickle.load(handle)
        first_epoch_losses = pickle.load(handle)
        try:
            config_info = pickle.load(handle)
        except:
            config_info = [num_outer, num_inner, num_inner_printed]
    num_out, num_in, num_print = config_info
    total_points = num_out * num_in
    xs_for_first = [x * num_in for x in range(num_out)]
    xs_for_total = [i * num_print for i in range(len(train_losses))]
    
    fig, axs = plt.subplots()
    axs.plot(xs_for_total[:1000], train_losses[:1000], label = 'Training loss')
    # axs.plot(xs_for_first[:25], first_epoch_losses[:25], label = 'Reference energy loss')
    axs.set_title("Training losses update charges standalone dftb")
    axs.set_xlabel("Epochs")
    axs.set_ylabel("Total Epoch Loss (Energy)")
    axs.legend()
    plt.show()
    
    
    
    