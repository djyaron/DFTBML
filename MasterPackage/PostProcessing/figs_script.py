# -*- coding: utf-8 -*-
"""
Created on Thu May 19 15:28:00 2022

@author: fhu14
"""

"""
Module for generating the figures for the paper; there will be a separate
function for each figure and the paths will be hard-coded so that 
there is no ambiguity about what is being referenced

There will be a lot of dependence on the PlottingUtil package
"""

#%% Imports, definitions
import os, pickle
from PlottingUtil import visualize_loss_tracker


#%% Code behind

'''
The figures that we need are as follows:
    1) General loss/learning curves
    2) Plots of splines overlayed with each other
    3) Plots of splines overlayed with the distance distribution histogram
    4) Plots of splines on their own (representative)
'''