# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 23:58:42 2021

@author: fhu14

Methods for plotting SKF files for a directory of SKFs

We are generally more interested in the functional form of the functions 
than the values attained. For this reason, the x-axis is plotted with
more granular information than the y-axis. 

This code is pretty messy, might want to consider a redesign/rewrite of
the software.
"""
#%% Imports, definitions
import pickle, os
from DFTBrepulsive import SKFSet
from MasterConstants import ANGSTROM2BOHR, Model
import matplotlib.pyplot as plt
from typing import Dict, List
from Spline import get_dftb_vals
import numpy as np
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from .util import get_dist_distribution
import numpy as np

nums = { #Switch this out with MasterConstants later
        'H' : 1,
        'C' : 6,
        'N' : 7,
        'O' : 8
        }


#%% Code behind

def read_skf_set(skf_dir: str) -> SKFSet:
    r"""Reads in a set of SKF files from the given directory
    
    Arguments:
        skf_dir (str): The directory containing the skf files
    
    Returns:
        skfset (SKFSet): The SKFSet object created from the files contained in
            skf_dir
    
    Notes: The returned skfset has a dictionary interface for accessing the 
        skf H and S grids of different element pairs.
    """
    skfset = SKFSet.from_dir(skf_dir)
    return skfset

def plot_skf_int(elems: tuple, op: str, int_name: str, skf_set: SKFSet, axs, label: str = None,
                 mode: str = 'scatter') -> None:
    r"""Plots a single SKF integral
    
    Arguments:
        elems (tuple): The elements indicating which SKF is being used
        op (str): The operator, one of "H" or "S"
        int_name (str): The integral name (e.g. 'Hdd0')
        skf_set (SKFSet): The SKFSet object currently being used to write
        axs: The axes object being drawn on
        label (str): The label to use when writing this series of data. For example,
            lable could be 'Auorg' if the integral data comes from auorg or 'MIO' if
            it comes from MIO
        mode (str): scatter or plot. Defaults to scatter.
    """
    curr_skf = skf_set[elems]
    int_table = getattr(curr_skf, op)
    curr_data = int_table[int_name].to_numpy()
    rgrid = curr_skf.intGrid() / ANGSTROM2BOHR
    if mode == 'scatter':
        axs.scatter(rgrid, curr_data, label = label)
    elif mode == 'plot':
        axs.plot(rgrid, curr_data, label = label)

def empty_int(int_series) -> bool:
    r"""Tests if an integral is zero or not
    
    Arguments:
        int_series (Series): A pandas series that represents the data for a 
            given integral
    
    Returns:
        Whether the integral is empty or not. True for empty false for not empty.
    
    Notes: An integral is empty if its maximum and minimum value are both
        0
    """
    return int_series.max() == int_series.min() == 0

def plot_single_skf_set(skf_set: SKFSet, dest: str, mode: str,
                        x_major: int = 1, x_minor: float = 0.1) -> None:
    r"""Plots all the integrals of a single skf set
    
    Arguments:
        skf_set (SKFSet): The set of SKF files that need to be plotted
        dest (str): The destination to save the plots to. If the value of 
            dest is set to None, then the figures are not saved. Useful for just 
            debugging.
        mode (str): plotting mode, either 'scatter' or 'plot'
        x_major (int): The argument used for major multiple locator (increments
             for major tick marks)
        x_minor (float): the argument used for minor multiple locator (increments
             for minor tick marks)
    
    Returns:
        None
    """
    all_ops = ['H', 'S'] #Hamiltonian and overlap operators
    for elem_pair in skf_set.keys():
        for op in all_ops:
            for int_name in getattr(skf_set[elem_pair], op).keys():
                if (not empty_int(getattr(skf_set[elem_pair], op)[int_name]) ):
                    fig, axs = plt.subplots()
                    title = f"{elem_pair}, {op}, {int_name}"
                    axs.set_title(title)
                    axs.set_xlabel("Angstroms")
                    if op == 'H':
                        axs.set_ylabel("Hartrees")
                    elif op == 'S':
                        axs.set_ylabel("A.U.")
                    plot_skf_int(elem_pair, op, int_name, skf_set, axs, mode = mode)
                    #In plotting these graphs, the distance along the x-axis is
                    #   going to be important for figuring out spline
                    #   differences. y-axis vlaues
                    axs.yaxis.set_minor_locator(AutoMinorLocator())
                    axs.xaxis.set_major_locator(MultipleLocator(x_major))
                    axs.xaxis.set_minor_locator(MultipleLocator(x_minor))
                    if (dest is not None):
                        fig.savefig(os.path.join(dest, f"{title}_skf.png"))
                    plt.show()

def plot_overlay_skf_sets(set_1: SKFSet, set_2: SKFSet,
                          set_1_label: str, set_2_label: str, dest: str, mode: str,
                          x_major: int = 1, x_minor: float = 0.1) -> None:
    r"""Plots all integrals of two SKF sets and overlays them.
    
    Arguments:
        set_1 (SKFSet): First SKFSet object
        set_2 (SKFSet): Second SKFSet object
        set_1_label (str): Label for data plotted for set 1
        set_2_label (str): label for data plotted for set 2
        dest (str): Where to save the figures. If dest is set to None,
            the figures are not saved (useful for debugging).
        mode (str): plotting mode, either 'scatter' or 'plot'
        x_major (int): The argument used for major multiple locator (increments
             for major tick marks)
        x_minor (float): the argument used for minor multiple locator (increments
             for minor tick marks)
    
    Returns:
        None
    """
    all_ops = ['H', 'S']
    if len(set_1.keys()) < len(set_2.keys()):
        assert( set(set_1.keys()).issubset(set(set_2.keys())) )
    elif len(set_1.keys()) > len(set_2.keys()):
        assert( set(set_2.keys()).issubset(set(set_1.keys())) )
    else:
        assert( set(set_1.keys()) == set(set_2.keys()) )
    #Assume for now that skf_set_1 has fewer element files than skf_set_2
    for elem_pair in set_1.keys(): #Unsafe, only works for SKF sets with fewer files than the reference set
        for op in all_ops:
            for int_name in getattr(set_1[elem_pair], op).keys():
                series1 = getattr(set_1[elem_pair], op)[int_name]
                series2 = getattr(set_2[elem_pair], op)[int_name]
                if (not empty_int(series1)) and (not empty_int(series2)):
                    fig, axs = plt.subplots()
                    title = f"{elem_pair}, {op}, {int_name}"
                    axs.set_title(title)
                    axs.set_xlabel("Angstroms")
                    if op == 'H':
                        axs.set_ylabel("Hartrees")
                    elif op == 'S':
                        axs.set_ylabel("A.U.")
                    plot_skf_int(elem_pair, op, int_name, set_1, axs, label = set_1_label, mode = mode)
                    plot_skf_int(elem_pair, op, int_name, set_2, axs, label = set_2_label, mode = mode)
                    axs.legend()
                    #In plotting these graphs, the distance along the x-axis is
                    #   going to be important for figuring out spline
                    #   differences. y-axis vlaues
                    axs.yaxis.set_minor_locator(AutoMinorLocator())
                    axs.xaxis.set_major_locator(MultipleLocator(x_major))
                    axs.xaxis.set_minor_locator(MultipleLocator(x_minor))
                    if (dest is not None):
                        fig.savefig(os.path.join(dest, f"{title}_skf.png"))
                    plt.show()
                    
def correct_elem_key(key: str) -> tuple:
    elem_1, elem_2 = key.split("-")
    elem_1, elem_2 = nums[elem_1], nums[elem_2]
    return (int(elem_1), int(elem_2))
                    
def plot_repulsive_overlay(pardict_1: Dict, pardict_2: Dict, 
                           data_1_label: str, data_2_label: str, dest: str,
                           x_major: int = 1, x_minor: float = 0.1) -> None:
    r"""Extracts and plots the repulsive blocks of a set of SKFs in an overlay format
    
    Arguments:
        pardict_1 (Dict): The dictionary with SKFInfo for first SKF set
        pardict_2 (Dict): The dictionary with SKFInfo for the second SKF set
        data_1_label (str): The label for the plots of the first data
        data_2_label (str): The label for the plots of the second set of data
        dest (str): Where to save the plots. If dest is set to None,
            the figures are not saved (useful for debugging).
        x_major (int): The argument used for major multiple locator (increments
             for major tick marks)
        x_minor (float): the argument used for minor multiple locator (increments
             for minor tick marks)
    
    Returns:
        None
    
    TODO: Include exclusion for reverse element pairs to prevent counting of 
        the reflexive
    """
    #Hardcode r_grid
    if len(pardict_1.keys()) < len(pardict_2.keys()):
        assert( set(pardict_1.keys()).issubset(pardict_2.keys()) )
    elif len(pardict_1.keys()) > len(pardict_2.keys()):
        assert( set(pardict_2.keys()).issubset(pardict_1.keys()) )
    else:
        assert( set(pardict_1.keys()) == set(pardict_2.keys()) )
    #Assume for now that pardict_1 has fewer element pairs than pardict_2
    for elems in pardict_1:
        rs = np.linspace(0, 10, 500)
        mod_spec = Model('R', correct_elem_key(elems), 'ss')
        vals_1 = get_dftb_vals(mod_spec, pardict_1, rs)
        vals_2 = get_dftb_vals(mod_spec, pardict_2, rs)
        fig, axs = plt.subplots()
        title = f"{elems}, R"
        axs.set_title(title)
        axs.set_xlabel("Angstroms")
        axs.set_ylabel("Hartrees")
        axs.plot(rs, vals_1, label = data_1_label)
        axs.plot(rs, vals_2, label = data_2_label)
        axs.legend()
        #In plotting these graphs, the distance along the x-axis is
        #   going to be important for figuring out spline
        #   differences. y-axis vlaues
        axs.yaxis.set_minor_locator(AutoMinorLocator())
        axs.xaxis.set_major_locator(MultipleLocator(x_major))
        axs.xaxis.set_minor_locator(MultipleLocator(x_minor))
        if (dest is not None):
            fig.savefig(os.path.join(dest, f"{title}.png"))
        plt.show()

def plot_skf_dist_overlay(skf_set: SKFSet, dest: str, mode: str, dset: List[Dict],
                          x_major: int = 1, x_minor: float = 0.1) -> None:
    r"""Plots an SKF set with the distance distribution overlayed. 
    
    Arguments:
        skf_set (SKFSet): The SKF set to be plotted
        dest (str): Where to save plots. If the dest is set as None, plots will
            not be saved
        mode (str): The mode to be used for plotting ('scatter' or 'plot')
        dset (List[Dict]): The dataset whose distance distribution will be 
            plotted underneath. Fed in as a list of molecule dictionaries
        x_major (int): The argument used for major multiple locator (increments
             for major tick marks)
        x_minor (float): the argument used for minor multiple locator (increments
             for minor tick marks)
    
    Returns:
        None
    
    Notes: The distance distribution histogram will be based on the incrementation
        defined by x_major and x_minor. The bin width will be set to x_minor, and 
        bin spacing is taken from this question:
            
        https://stackoverflow.com/questions/6986986/bin-size-in-matplotlib-histogram
        
        For overlaying plots with different axes, we use twinx(), as described
        here: https://matplotlib.org/stable/gallery/subplots_axes_and_figures/two_scales.html#sphx-glr-gallery-subplots-axes-and-figures-two-scales-py
    """
    d_distr = get_dist_distribution(dset)
    
    all_ops = ['H', 'S'] #Hamiltonian and overlap operators
    for elem_pair in skf_set.keys():
        for op in all_ops:
            for int_name in getattr(skf_set[elem_pair], op).keys():
                if (not empty_int(getattr(skf_set[elem_pair], op)[int_name]) ):
                    fig, axs = plt.subplots()
                    title = f"{elem_pair}, {op}, {int_name}"
                    axs.set_title(title)
                    axs.set_xlabel("Angstroms")
                    if op == 'H':
                        axs.set_ylabel("Hartrees")
                    elif op == 'S':
                        axs.set_ylabel("A.U.")
                        
                    #Now plot out the histogram of distances
                    elem_rev = (elem_pair[-1], elem_pair[0])
                    curr_dist_data = d_distr[elem_pair] if elem_pair in d_distr else d_distr[elem_rev]
                    bins = np.arange( min(curr_dist_data), max(curr_dist_data) + x_minor, x_minor)
                    print( min(curr_dist_data), max(curr_dist_data) )
                    axs2 = axs.twinx()
                    axs2.hist(curr_dist_data, bins = bins, color = 'red', alpha = 0.3)
                    axs2.set_ylabel("Frequency")
                    
                    #Plot the spline second
                    plot_skf_int(elem_pair, op, int_name, skf_set, axs, mode = mode)
                    #In plotting these graphs, the distance along the x-axis is
                    #   going to be important for figuring out spline
                    #   differences. y-axis vlaues
                    axs.yaxis.set_minor_locator(AutoMinorLocator())
                    axs.xaxis.set_major_locator(MultipleLocator(x_major))
                    axs.xaxis.set_minor_locator(MultipleLocator(x_minor))
                    if (dest is not None):
                        fig.savefig(os.path.join(dest, f"{title}_skf.png"))
                    plt.show()
    
    
    
    
    
    
        
        
        

    


