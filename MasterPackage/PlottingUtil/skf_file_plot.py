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
from typing import Dict, List, Union
from Spline import get_dftb_vals
import numpy as np
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from .util import get_dist_distribution
import numpy as np
from .pardict_gen import generate_pardict
from functools import reduce

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
        
    TODO: Fix safety issue in determining which set to iterate over
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

def infer_elements(elems: List[tuple]) -> List[int]:
    r"""Given a list of element tuples, infers all the unique elements
        within the list elems.
    
    Arguments:
        elems (List[tuple]): The pairwise list of elements that are 
            involved in the SKF files.
        
    Returns:
        elem_nums (List[int]): The list of atomic numbers corresponding to 
            each of those unique elements.
    
    Example:
        elems = [(1,1), (1,6), (6,7), (7,7)]
        res = infer_elements(elems)
        res == [1, 6, 7]
    """
    l_iter = map(lambda x : list(x), elems)
    f_iter = reduce(lambda x, y : x + y, l_iter)
    elem_nums = sorted(set(f_iter))
    return elem_nums

def extract_orb(integral_name: str) -> str:
    r"""Given an integral label, extracts the involved orbitals from the 
        interaction.
        
    Arguments: 
        integral_name (str): The name of the integral
    
    Returns:
        orb (str): The orbital type
    
    Example:
        name = "Spp0"
        orb = extract_elem(name)
        orb == "pp_sigma"
    """
    assert(len(integral_name) == 4)
    op, orb, orb_num = integral_name[0], integral_name[1:-1], int(integral_name[-1])
    ind_shell = ['pp']
    if orb in ind_shell:
        orb = orb + "_sigma" if orb_num == 0 else orb + "_pi"
    return orb

def skf_interpolation_plot(skf_dir: str, mode: str, dest: str = None) -> None:
    r"""Reads in a set of SKF files and plots an overlay of the SKF
        integral table and the form of the interpolated spline generated
        by dftbpy
        
    Arguments:
        skf_dir (str): The path to the directory containing the SKF files
        mode (str): The mode to use for plotting. One of 'scatter' or 'plot'
        dest (str): The destination to save the plots to. Defaults to None,
            in which case the plots are not saved.
    
    Returns:
        None
    
    Notes: This function is written for debugging. The goal is to see the 
        discrepancies between the spline in the SKF (as represented as a series of values
        on an integral table) and the spline that is interpolated from that SKF 
        table. The hypothesis is that there is more of a discrepancy when splines are
        read in than when written out. 
        
        This makes sense because interpolation involves a smooth function,
        so if the values on the integral table encode a kinked functional 
        form, the interpolation error on reading in would lead to discrepancies.
        
        This problem does not involve the repulsive splines and only involves
        the quantum electronic Hamiltonian. 
    
    Example:
        skf_dir = "zero_epoch_run"
        mode = "plot"
        dest = None
        skf_interpolation_plot(skf_dir, mode, dest)
        #Splines should overlay exactly for zero_epoch_run
    """
    skfset = SKFSet.from_dir(skf_dir)
    elem_pairs = list(skfset.keys())
    elem_nums = infer_elements(elem_pairs)
    
    par_dict = generate_pardict(skf_dir, elem_nums)
    assert(len(par_dict.keys()) == len(skfset.keys()))
    
    #Now to actually plot everything together.
    all_ops = ['H', 'S']
    for elem_pair in skfset.keys():
        curr_skf = skfset[elem_pair]
        for op in all_ops:
            for int_name in getattr(curr_skf, op).keys():
                if not (empty_int(getattr(curr_skf, op)[int_name])):
                    fig, axs = plt.subplots()
                    title = f"{elem_pair}, {op}, {int_name}"
                    axs.set_title(title)
                    axs.set_xlabel("Angstroms")
                    if op == 'S':
                        axs.set_ylabel("A.U.")
                    elif op == 'H':
                        axs.set_ylabel("Hartrees")
                    plot_skf_int(elem_pair, op, int_name, skfset, axs, label = "skf_table",
                                 mode = mode)
                    #Now to plot the values from the parameter dictionary
                    rgrid = (curr_skf.intGrid() / ANGSTROM2BOHR)  #rgrid in angstroms
                    orb = extract_orb(int_name)
                    model = Model(op, elem_pair, orb)
                    ygrid = get_dftb_vals(model, par_dict, rgrid)
                    if mode == 'scatter':
                        axs.scatter(rgrid, ygrid, label = "interpolated spline")
                    elif mode == 'plot':
                        axs.plot(rgrid, ygrid, label = "interpolated spline")
                    axs.legend()
                    if (dest is not None):
                        fig.savefig(os.path.join(os.getcwd(), dest, f"{title}_skf.png"))
                    plt.show()
                    #Also do a comparison of the ygrid values to the values 
                    #   read in from the SKF files:
                    numpy_skf = getattr(curr_skf, op)[int_name].to_numpy()
                    MAE = np.mean(np.abs(numpy_skf - ygrid))
                    print(f"MAE difference in values is {MAE}")
                    #Of course this comparison is going to be nearly exact!
                    #   the SKFInfo backend uses an InterpolatedUnivariateSpline
                    #   whose points are the grid points of the SKF integrl table.
                    #   if rgrid was not the same as the set of SKF grid points,
                    #   I'm sure a greater difference would emerge. 

def compare_differences(skset_1_name: str, skset_2_name: str, dest: str,
                        mode: str, x_major: Union[int, float] = 1,
                        x_minor: Union[int, float] = 0.1, units: str = "Ha") -> None:
    r"""Plots the differences of two spline plots with the ability to specify
        the units of the plot.
    
    Arguments:
        skset_1_name (str): First SKFSet directory path
        skset_2_name (str): Second SKFSet directory path
        dest (str): The destination to save the plots; plots are not saved
            if the dest is set to None
        mode (str): The mode for the plots, one of 'plot' and 'scatter'
        units (str): The units to use for plotting. Defaults to 'Ha', which
            is the atomic unit of energy. Can also specify 'kcal' for Kcal/mol.
    
    Returns:
        None
    
    Notes: The energy unit only matters for splines representing matrix elements of the 
        Hamiltonian operator; otherwise, the overlap operators have arbitrary units.
        
        The conversion from Ha to kcal/mol is multiplcation by 627.5.
        
        For each skf set, the corresponding SKFs (e.g. C-C.skf in set 1 and C-C.skf in set 2)
        must have been evaluated over the same grid. 
    """
    # raise NotImplementedError()
    all_ops = ['H', 'S']
    skset_1 = read_skf_set(skset_1_name)
    skset_2 = read_skf_set(skset_2_name)
    min_set = None
    max_set = None
    if len(skset_1.keys()) < len(skset_2.keys()):
        assert( set(skset_1.keys()).issubset(set(skset_2.keys())) )
        min_set = skset_1
        max_set = skset_2
    elif len(skset_1.keys()) > len(skset_2.keys()):
        assert( set(skset_2.keys()).issubset(set(skset_1.keys())) )
        min_set = skset_2
        max_set = skset_1
    else:
        assert( set(skset_1.keys()) == set(skset_2.keys()) )
        min_set = skset_1
        max_set = skset_2
    
    for elem_pair in min_set.keys():
        for op in all_ops:
            for int_name in getattr(min_set[elem_pair], op).keys():
                series1 = getattr(min_set[elem_pair], op)[int_name]
                series2 = getattr(max_set[elem_pair], op)[int_name]
                if (not empty_int(series1)) and (not empty_int(series2)):
                    skf1 = min_set[elem_pair]
                    skf2 = max_set[elem_pair]
                    rgrid1 = skf1.intGrid() / ANGSTROM2BOHR
                    rgrid2 = skf2.intGrid() / ANGSTROM2BOHR
                    assert(all(rgrid1 == rgrid2)) #rgrids must be the same.
                    int_table1 = getattr(skf1, op)
                    int_table2 = getattr(skf2, op)
                    data_1 = int_table1[int_name].to_numpy()
                    data_2 = int_table2[int_name].to_numpy()
                    diff = np.abs(data_2 - data_1)
                    if (op == 'H') and (units == 'kcal'):
                        diff = diff * 627.5
                    #Now to plot the differences over the rgrid
                    fig, axs = plt.subplots()
                    title = f"{elem_pair}, {op}, {int_name}"
                    axs.set_title(title)
                    axs.set_xlabel("Angstroms")
                    if op == 'H':
                        if units == 'kcal':
                            axs.set_ylabel("Absolute difference in kcal/mol")
                        elif units == 'Ha':
                            axs.set_ylabel("Absolute difference in Ha")
                    elif op == 'S':
                        axs.set_ylabel("Absolute difference")
                    if mode == 'scatter':
                        axs.scatter(rgrid1, diff)
                    else:
                        axs.plot(rgrid1, diff)
                    axs.yaxis.set_minor_locator(AutoMinorLocator())
                    axs.xaxis.set_major_locator(MultipleLocator(x_major))
                    axs.xaxis.set_minor_locator(MultipleLocator(x_minor))
                    if (dest is not None):
                        fig.savefig(os.path.join(dest, f"{title}_skf.png"))
                    plt.show()
    
    if (dest is not None):
        note_path = os.path.join(dest, "info.txt")
        with open(note_path, 'w+') as handle:
            msg = f"Absolute differences plotted for electronic splines between sets {skset_1_name} and {skset_2_name}"
            handle.write(msg + "\n")
            handle.close()
        
        print("Plots generated and saved")
    
    print("Plots generated")
    pass

