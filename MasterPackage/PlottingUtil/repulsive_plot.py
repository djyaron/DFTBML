# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 15:17:04 2022

@author: fhu14
"""
#%% Imports, definitions
import pickle, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.interpolate import CubicSpline

#%% Code behind

def plot_repulsive_model(model_filename: str, dest: str = None,
                         x_label: str = "Angstrom", y_label: str = "Ha", spl_ngrid = 500,
                         mode: str = "scatter") -> None:
    r"""Plots the repulsive model but under new scheme, i.e. using the DFTBrepulsive model
    
    Arguments:
        model_filename (str): The name/path to the pickle file containing the saved models
        dest (str): The location to save the plots to. Defaults to None
        x_label (str): The x label for the plots. Defaults to Angstrom
        y_label (str): The y label for the plots. Defaults to Ha
        spl_ngrid (int): The number of grid points to evaluate for plotting. Defaults to a 
            dense grid, i.e. 500 points
        mode (str): The mode used for plotting the repulsive potential. One of 'plot' or 'scatter',
            defaults to 'scatter'
    
    Returns:
        None
    
    Raises: 
        ValueError if the plotting mode is not recognized in ["plot", "scatter"]
    
    Notes: This is just isolated logic from what's done in skfwriter for the repulsive 
        model
    """
    models = pickle.load(open(model_filename, 'rb'))
    repulsive_model = models['rep']
    grid_dict = {elems : np.linspace(v[0], v[1], spl_ngrid) for elems, v in repulsive_model.mod.opts.cutoff.items()}
    xy_data = repulsive_model.mod.create_xydata(grid_dict, expand = True)
    # assert(set(xy_data.keys()) == set(models.keys()))
    for elem_pair in xy_data:
        x, y = xy_data[elem_pair][:,0], xy_data[elem_pair][:,1]
        fig, ax = plt.subplots()
        if mode == 'scatter':
            ax.scatter(x, y)
        elif mode == 'plot':
            ax.plot(x, y)
        else:
            raise ValueError("Plot mode not recognized")
        ax.set_title(f"{elem_pair} repulsive model")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        if dest is not None:
            if not os.path.exists(dest):
                os.mkdir(dest)
            fig.savefig(os.path.join(dest, f"{elem_pair}_rep.png"))
        plt.show()
    print("All the repulsive potentials plotted")

def compare_repulsive_value_magnitudes(mod_file1: str, mod_file2: str,
                                       method: str = "mean", spl_ngrid: int = 500) -> None:
    r"""Computes the difference in the magnitudes of the values between
        two repulsive models for each pairwise interaction
    
    Arguments:
        mod_file1 (str): The ame of the first model pickle file
        mod_file2 (str): The name of the second model pickle file
        method (str): The method used to quantify the difference, one of 
            'mean' to compare the mean value and 'absolute' to just compare the 
            absolute deviation between values
        spl_ngrid (int): Number of points to use for the spline grid
    
    Returns:
        None
    
    Raises:
        ValueError if the method passed to the function is not recognized in 
            'mean' or 'absolute'
    
    Notes: Because the spacing of values are not consistent between two different repulsive models, 
        'absolute' is implemented such that the values from the closest distances are 
        compared. This method should be used for cases where the intervals don't overlap,
        i.e. the set of distances for one repulsive model is a subset of the distances
        for the other repulsive model for all pairwise interactions. It's safest to compare
        against the full cutoff in most cases.
    """
    if method not in ['mean', 'absolute']:
        raise ValueError("Unrecognized method parameter")
    mods1 = pickle.load(open(mod_file1, 'rb'))
    mods2 = pickle.load(open(mod_file2, 'rb'))
    repulsive_mod_1 = mods1['rep']
    repulsive_mod_2 = mods2['rep']
    gd1 = {elems : np.linspace(v[0], v[1], spl_ngrid) for elems, v in repulsive_mod_1.mod.opts.cutoff.items()}
    gd2 = {elems : np.linspace(v[0], v[1], spl_ngrid) for elems, v in repulsive_mod_2.mod.opts.cutoff.items()}
    xy_data1 = repulsive_mod_1.mod.create_xydata(gd1, expand = True)
    xy_data2 = repulsive_mod_2.mod.create_xydata(gd2, expand = True)
    assert(set(xy_data1.keys()) == set(xy_data2.keys()))
    for elem_pair in xy_data1:
        d1, d2 = xy_data1[elem_pair], xy_data2[elem_pair]
        if (d1[0,0] > d2[0,0]) and (d1[-1,0] < d2[-1,0]):
            print("Case 1, data1 is subset of data2")
            r_range = (d1[0,0], d1[-1,0])
        elif (d2[0,0] > d1[0,0]) and (d2[-1,0] < d1[-1,0]):
            print("Case 2, data2 is subset of data1")
            r_range = (d2[0,0], d2[-1,0])
        else:
            if (d1[0,0] < d2[0,0]) and (d2[0,0] < d1[-1,0]) and (d1[-1,0] < d2[-1,0]):
                print("Case 3, 2 overlaps 1, 2 is greater")
                r_range = (d2[0,0], d1[-1,0])
            elif (d2[0,0] < d1[0,0]) and (d1[0,0] < d2[-1,0]) and (d2[-1,0] < d1[-1,0]):
                print("Case 4, 1 overlaps 2, 1 is greater")
                r_range = (d1[0,0], d2[-1,0])
        rows1 = np.where((d1[:, 0] >= r_range[0]) & (d1[:,0] <= r_range[1]))[0]
        rows2 = np.where((d2[:, 0] >= r_range[0]) & (d2[:,0] <= r_range[1]))[0]
        assert(len(rows1) > 0 and len(rows2) > 0)
        #Extract just the y values
        matched_data1 = d1[rows1][:,1]
        matched_data2 = d2[rows2][:,1]
        print(f"Comparing {mod_file1} repulsive and {mod_file2} repulsive for {elem_pair}")
        print(f"Comparing over distance range {r_range}")
        if method == 'mean':
            print("Method is mean")
            #We take advantage of the fact that all repulsive values should be
            #   positive
            print(np.abs(np.mean(matched_data1) - np.mean(matched_data2)))
        elif method == 'absolute':
            print("Method is absolute")
            #Calculate differences for beginning, end, and range
            beginning_diff = np.abs(matched_data1[0] - matched_data2[0])
            ending_diff = np.abs(matched_data1[-1] - matched_data2[-1])
            range_diff = np.abs( (matched_data1[-1] - matched_data1[0]) - (matched_data2[-1] - matched_data2[0]) )
            print(beginning_diff, ending_diff, range_diff)
    print(f"Analysis completed for {mod_file1} repulsive and {mod_file2} repulsive")
    
def test_rep_spline_interp(mod_filename: str, spl_ngrid: int = 500,
                           error_metric: str = "MAE") -> None:
    r"""This method is designed to test the fidelity of interpolations over 
        the repulsive data for different cutoff conditions
    
    Arguments:
        mod_filename (str): The name of the filename with losslessly saved models contained
        spl_ngrid (int): The number of grid points to evaluate the repulslive 
            spline on and to perform the interpolation. Defaults to a dense grid
            of 500 points
        error_metric (str): The error metric for quantifying the difference 
            between the original values and the interpolated values. 
            One of "MAE" or "RMSE", defaults to "MAE"
        
    Returns:
        ValueError if error_metric not in ["MAE", "RMSE"]
    
    Notes: In writing out the SKF files, an interpolation is performed using the 
        scipy.interpolate library. The interpolation is performed in the skf.py 
        module within the DFTBrepulsive sub-package.
        
        One consideration here is that we want to see the power of the interpolation
        between breakpoints and at breakpoints, so we want to re-evaluate on a 
        dense grid. We will assess both error in reproducing values at breakpoints
        and the error in interpolating values between breakpoints
    """
    if error_metric not in ["MAE", "RMSE"]:
        raise ValueError("Unrecognized error metric!")
    mods = pickle.load(open(mod_filename, 'rb'))
    repulsive_model = mods['rep']
    
    grid_dict = {elems : np.linspace(v[0], v[1], spl_ngrid) for elems, v in repulsive_model.mod.opts.cutoff.items()}
    grid_dict_dense = {elems : np.linspace(v[0], v[1], 2 * spl_ngrid) for elems, v in repulsive_model.mod.opts.cutoff.items()}
   
    xy_data = repulsive_model.mod.create_xydata(grid_dict, expand = True)
    xy_data_dense = repulsive_model.mod.create_xydata(grid_dict_dense, expand = True)
    
    for elem_pair in xy_data:
        spl_grid, spl_vals = xy_data[elem_pair][:,0], xy_data[elem_pair][:,1]
        fig, axs = plt.subplots()
        axs.scatter(spl_grid, spl_vals) #Plot it quickly as a sanity check
        plt.show()
        #Perform the interpolation
        spline = CubicSpline(spl_grid, spl_vals)
        #Re-evaluate the spline on the grid to try and reproduce the breakpoints spl_grid
        knot_interp_vals = spline(spl_grid)
        #Interp between breakpoints
        super_dense_grid, super_dense_vals = xy_data_dense[elem_pair][:,0], xy_data_dense[elem_pair][:,1]
        dense_interp_vals = spline(super_dense_grid)
        #Calculate the errors for both of these interpolations
        if error_metric == "MAE":
            knot_interp_err = np.mean(np.abs(knot_interp_vals - spl_vals))
            super_dense_interp_vals = np.mean(np.abs(dense_interp_vals - super_dense_vals))
        elif error_metric == "RMSE":
            knot_interp_err = np.sqrt(np.mean(np.square(knot_interp_vals - spl_vals)))
            super_dense_interp_vals = np.sqrt(np.mean(np.square(dense_interp_vals - super_dense_vals)))
        print(f"For {elem_pair}, knot interpolation error, repulsive potential recovered with {error_metric} {knot_interp_err} ")
        print(f"For {elem_pair}, dense interpolation error, repulsive potential recovered with {error_metric} {super_dense_interp_vals}")
    
    print("Completed interpolation test for all element pairs")