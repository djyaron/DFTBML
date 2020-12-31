# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 16:32:51 2020

@author: Frank
"""
'''
New file to hold the different methods for computing the losses for backpropagation

Will include the present losses for Eelec, but will also include methods for
getting different penalties and eventually other properties of interest
'''

import torch
import torch.nn as nn
import numpy as np
from batch import Model
from auorg_1_1 import ParDict
from tfspline import Bcond,spline_linear_model
from modelspline import get_dftb_vals
import matplotlib.pyplot as plt
import os, os.path
from typing import Union, List, Optional, Dict, Any, Literal
Array = np.ndarray

#%% Previous loss methods 
def loss_refactored(output, data_dict, targets):
    '''
    Slightly refactored loss, will be used within the objet oriented handler
    '''
    all_bsizes = list(output[targets[0]].keys())
    loss_criterion = nn.MSELoss()
    target_tensors, computed_tensors = list(), list()
    for bsize in all_bsizes:
        for target in targets:
            computed_tensors.append(output[target][bsize])
            target_tensors.append(output[target][bsize])
    assert(len(target_tensors) == len(computed_tensors))
    for i in range(len(target_tensors)):
        # Can only concatenate tensors, so make scalars into tensors
        if len(target_tensors[i].shape) == 0:
            target_tensors[i] = target_tensors[i].unsqueeze(0)
        if len(computed_tensors[i].shape) == 0:
            computed_tensors[i] = computed_tensors[i].unsqueeze(0)
    total_targets = torch.cat(target_tensors)
    total_computed = torch.cat(computed_tensors)
    return loss_criterion(total_computed, total_targets)

def total_energy_loss (output, data_dict, targets):
    '''
    Calculates the MSE loss for a given minibatch using the torch implementation of 
    MSELoss
    
    targets is not used in the function but is there to stay consistent with interface
    '''
    all_bsizes = list(output['Eelec'].keys())
    loss_criterion = nn.MSELoss() #Compute MSE loss by the pytorch specification
    target_tensors, computed_tensors = list(), list()
    for bsize in all_bsizes:
        computed_result = output['Erep'][bsize] + output['Eelec'][bsize] + output['Eref'][bsize] 
        target_result = data_dict['Etot'][bsize]
        if len(computed_result.shape) == 0:
            computed_result = computed_result.unsqueeze(0)
        if len(target_result.shape) == 0:
            target_result = target_result.unsqueeze(0)
        computed_tensors.append(computed_result)
        target_tensors.append(target_result)
    total_targets = torch.cat(target_tensors)
    total_computed = torch.cat(computed_tensors)
    return loss_criterion(total_computed, total_targets) 

#%% Adaptation of spline penalties
'''
This adaptation for the penalties comes from solver.py
which is a part of the DFTBrepulsive project. The file is included in this repo,
but only use it as a reference

Do a generic class here to get all the different penalties

For this version of the penalties, working with the pairwise linear interface
defined in dftb_layer_splines_3.py

Note: torch.einsum is used here in place of np.einsum because only torch.einsum 
works on tensors that require gradients, and the gradients need to be maintained throughout
forward prop for back prop

TODO: Figure out system for classifying if model should concave up or concave down
    Possible solution: Know this initially and pass in a flag? Probably easier than
    trying to figure it out on the fly...
'''
class ModelPenalty:
    '''
    This class takes in two things:
        input_pairwise_linear: model conforming to the input_pairwise_linear interface
                               defined in dftb_layer_splines_3.py
        n_grid (int): number of points to use when evaluating the second derivative and the penalty. Default = 500
        neg_integral (bool): Indicates whether the spline represented by this model should be concave up (v''(r) > 0) or
                             concave down (v''(r) < 0). If True, then concave down; otherwise, concave up
        penalties (dict): Should map the penalty with the weight that that penalty should have
    '''
    def __init__ (self, input_pairwise_linear, n_grid = 500, neg_integral = False, penalties = None):
        self.input_pairwise_lin = input_pairwise_linear
        self.penalties = penalties
        self.neg_integral = neg_integral
        
        #Compute the x-grid
        r_low, r_high = self.input_pairwise_lin.pairwise_linear_model.r_range()
        self.xgrid = np.linspace(r_low, r_high, n_grid)
        
        #Compute the derivative_grid for 0th, 1st, and 2nd order
        self.dgrid = dict()
        for i in range(3):
            self.dgrid[i] = self.input_pairwise_lin.pairwise_linear_model.linear_model(self.xgrid, i)[0]
        
        #Do a mini penalty check for the weights of each penalty
        self.monotonic_enabled = False
        self.convex_enabled = False
        self.smooth_enabled = False
        
        self.penalty_check()
    
    def penalty_check(self):
        '''
        Checks the penalties and toggle the flags
        '''
        if self.penalties is not None:
            if "monotonic" in self.penalties:
                self.monotonic_enabled = True
            if "convex" in self.penalties:
                self.convex_enabled = True
            if "smooth" in self.penalties:
                self.smooth_enabled = True
    
    # Now some methods for computing the different penalties
    def get_monotonic_penalty(self):
        '''
        Computes the monotonic penalty similar to that done in solver.py
        
        TODO: Come back to this and consider how monotonic penalty changes given spline sign
        '''
        lambda_monotonic = self.penalties['monotonic']
        monotonic_penalty = 0
        c = self.input_pairwise_lin.get_variables()
        deriv = self.dgrid[1]
        deriv = torch.tensor(deriv)
        p_monotonic = torch.einsum('j,ij->i', c, deriv)
        #For a monotonically increasing potential (i.e. concave down integral), the
        # First derivative should be positive, so penalize the negative terms. Otherwise,
        # penalize the positive terms for concave up
        if self.neg_integral:
            p_monotonic [p_monotonic > 0] = 0
        else:
            p_monotonic [p_monotonic < 0] = 0 
        monotonic_penalty = torch.einsum('i,i->', p_monotonic, p_monotonic)
        monotonic_penalty *= lambda_monotonic
        return monotonic_penalty
        
    def get_convex_penalty(self):
        '''
        Computes the convex penalty similar to that done in solver.py
        '''
        lambda_convex = self.penalties['convex']
        convex_penalty = 0
        c = self.input_pairwise_lin.get_variables()
        deriv = self.dgrid[2]
        deriv = torch.tensor(deriv)
        p_convex = torch.einsum('j,ij->i', c, deriv)
        # Case on whether the spline should be concave up or down
        if self.neg_integral:
            p_convex [p_convex < 0] = 0
        else:
            p_convex [p_convex > 0] = 0
        convex_penalty = torch.einsum('i,i->', p_convex, p_convex)
        convex_penalty *= lambda_convex
        return convex_penalty
    
    def get_smooth_penalty(self):
        '''
        Computes the smooth penalty similar to that done in solver.py
        
        Pretty sure this is going ot have to change
        '''
        lambda_smooth = self.penalties['smooth']
        smooth_penalty = 0
        c = self.input_pairwise_lin.get_variables()
        deriv = self.dgrid[2]
        deriv = torch.tensor(deriv)
        p_smooth = torch.einsum('j,ij->i',c,deriv)
        smooth_penalty = torch.einsum('i,i->', p_smooth, p_smooth)
        smooth_penalty *= lambda_smooth
        return smooth_penalty
    
    #Compute the actual loss
    def get_loss(self):
        '''
        Computes the overall loss as a sum of the individual penalties
        
        Nothing fancy like shuttling information back and forth in solver.py
        '''
        loss = 0
        if self.monotonic_enabled:
            loss += self.get_monotonic_penalty()
        if self.smooth_enabled:
            loss += self.get_smooth_penalty()
        if self.convex_enabled:
            loss += self.get_convex_penalty()
        return loss

#%% Total loss method construction
'''
Need to devise a complete loss method that incorporates regularization into the spline
elements

We only want to concentrate on the electronic energy here, the repulsive and 
reference energies will be dealt with separately
'''

atom_dict = {
    1 : 'H',
    6 : 'C',
    7 : 'N',
    8 : 'O'
    }

def apx_equal(a, b, tol = 1E-8):
    return abs(a - b) < 1E-8

def find_concavity_H(model_spec, par_dict, r_max = 9, threshold = 0.75):
    '''
    Parameters
    ----------
    model_spec : Model named tuple, used in batch.py
        The spec representing a given model. This shoudl 
        
    par_dict : parameter dictionary taken from skf files. 
    r_max (int or float): The maximum distance to go to
    threshold (float): The proportion of values that needs to be met for 

    Returns
    -------
    Bool, False: The spline should be concave up (+ -> 0)
          True: The spline should be concave down (- -> 0)
        
    Note: The electron repulsion splines (G) are always concave up, so no
    need to do in-depth concavity analysis for them.
    
    By the same reasoning, the core-core repulsions should also be concave up
    
    TODO: This needs work
    '''
    #Extract the relevant information from the model_spec
    assert(len(model_spec.Zs) == 2)
    assert(model_spec.oper == 'H')
    elem1, elem2 = model_spec.Zs
    elem1_str, elem2_str = None, None
    try:
        elem1_str, elem2_str = atom_dict[elem1], atom_dict[elem2]
    except:
        raise ValueError("Element not found!")
    key = f"{elem1_str}-{elem2_str}"
    orbitals = model_spec.orb
    #Figure out the inner key based on the orbitals 
    par_dict_key = None
    if orbitals == 'ss':
        par_dict_key = 'Hss0'
    elif orbitals == 'sp':
        par_dict_key = 'Hsp0'
    elif orbitals == 'sd':
        par_dict_key = 'Hsd0'
    elif orbitals == 'pp_pi':
        par_dict_key = 'Hpp1'
    elif orbitals == 'pp_sigma':
        par_dict_key = 'Hpp0'
    elif orbitals == 'pd_pi':
        par_dict_key = 'Hpd1'
    elif orbitals == 'pd_sigma':
        par_dict_key = 'Hpd0'
    elif orbitals == 'dd_del':
        par_dict_key = 'Hdd2'
    elif orbitals == 'dd_pi':
        par_dict_key = 'Hdd1'
    elif orbitals == 'dd_sigma':
        par_dict_key = 'Hdd0'
    
    grid, data = par_dict[key].GetSkData(par_dict_key)    
    start_index, end_index = None, None
    for i in range(len(data)):
        if not apx_equal(data[i], 0):
            start_index = i
            break
        
    for i in range(len(grid) - 1, -1, -1):
        if apx_equal(grid[i], r_max):
            end_index = i
            break
    
    data_subset = data[start_index : end_index + 1]
    diff = np.diff(data_subset) 
    
    num_negative = np.sum(diff < 0)
    num_positive = np.sum(diff > 0)
    total = len(diff)
    
    #Really hacky way of checking concave up or down
    if num_negative / total >= threshold: 
        return False
    elif num_positive / total >= threshold:
        return True

def concavity_test(concavity_dict):
    '''
    Test method for the concavity dictionary, at least for the 'H' models
    '''
    H_double_mods_only = [Model(oper='H', Zs=(1, 6), orb='sp'), Model(oper='H', Zs=(1, 7), orb='sp'), 
                 Model(oper='H', Zs=(1, 8), orb='sp'), Model(oper='H', Zs=(6, 1), orb='ss'), 
                 Model(oper='H', Zs=(6, 6), orb='pp_pi'), Model(oper='H', Zs=(6, 6), orb='pp_sigma'), 
                 Model(oper='H', Zs=(6, 6), orb='sp'), Model(oper='H', Zs=(6, 6), orb='ss'), 
                 Model(oper='H', Zs=(6, 7), orb='sp'), Model(oper='H', Zs=(6, 8), orb='sp'), 
                 Model(oper='H', Zs=(7, 1), orb='ss'), Model(oper='H', Zs=(7, 6), orb='pp_pi'), 
                 Model(oper='H', Zs=(7, 6), orb='pp_sigma'), Model(oper='H', Zs=(7, 6), orb='sp'), 
                 Model(oper='H', Zs=(7, 6), orb='ss'), Model(oper='H', Zs=(7, 7), orb='pp_pi'), 
                 Model(oper='H', Zs=(7, 7), orb='pp_sigma'), Model(oper='H', Zs=(7, 7), orb='sp'), 
                 Model(oper='H', Zs=(7, 7), orb='ss'), Model(oper='H', Zs=(7, 8), orb='sp'),
                 Model(oper='H', Zs=(8, 1), orb='ss'), Model(oper='H', Zs=(8, 6), orb='pp_pi'), 
                 Model(oper='H', Zs=(8, 6), orb='pp_sigma'), Model(oper='H', Zs=(8, 6), orb='sp'),
                 Model(oper='H', Zs=(8, 6), orb='ss'), Model(oper='H', Zs=(8, 7), orb='pp_pi'),
                 Model(oper='H', Zs=(8, 7), orb='pp_sigma'), Model(oper='H', Zs=(8, 7), orb='sp'),
                 Model(oper='H', Zs=(8, 7), orb='ss'), Model(oper='H', Zs=(8, 8), orb='pp_pi'), 
                 Model(oper='H', Zs=(8, 8), orb='pp_sigma'), Model(oper='H', Zs=(8, 8), orb='sp'), 
                 Model(oper='H', Zs=(8, 8), orb='ss')]
    result_keys = list(concavity_dict.keys())
    ss_keys = [mod for mod in result_keys if mod.orb == 'ss' and mod in H_double_mods_only]
    sp_keys = [mod for mod in result_keys if mod.orb == 'sp' and mod in H_double_mods_only]
    pp_sig = [mod for mod in result_keys if mod.orb == 'pp_sigma' and mod in H_double_mods_only]
    pp_pi = [mod for mod in result_keys if mod.orb == 'pp_pi' and mod in H_double_mods_only]

    for k in ss_keys:
        assert(concavity_dict[k])
    for k in sp_keys:
        assert(not concavity_dict[k])
    for k in pp_sig:
        assert(not concavity_dict[k])
    for k in pp_pi:
        assert(concavity_dict[k])
    
    print("Tests passed!")
    pass

def generate_concavity_dict(model_spline_dict):
    '''
    Takes in a dictionary containing spline information for all the dftb models and
    returns a dictionary mapping each model to whether it's concave up (False) or concave down (True)
    '''
    concavity_dict = dict()
    for model_spec in model_spline_dict:
        mod_dict = model_spline_dict[model_spec]
        y_vals = np.dot(mod_dict['X'][2], mod_dict['coefs']) + mod_dict['const'][2]
        #Assume also that the mean will take care of it, and that the mean should not be 0
        concavity = np.sign(np.mean(y_vals))
        #Positive mean value means it should be concave up
        if concavity == 1:
            concavity_dict[model_spec] = False
        #Negative mean value means it should be concave down
        elif concavity == -1:
            concavity_dict[model_spec] = True
        elif concavity == 0:
            raise ValueError("Concavity should be non-zero for two-body splines!")
    concavity_test(concavity_dict)
    return concavity_dict

def compute_mod_vals_derivs(all_models, par_dict, ngrid = 100, bcond = [Bcond(0, 2, 0.0), Bcond(-1, 2, 0.0)], op_ignore = ['R']):
    '''
    Takes in all_models and creates a dictionary mapping each model to the original dftb equivalent
    
    Default boundary conditions is 'natural', and default number of gridpoints is 100.
    
    This is just used so that first and second derivative information are all stored and easily accessible, cuts down
    on computation time. The dictionary is only for two-body potentials for either the H or G operator. Right now,
    ignore the 'G' operator
    '''
    model_spline_dict = dict()
    for model in all_models:
        try:
            #Only two-body G and H models rn
            if (model.oper not in op_ignore) and (len(model.Zs) == 2):
                pairwise_lin = all_models[model]
                r_low, r_high = pairwise_lin.pairwise_linear_model.r_range()
                rgrid = np.linspace(r_low, r_high, ngrid)
                ygrid = get_dftb_vals(model, par_dict, rgrid)
                model_spline_dict[model] = spline_linear_model(rgrid, None, (rgrid, ygrid), bcond)
        except:
            pass
    return model_spline_dict

def compute_model_deviation(all_models, penalties, concavity_dict):
    '''
    all_models (dict): dictionary of all models mapping the model_spec to the object 
                instance of that model
    penalties (dict): The different penalties to use and their relative weights
    concavity_dict (dict): Dictionary saying which two-body splines should be concave up (False) or concave down (True)

    Computes the losses using the ModelPenalty class
    
    Only the models doing 2-body potentials for the H operator should have to have their
    concavity figured out. For now, only regularize the splines for the hamiltonian elements
    '''
    deviation_loss = 0
    for model_spec in all_models:
        # Only compute deviation for the off-diagonal H elements for now, 
        # include other matrix elements later!
        try:
            if len(model_spec.Zs) == 2 and model_spec.oper == 'H':
                curr_model = all_models[model_spec]
                # False -> concave up, True -> concave down
                neg_integral = concavity_dict[model_spec]
                mod_penalty = ModelPenalty(curr_model, neg_integral = neg_integral, penalties = penalties)
                deviation_loss += mod_penalty.get_loss()
        except:
            pass
    return deviation_loss

def compute_variable_loss(output, data_dict, targets, scheme = 'MSE'):
    '''
    Similar to previous loss method, computes the loss between the output and the
    data_dict reference
    
    Fill in later, for now use loss based on total energy
    '''
    return total_energy_loss(output, data_dict, targets)

def compute_all_dipoles(output, data_dict, dipole_mats):
    '''
    Computes the dipoles for all molecules, similar to what is done in losschemtools.py
    
    The data_dict must store the dQ's by basis size, and the dipole_mats must also be a dict
    with the basis sizes as keys.
    
    To compute the dipoles for all molecules of a given basis size, we do the following:
        np.dot(A, dQ)
        A : (ngeom, 3, nbasis)
        dQ: (ngeom, nbasis, 1)
    The batch dimension (ngeom) is broadcast for this multiplication.
    
    The result will be a (ngeom, 3, 1), which will be squeezed to (ngeom, 3)
    '''
    all_bsizes = list(dipole_mats.keys())
    computed_dipoles = list()
    true_dipoles = list()
    for bsize in all_bsizes:
        comp_dQ = output['dQ'][bsize]
        true_dQ = data_dict['dQ'][bsize]
        curr_mat = dipole_mats[bsize]
        comp_dip = torch.matmul(curr_mat, comp_dQ).squeeze(2)
        true_dip = torch.matmul(curr_mat, true_dQ).squeeze(2)
        for i in range(comp_dip.shape[0]): #Save all the dipoles together
            computed_dipoles.append(comp_dip[i])
            true_dipoles.append(true_dip[i])
    return torch.cat(computed_dipoles), torch.cat(true_dipoles)

def dimension_correction(all_tensors):
    '''
    Makes sure scalars have at least one dimension
    '''
    for tensor in all_tensors:
        if len(tensor.shape) == 0:
            tensor = tensor.unsqueeze(0)
        
def compute_variable_loss_alt(output, data_dict, targets, scheme = 'MSE'):
    '''
    Variant for computing total loss, this time with an actual list of targets rather than 
    only computing the electonic energy. This will require casing on the targets and applying the correct
    loss calculation
    '''
    if targets == []:
        raise ValueError("Need at least one target for computing loss!")
    all_bsizes = data_dict['basis_sizes']
    loss_criterion = nn.MSELoss()
    target_tensors, computed_tensors = list(), list()
    for target in targets:
        for bsize in all_bsizes:
            if target == "Etot":
                computed_result = output['Erep'][bsize] + output['Eelec'][bsize] + output['Eref'][bsize]
                target_result = data_dict['Etot'][bsize]
                dimension_correction([computed_result, target_result])
                computed_tensors.append(computed_result)
                target_tensors.append(target_result)
            elif target == 'dipvec':
                computed_dipoles, target_dipoles = compute_all_dipoles(output, data_dict, data_dict['dipole_mat'])
                computed_tensors.append(computed_dipoles)
                target_tensors.append(target_dipoles)
    total_targets = torch.cat(target_tensors)
    total_computed = torch.cat(computed_tensors)
    return loss_criterion(total_computed, total_targets)
    

def compute_total_loss(output, data_dict, targets, all_models, concavity_dict, penalties, weights):
    '''
    Computes the total loss as the sum of penalty losses and the property losses. This final loss
    should be a PyTorch object with a backward() method, i.e. able to backpropagate
    
    Weights is a new dictionary with two keys, 'targets' and 'deviations'. They map to the weights 
    that should be used for the target loss and 
    '''
    target_loss = compute_variable_loss_alt(output, data_dict, targets)
    deviation_loss = 0
    if weights is not None:
        return weights['targets'] * target_loss + weights['deviations'] * deviation_loss
    else:
        return target_loss + deviation_loss

#%% Plotting functions

def plot_spline(spline_model, ngrid: int = 500) -> None:
    r"""Takes an instance of input_pairwise_linear and plots using present variable vector
    
    Arguments:
        spline_model (input_pairwise_linear): Instance of input_pairwise_linear for plotting
        ngrid (int): Number of grid points for evaluation. Defaults to 500
    
    Returns:
        None
    
    Notes: Deprecated method, only for non-joined splines.
    """
    rlow, rhigh = spline_model.pairwise_linear_model.r_range()
    rgrid = np.linspace(rlow, rhigh, ngrid)
    dgrids_consts = [spline_model.pairwise_linear_model.linear_model(rgrid, 0),
                     spline_model.pairwise_linear_model.linear_model(rgrid, 1),
                     spline_model.pairwise_linear_model.linear_model(rgrid, 2)]
    model_variables = spline_model.get_variables().detach().numpy()
    model = spline_model.model
    for i in range(3):
        fig, ax = plt.subplots()
        y_vals = np.dot(dgrids_consts[i][0], model_variables) + dgrids_consts[i][1]
        ax.plot(rgrid, y_vals)
        ax.set_title(f"{model} deriv {i}")
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


#%% Some testing
if __name__ == "__main__":
    pass


        

