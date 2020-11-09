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

def loss_temp(output, data_dict, targets):
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
        r_low, r_high = self.input_pairwise_lin.r_range()
        self.xgrid = np.linspace(r_low, r_high, n_grid)
        
        #Compute the derivative_grid for 0th, 1st, and 2nd order
        self.dgrid = dict()
        for i in range(3):
            self.dgrid[i] = self.input_pairwise_lin.linear_model(self.xgrid, i)[0]
        
        #Do a mini penalty check for the weights of each penalty
        self.monotonic_enabled = False
        self.convex_enabled = False
        self.smooth_enabled = False
    
    def penalty_check(self):
        '''
        Checks the penalties and toggle the flags
        '''
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
        p_monotonic = torch.einsum('j,ij->i', c, deriv)
        p_monotonic [p_monotonic > 0] = 0 #Care only about the negative terms
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
        '''
        lambda_smooth = self.penalties['smooth']
        smooth_penalty = 0
        c = self.input_pairwise_lin.get_variables()
        deriv = self.dgrid[2]
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
    
    This is a really hacky way of determining concave up or down, so
    TODO: revisit this! Find more robust way
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

def compute_model_deviation(all_models, thresholds, penalties, scheme = 'MSE'):
    '''
    all_models (dict): dictionary of all models mapping the model_spec to the object 
                instance of that model
    threshold (dict): value to be used as a comparison for deviations for each type of 
                      penalty. Number of elements in threshold should match penalties
    penalties (dict): The different penalties to use and their relative weights
    scheme (string): The method for computing the loss. Default is torch.MSE
    
    Computes the penalties using the ModelPenalty class and translates those penalties
    into losses by the weights
    '''
    loss_criterion = None
    if scheme == 'MSE':
        loss_criterion  = nn.MSELoss()
    for model_spec in all_models:
        # Only compute deviation for the off-diagonal elements
        if len(model_spec.Zs) == 2:
            #Compute the refactored loss for the
            pass
            
            
    pass

def compute_variable_loss(output, data_dict, targets, scheme = 'MSE'):
    '''
    Similar to previous loss method, computes the loss between the output and the
    data_dict reference
    
    Fill in later, for now use loss_temp
    '''
    return loss_temp(output, data_dict, targets)

def compute_total_loss(output, data_dict, targets, all_models, thresholds, penalties, scheme = 'MSE'):
    '''
    Computes the total loss as the sum of penalty losses and the property losses. This final loss
    should be a PyTorch object with a backward() method, i.e. able to backpropagate
    '''
    pass
    
    
    

#%% Some testing
H_mods = [Model(oper='H', Zs=(1,), orb='s'), Model(oper='H', Zs=(6,), orb='p'), 
                 Model(oper='H', Zs=(6,), orb='s'), Model(oper='H', Zs=(7,), orb='p'), 
                 Model(oper='H', Zs=(7,), orb='s'), Model(oper='H', Zs=(8,), orb='p'), 
                 Model(oper='H', Zs=(8,), orb='s'), Model(oper='H', Zs=(1, 1), orb='ss'),
                 Model(oper='H', Zs=(1, 6), orb='sp'), Model(oper='H', Zs=(1, 7), orb='sp'), 
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

par_dict = ParDict()
results = [find_concavity_H(model_spec, par_dict) for model_spec in H_double_mods_only]
result_dict = {key : val for key, val in zip(H_double_mods_only, results)}

result_keys = list(result_dict.keys())
ss_keys = [mod for mod in result_keys if mod.orb == 'ss']
sp_keys = [mod for mod in result_keys if mod.orb == 'sp']
pp_sig = [mod for mod in result_keys if mod.orb == 'pp_sigma']
pp_pi = [mod for mod in result_keys if mod.orb == 'pp_pi']

for k in ss_keys:
    assert(result_dict[k])
for k in sp_keys:
    assert(not result_dict[k])
for k in pp_sig:
    assert(not result_dict[k])
for k in pp_pi:
    assert(result_dict[k])
    
print("Tests passed")


        

