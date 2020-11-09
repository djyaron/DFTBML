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
        penalties (dict): Should map the penalty with the weight that that penalty should have
    '''
    def __init__ (self, input_pairwise_linear, n_grid = 500, penalties = None):
        self.input_pairwise_lin = input_pairwise_linear
        self.penalties = penalties
        
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
    
    pass

def compute_variable_loss(output, data_dict, targets, scheme = 'MSE'):
    '''
    Similar to previous loss method, computes the loss between the output and the
    data_dict reference
    '''
    pass

def compute_total_loss(output, data_dict, targets, all_models, thresholds, penalties, scheme = 'MSE'):
    '''
    Computes the total loss as the sum of penalty losses and the property losses. This final loss
    should be a PyTorch object with a backward() method, i.e. able to backpropagate
    '''
    pass
    
    
    


        
        
