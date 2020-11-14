# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 19:56:50 2020

@author: Frank
"""
'''
Module consisting of models for losses. Will adapt from loss_methods.py to have
class-based losses for each kind of loss. Similar code structure to the
all_models
'''
import torch
import torch.nn as nn
import numpy as np
from batch import Model
from auorg_1_1 import ParDict
from tfspline import Bcond, spline_linear_model
from modelspline import get_dftb_vals
import matplotlib.pyplot as plt
from geometry import Geometry
from batch import create_batch, create_dataset, DFTBList

#%% External Functions
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
    return concavity_dict

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
    def __init__ (self, input_pairwise_linear, penalty, n_grid = 500, neg_integral = False):
        self.input_pairwise_lin = input_pairwise_linear
        self.penalty = penalty
        self.neg_integral = neg_integral
        
        #Compute the x-grid
        r_low, r_high = self.input_pairwise_lin.pairwise_linear_model.r_range()
        self.xgrid = np.linspace(r_low, r_high, n_grid)
        
        #Compute the derivative_grid for 0th, 1st, and 2nd order
        self.dgrid = dict()
        for i in range(3):
            self.dgrid[i] = self.input_pairwise_lin.pairwise_linear_model.linear_model(self.xgrid, i)[0]
    
    # Now some methods for computing the different penalties
    def get_monotonic_penalty(self):
        '''
        Computes the monotonic penalty similar to that done in solver.py
        
        TODO: Come back to this and consider how monotonic penalty changes given spline sign
        '''
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
        return monotonic_penalty
        
    def get_convex_penalty(self):
        '''
        Computes the convex penalty similar to that done in solver.py
        '''
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
        return convex_penalty
    
    def get_smooth_penalty(self):
        '''
        Computes the smooth penalty similar to that done in solver.py
        
        Pretty sure this is going ot have to change
        '''
        smooth_penalty = 0
        c = self.input_pairwise_lin.get_variables()
        deriv = self.dgrid[2]
        deriv = torch.tensor(deriv)
        p_smooth = torch.einsum('j,ij->i',c,deriv)
        smooth_penalty = torch.einsum('i,i->', p_smooth, p_smooth)
        return smooth_penalty
    
    #Compute the actual loss
    def get_loss(self):
        '''
        Computes the overall loss as a sum of the individual penalties
        
        Nothing fancy like shuttling information back and forth in solver.py
        '''
        if self.penalty == "convex":
            return self.get_convex_penalty()
        elif self.penalty == "monotonic":
            return self.get_monotonic_penalty()
        elif self.penalty == "smooth":
            return self.get_smooth_penalty()
    
#%% Loss Model Interface
class LossModel:
    def get_feed(self, feed, molecs, all_models, par_dict, debug):
        '''
        Method for adding stuff to the feed for the given loss. Every loss
        will take in the feed and the associated molecules, but 
        
        The debug flag is set to true or false depending on whether or not to fit to DFTB.
        If performing a fit to DFTB, then the debug flag is set to true. Otherwise, it is False
        
        For consistency, the method should return a dictionary containing all relevant
        information for this loss and the key under which that dictionary should be saved
        in the feed dictionary
        '''
        raise NotImplementedError
    
    def get_value(self, output, feed):
        '''
        Computes the value of the loss directly against the feed from the output,
        returns the loss as a PyTorch object.
        '''
        raise NotImplementedError

#%% Loss Model Implementations
class TotalEnergyLoss(LossModel):
    def __init__(self):
        #Total energy loss does not require anything saved in its state
        pass
    def get_feed(self, feed, molecs, all_models, par_dict, debug):
        key = "Etot"
        result_dict = dict()
        all_bsizes = list(feed['glabels'].keys())
        if debug:
            for bsize in all_bsizes:
                result_dict[bsize] = feed['Eelec'][bsize] + feed['Erep'][bsize]
        else:
            for bsize in all_bsizes:
                glabels = feed['glabels'][bsize]
                total_energies = [molecs[x]['targets']['Etot'] for x in glabels]
                result_dict[bsize] = np.array(total_energies)
        return key, result_dict
    
    def get_value(self, output, feed):
        '''
        Computes the total energy loss using PyTorch MSEloss
        '''
        all_bsizes = list(output['Eelec'].keys())
        loss_criterion = nn.MSELoss() #Compute MSE loss by the pytorch specification
        target_tensors, computed_tensors = list(), list()
        for bsize in all_bsizes:
            computed_result = output['Erep'][bsize] + output['Eelec'][bsize] + output['Eref'][bsize] 
            target_result = feed['Etot'][bsize]
            if len(computed_result.shape) == 0:
                computed_result = computed_result.unsqueeze(0)
            if len(target_result.shape) == 0:
                target_result = target_result.unsqueeze(0)
            computed_tensors.append(computed_result)
            target_tensors.append(target_result)
        total_targets = torch.cat(target_tensors)
        total_computed = torch.cat(computed_tensors)
        return loss_criterion(total_computed, total_targets) 
    
class FormPenaltyLoss(LossModel):
    '''
    Takes in a penalty_type string that determines what kind of form penalty to compute, 
    i.e. monotonic, convex, smooth, etc.
    '''
    def __init__(self, penalty_type):
        self.type = penalty_type
        
    def get_feed(self, feed, molecs, all_models, par_dict, debug):
        '''
        No need to do anything with debug right now
        The FormPenaltyLoss deals with the convex, smooth, and monotonic penalties
        '''
        #If a previous FormPenaltyLoss has already added the concavity information,
        # no need to duplicate that computation
        
        if 'form_penalty' not in feed: 
            model_subset = dict()
            for model_spec in feed['models']:
                model_subset[model_spec] = all_models[model_spec]
            mod_spline_dict = compute_mod_vals_derivs(model_subset, par_dict)
            concavity_dict = generate_concavity_dict(mod_spline_dict)
            
            #Save both a dictionary pointing to the actual model and the concavity info
            final_dict = dict()
            for model_spec in concavity_dict:
                # Dictionary of the actual model and the concavity of said model
                final_dict[model_spec] = (model_subset[model_spec], concavity_dict[model_spec])
            return "form_penalty", final_dict
    
    def get_value(self, output, feed):
        form_penalty_dict = feed["form_penalty"]
        total_loss = 0
        for pairwise_lin_mod, concavity in form_penalty_dict:
            penalty_model = ModelPenalty(pairwise_lin_mod, self.type, neg_integral = concavity)
            total_loss += penalty_model.get_loss()
        return total_loss
    
class DipoleLoss(LossModel):
    '''
    Class to compute the dipole loss
    '''
    def __init__(self):
        pass
    def get_feed(self, feed, molecs, all_models, par_dict, debug):
        '''
        Gets the dipole mats required for computing the dipole loss
        
        Returns a tuple of dictionaries
        '''
        needed_fields = ['dipole_mat']
        geom_batch = list()
        for molecule in molecs:
            geom = Geometry(molecule['atomic_numbers'], molecule['coordinates'].T)
            geom_batch.append(geom)
                
        batch = create_batch(geom_batch)
        dftblist = DFTBList(batch)
        result = create_dataset(batch,dftblist,needed_fields)
        dipole_mats = result['dipole_mat']
        
        real_dipvecs = dict()
        #Also need to pull the dipoles from the molecules for comparison
        for bsize in feed['basis_sizes']:
            glabels = feed['glabels'][bsize]
            total_energies = [molecs[x]['targets']['dipole'] for x in glabels]
            real_dipvecs[bsize] = np.array(total_energies)
        return 'dipoles', (dipole_mats, real_dipvecs)
    
    def get_value(self, output, feed):
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
        
        The real dipoles will have shape (ngeom, 3) as well
        '''
        dipole_mats, real_dipoles = feed['dipoles']
        loss_criterion = nn.MSELoss()
        computed_dips = list()
        real_dips = list()
        for bsize in feed['basis_sizes']:
            curr_dQ = output['dQ'][bsize]
            curr_dipmat = dipole_mats[bsize]
            comp_result = torch.matmul(curr_dipmat, curr_dQ).squeeze(2)
            assert(comp_result.shape[0] == real_dipoles[bsize].shape[0])
            for i in range(comp_result.shape[0]):
                computed_dips.append(comp_result[i])
                real_dips.append(real_dipoles[bsize][i])
        total_comp_dips = torch.cat(computed_dips)
        total_real_dips = torch.cat(real_dips)
        return loss_criterion(total_comp_dips, total_real_dips)