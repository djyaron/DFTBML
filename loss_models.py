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
from loss_methods import plot_spline
import re

#%% External Functions
# The determination of concavity is independent of the current implementations of the 
# models; they are based on the model_spec and dftb values only!
def compute_mod_vals_derivs(all_models, par_dict, ngrid = 200, bcond = [Bcond(0, 2, 0.0), Bcond(-1, 2, 0.0)], op_ignore = []):
    '''
    Takes in all_models and creates a dictionary mapping each model to the original dftb equivalent since
    we want to be sure that the concavity is based on the dftb data
    
    Default boundary conditions is 'natural', and default number of gridpoints is 100.
    
    This is just used so that first and second derivative information are all stored and easily accessible, cuts down
    on computation time.
    '''
    model_spline_dict = dict()
    for model in all_models:
        try:
            #Only two-body H, G, R
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
        else:
            print(f"Zero concavity detected for {model_spec}")
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
        penalty (str): Indicates the type of penalty to apply
        dgrid (array): The derivative grid to use for the given penalty. 
        n_grid (int): number of points to use when evaluating the second derivative and the penalty. Default = 500
        neg_integral (bool): Indicates whether the spline represented by this model should be concave up (v''(r) > 0) or
                             concave down (v''(r) < 0). If True, then concave down; otherwise, concave up
    '''
    def __init__ (self, input_pairwise_linear, penalty, dgrid, n_grid = 500, neg_integral = False):
        self.input_pairwise_lin = input_pairwise_linear
        self.penalty = penalty
        self.neg_integral = neg_integral
        
        #Compute the x-grid
        r_low, r_high = self.input_pairwise_lin.pairwise_linear_model.r_range()
        self.xgrid = np.linspace(r_low, r_high, n_grid)
        
        #Compute the derivative_grid for the necessary derivative given the penalty
        self.dgrid = None
        if dgrid is None:
            if self.penalty == "convex" or self.penalty == "smooth":
                self.dgrid = self.input_pairwise_lin.pairwise_linear_model.linear_model(self.xgrid, 2)
            elif self.penalty == "monotonic":
                self.dgrid = self.input_pairwise_lin.pairwise_linear_model.linear_model(self.xgrid, 1)
        else:
            self.dgrid = dgrid
    
    # Now some methods for computing the different penalties
    def get_monotonic_penalty(self):
        '''
        Computes the monotonic penalty similar to that done in solver.py
        
        TODO: Come back to this and consider how monotonic penalty changes given spline sign
        '''
        monotonic_penalty = 0
        m = torch.nn.ReLU()
        c = self.input_pairwise_lin.get_variables()
        if hasattr(self.input_pairwise_lin, 'joined'):
            other_coefs = self.input_pairwise_lin.get_fixed()
            c = torch.cat([c, other_coefs])
        deriv, consts = self.dgrid[0], self.dgrid[1]
        deriv, consts = torch.tensor(deriv), torch.tensor(consts)
        p_monotonic = torch.einsum('j,ij->i', c, deriv) + consts
        #For a monotonically increasing potential (i.e. concave down integral), the
        # First derivative should be positive, so penalize the negative terms. Otherwise,
        # penalize the positive terms for concave up
        if self.neg_integral:
            p_monotonic = -1 * p_monotonic
            p_monotonic = m(p_monotonic)
            # p_monotonic [p_monotonic > 0] = 0
        else:
            # p_monotonic [p_monotonic < 0] = 0 
            p_monotonic = m(p_monotonic)
        monotonic_penalty = torch.einsum('i,i->', p_monotonic, p_monotonic) / len(p_monotonic)
        return monotonic_penalty
        
    def get_convex_penalty(self):
        '''
        Computes the convex penalty similar to that done in solver.py
        '''
        convex_penalty = 0
        m = torch.nn.ReLU()
        c = self.input_pairwise_lin.get_variables()
        if hasattr(self.input_pairwise_lin, 'joined'):
            other_coefs = self.input_pairwise_lin.get_fixed()
            c = torch.cat([c, other_coefs])
        deriv, consts = self.dgrid[0], self.dgrid[1]
        deriv, consts = torch.tensor(deriv), torch.tensor(consts)
        p_convex = torch.einsum('j,ij->i', c, deriv) + consts
        # Case on whether the spline should be concave up or down
        if self.neg_integral:
            p_convex = m(p_convex)
        else:
            p_convex = -1 * p_convex
            p_convex = m(p_convex)
        # Equivalent of RMS penalty
        convex_penalty = torch.einsum('i,i->', p_convex, p_convex) / len(p_convex)
        return convex_penalty
    
    def get_smooth_penalty(self):
        '''
        Computes the smooth penalty similar to that done in solver.py
        
        Pretty sure this is going ot have to change
        
        Not sure what the smooth penalty means here
        '''
        smooth_penalty = 0
        c = self.input_pairwise_lin.get_variables()
        deriv = self.dgrid
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
        
        For consistency, the method adds things directly to the feed wihtout returning any values.
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
    
    def get_nheavy(self, lst):
        '''
        Gets the nuber of heavy atoms out of a list of the molecule name
        '''
        assert(len(lst) % 2 == 0)
        n_heavy = 0
        for i in range(0, len(lst), 2):
            if lst[i].isalpha() and lst[i] != 'H':
                n_heavy += int(lst[i+1])
        return n_heavy
    
    def get_feed(self, feed, molecs, all_models, par_dict, debug):
        if "Etot" not in feed:
            key = "Etot"
            result_dict = dict()
            all_bsizes = list(feed['glabels'].keys())
                
            if debug:
                # NOTE: Debug mode is done on energy per molecule, not energy per heavy atom
                for bsize in all_bsizes:
                    result_dict[bsize] = feed['Eelec'][bsize] + feed['Erep'][bsize]
            else:
                for bsize in all_bsizes:
                    glabels = feed['glabels'][bsize]
                    total_energies = [molecs[x]['targets']['Etot'] for x in glabels]
                    result_dict[bsize] = np.array(total_energies)
            feed[key] = result_dict
        
        if "nheavy" not in feed:
            # Add the number of heavy atoms
            heavy_dict = dict()
            pattern = '[A-Z][a-z]?|[0-9]+'
            all_bsizes = feed['basis_sizes']
            for bsize in all_bsizes:
                names = feed['names'][bsize]
                # Regex approach from https://stackoverflow.com/questions/9782835/break-string-into-list-elements-based-on-keywords
                split_lsts = list(map(lambda x : re.findall(pattern, x), names))
                heavy_lst = list(map(lambda x : self.get_nheavy(x), split_lsts))
                heavy_dict[bsize] = np.array(heavy_lst)
            feed['nheavy'] = heavy_dict
    
    def get_value(self, output, feed, per_atom_flag):
        '''
        Computes the total energy loss using PyTorch MSEloss
        '''
        all_bsizes = list(output['Eelec'].keys())
        loss_criterion = nn.MSELoss() #Compute MSE loss by the pytorch specification
        target_tensors, computed_tensors = list(), list()
        for bsize in all_bsizes:
            n_heavy = feed['nheavy'][bsize].long()
            computed_result = output['Erep'][bsize] + output['Eelec'][bsize] + output['Eref'][bsize]
            if per_atom_flag:
                computed_result = torch.div(computed_result, n_heavy)
            target_result = feed['Etot'][bsize]
            if len(computed_result.shape) == 0:
                computed_result = computed_result.unsqueeze(0)
            if len(target_result.shape) == 0:
                target_result = target_result.unsqueeze(0)
            computed_tensors.append(computed_result)
            target_tensors.append(target_result)
        total_targets = torch.cat(target_tensors)
        total_computed = torch.cat(computed_tensors)
        # RMS loss for total energy
        return torch.sqrt(loss_criterion(total_computed, total_targets))
    
class FormPenaltyLoss(LossModel):
    '''
    Takes in a penalty_type string that determines what kind of form penalty to compute, 
    i.e. monotonic, convex, smooth, etc.
    
    Can specify the grid density, but the default is 500. Form penalty only applies
    to two-body potentials
    '''
    # DEBUGGING CODE
    # tst_mod = Model(oper='G', Zs=(1, 1), orb='ss')
    # tst_penalty = "convex"
    # tst_dgrid = None
    
    #Additional optimization: Keep track of the models and dgrid sets we've seen so far.
    # Also, keep track of the spline_dicts that we've seen so far for future use
    seen_dgrid_dict = dict()
    seen_concavity_dict = dict()
    
    def __init__(self, penalty_type, grid_density = 500):
        self.type = penalty_type
        self.density = grid_density
        
    def get_feed(self, feed, molecs, all_models, par_dict, debug):
        '''
        No need to do anything with debug right now
        The FormPenaltyLoss deals with the convex, smooth, and monotonic penalties
        '''
        #If a previous FormPenaltyLoss has already added the concavity information,
        # no need to duplicate that computation
        
        if 'form_penalty' not in feed: 
            # First, check to see what's already done
            concavity_dict = dict()
            # Models that have not had their concavity computed need to have that done and the results saved
            model_subset = dict()
            for mod_spec in feed['models']:
                if mod_spec in FormPenaltyLoss.seen_concavity_dict:
                    concavity_dict[mod_spec] = FormPenaltyLoss.seen_concavity_dict[mod_spec]
                elif (mod_spec not in FormPenaltyLoss.seen_concavity_dict) and (len(mod_spec.Zs) == 2):
                    model_subset[mod_spec] = all_models[mod_spec]
            mod_spline_dict = compute_mod_vals_derivs(model_subset, par_dict)
            temp_concav_dict = generate_concavity_dict(mod_spline_dict)
            # non-empty dictionaries evaluate to true in python
            if temp_concav_dict:
                concavity_dict.update(temp_concav_dict)
                FormPenaltyLoss.seen_concavity_dict.update(temp_concav_dict)
            
            #Optimization (push onto pre-compute), save the dgrid for the model as well
            final_dict = dict()
            for model_spec in concavity_dict:
                current_model = all_models[model_spec]
                if model_spec in FormPenaltyLoss.seen_dgrid_dict:
                    final_dict[model_spec] = (current_model, concavity_dict[model_spec], FormPenaltyLoss.seen_dgrid_dict[model_spec])
                else:
                    rlow, rhigh = current_model.pairwise_linear_model.r_range()
                    xgrid = np.linspace(rlow, rhigh, self.density)
                    #We only need the first and second derivative for the dgrids
                    #Including the constants, especially important for the joined splines!
                    dgrids = [current_model.pairwise_linear_model.linear_model(xgrid, 1),
                              current_model.pairwise_linear_model.linear_model(xgrid, 2)] 
                    final_dict[model_spec] = (current_model, concavity_dict[model_spec], dgrids)
                    FormPenaltyLoss.seen_dgrid_dict[model_spec] = dgrids
            feed['form_penalty'] = final_dict
    
    def get_value(self, output, feed):
        form_penalty_dict = feed["form_penalty"]
        total_loss = 0
        for model_spec in form_penalty_dict:
            # if model_spec.oper != 'G':
            pairwise_lin_mod, concavity, dgrids = form_penalty_dict[model_spec]
            penalty_model = None
            if self.type == "convex" or self.type == "smooth":
                penalty_model = ModelPenalty(pairwise_lin_mod, self.type, dgrids[1], n_grid = self.density, neg_integral = concavity)
            elif self.type == "monotonic":
                penalty_model = ModelPenalty(pairwise_lin_mod, self.type, dgrids[0], n_grid = self.density, neg_integral = concavity)
            total_loss += penalty_model.get_loss()
            # DEBUGGING CODE
            # if (model_spec == FormPenaltyLoss.tst_mod) and (self.type == FormPenaltyLoss.tst_penalty):
            #     if FormPenaltyLoss.tst_dgrid is None:
            #         FormPenaltyLoss.tst_dgrid = penalty_model.dgrid
            #     else:
            #         result = (penalty_model.dgrid == FormPenaltyLoss.tst_dgrid).all()
            #         if result:
            #             print("Arrays hold up over iterations")
            #         else:
            #             print("Arrays change over time")
        return torch.sqrt(total_loss / len(form_penalty_dict))
    
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
        if ('dipole_mat' not in feed) and ('dipoles' not in feed):
            needed_fields = ['dipole_mat']
            geom_batch = list()
            for molecule in molecs:
                geom = Geometry(molecule['atomic_numbers'], molecule['coordinates'].T)
                geom_batch.append(geom)
                    
            batch = create_batch(geom_batch)
            dftblist = DFTBList(batch)
            result = create_dataset(batch,dftblist,needed_fields)
            dipole_mats = result['dipole_mat']
            # In debugging case, we are just getting the dipole vectors computed from DFTB
            if debug:
                #Assert that the glabels and the dipole_mats have the same keys
                assert(set(feed['basis_sizes']).difference(set(dipole_mats.keys())) == set())
                dipvec_dict = dict()
                for bsize in feed['basis_sizes']:
                    dipoles = np.matmul(dipole_mats[bsize], feed['dQ'][bsize])
                    dipoles = np.squeeze(dipoles, 2) #Reduce to shape (ngeom, 3)
                    dipvec_dict[bsize] = dipoles
                feed['dipole_mat'] = dipole_mats
                feed['dipoles'] = dipvec_dict
            # If not debugging, we pull the real dipole vectors 
            else:
                real_dipvecs = dict()
                #Also need to pull the dipoles from the molecules for comparison
                for bsize in feed['basis_sizes']:
                    glabels = feed['glabels'][bsize]
                    total_dipoles = [molecs[x]['targets']['dipole'] for x in glabels]
                    real_dipvecs[bsize] = np.array(total_dipoles)
                feed['dipole_mat'] = dipole_mats
                feed['dipoles'] = real_dipvecs
            
    def get_value(self, output, feed):
        '''
        Computes the dipoles for all molecules, similar to what is done in losschemtools.py
        
        The data_dict must store the dQ's by basis size, and the dipole_mats must also be a dict
        with the basis sizes as keys.
        
        To compute the dipoles for all molecules of a given basis size, we do the following:
            torch.matmul(A, dQ)
            A : (ngeom, 3, nbasis)
            dQ: (ngeom, nbasis, 1)
        The batch dimension (ngeom) is broadcast for this multiplication.
        
        The result will be a (ngeom, 3, 1), which will be squeezed to (ngeom, 3)
        
        The real dipoles will have shape (ngeom, 3) as well, so everything matches
        '''
        dipole_mats, real_dipoles = feed['dipole_mat'], feed['dipoles']
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
        return torch.sqrt(loss_criterion(total_comp_dips, total_real_dips))

class ChargeLoss(LossModel):
    '''
    Class for handling training loss associated with charges
    
    The charges are computed as dQ + qneutral for a given species, since the charge 
    fluctuation is defined as dQ = Q_i + Q_0, where Q_0 is the neutral charge
    
    TODO: Fix the ragged tensor problem with charges!
    '''
    def __init__(self):
        pass
    
    def compute_charges(self, dQs, ids):
        '''
        Internal method to compute the charges 
        Uses scatter_add and returns a list of torch tensor objects
        '''
        charges = dQs
        #Should have the same dimensions (ngeom, nshells, 1)
        if isinstance(charges, np.ndarray) and isinstance(ids, np.ndarray):
            charges = torch.from_numpy(charges)
            ids = torch.from_numpy(ids)
        assert(charges.shape[0] == ids.shape[0])
        charge_tensors = []
        for i in range(charges.shape[0]):
            curr_ids = ids[i].squeeze(-1)
            curr_charges = charges[i].squeeze(-1)
            #Scale down by the minimum index
            scaling_val = curr_ids[0].item()
            curr_ids -= scaling_val
            temp = torch.zeros(int(curr_ids[-1].item()) + 1, dtype = curr_charges.dtype)
            temp = temp.scatter_add(0, curr_ids.long(), curr_charges)
            charge_tensors.append(temp)
        return charge_tensors
            
    def get_feed(self, feed, molecs, all_models, par_dict, debug):
        #Use 'charges' as the key for storing charge information for each molecule
        #Generates the charges used to compute the loss later on
        if 'charges' not in feed:
            if debug:
                all_bsizes = feed['basis_sizes']
                charge_dict = dict()
                for bsize in all_bsizes:
                    curr_dQ = feed['dQ'][bsize]
                    curr_ids = feed['atom_ids'][bsize]
                    #Now get true charges
                    true_charges = self.compute_charges(curr_dQ, curr_ids)
                    for i in range(len(true_charges)):
                        #Convert to numpy arrays for consistency
                        true_charges[i] = true_charges[i].numpy()
                    charge_dict[bsize] = true_charges
                feed['charges'] = charge_dict
            else:
                charge_dict = dict()
                for bsize in feed['basis_sizes']:
                    glabels = feed['glabels'][bsize]
                    total_charges = [molecs[x]['targets']['charges'] for x in glabels]
                    # starting_len = len(total_charges[0])
                    # try:
                    #     assert (all(len(ele) == starting_len for ele in total_charges))
                    # except:
                    #     print("Something went wrong with length of charges")
                    #     print([molecs[x] for x in glabels])
                    charge_dict[bsize] = total_charges
                feed['charges'] = charge_dict
    
    def get_value(self, output, feed):
        '''
        Because charges will have to be a ragged np array, will have to perform the 
        calculation per charge vector and convert to tensor as we go along
        '''
        all_bsizes = feed['basis_sizes']
        loss_criterion = nn.MSELoss()
        total_computed, total_targets = list(), list()
        for bsize in all_bsizes:
            real_charges = feed['charges'][bsize]
            curr_dQ_out = output['dQ'][bsize]
            curr_ids = feed['atom_ids'][bsize]
            computed_charges = self.compute_charges(curr_dQ_out, curr_ids)
            assert(len(computed_charges) == len(real_charges)) #Both are lists now since raggedness
            for i in range(len(computed_charges)):
                total_computed.append(computed_charges[i])
                tensor_reals = torch.from_numpy(real_charges[i])
                tensor_reals = tensor_reals.type(computed_charges[i].dtype)
                total_targets.append(tensor_reals)
                assert (total_computed[-1].shape == total_targets[-1].shape)
        computed_tensor = torch.cat(total_computed)
        target_tensor = torch.cat(total_targets)
        return torch.sqrt(loss_criterion(computed_tensor, target_tensor))