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
from typing import Union, List, Optional, Dict, Any, Literal
Array = np.ndarray
Tensor = torch.Tensor

#%% External Functions
# The determination of concavity is independent of the current implementations of the 
# models; they are based on the model_spec and dftb values only!
def compute_mod_vals_derivs(all_models: Dict, par_dict: Dict, ngrid: int = 200, 
                            bcond: List[Bcond] = [Bcond(0, 2, 0.0), Bcond(-1, 2, 0.0)], op_ignore: List[str] = []) -> Dict:
    r"""Computes the matrix X and vector const based on DFTB values for determining concavity
    
    Arguments:
        all_models (Dict): A dictionary referencing all the models that need to have their
            (X, const) pair computed
        par_dict (Dict): Dictionary of the DFTB Slater-Koster parameters for atomic interactions 
            between different elements, indexed by a string 'elem1-elem2'. For example, the
            Carbon-Carbon interaction is accessed using the key 'C-C'
        ngrid (int): The number of grid points to use for evaluating the spline. Defaults to 200
        bcond (List[Bcond]): A list of the boundary conditions to use for the spline. Defaults to 
            a list representing natural boundary conditions at start and end
        op_ignore (List[str]): A list of the operators to ignore
    
    Returns:
        model_spline_dict (Dict): Dictionary mapping the model_specs to the (X, b) pairs necessary for evaluating
            the curvature of the spline.
    
    Notes: By default the spline_linear_model computes a cubic spline up to the second derivative. 
        This behavior is necessary for this code to work, so that should not be changed in tfspline, 
        which has those defaults encoded.
        
        This computation is based on what the models should look like from the dftb slater-koster parameters
        alone. This is necessary to determine the correct concavity and curvature later on.
    """
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

def generate_concavity_dict(model_spline_dict: Dict) -> Dict[Model, bool]:
    r"""Generates a dictionary mapping each model to its concavity
    
    Arguments:
        model_spline_dict (Dict): output from compute_mod_vals_derivs
    
    Returns:
        concavity_dict (Dict[Model, bool]): A dictionary mapping each 
            model spec to its concavity. If the model should be concave down,
            then set to True; otherwise, set to False
    
    Notes: The concavity of the model is deteremined by the sign of the
        average value of the predictions of the second derivative. If the 
        average value is 0, then this means the spline is flat and has
        no curvature. In this case, the spline is not optimized since there
        is no interaction
    """
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

    def __init__ (self, input_pairwise_linear, penalty: str, dgrid: (Array, Array), n_grid: int = 500, neg_integral: bool = False) -> None:
        r"""Initializes the ModelPenalty object for computing the spline functional form penalty
        
        Arguments:
            input_pairwise_linear: This can be either an Input_layer_pairwise_linear or 
                input_layer_pairwise_linear_joined object.
            penalty (str): The pnalty to be computed. One of "convex", "monotonic", "smooth"
            dgrid (Array, Array): The matrix A and vector b for doing y = Ax + b. Depending on 
                the penalty,it's either the first or second derivative
            n_grid (int): Number of grid points to use. Defaults to 500
            neg_integral (bool): Whether the spline is concave down. Defaults to False (concave up)
        
        Returns:
            None
        
        Notes: This is adapted from solver.py from the dftbrepulsive project. For concavity,
            we define concave down as (v''(r) < 0) and concave up as (v''(r) > 0), where r is
            the interatomic distance. If no dgrids are given, then they are recomputed which is
            computationally inefficient.
        """
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
    def get_monotonic_penalty(self) -> float:
        r"""Computes the monotonic penalty for the model
        
        Arguments:
            None
            
        Returns:
            monotonic_penalty (float): The value for the monotonic penalty, with
                gradients for backpropagation
        
        Notes: The monotonic penalty depends on the first integral, v'(r). If the 
            spline should be monotonically increasing (concave down), we enforce that 
            v'(r) > 0. Otherwise, we enforce that v'(r) < 0 for a concave up, 
            monotonically decreasing potential. Enforcement of the penalty is done through
            the ReLU activation function, where if the spline should be concave down, we multiply 
            all the values by -1 and apply ReLU to only penalize the negative terms. Otherwise,
            we just apply ReLU to penalize the positive terms.
            
            The final penalty is computed as a dot product of the vector resulting from
            the application of ReLU.
        """
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
        
    def get_convex_penalty(self) -> float:
        r"""Computes the convex penalty
        
        Arguments:
            None
        
        Returns:
            convex_penalty (float): The value of the convex penalty with gradients 
                for backpropagation
        
        Notes: The convex penalty deals with the second derivative. For a concave up function,
            we enforce that v''(r) > 0, and for a concave down function we enforce that v''(r) < 0. 
            This us achieved again through the use of ReLU.
        """
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
    
    def get_smooth_penalty(self) -> float:
        r"""Computes the smooth penalty
        
        Arguments: 
            None
        
        Returns:
            smooth_penalty (float): The value for the smooth penalty
        
        Notes: The smooth penalty is computed as a dot product on the second
            derivative values, enforcing the second derivative goes to 0. 
            Not currently used.
        """
        smooth_penalty = 0
        c = self.input_pairwise_lin.get_variables()
        deriv = self.dgrid
        deriv = torch.tensor(deriv)
        p_smooth = torch.einsum('j,ij->i',c,deriv)
        smooth_penalty = torch.einsum('i,i->', p_smooth, p_smooth)
        return smooth_penalty
    
    #Compute the actual loss
    def get_loss(self) -> float:
        r"""Computes the loss
        
        Arguments: 
            None
        
        Returns:
            penalty (float): Calls on the requisite function depending on the
                set penalty to compute the value for the form penalty
        
        Notes: None
        """
        if self.penalty == "convex":
            return self.get_convex_penalty()
        elif self.penalty == "monotonic":
            return self.get_monotonic_penalty()
        elif self.penalty == "smooth":
            return self.get_smooth_penalty()
    
#%% Loss Model Interface
class LossModel:
    def get_feed(self, feed: Dict, molecs: List[Dict], all_models: Dict, par_dict: Dict, debug: bool) -> None:
        '''
        Method for adding stuff to the feed for the given loss. Every loss
        will take in the feed and the associated molecules, but 
        
        The debug flag is set to true or false depending on whether or not to fit to DFTB.
        If performing a fit to DFTB, then the debug flag is set to true. Otherwise, it is False
        
        For consistency, the method adds things directly to the feed wihtout returning any values.
        '''
        raise NotImplementedError
    
    def get_value(self, output: Dict, feed: Dict) -> None:
        '''
        Computes the value of the loss directly against the feed from the output,
        returns the loss as a PyTorch object.
        
        For consistency, all losses have torch.sqrt applied since we are interested in 
        RMSE loss.
        '''
        raise NotImplementedError

#%% Loss Model Implementations
class TotalEnergyLoss(LossModel):
    def __init__(self) -> None:
        #Total energy loss does not require anything saved in its state
        pass
    
    def get_nheavy(self, lst: List[str]) -> int:
        r"""Computes the number of heavy atoms from a list formed by the molecule's name
        
        Arguments:
            lst (List[str]): The properly formed list derived from the molecule's name.
                By properly formed, we mean that for a formula like "C10H11", the list is
                ["C", "10", "H", "11"]. 
        
        Returns:
            n_heavy (int): The number of heavy (non-hydrogen) atoms in the molecule

        Notes: The list is formed by using regex and the findall method with the 
            appropriate pattern.
        """
        assert(len(lst) % 2 == 0)
        n_heavy = 0
        for i in range(0, len(lst), 2):
            if lst[i].isalpha() and lst[i] != 'H':
                n_heavy += int(lst[i+1])
        return n_heavy
    
    def get_feed(self, feed: Dict, molecs: List[Dict], all_models: Dict, par_dict: Dict, debug: bool) -> None:
        r"""Adds the necessary information for the total energy into the feed
        
        Arguments:
            feed (Dict): The input dictionary representing the current batch to add 
                information to
            molecs (List[Dict]): A list of the molecular conformations used to generate this batch
                all_models (Dict): A dictionary containing references to all the spline models being used
            par_dict (Dict):  Dictionary of the DFTB Slater-Koster parameters for atomic interactions 
                between different elements, indexed by a string 'elem1-elem2'. For example, the
                Carbon-Carbon interaction is accessed using the key 'C-C'
            debug (bool): A flag indicating whether we are in debug mode.
        
        Returns:
            None
        
        Notes: The total energy is pulled out from the molecules and added to the feed dictionary.
            Additionally, the number of heavy atoms in each molecule is also extracted and added
            in. For determining the number of heavy atoms, a regex approach is used [1]
        
        References:
            [1] https://stackoverflow.com/questions/9782835/break-string-into-list-elements-based-on-keywords
        """
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
    
    def get_value(self, output: Dict, feed: Dict, per_atom_flag: bool) -> Tensor:
        r"""Computes the loss for the total energy
        
        Arguments:
            output (Dict): The output dictionary from the dftb layer
            feed (Dict): The original input dictionary for the DFTB layer
            per_atom_flag (bool): Whether the energy should be trained on a
                per heavy atom basis
            
        Returns:
            loss (Tensor): The value for the total energy loss with gradients
                attached that allow backpropagation
        
        Notes: If total energy is computed per heavy atom, torch.div is used
            to perform element-wise division with gradients. This is slightly
            interface breaking, but a better workaround than handling things externally
            in the training loop.
        """
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

class ReferenceEnergyLoss(TotalEnergyLoss):
    r""" Loss model for training the reference energy only
    
    Because the reference enregy is a part of the total energy, the 
    same components are needed for the feed, hence the class inheritance. 
    The only difference is in get_value, where we are interested in backpropagating along
    E['ref'] only. 
    """
    
    def __init__(self) -> None:
        super().__init__()
    
    def get_value(self, output: Dict, feed: Dict) -> Tensor:
        r"""Computes the reference energy loss
        
        Arguments: 
            output (Dict): Output dictionary from the DFTB layer
            feed (Dict): The original feed dictionary
        
        Returns:
            loss (Tensor): The value of the reference energy loss
        
        Notes: We are interested in learning the set of parameters C_z such that
            E_method2 = E_method1 + Sum_z[N_z * C_z], where z indexes over all element 
            types in a molecule and N_z is the number of that atom in the molecule.
            
            Because total energy is computed as E_tot = Eelec + Erep + Eref, we
            will take E_method1 to be Eelec + Erep and subtract it from E_method2, giving
            E_ref = E_method2 - E_method1
            
            We then backpropagate along Eref. This will give a crude starting point 
            for the reference energy parameters. Because we are interested in 
            learning the true starting point of reference energy for each
            molecule, we will find the reference energies for E_molec
            rather than for E_atom, so that when using the TotalEnergyLoss, we
            do not end up dividing by the number of heavy atoms twice
        """
        all_bsizes = feed['basis_sizes']
        loss_criterion = nn.MSELoss()
        target_tensors, computed_tensors = list(), list()
        for bsize in all_bsizes:
            Elec_rep = output['Erep'][bsize] + output['Eelec'][bsize]
            true_energy = feed['Etot'][bsize]
            true_ref = true_energy - Elec_rep #Crude estimate of Eref
            computed_ref = output['Eref'][bsize]
            if len(true_ref.shape) == 0:
                true_ref = true_ref.unsqueeze(0)
            if len(computed_ref.shape) == 0:
                computed_ref = computed_ref.unsqueeze(0)
            target_tensors.append(true_ref)
            computed_tensors.append(computed_ref)
        total_targets = torch.cat(target_tensors)
        total_computed = torch.cat(computed_tensors)
        return torch.sqrt(loss_criterion(total_computed, total_targets))
            
class FormPenaltyLoss(LossModel):
    
    seen_dgrid_dict = dict()
    seen_concavity_dict = dict()
    
    def __init__(self, penalty_type: str, grid_density: int = 500) -> None:
        r"""Initializes the FormPenaltyLoss object
        
        Arguments:
            penalty_type (str): The penalty to be computed. One of "convex", "monotonic", "smooth"
            grid_density (int): The number of grid points to use for evaluating the splines. 
                Defaults to 500
        
        Returns:
            None
        
        Notes: As an optimization step, there are two class-level variables seen_dgrid_dict and
            seen_concavity_dict. They keep track of which models have already had their dgrids and concavities
            calculated, and saves those values to be reused. This prevents repeated computation, 
            and is useful since the dgrids and target concavities of the models never change for 
            computing the form penalties.
        """
        self.type = penalty_type
        self.density = grid_density
        
    def get_feed(self, feed: Dict, molecs: List[Dict], all_models: Dict, par_dict: Dict, debug: bool) -> None:
        r"""Adds the information needed for the form penalty loss to the feed
        
        Arguments:
            feed (Dict): The feed dictionary to the DFTB layer
            molecs (List[Dict]): List of molecule dictionaries used to construct
                the current feed
            all_models (Dict): A dictionary containing references to all the spline models
                used
            par_dict (Dict): Dictionary of the DFTB Slater-Koster parameters for atomic interactions 
                between different elements, indexed by a string 'elem1-elem2'. For example, the
                Carbon-Carbon interaction is accessed using the key 'C-C'
            debug (bool): A flag indicating whether we are in debug mode.
            
        Returns:
            None
        
        Notes: If a concavity or dgrid has already been seen, it is called from 
            class-level seen_dgrid_dict or seen_concavity_dict rather than recomputed. Adds the dgrids, concavity, and current
            models into the feed. Because the spline models are all aliased, everything is connected.
        """
        
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
    
    def get_value(self, output: Dict, feed: Dict) -> Tensor:
        r"""Computes the form penalty for the spline functional form
        
        Arguments:
            output (Dict): Output from the DFTB layer
            feed (Dict): The original feed into the DFTB layer
        
        Returns:
            loss (Tensor): The value for the form penalty loss with gradients
                attached that allow backpropagation
        
        Notes: None
        """
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
    
class DipoleLoss(LossModel): #DEPRECATED
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

class DipoleLoss2(LossModel):
    
    def __init__(self):
        pass
    
    def compute_charges(self, dQs: Union[Array, Tensor], ids: Union[Array, Tensor]) -> List[Tensor]:
        r"""Computes the charges with a segment sum over dQs
        
        Arguments:
            dQs (Union[Array, Tensor]): The current orbital-resolved charge fluctuations
                predicted from the DFTB layer
            ids (Union[Array, Tensor]): The atom_ids for summing the dQs together
        
        Returns:
            charge_tensors (List[Tensor]): List of charge tensors computed from the 
                dQ summation
        
        Notes: To get the charges, we first flatten the dQs and ids into 1-dimensional tensors.
            We then perform a scatter_add (same as tf.segsum) using the ids as a map for 
            summing the dQs together into on-atom charges.
        """
        charges = dQs
        #Should have the same dimensions (ngeom, nshells, 1)
        if isinstance(charges, np.ndarray) and isinstance(ids, np.ndarray):
            charges = torch.from_numpy(charges)
            ids = torch.from_numpy(ids)
        assert(charges.shape == ids.shape)
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
    
    def get_feed(self, feed: Dict, molecs: List[Dict], all_models: Dict, par_dict: Dict, debug: bool) -> None:
        r"""Adds the information needed for the dipole loss to the feed
        
        Arguments:
            feed (Dict): The feed dictionary to the DFTB layer
            molecs (List[Dict]): List of molecule dictionaries used to construct
                the current feed
            all_models (Dict): A dictionary containing references to all the spline models
                used
            par_dict (Dict): Dictionary of the DFTB Slater-Koster parameters for atomic interactions 
                between different elements, indexed by a string 'elem1-elem2'. For example, the
                Carbon-Carbon interaction is accessed using the key 'C-C'
            debug (bool): A flag indicating whether we are in debug mode.
            
        Returns:
            None
        
        Notes: Adds the dipole matrices (3 * Natom matrices of cartesian coordinates) and the
            real dipoles to the feed. Note that the debug mode does not currently work for DipoleLoss2.
        """
        if ('dipole_mat' not in feed) and ('dipoles' not in feed):
            # needed_fields = ['dipole_mat']
            # geom_batch = list()
            # for molecule in molecs:
            #     geom = Geometry(molecule['atomic_numbers'], molecule['coordinates'].T)
            #     geom_batch.append(geom)
                    
            # batch = create_batch(geom_batch)
            # dftblist = DFTBList(batch)
            # result = create_dataset(batch,dftblist,needed_fields)
            dipole_mat_dict = dict()
            for bsize, glabels in feed['glabels'].items():
                bsize_coords = [molecs[x]['coordinates'].T for x in glabels] #Should be list of (3, Natom) arrays
                dipole_mat_dict[bsize] = bsize_coords
            dipole_mats = dipole_mat_dict
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
                #Trick is going to be here
                for bsize, glabels in feed['glabels'].items():
                    curr_molec_rs = [molecs[x]['coordinates'].T for x in glabels]
                    curr_molec_charges = [molecs[x]['targets']['charges'] for x in glabels]
                    assert(len(curr_molec_rs) == len(curr_molec_charges))
                    indices = [i for i in range(len(curr_molec_rs))]
                    results = list(map(lambda x : np.matmul(curr_molec_rs[x], curr_molec_charges[x]), indices))
                    real_dipvecs[bsize] = np.array(results)
                feed['dipole_mat'] = dipole_mats
                feed['dipoles'] = real_dipvecs
        pass
    
    def get_value(self, output: Dict, feed: Dict) -> Tensor:
        r"""Computes the penalty for the dipoles
        
        Arguments:
            output (Dict): Output from the DFTB layer
            feed (Dict): The original feed into the DFTB layer
        
        Returns:
            loss (Tensor): The value for the dipole loss with gradients 
                attached for backpropagation
        
        Notes: Dipoles are computed from the predicted values for dQ as 
            mu = R @ q where q is the dQs summed into on-atom charges and 
            R is the matrix of cartesian coordinates. This mu is therefore the
            ESP dipoles, and they are used to prevent competition between 
            atomic charge and dipole optimization.
        """
        dipole_mats, real_dipoles = feed['dipole_mat'], feed['dipoles']
        loss_criterion = nn.MSELoss()
        computed_dips = list()
        real_dips = list()
        for bsize in feed['basis_sizes']:
            curr_dQ = output['dQ'][bsize]
            curr_ids = feed['atom_ids'][bsize]
            curr_dipmats = dipole_mats[bsize]
            curr_charges = self.compute_charges(curr_dQ, curr_ids)
            assert(len(curr_charges) == len(curr_dipmats) == len(real_dipoles[bsize]))
            for i in range(len(curr_charges)):
                cart_mat = torch.from_numpy(curr_dipmats[i])
                #dipoles computed as coords @ charges
                cart_mat = cart_mat.type(curr_charges[i].dtype)
                comp_res = torch.matmul(cart_mat, curr_charges[i])
                computed_dips.append(comp_res)
                real_dips.append(real_dipoles[bsize][i])
        total_comp_dips = torch.cat(computed_dips)
        total_real_dips = torch.cat(real_dips)
        return torch.sqrt(loss_criterion(total_comp_dips, total_real_dips))        

class ChargeLoss(LossModel):

    def __init__(self):
        pass
    
    def compute_charges(self, dQs: Union[Array, Tensor], ids: Union[Array, Tensor]) -> List[Tensor]:
        r"""Computes the charges with a segment sum over dQs
        
        Arguments:
            dQs (Union[Array, Tensor]): The current orbital-resolved charge fluctuations
                predicted from the DFTB layer
            ids (Union[Array, Tensor]): The atom_ids for summing the dQs together
        
        Returns:
            charge_tensors (List[Tensor]): List of charge tensors computed from the 
                dQ summation
        
        Notes: To get the charges, we first flatten the dQs and ids into 1-dimensional tensors.
            We then perform a scatter_add (same as tf.segsum) using the ids as a map for 
            summing the dQs together into on-atom charges.
        """
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
            
    def get_feed(self, feed: Dict, molecs: List[Dict], all_models: Dict, par_dict: Dict, debug: bool) -> None:
        r"""Adds the information needed for the charge loss to the feed
        
        Arguments:
            feed (Dict): The feed dictionary to the DFTB layer
            molecs (List[Dict]): List of molecule dictionaries used to construct
                the current feed
            all_models (Dict): A dictionary containing references to all the spline models
                used
            par_dict (Dict): Dictionary of the DFTB Slater-Koster parameters for atomic interactions 
                between different elements, indexed by a string 'elem1-elem2'. For example, the
                Carbon-Carbon interaction is accessed using the key 'C-C'
            debug (bool): A flag indicating whether we are in debug mode.
            
        Returns:
            None
        
        Notes: Adds atomic charges for each molecule. Because molecules are organized by bsize and 
            not number of atoms, the charges are stored in the feed dictionary as a list to prevent
            having to deal with ragged tensors or arrays.
        """
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
    
    def get_value(self, output: Dict, feed: Dict) -> Tensor:
        r"""Computes the loss for charges
        
        Arguments:
            output (Dict): Output from the DFTB layer
            feed (Dict): The original feed into the DFTB layer
        
        Returns:
            loss (Tensor): The value for the dipole loss with gradients 
                attached for backpropagation
        
        Notes: Charges are compouted by summing the orbital-resolved charge fluctuations
            into on-atom charges using the compute_charges function and a segment sum.
        """
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