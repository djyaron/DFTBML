# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 19:12:34 2021

@author: fhu14
"""
#%% Imports, definitions
from base_classes import LossModel, ModelPenalty
from external_funcs import compute_mod_vals_derivs, generate_concavity_dict
from typing import List, Dict
import numpy as np
import re
import torch
Tensor = torch.Tensor
import torch.nn as nn

#%% Code behind

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
            
            #Optimization (push onto pre-compute), save the dgrid and xgrid for the model as well
            final_dict = dict()
            for model_spec in concavity_dict:
                current_model = all_models[model_spec]
                if model_spec in FormPenaltyLoss.seen_dgrid_dict:
                    final_dict[model_spec] = (current_model, concavity_dict[model_spec], FormPenaltyLoss.seen_dgrid_dict[model_spec][0],
                                              FormPenaltyLoss.seen_dgrid_dict[model_spec][1])
                else:
                    rlow, rhigh = current_model.pairwise_linear_model.r_range()
                    xgrid = np.linspace(rlow, rhigh, self.density)
                    #We only need the first and second derivative for the dgrids
                    #Including the constants, especially important for the joined splines!
                    dgrids = [current_model.pairwise_linear_model.linear_model(xgrid, 1),
                              current_model.pairwise_linear_model.linear_model(xgrid, 2)] 
                    final_dict[model_spec] = (current_model, concavity_dict[model_spec], dgrids, xgrid)
                    FormPenaltyLoss.seen_dgrid_dict[model_spec] = (dgrids, xgrid)
            feed['form_penalty'] = final_dict
    
    def get_value(self, output: Dict, feed: Dict, rep_method: str) -> Tensor:
        r"""Computes the form penalty for the spline functional form
        
        Arguments:
            output (Dict): Output from the DFTB layer
            feed (Dict): The original feed into the DFTB layer
            rep_method (str): The repulsive method used. 'old' means the form penalties 
                are applied to the old spline-based Rs and 'new' means the form penalties
                are excluded since the new DFTBrepulsive splines are used.
        
        Returns:
            loss (Tensor): The value for the form penalty loss with gradients
                attached that allow backpropagation
        
        Notes: None
        """
        form_penalty_dict = feed["form_penalty"]
        total_loss = 0
        for model_spec in form_penalty_dict:
            # if model_spec.oper != 'G':
            if (rep_method == 'new') and (model_spec.oper == 'R'):
                continue #Skip built-in repulsive models if using DFTBrepulsive implementation
            pairwise_lin_mod, concavity, dgrids, xgrid = form_penalty_dict[model_spec]
            inflection_point_val = pairwise_lin_mod.get_inflection_pt()
            # print(model_spec, inflection_point_val)
            penalty_model = None
            if self.type == "convex" or self.type == "smooth":
                penalty_model = ModelPenalty(pairwise_lin_mod, self.type, dgrids[1], inflect_point_val = inflection_point_val, n_grid = self.density, neg_integral = concavity,
                                              pre_comp_xgrid = xgrid)
                # loss_val = get_loss(pairwise_lin_mod, self.type, dgrids[1], inflect_point_val = inflection_point_val, neg_integral = concavity,
                #                         pre_comp_xgrid = xgrid)
            elif self.type == "monotonic":
                penalty_model = ModelPenalty(pairwise_lin_mod, self.type, dgrids[0], inflect_point_val = inflection_point_val, n_grid = self.density, neg_integral = concavity,
                                             pre_comp_xgrid = xgrid)
            total_loss += penalty_model.get_loss()
        return torch.sqrt(total_loss / len(form_penalty_dict))