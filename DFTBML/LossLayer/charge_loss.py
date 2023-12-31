# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 19:47:44 2021

@author: fhu14
"""
#%% Imports, definitions
from .base_classes import LossModel
from .external_funcs import compute_charges
from typing import List, Dict
import numpy as np
Array = np.ndarray
import torch
Tensor = torch.Tensor
import torch.nn as nn

#%% Code behind

class ChargeLoss(LossModel):

    def __init__(self):
        pass
            
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
                    true_charges = compute_charges(curr_dQ, curr_ids)
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
                    charge_dict[bsize] = total_charges
                feed['charges'] = charge_dict
    
    def get_value(self, output: Dict, feed: Dict, rep_method: str) -> Tensor:
        r"""Computes the loss for charges
        
        Arguments:
            output (Dict): Output from the DFTB layer
            feed (Dict): The original feed into the DFTB layer
            rep_method (str): The repulsive method being used. 'new' is the built-in
                splines and 'old' is the DFTBrepulsive splines. Has no effect here, included
                for consistency with interface.
        
        Returns:
            loss (Tensor): The value for the dipole loss with gradients 
                attached for backpropagation
            prediction_dict (Dict): Dictionary of predicted total energies organized
                by basis size
        
        Notes: Charges are compouted by summing the orbital-resolved charge fluctuations
            into on-atom charges using the compute_charges function and a segment sum.
        """
        all_bsizes = feed['basis_sizes']
        loss_criterion = nn.MSELoss()
        total_computed, total_targets = list(), list()
        
        #Return a dictionary to keep track of the values predicted.
        prediction_dict = dict()
        
        for bsize in all_bsizes:
            real_charges = feed['charges'][bsize]
            curr_dQ_out = output['dQ'][bsize]
            curr_ids = feed['atom_ids'][bsize]
            computed_charges = compute_charges(curr_dQ_out, curr_ids)
            
            prediction_dict[bsize] = [elem.detach().cpu().numpy() for elem in computed_charges]
            
            assert(len(computed_charges) == len(real_charges)) #Both are lists now since raggedness
            for i in range(len(computed_charges)):
                total_computed.append(computed_charges[i])
                tensor_reals = torch.tensor(real_charges[i], dtype = computed_charges[i].dtype,
                                            device = computed_charges[i].device)
                total_targets.append(tensor_reals)
                assert (total_computed[-1].shape == total_targets[-1].shape)
        computed_tensor = torch.cat(total_computed)
        target_tensor = torch.cat(total_targets)
        return torch.sqrt(loss_criterion(computed_tensor, target_tensor)), prediction_dict