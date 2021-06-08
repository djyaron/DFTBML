# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 19:17:13 2021

@author: fhu14
"""
#%% Imports, definitions
from .base_classes import LossModel
from .external_funcs import compute_charges
from typing import List, Dict
import numpy as np
import re
import torch
Tensor = torch.Tensor
import torch.nn as nn

#%% Code behind

class DipoleLoss(LossModel):
    
    def __init__(self):
        pass
    
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
    
    def get_value(self, output: Dict, feed: Dict, rep_method: str) -> Tensor:
        r"""Computes the penalty for the dipoles
        
        Arguments:
            output (Dict): Output from the DFTB layer
            feed (Dict): The original feed into the DFTB layer
            rep_method (str): The repulsive method, 'new' for built-in splines
                and 'old' for DFTBrepulsive splines. Has no effect here.
        
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
            curr_charges = compute_charges(curr_dQ, curr_ids)
            assert(len(curr_charges) == len(curr_dipmats) == len(real_dipoles[bsize]))
            for i in range(len(curr_charges)):
                #Make sure the cartesian matrix has the same device and datatype
                #   as the charges.
                cart_mat = torch.tensor(curr_dipmats[i], dtype = curr_charges[i].dtype,
                                        device = curr_charges[i].device)
                #dipoles computed as coords @ charges
                comp_res = torch.matmul(cart_mat, curr_charges[i])
                computed_dips.append(comp_res)
                real_dips.append(real_dipoles[bsize][i])
        total_comp_dips = torch.cat(computed_dips)
        total_real_dips = torch.cat(real_dips)
        return torch.sqrt(loss_criterion(total_comp_dips, total_real_dips))        