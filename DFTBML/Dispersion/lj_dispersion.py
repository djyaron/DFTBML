# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 16:21:33 2021

@author: fhu14

Module handling the implementation of the Lennard-Jones dispersion
correction.

The form for the total energy is now defined as follows:
    
    E_tot = E_elec + E_rep + E_ref + E_disp

Where the electronic energy (E_elec) comes from the DFTB layer calculations,
E_rep comes from the spline form from either the skf files or from 
DFTBrepulsive, E_ref is a linear model in the number of atoms, and 
E_disp is computed from this module.

The Lennard-Jones dispersion model is defined as follows [1]:
    
    U_{i,j}(r) = d_{i,j} [-2 (r_{i,j}/r)^6 + (r_{i,j} / r) ^ 12]

Where d_{i,j} = (d_i * d_j) ^ 0.5 and r_{i,j} = (r_i * r_j)  ^ 0.5, and
r is the distance between atoms i and j. 

Due to the different power terms within the expression, computation of this
potential cannot be reduced to a linear problem, and thus cannot be 
turned into a matrix multiply.

The parameters being optimized are the VDW distance and well depth on a per
atom basis, and the starting values come from the original UFF paper [2].

References:
    [1] L. Zhechkov, Th. Heine, S. Patchkovskii, G. Seifert, and H. A. Duarte. An efficient a posteriori
    treatment for dispersion interaction in density-functional-based tight binding. J. Chem. Theory
    Comput., 1:841–847, 2005.
    
    [2] Rappe´, A. K.; Casewit, C. J.; Colwell, K. S.; Goddard, W.
    A., III; Skiff, W. M. J. Am. Chem. Soc. 1992, 114, 10024-10035.

TODO:
    1) As an optimization, the pairwise distances for each geometry could
        be stored in the feed via a get_feed() method perhaps?
    2) Can also precompute the geometric means and have those be the true 
        variables. Since the geom mean is differentiable, optimizing the 
        geometric mean indirectly optimizes the elemental parameters. (X) 
        
        In this case, we are optimizing parameters on a pairwise element basis.
    3) Introduce another Model (namedtuple) with 'D' as the oper for dispersion!
    4) It is likely the case that the value of r_0 for when VDW becomes attractive
        differs for element pair, so going to allow for input specification
        of r_0 values per element pair.
"""

#%% Imports, definitions
from .lj_dispersion_parameters import VDW_dists, VDW_well
from typing import Dict, Union
from .util import torch_geom_mean, np_geom_mean
import torch
from scipy.spatial.distance import pdist
from functools import reduce
import itertools
import numpy as np
Array = np.ndarray
Tensor = torch.Tensor


#%% Code behind

class LJ_Dispersion:
    
    def __init__(self, device: torch.device, dtype: torch.dtype) -> None:
        r"""Constructor for the Dispersion class
        
        Arguments:
            device (torch.device): The device to use for tensors.
            dtype (torch.dtype): The dtype to use for tensors.
        
        Returns: 
            None
        """
        #Load the values in 
        self.dist_vars = VDW_dists
        self.well_vars = VDW_well
        self.device = device
        self.dtype = dtype
        
        dist_keys = sorted(list(self.dist_vars.keys()))
        well_keys = sorted(list(self.well_vars.keys()))
        
        assert(dist_keys == well_keys) #Should be the same
        unique_elem_combos = list(itertools.combinations(dist_keys, 2))
        non_unique_elem_combos = [(x, x) for x in dist_keys] #need to account for repeats
        
        d_ij = dict()
        r_ij = dict()
        
        for elem1, elem2 in unique_elem_combos + non_unique_elem_combos:
            r_ij[(elem1, elem2)] = np_geom_mean([self.dist_vars[elem1], self.dist_vars[elem2]])
            d_ij[(elem1, elem2)] = np_geom_mean([self.well_vars[elem1], self.well_vars[elem2]])
        
        self.r_ij = r_ij
        self.d_ij = d_ij
        
        self.var_correction(self.r_ij)
        self.var_correction(self.d_ij)
            
            
    def var_correction(self, dict_to_correct: Dict) -> None:
        r"""Corrects a dictionary into a dictionary mapped to tensors
        
        Arguments:
            dict_to_correct (Dict): The dictionary to correct
        
        Returns:
            None
        
        Notes: dtype and device are assumed to be established attributes of self.
        """
        for key, value in dict_to_correct.items():
            dict_to_correct[key] = torch.tensor(value, dtype = self.dtype, 
                                                device = self.device, requires_grad = True)
    
    def compute_cutoff(self, elems: tuple) -> Tensor:
        r"""Computes the cutoff value based on the pairwise element
            parameter r_ij
        
        Arguments:
            elems (tuple): The element pair whose cutoff needs to be computed
        
        Returns:
            cutoff (float): The cutoff value as a floating point number
        
        Notes: The cutoff value r_0 is defined as r_0 := 2^(-1/6) * r_ij for 
            any two elements i and j [1]. Thus, the cutoff is 
            element pair dependent and computed on a case-by-case basis. This
            restriction on the value of r_0 also ensures that only 
            meaningful interactions are computed for the atoms, and that the 
            magnitude of the term r_ij / r for r > r_0 is not major, as the 
            maximal value (for when r apx equals r_0) of r_ij / r is 2^(1/6) or
            ~1.122. 
        """
        curr_rij = self.r_ij[elems] if elems in self.r_ij else self.r_ij[(elems[-1], elems[0])]
        cutoff = (2**(-1/6)) * curr_rij
        return cutoff
    
    def get_variables(self):
        r"""This method returns the variables of the dispersion model,
            and should not be invokved until after __init__() runs. 
            
            Because of the nature of the dispersion model, addition of the 
            variables will have to be done separately.
        """
        return self.r_ij, self.d_ij

    def lj_dispersion(self, elems: tuple, r: float) -> Tensor:
        r"""Performs the calculation for the dispersion energy according to the
            lennard jones functional form.
        
        Arguments:
            elems (tuple): The elements to use for the calculation.
            r (float): The distance between the two atoms.
        
        Returns:
            disp (float): The dispersion energy.
            
        Notes: See explanation at top of file for how dispersion is calculated.
        """
        
        r_ij = self.r_ij[elems] if elems in self.r_ij else self.r_ij[(elems[-1], elems[0])]
        d_ij = self.d_ij[elems] if elems in self.d_ij else self.d_ij[(elems[-1], elems[0])]
        
        term_6 = torch.pow(torch.div(r_ij, r), 6)
        term_12 = torch.pow(torch.div(r_ij, r), 12)
        
        disp = d_ij * ((-2 * term_6) + (term_12))
        
        return disp
        
    def calc_disp_energy(self, atomic_nums: Array, rcart: Array) -> Tensor:
        """Performs the calculation for the dispersion energy on a single
            geometry.
        
        Arguments: 
            atomic_nums (Array): The atomic numbers, vector of shape (Natom,)s
            rcart (Array): The coordinates in angstroms, matrix of shape 
                (Natom, 3)
        
        Returns:
            0th dimensional Tensor (A scalar) of the dispersion energy.
        """
        assert(atomic_nums.shape[0] == rcart.shape[0])
        atom_num_combo = list(itertools.combinations(atomic_nums, 2))
        pairwise_dists = pdist(rcart) #Uses L2 norm by default, distances are in Angstroms
        assert(len(pairwise_dists) == len(atom_num_combo))
        # dispersion_energies = [ self.lj_dispersion(combo, dist) for combo, dist 
        #                        in enumerate(zip(atom_num_combo, pairwise_dists)) if (dist >= 
        #                        self.cutoff if (not isinstance(self.cutoff, dict)) else self.get_cutoff_for_pair(combo)) ]
        dispersion_energies = []
        for _, comb_dist in enumerate(zip(atom_num_combo, pairwise_dists)):
            combo, dist = comb_dist
            cutoff_val = self.compute_cutoff(combo)
            valid_dist = dist >= cutoff_val
            if valid_dist:
                dispersion_energies.append(self.lj_dispersion(combo, dist))
            pass
        if len(dispersion_energies) > 0: 
            return reduce(lambda x, y : x + y, dispersion_energies)
        else: 
            return torch.tensor(0.0, dtype = self.dtype, device = self.device)
    
    
    def get_disp_energy(self, feed: Dict) -> Tensor:
        r"""Computes the dispersion energy based off the given feed.
        
        Arguments:
            feed (Dict): The feed dictionary to generate dispersion energies for
        
        Returns:
            disp_dict (Dict): The tensor of calculated dispersion energies.
        """
        all_bsizes = feed['basis_sizes']
        disp_dict = dict()
        for bsize in all_bsizes:
            glabels = feed['glabels'][bsize]
            curr_geoms = [feed['geoms'][x] for x in glabels]
            # Perform the transposition of the rcart matrix so it's of dimension (Natom, 3)
            raw_disps = [self.calc_disp_energy(geom.z, geom.rcart.T) for geom in curr_geoms]
            disp_dict[bsize] = torch.stack(raw_disps)
        return disp_dict