# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 19:03:43 2021

@author: fhu14
"""
#%% Imports, definitions
from typing import List, Dict
import numpy as np
Array = np.ndarray
import torch
Tensor = torch.Tensor
from copy import deepcopy

#%% Code behind

class Reference_energy:

    def __init__(self, allowed_Zs: List[int], device: torch.device, dtype: torch.dtype,
                 prev_values: List[float] = None) -> None:
        r"""Initializes the reference energy model.
        
        Arguments:
            allowed_Zs (List[int]): List of the allowed atomic numbers
            device (torch.device): The device to run the computations on (CPU vs GPU).
                If running on GPU, must be CUDA enabled GPU.
            dtype (torch.dtype): The torch datatype for the calculations
            prev_values (List[float]): Previous values for the reference energy to start from.
                Defaults to None
        
        Returns:
            None
        
        Notes: The reference energy is computed for each element type, and is used
            as a term in computing the total energy. For calculating Etot, the total
            energy is computed as
            
            Etot = Eelec + Erep + Eref
            
            Where Eelec, Erep, and Eref are the electronic, repulsive, and reference 
            energies respectively. The reference energy values are all initialized to 0,
            and the tensor representing the reference energies has a required gradient as
            they are trainable.
            
            To compute the reference energy contribution, for each basis size,
            we do feed[zcounts][bsize] @ self.variables where feed[zcounts][bsize]
            will be a (ngeom, natom) matrix consisting of the molecules of that 
            basis size with the atom counts sorted from lowest to highest atomic number,
            and self.variables is a (natom, 1) vector of the reference energy variables.
            This gives a vector of (ngeom, 1) with the reference energy terms for each 
            variable. natom here does not mean the number of atoms in the molecule, but the
            number of unique atom types across all molecules in the data.
            
            An additional constant needs to be added since the reference energy 
            contains an additional constant term. Will work on adding this in, so that the 
            reference energy is computed as 
            
            Eref = Sum_z[C_z * N_z] + C_0, where the sum goes over all atom types z in the 
            dataset, C_z is the coefficient for element z, N_z is the number of that element 
            in the molecule, and C_0 is the additional coefficient.
        """
        self.dtype, self.device = dtype, device
        self.allowed_Zs = np.sort(np.array(allowed_Zs))
        self.values = np.zeros(self.allowed_Zs.shape)
        # Add in the constant term
        self.values = np.append(self.values, np.array([0]))
        if (not (prev_values is None)) and  len(prev_values) > 0:
            #Load previous values if they are given
            #FOR DEBUGGING PURPOSES ONLY
            assert(len(prev_values) == len(self.values))
            self.values = np.array(prev_values)
        self.variables = torch.tensor(self.values, dtype = self.dtype, device = self.device)
        self.variables.requires_grad = True

    def get_variables(self) -> Tensor:
        r"""Returns the trainable variables for the reference energy.
        
        Arguments:
            None
        
        Returns:
            self.variables (Tensor): The tensor of the trainable reference
                energy variables, with gradients enabled.
        
        Notes: None
        """
        return self.variables
    
    def get_ref_ener_info(self) -> Dict:
        r"""Returns the reference energy information from the model as a 
            dictionary
        
        Arguments:
            None
        
        Returns: 
            ref_dict (Dict): A dictionary that contains the coefficients,
                intercept, and atype ordering.
        
        Notes: The return type of this function should have the same structure as 
            the dictionary returned by the new repulsive model. The dictionary 
            should have the following keys:
                'coef' : Coefficients for the atoms
                'intercept' : The constant term
                'atype_ordering' : The ordering of the atomic numbers (should be sorted)
        """
        assert(len(self.variables) == len(self.allowed_Zs) + 1)
        numpy_variables = deepcopy(self.variables.detach().numpy())
        coeffs = numpy_variables[:len(self.allowed_Zs)]
        intercept = np.array([numpy_variables[len(self.allowed_Zs)]])
        atype_ordering = tuple(self.allowed_Zs)
        ref_dict = {
            'coef' : coeffs,
            'intercept' : intercept,
            'atype_ordering' : atype_ordering
            }
        return ref_dict
        