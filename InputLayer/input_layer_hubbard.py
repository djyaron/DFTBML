# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 19:02:56 2021

@author: fhu14
"""
#%% Imports, definitions
from constants import Model, RawData
from typing import List, Dict
import numpy as np
Array = np.ndarray
import torch
Tensor = torch.Tensor

#%% Code behind

class Input_layer_hubbard:
    
    def __init__(self, model: Model, model_variables: Dict, device: torch.device, dtype: torch.dtype) -> None:
        r"""Initializes the off-diagonal model
        
        Arguments:
            model (Model): Named tuple describing the interaction to be modeled
            model_variables (Dict): Dictionary referencing all the variables of models
                being used
            device (torch.device): The device to run the computations on (CPU vs GPU).
                If running on GPU, must be CUDA enabled GPU.
            dtype (torch.dtype): The torch datatype for the calculations
        
        Returns:
            None
        
        Notes: The off diagonal model is used to construct all off-diagonal elements of the
            operator matrix from the on-diagonal elements. This approach will be primarily used
            for the G operator matrix using the _Gamma12() function provided in sccparam.py
            
            Initialization of this model requires initializing the on-diagonal elements of the matrix first, 
            such as the G diagonal element for C s or C p. Then, to get the off-digaonal element, 
            we do 
            
            G(C s| C p) (r) = _Gamma12(r, C s, C p)
            
            Where C s and C p are the two digaonal elements for the G operator matrix corresponding
            to the s-orbital interactions on C and p orbital interactions on C, respectively. The 
            distances r to evaluate this model at will be determined from the mod_raw data.
            
            Because the OffDiagModel uses the same variables as the diagonal models,
            it will not have its variables added to the model_variables dictionary.
        """
        if len(model.Zs) < 2: 
            return
        elem1, elem2 = model.Zs
        orb1, orb2 = model.orb[0], model.orb[1]
        oper = model.oper
        if oper == 'G':
            # Double the orbitals for a G operator
            orb1, orb2 = orb1 + orb1, orb2 + orb2
        mod1 = Model(oper, (elem1, ), orb1)
        mod2 = Model(oper, (elem2, ), orb2)
        
        # Use the created orbitals to index into the model variables and 
        # get the appropriate variables out
        elem1_var = model_variables[mod1]
        elem2_var = model_variables[mod2]
        # Keep references to the variables in a list
        self.variables = [elem1_var, elem2_var] 
        self.device = device
        self.dtype = dtype
    
    def _Expr(self, r12: Tensor, tau1: Tensor, tau2: Tensor) -> Tensor:
        r"""Computes expression for off-diagonal elements (between atoms)
        
        Arguments:
            r12 (Tensor): Tensor of non-zero distances to compute the elements for
            tau1 (Tensor): Computed as 3.2 * hub1
            tau2 (Tensor): Computed as 3.2 * hub2
        
        Returns:
            computed expression (Tensor)
        """
        sq1, sq2 = tau1**2, tau2**2
        sq1msq2 = sq1 - sq2
        quad2 = sq2**2
        termExp = torch.exp(-tau1 * r12)
        term1 = 0.5 * quad2 * tau1 / sq1msq2**2
        term2 = sq2**3 - 3.0 * quad2 * sq1
        term3 = r12 * sq1msq2**3
        return termExp * (term1 - term2 / term3)
    
    def get_variables(self) -> Tensor:
        return self.variables
    
    def get_feed(self, mod_raw: List[RawData]) -> Dict:
        r"""New method for grabbing distances for feed
        
        Arguments:
            mod_raw (List[RawData]): A list of RawData named tiples used to extract the 
                distances at which to evaluate model
        
        Returns:
            distances (Dict): A dictionary with three keys: 
                'zero_indices' (Array): Array of indices for distances less than a threshold
                    value of 1e-5
                'nonzero_indices' (Array): Complement indices of 'zero_indices'
                'nonzero_distances' (Array): The distances corresponding to 'nonzero_indices'
        """
        xeval = np.array([elem.rdist for elem in mod_raw])
        zero_indices = np.where(xeval < 1.0e-5)[0]
        nonzero_indices = np.where(xeval >= 1.0e-5)[0]
        assert( len(zero_indices) + len(nonzero_indices) == len(xeval))
        nonzero_distances = xeval[nonzero_indices]
        return {'zero_indices'      : zero_indices,
                'nonzero_indices'   : nonzero_indices,
                'nonzero_distances' : nonzero_distances,
                'xeval' : xeval} #Return the distances too for debugging purposes
    
    def get_values(self, feed: Dict) -> Tensor:
        r"""Obtain the predicted values in a more efficient way
        
        Arguments:
            feed (Dict): The dictionary containing the information for getting
                the value
        
        Returns:
            results (Tensor): The calculated results
        """
        zero_indices = feed['zero_indices'].long()
        nonzero_indices = feed['nonzero_indices'].long()
        r12 = feed['nonzero_distances'] * 1.889725989 #Multiply by ANGSTROM2BOHR to get correct values, need to verify why this is the case?
        nelements = len(zero_indices) + len(nonzero_indices)
        results = torch.zeros([nelements], dtype = self.dtype, device = self.device)
        
        hub1, hub2 = self.variables
        smallHubDiff = abs(hub1-hub2).item() < 0.3125e-5
        tau1 = 3.2 * hub1
        tau2 = 3.2 * hub2
        
        # G between shells on the same atom
        if len(zero_indices) > 0:
            if smallHubDiff:
                onatom = 0.5 * (hub1 + hub2)
            else:
                p12 = tau1 * tau2
                s12 = tau1 + tau2
                pOverS = p12 / s12
                onatom = 0.5 * ( pOverS + pOverS**2 / s12 )
            results[zero_indices] = onatom
        
        # G between atoms
        if len(nonzero_indices) > 0:
            if smallHubDiff:
                tauMean = 0.5 * (tau1 + tau2)
                termExp = torch.exp(-tauMean * r12)
                term1 = 1.0/r12 + 0.6875 * tauMean + 0.1875 * r12 * tauMean**2
                term2 = 0.02083333333333333333 * r12**2 * tauMean**3
                expr = termExp * (term1 + term2)        
            else:
                expr = self._Expr(r12, tau1, tau2) + self._Expr(r12, tau2, tau1)
            results[nonzero_indices] = 1.0 / r12 - expr                      
            
        return results