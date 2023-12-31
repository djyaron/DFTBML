# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 18:33:29 2021

@author: fhu14
"""

#%% Imports, definitions
from typing import List, Dict, Union
from Spline import get_dftb_vals, spline_linear_model
import numpy as np
from MasterConstants import Model, Bcond
Array = np.ndarray
import torch
Tensor = torch.Tensor
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

#%% Code behind

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
        
        The boundary conditions here are natural, i.e. second derivative goes to zero at both end points.
    """
    model_spline_dict = dict()
    for model in all_models:
        try:
            #Only two-body H, R, S
            # THE TRY AND EXCEPT IS EXCLUDING THE G MODELS
            if (model.oper not in op_ignore) and (len(model.Zs) == 2):
                pairwise_lin = all_models[model]
                r_low, r_high = pairwise_lin.pairwise_linear_model.r_range()
                rgrid = np.linspace(r_low, r_high, ngrid) 
                ygrid = get_dftb_vals(model, par_dict, rgrid)
                model_spline_dict[model] = (spline_linear_model(rgrid, None, (rgrid, ygrid), bcond), rgrid, ygrid)
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
            then set to True; otherwise, set to False (concave up)
    
    Notes: The concavity of the model is deteremined by the sign of the
        average value of the predictions of the second derivative. If the 
        average value is 0, then this means the spline is flat and has
        no curvature. In this case, the spline is not optimized since there
        is no interaction
    """
    concavity_dict = dict()
    for model_spec in model_spline_dict:
        mod_dict = model_spline_dict[model_spec][0]
        y_vals_deriv1 = np.dot(mod_dict['X'][1], mod_dict['coefs']) + mod_dict['const'][1]
        y_vals_deriv0 = np.dot(mod_dict['X'][0], mod_dict['coefs']) + mod_dict['const'][0]
        #Assume also that the mean will take care of it, and that the mean should not be 0
        concavity_deriv1 = np.sign(np.mean(y_vals_deriv1))
        concavity_deriv0 = np.sign(np.mean(y_vals_deriv0))
        #Positive mean value means it should be concave up
        if (concavity_deriv1 == 1) and (concavity_deriv0 == -1):
            #Concave down should have negative values and positive first derivative
            concavity_dict[model_spec] = True
        elif (concavity_deriv1 == -1) and (concavity_deriv0 == 1):
            #Concave up should have positive values and negative first derivative
            concavity_dict[model_spec] = False
        elif (concavity_deriv1 == 0) and (concavity_deriv0 == 0):
            #Flat splines with no value other than 0 are ignored by optimization
            print(f"Zero concavity detected for {model_spec}")
        elif concavity_deriv1 == concavity_deriv0:
            #Should not happen
            print(f"Concavity mismatch for {model_spec}")
    return concavity_dict

def generate_third_deriv_dict(concavity_dict: Dict) -> Dict:
    r"""Generates the dictionary specifying the correct sign of the third derivatives
    
    Arguments:
        concavity_dict (Dict): The dictionary of the concavities, i.e. signs of the 
            second derivatives
    
    Returns:
        third_deriv_dict (Dict): The dictionary of the third derivative, indicating 
            which sign the third derivative should be for each model_spec
    
    Notes:
        For the simpler cases, the third derivative should be the opposite sign of the 
        second derivative. As per the specification of generate_concavity_dict,a value of 
        True corresponds to a concave down function (negative second derivative) and a 
        value of False corresponds to a concave up function (positive second derivative).
        Thus, if v'' < 0 (True), we note that v''' > 0 ('pos'), while if v'' > 0 (False), we note that
        v''' < 0 ('neg')
    """
    third_model_deriv_dict = dict()
    for model_spec in concavity_dict:
        if concavity_dict[model_spec] == True:
            third_model_deriv_dict[model_spec] = 'pos'
        if concavity_dict[model_spec] == False:
            third_model_deriv_dict[model_spec] = 'neg'
    return third_model_deriv_dict

def compute_charges(dQs: Union[Array, Tensor], ids: Union[Array, Tensor]) -> List[Tensor]:
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
            temp = torch.zeros(int(curr_ids[-1].item()) + 1, dtype = curr_charges.dtype,
                               device = curr_charges.device)
            temp = temp.scatter_add(0, curr_ids.long(), curr_charges)
            charge_tensors.append(temp)
        return charge_tensors
