# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 19:02:00 2021

@author: fhu14
"""
#%% Imports, definitions
from constants import Model, RawData
from typing import List, Dict
import numpy as np
Array = np.ndarray
from Spline import JoinedSplineModel, get_dftb_vals
import torch
Tensor = torch.Tensor

#%% Code behind

class Input_layer_pairwise_linear_joined:

    def __init__(self, model: Model, pairwise_linear_model: JoinedSplineModel, par_dict: Dict,
                 cutoff: float, device: torch.device, dtype: torch.dtype,
                 inflection_point_var: List[float] = [], ngrid: int = 100, 
                 noise_magnitude: float = 0.0) -> None:
        r"""Interface for a joined spline model with flexible and infleixble regions.
        
        Arguments:
            model (Model): A named tuple of the form ('oper', 'Zs', 'orb'), where
                'oper' is the operater the model is modelling represented as a string
                (e.g. 'G', 'H', 'R'), 'Zs' is a tuple of the atomic numbers that are needed
                (e.g. (1,)), and 'orb' is a string representing the orbitals being considered
                (e.g. 'ss' for two s-orbital interactions)
            pairwise_linear_model (JoinedSplineModel): An instance of the JoinedSplineModel
                class for handling joined splines
            par_dict (Dict): Dictionary of the DFTB Slater-Koster parameters for atomic interactions 
                between different elements, indexed by a string 'elem1-elem2'. For example, the
                Carbon-Carbon interaction is accessed using the key 'C-C'
            cutoff (float): The cutoff distance for the joined spline
            device (torch.device): The device to run the computations on (CPU vs GPU).
                If running on GPU, must be CUDA enabled GPU.
            dtype (torch.dtype): The torch datatype for the calculations
            inflection_point_var (list[float]): The list of length 1 containing the 
                variable used to compute the inflection point. Defaults to [], in which case
                there is no inflection point.
            ngrid (int): The number of points for initially fitting the model to the DFTB
                parameters. Defaults to 100
            noise_magnitude (float): Factor to distort the DFTB-initialized value by. Can be used
                to test the effectiveness of the training by introducing artificial error. Defaults
                to 0.0
        
        Returns:
            None
        
        Notes: For a joined spline, a cutoff distance r_0 is given. The spline functional
            form is then allowed to change for distances r_i < r_0, but not for distances
            r_j > r_0. The coefficients are also partitioned into two sections, coeffs and c_fixed.
            To generate a prediction from the joined spline, we first perform a merge operation
            to generate the necessary matrix A and vector b. Then, the predictions are generated as 
            y = concatenate(coeffs, c_fixed) + b. Because we only want the model to train the variable
            portion of the spline, only the vector for coeffs is converted to a PyTorch tensor that 
            is optimizable.
            
            The inflection point variable is optional, and is only used when the model in question 
            has a very strongly defined inflection point (commonly seen among models of the overlap operator S).
            This variable is returned separately from the normal coefficients of the model, and is only 
            used internally for the calculation of the convex/monotonic penalties.
        """
        self.dtype = dtype
        self.device = device
        self.model = model
        self.pairwise_linear_model = pairwise_linear_model
        (rlow, rhigh) = pairwise_linear_model.r_range()
        rgrid = np.linspace(rlow, rhigh, ngrid)
        ygrid = get_dftb_vals(model, par_dict, rgrid)
        ygrid = ygrid + noise_magnitude * np.random.randn(len(ygrid))
        # fig, axs = plt.subplots()
        # axs.scatter(rgrid, ygrid)
        # axs.set_title(f"{model}")
        # plt.show()
        variable_vars, fixed_vars = pairwise_linear_model.fit_model(rgrid, ygrid)
        #Initialize the optimizable torch tensor for the variable coefficients
        # of the spline and the fixed part that's cat'd on each time
        self.variables = torch.tensor(variable_vars, dtype = self.dtype, device = self.device)
        self.variables.requires_grad = True
        self.constant_coefs = torch.tensor(fixed_vars, dtype = self.dtype, device = self.device)
        self.joined = True #A flag used by later functions to identify joined splines
        self.cutoff = cutoff #Used later for outputting skf files
        if len(inflection_point_var) == 1:
            self.inflection_point_var = torch.tensor(inflection_point_var, dtype = self.dtype, device = self.device)
            self.inflection_point_var.requires_grad = True #Requires gradient
        else:
            self.inflection_point_var = None
        
    def get_variables(self) -> Tensor:
        r"""Returns the trainable coefficients for the given joined spline
        
        Arguments:
            None
        
        Returns:
            self.variables (Tensor): The trainable variables for this model 
                as a PyTorch tensor object with gradients enabled.
        
        Notes: Only coeffs is returned, not c_fixed
        """
        return self.variables

    def get_fixed(self) -> Tensor:
        r"""Returns the non-trainable coefficients for the given joined spline
        
        Arguments:
            None
        
        Returns:
            self.constant_coefs (Tensor): The non-trainable coefficients for this model
                as a PyTorch tensor object without gradients enabled.
        
        Notes: None
        """
        return self.constant_coefs

    def get_total(self) -> Tensor:
        r"""Returns the total coefficient vector for the joined spline
        
        Arguments:
            None
        
        Returns:
            total coefficients (Tensor): A tensor of the coeffs and c_fixed 
                concatenated together.
        
        Notes: Because coeffs has gradients enabled, the total coefficient 
            tensor will also have gradients enabled.
        """
        return torch.cat([self.variables, self.constant_coefs])
    
    def get_inflection_pt(self) -> Tensor:
        r"""Returns the inflection point variable if there is one created
        
        Arguments:
            None
        
        Returns:
            inflec_var (Tensor): The variable tensor used to compute the location
                of the inflection point
                
        Note: In the case of there not being an inflection point variable, the
            NoneType is returned instead
        """
        return self.inflection_point_var
    
    def set_inflection_pt(self, value: List[float]) -> None:
        r"""Sets the inflection point variable value for the given model
        
        Arguments:
            value (List[float]): A 1 element list containing the value for the inflection point variable
        
        Returns:
            None
        """
        if len(value) == 1:
            self.inflection_point_var = torch.tensor(value, dtype = self.dtype, device = self.device)
            self.inflection_point_var.requires_grad = True
    
    def get_feed(self, mod_raw: List[RawData]) -> Dict:
        r"""Returns the necessary information for the feed dictionaries into the DFTB layer
        
        Arguments:
            mod_raw (List[RawData]): A list of RawData named tuples that contains the
                index, glabel, Zs, oper string, dftb value, and distance for each occurence
                of a given Model within the data. Used to determine the distances (xeval) that 
                the spline needs to be evaluated at, and this is used to generate the matrix A and vector b
                that is needed for generating predictions from the model.
        
        Returns:
            feed dictionary (Dict): A dictionary containing the matrix A and vector b needed by the given
                spline for generating a prediction. These are added to the feed dictionary for the 
                DFTB layer.
        
        Notes: The matrix A and vector b returned by this function in the dictionary
            are initally numpy arrays, but they are converted to PyTorch tensors
            later on. The matrix A and vector b requires a spline merge operation under the hood.
        """
        xeval = np.array([elem.rdist for elem in mod_raw])
        A, b = self.pairwise_linear_model.linear_model(xeval)
        return {'A' : A, 'b' : b}

    def get_values(self, feed: Dict) -> Tensor:
        r"""Generates a prediction from the joined spline.
        
        Arguments: 
            feed (Dictionary): The dictionary containing the matrix A and vector b needed
                for generating the predictions
        
        Returns:
            result (Tensor): PyTorch tensor containing the predictions from the spline
        
        Notes: For a joined spline, the predictions are computed as 
        
            y = (A @ cat(coeffs, c_fixed)) + b
        
            Where coeffs are the trainable coefficients, c_fixed are the fixed coefficients. 
            Cat is the concatenation operation to generate the total coefficient Tensor.
        """
        A = feed['A']
        b = feed['b']
        total_var_tensor = torch.cat([self.variables, self.constant_coefs])
        result = torch.matmul(A, total_var_tensor) + b
        return result