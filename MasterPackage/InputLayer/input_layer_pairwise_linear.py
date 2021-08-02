# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 19:00:06 2021

@author: fhu14
"""
#%% Imports, definitions
from MasterConstants import Model, RawData
from typing import List, Dict
import numpy as np
Array = np.ndarray
from Spline import SplineModel, get_dftb_vals, fit_linear_model
import torch
Tensor = torch.Tensor

#%% Code behind

class Input_layer_pairwise_linear:

    def __init__(self, model: Model, pairwise_linear_model: SplineModel, par_dict: Dict,
                 cutoff: float, device: torch.device, dtype: torch.dtype,
                 inflection_point_var: List[float] = [], ngrid: int = 100, 
                 noise_magnitude: float = 0.0) -> None:
        r"""Creates a cubic spline model that is allowed to vary over the entire spanned distance
        
        Arguments:
            model (Model): A named tuple of the form ('oper', 'Zs', 'orb'), where
                'oper' is the operater the model is modelling represented as a string
                (e.g. 'G', 'H', 'R'), 'Zs' is a tuple of the atomic numbers that are needed
                (e.g. (1,)), and 'orb' is a string representing the orbitals being considered
                (e.g. 'ss' for two s-orbital interactions)
            pairwise_linear_model (SplineModel): An instance of the SplineModel from
                SplineModel_v3.py, used for managing cubic splines that vary
                over the entire distance
            par_dict (Dict): Dictionary of the DFTB Slater-Koster parameters for atomic interactions 
                between different elements, indexed by a string 'elem1-elem2'. For example, the
                Carbon-Carbon interaction is accessed using the key 'C-C'
            cutoff (float): The distance in angstroms above which all predicted 
                values are set to 0.
            device (torch.device): The device to run the computations on (CPU vs GPU).
                If running on GPU, must be CUDA enabled GPU.
            dtype (torch.dtype): The torch datatype for the calculations
            inflection_point_var (List[float]): The variable value used to compute the 
                inflection point for the model. Defaults to []
            ngrid: (int): The number of points for initially fitting the model to the DFTB
                parameters. Defaults to 100
            noise_magnitude (float): Factor to distort the DFTB-initialized value by. Can be used
                to test the effectiveness of the training by introducing artificial error. Defaults
                to 0.0
        
        Returns:
            None
        
        Notes: The model is initialized to DFTB values by a least squares fit, which 
            is solved by the fit_linear_model function. Getting the predictions from the spline
            is done using the equation
            
            y = Ax + b
            
            where x is the coefficient vector. The least squares problem is solved for the vector
            x. Once the coefficients are obtained, they are converted to a PyTorch tensor and
            their gradients are initialized to allow training.
        """
        self.model = model
        self.pairwise_linear_model= pairwise_linear_model
        self.cutoff = cutoff
        self.dtype = dtype
        self.device = device
        (rlow,rhigh) = pairwise_linear_model.r_range()
        ngrid = 1000 #Number of grid points used to fit the initial variables
        rgrid = np.linspace(rlow,rhigh,ngrid) #This is in angstroms
        ygrid = get_dftb_vals(model, par_dict, rgrid)
        ygrid = ygrid + noise_magnitude * np.random.randn(len(ygrid))
        model_vars, A, b = fit_linear_model(self.pairwise_linear_model, rgrid,ygrid) #Model vars fit based on angstrom x-axis
        
        ypred = np.dot(A, model_vars) + b
        y_diff = np.abs(ypred - ygrid)
        print(f"Maximum absolute difference (kcal/mol): {np.max(y_diff) * 627}")
        print(f"MAE difference (kcal/mol): {np.mean(y_diff) * 627}")
        
        ### TESTING CODE, REMOVE THIS CONDITIONAL LATER! INITIALIZING REPULSIVE
        ### VARIABLES TO A VECTOR OF 0'S
        # if (self.model.oper == 'R'):
        #     self.variables = torch.zeros(len(model_vars), dtype = self.dtype, device = self.device)
        #     print(f"Setting coefficient vector to zero for model {self.model}")
        # else:
        self.variables = torch.tensor(model_vars, dtype = self.dtype, device = self.device)
        ### END TESTING CODE
        
        self.variables.requires_grad = True
        if len(inflection_point_var) == 1:
            self.inflection_point_var = torch.tensor(inflection_point_var, dtype = self.dtype, device = self.device)
            self.inflection_point_var.requires_grad = True
        else:
            self.inflection_point_var = None

    def get_variables(self) -> Tensor:
        r"""Returns the coefficient vector for the spline model.
        
        Arguments:
            None
        
        Returns:
            self.variables (Tensor): The trainable variables for this model 
                as a PyTorch tensor object with gradients enabled.
        
        Notes: The same variable tensor can be used for evaluating the spline
            at any derivative. However, the matrix A and vector b in y = Ax + b must
            be recomputed for each derivative.
        """
        return self.variables
    
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
            self.inflection_point_var = torch.tensor(value, device = self.device, dtype = self.dtype)
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
            later on.
        
        TODO: Handle edge case of no non-zero values!
        """
        xeval = np.array([elem.rdist for elem in mod_raw]) #xeval are in angstroms
        nval = len(xeval)
        izero = np.where(xeval > self.cutoff)[0]
        inonzero = np.where(xeval <= self.cutoff)[0]
        xnonzero = xeval[inonzero] # Predict only on the non-zero x vals
        if len(inonzero) > 0:
            A,b = self.pairwise_linear_model.linear_model(xnonzero) #Computed based on angstrom values
        elif len(inonzero) == 0:
            A, b = None, None
        return {'A': A, 'b': b, 'nval' : nval, 'izero' : izero, 'inonzero' : inonzero, 
                'xeval' : xeval}
    
    def get_values(self, feed: Dict) -> Tensor:
        r"""Generates a prediction from the spline
        
        Arguments:
            feed (Dict): The dictionary that contains the matrix A and vector b that
                are needed for generating a prediction.
        
        Returns:
            result (Tensor): A torch tensor of the predicted values.
        
        Notes: Because we are concerned with the values and not derivatives,
            the values returned correspond with the 0th derivative. The prediction
            is generated as y = (A @ x) + b.
        """
        A = feed['A']
        b = feed['b']
        nval = feed['nval']
        izero = feed['izero'].long()
        inonzero = feed['inonzero'].long()
        if len(inonzero) == 0:
            # If all values are zero, just return zeros with double datatype
            return torch.zeros([nval], dtype = self.dtype, device = self.device)
        result_temp = torch.matmul(A, self.variables) + b #Comes from angstrom values
        result = torch.zeros([nval], dtype = self.dtype, device = self.device)
        result[inonzero] = result_temp
        result[izero] = 0.0
        return result