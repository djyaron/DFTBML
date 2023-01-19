# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 18:58:12 2021

@author: fhu14
"""
#%% Imports, definitions
from MasterConstants import Model, RawData
from typing import List, Dict
import numpy as np
Array = np.ndarray
import torch
Tensor = torch.Tensor
from Spline import get_dftb_vals

#%% Code behind

class Input_layer_value:
    
    def __init__(self, model: Model, device: torch.device, dtype: torch.dtype, initial_value: float = 0.0) -> None:
        r"""Interface for models predicting on-diagonal elements
        
        Arguments:
            model (Model): A named tuple of the form ('oper', 'Zs', 'orb'), where
                'oper' is the operater the model is modelling represented as a string
                (e.g. 'G', 'H', 'R'), 'Zs' is a tuple of the atomic number that is needed
                (e.g. (1,)), and 'orb' is a string representing the orbitals being considered
                (e.g. 'ss' for two s-orbital interactions)
            device (torch.device): The device to run the computations on (CPU vs GPU).
                If running on GPU, must be CUDA enabled GPU.
            dtype (torch.dtype): The torch datatype for the calculations
            initial_value (float): The starting value for the model. Defaults to 0.0
        
        Returns:
            None
        
        Notes: Because this model is only used to model on-diagonal elements of the 
            various operator matrices, this constructor is only called when the 
            number of atomic numbers is 1 (i.e. len(model.Zs) == 1). The variable tensor
            has requires_grad set to true so that the variable is trainable by the network later on.
        """
        self.model = model
        self.dtype = dtype
        self.device = device
        if not isinstance(initial_value, float):
            raise ValueError('Val_model not initialized to float')
        self.value = np.array([initial_value])
        self.variables = torch.tensor(self.value, device = self.device, dtype = self.dtype)
        self.variables.requires_grad = True

    def initialize_to_dftb(self, pardict: Dict, noise_magnitude: float = 0.0) -> None:
        r"""Initializes the value model parameter to DFTB values
        
        Arguments:
            pardict (Dict): Dictionary of the DFTB Slater-Koster parameters for atomic interactions 
                between different elements, indexed by a string 'elem1-elem2'. For example, the
                Carbon-Carbon interaction is accessed using the key 'C-C'
            noise_magnitude (float): Factor to distort the DFTB-initialized value by. Can be used
                to test the effectiveness of the training by introducing artificial error. Defaults
                to 0.0
        
        Returns:
            None
        
        Notes: This method updates the value for the value being held within the self.value field. 
        The reason we do not have to re-initialize the torch tensor for the variable is because 
        the torch tensor and the numpy array share the same underlying location in memory, so changing one 
        will change the other.
        
        Note: TORCH.TENSOR DOES NOT HAVE THE SAME MEMORY ALIASING BEHAVIOR AS TORCH.FROM_NUMPY!!
        """
        if self.model.oper == 'G':
            init_value, val, hub1, hub2 = get_dftb_vals(self.model, pardict)
        else:
            init_value = get_dftb_vals(self.model, pardict)
        if not noise_magnitude == 0.0:
            init_value = init_value + noise_magnitude * np.random.randn(1)
        if (self.model.oper == 'G') and (not (hub1 == hub2 == val)):
            print(self.model, hub1, hub2, val)
        self.value[0]= init_value
        self.variables = torch.tensor(self.value, device = self.device, dtype = self.dtype)
        self.variables.requires_grad = True

    def get_variables(self) -> Tensor:
        r"""Returns the trainable variables for this model as a PyTorch tensor.
        
        Arguments:
            None
        
        Returns:
            self.variables (Tensor): The trainable variables for this model 
                as a PyTorch tensor object with gradients enabled.
        
        Notes: None
        """
        return self.variables

    def get_feed(self, mod_raw: List[RawData]) -> Dict[str, int]:
        r"""Returns a dictionary indicating how to use the variable for this model.
        
        Arguments:
            mod_raw (List[RawData]): A list of RawData named tuples that contains the
                index, glabel, Zs, oper string, dftb value, and distance for each occurence
                of a given Model within the data. Used to determine how many times the variable for the
                value model is needed.
        
        Returns:
            feed dictionary (Dict): A dictionary indicating how many times the model's variable needs
                to be repeated in the initial input to the DFTB layer before the Slater-Koster rotations
                and gather/reshape operations.
        
        Notes: The number of times the variable is needed is equal to the number of times the model is
            used within the given batch.
        """
        return {'nval': len(mod_raw)}

    def get_values(self, feed: Dict) -> Tensor:
        r"""Returns the values necessary for the DFTB layer
        
        Arguments: 
            feed (Dict): The dictionary that indicates how many times to repeat the value
        
        Returns:
            result (Tensor): A tensor with the model value repeated the necessary number of times for 
                the initial layer for the gather/reshape operations to work properly in assembling the
                operator matrices.
        
        Notes: The number of times that the value needs to be repeated is determined by the number 
            of times the model appears in mod_raw.
        """
        result = self.variables.repeat(feed['nval'])
        return result