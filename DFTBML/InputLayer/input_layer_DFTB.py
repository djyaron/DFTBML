# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 18:54:37 2021

@author: fhu14
"""
#%% Imports, definitions
from MasterConstants import Model, RawData
from typing import List, Dict
import numpy as np
Array = np.ndarray

#%% Code behind

class Input_layer_DFTB:

    def __init__(self, model: Model) -> None:
        r"""Initializes a debugging model that just uses the DFTB values rather than
            spline interpolations.
        
        Arguments:
            model (Model): Named tuple describing the interaction being modeled
        
        Returns:
            None
        
        Notes: This interface is mostly a debugging tool
        """
        self.model = model

    def get_variables(self) -> List:
        r"""Returns variables for the model
        
        Arguments:
            None
        
        Returns:
            [] (List): Empty list
        
        Notes: There are no variables for this model.
        """
        return []
    
    def get_feed(self, mod_raw: List[RawData]) -> Dict:
        r"""Returns the necessary values for the feed dictionary
        
        Arguments:
            mod_raw (List[RawData]): List of RawData tuples from the feed dictionary
        
        Returns:
            value dict (Dict): The dftb values extracted from the mod_raw list
        
        Notes: None
        """
        return {'values' : np.array([x.dftb for x in mod_raw])}

    def get_values(self, feed: Dict) -> Array:
        r"""Generates a prediction from this model
        
        Arguments:
            feed (Dict): The dictionary containing the needed values
        
        Returns:
            feed['values'] (Array): Numpy array of the predictions (just the original values)
        
        Notes: None
        """
        return feed['values']