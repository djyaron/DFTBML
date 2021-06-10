# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 18:56:50 2021

@author: fhu14
"""
#%% Imports, definitions
from MasterConstants import Model, RawData
from typing import List, Dict
import numpy as np
Array = np.ndarray

#%% Code behind

class Input_layer_DFTB_val:
    
    def __init__(self, model: Model):
        r"""DEBUGGING interface for models predicting on-diagonal elements
        
        Arguments:
            model (Model): A named tuple of the form ('oper', 'Zs', 'orb'), where
                'oper' is the operater the model is modelling represented as a string
                (e.g. 'G', 'H', 'R'), 'Zs' is a tuple of the atomic number that is needed
                (e.g. (1,)), and 'orb' is a string representing the orbitals being considered
                (e.g. 'ss' for two s-orbital interactions)
        
        Returns:
            None
        
        Notes: This model just takes the mod_raw value for the single element operator.
        """
        self.model = model
        if len(model.Zs) > 1:
            raise ValueError("On-diagonals consist of single-element interactions")
    
    def get_variables (self) -> List:
        r"""Dummy method to respect the model interface; there are no
            variables for this model
        """
        return []
    
    def get_feed(self, mod_raw: List[RawData]) -> Dict[str, Array]:
        r"""Generates the elements for this model that need to be included in the
            feed
        """
        return {'values' : np.array([x.dftb for x in mod_raw])}
    
    def get_values(self, feed: Dict) -> Array:
        r"""Extracts the elements from the feed
        """
        return feed['values']