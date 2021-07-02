# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 17:34:14 2021

@author: fhu14
"""

#%% Imports, definitions
import numpy as np
import torch
Tensor = torch.Tensor

#%% Code behind
def torch_geom_mean(vals) -> Tensor:
    r"""Computes the geometric mean using torch functions to allow for 
        backpropagation differentiability
    
    Arguments
        vals (Iterable[Tensor]): A sequence of tensors used to compute
            the geometric mean. The tensors of the input should be 
            zero dimensional (scalars)
    
    Returns:
        The geometric mean.
    
    Notes: To ensure that the geometric mean is a differentiable operation, 
        calculation method is used:
            
            gmean = exp[mean(log(x))]
        
        where x is the concatenated input tensor to allow for vectorization
        without a for loop. The algorithm can be found here:
        
        https://stackoverflow.com/questions/59722983/how-to-calculate-geometric-mean-in-a-differentiable-way
        https://en.wikipedia.org/wiki/Geometric_mean#Relationship_with_logarithms
    """
    #torch.stack joins our 0 dimensional vectors (scalars) along a new axis
    concat_vec = torch.stack(vals)
    log_vec = torch.log(concat_vec)
    return torch.exp(torch.mean(log_vec))