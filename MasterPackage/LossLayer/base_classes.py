# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 18:49:29 2021

@author: fhu14
"""
#%% Imports, definitions
import numpy as np
Array = np.ndarray
import torch
Tensor = torch.Tensor
import math
from typing import List, Dict

#%% Code behind

'''
This adaptation for the penalties comes from solver.py
which is a part of the DFTBrepulsive project. The file is included in this repo,
but only use it as a reference

Do a generic class here to get all the different penalties

For this version of the penalties, working with the pairwise linear interface
defined in dftb_layer_splines_3.py

Note: torch.einsum is used here in place of np.einsum because only torch.einsum 
works on tensors that require gradients, and the gradients need to be maintained throughout
forward prop for back prop

TODO: Figure out system for classifying if model should concave up or concave down
    Possible solution: Know this initially and pass in a flag? Probably easier than
    trying to figure it out on the fly...
'''
class ModelPenalty:

    def __init__ (self, input_pairwise_linear, penalty: str, dgrid: (Array, Array), inflect_point_val: Tensor = None,
                  n_grid: int = 500, neg_integral: bool = False, pre_comp_xgrid: Array = None,
                  third_deriv_sign: str = 'pos') -> None:
        r"""Initializes the ModelPenalty object for computing the spline functional form penalty
        
        Arguments:
            input_pairwise_linear: This can be either an Input_layer_pairwise_linear or 
                input_layer_pairwise_linear_joined object.
            penalty (str): The pnalty to be computed. One of "convex", "monotonic", "smooth"
            dgrid (Array, Array): The matrix A and vector b for doing y = Ax + b. Depending on 
                the penalty,it's either the first or second derivative
            inflect_point_val (Tensor): The tensor containing the value used to compute the
                inflection point. Defaults to None
            n_grid (int): Number of grid points to use. Defaults to 500
            neg_integral (bool): Whether the spline is concave down. Defaults to False (concave up)
            pre_comp_xgrid (Array): The precomputed xgrid. Defaults to None.
            third_deriv_sign (str): The sign of the third derivative. One of 'pos' for positive or 
                'neg' for negative. Defaults to 'pos', but should be assigned if the penalty type is 
                'smooth'
        
        Returns:
            None
        
        Notes: This is adapted from solver.py from the dftbrepulsive project. For concavity,
            we define concave down as (v''(r) < 0) and concave up as (v''(r) > 0), where r is
            the interatomic distance. If no dgrids are given, then they are recomputed which is
            computationally inefficient.
        """
        self.input_pairwise_lin = input_pairwise_linear
        self.penalty = penalty
        self.neg_integral = neg_integral
        self.third_deriv_sign = third_deriv_sign
        self.n_grid = n_grid
        
        #Compute the x-grid
        if (pre_comp_xgrid is None):
            r_low, r_high = self.input_pairwise_lin.pairwise_linear_model.r_range()
            self.xgrid = np.linspace(r_low, r_high, n_grid)
        else:
            self.xgrid = pre_comp_xgrid
        
        #Compute the derivative_grid for the necessary derivative given the penalty
        self.dgrid = None
        if dgrid is None:
            if self.penalty == "convex":
                self.dgrid = self.input_pairwise_lin.pairwise_linear_model.linear_model(self.xgrid, 2)
            elif self.penalty == "smooth": #Smooth penalty is applied based on the third derivative
                self.dgrid = self.input_pairwise_lin.pairwise_linear_model.linear_model(self.xgrid, 3)
            elif self.penalty == "monotonic":
                self.dgrid = self.input_pairwise_lin.pairwise_linear_model.linear_model(self.xgrid, 1)
        else:
            self.dgrid = dgrid
        self.inflect_x_val = inflect_point_val #Either tensor or nonetype
            
    def compute_inflection_point(self) -> Tensor:
        r"""Computes the value of the inflection point based on the optimized variable x_val
        
        Arguments:
            None
        
        Returns:
            inflection_point (Tensor): The position of the inflection point in 
                angstroms
        
        Notes: The inflection point is computed from the x_val using the following
            formula: 
            
            r_inflect = lrow + ((rhigh - rlow) / 2) * ((atan(x) / (pi/2)) + 1)
            
            The arctangent function is used for smooth differentiability on backpropagation.
            Division by pi/2 fixes the range such that x can range from (-inf, inf) but
            r_inflect will range from (rlow, rhigh)
            
            This method should only be invoked if it is confirmed that
            self.inflect_x_val is not None
        """
        rlow, rhigh = self.input_pairwise_lin.pairwise_linear_model.r_range()
        first_term = (rhigh - rlow) / 2
        const = torch.tensor([math.pi / 2], dtype = self.inflect_x_val.dtype,
                             device = self.inflect_x_val.device)
        second_term = (torch.atan(self.inflect_x_val) / const) + 1
        return rlow + (first_term * second_term)
    
    def compute_penalty_vec(self) -> Tensor:
        r"""Computes the penalty vector to multiply p_convex by based on atan approach
        
        Arguments:
            None
        
        Returns:
            penalty_grid (Tensor): The penalty grid computed using atan approach,
                which will be multiplied by p_convex
        
        Notes: The penalty grid is computed as 
            
            p_i = arctan(10 * (r_i - r_inflect))
            
            Where p_i is the penalty for the r_i point, and r_inflect is computed
            as the defined in compute_inflection_point. This penalty grid is then
            multiplied into p_convex. This ensures that smooth curvatures withn consistent
            second derivatives are not penalized, and accounts for one inflection point, 
            which occurs in the short range for the models of the overlap operator S.
            
            The 10 is an arbitrary scaling constant.
        """
        r_inflect = self.compute_inflection_point()
        torch_xgrid = torch.tensor(self.xgrid, dtype = r_inflect.dtype, device = r_inflect.device)
        corrected_xgrid = torch_xgrid - r_inflect
        penalty_grid = torch.atan(10 * corrected_xgrid)
        return penalty_grid
    
    # Now some methods for computing the different penalties
    def get_monotonic_penalty(self) -> float:
        r"""Computes the monotonic penalty for the model
        
        Arguments:
            None
            
        Returns:
            monotonic_penalty (float): The value for the monotonic penalty, with
                gradients for backpropagation
        
        Notes: The monotonic penalty depends on the first integral, v'(r). If the 
            spline should be monotonically increasing (concave down), we enforce that 
            v'(r) > 0. Otherwise, we enforce that v'(r) < 0 for a concave up, 
            monotonically decreasing potential. Enforcement of the penalty is done through
            the ReLU activation function, where if the spline should be concave down, we multiply 
            all the values by -1 and apply ReLU to only penalize the negative terms. Otherwise,
            we just apply ReLU to penalize the positive terms.
            
            The final penalty is computed as a dot product of the vector resulting from
            the application of ReLU.
        """
        monotonic_penalty = 0
        m = torch.nn.ReLU()
        c = self.input_pairwise_lin.get_variables()
        if hasattr(self.input_pairwise_lin, 'joined'):
            other_coefs = self.input_pairwise_lin.get_fixed()
            c = torch.cat([c, other_coefs])
        deriv, consts = self.dgrid[0], self.dgrid[1]
        deriv, consts = torch.tensor(deriv), torch.tensor(consts)
        #p_monotonic = torch.einsum('j,ij->i', c, deriv) + consts
        p_monotonic = torch.matmul(deriv, c) + consts
        #For a monotonically increasing potential (i.e. concave down integral), the
        # First derivative should be positive, so penalize the negative terms. Otherwise,
        # penalize the positive terms for concave up
        if self.neg_integral:
            p_monotonic = -1 * p_monotonic
            p_monotonic = m(p_monotonic)
            # p_monotonic [p_monotonic > 0] = 0
        else:
            # p_monotonic [p_monotonic < 0] = 0 
            p_monotonic = m(p_monotonic)
        monotonic_penalty = torch.matmul(p_monotonic, p_monotonic) / len(p_monotonic)
        return monotonic_penalty
        
    def get_convex_penalty(self) -> float:
        r"""Computes the convex penalty
        
        Arguments:
            None
        
        Returns:
            convex_penalty (float): The value of the convex penalty with gradients 
                for backpropagation
        
        Notes: The convex penalty deals with the second derivative. For a concave up function,
            we enforce that v''(r) > 0, and for a concave down function we enforce that v''(r) < 0. 
            This us achieved again through the use of ReLU.
            
            In the case of an inflection point, we allow for smooth curvatures on either side
            by multiplying p_convex by a tensor of values computed using compute_penalty_vec
        """
        convex_penalty = 0
        m = torch.nn.ReLU()
        c = self.input_pairwise_lin.get_variables()
        if hasattr(self.input_pairwise_lin, 'joined'):
            other_coefs = self.input_pairwise_lin.get_fixed()
            c = torch.cat([c, other_coefs])
        deriv, consts = self.dgrid[0], self.dgrid[1]
        deriv, consts = torch.tensor(deriv, dtype = c.dtype, device = c.device), torch.tensor(consts, dtype = c.dtype, device = c.device)
        #p_convex = torch.einsum('j,ij->i', c, deriv) + consts
        p_convex = torch.matmul(deriv, c) + consts
        if (self.inflect_x_val is not None): #Compute the penalty grid if an inflection point is present
            penalty_grid = self.compute_penalty_vec()
            #p_convex = torch.einsum('i,i->i', p_convex, penalty_grid) #Multiply the p_convex by the penalty_grid
            p_convex = p_convex * penalty_grid
        # Case on whether the spline should be concave up or down
        if self.neg_integral:
            p_convex = m(p_convex)
        else:
            p_convex = -1 * p_convex
            p_convex = m(p_convex)
        # Equivalent of RMS penalty
        #convex_penalty = torch.einsum('i,i->', p_convex, p_convex) / len(p_convex)
        convex_penalty = torch.matmul(p_convex, p_convex) / p_convex.shape[0]
        return convex_penalty
    
    def get_smooth_penalty(self) -> float:
        r"""Computes the smooth penalty based on the third derivative
        
        Arguments:
            None
        
        Returns:
            smooth_penalty (float): The value of the smoot penalty with gradients for 
                backpropagation
        
        Notes:
            Unlike the convex penalty, the smooth penalty is applied on the sign of the 
            third derivative. This is determined as the opposite sign of the second derivative,
            which is the case because of the simple univariate curvature of the splines. The 
            implementation of the penalty is analagous to the convex penalty
        """
        
        # raise NotImplementedError("Method get_smooth_penalty() not implemented!")
        
        smooth_penalty = 0
        m = torch.nn.ReLU() #Use ReLU to select for sign
        c = self.input_pairwise_lin.get_variables()
        if hasattr(self.input_pairwise_lin, 'joined'):
            other_coefs = self.input_pairwise_lin.get_fixed()
            c = torch.cat([c, other_coefs])
        deriv, consts = self.dgrid[0], self.dgrid[1]
        deriv, consts = torch.tensor(deriv, dtype = c.dtype, device = c.device), torch.tensor(consts, dtype = c.dtype, device = c.device)
        p_smooth = torch.matmul(deriv, c) + consts
        #Assuming for right now that there is no penalty grid/inflection point;
        #   will have to come back and take care of this later
        if self.third_deriv_sign == 'neg':
            p_smooth = m(p_smooth)
        else:
            p_smooth = -1 * p_smooth
            p_smooth = m(p_smooth)
        smooth_penalty = torch.matmul(p_smooth, p_smooth) / p_smooth.shape[0]
        return smooth_penalty
    
    def get_smooth_L2_loss(self) -> float:
        r"""Computes the L2 loss over the third derivative penalty
        
        Arguments:
            None
        
        Returns:
            L2_norm (float): The computed value of the L2-norm of the 
                the third derivative.
        
        Notes:
            Unlike the sign penalty being used for the second derivative
            (which is essentially an inequality penalty), the L2 loss seeks 
            to minimize the L2-norm of the vector of third derivative predictions.
            The L2-norm is defined as sqrt(Sum_i |x_i|^2).
        """
        c = self.input_pairwise_lin.get_variables()
        if hasattr(self.input_pairwise_lin, 'joined'):
            other_coefs = self.input_pairwise_lin.get_fixed()
            c = torch.cat([c, other_coefs])
        deriv, consts = self.dgrid[0], self.dgrid[1]
        deriv, consts = torch.tensor(deriv, dtype = c.dtype, device = c.device), torch.tensor(consts, dtype = c.dtype, device = c.device)
        pred_third_der = torch.matmul(deriv, c) + consts
        # L2_norm = torch.sqrt(torch.sum(torch.square(pred_third_der)))
        #Make this definition more consistent with other penalties
        L2_norm = torch.matmul(pred_third_der, pred_third_der) / pred_third_der.shape[0]
        L2_norm = torch.sqrt(L2_norm)
        return L2_norm
            
    
    #Compute the actual loss
    def get_loss(self) -> float:
        r"""Computes the loss
        
        Arguments: 
            None
        
        Returns:
            penalty (float): Calls on the requisite function depending on the
                set penalty to compute the value for the form penalty
        
        Notes: None
        """
        if self.penalty == "convex":
            return self.get_convex_penalty()
        elif self.penalty == "monotonic":
            return self.get_monotonic_penalty()
        elif self.penalty == "smooth":
            #Switching over to the L2 Loss
            return self.get_smooth_L2_loss()
        
class LossModel:
    def get_feed(self, feed: Dict, molecs: List[Dict], all_models: Dict, par_dict: Dict, debug: bool) -> None:
        '''
        Method for adding stuff to the feed for the given loss. Every loss
        will take in the feed and the associated molecules, but 
        
        The debug flag is set to true or false depending on whether or not to fit to DFTB.
        If performing a fit to DFTB, then the debug flag is set to true. Otherwise, it is False
        
        For consistency, the method adds things directly to the feed wihtout returning any values.
        '''
        raise NotImplementedError
    
    def get_value(self, output: Dict, feed: Dict, rep_method: str) -> None:
        '''
        Computes the value of the loss directly against the feed from the output,
        returns the loss as a PyTorch object.
        
        For consistency, all losses have torch.sqrt applied since we are interested in 
        RMSE loss. Also include argument for the rep_setting since different losses
        (e.g. form penalty loss) are computed differently if new repulsive setting used
        '''
        raise NotImplementedError