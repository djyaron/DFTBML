# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 14:05:22 2020

@author: Frank
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 23:03:05 2020

@author: Frank
"""
"""
TODO:
    1) Encode new ranges for the spline models, so that they are not dependent on the 
        ranges in the data
        (The H block and S block should align in the skf files, similar to in the 
         original ones)

"""
import pdb, traceback, sys, code

import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from collections import OrderedDict
import collections
import torch
torch.set_printoptions(precision = 10)
from copy import deepcopy
import torch.nn as nn
import torch.optim as optim
import time
from tfspline import Bcond

from geometry import Geometry, to_cart
from auorg_1_1 import ParDict
from dftb import DFTB
from eig import SymEigB
from batch import create_batch, create_dataset, DFTBList

from modelspline import get_dftb_vals
from SplineModel_v3 import SplineModel, fit_linear_model, JoinedSplineModel

from numbers import Real
from typing import Union, List, Optional, Dict, Any, Literal
Tensor = torch.Tensor
Array = np.ndarray
from batch import Model, RawData

from dftb_layer_splines_ani1ccx import get_targets_from_h5file
from h5handler import model_variable_h5handler, per_molec_h5handler, per_batch_h5handler,\
    total_feed_combinator, compare_feeds
from loss_models import TotalEnergyLoss, FormPenaltyLoss, DipoleLoss, ChargeLoss, DipoleLoss2,\
    ReferenceEnergyLoss

from sccparam_torch import _Gamma12 #Gamma func for computing off-diagonal elements
from functools import partial

#Fix the ani1_path for now
ani1_path = 'data/ANI-1ccx_clean_fullentry.h5'

def apx_equal(x: Union[float, int], y: Union[float, int], tol: float = 1e-12) -> bool:
    r"""Compares two floating point numbers for equality with a given threshold
    
    Arguments:
        x (float): The first number to be compared
        y (float): The second number to be compared
        
    Returns:
        equality (bool): Whether the two given numbers x and y are equal
            within the specified threshold by comparing the absolute value
            of their difference.
            
    Notes: This method works with both integers and floats, which are the two 
        numeric types. Chosen workaround for determining float equality

    """
    return abs(x - y) < tol

def get_ani1data(allowed_Z: List[int], heavy_atoms: List[int], max_config: int, 
                 target: Dict[str, str], exclude: List[str] = []) -> List[Dict]:
    r"""Extracts data from the ANI-1 data files
    
    Arguments:
        allowed_Z (List[int]): Include only molecules whose elements are in
            this list
        heavy_atoms (List[int]): Include only molecules for which the number
            of heavy atoms is in this list
        max_config (int): Maximum number of configurations included for each
            molecule.
        target (Dict[str,str]): entries specify the targets to extract
            key: target_name name assigned to the target
            value: key that the ANI-1 file assigns to this target
        exclude (List[str], optional): Exclude these molecule names from the
            returned molecules
            Defaults to [].
            
    Returns:
        molecules (List[Dict]): Each Dict contains the data for a single
            molecular structure:
                {
                    'name': str with name ANI1 assigns to this molecule type
                    'iconfig': int with number ANI1 assignes to this structure
                    'atomic_numbers': List of Zs
                    'coordinates': numpy array (:,3) with cartesian coordinates
                    'targets': Dict whose keys are the target_names in the
                        target argument and whose values are numpy arrays
                        with the ANI-1 data
                        
    Notes: The ANI-1 data h5 files are indexed by a molecule name. For each
        molecule, the data is stored in arrays whose first dimension is the
        configuration number, e.g. coordinates(iconfig,atom_num,3). This
        function treats each configuration as its own molecular structure. The
        returned dictionaries include the ANI1-name and configuration number
        in the dictionary, along with the data for that individual molecular
        structure.
    """
    target_alias, h5keys = zip(*target.items())
    target_alias, h5keys = list(target_alias), list(h5keys)
    all_zs = get_targets_from_h5file('atomic_numbers', ani1_path)
    all_coords =  get_targets_from_h5file('coordinates', ani1_path)
    all_targets = get_targets_from_h5file(h5keys, ani1_path)
    if exclude is None:
        exclude = []
    batches = list()
    for name in all_zs.keys():
        if name in exclude:
            continue
        zs = all_zs[name][0] #Extract np array of the atomic numbers
        zcount = collections.Counter(zs)
        ztypes = list(zcount.keys())
        zheavy = [x for x in ztypes if x > 1]
        nheavy = sum([zcount[x] for x in zheavy])
        ztypes.sort()
        zkeep = deepcopy(allowed_Z)
        zkeep.sort()
        if any([zz not in allowed_Z for zz in ztypes]):
            continue
        if nheavy not in heavy_atoms:
            continue
        nconfig = all_coords[name][0].shape[0] #Extract the np array of the atomic coordinates
        for iconfig in range(min(nconfig, max_config)):
            batch = dict()
            batch['name'] = name
            batch['iconfig'] = iconfig
            batch['atomic_numbers'] = zs
            batch['coordinates'] = all_coords[name][0][iconfig,:,:]
            batch['targets'] = dict()
            for i in range(len(target_alias)):
                targ_key = target_alias[i]
                batch['targets'][targ_key] = all_targets[name][i][iconfig]
            batches.append([batch])
    return batches

def create_graph_feed(config: Dict, batch: List[Dict], allowed_Zs: List[int], par_dict: Dict) -> (Dict, DFTBList):
    r"""Takes in a list of geomtries and creates the graph and feed dictionaries for the batch
    
    Arguments:
        config (Dict): Configuration dictionary that indicates which operators to model (e.g. "H", "G", "R")
        batch (List[Dict]): List of dictionaries representing the molecular configurations that 
            are going into the batch
        allowed_Zs (List[int]): List of allowed atomic numbers
        par_dict (Dict): Dictionary of skf parameters, indexed by atom symbol pairs (e.g. 'C-C')
        
    Returns:
        feed (Dict): Dictionary containing information for all the requested fields
            for the batch, with the field names as the keys. This is the feed dictionary
            for the given molecular batch.
        dftblist (DFTBList): Instance of the DFTBList object for this batch that contains
            the DFTB objects necessary for charge updates.
    
    Notes: The values for the feed can change depending on which fields are needed. 
        Those fields that can be linked to individual molecules are organized by basis size,
        and the information can be connected back to the molecules using the 'glabels' key. 
        The molecular conformations are listed under the 'geom' key, and they are inserted 
        in the same order as they appear in batch.
        
        Information that depends on the batch composition is not organized by bsize, so things 
        like 'models', 'onames', and 'basis_sizes' are given as lists.
    """
    fields_by_type = dict()
    fields_by_type['graph'] = \
      ['models','onames','basis_sizes','glabels',
       'gather_for_rot', 'gather_for_oper',
       'gather_for_rep','segsum_for_rep', 'occ_rho_mask',
       'occ_eorb_mask','qneutral','atom_ids','norbs_atom','zcounts']
    
    fields_by_type['feed_constant'] = \
        ['geoms','mod_raw', 'rot_tensors', 'dftb_elements', 'dftb_r','Erep', 'rho']
    
    if 'S' not in config['opers_to_model']:
        fields_by_type['feed_constant'].extend(['S','phiS','Sevals'])
    if 'G' not in config['opers_to_model']: #Testing not modelling G CHANGE BACK LATER
        fields_by_type['feed_constant'].extend(['G'])
    
    fields_by_type['feed_SCF'] = \
        ['dQ','Eelec','eorb']
        
    needed_fields = fields_by_type['feed_constant'] + \
       fields_by_type['feed_SCF'] + fields_by_type['graph']
    
    geom_batch = []
    for molecule in batch:
        geom = Geometry(molecule['atomic_numbers'], molecule['coordinates'].T)
        geom_batch.append(geom)
        
    batch = create_batch(geom_batch, opers_to_model = config['opers_to_model'], 
                         parDict = par_dict, FIXED_ZS=allowed_Zs) #Add in opers to model
    dftblist = DFTBList(batch)
    feed = create_dataset(batch,dftblist,needed_fields)

    return feed, dftblist

class Input_layer_DFTB:

    def __init__(self, model: Model) -> None:
        r"""Initializes a debugging model that just uses the DFTB values
        
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

class Input_layer_value:
    
    def __init__(self, model: Model, initial_value: float = 0.0) -> None:
        r"""Interface for models predicting on-diagonal elements
        
        Arguments:
            model (Model): A named tuple of the form ('oper', 'Zs', 'orb'), where
                'oper' is the operater the model is modelling represented as a string
                (e.g. 'G', 'H', 'R'), 'Zs' is a tuple of the atomic number that is needed
                (e.g. (1,)), and 'orb' is a string representing the orbitals being considered
                (e.g. 'ss' for two s-orbital interactions)
            initial_value (float): The starting value for the model. Defaults to 0.0
        
        Returns:
            None
        
        Notes: Because this model is only used to model on-diagonal elements of the 
            various operator matrices, this constructor is only called when the 
            number of atomic numbers is 1 (i.e. len(model.Zs) == 1). The variable tensor
            has requires_grad set to true so that the variable is trainable by the network later on.
        """
        self.model = model
        if not isinstance(initial_value, float):
            raise ValueError('Val_model not initialized to float')
        self.value = np.array([initial_value])
        self.variables = torch.from_numpy(self.value)
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
        """
        init_value = get_dftb_vals(self.model, pardict)
        if not noise_magnitude == 0.0:
            init_value = init_value + noise_magnitude * np.random.randn(1)
        self.value[0]= init_value

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
   
class Input_layer_pairwise_linear:

    def __init__(self, model: Model, pairwise_linear_model: SplineModel, par_dict: Dict, ngrid: int = 100, 
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
        (rlow,rhigh) = pairwise_linear_model.r_range()
        ngrid = 100
        rgrid = np.linspace(rlow,rhigh,ngrid)
        ygrid = get_dftb_vals(model, par_dict, rgrid)
        ygrid = ygrid + noise_magnitude * np.random.randn(len(ygrid))
        model_vars,_,_ = fit_linear_model(self.pairwise_linear_model, rgrid,ygrid) 
        self.variables = torch.from_numpy(model_vars)
        self.variables.requires_grad = True

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
        """
        xeval = np.array([elem.rdist for elem in mod_raw])
        A,b = self.pairwise_linear_model.linear_model(xeval)
        return {'A': A, 'b': b}
    
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
        result = torch.matmul(A, self.variables) + b
        return result

class Input_layer_pairwise_linear_joined:

    def __init__(self, model: Model, pairwise_linear_model: JoinedSplineModel, par_dict: Dict,
                 cutoff: float, ngrid: int = 100, 
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
        """
        self.model = model
        self.pairwise_linear_model = pairwise_linear_model
        (rlow, rhigh) = pairwise_linear_model.r_range()
        rgrid = np.linspace(rlow, rhigh, ngrid)
        ygrid = get_dftb_vals(model, par_dict, rgrid)
        ygrid = ygrid + noise_magnitude * np.random.randn(len(ygrid))
        variable_vars, fixed_vars = pairwise_linear_model.fit_model(rgrid, ygrid)
        #Initialize the optimizable torch tensor for the variable coefficients
        # of the spline and the fixed part that's cat'd on each time
        self.variables = torch.from_numpy(variable_vars)
        self.variables.requires_grad = True
        self.constant_coefs = torch.from_numpy(fixed_vars)
        self.joined = True #A flag used by later functions to identify joined splines
        self.cutoff = cutoff #Used later for outputting skf files
        
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
    
class OffDiagModel:
    
    def __init__(self, model: Model, model_variables: Dict) -> None:
        r"""Initializes the off-diagonal model
        
        Arguments:
            model (Model): Named tuple describing the interaction to be modeled
            model_variables (Dict): Dictionary referencing all the variables of models
                being used
        
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
        # Keep the variables in a list
        self.variables = [elem1_var, elem2_var] 
    
    def get_variables(self) -> List[Tensor]:
        r"""Returns the variables that the model needs
        
        Arguments:
            None
        
        Returns:
            var1 (Tensor): The tensor for the first variable in the model
            var2 (Tensor): The tensor for the second variable in the model
        
        Notes: The variables here are references to the variables in model_variables,
            which in turn are references to the on-diagonal matrix elements (Input_layer_value).
            So for G(C s|C p) (r), the two variables would be the on-diagonal element for 
            G(C s) and G(C p). Both variables should have gradients enabled.
        """
        return self.variables
    
    def get_feed(self, mod_raw: List[RawData]) -> Dict:
        r"""Gets information for the feed dictionary for the OffDiagModel
        
        Arguments:
            mod_raw (List[RawData]): A list of raw data used to extract the distances
                at which to evaluate the model
        
        Returns:
            distances (Dict): A dictionary containing an array of all the distances
                at which to evaluate the model
        
        Notes: The only thing that needs to be added to the feed dictionary for the 
            OffDiagModel are the distances at which to evaluate them.
        """
        xeval = np.array([elem.rdist for elem in mod_raw])
        return {'distances' : xeval}
    
    def get_values(self, feed: Dict) -> Tensor:
        r"""Obtains the predicted values from the model
        
        Arguments:
            feed (Dict): Dictionary containing the distances to evaluate the model at
        
        Returns:
            results (Tensor): The predicted values for the model at the necessary 
                distances
        
        Notes: Computed by a map of _Gamma12() across all the distances with
            the given variables. Switching variable order does not affect
            computed result, i.e. _Gamma12(r, x, y) == _Gamma12(r, y, x).
        """
        distances = feed['distances']
        elem1_var, elem2_var = self.variables
        gamma_partial = partial(_Gamma12, hub1 = elem1_var, hub2 = elem2_var)
        results = list(map(gamma_partial, distances))
        return torch.cat(results)

class OffDiagModel2:
    
    def __init__(self, model: Model, model_variables: Dict) -> None:
        r"""Initializes the off-diagonal model
        
        Arguments:
            model (Model): Named tuple describing the interaction to be modeled
            model_variables (Dict): Dictionary referencing all the variables of models
                being used
        
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
                'nonzero_distances' : nonzero_distances}
    
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
        r12 = feed['nonzero_distances']
        nelements = len(zero_indices) + len(nonzero_indices)
        results = torch.zeros([nelements], dtype = r12.dtype)
        
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
        
class Reference_energy:

    #TODO: add const to this
    def __init__(self, allowed_Zs: List[int], 
                 prev_values: List[float] = [ -0.2323322747, -36.3256865272, -52.3187836247, -71.8383273595]) -> None:
        r"""Initializes the reference energy model.
        
        Arguments:
            allowed_Zs (List[int]): List of the allowed atomic numbers
            prev_values (List[float]): Previous values for the reference energy to start from.
                Defaults to []
        
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
        """
        self.allowed_Zs = np.sort(np.array(allowed_Zs))
        self.values = np.zeros(self.allowed_Zs.shape)
        if len(prev_values) > 0:
            #Load previous values if they are given
            #FOR DEBUGGING PURPOSES ONLY
            assert(len(prev_values) == len(self.values))
            self.values = np.array(prev_values)
        self.variables = torch.from_numpy(self.values)
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
    
    
def get_model_dftb(model_spec: Model) -> Input_layer_DFTB:
    r"""Wrapper method for generating an instance of the debugging model for a given model_spec
    
    Arguments: 
        model_spec (Model): The named tuple describing the interaction
    
    Returns:
        Input_layer_DFTB: An instance of the debugging model
    
    Notes: None
    """
    return Input_layer_DFTB(model_spec)


class data_loader:

    def __init__(self, dataset: List[Dict], batch_size: int, shuffle: bool = True, 
                 shuffle_scheme: str = 'random') -> None:
        r"""Initializes the data_loader object for batching data.
        
        Arguments:
            dataset (List[Dict]): A list of dictionaries containing information for
                all the molecules, with each molecule represented as a dictionary.
            batch_size (int): The number of molecules to have per batch
            shuffle (bool): Whether or not to shuffle batches. Defaults to True
            shuffle_scheme (str): The scheme for shuffling. Defaults to "random"
        
        Returns:
            None
        
        Notes: This is a very simple data_loader implementation that uses sequential
            batching on a list of data. Once all the data has been iterated over, the
            loader can be re-iterated over and it will shuffle the batches between iterations.
            
            This loader can be used for any list-type dataset, but for this use case we 
            specify List[Dict] since that is the data format being used here.
        """
        self.data = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.shuffle_method = shuffle_scheme
        self.batch_creation()
        self.batch_index = 0
    
    def create_batches(self, data: List[Dict]) -> List[List[Dict]]:
        r"""Generates the batches from the list of data.
        
        Arguments:
            data (List[Dict]): List of molecule dictionaries from which to generate
                the batches
        
        Returns:
            batches (List[List[Dict]]): A list of lists of molecule dictionaries, where
                each inner list represents one batch
                
        Notes: If the number of elements in the data is not a multiple of the batch size,
            the last batch will have less elements than the batch size.
        """
        batches = list()
        for i in range(0, len(data), self.batch_size):
            current_batch = data[i : i + self.batch_size]
            batches.append(current_batch)
        return batches
    
    def batch_creation(self) -> None:
        r"""Wrapper method for creating the batches
        
        Arguments:
            None
        
        Returns: 
            None
        
        Notes: Initializes the batches by calling self.create_batches, and 
            also initializes the batch index used for iterating over the 
            data_loader
        """
        self.batches = self.create_batches(self.data)
        self.batch_index = 0
        
    def shuffle_batches(self) -> None:
        r"""Shuffles batches and resets the batch_index
        
        Arguments: 
            None
            
        Returns:
            None
        
        Notes: Called at end of iteration over data_loader
        """
        random.shuffle(self.batches)
        self.batch_index = 0
        
    def shuffle_total_data(self) -> None:
        r"""Reshuffles the original data from which the batches are generated
        
        Arguments:
            None
        
        Returns:
            None
        
        Notes: None
        """
        random.shuffle(self.data)
        self.batch_creation()
    
    def __iter__(self):
        r"""
        Method for treating the data_loader as the iterator.
        """
        return self
    
    def __next__(self) -> List[Dict]:
        r"""Method for retrieving next element in iterator
        
        Arguments:
            None
        
        Returns:
            return_batch (List[Dict]): The current batch to be used
        
        Notes: None
        
        Raises:
            StopIteration: If iteration over all batches in data_loader is complete
        """
        if self.batch_index < len(self.batches):
            return_batch = self.batches[self.batch_index]
            self.batch_index += 1
            return return_batch
        else:
            # Automatically shuffle the batches after a full iteration
            self.shuffle_batches()
            raise StopIteration

def get_model_value_spline_2(model_spec: Model, model_variables: Dict, spline_dict: Dict, par_dict: Dict, num_knots: int = 50, 
                             num_grid: int = 200, buffer: float = 0.0, 
                             joined_cutoff: float = 3.0, cutoff_dict: Dict = None,
                             off_diag_opers: List[str] = ["G"]) -> (Input_layer_pairwise_linear_joined, str):
    r"""Generates a joined spline model for the given model_spec
    
    Arguments:
        model_spec (Model): Named tupled describing the interaction being modeled by the spline
        model_variables (Dict): Dictionary containing references to all the model variables
            instantiated so far. This is necessary for initializing the instances of the OffDiagModel
        spline_dict (Dict): Dictionary of the minimum and maximum distance, i.e. endpoints of the range
            spanned by the spline
        par_dict (Dict): Dictionary of the DFTB Slater-Koster parameters for atomic interactions 
            between different elements, indexed by a string 'elem1-elem2'. For example, the
            Carbon-Carbon interaction is accessed using the key 'C-C'
        num_knots (int): The number of knots to use for the spline. Defaults to 50
        num_grid (int): The number of grid points, used for old approach for identifying
            if the spline is completely 0. Defaults to 200
        buffer (float): The value to shift the starting and ending distance by. The starting distance
            is rlow - buffer, and the ending distance is rhigh + buffer, where rlow, rhigh are the 
            minimum and maximum distances for the spline from the spline_dict. Defaults to 0.0 angstroms
        joined_cutoff (float): The cutoff distance for separating the variable and fixed
            regions of the spline. Defaults to 3.0 angstroms
        cutoff_dict (Dict): Dictionary of cutoffs to use for each joined spline model, indexed
            by the model_spec. Defaults to None, in which case the joined_cutoff default of 3.0 angstroms is used
        off_diag_opers (List[str]): A list of the operators to model using the 
            OffDiagModel. Defaults to ['G']
    
    Returns:
        model (Input_layer_pairwise_linear_joined): The instance of the Input_layer_pairwise_linear_joined
            object for working with the joined spline
        tag (str): A tag indicating whether to optimize the spline ('opt') or to ignore the spline in
            optimizations ('noopt')
    
    Notes: The tag is included because if a spline is flat at 0 (i.e., no interactions), then the 
        spline is not optimized. 
        
        The inclusion of model_variables is for the OffDiag model. It is assumed that by the time
        a two-body interaction is encountered (e.g. G, (1,6), sp) that all the one-body interactions for
        that model have been handled. Right now, it seems like that is a safe assumption.
    """
    noise_magnitude = 0.0
    if len(model_spec.Zs) == 1:
        model = Input_layer_value(model_spec)
        model.initialize_to_dftb(par_dict, noise_magnitude)
        return (model, 'vol')
    elif len(model_spec.Zs) == 2:
        if model_spec.oper not in off_diag_opers:
            minimum_value, maximum_value = spline_dict[model_spec]
            minimum_value -= buffer
            maximum_value += buffer
            xknots = np.linspace(minimum_value, maximum_value, num = num_knots)
            # Joined splines for all operators
            config = {'xknots' : xknots,
                      'equal_knots' : False,
                      'cutoff' : cutoff_dict[model_spec] if (cutoff_dict is not None) else joined_cutoff,
                      'bconds' : 'natural'}
            spline = JoinedSplineModel(config)
            model = Input_layer_pairwise_linear_joined(model_spec, spline, par_dict, config['cutoff'])
            variables = model.get_variables().detach().numpy()
            if apx_equal(np.sum(variables), 0):
                return (model, 'noopt')
            return (model, 'opt')
        else:
            # Case of using OffDiagModel
            model = OffDiagModel2(model_spec, model_variables)
            return (model, 'opt')

def torch_segment_sum(data: Tensor, segment_ids: Tensor, device: torch.device, dtype: torch.dtype) -> Tensor: 
    r"""Function for summing elements together based on index
    
    Arguments:
        data (Tensor): The data to sum together
        segment_ids (Tensor): The indices used to sum together corresponding elements
        device (torch.device): The device to execute the operations on (CPU vs GPU)
        dtype (torch.dtype): The datatype for the result
    
    Returns:
        res (Tensor): The resulting tensor from executing the segment sum
    
    Notes: This is similar to scatter_add for PyTorch, but this is easier to deal with.
        The segment_ids, since they are being treated as indices, must be a tensor
        of integers
    """
    max_id = torch.max(segment_ids)
    res = torch.zeros([max_id + 1], device = device, dtype = dtype)
    for i, val in enumerate(data):
        res[segment_ids[i]] += val
    return res

class DFTB_Layer(nn.Module):
    
    def __init__(self, device: torch.device, dtype: torch.dtype, eig_method: str = 'new') -> None:
        r"""Initializes the DFTB deep learning layer for the network
        
        Arguments:
            device (torch.device): The device to run the computations on (CPU vs GPU)
            dtype (torch.dtype): The torch datatype for the calculations
            eig_method (str): The eigenvalue method for the symmetric eigenvalue decompositions.
                'old' means the original PyTorch symeig method, and 'new' means the 
                eigenvalue broadening method implemented in eig.py. Defaults to 'new'.
        
        Returns:
            None
        
        Notes: The eigenvalue broadening method implemented in eig.py is the default 
            because with eigenvalue broadening degenerate molecules are handled and do not
            need to be sorted out explicitly. Further documentation for the methods 
            can be found in the eig.py file.
            
            The theory and derviations for the tensor operations comprising the DFTB layer
            can be found in the paper by Collins, Tanha, Gordon, and Yaron [1]
        
        References:
            [1] Li, H.; Collins, C.; Tanha, M.; Gordon, G. J.; Yaron, D. J. A Density
            Functional Tight Binding Layer for Deep Learning of Chemical Hamiltonians. 2018,
        """
        super(DFTB_Layer, self).__init__()
        self.device = device
        self.dtype = dtype
        self.method = eig_method
    
    def forward(self, data_input: Dict, all_models: Dict) -> Dict:
        r"""Forward pass through the DFTB layer to generate molecular properties
        
        Arguments: 
            data_input (Dict): The feed dictionary for the current batch being pushed
                through the network
            all_models (Dict): The dictionary containing references to all the spline 
                model objects being used to predict operator elements
        
        Returns:
            calc (Dict): A dictionary contianing the molecular properties predicted from the 
                DFTB layer using the predicted operator element values from the spline models.
                The calc dict contains important values like 'dQ' used for charges and dipoles, and 
                'Erep', 'Eelec', and 'Eref', which are used to compute the total energy.
        
        Notes: The DFTB layer operations are separated into 5 stages: forming the initial input layer,
            performing Slater-Koster rotations, assembling values into operators, constructing the fock operators, and
            solving the generalized eigenvalue problem for the the fock operator. 
        """
        model_vals = list()
        for model_spec in data_input['models']:
            model_vals.append( all_models[model_spec].get_values(data_input[model_spec]) )
        net_vals = torch.cat(model_vals)
        calc = OrderedDict() 
        ## SLATER-KOSTER ROTATIONS ##
        rot_out = torch.tensor([0.0, 1.0], dtype = self.dtype, device = self.device, requires_grad = True)
        for s, gather in data_input['gather_for_rot'].items():
            gather = gather.long()
            if data_input['rot_tensors'][s] is None:
                rot_out = torch.cat((rot_out, net_vals[gather]))
            else:
                vals = net_vals[gather].reshape((-1, s[1])).unsqueeze(2)
                tensor = data_input['rot_tensors'][s]
                rot_out_temp = torch.matmul(tensor, vals).squeeze(2)
                rot_out = torch.cat((rot_out, torch.flatten(rot_out_temp)))
        
        ## ASSEMBLE SK ROTATION VALUES INTO OPERATORS ##
        for oname in data_input['onames']:
            calc[oname] = {}
            if oname != "R":
                for bsize in data_input['basis_sizes']:
                    gather = data_input['gather_for_oper'][oname][bsize].long()
                    calc[oname][bsize] = rot_out[gather].reshape((len(data_input['glabels'][bsize]),bsize,bsize))
        
        if 'S' not in data_input['onames']:
            calc['S'] = deepcopy(data_input['S']) #Deepcopy operations may be too inefficient...
        if 'G' not in data_input['onames']:
            calc['G'] = deepcopy(data_input['G'])
        
        #At this point, we have H and G operators with requires_grad...
        
        ## CONSTRUCT THE FOCK OPERATORS ##
        calc['F'] = {}
        calc['dQ'] = {}
        calc['Erep'] = {}
        for bsize in data_input['basis_sizes']:
        
            # rho = data_input['rho'][bsize] #Unnecessary, just copying dQ
            # qbasis = rho * calc['S'][bsize] #Unnecessary, just copying dQ
            # GOP  = torch.sum(qbasis,2,keepdims=True) #Unnecessary, just copying dQ
            # qNeutral = data_input['qneutral'][bsize] #Unnecessary, just copying dQ
            #calc['dQ'][bsize] = data_input['dQ'][bsize] #Use data_input['dQ'][bsize] here
            ep = torch.matmul(calc['G'][bsize], data_input['dQ'][bsize])
            couMat = ((-0.5 * calc['S'][bsize]) *  (ep + torch.transpose(ep, -2, -1)))
            calc['F'][bsize] = calc['H'][bsize] + couMat 
            vals = net_vals[data_input['gather_for_rep'][bsize].long()] # NET VALS ERROR OCCURS HERE
            #The segment_sum is going to be problematic
            calc['Erep'][bsize] = torch_segment_sum(vals,
                                data_input['segsum_for_rep'][bsize].long(), self.device, self.dtype)
        
        ## SOLVE GEN-EIG PROBLEM FOR FOCK ##
        calc['Eelec']= {}
        calc['eorb'] = {}
        calc['rho'] = {}
        calc['Eref'] = {}
        for bsize in data_input['basis_sizes']:
            S1 = calc['S'][bsize]
            fock = calc['F'][bsize]
            if 'phiS' not in list(data_input.keys()):
                # Eigenvalues in ascending order, eigenvectors are orthogonal, use conditional broadening
                # Try first with default broadening factor of 1E-12
                if self.method == 'new':
                    symeig = SymEigB.apply
                    Svals, Svecs = symeig(S1, 'cond')
                elif self.method == 'old':
                    Svals, Svecs = torch.symeig(S1, eigenvectors = True)
                phiS = torch.matmul(Svecs, torch.diag_embed(torch.pow(Svals, -0.5))) #Changed this to get the diagonalization and multiplication to work
            else:
                phiS = data_input['phiS'][bsize]
            fockp = torch.matmul(torch.transpose(phiS, -2, -1), torch.matmul(fock, phiS))
            try:
                if self.method == 'new':
                    symeig = SymEigB.apply
                    Eorb, temp2 = symeig(fockp, 'cond')
                elif self.method == 'old':
                    Eorb, temp2 = torch.symeig(fockp, eigenvectors = True)
            except Exception as e:
                print('diagonalization failed for batch ', data_input['names'])
                print(e)
                extype, value, tb = sys.exc_info()
                traceback.print_exc()
                pdb.post_mortem(tb)
            calc['eorb'][bsize] = Eorb
            orb = torch.matmul(phiS, temp2)
            occ_mask = data_input['occ_rho_mask'][bsize]
            orb_filled = torch.mul(occ_mask, orb)
            rho = 2.0 * torch.matmul(orb_filled, torch.transpose(orb_filled, -2, -1))
            calc['rho'][bsize] = rho
            ener1 = torch.sum(torch.mul(rho.view(rho.size()[0], -1), calc['H'][bsize].view(calc['H'][bsize].size()[0], -1)), 1) #I think this is fine since calc['Eelec'] is a 1D array
            #dQ = calc['dQ'][bsize] #dQ is just copied but not calculated
            qbasis = rho * calc['S'][bsize] 
            GOP  = torch.sum(qbasis,2,keepdims=True)
            qNeutral = data_input['qneutral'][bsize]
            dQ = qNeutral - GOP
            calc['dQ'][bsize] = dQ
            Gamma = calc['G'][bsize]
            ener2 = 0.5 * torch.matmul(torch.transpose(dQ, -2, -1), torch.matmul(Gamma, dQ))
            ener2 = ener2.view(ener2.size()[0])
            calc['Eelec'][bsize] = ener1 + ener2
            ref_energy_variables = all_models['Eref'].get_variables()
            ref_res = torch.matmul(data_input['zcounts'][bsize], ref_energy_variables.unsqueeze(1))
            calc['Eref'][bsize] = ref_res.squeeze(1)
        return calc

def recursive_type_conversion(data: Dict, ignore_keys: List[str], device: torch.device = None, 
                              dtype: torch.dtype = torch.double, grad_requires: bool = False) -> None:
    r"""Performs destructive conversion of elements in data from np arrays to Tensors
    
    Arguments:
        data (Dict): The dictionary to perform the recursive type conversion on
        ignore_keys (List[str]): The list of keys to ignore when doing
            the recursive type conversion
        device (torch.device): Which device to put the tensors on (CPU vs GPU).
            Defaults to None.
        dtype (torch.dtype): The datatype for all created tensors. Defaults to torch.double
        grad_requires (bool): Whether or not created tensors should have their 
            gradients enabled. Defaults to False
    
    Returns:
        None
    
    Notes: None
    """
    for key in data:
        if key not in ignore_keys:
            if isinstance(data[key], np.ndarray):
                data[key] = torch.tensor(data[key], dtype = dtype, device = device)            
            elif isinstance(data[key], collections.OrderedDict) or isinstance(data[key], dict):
                recursive_type_conversion(data[key], ignore_keys)

def assemble_ops_for_charges(feed: Dict, all_models: Dict) -> Dict:
    r"""Generates the H and G operators for the charge update operation
    
    Arguments:
        feed (Dict): The current feed whose charges need to be udpated
        all_models (Dict): A dictionary referencing all the spline models being used
            to predict operator elements
    
    Returns:
        calc (Dict): A dictionary containing the tensors for the H and G operators, organized
            by basis size. 
    
    Notes: This replicates the first three steps of the DFTB layer.
    """
    model_vals = list()
    for model_spec in feed['models']:
        model_vals.append( all_models[model_spec].get_values(feed[model_spec]) )
    net_vals = torch.cat(model_vals)
    calc = OrderedDict() 
    ## SLATER-KOSTER ROTATIONS ##
    rot_out = torch.tensor([0.0, 1.0])
    for s, gather in feed['gather_for_rot'].items():
        gather = gather.long()
        if feed['rot_tensors'][s] is None:
            rot_out = torch.cat((rot_out, net_vals[gather]))
        else:
            vals = net_vals[gather].reshape((-1, s[1])).unsqueeze(2)
            tensor = feed['rot_tensors'][s]
            rot_out_temp = torch.matmul(tensor, vals).squeeze(2)
            rot_out = torch.cat((rot_out, torch.flatten(rot_out_temp)))
    
    ## ASSEMBLE SK ROTATION VALUES INTO OPERATORS ##
    for oname in feed['onames']:
        calc[oname] = {}
        if oname != "R":
            for bsize in feed['basis_sizes']:
                gather = feed['gather_for_oper'][oname][bsize].long()
                calc[oname][bsize] = rot_out[gather].reshape((len(feed['glabels'][bsize]),bsize,bsize))
    
    if 'S' not in feed['onames']:
        calc['S'] = deepcopy(feed['S']) #Deepcopy operations may be too inefficient...
    if 'G' not in feed['onames']:
        calc['G'] = deepcopy(feed['G'])
    
    return calc

def update_charges(feed: Dict, op_dict: Dict, dftblst: DFTBList) -> None:
    r"""Destructively updates the charges in the feed
    
    Arguments:
        feed (Dict): The feed dictionary whose charges need to be updated
        op_dict (Dict): The dictionary containing the H and G operators, separated by bsize
        dftblst (DFTBList): The DFTBList instance for this feed
    
    Returns:
        None
    
    Notes: Both dQ and the occ_rho_mask are udpated for the feed
    """
    for bsize in op_dict['H'].keys():
        np_Hs = op_dict['H'][bsize].detach().numpy() #Don't care about gradients here
        np_Gs = op_dict['G'][bsize].detach().numpy()
        for i in range(len(dftblst.dftbs_by_bsize[bsize])):
            curr_dftb = dftblst.dftbs_by_bsize[bsize][i]
            curr_H = np_Hs[i]
            curr_G = np_Gs[i] 
            newQ, occ_rho_mask_upd, _ = curr_dftb.get_dQ_from_H(curr_H, newG = curr_G) #Not modelling G (for now) CHANGE BACK LATER
            newQ, occ_rho_mask_upd = torch.tensor(newQ).unsqueeze(1), torch.tensor(occ_rho_mask_upd)
            feed['dQ'][bsize][i] = newQ # Change dQ to newQ instead
            feed['occ_rho_mask'][bsize][i] = occ_rho_mask_upd

def create_spline_config_dict(data_dict_lst: List[Dict]) -> Dict:
    r"""Finds the distance range (min, max) for each model across the entire data
    
    Arguments:
        data_dict_lst (List[Dict]): List of all feeds, train and validation
    
    Returns:
        model_range_dict (Dict): A dictionary mapping each model_spec to 
            a tuple representing the minimum and maximum distances spanned by the model
            (i.e. (min, max))
    
    Notes: The distance ranges for each model depends on all the data points
    """
    model_range_dict = dict()
    for feed in data_dict_lst:
        models_for_feed = list(feed['mod_raw'].keys())
        for model_spec in models_for_feed:
            if len(model_spec.Zs) == 2:
                raw_dat_for_mod = feed['mod_raw'][model_spec]
                distances = [x.rdist for x in raw_dat_for_mod]
                range_low, range_max = min(distances), max(distances)
                if model_spec not in model_range_dict:
                    model_range_dict[model_spec] = (range_low, range_max)
                else:
                    curr_min, curr_max = model_range_dict[model_spec]
                    new_min = curr_min if curr_min < range_low else range_low
                    new_max = curr_max if curr_max > range_max else range_max
                    model_range_dict[model_spec] = (new_min, new_max)
    return model_range_dict


#%% Refactored functions for the top-level code
def load_data(train_molec_file: str, train_batch_file: str, 
                       valid_molec_file: str, valid_batch_file: str,
                       ref_train_data: str, ref_valid_data: str,
                       train_dftblsts: str, valid_dftblsts: str, 
                       ragged_dipole: bool = True, run_check: bool = True) -> (List[Dict], List[Dict], List[DFTBList], List[DFTBList]):
    r"""Loads the trainng and validation data saved in h5 files
    
    Arguments:
        train_molec_file (str): h5 file containing the per-molecule information for the 
            training molecules
        train_batch_file (str): h5 file containing the per-batch information for the 
            training batches
        valid_molec_file (str): h5 file containing the per-molecule information for the 
            validation molecules
        valid_batch_file (str): h5 file containing the per-batch information for the 
            validation batches
        ref_train_data (str): h5 file containing the reference training data for comparison
        ref_valid_data (str): h5 file containing the reference validation data for comparison
        train_dftblsts (str): pickle file containing the training DFTBLists
        valid_dftblsts (str): pickle file containing the validation DFTBLists
        ragged_dipole (bool): Whether the dipole matrices are expected to be 
            ragged. Defaults to True
        run_check (bool): Whether or not to check if loaded data matches reference data. 
            Defaults to True
    
    Returns: 
        training_feeds (List[Dict]): The reconstructed training feeds
        validation_feeds (List[Dict]): The reconstructed validation feeds
        training_dftblsts (List[DFTBList]): The training DFTBLists
        validation_dftblsts (List[DFTBList]): The validation DFTBLists
    
    Notes: Right now, the DFTBList objects have to be stored in a pickle file 
        because there is no way to deconstruct them into an h5 friendly format
    """
    x = time.time()
    training_feeds = total_feed_combinator.create_all_feeds(train_batch_file, train_molec_file, ragged_dipole)
    validation_feeds = total_feed_combinator.create_all_feeds(valid_batch_file, valid_molec_file, ragged_dipole)
    print(f"{time.time() - x}")
    if run_check:
        compare_feeds(ref_train_data, training_feeds)
        compare_feeds(ref_valid_data, validation_feeds)
    
    training_dftblsts = pickle.load(open(train_dftblsts, "rb"))
    validation_dftblsts = pickle.load(open(valid_dftblsts, "rb"))
    
    return training_feeds, validation_feeds, training_dftblsts, validation_dftblsts

def dataset_sorting(dataset: List[Dict], prop_train: float, transfer_training: bool = False, transfer_train_params: Dict = None,
                    train_ener_per_heavy: bool = True) -> (List[Dict], List[Dict]):
    r"""Generates the training and validation sets of molecules
    
    Arguments:
        dataset (List[Dict]): Output of get_ani1data, all the raw molecular configurations to use
        prop_train (float): The proportion of the data that should be allocated as training data
        transfer_training (bool): Whether to do transfer training, i.e. training on up to some lower_limit 
            heavy atoms and testing on lower_limit+. Defaults to False
        transfer_train_params (Dict): A dictionary that contains the following information:
            test_set (str): One of 'pure' and 'impure'. 'pure' means a total separation
                of train and validation set in terms of heavy atoms; 'impure' means some 
                fraction of the molecules with up to lower_limit heavy atoms are added 
                to the validation set
            impure_ratio (float): The ratio of molecules from train set that should be moved
                to validation set instead
            lower_limit (int): Number of heavy atoms to train up to
        train_ener_per_heavy (bool): Whether to fit total energy per heavy atom. Defaults to True
    
    Returns:
        training_molecs (List[Dict]): List of dictionaries for the training molecules
        validation_molecs (List[Dict]): List of dictionaries for the validation molecules
    
    Notes: In previous versions of dataset sorting, a degeneracy rejection was performed.
        Now that eigenvalue broadening is applied in the backpropagation, the degeneracy
        rejection has been removed as a step of the computation.
    """
    
    cleaned_dataset = list()
    for index, item in enumerate(dataset, 0):
        cleaned_dataset.append(item[0])
            
    print('Number of total molecules', len(cleaned_dataset))

    if transfer_training:
        print("Transfer training dataset")
        # Separate into molecules with up to lower_limit heavy atoms and those with
        # more
        
        impure_ratio = transfer_train_params['impure_ratio']
        test_set = transfer_train_params['test_set']
        lower_limit = transfer_train_params['lower_limit']
        
        up_to_ll, more = list(), list()
        for molec in cleaned_dataset:
            zcount = collections.Counter(molec['atomic_numbers'])
            ztypes = list(zcount.keys())
            heavy_counts = [zcount[x] for x in ztypes if x > 1]
            num_heavy = sum(heavy_counts)
            if num_heavy > lower_limit:
                if train_ener_per_heavy: 
                    molec['targets']['Etot'] = molec['targets']['Etot'] / num_heavy
                more.append(molec)
            else:
                if train_ener_per_heavy:
                    molec['targets']['Etot'] = molec['targets']['Etot'] / num_heavy
                up_to_ll.append(molec)
    
        # Check whether test_set should be pure
        training_molecs, validation_molecs = None, None
        if test_set == 'pure':
            random.shuffle(up_to_ll)
            training_molecs = up_to_ll
            num_valid = (int(len(up_to_ll) / prop_train)) - len(up_to_ll)
            validation_molecs = random.sample(more, num_valid)
        elif test_set == 'impure':
            indices = [i for i in range(len(up_to_ll))]
            chosen_for_blend = set(random.sample(indices, int(len(up_to_ll) * impure_ratio)))
            training_molecs, blend_temp = list(), list()
            for ind, elem in enumerate(up_to_ll, 0):
                if ind not in chosen_for_blend:
                    training_molecs.append(elem)
                else:
                    blend_temp.append(elem)
            num_valid = (int(len(training_molecs) / prop_train)) - (len(training_molecs) + len(blend_temp))
            rest_temp = random.sample(more, num_valid)
            validation_molecs = blend_temp + rest_temp
            random.shuffle(validation_molecs)
    
    else:
        #Shuffle the dataset before feeding into data_loader
        print("Non-transfer training dataset")
        random.shuffle(cleaned_dataset)
        
        #Sample the indices that will be used for the training dataset randomly from the shuffled data
        indices = [i for i in range(len(cleaned_dataset))]
        sampled_indices = set(random.sample(indices, int(len(cleaned_dataset) * prop_train)))
        
        #Separate into the training and validation sets
        training_molecs, validation_molecs = list(), list()
        for i in range(len(cleaned_dataset)):
            molec = cleaned_dataset[i]
            if train_ener_per_heavy:
                zcount = collections.Counter(molec['atomic_numbers'])
                ztypes = list(zcount.keys())
                heavy_counts = [zcount[x] for x in ztypes if x > 1]
                num_heavy = sum(heavy_counts)
                molec['targets']['Etot'] = molec['targets']['Etot'] / num_heavy
            if i in sampled_indices:
                training_molecs.append(molec)
            else:
                validation_molecs.append(molec)
    
    print(f'Number of molecules used for training: {len(training_molecs)}')
    print(f"Number of molecules used for testing: {len(validation_molecs)}")
    
    return training_molecs, validation_molecs

def graph_generation(molecules: List[Dict], config: Dict, allowed_Zs: List[int], 
                     par_dict: Dict, num_per_batch: int = 10) -> (List[Dict], List[DFTBList], List[List[Dict]]):
    r"""Generates the initial feed dictionaries for the given set of molecules
    
    Arguments:
        molecules (List[Dict]): Set of molecules to generate feed dicts for
        config (Dict): A dictionary containing some parameters for how the 
            feed dictionaries should be generated. It should contain the 'opers_to_model', 
            which is a list of strings representing which operators to model
        allowed_Zs (List[int]): List of integers representing the allowed elements 
            in the dataset
        par_dict (Dict): Dictionary of skf parameters indexed by atom pairs (e.g. 'C-C')
        num_per_batch (int): Number of molecules per feed dictionary
    
    Returns:
        feeds (List[Dict]): The feed dictionaries
        feed_dftblsts (List[DFTBList]): The DFTBLists for each feed dictionary
        feed_batches (List[List[Dict]]): The original molecules used to generate
            each feed dictionary
    
    Notes: None
    """
    train_dat_set = data_loader(molecules, batch_size = num_per_batch)
    feeds, feed_dftblsts = list(), list()
    feed_batches = list()
    for index, batch in enumerate(train_dat_set):
        feed, batch_dftb_lst = create_graph_feed(config, batch, allowed_Zs, par_dict)
        all_bsizes = list(feed['Eelec'].keys())
        
        # Better organization for saved names and config numbers
        feed['names'] = dict()
        feed['iconfigs'] = dict()
        for bsize in all_bsizes:
            glabels = feed['glabels'][bsize]
            all_names = [batch[x]['name'] for x in glabels]
            all_configs = [batch[x]['iconfig'] for x in glabels]
            feed['names'][bsize] = all_names
            feed['iconfigs'][bsize] = all_configs
                        
        feeds.append(feed)
        feed_dftblsts.append(batch_dftb_lst)
        feed_batches.append(batch) #Save the molecules to be used later for generating feeds
    
    return feeds, feed_dftblsts, feed_batches

def model_loss_initialization(training_feeds: List[Dict], validation_feeds: List[Dict], allowed_Zs: List[int], losses: Dict) -> tuple:
    r"""Initializes the losses and generates the models and model_variables dictionaries
    
    Arguments:
        training_feeds (List[Dict]): The training feed dictionaries
        validation_feeds (List[Dict]): The validation feed dictionaries
        allowed_Zs (List[int]): The atomic numbers of allowed elements
        losses (Dict): Dictionary of the targets and their target accuracies
    
    Returns:
        all_models (Dict): Dictionary that will contain references to all models
        model_variables (Dict): Dictionary that will contain references to all 
            model variables
        loss_tracker (Dict): Dictionary to keep track of the losses for each
            target
        all_losses (Dict): Dictionary containing references to all loss models used
            to compute losses for each target
        model_range_dict (Dict): Dictionary mapping each model to the range of 
            distances that it spans
    
    Notes: Everything is aliased so that the same set of models is being optimized
        across all feed dictionaries.
    """
    all_models = dict()
    model_variables = dict() #This is used for the optimizer later on
    
    all_models['Eref'] = Reference_energy(allowed_Zs)
    model_variables['Eref'] = all_models['Eref'].get_variables()
    
    #More nuanced construction of config dictionary
    model_range_dict = create_spline_config_dict(training_feeds + validation_feeds)
    
    #Constructing the losses using the models implemented in loss_models
    all_losses = dict()
    
    #loss_tracker to keep track of values for each 
    #Each loss maps to tuple of two lists, the first is the validation loss,the second
    # is the training loss, and the third is a temp so that average losses for validation/train 
    # can be computed
    loss_tracker = dict() 
    
    for loss in losses:
        if loss == "Etot":
            all_losses['Etot'] = TotalEnergyLoss()
            loss_tracker['Etot'] = [list(), list(), 0]
        elif loss in ["convex", "monotonic", "smooth"]:
            all_losses[loss] = FormPenaltyLoss(loss)
            loss_tracker[loss] = [list(), list(), 0]
        elif loss == "dipole":
            all_losses['dipole'] = DipoleLoss2() #Use DipoleLoss2 for dipoles computed from ESP charges!
            loss_tracker['dipole'] = [list(), list(), 0]
        elif loss == "charges":
            all_losses['charges'] = ChargeLoss()
            loss_tracker['charges'] = [list(), list(), 0]
    
    return all_models, model_variables, loss_tracker, all_losses, model_range_dict
    
def feed_generation(feeds: List[Dict], feed_batches: List[List[Dict]], all_losses: Dict, 
                    all_models: Dict, model_variables: Dict, model_range_dict: Dict, 
                    par_dict: Dict, debug: bool = False, loaded_data: bool = False) -> None:
    r"""Destructively adds all the necessary information to each feed dictionary
    
    Arguments:
        feeds (List[Dict]): List of feed dictionaries to add information to
        feed_batches (List[List[Dict]]): The original molecule batches used
            to generate each feed in feeds
        all_losses (Dict): Dictionary referencing each loss model to be used for
            computing losses per target
        all_models (Dict): Dictionary referencing all the models used
        model_variables (Dict): Dictionary referencing all the model variables
        model_range_dict (Dict): Dictionary of ranges for each model
        par_dict (Dict): Dictionary of the DFTB Slater-Koster parameters for atomic interactions 
            between different elements, indexed by a string 'elem1-elem2'. For example, the
            Carbon-Carbon interaction is accessed using the key 'C-C'
        debug (bool): Whether or not to use debug mode. Defaults to False
        loaded_data (bool): Whether or not using pre-loaded data. Defaults to False
    
    Returns:
        None
    
    Notes: None
    """
    for ibatch,feed in enumerate(feeds):
        for model_spec in feed['models']:
            print(model_spec)
            if (model_spec not in all_models):
                mod_res, tag = get_model_value_spline_2(model_spec, model_variables, model_range_dict, par_dict)
                all_models[model_spec] = mod_res
                #all_models[model_spec] = get_model_dftb(model_spec)
                if tag != 'noopt' and not isinstance(mod_res, OffDiagModel2):
                    # Do not add redundant variables for the OffDiagModel. Nothing
                    # is done for off-diagonal model variables
                    model_variables[model_spec] = all_models[model_spec].get_variables()
                # Detach it from the computational graph (unnecessary)
                elif tag == 'noopt':
                    all_models[model_spec].variables.requires_grad = False
            model = all_models[model_spec]
            feed[model_spec] = model.get_feed(feed['mod_raw'][model_spec])
        
        for loss in all_losses:
            try:
                all_losses[loss].get_feed(feed, [] if loaded_data else feed_batches[ibatch], all_models, par_dict, debug)
            except Exception as e:
                print(e)

def saving_data(training_feeds: List[Dict], validation_feeds: List[Dict], 
                training_dftblsts: List[DFTBList], validation_dftblsts: List[DFTBList],
                molec_names: List[str], batch_names: List[str], ref_names: List[str],
                dftblst_names: List[str]) -> None:
    r"""Saves all the information for the training data to h5 files
    
    Arguments:
        training_feeds (List[Dict]): The training feed dictionaries to save
        validation_feeds (List[Dict]): The validation feed dictionaries to save
        training_dftblsts (List[DFTBList]): DFTBList objects for each feed in training_feeds
        validation_dftblsts (List[DFTBList]): DFTBList objects for eachfeed in validation_feeds
        molec_names (List[str]): The names for the files for saving the per-molecule
            information. The first name is for the training molecules, the second is for the 
            validation molecules
        batch_names (List[str]): The names for the files for saving the batch information. The 
            first name is for the training batches, the second is for the validation batches
        ref_names (List[str]): The names for the files for saving the original data. The 
            first name is for the training data, the second is for the validation data
        dftblst_names (List[str]): The names for the files for saving the dftblsts. The 
            first name is for the training dftblsts, the second is for the validation dftblsts
    
    Returns:
        None
    
    Notes: Saving the dftblsts requires a pickle file while the others information is saved to 
        h5 files.
    """
    per_molec_h5handler.save_all_molec_feeds_h5(training_feeds, molec_names[0])
    per_batch_h5handler.save_multiple_batches_h5(training_feeds, batch_names[0])
    
    per_molec_h5handler.save_all_molec_feeds_h5(validation_feeds, molec_names[1])
    per_batch_h5handler.save_multiple_batches_h5(validation_feeds, batch_names[1])
    
    with open(ref_names[0]) as handle:
        pickle.dump(training_feeds, handle)
    
    with open(ref_names[1]) as handle:
        pickle.dump(validation_feeds, handle)
    
    with open(dftblst_names[0]) as handle:
        pickle.dump(training_dftblsts, handle)
    
    with open(dftblst_names[1]) as handle:
        pickle.dump(validation_dftblsts, handle)
    
    print("Molecular and batch information saved successfully, along with reference data")

def total_type_conversion(training_feeds: List[Dict], validation_feeds: List[Dict], ignore_keys: List[str]) -> None:
    r"""Does a recursive type conversion for each feed in training_feeds and each feed in validation_feeds
    
    Arguments:
        training_feeds (List[Dict]): The training feed dictionaries to do the correction for
        validation_feeds (List[Dict]): The validation feed dictionaries to do the correction for
        ignore_keys (List[str]): The keys to ignore the type conversion for
    
    Returns:
        None
    
    Notes: None
    """
    for feed in training_feeds:
        recursive_type_conversion(feed, ignore_keys)
    for feed in validation_feeds:
        recursive_type_conversion(feed, ignore_keys)

#%% Top level variable declaration
if __name__ == "__main__":
    '''
    If loading data from h5 files, make sure to note the allowed_Zs and heavy_atoms of the dataset and
    set them accordingly!
    '''
    allowed_Zs = [1,6,7,8]
    heavy_atoms = [1,2,3,4,5]
    #Still some problems with oxygen, molecules like HNO3 are problematic due to degeneracies
    max_config = 10
    # target = 'dt'
    target = {'Etot' : 'dt',
              'dipole' : 'wb97x_dz.dipole',
              'charges' : 'wb97x_dz.cm5_charges'}
    exclude = ['O3', 'N2O1', 'H1N1O3']
    # Parameters for configuring the spline
    num_knots = 50
    max_val = None
    num_per_batch = 10
    
    #Method for eigenvalue decomposition
    eig_method = 'new'
    
    #Proportion for training and validation
    prop_train = 0.8
    prop_valid = 0.2
    
    reference_energies = list() # Save the reference energies to see how the losses are really changing
    training_losses = list()
    validation_losses = list()
    times = collections.OrderedDict()
    times['init'] = time.process_time()
    dataset = get_ani1data(allowed_Zs, heavy_atoms, max_config, target, exclude=exclude)
    times['dataset'] = time.process_time()
    print('number of molecules retrieved', len(dataset))
    
    config = dict()
    config['opers_to_model'] = ['H', 'R', 'G'] #This actually matters now
    
    #loss weights
    losses = dict()
    target_accuracy_energy = 6270 #Ha^-1
    target_accuracy_dipole = 100 # debye
    target_accuracy_charges = 100
    target_accuracy_convex = 1000
    target_accuracy_monotonic = 1000
    
    losses['Etot'] = target_accuracy_energy
    losses['dipole'] = target_accuracy_dipole 
    losses['charges'] = target_accuracy_charges #Not working on charge loss just yet
    losses['convex'] = target_accuracy_convex
    losses['monotonic'] = target_accuracy_monotonic
    
    #Initialize the parameter dictionary
    par_dict = ParDict()
    
    #Compute or load?
    loaded_data = False
    
    #Training scheme
    # If this flag is set to true, the dataset will be changed such that you 
    # train on up to lower_limit heavy atoms and test on the rest
    
    # If test_set is set to 'pure', then the test set will only have molecules with
    # more than lower_limit heavy atoms; otherwise, test set will have a blend of 
    # molecules between those with up to lower_limit heavy atoms and those with more
    
    # impure_ratio indicates what fraction of the molecules found with up to lower_limit
    # heavy atoms should be added to the test set if the test_set is not 'pure'
    transfer_training = False
    test_set = 'pure' #either 'pure' or 'impure'
    impure_ratio = 0.2
    lower_limit = 4
    
    # Flag indicates whether or not to fit to the total energy per molecule or the 
    # total energy as a function of the number of heavy atoms. 
    train_ener_per_heavy = True
    
    # Debug flag. If set to true, get_feeds() for the loss models adds data based on
    # dftb results rather than from ANI-1
    # Note that for total energy, debug mode gives total energy per molecule,
    # NOT total energy per heavy atom!
    debug = False
    
    # debug and train_ener_per_heavy should be opposite
    assert(not(debug and train_ener_per_heavy))
    
    # Flag indicating whether or not to include the dipole in backprop
    include_dipole_backprop = True
    
    #%% Degbugging h5 (Extraction and combination)
    x = time.time()
    training_feeds = total_feed_combinator.create_all_feeds("final_batch_test.h5", "final_molec_test.h5", True)
    validation_feeds = total_feed_combinator.create_all_feeds("final_valid_batch_test.h5", "final_valid_molec_test.h5", True)
    print(f"{time.time() - x}")
    compare_feeds("reference_data1.p", training_feeds)
    compare_feeds("reference_data2.p", validation_feeds)
    
    training_molec_batches = []
    validation_molec_batches = []
    
    #Need to regenerate the molecule batches for both train and validation
    # master_train_molec_dict = per_molec_h5handler.extract_molec_feeds_h5("final_molec_test.h5")
    # master_valid_molec_dict = per_molec_h5handler.extract_molec_feeds_h5("final_valid_molec_test.h5")
    
    # #Reconstitute the lists 
    # training_molec_batches = per_molec_h5handler.create_molec_batches_from_feeds_h5(master_train_molec_dict,
    #                                                                         training_feeds, ["Etot", "dipoles", "charges"])
    # validation_molec_batches = per_molec_h5handler.create_molec_batches_from_feeds_h5(master_valid_molec_dict,
    #                                                                         validation_feeds, ["Etot", "dipoles", "charges"])
    
    #Load dftb_lsts
    training_dftblsts = pickle.load(open("training_dftblsts.p", "rb"))
    validation_dftblsts = pickle.load(open("validation_dftblsts.p", "rb"))
    
    print("Check me!")
    
    #%% Dataset Sorting
    print("Running degeneracy rejection")
    degeneracy_tolerance = 1.0e-3
    bad_indices = set()
    # NOTE: uncomment this section if using torch.symeig; if using new symeig, 
    #       can leave this step out
    # for index, batch in enumerate(dataset, 0):
    #     try:
    #         feed, _ = create_graph_feed(config, batch, allowed_Zs)
    #         eorb = list(feed['eorb'].values())[0]
    #         degeneracy = np.min(np.diff(np.sort(eorb)))
    #         if degeneracy < degeneracy_tolerance:
    #             bad_indices.add(index)
    #     except:
    #         print(batch[0]['name'])
    
    cleaned_dataset = list()
    for index, item in enumerate(dataset, 0):
        if index not in bad_indices:
            cleaned_dataset.append(item[0])
    
    print('Number of total molecules after degeneracy rejection', len(cleaned_dataset))
    
    if transfer_training:
        print("Transfer training dataset")
        # Separate into molecules with up to lower_limit heavy atoms and those with
        # more
        up_to_ll, more = list(), list()
        for molec in cleaned_dataset:
            zcount = collections.Counter(molec['atomic_numbers'])
            ztypes = list(zcount.keys())
            heavy_counts = [zcount[x] for x in ztypes if x > 1]
            num_heavy = sum(heavy_counts)
            if num_heavy > lower_limit:
                if train_ener_per_heavy: 
                    molec['targets']['Etot'] = molec['targets']['Etot'] / num_heavy
                more.append(molec)
            else:
                if train_ener_per_heavy:
                    molec['targets']['Etot'] = molec['targets']['Etot'] / num_heavy
                up_to_ll.append(molec)
        
        # Check whether test_set should be pure
        training_molecs, validation_molecs = None, None
        if test_set == 'pure':
            random.shuffle(up_to_ll)
            training_molecs = up_to_ll
            num_valid = (int(len(up_to_ll) / prop_train)) - len(up_to_ll)
            validation_molecs = random.sample(more, num_valid)
        elif test_set == 'impure':
            indices = [i for i in range(len(up_to_ll))]
            chosen_for_blend = set(random.sample(indices, int(len(up_to_ll) * impure_ratio)))
            training_molecs, blend_temp = list(), list()
            for ind, elem in enumerate(up_to_ll, 0):
                if ind not in chosen_for_blend:
                    training_molecs.append(elem)
                else:
                    blend_temp.append(elem)
            num_valid = (int(len(training_molecs) / prop_train)) - (len(training_molecs) + len(blend_temp))
            rest_temp = random.sample(more, num_valid)
            validation_molecs = blend_temp + rest_temp
            random.shuffle(validation_molecs)
    else:
        #Shuffle the dataset before feeding into data_loader
        print("Non-transfer training dataset")
        random.shuffle(cleaned_dataset)
        
        #Sample the indices that will be used for the training dataset randomly from the shuffled data
        indices = [i for i in range(len(cleaned_dataset))]
        sampled_indices = set(random.sample(indices, int(len(cleaned_dataset) * prop_train)))
        
        #Separate into the training and validation sets
        training_molecs, validation_molecs = list(), list()
        for i in range(len(cleaned_dataset)):
            molec = cleaned_dataset[i]
            if train_ener_per_heavy:
                zcount = collections.Counter(molec['atomic_numbers'])
                ztypes = list(zcount.keys())
                heavy_counts = [zcount[x] for x in ztypes if x > 1]
                num_heavy = sum(heavy_counts)
                molec['targets']['Etot'] = molec['targets']['Etot'] / num_heavy
            if i in sampled_indices:
                training_molecs.append(molec)
            else:
                validation_molecs.append(molec)
    
    #Logging data
    total_num_molecs = len(cleaned_dataset)
    total_num_train_molecs = len(training_molecs)
    total_num_valid_molecs = len(validation_molecs)
    
    #Now run through the graph and feed generation procedures for both the training
    #   and validation molecules
    
    #NOTE: The order of the geometries in feed corresponds to the order of the 
    # geometries in batch, i.e. the glabels match the indices of batch (everything
    # is added sequentially)
    
    # Can go based on the order of the 'glabels' key in feeds, which dictates the 
    # ordering for everything as a kvp with bsize -> values for each molecule, glabels are sorted
    print(f'Number of molecules used for training: {len(training_molecs)}')
    print(f"Number of molecules used for testing: {len(validation_molecs)}")
    #%% Graph generation
    x = time.time()
    print("Making Training Graphs")
    train_dat_set = data_loader(training_molecs, batch_size = num_per_batch)
    training_feeds, training_dftblsts = list(), list()
    training_molec_batches = list()
    for index, batch in enumerate(train_dat_set):
        feed, batch_dftb_lst = create_graph_feed(config, batch, allowed_Zs, par_dict)
        all_bsizes = list(feed['Eelec'].keys())
        
        # Better organization for saved names and config numbers
        feed['names'] = dict()
        feed['iconfigs'] = dict()
        for bsize in all_bsizes:
            glabels = feed['glabels'][bsize]
            all_names = [batch[x]['name'] for x in glabels]
            all_configs = [batch[x]['iconfig'] for x in glabels]
            feed['names'][bsize] = all_names
            feed['iconfigs'][bsize] = all_configs
                        
        training_feeds.append(feed)
        training_dftblsts.append(batch_dftb_lst)
        training_molec_batches.append(batch) #Save the molecules to be used later for generating feeds
    
    print("Making Validation Graphs")
    validation_dat_set = data_loader(validation_molecs, batch_size = num_per_batch)
    validation_feeds, validation_dftblsts = list(), list()
    validation_molec_batches = list()
    for index, batch in enumerate(validation_dat_set):
        feed, batch_dftb_lst = create_graph_feed(config, batch, allowed_Zs, par_dict)
        all_bsizes = list(feed['Eelec'].keys())
        
        feed['names'] = dict()
        feed['iconfigs'] = dict()
        for bsize in all_bsizes:
            glabels = feed['glabels'][bsize]
            all_names = [batch[x]['name'] for x in glabels]
            all_configs = [batch[x]['iconfig'] for x in glabels]
            feed['names'][bsize] = all_names
            feed['iconfigs'][bsize] = all_configs
    
        validation_feeds.append(feed)
        validation_dftblsts.append(batch_dftb_lst) #Save the validation dftblsts for charge updates on the validation set
        validation_molec_batches.append(batch)
    print(f"{time.time() - x}")
    #%% Model and loss initialization
    all_models = dict()
    model_variables = dict() #This is used for the optimizer later on
    
    all_models['Eref'] = Reference_energy(allowed_Zs)
    model_variables['Eref'] = all_models['Eref'].get_variables()
    
    #More nuanced construction of config dictionary
    model_range_dict = create_spline_config_dict(training_feeds + validation_feeds)
    
    #Constructing the losses using the models implemented in loss_models
    all_losses = dict()
    
    #loss_tracker to keep track of values for each 
    #Each loss maps to tuple of two lists, the first is the validation loss,the second
    # is the training loss, and the third is a temp so that average losses for validation/train 
    # can be computed
    loss_tracker = dict() 
    
    for loss in losses:
        if loss == "Etot":
            all_losses['Etot'] = TotalEnergyLoss()
            loss_tracker['Etot'] = [list(), list(), 0]
        elif loss in ["convex", "monotonic", "smooth"]:
            all_losses[loss] = FormPenaltyLoss(loss)
            loss_tracker[loss] = [list(), list(), 0]
        elif loss == "dipole":
            all_losses['dipole'] = DipoleLoss2() #Use DipoleLoss2 for dipoles computed from ESP charges!
            loss_tracker['dipole'] = [list(), list(), 0]
        elif loss == "charges":
            all_losses['charges'] = ChargeLoss()
            loss_tracker['charges'] = [list(), list(), 0]
    
    #%% Feed generation
    x = time.time()
    print('Making training feeds')
    for ibatch,feed in enumerate(training_feeds):
       for model_spec in feed['models']:
           if (model_spec not in all_models):
               mod_res, tag = get_model_value_spline_2(model_spec, model_variables, model_range_dict, par_dict)
               all_models[model_spec] = mod_res
               #all_models[model_spec] = get_model_dftb(model_spec)
               if tag != 'noopt' and not isinstance(mod_res, OffDiagModel):
                   model_variables[model_spec] = all_models[model_spec].get_variables()
               # Detach it from the computational graph (unnecessary)
               elif tag == 'noopt':
                   all_models[model_spec].variables.requires_grad = False
           model = all_models[model_spec]
           feed[model_spec] = model.get_feed(feed['mod_raw'][model_spec])
       
       for loss in all_losses:
           try:
               all_losses[loss].get_feed(feed, [] if loaded_data else training_molec_batches[ibatch], all_models, par_dict, debug)
           except Exception as e:
               print(e)
    
    
    print('Making validation feeds')
    for ibatch, feed in enumerate(validation_feeds):
        for model_spec in feed['models']:
            if (model_spec not in all_models):
                mod_res, tag = get_model_value_spline_2(model_spec, model_variables, model_range_dict, par_dict)
                all_models[model_spec] = mod_res
                #all_models[model_spec] = get_model_dftb(model_spec)
                if tag != 'noopt' and not isinstance(mod_res, OffDiagModel):
                    model_variables[model_spec] = all_models[model_spec].get_variables()
                elif tag == 'noopt':
                    all_models[model_spec].variables.requires_grad = False
            model = all_models[model_spec]
            feed[model_spec] = model.get_feed(feed['mod_raw'][model_spec])
        
        for loss in all_losses:
            try:
                all_losses[loss].get_feed(feed, [] if loaded_data else validation_molec_batches[ibatch], all_models, par_dict, debug)
            except Exception as e:
                print(e)
    print(f"{time.time() - x}")
    #%% Debugging h5 (Saving)
    
    #Save all the molecular information
    per_molec_h5handler.save_all_molec_feeds_h5(training_feeds, 'final_molec_test.h5')
    per_batch_h5handler.save_multiple_batches_h5(training_feeds, 'final_batch_test.h5')
    
    per_molec_h5handler.save_all_molec_feeds_h5(validation_feeds, 'final_valid_molec_test.h5')
    per_batch_h5handler.save_multiple_batches_h5(validation_feeds, 'final_valid_batch_test.h5')
    
    with open("reference_data1.p", "wb") as handle:
        pickle.dump(training_feeds, handle)
    with open("reference_data2.p", "wb") as handle:
        pickle.dump(validation_feeds, handle)
        
    # Also save the dftb_lsts for the training_feeds and validation feeds. Can do this using pickle for now
    with open("training_dftblsts.p", "wb") as handle:
        pickle.dump(training_dftblsts, handle)
        
    with open("validation_dftblsts.p", "wb") as handle:
        pickle.dump(validation_dftblsts, handle)
        
    print("molecular and batch information successfully saved, along with reference data")
    
    #%% Debugging inflection point analysis
    g_mods = [mod for mod in all_models.keys() if mod != 'Eref' and mod.oper == 'G' and len(mod.Zs) == 2]
    num_per_plot = 4
    num_row = num_col = 2
    sections = [g_mods[i : i + num_per_plot] for i in range(0, len(g_mods), num_per_plot)]
    new_dict = dict()
    rgrid = np.linspace(0, 10, 1000) #dense grid
    for sect in sections:
        fig, axs = plt.subplots(num_row, num_col) #sqrt of num_per_plot
        pos = 0
        for row in range(num_row):
            for col in range(num_col):
                axs[row, col].plot(rgrid, get_dftb_vals(sect[pos], par_dict, rgrid))
                axs[row, col].set_title(f"{sect[pos].oper}, {sect[pos].Zs}, {sect[pos].orb}")
                pos += 1
        fig.tight_layout()
        #save the figure...
        plt.show()
    
    #%% Recursive type conversion
    # Not an elegant solution but these two keys need to be ignored since they
    # should not be tensors!
    # Charges are ignored because of raggedness coming from bsize organization
    
    #If you are using the second version of dipole loss, ignore the dipole_mats too
    # because they are going to be a list of arrays
    ignore_keys = ['glabels', 'basis_sizes', 'charges', 'dipole_mat']
    
    for feed in training_feeds:
        recursive_type_conversion(feed, ignore_keys)
    for feed in validation_feeds:
        recursive_type_conversion(feed, ignore_keys)
    times['feeds'] = time.process_time()
    
    #%% Training loop
    '''
    Two different eig methods are available for the dftblayer now, and they are 
    denoted by flags 'new' and 'old'.
        'new': Implemented eigenvalue broadening method to work around vanishing 
        eigengaps, refer to eig.py for more details. Only using conditional broadening
        to cut down on gradient errors. Broadening factor is 1E-12.
        
        'old': Implementation using torch.symeig, standard approach from before
    
    Note: If you are using the old method for symmetric eigenvalue decomp, make sure
    to uncomment the section that runs the degeneracy rejection! Diagonalization will fail for 
    degenerate molecules in the old method.
    '''
    dftblayer = DFTB_Layer(device = None, dtype = torch.double, eig_method = eig_method)
    learning_rate = 1.0e-5
    optimizer = optim.Adam(list(model_variables.values()), lr = learning_rate, amsgrad=True)
    #TODO: Experiment with alternative learning rate schedulers
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 10, threshold = 0.01) 
    
    #Instantiate the loss layer here
    
    times_per_epoch = list()
    
    nepochs = 300
    for i in range(nepochs):
        #Initialize epoch timer
        start = time.time()
        
        #Validation routine
        #Comment out, testing new loss
        validation_loss = 0
        for elem in validation_feeds:
            with torch.no_grad():
                output = dftblayer(elem, all_models)
                # loss = loss_layer.get_loss(output, elem)
                tot_loss = 0
                for loss in all_losses:
                    if loss == 'Etot':
                        if train_ener_per_heavy:
                            val = losses[loss] * all_losses[loss].get_value(output, elem, True)
                        else:
                            val = losses[loss] * all_losses[loss].get_value(output, elem, False)
                        tot_loss += val
                        loss_tracker[loss][2] += val.item()
                    elif loss == 'dipole':
                        val = losses[loss] * all_losses[loss].get_value(output, elem)
                        loss_tracker[loss][2] += val.item()
                        if include_dipole_backprop:
                            tot_loss += val
                    else:
                        val = losses[loss] * all_losses[loss].get_value(output, elem)
                        tot_loss += val 
                        loss_tracker[loss][2] += val.item()
                validation_loss += tot_loss.item()
        print("Validation loss:",i, (validation_loss/len(validation_feeds)))
        validation_losses.append((validation_loss/len(validation_feeds)))
        
        for loss in all_losses:
            loss_tracker[loss][0].append(loss_tracker[loss][2] / len(validation_feeds))
            #Reset the loss tracker after being done with all feeds
            loss_tracker[loss][2] = 0
    
        #Shuffle the validation data
        # random.shuffle(validation_feeds)
        temp = list(zip(validation_feeds, validation_dftblsts))
        random.shuffle(temp)
        validation_feeds, validation_dftblsts = zip(*temp)
        validation_feeds, validation_dftblsts = list(validation_feeds), list(validation_dftblsts)
        
        #Training routine
        epoch_loss = 0.0
        for feed in training_feeds:
            optimizer.zero_grad()
            output = dftblayer(feed, all_models)
            #Comment out, testing new loss
            # loss = loss_layer.get_loss(output, feed) #Loss still in units of Ha^2 ?
            tot_loss = 0
            for loss in all_losses:
                if loss == 'Etot':
                    if train_ener_per_heavy:
                        val = losses[loss] * all_losses[loss].get_value(output, feed, True)
                    else:
                        val = losses[loss] * all_losses[loss].get_value(output, feed, False)
                    tot_loss += val
                    loss_tracker[loss][2] += val.item()
                elif loss == 'dipole':
                    val = losses[loss] * all_losses[loss].get_value(output, feed)
                    loss_tracker[loss][2] += val.item()
                    if include_dipole_backprop:
                        tot_loss += val
                else:
                    val = losses[loss] * all_losses[loss].get_value(output, feed)
                    tot_loss += val
                    loss_tracker[loss][2] += val.item()
    
            epoch_loss += tot_loss.item()
            tot_loss.backward()
            optimizer.step()
        scheduler.step(epoch_loss) #Step on the epoch loss
        
        #Perform shuffling while keeping order b/w dftblsts and feeds consistent
        temp = list(zip(training_feeds, training_dftblsts))
        random.shuffle(temp)
        training_feeds, training_dftblsts = zip(*temp)
        training_feeds, training_dftblsts = list(training_feeds), list(training_dftblsts)
        
        print(i, (epoch_loss/len(training_feeds)))
        training_losses.append((epoch_loss/len(training_feeds)))
        
        for loss in all_losses:
            loss_tracker[loss][1].append(loss_tracker[loss][2] / len(training_feeds))
            loss_tracker[loss][2] = 0
        
        # Update charges every 10 epochs
        # Do the charge update for the validation and the training sets
        if (i % 10 == 0):
            print("running training set charge update")
            for j in range(len(training_feeds)):
                # Charge update for training_feeds
                feed = training_feeds[j]
                dftb_list = training_dftblsts[j]
                op_dict = assemble_ops_for_charges(feed, all_models)
                try:
                    update_charges(feed, op_dict, dftb_list)
                except Exception as e:
                    print(e)
                    glabels = feed['glabels']
                    basis_sizes = feed['basis_sizes']
                    result_lst = []
                    for bsize in basis_sizes:
                        result_lst += list(zip(feed['names'][bsize], feed['iconfigs'][bsize]))
                    print("Charge update failed for")
                    print(result_lst)
            print("training charge update done, doing validation set")
            for k in range(len(validation_feeds)):
                # Charge update for validation_feeds
                feed = validation_feeds[k]
                dftb_list = validation_dftblsts[k]
                op_dict = assemble_ops_for_charges(feed, all_models)
                try:
                    update_charges(feed, op_dict, dftb_list)
                except Exception as e:
                    print(e)
                    glabels = feed['glabels']
                    basis_sizes = feed['basis_sizes']
                    result_lst = []
                    for bsize in basis_sizes:
                        result_lst += list(zip(feed['names'][bsize], feed['iconfigs'][bsize]))
                    print("Charge update failed for")
                    print(result_lst)
            print(f"charge updates done for epoch {i}")
        #Save timing information for diagnostics
        times_per_epoch.append(time.time() - start)
    
    print(f"Finished with {nepochs} epochs")
    times['train'] = time.process_time()
    #%% Logging
    print('dataset with', len(training_feeds), 'batches')
    time_names  = list(times.keys())
    time_vals  = list(times.values())
    for itime in range(1,len(time_names)):
        if time_names[itime] == 'train':
            print(time_names[itime], (time_vals[itime] - time_vals[itime-1])/nepochs)
        else:
            print(time_names[itime], time_vals[itime] - time_vals[itime-1])
    
    #Save the training and validation losses for visualization later
    with open("losses.p", "wb") as handle:
        pickle.dump(training_losses, handle)
        pickle.dump(validation_losses, handle)
    
    print(f"total time taken (sum epoch times): {sum(times_per_epoch)}")
    print(f"average epoch time: {sum(times_per_epoch) / len(times_per_epoch)}")
    # print(f"total number of molecules per epoch: {total_num_molecs}")
    # print(f"total number of training molecules: {total_num_train_molecs}")
    # print(f"total number of validation molecules: {total_num_valid_molecs}")
    
    #Plotting the change in each kind of loss per epoch
    for loss in all_losses:
        validation_loss = loss_tracker[loss][0]
        training_loss = loss_tracker[loss][1]
        # assert(len(validation_loss) == nepochs)
        # assert(len(training_loss) == nepochs)
        fig, axs = plt.subplots()
        axs.plot(training_loss, label = 'Training loss')
        axs.plot(validation_loss, label = 'Validation loss')
        axs.set_title(f"{loss} loss")
        axs.set_xlabel("Epoch")
        axs.set_ylabel("Average Epoch Loss (unitless)")
        axs.yaxis.set_minor_locator(AutoMinorLocator())
        axs.xaxis.set_minor_locator(AutoMinorLocator())
        axs.legend()
        plt.show()
        
    from loss_methods import plot_multi_splines
    double_mods = [mod for mod in all_models.keys() if mod != 'Eref' and len(mod.Zs) == 2]
    plot_multi_splines(double_mods, all_models)
        
    # #Writing diagnostic information for later user
    # with open("timing.txt", "a+") as handle:
    #     handle.write(f"Current time: {datetime.now()}\n")
    #     handle.write(f"Allowed Zs: {allowed_Zs}\n")
    #     handle.write(f"Heavy Atoms: {heavy_atoms}\n")
    #     handle.write(f"Molecules per batch: {num_per_batch}\n")
    #     handle.write(f"Total molecules per epoch: {total_num_molecs}\n")
    #     handle.write(f"Total number of training molecules: {total_num_train_molecs}\n")
    #     handle.write(f"Total number of validation molecules: {total_num_valid_molecs}\n")
    #     handle.write(f"Number of epochs: {nepochs}\n")
    #     handle.write(f"Eigen decomp method: {eig_method}\n")
    #     handle.write(f"Total training time, sum of epoch times (seconds): {sum(times_per_epoch)}\n")
    #     handle.write(f"Average time per epoch (seconds): {sum(times_per_epoch) / len(times_per_epoch)}\n")
    #     handle.write("Infrequent charge updating for dipole loss\n")
    #     handle.write("Switched over to using non-shifted dataset\n")
    #     handle.write("Testing with new loss framework\n")
    #     handle.write("Testing dipole loss against actual dipoles\n")
    #     handle.write("\n")



