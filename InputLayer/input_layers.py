# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 16:28:36 2021

@author: fhu14

Code behind with all the input layers used by the model

TODO:
    1) Add in new repulsive model once DFTBrepulsive can be turned into a 
        package
"""
#%% Imports, definitions
import numpy as np
Array = np.ndarray
from typing import List, Dict
from collections import namedtuple
Model = namedtuple('Model',['oper', 'Zs', 'orb'])
RawData = namedtuple('RawData',['index','glabel','Zs','atoms','oper','orb','dftb','rdist'])
import torch
Tensor = torch.Tensor
from Spline import get_dftb_vals, fit_linear_model, SplineModel, JoinedSplineModel


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
            raise ValueError("Hubbard inconsistency detected!")
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
        model_vars,_,_ = fit_linear_model(self.pairwise_linear_model, rgrid,ygrid) #Model vars fit based on angstrom x-axis
        self.variables = torch.tensor(model_vars, dtype = self.dtype, device = self.device)
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

class Reference_energy:

    def __init__(self, allowed_Zs: List[int], device: torch.device, dtype: torch.dtype,
                 prev_values: List[float] = None) -> None:
        r"""Initializes the reference energy model.
        
        Arguments:
            allowed_Zs (List[int]): List of the allowed atomic numbers
            device (torch.device): The device to run the computations on (CPU vs GPU).
                If running on GPU, must be CUDA enabled GPU.
            dtype (torch.dtype): The torch datatype for the calculations
            prev_values (List[float]): Previous values for the reference energy to start from.
                Defaults to None
        
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
            
            An additional constant needs to be added since the reference energy 
            contains an additional constant term. Will work on adding this in, so that the 
            reference energy is computed as 
            
            Eref = Sum_z[C_z * N_z] + C_0, where the sum goes over all atom types z in the 
            dataset, C_z is the coefficient for element z, N_z is the number of that element 
            in the molecule, and C_0 is the additional coefficient.
        """
        self.dtype, self.device = dtype, device
        self.allowed_Zs = np.sort(np.array(allowed_Zs))
        self.values = np.zeros(self.allowed_Zs.shape)
        self.values = np.append(self.values, np.array([0]))
        if (not (prev_values is None)) and  len(prev_values) > 0:
            #Load previous values if they are given
            #FOR DEBUGGING PURPOSES ONLY
            assert(len(prev_values) == len(self.values))
            self.values = np.array(prev_values)
        self.variables = torch.tensor(self.values, dtype = self.dtype, device = self.device)
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

