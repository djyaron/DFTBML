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
    1) Smarter initialization for the inflection point value?
"""
import pdb, traceback, sys

import numpy as np
import random
import pickle
import math

from collections import OrderedDict
import collections
import torch
torch.set_printoptions(precision = 10)
from copy import deepcopy
import time

from Geometry import Geometry
from .eig import SymEigB
from .batch import create_batch, create_dataset, DFTBList

from MasterConstants import Model

from Spline import SplineModel, JoinedSplineModel
from DFTBpy import _Gamma12

from typing import List, Dict, Tuple
Tensor = torch.Tensor
Array = np.ndarray

from DataManager import per_molec_h5handler, per_batch_h5handler,\
    total_feed_combinator, compare_feeds, data_loader
from LossLayer import TotalEnergyLoss, FormPenaltyLoss, ChargeLoss, DipoleLoss
from InputLayer import Input_layer_DFTB, Input_layer_DFTB_val, Input_layer_hubbard,\
    Input_layer_pairwise_linear_joined, Input_layer_pairwise_linear, Input_layer_value,\
        Reference_energy

from .util import apx_equal, torch_segment_sum, recursive_type_conversion

def create_graph_feed(config: Dict, batch: List[Dict], allowed_Zs: List[int], par_dict: Dict) -> Tuple[Dict, DFTBList]:
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

def solve_for_inflect_var(rlow: float, rhigh: float, r_target: float) -> float:
    r"""Solves the inflection point equation for the variable x
    
    Arguments:
        rlow (float): lower bound of distance range
        rhigh (float): upper bound of distance range
        r_target (float): The target starting guess for the inflection point
    
    Returns:
        x_val (float): The value of the variable for computing the inflection point
    
    Notes: This function solves the following equation for x:
        r_target = rlow + ((rhigh - rlow) / 2) * ((atan(x)/pi/2) + 1)
    """
    first_term = r_target - rlow
    second_term = 2 / (rhigh - rlow)
    pi_const = math.pi / 2
    operand = ((first_term * second_term) - 1) * pi_const
    x_val = math.tan(operand)
    return x_val

def get_model_value_spline(model_spec: Model, model_variables: Dict, spline_dict: Dict, par_dict: Dict,
                             device: torch.device, dtype: torch.dtype,
                             num_knots: int = 50, buffer: float = 0.0, 
                             joined_cutoff: float = 3.0, cutoff_dict: Dict = None,
                             spline_mode: str = 'joined', spline_deg: int = 3,
                             off_diag_opers: List[str] = ["G"],
                             include_inflect: bool = True) -> Tuple[Input_layer_pairwise_linear_joined, str]:
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
        device (torch.device): The device to run the computations on (CPU vs GPU).
                If running on GPU, must be CUDA enabled GPU.
        dtype (torch.dtype): The torch datatype for the calculations
        num_knots (int): The number of knots to use for the spline. Defaults to 50
        buffer (float): The value to shift the starting and ending distance by. The starting distance
            is rlow - buffer, and the ending distance is rhigh + buffer, where rlow, rhigh are the 
            minimum and maximum distances for the spline from the spline_dict. Defaults to 0.0 angstroms
        joined_cutoff (float): The cutoff distance for separating the variable and fixed
            regions of the spline. Defaults to 3.0 angstroms
        cutoff_dict (Dict): Dictionary of cutoffs to use for each joined spline model, indexed
            by the element pair. Defaults to None, in which case the joined_cutoff default of 3.0 angstroms is used
        spline_mode (str): The type of spline to use, one of 'joined', 'non-joined'. Defaults to 'joined'
        spline_deg (int): The degree of splines to use. Defaults to 3
        off_diag_opers (List[str]): A list of the operators to model using the 
            Input_layer_hubbard. Defaults to ['G']
        include_inflect (bool): Whether or not to use inflection point variables for the splines.
            Defaults to True, in which case it is used
    
    Returns:
        model (Input_layer_pairwise_linear_joined): The instance of the Input_layer_pairwise_linear_joined
            object for working with the joined spline
        tag (str): A tag indicating whether to optimize the spline ('opt') or to ignore the spline in
            optimizations ('noopt')
    
    Notes: The tag is included because if a spline is flat at 0 (i.e., no interactions), then the 
        spline is not optimized. 
        
        The inclusion of model_variables is for the OffDiag2 model. It is assumed that by the time
        a two-body interaction is encountered (e.g. G, (1,6), sp) that all the one-body interactions for
        that model have been handled. Right now, it seems like that is a safe assumption.
        
        Specifying the degree of the splines only affects the non-joined splines, as joined-splines only work
        with third degree splines
        
        The cutoff_dict entries should be of the following format:
            (oper, (elem1, elem2)) : cutoff
        
        This way, all the models using the given oper and elems will have the same cutoff
        
        TODO: Constrain the knots to be between 0 and cutoff, anything past that is 0. Need to apply 
            this fix for everyting spline-related (inflection point, form penalty, etc.)
    """
    noise_magnitude = 0.0
    if len(model_spec.Zs) == 1:
        if spline_mode != 'debugging':
            if (model_spec.oper not in off_diag_opers) or ( (len(model_spec.orb) == 2) and (model_spec.orb[0] == model_spec.orb[1]) ):
                print(f"Input to value layer: {model_spec}")
                model = Input_layer_value(model_spec, device, dtype)
                model.initialize_to_dftb(par_dict, noise_magnitude)
            else:
                print(f"Input to hubbard layer: {model_spec}")
                model = Input_layer_hubbard(model_spec, model_variables, device, dtype)
            #REMOVE THIS LINE LATER
            # model = Input_layer_DFTB_val(model_spec)
        else:
            print("Using a debugging model for on-diagonal elements")
            model = Input_layer_DFTB_val(model_spec)
        return (model, 'val')
    elif len(model_spec.Zs) == 2:
        if model_spec.oper not in off_diag_opers:
            minimum_value, maximum_value = spline_dict[model_spec]
            minimum_value -= buffer
            maximum_value += buffer
            xknots = np.linspace(minimum_value, maximum_value, num = num_knots) #Set maximum_value to cutoff, not 10 angstroms
            print(f"number of knots: {len(xknots)}")
            print(f"first knot: {xknots[0]}, second knot: {xknots[-1]}")
            
            #Get the correct cutoff for the model
            # if cutoff_dict is not None:
            #     Zs, Zs_rev = model_spec.Zs, (model_spec.Zs[1], model_spec.Zs[0])
            #     oper = model_spec.oper
            #     if (oper, Zs) in cutoff_dict:
            #         model_cutoff = cutoff_dict[(oper, Zs)]
            #     elif (oper, Zs_rev) in cutoff_dict:
            #         model_cutoff = cutoff_dict[(oper, Zs_rev)]
            #     else:
            #         model_cutoff = joined_cutoff
            # else:
            #     model_cutoff = joined_cutoff
            
            #model_cutoff is now the position of the last knot (i.e. maximum_value)
            model_cutoff = maximum_value
            print(f"model_cutoff: {model_cutoff}")
            assert(model_cutoff == xknots[-1])
            
            config = {'xknots' : xknots,
                      'equal_knots' : False,
                      'cutoff' : model_cutoff,
                      'bconds' : 'last_only', #last_only applies boundary conditions on the final knot only
                      'deg' : spline_deg,
                      'max_der' : 2} #Hard-code to get the second derivative as the maximum derivative
            if spline_mode == 'joined':
                spline = JoinedSplineModel(config)
                
                #REMOVE THESE LINES LATER
                # model = Input_layer_DFTB(model_spec)
                # return (model, 'opt')
                
            elif spline_mode == 'non-joined':
                spline = SplineModel(config)
                
                #REMOVE THESE LINES LATER
                # model = Input_layer_DFTB(model_spec)
                # return (model, 'opt')
                
            elif spline_mode == 'debugging':
                #Use this model for debugging purposes, just going to return the values computed in mod_raw
                #return early to prevent any unnecessary work
                model = Input_layer_DFTB(model_spec)
                print("Using a debugging spline model")
                return (model, 'opt')
            if model_spec.oper == 'S': #Inflection points are only instantiated for the overlap operator
                inflect_point_target = minimum_value + ((maximum_value - minimum_value) / 10) #Approximate guess for initial inflection point var
                inflect_point_var = solve_for_inflect_var(minimum_value, maximum_value, inflect_point_target)
                
                #Inflection point should be in the region of variation, i.e. less than cutoff
                try:
                    assert(inflect_point_target < model_cutoff)
                    print(f"Inflection point less than cutoff for {model_spec}")
                    print(f"Inflection point: {inflect_point_target}, cutoff: {model_cutoff}")
                except:
                    print(f"Warning: inflection point {inflect_point_target} not less than cutoff {model_cutoff} for {model_spec}")
                    
                if spline_mode == 'joined':
                    model = Input_layer_pairwise_linear_joined(model_spec, spline, par_dict, config['cutoff'], device, dtype, inflection_point_var = [inflect_point_var] if include_inflect else [])
                        
                elif spline_mode == 'non-joined':
                    model = Input_layer_pairwise_linear(model_spec, spline, par_dict, config['cutoff'], device, dtype, inflection_point_var = [inflect_point_var] if include_inflect else [])
            else:
                if spline_mode == 'joined':
                    model = Input_layer_pairwise_linear_joined(model_spec, spline, par_dict, config['cutoff'], device, dtype)
                elif spline_mode == 'non-joined':
                    model = Input_layer_pairwise_linear(model_spec, spline, par_dict, config['cutoff'], device, dtype)
            #To check for non-zero splines, do this trick to pull the coefficients and check if they are zero. Have to 
            #   transport to CPU first, but since this is done once at the beginning and the tensors are small, this is acceptable
            #   overhead.
            #cpu() after detach() to prevent superfluous copying, https://stackoverflow.com/questions/49768306/pytorch-tensor-to-numpy-array
            ### UNCOMMENT THESE THREE LINES LATER!
            variables = model.get_variables().detach().cpu().numpy() 
            if apx_equal(np.sum(variables), 0):
                return (model, 'noopt')
            return (model, 'opt')
        else:
            # Case of using OffDiagModel
            if spline_mode != 'debugging':
                model = Input_layer_hubbard(model_spec, model_variables, device, dtype)
                
                #REMOVE THIS LINE LATER
                #model = Input_layer_DFTB(model_spec)
            else:
                print("Using a debugging model for off-diagonal elements")
                model = Input_layer_DFTB(model_spec)
            return (model, 'opt')
        
class DFTB_Layer:
    
    def __init__(self, device: torch.device, dtype: torch.dtype, eig_method: str = 'new',
                 repulsive_method: str = 'old') -> None:
        r"""Initializes the DFTB deep learning layer for the network
        
        Arguments:
            device (torch.device): The device to run the computations on (CPU vs GPU).
                If running on GPU, must be CUDA enabled GPU.
            dtype (torch.dtype): The torch datatype for the calculations
            eig_method (str): The eigenvalue method for the symmetric eigenvalue decompositions.
                'old' means the original PyTorch symeig method, and 'new' means the 
                eigenvalue broadening method implemented in eig.py. Defaults to 'new'
            repulsive_method (str): The repulsive method to use. If the repulsive method is 
                'old', then the pairwise splines initiated are still used. If the repulsive method 
                if 'new', then the repulsive energy is not computed from the DFTB Layer 
                but rather from the DFTBrepulsive model externally.
        
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
        self.device = device
        self.dtype = dtype
        self.method = eig_method
        self.repulsive_method = repulsive_method
    
    def forward(self, data_input: Dict, all_models: Dict, mode: str = 'train') -> Dict:
        r"""Forward pass through the DFTB layer to generate molecular properties
        
        Arguments: 
            data_input (Dict): The feed dictionary for the current batch being pushed
                through the network
            all_models (Dict): The dictionary containing references to all the spline 
                model objects being used to predict operator elements
            mode (str): The mode being used with the layer. One of 'train' and 
                'eval', where 'train' means the model is being trained and 'eval'
                means the model is being evaluated.
        
        Returns:
            calc (Dict): A dictionary contianing the molecular properties predicted from the 
                DFTB layer using the predicted operator element values from the spline models.
                The calc dict contains important values like 'dQ' used for charges and dipoles, and 
                'Erep', 'Eelec', and 'Eref', which are used to compute the total energy.
        
        Notes: The DFTB layer operations are separated into 5 stages: forming the initial input layer,
            performing Slater-Koster rotations, assembling values into operators, constructing the fock operators, and
            solving the generalized eigenvalue problem for the the fock operator. 
            
            The mode argument is necessary because when training the models, there is a 
            cutoff that is used when invoking the get_values() method of the spline
            models in all_models. However, when evaluating the model, we want 
            the predictions to be generated accurately for the entire distance 
            range, not only for those values of r < cutoff.
            
        """
        model_vals = list()
        #Maybe won't need additional filtering here if going off feed['models']
        if mode == 'train':
            for model_spec in data_input['models']: 
                model_vals.append( all_models[model_spec].get_values(data_input[model_spec]) )
        elif mode == 'eval':
            for model_spec in data_input['models']:
                curr_model = all_models[model_spec]
                if hasattr(curr_model, "pairwise_linear_model"):
                    curr_mod_raw = data_input['mod_raw'][model_spec]
                    distances = np.array([elem.rdist for elem in curr_mod_raw])
                    dgrids_consts = curr_model.pairwise_linear_model.linear_model(distances, 0)
                    A, b = dgrids_consts
                    A = torch.tensor(A, dtype = self.dtype, device = self.device)
                    b = torch.tensor(b, dtype = self.dtype, device = self.device)
                    variables = curr_model.get_variables()
                    predictions = torch.matmul(A, variables) + b
                    model_vals.append(predictions)
                elif (model_spec.oper == 'G') and (len(model_spec.Zs) == 1) and (len(model_spec.orb) == 2)\
                    and (model_spec.orb[0] != model_spec.orb[1]):
                    #Specifically singling out the cases such as Model(oper = 'G', Zs = (7,), orb = 'ps')
                    #This is only for debugging. Also need to isolate the 
                    #Model("G", (1, ), 'ss') because there is no Model("G", (1, ), 's')
                    curr_mod_raw = data_input['mod_raw'][model_spec]
                    hub1_mod = Model("G", model_spec.Zs, model_spec.orb[0] * 2)
                    hub2_mod = Model("G", model_spec.Zs, model_spec.orb[1] * 2)
                    hub1 = all_models[hub1_mod].get_variables()[0].detach().item()
                    hub2 = all_models[hub2_mod].get_variables()[0].detach().item()
                    result = np.repeat(_Gamma12(0.0, hub1, hub2), len(curr_mod_raw))
                    result = torch.tensor(result, dtype = self.dtype, device = self.device)
                    model_vals.append(result)
                else:
                    model_vals.append(curr_model.get_values(data_input[model_spec]))
        else:
            raise ValueError("Unrecognized mode for method forward()")
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
            if self.repulsive_method == 'old':
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
                    symeig = SymEigB.apply #Can this be done on GPU?
                    Svals, Svecs = symeig(S1, 'cond')
                elif self.method == 'old':
                    Svals, Svecs = torch.symeig(S1, eigenvectors = True) #How does this work on GPU?
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
            zcount_vec = data_input['zcounts'][bsize]
            ones_vec = torch.tensor([1.0], device = self.device, dtype = self.dtype)
            ones_vec = ones_vec.repeat(zcount_vec.shape[0]).unsqueeze(1)
            zcount_vec = torch.cat((zcount_vec, ones_vec), dim = 1)
            ref_res = torch.matmul(zcount_vec, ref_energy_variables.unsqueeze(1))
            calc['Eref'][bsize] = ref_res.squeeze(1)
        return calc

def assemble_ops_for_charges(feed: Dict, all_models: Dict, device: torch.device, dtype: torch.dtype) -> Dict:
    r"""Generates the H and G operators for the charge update operation
    
    Arguments:
        feed (Dict): The current feed whose charges need to be udpated
        all_models (Dict): A dictionary referencing all the spline models being used
            to predict operator elements
        device (torch.device): The device to run the computations on (CPU vs GPU).
                If running on GPU, must be CUDA enabled GPU.
        dtype (torch.dtype): The torch datatype for the calculations
        
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
    rot_out = torch.tensor([0.0, 1.0], dtype = dtype, device = device)
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

def update_charges(feed: Dict, op_dict: Dict, dftblst: DFTBList, device: torch.device, dtype: torch.dtype,
                   modeled_opers: List[str] = ["S", "G"]) -> None:
    r"""Destructively updates the charges in the feed
    
    Arguments:
        feed (Dict): The feed dictionary whose charges need to be updated
        op_dict (Dict): The dictionary containing the H and G operators, separated by bsize
        dftblst (DFTBList): The DFTBList instance for this feed
        device (torch.device): The device to run the computations on (CPU vs GPU).
                If running on GPU, must be CUDA enabled GPU.
        dtype (torch.dtype): The torch datatype for the calculations
        additional_opers (List[str]): The list of additional operators being modeled that
            need to be passedto get_dQ_from_H. Only "S" and "G" are relevant; "R" is not used in
            charge updates and "H" is mandatory for charge updates
    
    Returns:
        None
    
    Notes: Both dQ and the occ_rho_mask are udpated for the feed
    """
    for bsize in op_dict['H'].keys():
        np_Hs = op_dict['H'][bsize].detach().cpu().numpy() #Have to shift things back over to CPU, but not sure about this overhead...
        np_Gs, np_Ss = None, None
        if "G" in modeled_opers:
            np_Gs = op_dict['G'][bsize].detach().cpu().numpy()
        if "S" in modeled_opers:
            np_Ss = op_dict['S'][bsize].detach().cpu().numpy()
        for i in range(len(dftblst.dftbs_by_bsize[bsize])):
            curr_dftb = dftblst.dftbs_by_bsize[bsize][i]
            curr_H = np_Hs[i]
            curr_G = np_Gs[i] if np_Gs is not None else None
            curr_S = np_Ss[i] if np_Ss is not None else None
            # if (curr_G is None):
            #     print("G is not included in charge update")
            # if (curr_S is None):
            #     print("S is not included in charge update")
            newQ, occ_rho_mask_upd, _ = curr_dftb.get_dQ_from_H(curr_H, newG = curr_G, newS = curr_S) #Modelling both S and G
            newQ, occ_rho_mask_upd = torch.tensor(newQ, dtype = dtype, device = device).unsqueeze(1), torch.tensor(occ_rho_mask_upd, dtype = dtype, device = device)
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
                       ragged_dipole: bool = True, run_check: bool = True) -> Tuple[List[Dict], List[Dict], List[DFTBList], List[DFTBList]]:
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
                    train_ener_per_heavy: bool = True) -> Tuple[List[Dict], List[Dict]]:
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
        cleaned_dataset.append(item)
            
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

def generate_cv_folds(dataset: List[Dict], num_folds: int, shuffle: bool = True) -> List[Array]:
    r"""Generates num_folds many folds from the dataset, performs a shuffle first
    
    Arguments:
        dataset (List[Dict]): A list of dictionaries returned by get_ani1data in dftb_layer_splines_4.py
        num_folds (int): The number of folds to generate
        shuffle (bool): If bool, will shuffle the dataset before splitting into folds
    
    Returns:
        folds (List[Array]): A list of arrays of indices representing the folds created from the 
            initial dataset
    
    Notes: This approach for generating folds works for non-transfer training, 
        this is also a very simple cv scheme
    """
    if shuffle:
        random.shuffle(dataset)
    indices = np.arange(len(dataset))
    folds = np.array_split(indices, num_folds)
    return folds

def graph_generation(molecules: List[Dict], config: Dict, allowed_Zs: List[int], 
                     par_dict: Dict, num_per_batch: int = 10) -> Tuple[List[Dict], List[DFTBList], List[List[Dict]]]:
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
        try:
            feed, batch_dftb_lst = create_graph_feed(config, batch, allowed_Zs, par_dict)
        except:
            continue
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

def model_loss_initialization(training_feeds: List[Dict], validation_feeds: List[Dict], allowed_Zs: List[int], losses: Dict, 
                              device: torch.device, dtype: torch.dtype, ref_ener_start: List = None) -> tuple:
    r"""Initializes the losses and generates the models and model_variables dictionaries
    
    Arguments:
        training_feeds (List[Dict]): The training feed dictionaries
        validation_feeds (List[Dict]): The validation feed dictionaries
        allowed_Zs (List[int]): The atomic numbers of allowed elements
        losses (Dict): Dictionary of the targets and their target accuracies
        device (torch.device): The device to run the computations on (CPU vs GPU).
                If running on GPU, must be CUDA enabled GPU.
        dtype (torch.dtype): The torch datatype for the calculations
        ref_ener_start (List): A list of starting value for the reference energy model.
            Defaults to None
    
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
        
        The default previous values are best suited for fitting the couple-clustered energy
    """
    all_models = dict()
    model_variables = dict() #This is used for the optimizer later on
    
    all_models['Eref'] = Reference_energy(allowed_Zs, device, dtype) if (ref_ener_start is None) else \
        Reference_energy(allowed_Zs, device, dtype, prev_values = ref_ener_start)
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
            #Additional step of clearing the FormPenaltyLoss
            all_losses[loss].clear_class_dicts()
        elif loss == "dipole":
            all_losses['dipole'] = DipoleLoss() #Use DipoleLoss2 for dipoles computed from ESP charges!
            loss_tracker['dipole'] = [list(), list(), 0]
        elif loss == "charges":
            all_losses['charges'] = ChargeLoss()
            loss_tracker['charges'] = [list(), list(), 0]
    
    return all_models, model_variables, loss_tracker, all_losses, model_range_dict

def model_sort(model_specs: List[Model]) -> List[Model]:
    r"""Sorts models so that earlier models that later models depend on are
        passed through first.
    
    Arguments:
        model_specs (list[Model]): The list of models that needs to be sorted.
    
    Returns:
        sorted_mods (List[Model]): The sorted list of models.
    
    Notes: Ensuring no dependency issues means that models that depend on 
        earlier models are sorted to later in the list of models 
        passed through get_model_value_spline() and the models that are
        independent are sorted to the front. This is only necessary for the 
        "G" models of the form Model(G, (6,), 'ps') which depends on the earlier 
        models Model(G, (6, ), 'ss') and Model(G, (6,), 'pp'). Since the order 
        of the other models does not matter, we are only interested in sorting G
        single element models.
        The order thus goes:
            1) G, single element, same orbital (e.g. Model(G, (6,), 'ss'))
            2) G, single element, different orbital (e.g. Model(G, (6,), 'ps'))
            3) G, double element (e.g. Model(G, (6, 7), 'ss'))
    """
    single_g_mods, all_other_mods = [], []
    for mod in model_specs:
        if (mod.oper == "G") and (len(mod.Zs) == 1):
            single_g_mods.append(mod)
        else:
            all_other_mods.append(mod)
    #The key here prioritizes the same orbital over different orbitals, 
    #   but without differentiating between different 
    sorted_g_single = sorted(single_g_mods, key = lambda x : 0 if x.orb[0] == x.orb[1] else 1)
    sorted_mods = sorted_g_single + all_other_mods
    return sorted_mods
    
def feed_generation(feeds: List[Dict], feed_batches: List[List[Dict]], all_losses: Dict, 
                    all_models: Dict, model_variables: Dict, model_range_dict: Dict,
                    par_dict: Dict, spline_mode: str, spline_deg: int,
                    device: torch.device, dtype: torch.dtype,
                    debug: bool = False, loaded_data: bool = False,
                    spline_knots: int = 50, buffer: float = 0.0, 
                    joined_cutoff: float = 3.0, cutoff_dict: Dict = None,
                    off_diag_opers: List[str] = ["G"],
                    include_inflect: bool = True) -> None:
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
        spline_mode (str): The type of spline to use, one of 'joined' or 'non-joined'
        spline_deg (int): The degree of the spline to use.
        device (torch.device): The device to run the computations on (CPU vs GPU).
                If running on GPU, must be CUDA enabled GPU.
        dtype (torch.dtype): The torch datatype for the calculations
        debug (bool): Whether or not to use debug mode. Defaults to False
        loaded_data (bool): Whether or not using pre-loaded data. Defaults to False
        spline_knots (int): How many knots for the basis splines. Defaults to 50
        buffer (float): The value to shift the starting and ending distance by. The starting distance
            is rlow - buffer, and the ending distance is rhigh + buffer, where rlow, rhigh are the 
            minimum and maximum distances for the spline from the spline_dict. 
            Defaults to 0.0 angstroms (no shift)
        joined_cutoff (float): The default cutoff for the joined splines. Defaults to 3.0 
            angstroms
        cutoff_dict (Dict): Optional dictionary mapping (oper, element pairs) tuples to cutoffs for specific
            combinations. Defaults to None, in which case the joined_cutoff is used for all models
        off_diag_opers (List[str]): The name of operators that are modeled using an off-diagonal 
            model rather than a spline functional form. Only used for "G" oper so 
            defaults to ["G"]
        include_inflect (bool): Whether or not to include the inflection point variable
            for each model. Defaults to True
    
    Returns:
        None
    
    Notes: The spline_deg parameter only matters if the spline_mode is 'non-joined'
        
        The cutoff_dict entries should be of the following format:
            (oper, (elem1, elem2)) : cutoff
        
        This way, all the models using the given oper and elems will have the same cutoff
    """
    for ibatch,feed in enumerate(feeds):
        #Need to do an intermediate step of sorting the models
        sorted_mods = model_sort(feed['models'])
        for model_spec in sorted_mods:
            # print(model_spec)
            if (model_spec not in all_models):
                print(model_spec)
                mod_res, tag = get_model_value_spline(model_spec, model_variables, model_range_dict, par_dict,
                                                        device, dtype, 
                                                        spline_knots, buffer,
                                                        joined_cutoff, cutoff_dict,
                                                        spline_mode, spline_deg,
                                                        off_diag_opers, include_inflect)
                all_models[model_spec] = mod_res
                #all_models[model_spec] = get_model_dftb(model_spec)
                if tag != 'noopt' and not isinstance(mod_res, Input_layer_hubbard):
                    # Do not add redundant variables for the OffDiagModel. Nothing
                    # is done for off-diagonal model variables
                    model_variables[model_spec] = all_models[model_spec].get_variables()
                    if (hasattr(mod_res, "inflection_point_var")) and (mod_res.inflection_point_var is not None):
                        old_oper, old_zs, old_orb = model_spec
                        new_mod = Model(old_oper, old_zs, old_orb + '_inflect')
                        model_variables[new_mod] = all_models[model_spec].get_inflection_pt()
                # Detach it from the computational graph (unnecessary)
                elif tag == 'noopt':
                    all_models[model_spec].variables.requires_grad = False
            model = all_models[model_spec]
            # print(model_spec)
            # print(model_spec == Model(oper='R', Zs=(8, 8), orb='ss'))
            feed[model_spec] = model.get_feed(feed['mod_raw'][model_spec])
        
        for loss in all_losses:
            try:
                all_losses[loss].get_feed(feed, [] if loaded_data else feed_batches[ibatch], all_models, par_dict, debug)
            except Exception as e:
                print("Something went wrong in feed_generation!")
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

def total_type_conversion(training_feeds: List[Dict], validation_feeds: List[Dict], ignore_keys: List[str],
                          device: torch.device, dtype: torch.dtype) -> None:
    r"""Does a recursive type conversion for each feed in training_feeds and each feed in validation_feeds
    
    Arguments:
        training_feeds (List[Dict]): The training feed dictionaries to do the correction for
        validation_feeds (List[Dict]): The validation feed dictionaries to do the correction for
        ignore_keys (List[str]): The keys to ignore the type conversion for
        device (torch.device): The device to run the computations on (CPU vs GPU).
                If running on GPU, must be CUDA enabled GPU.
        dtype (torch.dtype): The torch datatype for the calculations
    Returns:
        None
    
    Notes: None
    """
    for feed in training_feeds:
        recursive_type_conversion(feed, ignore_keys, device, dtype)
    for feed in validation_feeds:
        recursive_type_conversion(feed, ignore_keys, device, dtype)






# Old version of function
# def model_range_correction(model_range_dict: Dict, correction_dict: Dict, universal_high: float = None) -> Dict:
#     r"""Corrects the lower bound values of the spline ranges in model_range_dict using correction_dict
    
#     Arguments:
#         model_range_dict (Dict): Dictionary mapping model_specs to their ranges in angstroms as tuples of the form
#             (rlow, rhigh)
#         correction_dict (Dict): Dictionary mapping model_specs to their new low ends in angstroms. Here,
#             The keys are element tuples (elem1, elem2) and the values are the new low ends.
#             All models modelling the interactions between elements (a, b) will have the
#             same low end
#         universal_high (float): The maximum distance bound for all spline models.
#             Defaults to None
    
#     Returns:
#         new_dict (Dict): The corrected model range dictionary
    
#     Notes: If a models' element pair does not appear in the correction_dict, its range
#         is left unchanged, i.e. it is determined by the Modraw values of the 
#         used data.
#     """
#     new_dict = dict()
#     for mod, dist_range in model_range_dict.items():
#         xlow, xhigh = dist_range
#         Zs, Zs_rev = mod.Zs, (mod.Zs[1], mod.Zs[0])
#         if Zs in correction_dict:
#             #Safeguard to ensure that low-end distances are a valid correction
#             assert(xlow > correction_dict[Zs])
#             xlow = correction_dict[Zs]
#         elif Zs_rev in correction_dict:
#             assert(xlow > correction_dict[Zs_rev])
#             xlow = correction_dict[Zs_rev]
#         new_dict[mod] = (xlow, xhigh if universal_high is None else universal_high)
#     return new_dict

def model_range_correction(model_range_dict: Dict, low_correction_dict: Dict, cutoff_dictionary: Dict, joined_cutoff: float) -> Dict:
    r"""Corrects the lower bound values of the spline ranges in model_range_dict using correction_dict
    
    Arguments:
        model_range_dict (Dict): Dictionary mapping model_specs to their ranges in angstroms as tuples of the form
            (rlow, rhigh)
        low_correction_dict (Dict): Dictionary mapping model_specs to their new low ends in angstroms. Here,
            The keys are element tuples (elem1, elem2) and the values are the new low ends.
            All models modelling the interactions between elements (a, b) will have the
            same low end
        cutoff_dictionary (Dict): Dictionary mapping model_specs to their cutoff distances if specified. 
            For values beyond the cutoff, the value returned by the model returns 0.
        joined_cutoff (float): The maximum cutoff bound for all models and is used if the model is 
            not found in cutoff dictionary.
    
    Returns:
        new_dict (Dict): The corrected model range dictionary
    
    Notes: If a models' element pair does not appear in the correction_dict, its range
        is left unchanged, i.e. it is determined by the Modraw values of the 
        used data.
        
        All spline models are going to span the distance from 0 -> rcut, where 
        the value predicted for r > rcut is set to 0.
    """
    new_dict = dict()
    for mod, dist_range in model_range_dict.items():
        xlow, xhigh = dist_range
        Zs, Zs_rev = mod.Zs, (mod.Zs[-1], mod.Zs[0])
        #Query the low end correction dictionary with the element pairs
        if Zs in low_correction_dict:
            #Safeguard to ensure that low-end distances are a valid correction
            print(Zs, xlow)
            assert(xlow > low_correction_dict[Zs])
            xlow = low_correction_dict[Zs]
        elif Zs_rev in low_correction_dict:
            assert(xlow > low_correction_dict[Zs_rev])
            xlow = low_correction_dict[Zs_rev]
        
        if cutoff_dictionary is not None:
            #Query the cutoff dictionary with the correct tuple format
            q_forward, q_rev = (mod.oper, Zs), (mod.oper, Zs_rev)
            if q_forward in cutoff_dictionary:
                xhigh = cutoff_dictionary[q_forward]
            elif q_rev in cutoff_dictionary:
                xhigh = cutoff_dictionary[q_rev]
            else: #The case where the model's cutoff is not specified so the joined cutoff is used
                xhigh = joined_cutoff
            new_dict[mod] = (xlow, xhigh)
        else:
            new_dict[mod] = (xlow, joined_cutoff)
            
    return new_dict









def energy_correction(molec: Dict) -> None:
    r"""Performs in-place total energy correction for the given molecule by dividing Etot/nheavy
    
    Arguments:
        molec (Dict): The dictionary in need of correction
    
    Returns:
        None
    """
    zcount = collections.Counter(molec['atomic_numbers'])
    ztypes = list(zcount.keys())
    heavy_counts = [zcount[x] for x in ztypes if x > 1]
    num_heavy = sum(heavy_counts)
    molec['targets']['Etot'] = molec['targets']['Etot'] / num_heavy



if __name__ == "__main__":
    pass

