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
    1) Workaround for the R operator, since the H-H repulsion at longer range is 0 (basically, handle cases where splines are flat)
        Don't optimize models with 0 concavity and a value of 0! (X)
    1) Need more work on the H-H repulsion model for the spline
    2) General optimizations for the code throughout, work based on profiler
        To actually use the initial_results, run the file with a small molec batch and compare the results. Find the largest differences,
        focus on those not in the pre-compute stages of the model!
        
        Keep in mind, initial results were computed for 1,6,7,8 and heavy 1-8, 1193 total molecules, 
        max_config num = 2
        300 epochs
        
    3) Revise h5 file to store dipole information (store dipole_mat per molec)
"""
import pdb, traceback, sys, code

import math
import numpy as np
import os
import random
import pickle
from datetime import datetime
import h5py
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

from collections import OrderedDict, Counter
import collections
import torch
torch.set_printoptions(precision = 10)
from copy import deepcopy
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import time
from tfspline import Bcond

from geometry import Geometry, random_triatomics, to_cart
from auorg_1_1 import ParDict
from dftb import DFTB
from eig import SymEigB
from batch import create_batch, create_dataset, DFTBList
#from modelval import Val_model
# from modelspline import Spline_model
from modelspline import get_dftb_vals
from SplineModel_v3 import SplineModel, fit_linear_model
import pickle

from dftb_layer_splines_ani1ccx import get_targets_from_h5file
from h5handler import model_variable_h5handler, per_molec_h5handler, per_batch_h5handler,\
    total_feed_combinator, compare_feeds
from loss_models import TotalEnergyLoss, FormPenaltyLoss, DipoleLoss, ChargeLoss

#Fix the ani1_path for now
ani1_path = 'data/ANI-1ccx_clean_fullentry.h5'

def apx_equal(x, y, tol = 1e-12):
    return abs(x - y) < tol

def get_ani1data(allowed_Z, heavy_atoms, max_config, target, exclude = None):
    '''
    Method for extracting data from the ani1 data h5 files. 
    
    allowed_Zs: List of allowed atomic numbers
    heavy_atoms: List of allowed number of heavy atoms
    max_config: Maximum number of configurations allowed per molecule
    target: Dictionary mapping each type of target to the corresponding ANI-1 key
    exclude: Nonetype or list of molecules to ignore
    '''
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

# Fields that come from SCF:
#    ['F','Eelec','rho','eorb','occ_rho_mask', 'entropy', 'fermi_energy']
#    ['dQ','dQinit','dQcurr']
def create_graph_feed(config, batch, allowed_Zs):
    '''
    Takes in a list of geometries and creates the graph and feed dictionaries for that batch
    '''
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
    if 'G' not in config['opers_to_model']:
        fields_by_type['feed_constant'].extend(['G'])
    
    fields_by_type['feed_SCF'] = \
        ['dQ','Eelec','eorb']
        
    needed_fields = fields_by_type['feed_constant'] + \
       fields_by_type['feed_SCF'] + fields_by_type['graph']
    
    geom_batch = []
    for molecule in batch:
        geom = Geometry(molecule['atomic_numbers'], molecule['coordinates'].T)
        geom_batch.append(geom)
        
    batch = create_batch(geom_batch,FIXED_ZS=allowed_Zs)
    dftblist = DFTBList(batch)
    feed = create_dataset(batch,dftblist,needed_fields)

    return feed, dftblist

def create_graph_feed_loaded (config, batch, allowed_Zs):
    '''
    Method to generate the feeds for the dftblayer from data loaded in from
    h5 file. This requires running on a subset of the fields used in the 
    original method
    
    Ideally, since this side-steps the SCF cycle, this should require less compute time
    
    Defunct method, not necessary anymore.
    '''
    fields_by_type = dict()
    fields_by_type['graph'] = \
      ['models','onames','basis_sizes','glabels',
       'gather_for_rot', 'gather_for_oper',
       'gather_for_rep','segsum_for_rep','atom_ids','norbs_atom']
    
    fields_by_type['feed_constant'] = \
        ['geoms','mod_raw', 'rot_tensors']
        
    needed_fields = fields_by_type['feed_constant'] + fields_by_type['graph']
    
    geom_batch = []
    for molecule in batch:
        geom = Geometry(molecule['atomic_numbers'], molecule['coordinates'].T)
        geom_batch.append(geom)
    

    batch = create_batch(geom_batch, FIXED_ZS = allowed_Zs)
    dftblist = DFTBList(batch)
    feed = create_dataset(batch, dftblist, needed_fields)

    return feed, dftblist

class Input_layer_DFTB:
    '''
    Object oriented interface replacement for original value model input
    
    Ths value model should return a single torch tensor for its variables too...
    '''
    def __init__(self,model):
        self.model = model
    def get_variables(self):
        return []
    def get_feed(self, mod_raw):
        return {'values' : np.array([x.dftb for x in mod_raw])}
    def get_values(self, feed):
       return feed['values']

class Input_layer_value:
    '''
    Object oriented interface replacement for original value model input
    
    Ths value model should return a single torch tensor for its variables too...
    '''
    def __init__(self,model, initial_value=0.0):
        self.model = model
        if not isinstance(initial_value, float):
            raise ValueError('Val_model not initialized to float')
        self.value = np.array([initial_value])
        self.variables = torch.from_numpy(self.value)
        self.variables.requires_grad = True
    def initialize_to_dftb(self,pardict, noise_magnitude = 0.0):
        init_value = get_dftb_vals(self.model, pardict)
        if not noise_magnitude == 0.0:
            init_value = init_value + noise_magnitude * np.random.randn(1)
        self.value[0]= init_value
    def get_variables(self):
        return self.variables
    def get_feed(self, mod_raw):
        return {'nval': len(mod_raw)}
    def get_values(self, feed):
       result = self.variables.repeat(feed['nval'])
       return result
   
class Input_layer_pairwise_linear:
    '''
    Object oriented interface replacement for original spline model
    
    The spline model already returns a torch tensor for its variables
    '''
    def __init__(self, model, pairwise_linear_model, par_dict, ngrid = 100, 
                 noise_magnitude = 0.0):
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
    def get_variables(self):
        return self.variables
    def get_feed(self, mod_raw):
        ''' this returns numpy arrays '''
        xeval = np.array([elem.rdist for elem in mod_raw])
        A,b = self.pairwise_linear_model.linear_model(xeval)
        return {'A': A, 'b': b}
    def get_values(self, feed):
        '''
        No need to convert from numpy to torch, the recursive type correction
        should catch everything
        '''
        A = feed['A']
        b = feed['b']
        result = torch.matmul(A, self.variables) + b
        return result
    
class Reference_energy:
    '''
    Object oriented interface replacement for original value model input
    
    Ths value model should return a single torch tensor for its variables too...
    '''
    #TODO: add const to this
    def __init__(self,allowed_Zs, prev_values = None):
        self.allowed_Zs = np.sort(np.array(allowed_Zs))
        self.values = np.zeros(self.allowed_Zs.shape)
        if prev_values is not None:
            #Load previous values if they are given
            #FOR DEBUGGING PURPOSES ONLY
            self.values[0] = prev_values[0]
            self.values[1] = prev_values[1]
        self.variables = torch.from_numpy(self.values)
        self.variables.requires_grad = True
    def get_variables(self):
        return self.variables
   
    
def get_model_dftb(model_spec): 
    return Input_layer_DFTB(model_spec)


class data_loader:
    '''
    This is a new data loader class specifically designed for our use
    '''
    def __init__(self, dataset, batch_size, shuffle = True, shuffle_scheme = 'random'):
        self.data = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.shuffle_method = shuffle_scheme
        self.batch_creation()
        self.batch_index = 0
    
    def create_batches(self, data):
        '''
        Implements sequential batching of the data
        '''
        batches = list()
        for i in range(0, len(data), self.batch_size):
            current_batch = data[i : i + self.batch_size]
            batches.append(current_batch)
        return batches
    
    def batch_creation(self):
        '''
        Creates all the batches so that they are accessible in the class
        '''
        self.batches = self.create_batches(self.data)
        self.batch_index = 0
        
    def shuffle_batches(self):
        '''
        Takes the train and valid data, shuffles them, and then re-batches them
        
        Currently only takes in the random shuffling scheme
        '''
        random.shuffle(self.batches)
        self.batch_index = 0
        
    def shuffle_total_data(self):
        '''
        This reshuffles all the original data to create new train and valid batch possibilities
        
        This will need to be manually called
        '''
        random.shuffle(self.data)
        self.batch_creation()
    
    def __iter__(self):
        '''
        Treat this as our iterator
        '''
        return self
    
    def __next__(self):
        if self.batch_index < len(self.batches):
            return_batch = self.batches[self.batch_index]
            self.batch_index += 1
            return return_batch
        else:
            # Automatically shuffle the batches after a full iteration
            self.shuffle_batches()
            raise StopIteration

'''
The minimum and maximum should be determined dynamically each training session
'''
def get_model_value_spline(model_spec, max_val = 7.1, num_knots = 50, buffer = 0.1):
    par_dict = ParDict()
    noise_magnitude = 0.0
    if len(model_spec.Zs) == 1:
        model = Input_layer_value(model_spec)
        model.initialize_to_dftb(par_dict, noise_magnitude)            
    elif len(model_spec.Zs) == 2:
        cutoffs = OrderedDict(
            [( (1, 1) , (0.63, 5.00) ),
             ( (1, 6) , (0.60, 5.00) ),
             ( (1, 7) , (0.59, 5.00) ),
             ( (1, 8) , (0.66, 5.00) ),
             ( (6, 6) , (1.04, 5.00) ),
             ( (6, 7) , (0.95, 5.00) ),
             ( (6, 8) , (1.00, 5.00) ),
             ( (7, 7) , (0.99, 5.00) ),
             ( (7, 8) , (0.93, 5.00) ),
             ( (8, 8) , (1.06, 5.00) )] )
        flipped = dict()
        for z,rs in cutoffs.items():
            if z[0] != z[1]:
                flipped[(z[1],z[0])] = rs
        cutoffs.update(flipped)
        num_knots = num_knots
        # minimum_val = cutoffs[model_spec.Zs][0]
        # maximum_val = cutoffs[model_spec.Zs][1]
        minimum_val = 0.0
        maximum_val = max_val + buffer
        xknots = np.linspace(minimum_val, maximum_val, num = num_knots)
        # xknots = np.concatenate([xknots,[6.8]],0)
        config = {'xknots' : xknots,
                  'deg'    : 3,
                  'bconds' : 'vanishing'}  
        spline = SplineModel(config)
        model = Input_layer_pairwise_linear(model_spec, spline, par_dict)
 
    return model

def get_model_value_spline_2(model_spec, spline_dict, par_dict, num_knots = 50, num_grid = 200, buffer = 0.0):
    '''
    A hypothetical approach to spline configuration for off-diagonal models, no change for on-diagonal
    elements
    
    Will also need to check that the spline is not already flat at 0 (e.g. long range H-H repulsion). If the
    spline is already 0, don't change it!
    '''
    noise_magnitude = 0.0
    if len(model_spec.Zs) == 1:
        model = Input_layer_value(model_spec)
        model.initialize_to_dftb(par_dict, noise_magnitude)
        return (model, 'vol')
    elif len(model_spec.Zs) == 2:
        minimum_value, maximum_value = spline_dict[model_spec]
        minimum_value -= buffer
        maximum_value += buffer
        xknots = np.linspace(minimum_value, maximum_value, num = num_knots)
        config = {'xknots' : xknots,
                  'deg'    : 3,
                  'bconds' : 'natural'}  #CHANGED THE BOUNDARY CONDITION FROM VANISHING TO NATURAL
        spline = SplineModel(config)
        model = Input_layer_pairwise_linear(model_spec, spline, par_dict)
        variables = model.get_variables().detach().numpy()
        if apx_equal(np.sum(variables), 0):
            return (model, 'noopt')
        return (model, 'opt')
        # Check that the model is not already flat at 0
        # More thorough check but probably unnecessary
        # rlow, rhigh = model.pairwise_linear_model.r_range()
        # rgrid = np.linspace(rlow, rhigh, num_grid)
        # dgrids_consts = [model.pairwise_linear_model.linear_model(rgrid, 0),
        #              model.pairwise_linear_model.linear_model(rgrid, 2)]
        # y_vals_0 = np.dot(dgrids_consts[0][0], variables) + dgrids_consts[0][1]
        # y_vals_2 = np.dot(dgrids_consts[1][0], variables) + dgrids_consts[1][1]
        # sum_res = np.sum(y_vals_0) #values
        # sum_res2 = np.sum(y_vals_2) #2nd deriv
        # if apx_equal(sum_res, 0) and apx_equal(sum_res2, 0):
        #     return (model, 'noopt')
        # else:
        #     return (model, 'opt')
        

def form_initial_layer(all_models, feeds, device = None, dtype = torch.double):
    '''
    Form the initial layer (the net_vals) for dftb layer from the data contained in the 
    input dictionary, no more graph and feed separation. Also, now need to get the
    all_models dictionary going on in here to take advantage of the class method
    '''
    net_vals = list()
    for model_spec in feeds['models']:
        net_vals.append( all_models[model_spec].get_values(feeds[model_spec]) )
    result = torch.cat(net_vals)
    return result

def torch_segment_sum(data, segment_ids, device, dtype): # Can potentially replace with scatter_add, but not part of main ptyorch distro
    '''
     just going to use pytorch tensor instead of np array
    '''
    max_id = torch.max(segment_ids)
    res = torch.zeros([max_id + 1], device = device, dtype = dtype)
    for i, val in enumerate(data):
        res[segment_ids[i]] += val
    return res

class loss_model:
    '''
    General class for handling losses; users will have to supply list of targets to use in
    loss computation and a function for actually computing it.
    
    Targets will be a list of strings.
    '''
    def __init__(self, targets, loss_function):
        self.targets = targets
        self.loss_func = loss_function
 
    def get_targets(self):
        return self.targets
    
    def compute_loss(self, output, feed):
        '''
        Computes the loss using the user-provided loss function by comparing the 
        differences in target fields. The user defined loss function should have
        the following signature:
        
        def loss(output, feed, targets):
            # Logic
            # Returns a pytorch loss object with a backward method for backpropagation
        '''
        return self.loss_func(output, feed, self.targets)

class loss_model_alt:
    
    '''
    Take 2 on class for handling losses
    
    This time, incorporate penalties and weights for overall loss. Penalties and
    weights are optional, but all_models is not
    
    Also required to pass in concavity information. To save computational time, 
    do this as a pre-compute for all the models and toss it in. In fact, the 
    concavity can be determined for all models independently since it only 
    depends on the Model named tuple and the slater-koster files
    '''
    def __init__ (self, targets, loss_function, all_models, concavity, penalties = None, weights = None):
        self.targets = targets
        self.loss_func = loss_function
        self.penalties = penalties
        self.weights = weights
        self.all_models = all_models
        self.concavity = concavity
    
    '''
    This loss model does not need anything added to the feed
    '''
    def get_feed(self):
        return {}
    
    '''
    Method to compute the loss, taking in the output and the original feed to 
    compare the values against
    
    Since everything is aliased, we can be sure that references are properly maintained to all the dictionaries
    we are passing in.
    '''
    def get_loss(self, output, feed):
        return self.loss_func(output, feed, self.targets, self.all_models, self.concavity, self.penalties,
                              self.weights)

class DFTB_Layer(nn.Module):
    def __init__(self, device, dtype, eig_method = 'old'):
        super(DFTB_Layer, self).__init__()
        self.device = device
        self.dtype = dtype
        self.method = eig_method
    
    def forward(self, data_input, all_models):
        '''
        The forward pass now takes a full data dictionary (combination of graph and feed) and
        the all_models dictionary because we need that for form initial layer
        '''
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
            calc['dQ'][bsize] = data_input['dQ'][bsize] #Use data_input['dQ'][bsize] here
            ep = torch.matmul(calc['G'][bsize], calc['dQ'][bsize])
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
                phiS = torch.matmul(Svecs, torch.diag(torch.pow(Svals, -0.5).view(-1)))
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
            dQ = calc['dQ'][bsize] #dQ is just copied but not calculated
            Gamma = calc['G'][bsize]
            ener2 = 0.5 * torch.matmul(torch.transpose(dQ, -2, -1), torch.matmul(Gamma, dQ))
            ener2 = ener2.view(ener2.size()[0])
            calc['Eelec'][bsize] = ener1 + ener2
            ref_energy_variables = all_models['Eref'].get_variables()
            ref_res = torch.matmul(data_input['zcounts'][bsize], ref_energy_variables.unsqueeze(1))
            calc['Eref'][bsize] = ref_res.squeeze(1)
        return calc

def recursive_type_conversion(data, ignore_keys, device = None, dtype = torch.double, grad_requires = False):
    '''
    Transports all the tensors stored in data to a tensor with the correct dtype
    on the correct device
    
    list_keys specifies those keys whose values should be a list rather than an np array
    
    Can also instantiate gradient requirements for the recursive type conversion
    '''
    for key in data:
        if key not in ignore_keys:
            if isinstance(data[key], np.ndarray):
                data[key] = torch.tensor(data[key], dtype = dtype, device = device)            
            elif isinstance(data[key], collections.OrderedDict) or isinstance(data[key], dict):
                recursive_type_conversion(data[key], ignore_keys)

def assemble_ops_for_charges(feed, all_models):
    '''
    This is just to assemble the operators for updating the charges
    '''
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

def update_charges(feed, op_dict, dftblst):
    '''
    Test code right now, only works for current configuration of one molecule per batch
    
    Since there is a correspondence between the feed dictionary and dftblst and the op_dict is generated
    from the feed, the basis sizes used should all match
    '''
    for bsize in op_dict['H'].keys():
        np_Hs = op_dict['H'][bsize].detach().numpy() #Don't care about gradients here
        for i in range(len(dftblst.dftbs_by_bsize[bsize])):
            curr_dftb = dftblst.dftbs_by_bsize[bsize][i]
            curr_H = np_Hs[i]
            newQ, occ_rho_mask_upd, _ = curr_dftb.get_dQ_from_H(curr_H) #Ignore the entropy term for now
            newQ, occ_rho_mask_upd = torch.tensor(newQ).unsqueeze(1), torch.tensor(occ_rho_mask_upd)
            feed['dQ'][bsize][i] = newQ # Change dQ to newQ instead
            feed['occ_rho_mask'][bsize][i] = occ_rho_mask_upd

def update_charges_2(feed, op_dict):
    '''
    Try again, this time without batches, create a standalone dftb object much like how we did in
    dftb_layer_splines_1.py
    
    Test code right now, only works for one molecule per batch
    '''
    geom_labels = list(feed['geoms'].keys())
    geom_labels.sort()
    geom_lst = [feed['geoms'][x] for x in geom_labels]
    for geom in geom_lst:
        dftb = DFTB(ParDict(), to_cart(geom), charge = 0)
        curr_basis_size = dftb.nBasis()
        curr_H = op_dict['H'][curr_basis_size]
        curr_H = curr_H.detach().numpy()[0]
        newQ, new_occ_mask, _ = dftb.get_dQ_from_H(curr_H)
        newQ, new_occ_mask = torch.tensor(newQ).unsqueeze(1), torch.tensor(new_occ_mask)
        feed['dQ'][curr_basis_size][0] = newQ # Change dQ to newQ instead
        feed['occ_rho_mask'][curr_basis_size][0] = new_occ_mask
    pass

def find_config_range(data_dict_lst):
    '''
    Finds the range of distances for configuring the splines so that the models do 
    not behave crazily
    
    Too naive of an approach, best not to use this approach
    '''
    minimum, maximum = 0.0, 0.0
    for data_dict in data_dict_lst:
        mod_raw_lsts = [data for _, data in data_dict['mod_raw'].items()]
        distances = [x.rdist for lst in mod_raw_lsts for x in lst]
        (tempMax, tempMin) = max(distances), min(distances)
        if tempMax > maximum: maximum = tempMax
        if tempMin < minimum: minimum = tempMin
    return minimum, maximum

def create_spline_config_dict(data_dict_lst):
    '''
    Finds the range of distances FOR EACH TYPE OF MODEL. Models that are not spline models
    (i.e. len(Zs) == 1) are ignored. This is only for spline models
    Here, data_dict_lst is a list of feeds
    '''
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


#%% Top level variable declaration
'''
If loading data from h5 files, make sure to note the allowed_Zs and heavy_atoms of the dataset and
set them accordingly!
'''
allowed_Zs = [1,6,7,8]
heavy_atoms = [1,2,3,4,5,6,7,8]
#Still some problems with oxygen, molecules like HNO3 are problematic due to degeneracies
max_config = 50
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
config['opers_to_model'] = ['H', 'R']

#loss weights
losses = dict()
target_accuracy_energy = 6270 #Ha^-1
target_accuracy_dipole = 100 # debye
target_accuracy_charges = 100
target_accuracy_convex = 1000
target_accuracy_monotonic = 1000

losses['Etot'] = target_accuracy_energy
# Note: dipole loss cannot be optimized on its own since dQ is updated separately of the s
# model variables. Thus, dQ and teh dipole mats are technically not part of the computational
# graph since S is not being trained.
losses['dipole'] = target_accuracy_dipole 
losses['charges'] = target_accuracy_charges #Not working on charge loss just yet
losses['convex'] = target_accuracy_convex
losses['monotonic'] = target_accuracy_monotonic

#Initialize the parameter dictionary
par_dict = ParDict()

#Compute or load?
loaded_data = True

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

#%% Degbugging h5 (Extraction and combination)
x = time.time()
training_feeds = total_feed_combinator.create_all_feeds("final_batch_test.h5", "final_molec_test.h5")
validation_feeds = total_feed_combinator.create_all_feeds("final_valid_batch_test.h5", "final_valid_molec_test.h5")
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
    feed, batch_dftb_lst = create_graph_feed(config, batch, allowed_Zs)
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
validation_feeds = list()
validation_molec_batches = list()
for index, batch in enumerate(validation_dat_set):
    feed, batch_dftb_lst = create_graph_feed(config, batch, allowed_Zs)
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
        all_losses['dipole'] = DipoleLoss()
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
           mod_res, tag = get_model_value_spline_2(model_spec, model_range_dict, par_dict)
           all_models[model_spec] = mod_res
           #all_models[model_spec] = get_model_dftb(model_spec)
           if tag != 'noopt':
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
            mod_res, tag = get_model_value_spline_2(model_spec, model_range_dict, par_dict)
            all_models[model_spec] = mod_res
            #all_models[model_spec] = get_model_dftb(model_spec)
            if tag != 'noopt':
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
    
# Also save the dftb_lsts for the training_feeds. Can do this using pickle for now
with open("training_dftblsts.p", "wb") as handle:
    pickle.dump(training_dftblsts, handle)
    
print("molecular and batch information successfully saved, along with reference data")

#%% Recursive type conversion
# Not an elegant solution but these two keys need to be ignored since they
# should not be tensors!
# Charges are ignored because of raggedness coming from bsize organization
ignore_keys = ['glabels', 'basis_sizes', 'charges']
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

nepochs = 150
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
    random.shuffle(validation_feeds)
    
    
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
    if (i % 10 == 0):
        for j in range(len(training_feeds)):
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
    
    #Save timing information for diagnostics
    times_per_epoch.append(time.time() - start)

print(f"Finished with {nepochs} epochs")
times['train'] = time.process_time()

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
    assert(len(validation_loss) == nepochs)
    assert(len(training_loss) == nepochs)
    fig, axs = plt.subplots()
    axs.plot(training_loss, label = 'Training loss')
    axs.plot(validation_loss, label = 'Validation loss')
    axs.set_title(f"{loss} loss")
    axs.set_xlabel("Epoch")
    axs.set_ylabel("Average Epoch Loss (unitless)")
    axs.yaxis.set_minor_locator(AutoMinorLocator())
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



