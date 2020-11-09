# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 23:03:05 2020

@author: Frank
"""
"""
TODO:
    1) Incorporate regularization for all splines as additional weighted loss term (High priority)
        See solver.py and lossspline.py in repulsive branch
    2) Data management via h5 files (High priority)
        1) Test and incorporate correct_loaded_feeds() (IP)
    3) Revise loss functions to include weighting (High priority) 
"""
import math
import numpy as np
import os
import random
import pickle
from datetime import datetime
import h5py

from collections import OrderedDict, Counter
import collections
import torch
torch.set_printoptions(precision = 10)
from copy import deepcopy
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import time

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
from loss_methods import loss_refactored, loss_temp

#Fix the ani1_path for now
ani1_path = 'data/ANI-1ccx_clean.h5'

def get_ani1data(allowed_Z, heavy_atoms, max_config, target, exclude = None):
    all_zs = get_targets_from_h5file('atomic_numbers', ani1_path) 
    all_coords =  get_targets_from_h5file('coordinates', ani1_path) 
    all_targets = get_targets_from_h5file(target, ani1_path)
    if exclude is None:
        exclude = []
    batches = list()
    for name in all_zs.keys():
        if name in exclude:
            continue
        zs = all_zs[name]
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
        nconfig = all_coords[name].shape[0]
        for iconfig in range(min(nconfig, max_config)):
            batch = dict()
            batch['name'] = name
            batch['iconfig'] = iconfig
            batch['atomic_numbers'] = zs
            batch['coordinates'] = all_coords[name][iconfig,:,:]
            batch['targets'] = dict()
            batch['targets']['Etot'] = all_targets[name][iconfig]
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

def get_model_value_spline_2(model_spec, spline_dict, num_knots = 50, buffer = 0.1):
    '''
    A hypothetical approach to spline configuration for off-diagonal models, no change for on-diagonal
    elements
    '''
    par_dict = ParDict()
    noise_magnitude = 0.0
    if len(model_spec.Zs) == 1:
        model = Input_layer_value(model_spec)
        model.initialize_to_dftb(par_dict, noise_magnitude)
    elif len(model_spec.Zs) == 2:
        minimum_value, maximum_value = spline_dict[model_spec]
        minimum_value -= buffer
        maximum_value += buffer
        xknots = np.linspace(minimum_value, maximum_value, num = num_knots)
        config = {'xknots' : xknots,
                  'deg'    : 3,
                  'bconds' : 'vanishing'}  
        spline = SplineModel(config)
        model = Input_layer_pairwise_linear(model_spec, spline, par_dict)
    return model
        

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
        
            rho = data_input['rho'][bsize]
            qbasis = rho * calc['S'][bsize]
            GOP  = torch.sum(qbasis,2,keepdims=True)
            qNeutral = data_input['qneutral'][bsize]
            calc['dQ'][bsize] = qNeutral - GOP
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
            except:
                print('diagonalization failed for batch ', data_input['name'])
            calc['eorb'][bsize] = Eorb
            orb = torch.matmul(phiS, temp2)
            occ_mask = data_input['occ_rho_mask'][bsize]
            orb_filled = torch.mul(occ_mask, orb)
            rho = 2.0 * torch.matmul(orb_filled, torch.transpose(orb_filled, -2, -1))
            calc['rho'][bsize] = rho
            ener1 = torch.sum(torch.mul(rho.view(rho.size()[0], -1), calc['H'][bsize].view(calc['H'][bsize].size()[0], -1)), 1) #I think this is fine since calc['Eelec'] is a 1D array
            dQ = calc['dQ'][bsize]
            Gamma = calc['G'][bsize]
            ener2 = 0.5 * torch.matmul(torch.transpose(dQ, -2, -1), torch.matmul(Gamma, dQ))
            ener2 = ener2.view(ener2.size()[0])
            calc['Eelec'][bsize] = ener1 + ener2
            ref_energy_variables = all_models['Eref'].get_variables()
            ref_res = torch.matmul(data_input['zcounts'][bsize], ref_energy_variables.unsqueeze(1))
            calc['Eref'][bsize] = ref_res.squeeze(1)
        return calc

def recursive_type_conversion(data, device = None, dtype = torch.double, grad_requires = False):
    '''
    Transports all the tensors stored in data to a tensor with the correct dtype
    on the correct device
    
    Can also instantiate gradient requirements for the recursive type conversion
    '''
    for key in data:
        if isinstance(data[key], np.ndarray):
            data[key] = torch.tensor(data[key], dtype = dtype, device = device)            
        elif isinstance(data[key], collections.OrderedDict) or isinstance(data[key], dict):
            recursive_type_conversion(data[key])

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
allowed_Zs = [1,6,7,8]
heavy_atoms = [1,2,3,4,5,6,7,8]
#Still some problems with oxygen, molecules like HNO3 are problematic due to degeneracies
max_config = 3
target = 'dt'
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

#Fit to DFTB or fit to ani1-ccx target
fit_to_DFTB = False

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
targets_for_loss = ['Eelec', 'Eref']

#%% Degbugging h5 stuff
# master_dict = per_molec_h5handler.extract_molec_feeds_h5('testfeeds.h5') #test file used locally 
# molec_lst = per_molec_h5handler.reconstitute_molecs_from_h5(master_dict)

# tstfeed, _ = create_graph_feed_loaded(config, molec_lst, allowed_Zs)

# all_bsizes = list(tstfeed['glabels'].keys())

# tstfeed['names'] = dict()
# tstfeed['iconfigs'] = dict()
# for bsize in all_bsizes:
#     glabels = tstfeed['glabels'][bsize]
#     all_names = [molec_lst[x]['name'] for x in glabels]
#     all_configs = [molec_lst[x]['iconfig'] for x in glabels]
#     tstfeed['names'][bsize] = all_names
#     tstfeed['iconfigs'][bsize] = all_configs


# # As part of tests for saving the parts of the feeds that depend on the composition
# # of the batch
# new_hf = h5py.File("graph_save_tst.h5", 'w')
# per_batch_h5handler.unpack_save_feed_batch_h5(tstfeed, new_hf, 0)

# x = [tstfeed]
# per_molec_h5handler.add_per_molec_info(x, master_dict, ['Coords', 'Zs'])

# #Testing loading and combination
x = time.time()
final_feeds = total_feed_combinator.create_all_feeds('final_batch_test.h5', 'final_molec_test.h5')
print(f"Feed constitution time: {time.time() - x}")
compare_feeds('reference_data.p', final_feeds)
print("Check me!")

#%% Graph generation

# print('making graphs')
# feeds = list()
# degeneracies = list()
# degeneracy_tolerance = 1.0e-3
# debug_Etarget = list()
# debug_Ecalc = list()
# dftblists  = list()
# for batch in dataset:
#     feed, batch_dftblist = create_graph_feed(config, batch, allowed_Zs)
#     # TODO: will work only if 1 molecule in each batch
#     eorb = list(feed['eorb'].values())[0]
#     degeneracy = np.min(np.diff(np.sort(eorb)))
#     if degeneracy < degeneracy_tolerance:
#         continue
#     degeneracies.append(degeneracy)
#     feed['name'] = batch[0]['name']
#     feed['iconfig'] = batch[0]['iconfig']
#     for target in batch[0]['targets']:
#         if target == 'Etot':
#             Etot = list(feed['Eelec'].values())[0][0] + list(feed['Erep'].values())[0][0]
#             feed[target] = np.array([Etot])
#             debug_Etarget.append(batch[0]['targets']['Etot'])
#             debug_Ecalc.append(Etot)
#         else:
#             feed[target] = np.array([x['targets'][target] for x in batch])
#     feeds.append(feed)
#     dftblists.append(batch_dftblist)
# print('number of molecules after degeneracy rejection',len(feeds))
# times['graph'] = time.process_time()

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

#Shuffle the dataset before feeding into data_loader
random.shuffle(cleaned_dataset)

#Sample the indices that will be used for the training dataset randomly from the shuffled data
indices = [i for i in range(len(cleaned_dataset))]
sampled_indices = set(random.sample(indices, int(len(cleaned_dataset) * prop_train)))

#Separate into the training and validation sets
training_molecs, validation_molecs = list(), list()
for i in range(len(cleaned_dataset)):
    if i in sampled_indices:
        training_molecs.append(cleaned_dataset[i])
    else:
        validation_molecs.append(cleaned_dataset[i])

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

print('Number of total molecules after degeneracy rejection', len(cleaned_dataset))

print("Making Training Graphs")
train_dat_set = data_loader(training_molecs, batch_size = num_per_batch)
training_feeds, training_dftblsts = list(), list()
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
        
    for target in batch[0]['targets']:
        if target == 'Etot': #Only dealing with total energy right now
            feed[target] = dict()
            
            if fit_to_DFTB:
                for bsize in all_bsizes:
                    feed[target][bsize] = feed['Eelec'][bsize] + feed['Erep'][bsize]
                    
            else:
                for bsize in all_bsizes:
                    glabels = feed['glabels'][bsize]
                    total_energies = [batch[x]['targets'][target] for x in glabels]
                    feed[target][bsize] = np.array(total_energies)
                    
    training_feeds.append(feed)
    training_dftblsts.append(batch_dftb_lst)

print("Making Validation Graphs")
validation_dat_set = data_loader(validation_molecs, batch_size = num_per_batch)
validation_feeds = list()
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
    
    for target in batch[0]['targets']:
        if target == 'Etot': #Only dealing with total energy right now
            feed[target] = dict()
            
            if fit_to_DFTB:
                for bsize in all_bsizes:
                    feed[target][bsize] = feed['Eelec'][bsize] + feed['Erep'][bsize]
            
            else:
                for bsize in all_bsizes:
                    glabels = feed['glabels'][bsize]
                    total_energies = [batch[x]['targets'][target] for x in glabels]
                    feed[target][bsize] = np.array(total_energies)

    validation_feeds.append(feed)
            
all_models = dict()
model_variables = dict() #This is used for the optimizer later on

loss_mod = loss_model(targets_for_loss, loss_temp) # Add this to the model dictionary

all_models['Eref'] = Reference_energy(allowed_Zs)
all_models['Loss'] = loss_mod
model_variables['Eref'] = all_models['Eref'].get_variables()

#More nuanced construction of config dictionary
model_range_dict = create_spline_config_dict(training_feeds + validation_feeds)

#%% Debugging h5 part 2

#Save all the molecular information
per_molec_h5handler.save_all_molec_feeds_h5(training_feeds, 'final_molec_test.h5')
per_batch_h5handler.save_multiple_batches_h5(training_feeds, 'final_batch_test.h5')

with open('reference_data.p', 'wb') as handle:
    pickle.dump(training_feeds, handle)

print("molecular and batch information successfully saved!")
print("reference solution also saved")



#%% Feed generation
print('Making training feeds')
for ibatch,feed in enumerate(training_feeds):
   for model_spec in feed['models']:
       if (model_spec not in all_models):
           all_models[model_spec] = get_model_value_spline_2(model_spec, model_range_dict)
           #all_models[model_spec] = get_model_dftb(model_spec)
           model_variables[model_spec] = all_models[model_spec].get_variables()
       model = all_models[model_spec]
       feed[model_spec] = model.get_feed(feed['mod_raw'][model_spec])

print('Making validation feeds')
for ibatch, feed in enumerate(validation_feeds):
    for model_spec in feed['models']:
        if (model_spec not in all_models):
            all_models[model_spec] = get_model_value_spline_2(model_spec, model_range_dict)
            #all_models[model_spec] = get_model_dftb(model_spec)
            model_variables[model_spec] = all_models[model_spec].get_variables()
        model = all_models[model_spec]
        feed[model_spec] = model.get_feed(feed['mod_raw'][model_spec])


for feed in training_feeds:
    recursive_type_conversion(feed)
for feed in validation_feeds:
    recursive_type_conversion(feed)
times['feeds'] = time.process_time()

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

times_per_epoch = list()
#%% Training loop
nepochs = 300
for i in range(nepochs):
    #Initialize epoch timer
    start = time.time()
    
    #Validation routine
    validation_loss = 0
    for elem in validation_feeds:
        output = dftblayer(elem, all_models)
        loss = all_models['Loss'].compute_loss(output, elem)
        validation_loss += loss.item()
    print("Validation loss:",i, np.sqrt(validation_loss/len(validation_feeds)) * 627.0, 'kcal/mol')
    validation_losses.append(np.sqrt(validation_loss/len(validation_feeds)) * 627.0)

    #Shuffle the validation data
    random.shuffle(validation_feeds)
    
    #Training routine
    epoch_loss = 0.0
    for feed in training_feeds:
        optimizer.zero_grad()
        output = dftblayer(feed, all_models)
        loss = all_models['Loss'].compute_loss(output, feed) #Loss in units of Ha^2
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    scheduler.step(epoch_loss) #Step on the epoch loss
    
    #Perform shuffling while keeping order b/w dftblsts and feeds consistent
    temp = list(zip(training_feeds, training_dftblsts))
    random.shuffle(temp)
    training_feeds, training_dftblsts = zip(*temp)
    training_feeds, training_dftblsts = list(training_feeds), list(training_dftblsts)
    
    print(i,np.sqrt(epoch_loss/len(training_feeds)) * 627.0, 'kcal/mol')
    training_losses.append(np.sqrt(epoch_loss/len(training_feeds)) * 627.0)
    
    #Update charges at specified epoch intervals
    # Uncomment later, not working with charge updates right now
    # if (i % 10 == 0):
    #     for j in range(len(training_feeds)):
    #         feed = training_feeds[j]
    #         dftb_list = training_dftblsts[j]
    #         op_dict = assemble_ops_for_charges(feed, all_models)
    #         update_charges(feed, op_dict, dftb_list)
    
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
print(f"total number of molecules per epoch: {total_num_molecs}")
print(f"total number of training molecules: {total_num_train_molecs}")
print(f"total number of validation molecules: {total_num_valid_molecs}")

#Writing diagnostic information for later user
with open("timing.txt", "a+") as handle:
    handle.write(f"Current time: {datetime.now()}\n")
    handle.write(f"Allowed Zs: {allowed_Zs}\n")
    handle.write(f"Heavy Atoms: {heavy_atoms}\n")
    handle.write(f"Molecules per batch: {num_per_batch}\n")
    handle.write(f"Total molecules per epoch: {total_num_molecs}\n")
    handle.write(f"Total number of training molecules: {total_num_train_molecs}\n")
    handle.write(f"Total number of validation molecules: {total_num_valid_molecs}\n")
    handle.write(f"Number of epochs: {nepochs}\n")
    handle.write(f"Eigen decomp method: {eig_method}\n")
    handle.write(f"Total training time, sum of epoch times (seconds): {sum(times_per_epoch)}\n")
    handle.write(f"Average time per epoch (seconds): {sum(times_per_epoch) / len(times_per_epoch)}\n")
    handle.write("No charge updating!\n")
    handle.write("Switched over to using non-shifted dataset\n")
    handle.write("\n")



