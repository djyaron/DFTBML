# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 23:03:05 2020

@author: Frank
"""
"""
TODO:
    1) Fix the losses, write it like a model (X)
    2) Simplify some things (where applicable) (X)
    3) Very stability of validation and training loss (X)
        Managed to keep training losses relatively stable using learning rate scheduler
        Random loss spikes / oscillations, unstable training (problym with Adam maybe?)
        1) linearly decrease training rate every 20 epochs (adaptive LR)
        2) Epsilon workaround https://stackoverflow.com/questions/42327543/adam-optimizer-goes-haywire-after-200k-batches-training-loss-grows
    4) Fix charge updating method (Update the correct field(s)) (X) 
"""
import math
import numpy as np
import os
import random
import pickle

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
from batch import create_batch, create_dataset, DFTBList
#from modelval import Val_model
# from modelspline import Spline_model
from modelspline import get_dftb_vals
from SplineModel_v3 import SplineModel, fit_linear_model
import pickle

from dftb_layer_splines_ani1ccx import get_targets_from_h5file

#Fix the ani1_path for now
ani1_path = 'data/ANI-1ccx_clean_shifted.h5'

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
    def __init__(self, device, dtype):
        super(DFTB_Layer, self).__init__()
        self.device = device
        self.dtype = dtype
    
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
            # ngeom = len(self.data['glabels'][bsize])
            # calc['Eelec'][bsize] = torch.zeros([ngeom], device = self.device).double()
            # calc['eorb'][bsize] = torch.zeros([ngeom,bsize], device = self.device).double()
            # calc['rho'][bsize]  = torch.zeros([ngeom,bsize,bsize], device = self.device).double()
            S1 = calc['S'][bsize]
            fock = calc['F'][bsize]
            if 'phiS' not in list(data_input.keys()):
                Svals, Svecs = torch.symeig(S1, eigenvectors = True) #The eigenvalues from torch.symeig are in ascending order, but Svecs remains orthogonal
                phiS = torch.matmul(Svecs, torch.diag(torch.pow(Svals, -0.5).view(-1)))
            else:
                phiS = data_input['phiS'][bsize]
            fockp = torch.matmul(torch.transpose(phiS, -2, -1), torch.matmul(fock, phiS))
            try:
                Eorb, temp2 = torch.symeig(fockp, eigenvectors = True)
            except:
                print('diagonalization failed for ',data_input['name'],data_input['iconfig'])
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
            calc['Eref'][bsize] = torch.dot(data_input['zcounts'][bsize].squeeze(0), ref_energy_variables)
        return calc

def loss_temp(output, data_dict, targets):
    '''
    Calculates the MSE loss for a given minibatch using the torch implementation of 
    MSELoss
    '''
    all_bsizes = list(output['Eelec'].keys())
    loss_criterion = nn.MSELoss() #Compute MSE loss by the pytorch specification
    target_tensors, computed_tensors = list(), list()
    for bsize in all_bsizes:
        # if (target_tensor == None) and (computed_tensor == None):
        #     # computed_tensor = output['Eelec'][bsize] + output['Eref'][bsize]
        #     computed_tensor = output['Eref'][bsize]
        #     target_tensor = data_dict['Etot']
        # else:
        #     # current_computed_tensor = output['Eelec'][bsize] + output['Eref'][bsize]
        #     current_computed_tensor = output['Eref'][bsize]
        #     current_target_tensor = data_dict['Etot']
        #     target_tensor = torch.cat((target_tensor, current_target_tensor))
        #     computed_tensor = torch.cat((computed_tensor, current_computed_tensor))
        computed_result = output['Erep'][bsize] + output['Eelec'][bsize] + output['Eref'][bsize] 
        target_result = data_dict['Etot']
        if len(computed_result.shape) == 0:
            computed_result = computed_result.unsqueeze(0)
        if len(target_result.shape) == 0:
            target_result = target_result.unsqueeze(0)
        computed_tensors.append(computed_result)
        target_tensors.append(target_result)
    total_targets = torch.cat(target_tensors)
    total_computed = torch.cat(computed_tensors)
    return loss_criterion(total_computed, total_targets) 

def loss_refactored(output, data_dict, targets):
    '''
    Slightly refactored loss, will be used within the objet oriented handler
    '''
    all_bsizes = list(output[targets[0]].keys())
    loss_criterion = nn.MSELoss()
    target_tensors, computed_tensors = list(), list()
    for bsize in all_bsizes:
        for target in targets:
            computed_tensors.append(output[target][bsize])
            target_tensors.append(output[target][bsize])
    assert(len(target_tensors) == len(computed_tensors))
    for i in range(len(target_tensors)):
        if len(target_tensors[i].shape) == 0:
            target_tensors[i] = target_tensors[i].unsqueeze(0)
        if len(computed_tensors[i].shape) == 0:
            computed_tensors[i] = computed_tensors[i].unsqueeze(0)
    total_targets = torch.cat(target_tensors)
    total_computed = torch.cat(computed_tensors)
    return loss_criterion(total_computed, total_targets)

def recursive_type_conversion(data, device = None, dtype = torch.double):
    '''
    Transports all the tensors stored in data to a tensor with the correct dtype
    on the correct device
    '''
    for key in data:
        if isinstance(data[key], np.ndarray):
            data[key] = torch.tensor(data[key], dtype = dtype, device = device)            
        elif isinstance(data[key], collections.OrderedDict) or isinstance(data[key], dict):
            recursive_type_conversion(data[key])

# Leave this off for now because all data will be in the repo (small datasets)
# Also going to ignore gammas_path
# if os.getenv("USER") == "yaron":
#     ani1_path = 'data/ANI-1ccx_clean_shifted.h5'
#     gammas_path = 'data/gammas_50_5_extend.h5'
# elif os.getenv("USER") == "francishe":
#     ani1_path = "/home/francishe/Downloads/ANI-1ccx_clean_shifted.h5"
#     gammas_path = "/home/francishe/Downloads/gammas_50_5_extend.h5"
# else:
#     raise ValueError("Invalid user")

# Update charges based on method in dftb.py:
    
# def get_dQ_from_H(self, newH, newG = None):
#     Hsave = self._coreH
#     self._coreH = newH
#     if newG is not None:
#         Gsave = self._gamma
#         self._gamma = self.FullBasisToShell(newG)
#     E,Flist,rholist, occ_rho_mask = self.SCF(get_occ_rho_mask=True)                            
#     S  = self.GetOverlap()
#     rho = 2.0 * rholist[0]
#     qBasis = (rho)*S
#     GOP = np.sum(qBasis,axis=1)
#     self._coreH = Hsave

#     # Current hack to deal with smearing = None
#     if self.smearing:
#         entropy_term = self.entropy * self.smearing
#     else:
#         entropy_term = 0.0
#     if newG is not None:
#         self._gamma = Gsave
#     return self._qN - GOP, occ_rho_mask, entropy_term


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
                


#%%

allowed_Zs = [1,6,7]
heavy_atoms = [1,2,3,4,5,6,7]
max_config = 3
target = 'dt'
exclude = ['O3', 'N2O1']
# Parameters for configuring the spline
num_knots = 50
max_val = None

reference_energies = list() # Save the reference energies to see how the losses are really changing
training_losses = list()
first_epoch_losses = list()
times = collections.OrderedDict()
times['init'] = time.process_time()
dataset = get_ani1data(allowed_Zs, heavy_atoms, max_config, target, exclude=exclude)
times['dataset'] = time.process_time()
print('number of molecules retrieved', len(dataset))
#%%
config = dict()
config['opers_to_model'] = ['H', 'R']
targets_for_loss = ['Eelec', 'Eref']

print('making graphs')
feeds = list()
degeneracies = list()
degeneracy_tolerance = 1.0e-3
debug_Etarget = list()
debug_Ecalc = list()
dftblists  = list()
for batch in dataset:
    feed, batch_dftblist = create_graph_feed(config, batch, allowed_Zs)
    # TODO: will work only if 1 molecule in each batch
    eorb = list(feed['eorb'].values())[0]
    degeneracy = np.min(np.diff(np.sort(eorb)))
    if degeneracy < degeneracy_tolerance:
        continue
    degeneracies.append(degeneracy)
    feed['name'] = batch[0]['name']
    feed['iconfig'] = batch[0]['iconfig']
    for target in batch[0]['targets']:
        if target == 'Etot':
            Etot = list(feed['Eelec'].values())[0][0] + list(feed['Erep'].values())[0][0]
            feed[target] = np.array([Etot])
            debug_Etarget.append(batch[0]['targets']['Etot'])
            debug_Ecalc.append(Etot)
        else:
            feed[target] = np.array([x['targets'][target] for x in batch])
    feeds.append(feed)
    dftblists.append(batch_dftblist)
print('number of molecules after degeneracy rejection',len(feeds))
times['graph'] = time.process_time()

all_models = dict()
model_variables = dict() #This is used for the optimizer later on

loss_mod = loss_model(targets_for_loss, loss_temp) # Add this to the model dictionary

all_models['Eref'] = Reference_energy(allowed_Zs)
all_models['Loss'] = loss_mod
model_variables['Eref'] = all_models['Eref'].get_variables()

#Need more nuanced construction of the configuration dictionary
model_range_dict = create_spline_config_dict(feeds)

#%%
print('making feeds')
for ibatch,feed in enumerate(feeds):
   for model_spec in feed['models']:
       if (model_spec not in all_models):
           all_models[model_spec] = get_model_value_spline_2(model_spec, model_range_dict)
           #all_models[model_spec] = get_model_dftb(model_spec)
           model_variables[model_spec] = all_models[model_spec].get_variables()
       model = all_models[model_spec]
       feed[model_spec] = model.get_feed(feed['mod_raw'][model_spec])


for feed in feeds:
    recursive_type_conversion(feed)
times['feeds'] = time.process_time()


dftblayer = DFTB_Layer(device = None, dtype = torch.double)
learning_rate = 1.0e-5
optimizer = optim.Adam(list(model_variables.values()), lr = learning_rate, amsgrad=True)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 10, threshold = 0.01) #Experiment withb alternative schedulers?

#%%
nepochs = 300
for i in range(nepochs):
    epoch_loss = 0.0
    for feed in feeds:
        optimizer.zero_grad()
        output = dftblayer(feed, all_models)
        loss = all_models['Loss'].compute_loss(output, feed) #Loss in units of Ha^2
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    scheduler.step(epoch_loss) #Step on the epoch loss
    #Perform shuffling while keeping order b/w dftblsts and feeds consistent
    temp = list(zip(feeds, dftblists))
    random.shuffle(temp)
    feeds, dftblists = zip(*temp)
    feeds, dftblists = list(feeds), list(dftblists)
    if (i == 0):
        print(f"First epoch loss is {epoch_loss}")
        first_epoch_losses.append(epoch_loss)
    print(i,np.sqrt(epoch_loss/len(feeds)) * 627.0, 'kcal/mol')
    training_losses.append(np.sqrt(epoch_loss/len(feeds)) * 627.0)
    if (i % 10 == 0):
        # Update charges every 10 epochs
        for j in range(len(feeds)):
            feed = feeds[j]
            dftb_list = dftblists[j]
            op_dict = assemble_ops_for_charges(feed, all_models)
            update_charges(feed, op_dict, dftb_list)
print("Finished with 100 epochs")
times['train'] = time.process_time()

print('dataset with', len(feeds), 'batches')
time_names  = list(times.keys())
time_vals  = list(times.values())
for itime in range(1,len(time_names)):
    if time_names[itime] == 'train':
        print(time_names[itime], (time_vals[itime] - time_vals[itime-1])/nepochs)
    else:
        print(time_names[itime], time_vals[itime] - time_vals[itime-1])

with open("losses.p", "wb") as handle:
    pickle.dump(training_losses, handle)
    pickle.dump(first_epoch_losses, handle)
    pickle.dump([100, 100, 1], handle) #Plot configuration information for future use

