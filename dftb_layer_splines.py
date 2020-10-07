# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 23:03:05 2020

@author: Frank
"""

"""
Attempt at integrating the spline interface with the DFTB layer, hope that 
everything works out alright...


TODO:
    1) Figure out how the spline models work so that you can incorporate them (and maybe improve their interfaces)
    2) Reconfigure the DFTB Layer from previous pytorch implementations to make sure that it works.
    3) If the symeig convergence problem persists, may have to include gradient 
"""

import math
import numpy as np

from collections import OrderedDict
import collections
import torch
torch.set_printoptions(precision = 10)
from copy import deepcopy
import pdb
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import time
from datetime import date

from geometry import Geometry, random_triatomics, to_cart
from auorg_1_1 import ParDict
from dftb import DFTB, ANGSTROM2BOHR
from batch import create_batch, create_dataset, DFTBList
#from modelval import Val_model
# from modelspline import Spline_model
from modelspline import get_dftb_vals
from SplineModel_v3 import SplineModel, fit_linear_model
from mio_0_1 import ParDict
# For the version 3 of the splines, using the pairwise linear model for off-diagonal elements

"""
Global parameters available for changing
"""
ngeoms_total = 36 #Start with something small
Zs = [7,6,1]
opers_to_model = ['H','G','R']
bond_length_range = [0.7,1.1]
bond_angle_range = [(104.7+20.0)*math.pi/180.0,(104.-20.0)*math.pi/180.0]
size_per_batch = 4
num_epochs = 12_000 #number of times each batch is fed through

fields_by_type = dict()

fields_by_type['graph'] \
    = ['models','onames','basis_sizes','glabels',
       'gather_for_rot', 'gather_for_oper',
       'gather_for_rep','segsum_for_rep', 'occ_rho_mask',
       'occ_eorb_mask','qneutral','atom_ids','norbs_atom']
    
fields_by_type['feed_constant'] = \
    ['geoms','mod_raw', 'rot_tensors', 'dftb_elements', 'dftb_r','Erep', 'rho']
    
if 'S' not in opers_to_model:
    fields_by_type['feed_constant'].extend(['S','phiS','Sevals'])
if 'G' not in opers_to_model:
    fields_by_type['feed_constant'].extend(['G'])
    
fields_by_type['feed_SCF'] = \
    ['dQ','Eelec']

feed_fields = fields_by_type['feed_constant'] + fields_by_type['feed_SCF']
graph_fields = fields_by_type['graph']

#Need to construct a config dictionary for splines, similar to the modelspline
config = dict() #Set as empty dictionary now and tune later
config['xknots'] = np.linspace(0.5, 2, num = 20) #Set 20 equidistant

"""
Methods for dealing with splines for computations

The idea is to use the spline models for modeling the on diagonal and off-diagonal elements 
(but this will require initializing the models that are already implemented)

For each batch, graph and feeds are calculated and saved because each batch will be pushed
through the layer a large number of times. The models (and losses too) will have to 
add things to the feed and the graph

Will have to interface with the splines, though they should all have the methods contained
within their classes (closer inspection of their class files is necessary)
"""

class geometry_dataset(Dataset):
    '''
    Generic dataset for containing the geometries to be batched and used
    Very simple wrapper with no complex logic
    '''
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def generate_dummy_data(): #For testing trainloader
    dict_list = list()
    for i in range(20):
        temp_dict = dict()
        temp_dict['geom'] = {'z': np.array([i,i,i]), 'r': np.array([i,i,i,i])}
        temp_dict['targets'] = {'E' : i, 'F' : np.array([[i,i,i], [i,i,i], [i,i,i]])}
        dict_list.append(temp_dict)
    return dict_list

def create_batch_graph_feed(geom_batch):
    '''
    Takes in a list of geometries and creates the graph and feed dictionaries for that batch
    '''
    batch = create_batch(geom_batch)
    dftblist = DFTBList(batch)
    feed = create_dataset(batch,dftblist,feed_fields)
    graph = create_dataset(batch, dftblist, graph_fields)
    return feed, graph

def vals(A, b, coeffs):
    '''
    Evaluates the spline at the given distances using the saved A and b from the feed
    and the coefficients from the graph. The xvalues to evaluate at are assumed to not change throughout the training process
    as A and b are defined with regards to the distances of interest. Thus, xvals are not a parameter
    '''
    return torch.matmul(A, coeffs) + b

def torch_segment_sum(data, segment_ids, device, dtype): # Can potentially replace with scatter_add, but not part of main ptyorch distro
    '''
     just going to use pytorch tensor instead of np array
    '''
    max_id = torch.max(segment_ids)
    res = torch.zeros([max_id + 1], device = device, dtype = dtype)
    for i, val in enumerate(data):
        res[segment_ids[i]] += val
    return res

def compress_back_to_geoms(batch_dictionary):
    '''
    This method takes a batch_dictionary of deconstructed geom objects and turns them back into geoms
    '''
    geom_list = []
    for i in range(len(batch_dictionary['geom']['z'])):
        constructed_geom = Geometry(batch_dictionary['geom']['z'][i], batch_dictionary['geom']['rcart'][i])
        geom_list.append(constructed_geom)
    return geom_list

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

class Val_model:
    def __init__(self,model, initial_value=0.0):
        self.model = model
        if not isinstance(initial_value, float):
            raise ValueError('Val_model not initialized to float')
        self.value = np.array([initial_value])
    def get_variables(self):
        return self.value
    def initialize_to_dftb(self,pardict, noise_magnitude = 0.0):
        init_value = get_dftb_vals(self.model, pardict)
        if not noise_magnitude == 0.0:
            init_value = init_value + noise_magnitude * np.random.randn(1)
        self.value[0]= init_value
    
def form_initial_layer(graph, feed, ngeom, device = None, dtype = torch.double):
    '''
    Form the initial layer (the net_vals) for dftb layer from the data contained
    in the graph and the feed
    '''
    net_vals = torch.empty(0, dtype = dtype, device = device)
    for m in graph['models']:
        if len(m.Zs) == 1:
            net_vals = torch.cat((net_vals, graph['variables'][m].repeat(ngeom)))
        elif len(m.Zs) == 2:
            A, b = feed['spline_A_b'][m]
            coeffs = graph['variables'][m]
            result = vals(A, b, coeffs)
            net_vals = torch.cat((net_vals, result))
    return net_vals

def compute_E_based_loss(output, feed, dtype):
    '''
    Compares the total energy (Erep + Eelec) for the feed against the sum created by the output
    The value i is only used for debugging purposes
    
    Placeholder loss for now until more defined loss functions later, just use this to get the 
    network up and running
    '''
    all_bsizes = list(output['Eelec'].keys())
    max_loss = 0.0 #point of comparison
    for bsize in all_bsizes:
        computed_total = output['Eelec'][bsize] + output['Erep'][bsize]
        comparison_total = feed['Eelec'][bsize] + feed['Erep'][bsize]
        loss = (torch.max(torch.abs(computed_total - comparison_total)))
        if loss > max_loss:
            max_loss = loss
    return max_loss

class DFTB_Layer(nn.Module):
    def __init__(self, device, dtype, ngeom):
        super(DFTB_Layer, self).__init__()
        self.device = device
        self.dtype = dtype
        self.ngeom = ngeom #Number of geometries per batch
    
    def forward(self, graph, feed):
        '''
        The forward method this time only takes in the graph and the feed and nothing else;
        the variables for optimization are contained in the graph, other information 
        is contained in the feed
        '''
        calc = OrderedDict() #calc object for maintaining the computation results
        # start = time.time()
        ## SLATER-KOSTER ROTATIONS ##
        net_vals = form_initial_layer(graph, feed, self.ngeom, self.device, self.dtype)
        #Need to recrusive correction here
        rot_out = torch.tensor([0.0, 1.0], dtype = self.dtype, device = self.device, requires_grad = True)
        for s, gather in graph['gather_for_rot'].items():
            gather = gather.long()
            if feed['rot_tensors'][s] is None:
                rot_out = torch.cat((rot_out, net_vals[gather]))
            else:
                vals = net_vals[gather].reshape((-1, s[1])).unsqueeze(2)
                tensor = feed['rot_tensors'][s]
                rot_out_temp = torch.matmul(tensor, vals).squeeze(2)
                rot_out = torch.cat((rot_out, torch.flatten(rot_out_temp)))
        # print(f"Time taken for SK Rotations is {time.time() - start}")
        
        # start = time.time()
        ## ASSEMBLE SK ROTATION VALUES INTO OPERATORS ##
        for oname in graph['onames']:
            calc[oname] = {}
            if oname != "R":
                for bsize in graph['basis_sizes']:
                    gather = graph['gather_for_oper'][oname][bsize].long()
                    calc[oname][bsize] = rot_out[gather].reshape((len(graph['glabels'][bsize]),bsize,bsize))
        
        if 'S' not in graph['onames']:
            calc['S'] = deepcopy(feed['S']) #Deepcopy operations may be too inefficient...
        if 'G' not in graph['onames']:
            calc['G'] = deepcopy(feed['G'])
        # print(f"Time taken for operator assembly is {time.time() - start}")
        
        # start = time.time()
        ## CONSTRUCT THE FOCK OPERATORS ##
        calc['F'] = {}
        calc['dQ'] = {}
        calc['Erep'] = {}
        for bsize in graph['basis_sizes']:
        #Could potentially remove this outer loop by just having well-ordered tensors...
        #Might be able to do away with these castings and conditionals, may be too inefficient...
        #Thanks to data conversion, avoid casting
            # if not isinstance(calc['S'][bsize], torch.Tensor):
            #     calc['S'][bsize] = torch.DoubleTensor(calc['S'][bsize])
            # if not isinstance(calc['G'][bsize], torch.Tensor):
            #     calc['G'][bsize] = torch.DoubleTensor(calc['G'][bsize])
            # if not isinstance(calc['H'][bsize], torch.Tensor):
            #     calc['H'][bsize] = torch.DoubleTensor(calc['H'][bsize])
            rho = feed['rho'][bsize]
            qbasis = rho * calc['S'][bsize]
            GOP  = torch.sum(qbasis,2,keepdims=True)
            qNeutral = graph['qneutral'][bsize]
            calc['dQ'][bsize] = qNeutral - GOP
            ep = torch.matmul(calc['G'][bsize], calc['dQ'][bsize])
            couMat = ((-0.5 * calc['S'][bsize]) *  (ep + torch.transpose(ep, -2, -1)))
            calc['F'][bsize] = calc['H'][bsize] + couMat 
            vals = net_vals[graph['gather_for_rep'][bsize].long()] # Already a tensor
            #The segment_sum is going to be problematic
            calc['Erep'][bsize] = torch_segment_sum(vals,
                                graph['segsum_for_rep'][bsize].long(), self.device, self.dtype)
        # print(f"Time taken for fock operator construction is {time.time() - start}")
        # start = time.time()
        ## SOLVE GEN-EIG PROBLEM FOR FOCK ##
        # Maybe switch out symeig for eig?
        calc['Eelec']= {}
        calc['eorb'] = {}
        calc['rho'] = {}
        for bsize in graph['basis_sizes']:
            # ngeom = len(self.data['glabels'][bsize])
            # calc['Eelec'][bsize] = torch.zeros([ngeom], device = self.device).double()
            # calc['eorb'][bsize] = torch.zeros([ngeom,bsize], device = self.device).double()
            # calc['rho'][bsize]  = torch.zeros([ngeom,bsize,bsize], device = self.device).double()
            S1 = calc['S'][bsize]
            fock = calc['F'][bsize]
            if 'phiS' not in list(feed.keys()):
                Svals, Svecs = torch.symeig(S1, eigenvectors = True) #The eigenvalues from torch.symeig are in ascending order, but Svecs remains orthogonal
                phiS = torch.matmul(Svecs, torch.diag(torch.pow(Svals, -0.5).view(-1)))
            else:
                phiS = feed['phiS'][bsize]
            fockp = torch.matmul(torch.transpose(phiS, -2, -1), torch.matmul(fock, phiS))
            Eorb, temp2 = torch.symeig(fockp, eigenvectors = True)
            calc['eorb'][bsize] = Eorb
            orb = torch.matmul(phiS, temp2)
            occ_mask = graph['occ_rho_mask'][bsize]
            orb_filled = torch.mul(occ_mask, orb)
            rho = 2.0 * torch.matmul(orb_filled, torch.transpose(orb_filled, -2, -1))
            calc['rho'][bsize] = rho
            ener1 = torch.sum(torch.mul(rho.view(rho.size()[0], -1), calc['H'][bsize].view(calc['H'][bsize].size()[0], -1)), 1) #I think this is fine since calc['Eelec'] is a 1D array
            dQ = calc['dQ'][bsize]
            Gamma = calc['G'][bsize]
            ener2 = 0.5 * torch.matmul(torch.transpose(dQ, -2, -1), torch.matmul(Gamma, dQ))
            ener2 = ener2.view(ener2.size()[0])
            calc['Eelec'][bsize] = ener1 + ener2
        # print(f"Time taken for solving the general eigenvalue problem is {time.time() - start}")
        return calc
## TOP LEVEL CODE ##
#Make the geoms first
geoms = random_triatomics(ngeoms_total, Zs, bond_length_range, bond_length_range,
                              bond_angle_range)

#Go ahead and do some pre-compute for all the molecules to get them in the dictionary form
# (Taken from batch_demo_multibatch.py)

molecs = list()
#Generates the molecule data, now for batching and precompute
for geom in geoms:
    dftb = DFTB(ParDict(), to_cart(geom), charge = 0)
    E, Fs, rhos = dftb.SCF()
    molecule_data = dict()
    molecule_data['geom'] = vars(geom)
    molecule_data['targets'] = {'Etot' : E, 'F' : Fs[0]} #Targets for fitting
    molecs.append(molecule_data)

molec_data = geometry_dataset(molecs)
# molec_data = geometry_dataset(generate_dummy_data())
trainloader = DataLoader(molec_data, batch_size = size_per_batch, shuffle = True) 
"""
For the spline models, there seems to be a need for style, which is something indicated in the 
tfnet.py model

How exactly this style is determined remains to be determined

TODO: Continue from here!
"""
par_dict = ParDict()
All_models = dict()
model_variables = dict()
noise_magnitude = 0.5


saved_mod = None
# to continuously optimize them!
for index, batch in enumerate(trainloader):
    #Each batch needs to go through the pre-compute process
    geom_list = compress_back_to_geoms(batch)
    #Can also get the targets out from the batch here
    feed, graph = create_batch_graph_feed(geom_list)
    feed['spline_A_b'] = dict() #For containing the A and b of the splines
    graph['variables'] = dict() #For containing the coefficients for each model
    # valueb models will have a single array, non-value models will have (A, b) tuple
    #The feed will contain the distances for evaluation
    #We know all the models that we need from the graph
    for model in graph['models']:
        
        if len(model.Zs) == 1:
            #Fit on-diagonal using value model
            if (model not in All_models):
                value_model = Val_model(model)
                value_model.initialize_to_dftb(par_dict, noise_magnitude)
                All_models[model] = value_model
            else:
                value_model = All_models[model]
            if (model not in model_variables):
                #TODO: should maybe read this from skf files, instead of mod_raw
                #variable_value = list(value_model.get_variables().values())[0] #Should just be one value
                val_tensor = torch.from_numpy(value_model.get_variables())
                val_tensor.requires_grad = True
                model_variables[model] = val_tensor
                graph['variables'][model] = val_tensor
            else:
                graph['variables'][model] = model_variables[model]
                
        elif len(model.Zs) == 2:
            if (model not in All_models):
                spline_model = SplineModel(config)
                All_models[model] = spline_model
            else:
                spline_model = All_models[model]
            if (saved_mod is None): #DEBUGGING ONLY
                saved_mod = model
            xeval = np.array([elem.rdist for elem in feed['mod_raw'][model]])
            A, b = spline_model.linear_model(xeval)
            feed['spline_A_b'][model] = (torch.from_numpy(A), torch.from_numpy(b))
            if (model not in model_variables):
                # Initialize the spline to MIO values
                (rlow,rhigh) = spline_model.r_range()
                ngrid = 100
                rgrid = np.linspace(rlow,rhigh,ngrid)
                ygrid = get_dftb_vals(model, par_dict, rgrid)
                ygrid = ygrid + noise_magnitude * np.random.randn(len(ygrid))
                spline_vars,_,_ = fit_linear_model(spline_model, rgrid,ygrid) 
                variable_tensor = torch.from_numpy(spline_vars)
                variable_tensor.requires_grad = True
                model_variables[model] = variable_tensor
                graph['variables'][model] = variable_tensor
            else:
                graph['variables'][model] = model_variables[model]
            # This is basically the same as the get A, b for feeds
            # The linear model gives the A and b matrices for A*c + b to get the values,
            # but where do you get the variables? Or should those be initialized outside the interface of the spline?
            # the variables of the model can be initialized with nvars, as we can get the number of variables.
            # Can initialize variables as np.random.normal(size = self.nvars)

    recursive_type_conversion(feed)
    recursive_type_conversion(graph)
    ## SOME TESTING CODE ##
    # splines_only = [x for x in All_models if isinstance(x, SplineModel)]
    # vals_only = [x for x in All_models if isinstance(x, Val_model)]
    # test_spline = splines_only[0]
    # test_model_name = test_spline.mod
    # coeffs = graph['variables'][test_model_name]
    # A, b = feed['spline_A_b'][test_model_name]
    # result = vals(A, b, coeffs)
    '''
    Now that we have the splines and the batch defined, we need to consider how to interface the splines with the 
    DFTB Layer
    In this case, the number of knots and the distances can almost be considered hyper parameters for the model?
    '''
    # Declare the model
    dftb_layer = DFTB_Layer(device = None, dtype = torch.double, ngeom = size_per_batch)
    #Do a single output for debugging purposes
    output = dftb_layer(graph, feed)
    learning_rate = 1e-5
    optimizer = optim.SGD(list(graph['variables'].values()), lr = learning_rate, momentum = 0.9)
    # print(feed['spline_A_b'][saved_mod])
    # start = deepcopy(feed['spline_A_b'][saved_mod])
    
    for i in range(num_epochs):
        optimizer.zero_grad()
        output = dftb_layer(graph, feed)
        loss = compute_E_based_loss(output, feed, dtype = torch.double)
        if (i % 1000 == 0):
            print(f"Iter {i}\tLoss: {loss}")
        loss.backward()
        optimizer.step()
    
    # for key in graph['variables']: # DEBUGGING
    #     print(graph['variables'][key])
    #     print(model_variables[key])
    #     print(f"The two are equal: {graph['variables'][key] == model_variables[key]}")
    #     print(graph['variables'][key] is model_variables[key])
        
    # print(feed['spline_A_b'][saved_mod])
    # end = feed['spline_A_b'][saved_mod]
    # print(torch.max(torch.abs(end[0] - start[0])))
    # print(torch.max(torch.abs(end[1] - start[1])))
    pass
    
    
    
        
    
    
    
            
            
        
        
# Model(oper='G', Zs=(6, 1), orb='ps')
    
    


    
    




