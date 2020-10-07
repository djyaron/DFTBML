# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 23:03:05 2020

@author: Frank
"""

"""
Attempt at integrating the spline interface with the DFTB layer, hope that 
everything works out alright...


TODO:
    1) Replace mentions of spline and value model with equivalent objects, replace trainloader with peronal loader (X)
    2) Remove graph-feed separation (forward pass of DFTB layer should take one dictionary of data) (X)
    3) Verify train-validation stability (X)
        Maybe this has to do with the splines not configured properly over the whole range of mod raw distances
"""

import math
import numpy as np
import random

from collections import OrderedDict
import collections
import torch
torch.set_printoptions(precision = 10)
from copy import deepcopy
import pdb
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

"""
Global parameters available for changing
"""
ngeoms_total = 100 #Start with something small
Zs = [7,6,1]
opers_to_model = ['H','R']
bond_length_range = [0.7,1.1]
bond_angle_range = [(104.7+20.0)*math.pi/180.0,(104.-20.0)*math.pi/180.0]
size_per_batch = 10
num_epochs = 3_000 #number of times each batch is fed through
num_outermost_loop = 1
num_valid_geoms = 16

#Fractional values for training and validation
fraction_valid = 0.20
fraction_train = 0.80

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
num_knots = 40
minimum_val = 0.0
maximum_val = 12.0
config['xknots'] = np.linspace(minimum_val, maximum_val, num = num_knots) #Set 20 equidistant

use_ani = True # Whether or not to use ani-1ccx data in the training instead of random triatomics
ani_path = 'dataset_40_molecs_5_permolec.p'

"""
Classes
"""
class geometry_dataset(Dataset):
    '''
    Generic dataset for containing the geometries to be batched and used
    Very simple wrapper with no complex logic
    
    This is for use with trainloader, but not necessary if not using PyTorch trainloader
    '''
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

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
    def get_variables(self):
        val_tensor = torch.from_numpy(self.value)
        val_tensor.requires_grad = True
        return val_tensor
    def initialize_to_dftb(self,pardict, noise_magnitude = 0.0):
        init_value = get_dftb_vals(self.model, pardict)
        if not noise_magnitude == 0.0:
            init_value = init_value + noise_magnitude * np.random.randn(1)
        self.value[0]= init_value

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
    def get_feed(self,xeval):
        ''' this returns numpy arrays '''
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

class DFTB_Layer(nn.Module):
    def __init__(self, device, dtype, ngeom):
        super(DFTB_Layer, self).__init__()
        self.device = device
        self.dtype = dtype
        self.ngeom = ngeom #Number of geometries per batch
    
    def forward(self, data_input, all_models):
        '''
        The forward pass now takes a full data dictionary (combination of graph and feed) and
        the all_models dictionary because we need that for form initial layer
        '''
        calc = OrderedDict() 
        ## SLATER-KOSTER ROTATIONS ##
        net_vals = form_initial_layer(data_input, all_models, self.device, self.dtype)
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
            Eorb, temp2 = torch.symeig(fockp, eigenvectors = True)
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
        return calc
    
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

def compress_back_to_geom_listwise(batch_list):
    '''
    This is an alternative method that works with a custom generated batch, i.e. a list of 
    molecule dictionaries that isn't stacked'
    '''
    geom_list = []
    for molecule in batch_list:
        constructed_geom = Geometry(molecule['geom']['z'], molecule['geom']['rcart'])
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

def form_initial_layer(data, all_models, device = None, dtype = torch.double):
    '''
    Form the initial layer (the net_vals) for dftb layer from the data contained in the 
    input dictionary, no more graph and feed separation. Also, now need to get the
    all_models dictionary going on in here to take advantage of the class method
    
    Will have to extend things by the length of the mod_raw, which is different for each molecule. 
    This will require major overhaul for the form_initial_layer function. Will it require an overhaul
    for the splines?
    '''
    net_vals = torch.empty(0, dtype = dtype, device = device)
    for m in data['models']:
        if len(m.Zs) == 1:
            number_to_expand = len(data['mod_raw'][m])
            net_vals = torch.cat((net_vals, data['variables'][m].repeat(number_to_expand)))
        elif len(m.Zs) == 2:
            feed = data['spline_A_b'][m]
            result = all_models[m].get_values(feed)
            net_vals = torch.cat((net_vals, result))
    return net_vals

def compute_E_based_loss(output, data_dict, dtype):
    '''
    Calculates the MSE loss for a given minibatch using the torch implementation of 
    MSELoss
    '''
    all_bsizes = list(output['Eelec'].keys())
    loss_criterion = nn.MSELoss() #Compute MSE loss by the pytorch specification
    target_tensor, computed_tensor = None, None
    for bsize in all_bsizes:
        if (target_tensor == None) and (computed_tensor == None):
            computed_tensor = output['Eelec'][bsize] + output['Erep'][bsize]
            target_tensor = data_dict['Eelec'][bsize] + data_dict['Erep'][bsize]
        else:
            current_computed_tensor = output['Eelec'][bsize] + output['Erep'][bsize]
            current_target_tensor = data_dict['Eelec'][bsize] + data_dict['Erep'][bsize]
            target_tensor = torch.cat((target_tensor, current_target_tensor))
            computed_tensor = torch.cat((computed_tensor, current_computed_tensor))
    return loss_criterion(computed_tensor, target_tensor) 

def check_alias(var_dict, graph_lst):
    '''
    Makes sure that all the variables in the var_dict are aliased by all the graphs
    (i.e. they refer to the same location in memory)
    
    Ensures that there is only one set of variables being optimized by each run 
    through the layer.
    '''
    mod_list = list(var_dict.keys())
    for graph in graph_lst:
        for model in mod_list:
            test_value, actual_value = var_dict[model], graph['variables'][model]
            if isinstance(test_value, tuple):
                assert(test_value[0] is actual_value[0])
                assert(test_value[1] is actual_value[1])
            else:
                assert(test_value is actual_value)

def check_object_alias(data_dict_lst, all_models):
    '''
    Checks three-fold aliasing between the variable dictionary, list of data dictionaries,
    and the variables themselves stored within the model objects in all_models
    
    At this level of alias checking, it is assumed that everything should be equal
    '''
    for data_dict in data_dict_lst:
        mod_lst = data_dict['variables'].keys()
        for model in mod_lst:
            data_dict_ptr = data_dict['variables'][model]
            mod_ptr = all_models[model].get_variables()
            assert(torch.allclose(mod_ptr, data_dict_ptr))

def find_config_range(data_dict_lst):
    '''
    Finds the range of distances for configuring the splines so that the models do 
    not behave crazily
    '''
    minimum, maximum = 0.0, 0.0
    for data_dict in data_dict_lst:
        mod_raw_lsts = [data for _, data in data_dict['mod_raw'].items()]
        distances = [x.rdist for lst in mod_raw_lsts for x in lst]
        (tempMax, tempMin) = max(distances), min(distances)
        if tempMax > maximum: maximum = tempMax
        if tempMin < minimum: minimum = tempMin
    return minimum, maximum

def validation_subroutine(valid_data_loader, all_models, layer):
    '''
    valid_data_loader: DataLoader class for loading in batches of validation data
    model_vars: the model variables that will be used
    layer: The DFTB layer to be used
    all_models: dictionary mapping the model named tuples to the actual instance of the models
    
    It is assumed that all the molecules in valid_data_laoder will use the same models
    or a subset of the models used for training, i.e. there does not exist a molecule in the validation
    set that uses different models than the ones given
    '''
    ## VALIDATION LOOP ##
    ## Here, we do not need to feed the same validation batch through a ton of times
    total_loss = 0.0 # accumulated loss for all validations
    loss_counter = 0
    for index, batch in enumerate (valid_data_loader):
        geom_list = compress_back_to_geom_listwise(batch)
        feed, graph = create_batch_graph_feed(geom_list)
        feed['spline_A_b'] = dict() #For containing the A and b of the splines
        graph['variables'] = dict() #For containing the coefficients for each model
        for model in graph['models']:
            if len(model.Zs) == 1:
                graph['variables'][model] = all_models[model].get_variables()
            elif len(model.Zs) == 2:
                spline_model = all_models[model]
                graph['variables'][model] = spline_model.get_variables()
                xeval = np.array([elem.rdist for elem in feed['mod_raw'][model]])
                AbDict = spline_model.get_feed(xeval)
                feed['spline_A_b'][model] = AbDict
        ## Now that the graph is ready, feed it through the layer, no need for optimizer
        full_data_dict = {**graph, **feed}
        recursive_type_conversion(full_data_dict)
        output = layer(full_data_dict, all_models)
        loss = compute_E_based_loss(output, full_data_dict, dtype = torch.double)
        total_loss += loss.item() #Only interested in the value of the los for validation
        loss_counter += 1
    return total_loss / loss_counter

"""
For the spline models, there seems to be a need for style, which is something indicated in the 
tfnet.py model

How exactly this style is determined remains to be determined

TODO: Continue from here!
"""
if __name__ == "__main__":
    ## Code for creating the data ##
    if not use_ani:
        #Make the geoms first
        geoms = random_triatomics(ngeoms_total, Zs, bond_length_range, bond_length_range,
                                      bond_angle_range)
        
        valid_geoms = random_triatomics(num_valid_geoms, Zs, bond_length_range, bond_length_range,
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
            
        validation_molecs = list() #Validation data for the model
        
        for geom in valid_geoms:
            dftb = DFTB(ParDict(), to_cart(geom), charge = 0)
            E, Fs, rhos = dftb.SCF()
            molecule_data = dict()
            molecule_data['geom'] = vars(geom)
            molecule_data['targets'] = {'Etot' : E, 'F' : Fs[0]} #Targets for fitting
            validation_molecs.append(molecule_data)
                
        molec_data = geometry_dataset(molecs)
        
        validation_data = geometry_dataset(validation_molecs)
    else:
        #For the ani-1ccx data, the train-valid split will be 80-20
        with open(ani_path, 'rb') as handle:
            full_molec_list = pickle.load(handle)
        random.shuffle(full_molec_list)
        num_train = int(len(full_molec_list) * fraction_train)
        molec_data = full_molec_list[ : num_train + 1]
        
        validation_data = full_molec_list[num_train + 1 : ]
        
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
  
    trainloader = data_loader(molec_data, batch_size = size_per_batch, shuffle = True) 
    validationloader = data_loader(validation_data, batch_size = size_per_batch, shuffle = True)
    
    ## Code for training and validation ##
    par_dict = ParDict()
    All_models = dict()
    model_variables = dict() #This is used for the optimizer later on
    noise_magnitude = 0.5
    learning_rate = 1e-4
    outer_start = time.time()
    training_losses = []
    validation_loss = []
    for i in range(num_outermost_loop): 
        data_dicts = [] 
        for index, batch in enumerate(trainloader):
            geom_list = compress_back_to_geom_listwise(batch)
            feed, graph = create_batch_graph_feed(geom_list)
            feed['spline_A_b'] = dict() 
            graph['variables'] = dict()
            for model in graph['models']:
                
                if len(model.Zs) == 1:
                    #Fit on-diagonal using value model
                    if (model not in All_models):
                        value_model = Input_layer_value(model)
                        value_model.initialize_to_dftb(par_dict, noise_magnitude)
                        All_models[model] = value_model
                        # From numpy should preserve aliasing
                        graph['variables'][model] = value_model.get_variables()
                        model_variables[model] = value_model.get_variables()
                    else:
                        value_model = All_models[model]
                        graph['variables'][model] = value_model.get_variables()
                        
                elif len(model.Zs) == 2:
                    if (model not in All_models):
                        # The SplineModel is a pairwise linear
                        spline_model = Input_layer_pairwise_linear(model, SplineModel(config), par_dict)
                        All_models[model] = spline_model
                        graph['variables'][model] = spline_model.get_variables()
                        model_variables[model] = spline_model.get_variables()
                    else:
                        spline_model = All_models[model]
                        graph['variables'][model] = spline_model.get_variables()
                    xeval = np.array([elem.rdist for elem in feed['mod_raw'][model]])
                    AbDict = spline_model.get_feed(xeval)
                    feed['spline_A_b'][model] = AbDict
                
            full_data_dict = {**graph, **feed} # We know that there are no overlapping keys between the dictionaries.
            recursive_type_conversion(full_data_dict)
            #recursive correction is not very efficient, but it guarantees everything that can/should be a torch
            # tensor is a torch tensor
            data_dicts.append(full_data_dict)
        
        check_object_alias(data_dicts, All_models)
        #  Use the find_config_range function to find the range for the config dictionary for spline models
        #  Not necessary every time, just once at the start
        #minimum, maximum = find_config_range(data_dicts)
        dftblayer = DFTB_Layer(device = None, dtype = torch.double, ngeom = size_per_batch)
        optimizer = optim.Adam(list(model_variables.values()), lr = learning_rate)
        # model_variables should contain all the variables for all the models!
        start = time.time()
        ## TRAINING LOOP ##
        for i in range(num_epochs):
            #Each epoch, go through all the batches
            # FIX validation subroutine later
            if (i % 5 == 0):
                valid_loss = validation_subroutine(validationloader, All_models, dftblayer)
                print(f"Epoch average validation loss: {valid_loss}")
                validation_loss.append(valid_loss)
            epoch_loss = 0
            epoch_counter = 0
            for data_dict in data_dicts:
                # optimizer = optim.Adam(list(data_dict['variables'].values()), lr = learning_rate)
                # Because strcutures are not the same across all data_dicts, need new optimizer each time
                optimizer.zero_grad()
                output = dftblayer(data_dict, All_models)
                loss = compute_E_based_loss(output, data_dict, dtype = torch.double)
                epoch_loss += loss.item()
                epoch_counter += 1
                loss.backward()
                optimizer.step()
            #Right now validation loss is being done per epoch
            if (i % 5 == 0):
                print(f"Epoch average training loss: {epoch_loss / epoch_counter}")
                training_losses.append(epoch_loss / epoch_counter)
            #Alias check, can comment out as alias was already verified to work
            # check_alias(model_variables, graphs) #THIS STEP IS BOTTLENECK, REMOVE WHEN NOT TESTING
        print(f"Finished {num_epochs} training epochs in {time.time() - start}")
    print(f"finished {num_outermost_loop} outer epochs in {time.time() - outer_start}")
    
    #Saving training losses
    with open("model_loss.p", "wb") as handle:
        print("Saving losses")
        pickle.dump(training_losses, handle)
        pickle.dump(validation_loss, handle)
    
    with open("model_dict.p", "wb") as handle:
        print("Saving models and variables")
        pickle.dump(All_models, handle)
        pickle.dump(model_variables, handle)
    


    
    




