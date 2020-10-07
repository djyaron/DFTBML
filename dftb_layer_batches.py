# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 14:08:18 2020

@author: Frank
"""
"""
DFTBML layer with minibatching capabilities

Going to try and incorporate the dataset and dataloader class functionalities
since that's the direction of development anyway

"""
"""
#TODO:
    1) Fix stability issues with symeig non-convergence over large numbers of epochs
    2) Check through the code, make sure calculations are being done correctly
    3) Experiment with different optimizers and conditions and such

Thoughts:
    1) Could stability and weird behavior issues be because of how the loss is computed? Similar problems never occurred 
       in previous iteration of program that didn't use batching
       Maybe only looking at the energies is allowing the inputs to vary in uncontrolled manner?
    2) Are there alternatives to symeig for batch processing? Unlikely...
    3) How much of an effect would different optimizers really have? 
    4) Does shuffling in the dataloader actually work?
    5) It seems like whenever the loss gets stuck, symeig begins to behave strangely...this could be an indication of 
       something!
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

"""
Global parameters available for changing
"""
ngeoms_total = 20 #Start with something small
Zs = [7,6,1]
opers_to_model = ['H','G','R']
bond_length_range = [0.7,1.1]
bond_angle_range = [(104.7+20.0)*math.pi/180.0,(104.-20.0)*math.pi/180.0]
size_per_batch = 4
num_epochs = 100

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


class geometry_dataset(Dataset):
    '''
    Generic dataset for containing the geometries to be batched and used
    Very simple wrapper with no complex logic
    '''
    def __init__(self, data):
        self.data = list(map(lambda x : vars(x), data))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
def create_batch_graph_feed(geom_batch):
    '''
    Takes in a list of geometries and creates the graph and feed dictionaries for that batch
    '''
    batch = create_batch(geom_batch)
    dftblist = DFTBList(batch)
    feed = create_dataset(batch,dftblist,feed_fields)
    graph = create_dataset(batch, dftblist, graph_fields)
    return feed, graph

def torch_segment_sum(data, segment_ids, device, dtype): # Can potentially replace with scatter_add, but not part of main ptyorch distro
    '''
     just going to use pytorch tensor instead of np array
    '''
    max_id = torch.max(segment_ids)
    res = torch.zeros([max_id + 1], device = device, dtype = dtype)
    for i, val in enumerate(data):
        res[segment_ids[i]] += val
    return res

def find_input_params(feed, graph, oper, ngeoms):
    ## Going to apply this to different parameters this time, will
    ## have to modify because of the feed/graph separation, and negoms is the number
    ## of original geometries in the batch
    '''
    Parameters
    ----------
    feed : dictionary object
        The feed for the given minibatch
    graph: dictionary object
        The graph for the given minibatch
    oper : string
        The string representing the operator of interest.
    ngeoms : int
        The number of geometries

    Returns
    -------
    None.

    '''    
    trainable_models = [(index, elem) for index, elem in enumerate(graph['models'])
                        if len(elem.Zs) == 1 and elem.oper == oper]
    actual_values_indices = []
    for item in trainable_models:
        index = item[0]
        dftb_val = [elem.dftb for elem in feed['mod_raw'][item[1]]][0]
        actual_values_indices.append((index, dftb_val))
    #Since using on-diagonal elements, they will all have same DFTB value
    index_mapping = {}
    for pair in actual_values_indices:
        start, limit = pair[0] * ngeoms, (pair[0] + 1) * ngeoms
        if pair[1] in index_mapping:
            index_mapping[pair[1]].extend([ind for ind in range(start, limit)])
        else:
            index_mapping[pair[1]] = [ind for ind in range(start, limit)]
    return index_mapping

def compress_back_to_geoms(batch_dictionary):
    '''
    This method takes a batch_dictionary of deconstructed geom objects and turns them back into geoms
    '''
    geom_list = []
    for i in range(len(batch_dictionary['z'])):
        constructed_geom = Geometry(batch_dictionary['z'][i], batch_dictionary['rcart'][i])
        geom_list.append(constructed_geom)
    return geom_list
    
    
def generate_dummy_data(): #For testing trainloader
    dict_list = list()
    for i in range(20):
        dict_list.append({"z" : np.array([i, i, i]), "r" : np.array([i, i, i])})
    return dict_list
        

"""
Model Start
"""
class DFTB_Layer(nn.Module):
    '''
    Custom class representing the DFTBML layer in pytorch.
    
    This version of the layer is for CPU use only!
    
    The layer takes a few inputs, those being:
        input_mapping: A dictionary generated from find_input_params
        data: the data object to be used throughout
        ngeom: number of geometries
        ignore_oper: which operator was used in find_input_params
            (Assumed that only on-diagonal elements are considered)
        utilize_gpu: boolean flag whether to use GPU or not
        device: torch.cuda device, set to None if no gpu being used
        
    The rest of the logic should follow the code implemented in batch_demo_torch.py,
    though there are concerns as to whether the model (as it is currently) will be differentiable.
    Will have to be very vigilant about which tensors have their gradients tracked.
    
    NOTE: unlike tensorflow, pytorch is heavily dependent on the requires_grad value to
    determine whether or not to train a given input. This is important because requires_grad
    will propagate onwards throughout the rest of the values, so requiring_grad at the start 
    means that any other tensor / pytorch object that comes in contact with that input
    will also have requires_grad = True
    
    Pytorch gradient functions support things like indexing and slicing, and will be able to do that
    in reverse. This is great news!
    
    The convention followed here is that the variable 'x' is used to represent the input 
    to the network
    '''
    
    
    '''
    input_mapping, grapb, feed, ngeom should be given to forward
    '''
    def __init__(self, device, dtype, ignore_oper, ngeom):
        super(DFTB_Layer, self).__init__()
        self.device = device
        self.dtype = dtype
        self.ngeom = ngeom #Number of geometries will be constant due to batching
        self.ignore_oper = ignore_oper
    
    def separate_value_from_indices(self, input_mapping):
        #This function should be run the first time to truly get the inputs
        #MUST BE RUN BEFORE GOING THROUGH THE MODEL TO SET SELF.INDEX_MAPPING PROPERLY
        value_lst, indices_lst = [], []
        for pair in input_mapping.items():
            value_lst.append(pair[0])
            indices_lst.append(pair[1])
        #This guarantees that the index maps have the same order as the input
        self.index_mapping = indices_lst #In theory this will have to be done one time each batch?
        input_tensor = torch.tensor(value_lst, dtype = self.dtype, device = self.device, 
                                          requires_grad = True)
        ## FOR TESTING PURPOSES ONLY RETURN A TENSOR OF 5 1'S ##
        dummy_tensor = torch.tensor([1,1,1], dtype = self.dtype, device = self.device, 
                                          requires_grad = True)
        return input_tensor, dummy_tensor
    
    def recursive_type_conversion(self, data):
        '''
        Recursively traverse the dictionary and convert all instances of np.ndarray to 
        torch.DoubleTensor
        '''
        for key in data:
            if isinstance(data[key], np.ndarray):
                data[key] = torch.tensor(data[key], dtype = self.dtype, device = self.device)            
            elif isinstance(data[key], collections.OrderedDict) or \
                isinstance(data[key], dict):
                self.recursive_type_conversion(data[key])
    
    def form_initial_layer(self, x, graph, feed):
        '''
        create the net_vals input layer, filling in the constants before taking the input and filling in the holes
        Expected that x is a torch.DoubleTensor
        
        All these operations depend on data['model'] and data['mod_raw'] so they 
        will occur on the CPU first
        
        No need to require grad because as long as the input has requires_grad, 
        the entire model will propagate as requires_grad
        '''
        net_raw = []
        for m in graph['models']:
            if (m.oper == self.ignore_oper) and (len(m.Zs) == 1):
                net_raw.extend([0] * self.ngeom)
            else:
                net_raw.extend(feed['mod_raw'][m])
        #net_raw is now a list of numeric values with 0's in the 
        net_vals = torch.tensor([x.dftb if x != 0 else x for x in net_raw], dtype = self.dtype,
                                device = self.device)
        #Fill in the gaps with the input value
        for index, element in enumerate(x):
            target_indices = self.index_mapping[index]
            net_vals[target_indices] = element
        return net_vals
    
    def forward(self, x, input_mapping, graph, feed):
        '''
        Main overwrite for the derived class from nn.module
        
        For more information on each of the steps, refer to batch_demo_torch.py
        '''
        calc = OrderedDict() #calc object for maintaining the computation results
        # start = time.time()
        ## SLATER-KOSTER ROTATIONS ##
        net_vals = self.form_initial_layer(x, graph, feed)
        #Need to recrusive correction here
        self.recursive_type_conversion(graph), self.recursive_type_conversion(feed)
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
"""
Model End
"""

def compute_E_based_loss(output, feed, dtype, i):
    '''
    Compares the total energy (Erep + Eelec) for the feed against the sum created by the output
    The value i is only used for debugging purposes
    '''
    all_bsizes = list(output['Eelec'].keys())
    max_loss = 0.0 #point of comparison
    for bsize in all_bsizes:
        computed_total = output['Eelec'][bsize] + output['Erep'][bsize]
        comparison_total = feed['Eelec'][bsize] + feed['Erep'][bsize]
        if (i > 90):
            print("printing the total energies")
            print(computed_total)
            print(comparison_total)
        loss = (torch.max(torch.abs(computed_total - comparison_total)))
        if loss > max_loss:
            max_loss = loss
    return max_loss

def compute_total_loss(output, feed, dtype):
    '''
    Compares the loss in terms of all quantities represented in the feed
    '''
    pass

if __name__ == "__main__":
    """
    Going to have generic training loop and dataset generation techniques here
    """
    ## Generate geometries ##
    geoms = random_triatomics(ngeoms_total, Zs, bond_length_range, bond_length_range,
                              bond_angle_range)
    geom_dataset = geometry_dataset(geoms)
    # dummy_data = geometry_dataset(generate_dummy_data())
    #Data loader does automatic shuffling
    #Can set num_workers = 0 for improved performance through multithreading
    trainloader = DataLoader(geom_dataset, batch_size = size_per_batch, shuffle = True) 
    #Trainloader stacks things vertically within batches
    
    #Main training loop
    #Will loop through epochs and then through the samples in trainloader
    use_gpu = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if use_gpu else None  
    dtype = torch.float
    ignore_oper = 'H'
    input_actual, input_x_dummy = None, None #Because all molecules are the same, get the input once
    learning_rate = 1e-3
    optimizer = None
    dftb_layer = DFTB_Layer(device, dtype, ignore_oper, size_per_batch)
    for i in range(num_epochs):
        epoch_loss = 0
        for elem in trainloader:
            geom_list = compress_back_to_geoms(elem)
            feed, graph = create_batch_graph_feed(geom_list)
            input_mapping = find_input_params(feed, graph, 'H', len(geom_list))
            #This contidition only gets caught the first time
            if (input_x_dummy is None) and (input_actual is None):
                input_actual, input_x_dummy = dftb_layer.separate_value_from_indices(input_mapping)
                optimizer = optim.SGD([input_x_dummy], lr = learning_rate, momentum = 0.9) #Try adam for a bit
            optimizer.zero_grad() #Guaranteed to have one by now
            output = dftb_layer(input_x_dummy, input_mapping, graph, feed) #Graph and feed are converted over to tensors within the forward method
            loss = compute_E_based_loss(output, feed, dtype, i)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f"Epoch: {i}\tLoss: {epoch_loss}")
        print(input_actual)
        print(input_x_dummy)
        
    print(input_actual)
    print(input_x_dummy)
        
            
            
        
        
            
            
        
        
    
    

