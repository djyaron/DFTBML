# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 23:15:17 2020

@author: Frank
"""
'''
Functions for writing and handling h5 files

Note: to go back from bytestring to normal string, you have to do

    x.decode('UTF-8') where x is the byte string representation
    
TODO: 
    1) Finish implementing methods for feeds and for dictionaries
    2) Write better docstring detailing exactly how data is stored and how
       to use the functions in this module 
       
With this method of working with h5 files, the only certain pieces of information
still need to be computed in the pre-computation stage. Since most of the SCF
parameters are 
'''
import numpy as np
import h5py
from batch import Model
import collections

def get_model_from_string(model_spec : str):
    '''
    Converts the model_spec named tuple from a binary string representation to the
    named tuple representation. Will decode by UTF-8 first.
    '''
    return eval(model_spec.decode('UTF-8'))
    
def save_model_variables_h5(model_variables, filename):
    '''
    Saves the model_variables to an h5 file. Model variables are indexed by 
    the model_spec, i.e. the named tuple representing the model. Filename must
    be specified
    
    There will be two groups: off-diagonal and on-diagonal elements, i.e. splines
    versus model_values
    
    Save everything in different groups with the same order
    '''
    hf = h5py.File(filename, 'w')
    spline_keys, spline_vars = list(), list()
    value_keys, value_vars = list(), list()
    misc_keys, misc_vars = list(), list()
    for model_spec in model_variables:
        temp = model_variables[model_spec].detach().numpy()
        try:
            if len(model_spec.Zs) == 1:
                value_keys.append(str(model_spec))
                value_vars.append(temp)
            elif len(model_spec.Zs) == 2:
                spline_keys.append(str(model_spec))
                spline_vars.append(temp)
        except:
            misc_keys.append(str(model_spec))
            misc_vars.append(temp)
    
    spline_keys = [x.encode("ascii", "ignore") for x in spline_keys]
    value_keys = [x.encode("ascii", "ignore") for x in value_keys]
    misc_keys = [x.encode("ascii", "ignore") for x in misc_keys]
    
    hf.create_dataset('spline_mod_specs', data = spline_keys)
    hf.create_dataset('spline_mod_vars', data = np.array(spline_vars))
    
    hf.create_dataset('value_mod_specs', data = value_keys)
    hf.create_dataset('value_mod_vars', data = np.array(value_vars))
    
    hf.create_dataset('misc_mod_specs', data = misc_keys)
    hf.create_dataset('misc_mod_vars', data = misc_vars)
    
    hf.flush()
    hf.close()
    
def load_model_variables_h5(filename):
    '''
    Loads the model_variables from the model_variables dictionary saved in
    h5 format and reformats the model_variables dictionary. They are saved as
    np arrays.
    
    If intent is to use these variables in training, please use recursive type conversion
    to change into torch tensors with a gradient requirement.
    '''
    model_variables_np = dict()
    hf = h5py.File(filename, 'r')
    spline_mod_specs = list(hf['spline_mod_specs'])
    spline_mod_vars = list(hf['spline_mod_vars'])
    value_mod_specs = list(hf['value_mod_specs'])
    value_mod_vars = list(hf['value_mod_vars'])
    misc_mod_specs = list(hf['misc_mod_specs'])
    misc_mod_vars = list(hf['misc_mod_vars'])
    
    spline_mod_specs = [get_model_from_string(x) for x in spline_mod_specs]
    value_mod_specs = [get_model_from_string(x) for x in value_mod_specs]
    misc_mod_specs = [x.decode('UTF-8') for x in misc_mod_specs]
    
    spline_mod_vars = [np.array(x) for x in spline_mod_vars]
    value_mod_vars = [np.array(x) for x in value_mod_vars]
    misc_mod_vars = [np.array(x) for x in misc_mod_vars]
    
    specs = [spline_mod_specs, value_mod_specs, misc_mod_specs]
    variables = [spline_mod_vars, value_mod_vars, misc_mod_vars]
    for(spec_lst, var_lst) in zip(specs, variables):
        try:
            assert(len(spec_lst) == len(var_lst))
        except:
            raise ValueError("Spec and variable arrays not of same length")
        for i in range(len(spec_lst)):
            spec = spec_lst[i]
            mod_vars = var_lst[i]
            model_variables_np[spec] = mod_vars
    
    return model_variables_np

"""
Methods for handling feeds

Not going to do this class-wise because object oriented can be annoying
"""
def unpack_save_feed_h5(feed, hf):
    '''
    Here, the feed is the current feed to be saved and the hf is a created
    file pointer to an open and waiting h5py file.
    
    This method will make heavy use of glabels to access the correct molecules
    '''
    # First, get all the basis_sizes
    all_bsizes = list(feed['glabels'].keys())
    
    # It will be faster to go for each basis_size rather than each molecule given the
    # structure of the dictionaries
    
    for bsize in all_bsizes:
        curr_molec_labels = feed['glabels'][bsize]
        
        # Inner for loop to take care of each molecule
        for i in range(len(curr_molec_labels)):
            curr_label = curr_molec_labels[i] 
            curr_name = feed['names'][bsize][i]
            curr_iconfig = feed['iconfigs'][bsize][i]
            
            # Check if the molecule is already in the dataset. The molec_group
            # is what we're using from this point on
            molec_group = None
            if curr_name in hf.keys():
                molec_group = hf[curr_name]
            else:
                molec_group = hf.create_group(curr_name)
            
            # Create a group for the new iconfig (all iconfigs are unique). 
            # Save everyting to this iconfig group
            iconf_group = molec_group.create_group(str(curr_iconfig))
            
            # Unpack the geometry object
            curr_geom = feed['geoms'][curr_label]
            # No transpose applied to the extracted coordinate array so 
            # can be directly used to reconstitute geom object
            Zs, coords = curr_geom.z, curr_geom.rcart 
            
            iconf_group.create_dataset('Zs', data = Zs)
            iconf_group.create_dataset('Coords', data = coords)
            
            # Now need to save information for all the fields mentioned
            # Notes.txt, lots of extraction from the dictionary
            curr_Erep = feed['Erep'][bsize][i]
            curr_rho = feed['rho'][bsize][i]
            curr_S = feed['S'][bsize][i]
            curr_phis = feed['phiS'][bsize][i]
            curr_Sevals = feed['Sevals'][bsize][i]
            curr_G = feed['G'][bsize][i]
            curr_dQ = feed['dQ'][bsize][i]
            curr_Eelec = feed['Eelec'][bsize][i]
            curr_eorb = feed['eorb'][bsize][i]
            #No glabels
            #No gather_for_rep, depends on batch conformation
            #No segsum_for_rep, can get later out of batch
            curr_occ_rho_mask = feed['occ_rho_mask'][bsize][i]
            curr_occ_eorb_mask = feed['occ_eorb_mask'][bsize][i]
            curr_qneutral = feed['qneutral'][bsize][i]
            #No atom_ids
            #No norbs_atom, easy to reget
            curr_zcount = feed['zcounts'][bsize][i]
            curr_Etot = feed['Etot'][bsize][i]
            
            #Now save all this information to the created iconf group
            iconf_group.create_dataset('Erep', data = curr_Erep)
            iconf_group.create_dataset('rho', data = curr_rho)
            iconf_group.create_dataset('S', data = curr_S)
            iconf_group.create_dataset('phiS', data = curr_phis)
            iconf_group.create_dataset('Sevals', data = curr_Sevals)
            iconf_group.create_dataset('G', data = curr_G)
            iconf_group.create_dataset('dQ', data = curr_dQ)
            iconf_group.create_dataset('Eelec', data = curr_Eelec)
            iconf_group.create_dataset('eorb', data = curr_eorb)
            iconf_group.create_dataset('occ_rho_mask', data = curr_occ_rho_mask)
            iconf_group.create_dataset('occ_eorb_mask', data = curr_occ_eorb_mask)
            iconf_group.create_dataset('qneutral', data = curr_qneutral)
            iconf_group.create_dataset('zcounts', data = curr_zcount)
            iconf_group.create_dataset('Etot', data = curr_Etot)
    
    # Will not save the things related to the models because 
    

def save_all_feeds_h5(feeds, filename):
    '''
    Master method for saving a list of feeds into the h5 format
    '''
    hf = h5py.File(filename, 'w')
    for feed in feeds:
        unpack_save_feed_h5(feed, hf)
    print("feeds saved successfully")


"""
Extract and reconstitute the feed with the information provided
"""
def extract_feeds_h5(filename):
    '''
    Pulls the saved information for each molecule from the h5 file and
    saves the info as a dictionary.
    
    With the saved information, should be able to side-step the SCF cycle in 
    the initial precompute cycle
    
    This massive dictionary will have the exact same structure as the h5 file;
    a separate method will be in charge of taking this dictionary and 
    reconstituting the original feed with the SCF + other information 
    that's stored
    
    Accessing information requires an additional level of indexing by the empty 
    shape, i.e. dataset[()]. 
    '''
    master_molec_dict = dict()
    hf = h5py.File(filename, 'r')
    for molec in hf.keys():
        master_molec_dict[molec] = dict()
        
        #Deconstruct by the configuration numnbers
        for config_num in hf[molec].keys():
            master_molec_dict[molec][int(config_num)] = dict()
            
            for category in hf[molec][config_num].keys():
                master_molec_dict[molec][int(config_num)][category] = \
                    hf[molec][config_num][category]
    
    return master_molec_dict

def reconstitute_molecs_from_h5 (master_dict):
    '''
    Generates the list of dictionaries that is commonly used in
    dftb_layer_splines_3.py. 
    
    The keys for each molecule as used in dftb_layer_splines_3.py are as follows:
        name : name of molecule
        iconfig: configuration number
        atomic_numbers : Zs
        coordinates : Natom x 3 (will have to do a transpose from h5)
        targets : Etot : total energy 
    
    Current dictionary configuration, likely to change in the future
    
    It is important to keep track of the initial dictionary generated from
    the h5 file because that is going to be used to supplement any necessary SCF 
    information
    '''
    molec_list = list()
    for molec in master_dict:
        for config in master_dict[molec]:
            curr_molec_dict = dict()
            curr_molec_dict['name'] = molec
            curr_molec_dict['iconfig'] = config
            curr_molec_dict['coordinates'] = master_dict[molec][config]['Coords'][()].T
            curr_molec_dict['atomic_numbers'] = master_dict[molec][config]['Zs'][()]
            curr_molec_dict['targets'] = dict()
            curr_molec_dict['targets']['Etot'] = master_dict[molec][config]['Etot'][()]
            molec_list.append(curr_molec_dict)
    return molec_list
    

#%% Testing
if __name__ == "__main__":
    filename = 'testfeeds.h5'
    resulting_dict = extract_feeds_h5(filename)
    molec_lst = reconstitute_molecs_from_h5(resulting_dict)