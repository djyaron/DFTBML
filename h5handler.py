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
from batch import Model, RawData
import collections
from geometry import Geometry
import pickle

#%% Model variables h5

def get_model_from_string(model_spec : str):
    '''
    Converts the model_spec named tuple from a binary string representation to the
    named tuple representation. Will decode by UTF-8 first.
    '''
    return eval(model_spec.decode('UTF-8'))
    
class model_variable_h5handler:
    '''
    Class to just group methods together to save on import hassle. All of these
    are going to be static methods.
    '''
    @staticmethod
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
        
    @staticmethod
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
        
        hf.close()
        return model_variables_np

#%% Per molecule information for feeds
class per_molec_h5handler:
    """
    Methods for handling feeds
    
    This is saving information on a per-molecule basis
    """
    @staticmethod
    def unpack_save_molec_feed_h5(feed, hf):
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
        
    @staticmethod
    def save_all_molec_feeds_h5(feeds, filename):
        '''
        Master method for saving a list of feeds into the h5 format
        '''
        hf = h5py.File(filename, 'w')
        try:
            for feed in feeds:
                per_molec_h5handler.unpack_save_molec_feed_h5(feed, hf)
            print("feeds saved successfully per molecule")
            hf.flush()
            hf.close()
        except:
            print("something went wrong with saving feeds per molecule")
    
    """
    Extract and reconstitute the feed with the information provided on a per-molecule basis
    """
    @staticmethod
    def extract_molec_feeds_h5(filename):
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
                        hf[molec][config_num][category][()]
        
        hf.close()
        
        return master_molec_dict
    
    @staticmethod
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
                curr_molec_dict['coordinates'] = master_dict[molec][config]['Coords'].T
                curr_molec_dict['atomic_numbers'] = master_dict[molec][config]['Zs']
                curr_molec_dict['targets'] = dict()
                curr_molec_dict['targets']['Etot'] = master_dict[molec][config]['Etot']
                molec_list.append(curr_molec_dict)
        return molec_list
    
    @staticmethod
    def add_per_molec_info(feeds, master_dict, ignore_keys = ['Coords', 'Zs']):
        '''
        This adds the SCF information and everything else saved to the h5 file
        back into the feed using the glabels, names, and iconfigs as guidance.
        
        Also, need to specify which keys to ignore from the master_dict
        
        The master_dict comes from the method reading from the h5 file
        
        NOTE: the 'glables' key has to exist in all the feeds already!
        '''
        first_mol = list(master_dict.keys())[0]
        first_mol_first_conf = list(master_dict[first_mol].keys())[0]
        
        # These are all the keys to add to each feed
        # Coincidentally, they are also the fields that are indexed by bsize
        
        all_keys_to_add = list(master_dict[first_mol][first_mol_first_conf].keys())
        for feed in feeds:
            all_bsizes = list(feed['glabels'].keys())
            for key in all_keys_to_add:
                if (key not in feed) and (key not in ignore_keys):
                    feed[key] = dict()
                    for bsize in all_bsizes:
                        # number of values = number of molecules in that bsize
                        feed[key][bsize] = list()
                        
                        #Here, the configuration numbers and names match
                        current_iconfs = feed['iconfigs'][bsize]
                        current_names = feed['names'][bsize]
                        assert (len(current_iconfs) == len(current_names))
                        
                        for i in range(len(current_iconfs)):
                            name, conf = current_names[i], current_iconfs[i]
                            feed[key][bsize].append(master_dict[name][conf][key][()])
                        
                        feed[key][bsize] = np.array(feed[key][bsize])

#%% Handling batch information
class per_batch_h5handler:
    '''
    Collection of methods for dealing with information in each feed that
    depends on the composition of the feeds
    
    '''
    @staticmethod
    def unpack_save_feed_batch_h5(feed, hf, feed_id):
        '''
        This method is used to save information for each feed that depends on the 
        composition. The feed_id is used as the indexing method, and must be incremented
        by any wrapper function that calls this one repeatedly. The feed_id is an integer, but
        be sure to save 
        
        For a complete list of the information saved in these files, refer to 
        h5Notes.txt
        '''
        #Create a group for the current batch
        curr_batch_grp = hf.create_group(str(feed_id))
        
        #Ignore the geoms for, they can be reconstituted
        #Save the models
        curr_models = feed['models']
        curr_models = [str(x).encode('ascii', 'ignore') for  x in curr_models]
        curr_batch_grp.create_dataset ('models', data = curr_models)
        #Save the mod_raw information
        mod_raw_grp = curr_batch_grp.create_group('mod_raw')
        curr_mod_raw_keys = feed['mod_raw'].keys()
        for model in curr_mod_raw_keys:
            actual_mod_raw = feed['mod_raw'][model]
            actual_mod_raw = [str(x).encode('ascii', 'ignore') for x in actual_mod_raw]
            mod_raw_grp.create_dataset(str(model), data = actual_mod_raw)
        #Save the rot_tensors
        rot_tensor_grp = curr_batch_grp.create_group('rot_tensors')
        curr_rot_tensor_keys = feed['rot_tensors'].keys()
        for shp in curr_rot_tensor_keys:
            if feed['rot_tensors'][shp] is None:
                rot_tensor_grp.create_dataset(str(shp), data = ["None".encode('ascii', 'ignore')])
            else:
                rot_tensor_grp.create_dataset(str(shp), data = feed['rot_tensors'][shp])
        #Save the onames
        onames = feed['onames']
        onames = [x.encode('ascii', 'ignore') for x in onames]
        curr_batch_grp.create_dataset('onames', data = onames)
        #Save the basis_sizes
        curr_batch_grp.create_dataset('basis_sizes', data = feed['basis_sizes'])
        #Save the gather_for_rot
        gthr_for_rot_grp = curr_batch_grp.create_group('gather_for_rot')
        for shp in feed['gather_for_rot']:
            if feed['gather_for_rot'][shp] is None:
                gthr_for_rot_grp.create_dataset(str(shp), data = ["None".encode('ascii', 'ignore')])
            else:
                gthr_for_rot_grp.create_dataset(str(shp), data = feed['gather_for_rot'][shp])
        #Save the gather_for_oper
        gthr_for_oper_grp = curr_batch_grp.create_group('gather_for_oper')
        for oper in feed['gather_for_oper']:
            oper_grp = gthr_for_oper_grp.create_group(oper)
            for bsize in feed['gather_for_oper'][oper]:
                oper_grp.create_dataset(str(bsize), data = feed['gather_for_oper'][oper][bsize])
        
        #Following fields are all saved in one for loop because they have
        # very similar structure
        glabel_grp = curr_batch_grp.create_group('glabels')
        gthr_for_rep_grp = curr_batch_grp.create_group('gather_for_rep')
        sgsum_for_rep_grp = curr_batch_grp.create_group('segsum_for_rep')
        at_id_grp = curr_batch_grp.create_group('atom_ids')
        norbs_grp = curr_batch_grp.create_group('norbs_atom')
        name_grp = curr_batch_grp.create_group('names')
        iconfig_grp = curr_batch_grp.create_group('iconfigs')
        for bsize in feed['glabels']:
            glabel_grp.create_dataset(str(bsize), data = feed['glabels'][bsize])
            gthr_for_rep_grp.create_dataset(str(bsize), data = feed['gather_for_rep'][bsize])
            sgsum_for_rep_grp.create_dataset(str(bsize), data = feed['segsum_for_rep'][bsize])
            at_id_grp.create_dataset(str(bsize), data = feed['atom_ids'][bsize])
            norbs_grp.create_dataset(str(bsize), data = feed['norbs_atom'][bsize])
            name_grp.create_dataset(str(bsize), data = [x.encode('ascii', 'ignore') for x in feed['names'][bsize]])
            iconfig_grp.create_dataset(str(bsize), data = feed['iconfigs'][bsize])
    
    @staticmethod
    def save_multiple_batches_h5(batches, filename):
        '''
        Wrapper method for saving all the information
        '''
        save_file = h5py.File(filename, 'w')
        try:
            for i in range(len(batches)):
                per_batch_h5handler.unpack_save_feed_batch_h5(batches[i], save_file, i)
            print("batch info saved successfully")
        except:
            print("something went wrong with saving batch information")

    @staticmethod
    def extract_batch_info(filename):
        '''
        Creates a list of feeds from the informaiton saved in the given file
        represented by filename
        
        Will extract the feeds in the filename into a feeds list, which is the
        same format as when the data was first encoded in h5
        '''
        simple_keys = ['glabels', 'gather_for_rep', 'segsum_for_rep', 
                       'atom_ids', 'norbs_atom', 'iconfigs', 
                       'basis_sizes', 'names']
        #These keys require more more processing in the extraction process
        complex_keys = ['mod_raw', 'rot_tensors', 'gather_for_rot', 'gather_for_oper',
                        'models', 'onames', 'names']
        feeds_lst = list()
        feed_file = h5py.File(filename, 'r')
        #The top level keys should just be the numbers of each batch
        feed_ids = list(feed_file.keys())
        feed_ids = [int(x) for x in feed_ids]
        feed_ids.sort()
        for feed in feed_ids:
            #Construct each feed
            feed_dict = dict()
            all_keys = list(feed_file[str(feed)].keys())
            for key in all_keys:
                #Debugging, can remove this later
                assert((key in simple_keys) or (key in complex_keys))
                #Simple keys either have a level of bsize index or direct kvp
                if key in simple_keys:
                    try:
                        #All integers should be in string form due to encoding into h5
                        bsizes = list(feed_file[str(feed)][key].keys())
                        feed_dict[key] = dict()
                        for bsize in bsizes:
                            full_data = feed_file[str(feed)][key][bsize][()]
                            try:
                                full_data = [x.decode('UTF-8') for x in full_data]
                                feed_dict[key][int(bsize)] = full_data
                            except:
                                feed_dict[key][int(bsize)] = full_data
                    except:
                        feed_dict[key] = feed_file[str(feed)][key][()]
                #Complex keys have some trickery going on
                #For example, any information that is stored as a string has to be decoded
                elif key in complex_keys:
                    #Handle the mod_raw case
                    if key == 'mod_raw':
                        all_models = list(feed_file[str(feed)][key].keys())
                        feed_dict[key] = dict()
                        for model in all_models:
                            model_mod_raw = feed_file[str(feed)][key][model][()]
                            model_mod_raw = [get_model_from_string(x) for x in model_mod_raw]
                            model_key = eval(model) #Since the keys are valid python string, no need to decode byte string
                            feed_dict[key][model_key] = model_mod_raw
                    
                    #Handle the rot_tensors case
                    elif key == 'rot_tensors':
                        all_shps = list(feed_file[str(feed)][key].keys())
                        feed_dict[key] = dict()
                        for shp in all_shps:
                            shp_rot_tensors = feed_file[str(feed)][key][shp][()]
                            try:
                                #Saved an array of "None" if there was no rotational tensor
                                # for that shape. Use try-except block to test that out
                                nonetest = eval(shp_rot_tensors[0].decode('UTF-8')) is None
                                feed_dict[key][eval(shp)] = None
                            except:
                                feed_dict[key][eval(shp)] = shp_rot_tensors
                    
                    elif key == 'gather_for_rot':
                        all_shps = list(feed_file[str(feed)][key].keys())
                        feed_dict[key] = dict()
                        for shp in all_shps:
                            shp_rot_gthr = feed_file[str(feed)][key][shp][()]
                            try:
                                nonetes = eval(shp_rot_gthr[0].decode('UTF-8')) is None
                                feed_dict[key][eval(shp)] = None
                            except:
                                feed_dict[key][eval(shp)] = shp_rot_gthr
                    
                    elif key == 'gather_for_oper':
                        all_opers = list(feed_file[str(feed)][key].keys())
                        feed_dict[key] = dict()
                        for oper in all_opers:
                            feed_dict[key][oper] = dict()
                            all_bsizes = list(feed_file[str(feed)][key][oper].keys())
                            for bsize in all_bsizes:
                                feed_dict[key][oper][int(bsize)] = feed_file[str(feed)][key][oper][bsize][()]
                    
                    elif key == 'models':
                        all_models = feed_file[str(feed)][key][()]
                        all_models = [get_model_from_string(x) for x in all_models]
                        feed_dict[key] = all_models
                    
                    elif key == 'onames':
                        all_onames = feed_file[str(feed)][key][()]
                        all_onames = [x.decode('UTF-8') for x in all_onames]
                        feed_dict[key] = all_onames
                        
            #Add the completed feed to the feeds_lst
            feeds_lst.append(feed_dict)
        assert (len(feeds_lst) == len(feed_ids))
        return feeds_lst
    

#%% Combinator class
class total_feed_combinator:
    '''
    This class contains methods to join together per-molec information
    with batch information to formulate the original feeds that go into the 
    dftblayer. Will draw upon the methods of other classes defined so far in the code
    
    As an aside, we probably don't need the geometries of the molecules in there anymore,
    but will implement adding them in just in case
    '''
    
    @staticmethod
    def create_all_feeds(batch_filename, molec_filename):
        '''
        batch_filename: h5 file containing batch information
        molec_filename: h5 file containing molecule information
        
        Pulls all the molecules and feeds out of their respective files and 
        then assembles them into the list of complete feeds that is commonly used
        '''
        extracted_feeds = per_batch_h5handler.extract_batch_info(batch_filename)
        master_molec_dict = per_molec_h5handler.extract_molec_feeds_h5(molec_filename)
        
        #One of the things that needs to be manually added back in is the geoms category, which
        # has to be recreated
        for feed in extracted_feeds:
            feed['geoms'] = dict()
            all_bsizes = feed['glabels'].keys()
            for bsize in all_bsizes:
                curr_glabels = feed['glabels'][bsize]
                curr_names = feed['names'][bsize]
                curr_iconfigs = feed['iconfigs'][bsize]
                assert(len(curr_glabels) == len(curr_names) == len(curr_iconfigs))
                for i in range(len(curr_glabels)):
                    coordinates = master_molec_dict[curr_names[i]][curr_iconfigs[i]]['Coords'][()]
                    Zs = master_molec_dict[curr_names[i]][curr_iconfigs[i]]['Zs'][()]
                    feed['geoms'][curr_glabels[i]] = Geometry(Zs, coordinates) #No need to transpose
        
        #Now just use the master correction function in the per_molec_h5handler
        per_molec_h5handler.add_per_molec_info(extracted_feeds, master_molec_dict)
        return extracted_feeds

#%% Testing utilities 
def compare_feeds(reference_file, reconstituted_feeds):
    '''
    reference_file: pickle file name to load in the reference data
    reconstituted_feeds: list of reconstituted feeds
    
    Compares the values between each key between each feed of the reference_file
    feeds and the reconstituted feeds
    
    The reconstituted feeds and reference file feeds should have the same order in terms of their feeds
    '''
    reference_file_feeds = pickle.load(open(reference_file, 'rb'))
    assert(len(reconstituted_feeds) == len(reference_file_feeds))
    for i in range(len(reference_file_feeds)):
        curr_ref_fd = reference_file_feeds[i]
        feedi = reconstituted_feeds[i]
        #Assert same basis sizes
        assert( set(curr_ref_fd['basis_sizes']).difference(set(feedi['basis_sizes'])) == set() )
        assert( curr_ref_fd['onames'] == feedi['onames'] )
        
        assert( set(curr_ref_fd['models']).difference(set(feedi['models'])) == set() )
        
        #Need to check mod_raw
        for mod_spec in curr_ref_fd['mod_raw']:
            assert( curr_ref_fd['mod_raw'][mod_spec] == feedi['mod_raw'][mod_spec] )
        
        assert( set(curr_ref_fd['gather_for_rot'].keys()).difference(set(feedi['gather_for_rot'].keys())) == set() )
        
        for shp in curr_ref_fd['gather_for_rot']:
            assert( np.allclose (curr_ref_fd['gather_for_rot'][shp], feedi['gather_for_rot'][shp]) )
        
        for shp in curr_ref_fd['rot_tensors']:
            assert( 
                ((feedi['rot_tensors'][shp] is None) and (curr_ref_fd['rot_tensors'][shp] is None)) or\
                    np.allclose(curr_ref_fd['rot_tensors'][shp], feedi['rot_tensors'][shp])
                )
        #Assert the same for all things indexed by basis sizes
        for bsize in curr_ref_fd['glabels'].keys():
            assert( len(curr_ref_fd['glabels'][bsize]) == len(feedi['glabels'][bsize]) )
            assert( list(curr_ref_fd['glabels'][bsize]) == list(feedi['glabels'][bsize]) )
            
            assert( len(curr_ref_fd['names'][bsize]) == len(feedi['names'][bsize]) )
            assert( list(curr_ref_fd['names'][bsize]) == list(feedi['names'][bsize]) )
            
            assert( len(curr_ref_fd['iconfigs'][bsize]) == len(feedi['iconfigs'][bsize]) )
            assert( list(curr_ref_fd['iconfigs'][bsize]) == list(feedi['iconfigs'][bsize]) )
            
            assert( np.allclose(curr_ref_fd['gather_for_rep'][bsize], feedi['gather_for_rep'][bsize]))
            assert( np.allclose(curr_ref_fd['segsum_for_rep'][bsize], feedi['segsum_for_rep'][bsize]))
            
            assert( np.allclose(curr_ref_fd['norbs_atom'][bsize], feedi['norbs_atom'][bsize]) )
            
            assert( np.allclose(curr_ref_fd['atom_ids'][bsize], feedi['atom_ids'][bsize]) )
            
            assert( np.allclose (curr_ref_fd['S'][bsize], feedi['S'][bsize]) )

            assert( np.allclose (curr_ref_fd['G'][bsize], feedi['G'][bsize]) )
            
            assert( np.allclose (curr_ref_fd['Etot'][bsize], feedi['Etot'][bsize]) )      

            assert( np.allclose (curr_ref_fd['Eelec'][bsize], feedi['Eelec'][bsize]) )

            assert( np.allclose (curr_ref_fd['Erep'][bsize], feedi['Erep'][bsize]) ) 
            
            assert( np.allclose (curr_ref_fd['Sevals'][bsize], feedi['Sevals'][bsize]) ) 
            
            assert( np.allclose (curr_ref_fd['dQ'][bsize], feedi['dQ'][bsize]) ) 

            assert( np.allclose (curr_ref_fd['eorb'][bsize], feedi['eorb'][bsize]) ) 
            
            assert( np.allclose (curr_ref_fd['occ_rho_mask'][bsize], feedi['occ_rho_mask'][bsize]) )
            
            assert( np.allclose (curr_ref_fd['occ_eorb_mask'][bsize], feedi['occ_eorb_mask'][bsize]) )
            
            assert( np.allclose (curr_ref_fd['phiS'][bsize], feedi['phiS'][bsize]) )
            
            assert( np.allclose (curr_ref_fd['rho'][bsize], feedi['rho'][bsize]) ) 
            
            assert( np.allclose (curr_ref_fd['zcounts'][bsize], feedi['zcounts'][bsize]) ) 
            
            assert( np.allclose (curr_ref_fd['qneutral'][bsize], feedi['qneutral'][bsize]) )
            
            for oper in curr_ref_fd['gather_for_oper'].keys():
                assert(np.allclose(curr_ref_fd['gather_for_oper'][oper][bsize], feedi['gather_for_oper'][oper][bsize] ))
            
            for oper in curr_ref_fd['onames']:
                if oper != 'H':
                    assert(np.allclose(curr_ref_fd[oper][bsize], feedi[oper][bsize]))
                    
    print("Tests passed!")

            

#%% Testing
if __name__ == "__main__":
    filename = 'testfeeds.h5'
    resulting_dict = per_molec_h5handler.extract_molec_feeds_h5(filename)
    molec_lst = per_molec_h5handler.reconstitute_molecs_from_h5(resulting_dict)
    feeds_lst = per_batch_h5handler.extract_batch_info('graph_save_tst.h5')