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
        for feed in feeds:
            per_molec_h5handler.unpack_save_molec_feed_h5(feed, hf)
        print("feeds saved successfully")
    
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
                        hf[molec][config_num][category]
        
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
                curr_molec_dict['coordinates'] = master_dict[molec][config]['Coords'][()].T
                curr_molec_dict['atomic_numbers'] = master_dict[molec][config]['Zs'][()]
                curr_molec_dict['targets'] = dict()
                curr_molec_dict['targets']['Etot'] = master_dict[molec][config]['Etot'][()]
                molec_list.append(curr_molec_dict)
        return molec_list
    
    @staticmethod
    def add_per_molec_info(feeds, master_dict, ignore_keys = []):
        # TODO: FIX ME TO FIT!
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
        oname_grp = curr_batch_grp.create_group('onames')
        onames = feed['onames']
        oname_grp.create_dataset('onames', data = [x.encode('ascii', 'ignore') for x in onames])
        #Save the basis_sizes
        bsize_grp = curr_batch_grp.create_group('basis_sizes')
        bsize_grp.create_dataset('basis_sizes', data = feed['basis_sizes'])
        #Save the glabels
        glabel_grp = curr_batch_grp.create_group('glabels')
        for bsize in feed['glabels']:
            glabel_grp.create_dataset(str(bsize), data = feed['glabels'][bsize])
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
        #Save the gather_for_rep
        gthr_for_rep_grp = curr_batch_grp.create_group('gather_for_rep')
        for bsize in feed['gather_for_rep']:
            gthr_for_rep_grp.create_dataset(str(bsize), data = feed['gather_for_rep'][bsize])
        #Save the segsum_for_rep
        sgsum_for_rep_grp = curr_batch_grp.create_group('segsum_for_rep')
        for bsize in feed['segsum_for_rep']:
            sgsum_for_rep_grp.create_dataset(str(bsize), data = feed['segsum_for_rep'][bsize])
        #Save the atom_ids
        at_id_grp = curr_batch_grp.create_group('atom_ids')
        for bsize in feed['atom_ids']:
            at_id_grp.create_dataset(str(bsize), data = feed['atom_ids'][bsize])
        #Save the norbs_atom (AND WE'RE DONE!)
        norbs_grp = curr_batch_grp.create_group('norbs_atom')
        for bsize in feed['norbs_atom']:
            norbs_grp.create_dataset(str(bsize), data = feed['norbs_atom'][bsize])
        #Need to save names and iconfigs too
        name_grp = curr_batch_grp.create_group('names')
        iconfig_grp = curr_batch_grp.create_group('iconfigs')
        for bsize in feed['names']:
            name_grp.create_dataset(str(bsize), data = [x.encode('ascii', 'ignore') for x in feed['names'][bsize]])
            iconfig_grp.create_dataset(str(bsize), data = feed['iconfigs'][bsize])
    
    @staticmethod
    def extract_batch_info(filename):
        '''
        Creates a list of feeds from the informaiton saved in the given file
        represented by filename
        
        IMPLEMENT ME!!
        '''
        pass
    


#%% Testing
if __name__ == "__main__":
    filename = 'testfeeds.h5'
    resulting_dict = per_molec_h5handler.extract_molec_feeds_h5(filename)
    molec_lst = per_molec_h5handler.reconstitute_molecs_from_h5(resulting_dict)