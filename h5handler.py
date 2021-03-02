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
    1) Work on interface functions for loading a single graph at a time (X)
    2) Integration test the single graph loading mechanism
       
With this method of working with h5 files, the only certain pieces of information
still need to be computed in the pre-computation stage. 

Pleast note that saving to the h5 files should happen before any type conversions to
torch tensors. h5 works best with numpy arrays and should be treated as such
'''
import numpy as np
import h5py
from batch import Model, RawData
import collections
from geometry import Geometry
import pickle
from typing import Union, List, Optional, Dict, Any, Literal
import sys
Array = np.ndarray

#%% Model variables h5

def get_model_from_string(model_spec : str) -> Model:
    r"""Converts model_spec from string to named tuple object
    
    Arguments:
        model_spec (str): The representation of the model spec in string form
    
    Returns:
        model (Model): The named tuple representation of the model_spec, non-string 
            form
    
    Notes: This is a necessity because of the level of deconstruction data has 
        to go through to get saved in h5. Every key must be a binary string of some form,
        so decoding by UTF-8 is a necessary step.
    """
    return eval(model_spec.decode('UTF-8'))
    
class model_variable_h5handler:
    '''
    Class to just group methods together to save on import hassle. All of these
    are going to be static methods.
    '''
    @staticmethod
    def save_model_variables_h5(model_variables: Dict, filename: str) -> None:
        r"""Saves the model variables to an h5 file after training
        
        Arguments:
            model_variables (Dict): A dictionary contianing all the model variables,
                mapped by the model_spec tuple
            filename (str): The filename for the h5 file to write to
        
        Returns:
            None
        
        Notes: The value models and spline models have their variables saved separately. 
            Do note that for joined splines, only the variable coefficients will be saved. 
            This will have to be changed later if we intend on saving the variables. Alternatively, 
            saving the variables in a pickle file might be better since there aren't that many 
            weights. For N many models, we have at most num_knots * N weights so save.
        """
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
    def load_model_variables_h5(filename: str) -> Dict[str, Array]:
        r"""Loads the model variables from the model_variables dictionary saved in h5 
        
        Arguments:
            filename (str): The name of the h5 file from which to read model variables
        
        Returns:
            model_variables_np (Dict[str, Array]): A dictionary of the model variables as numpy arrays
        
        Notes: If you intend to use the loaded variables for training, please
            make sure to run a recursive type conversion to get the variables into tensors.
            Also, saving and loading variable names only works for non-joined splines
            right now, will fix that later if we end up using this approach.
        """
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
    def unpack_save_molec_feed_h5(feed: Dict, hf: h5py.File) -> None:
        r"""Saves a molecule feed to the given file pointer
        
        Arguments:
            feed (Dict): Current feed dictionary to be saved
            hf (h5py.File): Pointer to h5 file to save data to
        
        Returns:
            None
        
        Notes: None
        """
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
                #Try to get the dipole information saved in there too
                try:
                    curr_dipole_mats = feed['dipole_mat'][bsize][i]
                    curr_dipoles = feed['dipoles'][bsize][i]
                    curr_charges = feed['charges'][bsize][i]
                    iconf_group.create_dataset('dipole_mat', data = curr_dipole_mats)
                    iconf_group.create_dataset('dipoles', data = curr_dipoles)
                    iconf_group.create_dataset('charges', data = curr_charges)
                except:
                    print("Charge/dipole info was not found")
                try:
                    curr_G = feed['G'][bsize][i]
                    iconf_group.create_dataset('G', data = curr_G)
                except:
                    print("G info was not found")
                try:
                    curr_S = feed['S'][bsize][i]
                    curr_phis = feed['phiS'][bsize][i]
                    curr_Sevals = feed['Sevals'][bsize][i]
                    iconf_group.create_dataset('S', data = curr_S)
                    iconf_group.create_dataset('phiS', data = curr_phis)
                    iconf_group.create_dataset('Sevals', data = curr_Sevals)
                except:
                    print("S, phiS, Seval info was not found")
                    
                #Now save all this information to the created iconf group
                iconf_group.create_dataset('Erep', data = curr_Erep)
                iconf_group.create_dataset('rho', data = curr_rho)
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
    def save_all_molec_feeds_h5(feeds: List[Dict], filename: str) -> None:
        r"""Master method for saving a list of feeds into h5 format
        
        Arguments:
            feeds (List[Dict]): List of feed dictionaries to save
            filename (str): Name of h5 file to save to
        
        Returns:
            None
        
        Notes: None
        """
        hf = h5py.File(filename, 'w')
        try:
            for feed in feeds:
                per_molec_h5handler.unpack_save_molec_feed_h5(feed, hf)
            print("feeds saved successfully per molecule")
            hf.flush()
            hf.close()
        except Exception as e:
            print("something went wrong with saving feeds per molecule")
            print(e)
    
    """
    Extract and reconstitute the feed with the information provided on a per-molecule basis
    """
    @staticmethod
    def extract_molec_feeds_h5(filename: str) -> Dict:
        r"""Pulls saved information for each molecule from the h5 file into a dictionary
        
        Arguments:
            filename (str): h5 file to read from
        
        Returns:
            master_molec_dict (Dict): Dictionary where information about each molecule
                is indexed first by the molecule name and then the configuration number.
                To access information for the 5th conformation of C1H4, you would do
                master_molec_dict['C1H4'][5][()]
        
        Notes: Accessing information from the h5 data requires using the [()] as the 
            last indexing argument. This is important because without this, closing 
            the h5 file means shutting off access to the data.
        """
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
    def reconstitute_molecs_from_h5 (master_dict: Dict, targets: List[str]) -> List[Dict]:
        r"""Recreates a list of molecule dictionaries from the data in the h5 file
        
        Arguments:
            master_dict (Dict): Dictionary of molecular information generated from
                extract_molec_feeds_h5
            targets (List[str]): List of strings representing the name of the targets
                of interest
        
        Returns:
            molec_list (List[Dict]): A list of molecule dictionaries for each molecule contained within
                master_dict
            
        Notes: The keys for each molecule as used are as follows:
            name (str): Name of the molecule
            iconfig (int): Configuration number of the molecule
            atomic_numbers (Array[int]): Zs
            coordinates (Array): Natom x 3 of the cartesian coordinates
            targets (Dict): Dictionary mapping targets to values (e.g. Etot, dipole, etc.)
        """
        molec_list = list()
        for molec in master_dict:
            for config in master_dict[molec]:
                curr_molec_dict = dict()
                curr_molec_dict['name'] = molec
                curr_molec_dict['iconfig'] = config
                curr_molec_dict['coordinates'] = master_dict[molec][config]['Coords'].T
                curr_molec_dict['atomic_numbers'] = master_dict[molec][config]['Zs']
                curr_molec_dict['targets'] = dict()
                for target in targets:
                    try:
                        curr_molec_dict['targets'][target] = master_dict[molec][config][target]
                    except:
                        pass
                molec_list.append(curr_molec_dict)
        return molec_list
    
    @staticmethod
    def add_per_molec_info(feeds: List[Dict], master_dict: Dict, ragged_dipole_mat: bool, ignore_keys: List[str] = ['Coords', 'Zs']) -> None:
        r"""Adds the SCF information and everything else saved for molecules back into feeds
        
        Arguments:
            feeds (List[Dict]): List of feed dictionaries that need stuff added
                back in
            master_dict (Dict): Master dictionary containing information for all
                the molecules indexed by name and configuration number
            ragged_dipole_mat (bool): Flag indicating whether the dipole matrices
                are ragged or not
            ignore_keys (List[str]): List of keys to ignore. Defaults to ['Coords', 'Zs']
        
        Returns:
            None
            
        Raises:
            AssertionError: If the number of configuration numbers does not match the
                number of names
        
        Notes: The 'glabels', 'iconfigs', and 'names' keys must already exist in all the feeds that 
            need correcting, since we certainly need the names and configuration numbers to access
            the molecular information
        """
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
                        #Charges and dipole_mat are now ragged unfortunately
                        if key == 'dipole_mat':
                            if not ragged_dipole_mat:
                                feed[key][bsize] = np.array(feed[key][bsize])
                        elif key != 'charges':
                            feed[key][bsize] = np.array(feed[key][bsize])
        
    @staticmethod
    def create_molec_batches_from_feeds_h5(master_molec_dict: Dict, feeds: List[Dict], targets: List[str]) -> List[List[Dict]]:
        r"""Generates the batches of molecule dictionaries for each of the feeds
        
        Arguments:
            master_molec_dict (Dict): Dictionary contianing all the molecule infomration 
                indexed by name and configuration number
            feeds (List[Dict]): List of feed dictionaries that need their
                original molecule batches reconstructed
            
        Returns:
            master_batch_list (List[List[Dict]]): The list of lists of molecule
                dictionaries corresponding to the batches for every feed in feeds.
        
        Notes: Since the order of the molecules is preserved in creating each 
            feed dictionary, we can use glabels as a method of extracting the
            molecules and placing them in the correct places in each batch.
        """
        master_batch_list = list()
        for feed in feeds:
            num_geoms = len(feed['geoms'].keys())
            all_bsizes = feed['basis_sizes']
            #Placeholder for the batch
            batch = [None] * num_geoms
            for bsize in all_bsizes:
                curr_names = feed['names'][bsize]
                curr_iconfigs = feed['iconfigs'][bsize]
                curr_glabels = feed['glabels'][bsize]
                name_conf_zip = list(zip(curr_names, curr_iconfigs))
                assert(len(name_conf_zip) == len(curr_glabels))
                for i in range(len(name_conf_zip)):
                    name, config = name_conf_zip[i]
                    curr_molec = dict()
                    curr_molec['name'] = name
                    curr_molec['iconfig'] = config
                    curr_molec['coordinates'] = master_molec_dict[name][config]['Coords'].T
                    curr_molec['atomic_numbers'] = master_molec_dict[name][config]['Zs']
                    curr_molec['targets'] = dict()
                    for target in targets:
                        try:
                            curr_molec['targets'][target] = master_molec_dict[name][config][target]
                        except:
                            pass
                    batch[curr_glabels[i]] = curr_molec
            master_batch_list.append(batch)
        return master_batch_list

#%% Handling batch information
class per_batch_h5handler:
    '''
    Collection of methods for dealing with information in each feed that
    depends on the composition of the feeds
    
    '''
    @staticmethod
    def unpack_save_feed_batch_h5(feed: Dict, hf: h5py.File, feed_id: int) -> None:
        r"""Save information for each feed that depends on total composition
        
        Arguments:
            feed (Dict): Feed to save information for
            hf (h5py.File): Pointer to h5 file to save information to
            feed_id (int): Integer index for ith feed
        
        Returns:
            None
        
        Notes: Feeds are indexed by a generic number. They do not save
            any information that is organized by molecules, but only the 
            information that depends on the overall composition of the feed.
        """
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
    def save_multiple_batches_h5(batches: List[Dict], filename: str) -> None:
        r"""Wrapper method for saving all the information that depends on batch composition
        
        Arguments:
            batches (List[Dict]): List of feed dictionaries to save
            filename (str): Name of the h5 file to save the information to
        
        Returns:
            None
        
        Notes: Entry point for user.
        """
        save_file = h5py.File(filename, 'w')
        try:
            for i in range(len(batches)):
                per_batch_h5handler.unpack_save_feed_batch_h5(batches[i], save_file, i)
            print("batch info saved successfully")
        except Exception as e:
            print("something went wrong with saving batch information")
            print(e)
            
    @staticmethod
    def save_single_batch(batch: Dict, index: int, file: h5py.File) -> None:
        r"""Saves a single batch of a specified index
        
        Arguments:
            batch (Dict): The batch to save
            index (int): The identifying number of the current batch
            file (h5py.File): Pointer to open h5py file where the batch information
                is saved
        
        Returns:
            None
            
        Notes: The reason for this interface function is for memory purposes,
            whereby batches can be saved one at a time and then removed from memory.
            This prevents the need to have all the feed dictionaries open concurrently
            in memory.
        """
        try:
            per_batch_h5handler.unpack_save_feed_batch_h5(batch, file, index)
        except Exception as e:
            print(f"Something went wrong with saving batch info for batch {index}")
            print(e)
        

    @staticmethod
    def extract_batch_info(filename: str) -> List[Dict]:
        r"""Creates a list of feeds from the information saved in the given h5 file
        
        Arguments:
            filename (str): Name of h5 file to read from
        
        Returns:
            feeds_lst
            
        Raises:
            AssertionError: If the number of reconstructed feeds does not match the number of ids
        
        Notes: Simple keys are those which have their information organized by bsize 
            and can be easily retrieved by a direct index. Complex keys are those that
            will require more specialized handling to get out of the h5 files.
        """
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
        #Sort to make sure we are pulling things out in the correct order
        feed_ids = sorted([int(x) for x in feed_ids])
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
                                nonetest = eval(shp_rot_gthr[0].decode('UTF-8')) is None
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
    
    @staticmethod
    def extract_single_batch_info(file: h5py.File, batch_index: int) -> Dict:
        r"""Extracts a single graph based on the batch_index
        
        Arguments:
            file (h5py.File): Pointer to open h5py file to load a graph from
            batch_index (int): The batch to load from the h5py file
        
        Returns:
            batch (Dict): The reconstructed graph corresponding to the 
                batch_index within the h5py file
        
        Notes: The file pointer for h5py must remain open when reading from the 
            h5py file, otherwise the data becomes inaccessible. The pointer should be
            opened in read mode when trying to load data
        """
        simple_keys = ['glabels', 'gather_for_rep', 'segsum_for_rep', 
                       'atom_ids', 'norbs_atom', 'iconfigs', 
                       'basis_sizes', 'names']
        #These keys require more more processing in the extraction process
        complex_keys = ['mod_raw', 'rot_tensors', 'gather_for_rot', 'gather_for_oper',
                        'models', 'onames', 'names']
        
        stringed_index = str(batch_index)
        
        feed_file_dict = file[stringed_index]
        
        feed_dict = dict()
        
        all_keys = list(feed_file_dict.keys())
        for key in all_keys:
            #Debugging, can remove this later
            assert((key in simple_keys) or (key in complex_keys))
            #Simple keys either have a level of bsize index or direct kvp
            if key in simple_keys:
                try:
                    #All integers should be in string form due to encoding into h5
                    bsizes = list(feed_file_dict[key].keys())
                    feed_dict[key] = dict()
                    for bsize in bsizes:
                        full_data = feed_file_dict[key][bsize][()]
                        try:
                            full_data = [x.decode('UTF-8') for x in full_data]
                            feed_dict[key][int(bsize)] = full_data
                        except:
                            feed_dict[key][int(bsize)] = full_data
                except:
                    feed_dict[key] = feed_file_dict[key][()]
            #Complex keys have some trickery going on
            #For example, any information that is stored as a string has to be decoded
            elif key in complex_keys:
                #Handle the mod_raw case
                if key == 'mod_raw':
                    all_models = list(feed_file_dict[key].keys())
                    feed_dict[key] = dict()
                    for model in all_models:
                        model_mod_raw = feed_file_dict[key][model][()]
                        model_mod_raw = [get_model_from_string(x) for x in model_mod_raw]
                        model_key = eval(model) #Since the keys are valid python string, no need to decode byte string
                        feed_dict[key][model_key] = model_mod_raw
                
                #Handle the rot_tensors case
                elif key == 'rot_tensors':
                    all_shps = list(feed_file_dict[key].keys())
                    feed_dict[key] = dict()
                    for shp in all_shps:
                        shp_rot_tensors = feed_file_dict[key][shp][()]
                        try:
                            #Saved an array of "None" if there was no rotational tensor
                            # for that shape. Use try-except block to test that out
                            nonetest = eval(shp_rot_tensors[0].decode('UTF-8')) is None
                            feed_dict[key][eval(shp)] = None
                        except:
                            feed_dict[key][eval(shp)] = shp_rot_tensors
                
                elif key == 'gather_for_rot':
                    all_shps = list(feed_file_dict[key].keys())
                    feed_dict[key] = dict()
                    for shp in all_shps:
                        shp_rot_gthr = feed_file_dict[key][shp][()]
                        try:
                            nonetest = eval(shp_rot_gthr[0].decode('UTF-8')) is None
                            feed_dict[key][eval(shp)] = None
                        except:
                            feed_dict[key][eval(shp)] = shp_rot_gthr
                
                elif key == 'gather_for_oper':
                    all_opers = list(feed_file_dict[key].keys())
                    feed_dict[key] = dict()
                    for oper in all_opers:
                        feed_dict[key][oper] = dict()
                        all_bsizes = list(feed_file_dict[key][oper].keys())
                        for bsize in all_bsizes:
                            feed_dict[key][oper][int(bsize)] = feed_file_dict[key][oper][bsize][()]
                
                elif key == 'models':
                    all_models = feed_file_dict[key][()]
                    all_models = [get_model_from_string(x) for x in all_models]
                    feed_dict[key] = all_models
                
                elif key == 'onames':
                    all_onames = feed_file_dict[key][()]
                    all_onames = [x.decode('UTF-8') for x in all_onames]
                    feed_dict[key] = all_onames
        
        return feed_dict
    

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
    def create_all_feeds(batch_filename: str, molec_filename: str, ragged_dipole_mat: bool = True) -> List[Dict]:
        r"""Pulls all the molecules and feeds out of their respective files and comnbines to form feeds for DFTB layer
        
        Arguments:
            batch_filename (str): h5 file to read batch information (depends on composition)
            molec_filename (str): h5 file to read per-molecule information
            ragged_dipole_mat (bool): Indicates whether the dipole matrices are ragged (lists). Defaults to True
        
        Returns:
            extracted_feeds (List[Dict]): List of feed dictionaries correctly formatted
                with all the original data, ready to go for the DFTB layer
        
        Raises:
            AssertionError: If the glabels, names, and iconfigs are not the same length per basis size
        
        Notes: Keys such as 'glabels', 'names', and 'iconfigs' have to be added
            in manually first.
        """
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
        per_molec_h5handler.add_per_molec_info(extracted_feeds, master_molec_dict, ragged_dipole_mat)
        return extracted_feeds
    
    @staticmethod
    def create_single_feed(batch_file_ptr: h5py.File, master_molec_dict: Dict, batch_index: int, ragged_dipole_mat: bool = True) -> Dict:
        r"""Creates a single feed from the given batch file and the master molecule dictionary
        
        Arguments:
            batch_file_ptr (h5py.File): POinter to an opern h5py file containing the graph information
            master_molec_dict (Dict): The dictionary containing all the molecule information
            batch_index (int): The generic index of the batch that should be loaded
            ragged_dipole_mat (bool): Indication whether the dipoles are ragged. Defaults to True (ragged dipoles)
        
        Returns:
            feed (Dict): Completed feed dictionary
        """
        feed = per_batch_h5handler.extract_single_batch_info(batch_file_ptr, batch_index)
        
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
        single_feeds = [feed]
        #print(f"{sys.getsizeof(feed)}")
        per_molec_h5handler.add_per_molec_info(single_feeds, master_molec_dict, ragged_dipole_mat)
        
        return single_feeds[0]
        

#%% Testing utilities 
def compare_feeds(reference_file: str, reconstituted_feeds: List[Dict]) -> None:
    r"""Function for checkig that the reconstituted feeds matches the original data
    
    Arguments:
        reference_file (str): Name of the pickle file that stores the original data
        reconstituted_feeds (List[Dict]): Reconstructed feeds that need checking
    
    Returns:
        None
    
    Raises:
        AssertionError: If any of the tests fail
    
    Notes: For ragged data like dipole matrices and charges, that information is 
        handled with a small for loop.
    """
    reference_file_feeds = pickle.load(open(reference_file, 'rb'))
    assert(len(reconstituted_feeds) == len(reference_file_feeds))
    for i in range(len(reference_file_feeds)):
        curr_ref_fd = reference_file_feeds[i]
        feedi = reconstituted_feeds[i]
        #Assert same basis sizes
        assert( set(curr_ref_fd['basis_sizes']).difference(set(feedi['basis_sizes'])) == set() )
        assert( set(feedi['basis_sizes']).difference(set(curr_ref_fd['basis_sizes'])) == set() )
        
        assert( curr_ref_fd['onames'] == feedi['onames'] )
        
        assert( set(curr_ref_fd['models']).difference(set(feedi['models'])) == set() )
        assert( set(feedi['models']).difference(set(curr_ref_fd['models'])) == set() )
        
        #Need to check mod_raw
        for mod_spec in curr_ref_fd['mod_raw']:
            assert( curr_ref_fd['mod_raw'][mod_spec] == feedi['mod_raw'][mod_spec] )
        
        assert( set(curr_ref_fd['gather_for_rot'].keys()).difference(set(feedi['gather_for_rot'].keys())) == set() )
        assert( set(feedi['gather_for_rot'].keys()).difference(set(curr_ref_fd['gather_for_rot'].keys())) == set() )
        
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
            
            if 'S' in curr_ref_fd:
                assert( np.allclose (curr_ref_fd['S'][bsize], feedi['S'][bsize]) )
            
            if 'G' in curr_ref_fd:
                assert( np.allclose (curr_ref_fd['G'][bsize], feedi['G'][bsize]) )
            
            assert( np.allclose (curr_ref_fd['Etot'][bsize], feedi['Etot'][bsize]) )      

            assert( np.allclose (curr_ref_fd['Eelec'][bsize], feedi['Eelec'][bsize]) )

            assert( np.allclose (curr_ref_fd['Erep'][bsize], feedi['Erep'][bsize]) ) 
            
            if 'Sevals' in curr_ref_fd:
                assert( np.allclose (curr_ref_fd['Sevals'][bsize], feedi['Sevals'][bsize]) ) 
            
            assert( np.allclose (curr_ref_fd['dQ'][bsize], feedi['dQ'][bsize]) ) 

            assert( np.allclose (curr_ref_fd['eorb'][bsize], feedi['eorb'][bsize]) ) 
            
            for i in range(len(curr_ref_fd['dipole_mat'][bsize])):
                assert( np.allclose (curr_ref_fd['dipole_mat'][bsize][i], feedi['dipole_mat'][bsize][i]))
            
            assert( np.allclose (curr_ref_fd['dipoles'][bsize], feedi['dipoles'][bsize]))
            
            # assert( np.allclose (curr_ref_fd['charges'][bsize], feedi['charges'][bsize]))
            for i in range(len(curr_ref_fd['charges'][bsize])):
                assert(np.allclose(curr_ref_fd['charges'][bsize][i], feedi['charges'][bsize][i]))
            
            assert( np.allclose (curr_ref_fd['occ_rho_mask'][bsize], feedi['occ_rho_mask'][bsize]) )
            
            assert( np.allclose (curr_ref_fd['occ_eorb_mask'][bsize], feedi['occ_eorb_mask'][bsize]) )
            
            if 'phiS' in curr_ref_fd:
                assert( np.allclose (curr_ref_fd['phiS'][bsize], feedi['phiS'][bsize]) )
            
            assert( np.allclose (curr_ref_fd['rho'][bsize], feedi['rho'][bsize]) ) 
            
            assert( np.allclose (curr_ref_fd['zcounts'][bsize], feedi['zcounts'][bsize]) ) 
            
            assert( np.allclose (curr_ref_fd['qneutral'][bsize], feedi['qneutral'][bsize]) )
            
            for oper in curr_ref_fd['gather_for_oper'].keys():
                assert(np.allclose(curr_ref_fd['gather_for_oper'][oper][bsize], feedi['gather_for_oper'][oper][bsize] ))
            
            for oper in curr_ref_fd['onames']:
                if (oper != 'H') and (oper in curr_ref_fd):
                    assert(np.allclose(curr_ref_fd[oper][bsize], feedi[oper][bsize]))
                    
    print("Tests passed!")

            

#%% Testing
if __name__ == "__main__":
    filename = 'testfeeds.h5'
    resulting_dict = per_molec_h5handler.extract_molec_feeds_h5(filename)
    molec_lst = per_molec_h5handler.reconstitute_molecs_from_h5(resulting_dict)
    feeds_lst = per_batch_h5handler.extract_batch_info('graph_save_tst.h5')