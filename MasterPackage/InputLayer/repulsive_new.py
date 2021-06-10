# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 18:50:46 2021

@author: fhu14

TODO: Fix the link with DFTBrepulsive when everything's up
"""
#%% Imports, definitions
import torch
Tensor = torch.Tensor
from typing import List, Dict
import numpy as np
Array = np.ndarray
from .constants import Model
from Spline import SplineModel

#%% Code behind

class repulsive_energy:
    
    #TODO: REDESIGN SO THAT you just pass feeds into the model and it handles
    #   all the calculations internally
    
    def __init__(self, s, training_feeds: List[Dict], validation_feeds: List[Dict],
                 all_models: Dict, layer, dtype: torch.dtype, device: torch.device) -> None:
        r"""Initializes the repulsive_energy_2 model using the given training
            and validation feeds
        
        Arguments:
            s (Settings): Object containing all the hyperparameter settings
            training_feeds (List[Dict]): List of training feeds
            validation_feeds (List[Dict]): List of validation feeds
            all_models (Dict): Dictionary containing references to all the models
                used in training
            layer (DFTB_Layer): Instance of the DFTB_Layer object used for passing through
                the feeds.
            dtype (torch.dtype): The datatype for the generated energies to have.
            device (torch.device): The device to run the computations on (CPU vs GPU).
                If running on GPU, must be CUDA enabled GPU.
            
        Returns:
            None
        
        Notes: This method generates the coefficient vector from the training set and
            saves the loc, gammas, and config_tracker of both the training and validation
            sets. This serves to abstract away all the repulsive calculations from the
            higher level driver code.
        """
        repulsive_opts = self.obtain_repulsive_opts(s)
        self.repulsive_opts = repulsive_opts
        self.dtype = dtype
        self.device = device
        if len(training_feeds) > 0:
            total_dict_train, config_tracker_train = self.assemble_rep_input_all_feeds(training_feeds, all_models, layer)
            self.conversion_to_nparray(total_dict_train)
            self.c_sparse, self.loc_train, self.gammas_train = driver.train_repulsive_model(total_dict_train, repulsive_opts)
            self.config_tracker_train = config_tracker_train
        if len(validation_feeds) > 0:
            total_dict_valid, config_tracker_valid = self.assemble_rep_input_all_feeds(validation_feeds, all_models, layer)
            self.conversion_to_nparray(total_dict_valid)
            self.c_sparse_2, self.loc_valid, self.gammas_valid = driver.train_repulsive_model(total_dict_valid, repulsive_opts)
            self.config_tracker_valid = config_tracker_valid
            #Sanity check on coefficient vector
            assert(len(self.c_sparse) == len(self.c_sparse_2))
        #Set the spline models
        self.set_spline_models(all_models)
        self.cutoff_dictionary = s.cutoff_dictionary
        self.default_cutoff = s.joined_cutoff
    
    def generate_repulsive_energies(self, feed: Dict, flag: str) -> Dict:
        r"""Get values for the repulsive energies for the given feed, organized by basis size
        
        Arguments:
            feed (Dict): Dictionary that needs to have repulsive energies generated
                for the molecules
            flag (str): 'train' or 'valid', and it indicates which gamma, loc, config_tracker set to use
            
        Returns:
            repulsive_dict (Dict): Dictionary of repulsive energies for the configurations
                presented in the feed, organized by bsizes
        
        Notes: Predictions are generated for each specific configuration, and are 
            done by name and iconfig. The results are organized per basis size, so 
            things remain consistent with later parts of the workflow
        """
        gammas = self.gammas_train if flag == 'train' else self.gammas_valid
        config_tracker = self.config_tracker_train if flag == 'train' else self.config_tracker_valid
        all_bsizes = feed['basis_sizes']
        repulsive_dict = dict()
        for bsize in all_bsizes:
            curr_names = feed['names'][bsize]
            curr_iconfigs = feed['iconfigs'][bsize]
            assert(len(curr_names) == len(curr_iconfigs))
            #For efficiency, compute the dimension of the merged array first
            final_gammas_arr = np.zeros((len(curr_iconfigs), len(self.c_sparse)))
            #Assemble the reduced gammas
            for index, name in enumerate(curr_names):
                name, config = curr_names[index], curr_iconfigs[index]
                true_conf_num = config_tracker[name].index(config)
                final_gammas_arr[index, :] = gammas[name]['gammas'][true_conf_num]
            #Single dot to obtain the repulsive energies
            bsize_repulsive_eners = final_gammas_arr.dot(self.c_sparse)
            repulsive_dict[bsize] = torch.tensor(bsize_repulsive_eners, dtype = self.dtype, device = self.device)
        return repulsive_dict
    
    def assemble_rep_input_single_feed(self, feed: Dict, all_models: Dict, layer) -> Dict:
        r"""Generates the input dictionary for a single feed into the repulsive model
            training driver
        
        Arguments:
            feed (Dict): The feed dictionary to extract the information
            all_models (Dict): The dictionary containing all the models mapped by
                their model specification
            layer (DFTB_layer): The DFTB layer object that is currently being used
                in the driver
        
        Returns:
            final_dictionary (Dict): The dictionary containing the necessary
                information for running through the repulsive driver procedure.
                Each entry is indexed by the molecule name and then the configuration 
                number. 
        
        Notes: The organization by the configuration number is to ensure that
            the molecules with the same empirical molecular formula do not get 
            overwritten.
            
            This method also requires a pass through the dftblayer. The conex
            minimization problem solved by the repulsive only requires the 
            electronic from the output, which will be the value populating the
            'baseline' key.
        """
        final_dictionary = dict()
        all_bsizes = feed['basis_sizes']
        output = layer(feed, all_models)
        for bsize in all_bsizes:
            output_elec = output['Eelec'][bsize]
            true_ener = feed['Etot'][bsize]
            curr_names = feed['names'][bsize]
            curr_glabels = feed['glabels'][bsize]
            curr_iconfigs = feed['iconfigs'][bsize]
            assert(len(curr_names) == len(curr_glabels) == len(curr_iconfigs))
            for index, name in enumerate(curr_names):
                if name not in final_dictionary:
                    final_dictionary[name] = dict()
                    #All empirical formulas have the same atomic numbers
                    final_dictionary[name]['atomic_numbers'] = feed['geoms'][curr_glabels[index]].z 
                iconf = curr_iconfigs[index]
                glabel = curr_glabels[index]
                curr_coords = feed['geoms'][glabel].rcart.T #Transpose to get right shape (natom, 3)
                baseline = output_elec[index]
                target = true_ener[index]
                final_dictionary[name][iconf] = dict()
                final_dictionary[name][iconf]['coordinates'] = curr_coords
                final_dictionary[name][iconf]['baseline'] = baseline
                final_dictionary[name][iconf]['target'] = target
        return final_dictionary
    
    def combine_rep_dictionaries(self, total_dict: Dict, config_tracker: Dict, new_dict: Dict) -> None:
        r"""Combines new_dict into total_dict by joining the 'baseline', 'target', and 'coordinates' fields.
            All of this is according to the input specification for train_repulsive_model()
        
        Arguments:
            total_dict (Dict): The dict that will contain all the data for training the repulsive model
            config_tracker (Dict): The dictionary that will keep track of the configuration number order
                so that everything remains consistent
            new_dict (Dict): The dictionary contianing the information for a single fold that needs 
                to be added in
        
        Returns:
            None
        
        Notes: The contents of new_dict are used to destructively update the fields of total_dict
        """
        for molecule in new_dict:
            if molecule not in total_dict:
                total_dict[molecule] = dict()
                config_tracker[molecule] = list()
                #Copy over the atomic numbers and initialize the arrays for the 
                # various fields of interest
                total_dict[molecule]['atomic_numbers'] = new_dict[molecule]['atomic_numbers']
                total_dict[molecule]['coordinates'] = []
                total_dict[molecule]['baseline'] = []
                total_dict[molecule]['target'] = []
            config_nums = [key for key in new_dict[molecule] if key != 'atomic_numbers']
            for config in config_nums:
                total_dict[molecule]['baseline'].append(new_dict[molecule][config]['baseline'])
                assert(new_dict[molecule][config]['coordinates'].shape == (len(total_dict[molecule]['atomic_numbers']), 3)) #Must be (n_atom, 3)
                total_dict[molecule]['coordinates'].append(new_dict[molecule][config]['coordinates'])
                total_dict[molecule]['target'].append(new_dict[molecule][config]['target'])
                config_tracker[molecule].append(config) #Add the config number to the ordered list
                
    def assemble_rep_input_all_feeds(self, feeds: List[Dict], all_models: Dict, layer) -> Dict:
        r"""Generates the required information dictionary for each feed and then 
            combines the dictionaries together into the final dictionary for the 
            repulsive training.
            
        Arguments:
            feeds (List[Dict]): The list of all feed dictionaries involved in training
            all_models (Dict): Dictionary mapping models based on their specifications
            layer (DFTB_Layer): The current DFTB_Layer object to be used
        
        Returns:
            rep_dict (Dict): The final dictionary that is requires as input to the 
                repulsive training method
            config_tracker (Dict): The final dictionary keeping track of all configuration
                numbers for all molecules passed as input, and the ordering
        """
        total_dict = dict()
        config_tracker = dict()
        for feed in feeds:
            feed_rep_input = self.assemble_rep_input_single_feed(feed, all_models, layer)
            self.combine_rep_dictionaries(total_dict, config_tracker, feed_rep_input)
        return total_dict, config_tracker
    
    def obtain_repulsive_opts(self, s) -> Dict:
        r"""Generates a dictionary of options from the given Settings object
        
        Arguments:
            s (Settings): The Settings object containing the hyperparameter settings
        
        Returns:
            opt (Dict): The options dictionary required for train_repulsive_model()
            
        TODO: Clarify mapping of fields in opt to fields in settings files, add those
            fields to the settings files
        """
        opt = dict()
        opt['nknots'] = s.num_knots
        opt['deg'] = s.spline_deg
        opt['rmax'] = 'short' #not sure how this translates (NEEDS TO BE INCLUDED IN SETTINGS FILE)
        opt['bconds'] = 'vanishing' #makes sense for repulsive potentials to go 0
        opt['shift'] = False #energy shifter, default to True (NEEDS TO BE INCLUDED IN SETTINGS FILE)
        opt['scale'] = False
        opt['atom_types'] = [1, 6, 7, 8] #Let the program infer it automatically from the data
        opt['map_grid'] = 500 #not sure how this maps (NEEDS TO BE INCLUDED IN SETTINGS FILE)
        if 'convex' in s.losses:
            opt['constraint'] = 'convex'
        elif ('convex' not in s.losses) and ('monotonic' in s.losses):
            opt['constraint'] = 'monotonic'
        opt['pen_grid'] = 500 #Not sure what this is (NEEDS TO BE INCLUDED IN SETTINGS FILE)
        opt['n_worker'] = 1 #Will have to add this to the settings files
        return opt
    
    def conversion_to_nparray(self, total_dict: Dict) -> None:
        r"""Converts the fields for each molecule in total_dict to a numpy array
        
        Arguments:
            total_dict (Dict): The dictionary input to train_repulsive_model that will
                needs its fields converted
        
        Returns:
            None
        
        Notes: The fields of total_dict may contain torch tensor values, in which case the 
            value of the tensor is extracted using .item(). This only applies to the
            'baseline' and 'target' fields
        """
        for molecule in total_dict:
            total_dict[molecule]['coordinates'] = np.array(total_dict[molecule]['coordinates'])
            if all([isinstance(val, torch.Tensor) for val in total_dict[molecule]['baseline']]):
                new_baseline = np.array([elem.item() for elem in total_dict[molecule]['baseline']])
                total_dict[molecule]['baseline'] = new_baseline
            if all([isinstance(val, torch.Tensor) for val in total_dict[molecule]['target']]):
                new_target = np.array([elem.item() for elem in total_dict[molecule]['target']])
                total_dict[molecule]['target'] = new_target
    
    def update_model_training(self, s, training_feeds: List[Dict], all_models: Dict, layer) -> None:
        r"""Updates the component of the model from training the repulsive model
        
        Arguments:
            s (Settings): Object containing hyperparameter settings
            training_feeds (List[Dict]): List of training dictionaries
            all_models (Dict): Dictionary containing references to all the 
                models used in training
            layer (DFTB_Layer): The DFTB_Layer instance used for passthrough
        
        Returns:
            None
        
        Notes: Just sets the fields accordingly, but only update the training
            fields as we will never train the repulsive model on the
            validation set. This is also the only time to update the 
            coefficient vector.
        """
        total_dict, config_tracker = self.assemble_rep_input_all_feeds(training_feeds, all_models, layer)
        self.conversion_to_nparray(total_dict)
        self.c_sparse, self.loc_train, self.gammas_train = driver.train_repulsive_model(total_dict, self.repulsive_opts)
        self.config_tracker_train = config_tracker

    def update_model_crossover(self, s, training_feeds: List[Dict], validation_feeds: List[Dict],
                 all_models: Dict, layer, dtype: torch.dtype, device: torch.device) -> None:
        r"""Updates the gammas, locs, and config_trackers when doing a split crossover
        
        Arguments:
            Same as given in __init__() docstring
        
        Returns:
            None
        
        Notes: Just calls init with all the different variables. Coefficient vector
            does not need to be saved...
        """
        #c_save = self.c_sparse
        self.__init__(s, training_feeds, validation_feeds, all_models, layer, dtype, device)
        #self.c_sparse = c_save
    
    def set_spline_models(self, all_models: Dict)  -> None:
        r"""Sets up the matrices A and b for each interaction spline
        
        Arguments:
            all_models (Dict): Dictionary containing references to all the models
                used in training
        
        Returns:
            None
        
        Notes: The coefficients and gammas returned by driver are the dense
            5th order ones with nknots each and 'vanishing' boundary conditions.
            This function initializes basis spline models by calling SplineModel 
            once for each pair with the correct configuration settings.
            Resultant models are stored in the class in a dictionary indexed
            by element pairs. Everything is done based off the training gammas.
        """
        spline_mod_dict = dict()
        info = self.gammas_train['_INFO']
        Zs = list(map(lambda x : tuple(x), info['Zs']))
        nknots = info['Nknots']
        #Create and add the spline models to spline_mod_dict
        for index, pair in enumerate(Zs):
            curr_nknots = nknots[index]
            r_mod_1 = Model("R", pair, "ss")
            r_mod_2 = Model("R", (pair[1], pair[0]), "ss") 
            try:
                old_spline_mod = all_models[r_mod_1]
            except KeyError:
                old_spline_mod = all_models[r_mod_2]
            r_low, r_high = old_spline_mod.pairwise_linear_model.r_range()
            curr_xknots = np.linspace(r_low, r_high, curr_nknots[0])
            config = {
                'xknots' : curr_xknots,
                'deg' : 5, #5th degree with vanishing boundary conditions
                'bconds' : 'vanishing'
                }
            spline_mod = SplineModel(config)
            spline_mod_dict[pair] = spline_mod
        self.spline_mod_dict = spline_mod_dict
        
    def obtain_spline_vals(self, rgrid: Array, Z: tuple) -> Array:
        r"""Uses the established spline bases to return the predicted values 
            for a given element interaction spline evaluated on a grid
        
        Arguments:
            rgrid (Array): The array to evaluate the spline on
            Z (tuple): The element pair of interest as a tuple of numbers
                (e.g. (1,1))
        
        Returns:
            vals (Array): The values for the spline predicted using the 
                basis spline models
        
        Notes: This method generates the matrices A and b so that the spline
            coefficients for each element pair returned in c and indexed by
            loc can be treated like an ordinary basis spline, i.e. 
            val_Z = A_Z @ c[loc[Z]] + b_Z. A and b are generated by calling
            the linear_model() methods on rgrid for each of the initialized
            SplineModels
        """
        Z = Z if Z in self.spline_mod_dict else (Z[1], Z[0])
        curr_spline_mod = self.spline_mod_dict[Z] 
        A, b = curr_spline_mod.linear_model(rgrid)
        return np.dot(A, self.c_sparse[self.loc_train[Z]]) + b
    
    def obtain_range(self, Z: tuple) -> tuple:
        r"""For a given element pair Z, obtains the range spanned by the spline
        
        Arguments:
            Z (tuple): The element pair of interest as a tuple of numbers
                (e.g. (1,1))
        
        Returns:
            rlow, rhigh (tuple): A tuple of two values, representing the lowest
                distance and highest distance spanned by the spline.
        """
        curr_mod = self.spline_mod_dict[Z] if Z in self.spline_mod_dict else self.spline_mod_dict[(Z[1], Z[0])]
        return curr_mod.r_range()
    
    def obtain_cutoff(self, Z: tuple) -> float:
        r"""Obtain the cutoff for the given element-wise interaction
        
        Arguments:
            Z (tuple): The element pair of interest as a tuple of numbers
                (e.g. (1,1))
        
        Returns:
            cutoff (float): The cutoff value as a float
        """
        Zs = ('R',Z) if set(type(x) for x in self.cutoff_dictionary.keys()) == {tuple} else f"R,{Z[0]},{Z[1]}"
        Zs_rev = ('R',(Z[1], Z[0])) if set(type(x) for x in self.cutoff_dictionary.keys()) == {tuple} else f"R,{Z[1]},{Z[0]}"
        try:
            return self.cutoff_dictionary[Zs]
        except KeyError:
            return self.cutoff_dictionary[Zs_rev]