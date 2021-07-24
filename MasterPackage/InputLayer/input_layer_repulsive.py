# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 13:34:00 2021

@author: fhu14

This is the module that deals with integration with DFTBrepulsive. The methods
and variables contained within this class are intended to interact with the
driver methods taken from DFTBrepulsive.

USAGE OF DFTBRepulsieModel (external scheme):
    1) Initialization, invokes initialize_rep_model()
    2) initialize_rep_model() invokes train_update_model() to get initial model and predictions
    3) At regular intervals, train_update_model() invoked and self.pred_train, self.pred_valid, self.mod are updated
    4) Repeat step 3 until end of training
    5) On cross over from one split to the next, only self.gammas and self.config_tracker are updated
        (not sure about this though, think some more about this!)
"""

#%% Imports, definitions

import torch
Tensor = torch.Tensor
from typing import List, Dict
import numpy as np
Array = np.ndarray
from DFTBrepulsive import train_repulsive_model, repulsive_model_predict
from functools import reduce
from PredictionHandler import organize_predictions
import torch
from copy import deepcopy
from collections import Counter

#%% Code behind

def generate_gammas_input(all_batches: List[List[Dict]]) -> Dict:
    r"""Generates the input data necessary for computing gammas from the 
        entire dataset.
    
    Arguments:
        all_batches (List[List[Dict]]): The combined training and validation 
            containing the original molecule dictionaries
    
    Returns: 
        total_data_dict (Dict): The dictionary constructed from the total 
            dataset for input into gammas computations.
        config_tracker (Dict): Internal tracking for molecular configurations
    
    Notes: See DFTBrepulsive driver.py for documentation on the format of the 
        return/input dictionary into compute_gammas. This method is intended
        to be called during the precompute stage to compute the gammas for the 
        entire dataset once at the beginning.
    """
    all_molecs = list(reduce(lambda x, y : x + y, all_batches))
    total_data_dict = dict()
    config_tracker = dict()
    for molecule in all_molecs:
        name = molecule['name']
        if name not in total_data_dict:
            total_data_dict[name] = dict()
            total_data_dict[name]['atomic_numbers'] = molecule['atomic_numbers']
            total_data_dict[name]['coordinates'] = [molecule['coordinates']]
            config_tracker[name] = [molecule['iconfig']]
        else:
            total_data_dict[name]['coordinates'].append(molecule['coordinates'])
            config_tracker[name].append(molecule['iconfig'])
    
    #Convert all the coordinates to tensors of shape (nconf, natom, 3)
    for molecule in total_data_dict:
        total_data_dict[molecule]['coordinates'] = np.array(total_data_dict[molecule]['coordinates'])
    return total_data_dict, config_tracker

def fuse_gammas(tot_gammas, new_gammas) -> None:
    r"""Absorbs one gammas into the other.
    
    Arguments:
        tot_gammas (Gammas): The gammas whose contents are being continually updated
        new_gammas (Gammas): The gammas whose contents will be fused into
            tot_gammas
    
    Returns:
        None
        
    Notes: The gammas are combined by iterating over the molecules present in new_gammas.
        if the key is not present in tot_gammas, then tot_gammas is updated with that 
        (key, value) pair. If the key is present, the mol data gammas is concatenated 
        on using np.concatenate along axis = 0.
        
        In every concatenation operation, the order of the arguments matters! The concatenation will be:
            
            np.concatenate(tot_gammas[mol]['gammas'], new_gammas[mol]['gammas'])
        
        tot_gammas will always come first, likewise with tot_c_tracker in fuse_c_tracker()
    """
    for mol, moldata in new_gammas.items():
        if (mol not in tot_gammas):
            tot_gammas[mol] = moldata
        else:
            addn_gam = moldata['gammas']
            curr_gam = tot_gammas[mol]['gammas']
            res = np.concatenate((curr_gam, addn_gam))
            tot_gammas[mol]['gammas'] = res

def fuse_c_tracker(tot_c_tracker: Dict, new_c_tracker: Dict) -> None:
    r"""Absorbs one config tracker into another
    
    Arguments:
        tot_c_tracker (Dict): The config tracker that is continually updated
        new_c_tracker (Dict): The new config tracker to add into tot_c_tracker
    
    Returns:
        None
    
    Notes: Same rationale as with fusing gammas, but if the same molecule name
        exists in tot_c_tracker, the list is extended.
    """
    for mol, configs in new_c_tracker.items():
        if (mol not in tot_c_tracker):
            tot_c_tracker[mol] = configs
        else:
            tot_c_tracker[mol].extend(configs)

def combine_gammas_ctrackers(all_gammas: List, c_trackers: List):
    r"""Combines a sequence of gammas objects and c_trackers objects in the correct
        order for prediction generation.
    
    Arguments:
        all_gammas (List): List of gammas objects that need to be combined together
        c_trackers (List): List of corresponding configuration trackers that 
            need to be combined together
    
    Returns:
        total_gammas (Gammas): The combined gammas object
        total_config_tracker (Dict): The complete config tracker.
    
    Notes: For the two input lists, there needs to be an element-wise correspondence
        (i.e. the first gamma in all_gammas matches up with the first config_tracker in c_trackers, etc.)
        Combining gammas is necessary because gammas are saved per fold and folds 
        can be combined in aribitrary orders to create the training and validation sets.
        Because predictions need to be done for the training and validation sets 
        individually, the gammas need to be combined properly into a total 
        train gammas and valid gammas. The config trackers need to be correlated as well.
        
        Because the gammas inherits from dataset which implements a dictionary like 
        interface, the moldata can be accessed and modified like in a dictionary
    """
    assert(len(all_gammas) == len(c_trackers))
    total_gammas, total_config_tracker = None, None
    for i, gammas in enumerate(all_gammas):
        if (total_gammas is None) and (total_config_tracker is None):
            total_gammas, total_config_tracker = gammas, c_trackers[i]
        else:
            fuse_gammas(total_gammas, gammas)
            fuse_c_tracker(total_config_tracker, c_trackers[i])
    return total_gammas, total_config_tracker

def safety_check(total_gammas, total_c_tracker: Dict) -> None:
    r"""Checks that all the keys are the same and that the sizes match up.
    
    Arguments:
        total_gammas (Gammas): The total gammas constructed by fusing all the 
            individual fold gammas together
        total_c_tracker (Dict): The total c_tracker constructed by fusing all the 
            individual fold config trackers together
    
    Returns:
        None
    """
    assert(set(total_gammas.keys()) == set(total_c_tracker.keys()))
    for mol in total_gammas:
        assert(total_gammas[mol]['gammas'].shape[0] == len(total_c_tracker[mol]))
    print("Safety check passed")

def count_n_heavy_atoms(atomic_numbers):
    counts = sum([c for a, c in dict(Counter(atomic_numbers)).items() if a > 1])
    return counts

class DFTBRepulsiveModel:
    
    def __init__(self, config_trackers: List[Dict], gammas: List, device: torch.device = None, dtype: torch.dtype = None,
                 mode: str = 'external') -> None:
        r"""Initializes the repulsive model based on the DFTBrepulsive backend
        
        Arguments:
            config_tracker (List[Dict]): Internal data structure to keep track of the different
                conformations, and dictates the order of the dataset as was used to generate
                gammas. There should only be two config trackers in the list, with the training
                tracker first and validation tracker second
            gammas: A list of the training and validation gammas objects.
            device (torch.device): The torch device to use when generating 
                tensors internally.
            dtype (torch.dtype): The torch datatype for internal tensors
            mode (str): The way in which the repulsive model is applied. If the 
                mode is 'external', the model is used external of the gradient
                descent scheme. If 'internal', then the coefficients are 
                incorporated in the gradient descent scheme.
        
        Returns:
            None
        """
        #Training and validation set predictions need to be separated. 
        self.pred_train = None
        self.pred_valid = None
        self.pred_torch = None #Predictions for internally doing gammas * self.coef
        assert(len(gammas) == 2)
        self.gammas_train, self.gammas_valid = gammas
        self.loc = None
        self.atypes = None
        self.mod = None
        self.device, self.dtype = device, dtype
        assert(len(config_trackers) == 2)
        self.config_tracker_train, self.config_tracker_valid = config_trackers
        self.spl_block = None
        self.ref_info = None
        self.coef = None #Coefficients extracted from self.mod
        self.mode = mode

    def compute_repulsive_energies(self, pred_batches: List[List[Dict]], opts: Dict) -> None:
        r"""Generates the repulsive energies for all the molecules contained in the
            dataset
        
        DEPRECATED!
        
        Argument:
            pred_batches (List[List[Dict]]): The batches with the current predicted 
                energies.
            opts (Dict): The options dictionary required by DFTBrepulsive for 
                generating repulsive energy predictions
        
        Returns: None
        
        Notes: The dictionary that enters the driver file for DFTBrepulsive 
            must have the same ordering as the dictionary generated from
            generate_gammas_input. The config_tracker is the data structure used
            to keep track of everything. This internal bookkeeping is not very
            efficient, but given that repulsive energy predictions are not
            generated every epoch, the average cost in runtime is tolerable.
            
            See DFTBrepulsive driver.py docstring for more information on
            the format of function input. 
        """
        all_mols = list(reduce(lambda x, y : x + y, pred_batches))
        #Generate a dictionary for slightly faster lookup
        mol_dict = {(mol['name'], mol['iconfig']) : mol for mol in all_mols}
        #At this point, each mol should have predictions included.
        input_dict = dict()
        for molecule in self.config_tracker:
            input_dict[molecule] = dict()
            input_dict[molecule]['atomic_numbers'] = None
            input_dict[molecule]['coordinates'] = []
            input_dict[molecule]['target'] = []
            for iconfig in self.config_tracker[molecule]:
                query = (molecule, iconfig)
                curr_molec_dict = mol_dict[query]
                if input_dict[molecule]['atomic_numbers'] is None:
                    input_dict[molecule]['atomic_numbers'] = curr_molec_dict['atomic_numbers']
                input_dict[molecule]['coordinates'].append(curr_molec_dict['coordinates'])
                input_dict[molecule]['target'].append(curr_molec_dict['targets']['Etot'] - curr_molec_dict['predictions']['Etot']['Etot'])
            input_dict[molecule]['coordinates'] = np.array(input_dict[molecule]['coordinates'])
            input_dict[molecule]['target'] = np.array(input_dict[molecule]['target'])
        
        #Pass the input dictionary and the options into the driver file.
        mod, pred = train_repulsive_model(input_dict, opts)
        
        self.mod, self.pred = mod, pred
    
    def update_gammas_config(self, gammas_train, config_train: Dict, gammas_valid, config_valid: Dict) -> None:
        r"""Sets the new gammas and configuration trackers
        
        Arguments:
            gammas_train (Gammas): New training gammas
            config_train (Dict): New training configuration tracker
            gammas_valid (Gammas): New validation gammas
            config_valid (Dict): New validation configuration tracker
        
        Returns:
            None
        """
        self.gammas_train, self.gammas_valid = gammas_train, gammas_valid
        self.config_tracker_train, self.config_tracker_valid = config_train, config_valid
    
    def generate_repulsive_input(self, batches: List[List[Dict]], flag: str) -> Dict:
        r"""Compresses the dataset from a 2D list of molecule dictionaries into 
            the dataset format required for passing into the DFTBrepulsive.
        
        Arguments:
            batches (List[List[Dict]]): The batches that need be compressed into
                the correct dataset format.
            flag (str): Which set is the input being generated for? Flag can be one 
                of 'train' or 'valid'.
            
        Returns:
            input_dict (Dict): The dataset dictionary formatted as required by
                DFTBRepulsive
        
        Notes: The molecules contained in the batches of 'batches' must have 
            their electronic energy predictions included. This is obtained by
            passing through the layer and organizing the predictions using
            organize_predictions()
            
        """
        if flag == 'train':
            tracker = self.config_tracker_train
        elif flag == 'valid':
            tracker = self.config_tracker_valid
        all_mols = list(reduce(lambda x, y : x + y, batches))
        #Generate a dictionary for slightly faster lookup
        mol_dict = {(mol['name'], mol['iconfig']) : mol for mol in all_mols}
        #At this point, each mol should have predictions included.
        input_dict = dict()
        for molecule in tracker:
            input_dict[molecule] = dict()
            input_dict[molecule]['atomic_numbers'] = None
            input_dict[molecule]['coordinates'] = []
            input_dict[molecule]['target'] = []
            for iconfig in tracker[molecule]:
                query = (molecule, iconfig)
                curr_molec_dict = mol_dict[query]
                if input_dict[molecule]['atomic_numbers'] is None:
                    input_dict[molecule]['atomic_numbers'] = curr_molec_dict['atomic_numbers']
                input_dict[molecule]['coordinates'].append(curr_molec_dict['coordinates'])
                input_dict[molecule]['target'].append(curr_molec_dict['targets']['Etot'] - curr_molec_dict['predictions']['Etot']['Etot'])
            input_dict[molecule]['coordinates'] = np.array(input_dict[molecule]['coordinates'])
            input_dict[molecule]['target'] = np.array(input_dict[molecule]['target'])
        return input_dict
    
    def add_dummy_repulsive(self, output: Dict) -> None:
        r"""Adds in a dummy repulsive energy (zeros) so that the 
            initial repulsive + reference energies can be generated. 
        
        Arguments:
            output (Dict): The output dictionary from the DFTB_Layer without 
                the repulsive energies.
        
        Returns:
            None
            
        Notes: The repulsive energies for each bsize will be a tensor of zeros
            of the same size as the corresponding Eelec value. We don't care
            about the loss returned because we are only concerned with the
            currently predicted Eelec. This method is used when trying to get
            predictions for initializing the model.
        """
        output['Erep'] = dict()
        for bsize in output['Eelec']:
            output['Erep'][bsize] = torch.zeros(output['Eelec'][bsize].shape, device = self.device,
                                                dtype = self.dtype)
    
    def initialize_rep_model(self, training_feeds: List[Dict], validation_feeds: List[Dict],
                             training_batches: List[List[Dict]], validation_batches: List[List[Dict]],
                             dftblayer, all_models: Dict, opts: Dict, all_losses: Dict,
                             train_ener_per_heavy: bool = True):
        r"""Generates an initial guess for the repulsive energies of the 
            molecules.
        
        Arguments: 
            training_feeds (List[Dict]): The list of feed dictionaries for the 
                training set.
            validation_feeds (List[Dict]): The list of feed dictionaries for the 
                validation set.
            training_batches (List[List[Dict]]): The 2D list of the original 
                batches of training molecules.
            validation_batches (List[List[Dict]]): The 2D list of the original
                batches of validation molecules.
            dftblayer (DFTB_Layer): Instance of the DFTB_Layer object 
                used for passing feeds through.
            all_models (Dict): Dictionary of all model instances used by the 
                DFTB_Layer. 
            opts (Dict): The dictionary of hyperparameters used for calculations
                in the DFTBrepulsive model.
            all_losses (Dict): Dictionary containing instance of all loss objects.
                Because this method is for an initial repulsive guess, we are not
                concerned with the value of the loss, just the predictions
                dictionary.
            train_ener_per_heavy (bool): Whether the energy is trained per heavy atom. 
                Defaults to True. 
        
        Returns: None
        
        Notes: Because we are interested in just the numeric values of 
            the initial predictions for Erep + Eref, the entire function body
            is wrapped in a torch.no_grad() context manager. 
            
            Since this method is only used with DFTBrepulsive, the rep_setting
            parameter for getting the loss is hardcoded as "new". 
            
            This method should only be invoked once at the beginning of the 
            training loop.
            
        TODO: Separation of training and generation of predictions. 
        """
        assert(len(training_feeds) == len(training_batches))
        assert(len(validation_feeds) == len(validation_batches))
        
        with torch.no_grad():
            
            for i, feed in enumerate(validation_feeds):
                output = dftblayer.forward(feed, all_models)
                #Add in the dummy repulsive energies
                self.add_dummy_repulsive(output)
                for loss in all_losses:
                    if loss == 'Etot':
                        res = all_losses[loss].get_value(output, feed, train_ener_per_heavy, "new")
                        #Add in the prediction
                        feed['predicted_Etot'] = res[1]
                    elif loss == 'dipole':
                        res = all_losses[loss].get_value(output, feed, "new")
                        #Add in the prediction 
                        feed['predicted_dipole'] = res[1]
                    else:
                        res = all_losses[loss].get_value(output, feed, "new")
                        if isinstance(res, tuple):
                            feed[f"predicted_{loss}"] = res[1]
                organize_predictions(feed, validation_batches[i], all_losses, ['Eelec'], train_ener_per_heavy)
                
            for i, feed in enumerate(training_feeds):
                output = dftblayer.forward(feed, all_models)
                self.add_dummy_repulsive(output)
                for loss in all_losses:
                    if loss == 'Etot':
                        res = all_losses[loss].get_value(output, feed, train_ener_per_heavy, "new")
                        #Add in the prediction
                        feed['predicted_Etot'] = res[1]
                    elif loss == 'dipole':
                        res = all_losses[loss].get_value(output, feed, "new")
                        #Add in the prediction 
                        feed['predicted_dipole'] = res[1]
                    else:
                        res = all_losses[loss].get_value(output, feed, "new")
                        if isinstance(res, tuple):
                            feed[f"predicted_{loss}"] = res[1]
                organize_predictions(feed, training_batches[i], all_losses, ['Eelec'], train_ener_per_heavy)
        
        #Now with the predictions added to the batches, time to solve for the initial
        #   repulsive model
        self.train_update_model(training_batches, validation_batches, opts)
        # train_input = self.generate_repulsive_input(training_batches, 'train')
        # valid_input = self.generate_repulsive_input(validation_batches, 'valid')
        # self.mod = train_repulsive_model(train_input, opts, self.gammas_train)
        # self.pred_train = repulsive_model_predict(self.mod, train_input, opts, self.gammas_train)
        # self.pred_valid = repulsive_model_predict(self.mod, valid_input, opts, self.gammas_valid)
        
        # self.compute_repulsive_energies(training_batches + validation_batches, opts)
            
        #Having trained the repulsive model, extract the coefficients and get the 
        #   reference energy information
        _ = self.get_ref_ener_info(True)
        
        if self.mode == 'internal':
            
            #DO NOT TRY GRADIENT DESCENT RIGHT NOW!
            
            self.coef = self.mod.coef
            
            #Quick sanity check
            assert(self.coef is not None)
            assert(self.ref_info is not None)
            
            #Do the reference coefficient correction. Note: this also affects
            #   the internal coefficients of self.mod since self.coef and self.mod.coef
            #   are aliased
            self.ref_correct_coef()
            
            #initialize the coefficients as torch-optimizable variable tensors
            self.coef = torch.tensor(self.coef, device = self.device, dtype = self.dtype, requires_grad = True)
    
    def get_variables(self) -> Tensor:
        r"""Returns the coefficient tensor as a torch optimizable quantity.
            Should only be called after initialize_rep_model
        
            Raises ValueError if the variable is not initialized or of the 
                right form.
        """
        if isinstance(self.coef, Tensor) and (self.coef.requires_grad) and (self.mode == 'internal'):
            return self.coef
        else:
            raise ValueError("Repulsive coefficients are not present or not required!")
    
    def train_update_model(self, training_batches: List[List[Dict]], validation_batches: List[List[Dict]],
                           opts: Dict) -> None:
        r"""Trains the model on current predictions and updates the self.mod, self.pred_train, 
            self.pred_valid.
        
        Arguments:
            training_batches (List[List[Dict]]): Nested list of molecule dictionaries for
                the training molecules
            validation_batches (List[List[Dict]]): Nested list of molecule dictionaries for
                the validation molecules
            opts (Dict): The top-level hyperparameter dictionary used by the model
        
        Returns:
            None
        
        Notes: The batches need to have the predictions for the electronic energy already.
        """
        train_input = self.generate_repulsive_input(training_batches, 'train')
        valid_input = self.generate_repulsive_input(validation_batches, 'valid')
        self.mod = train_repulsive_model(train_input, opts, self.gammas_train)
        self.pred_train = repulsive_model_predict(self.mod, train_input, opts, self.gammas_train)
        self.pred_valid = repulsive_model_predict(self.mod, valid_input, opts, self.gammas_valid)
    
    def add_repulsive_eners(self, feed: Dict, flag: str) -> Dict:
        r"""Takes the current feed and adds in the predicted repulsive energies
            organized by bsize.
        
        Arguments:
            feed (Dict): The current feed
            flag (str): Which set is the input being generated for? Flag can be one 
                of 'train' or 'valid'.
        
        Returns: 
            rep_dict (Dict): The repulsive energies organized by bsize and ready for use
                in output dictionary.
        
        Notes: This function is another method used to massage the 
            output predictions into the correct format for use in the
            training loop calculations
        """
        if self.mode == 'internal':
            self.get_rep_eners_torch()
        all_bsizes = feed['basis_sizes']
        rep_dict = dict()
        if flag == 'train':
            tracker, pred = self.config_tracker_train, self.pred_train
        elif flag == 'valid':
            tracker, pred = self.config_tracker_valid, self.pred_valid
        for bsize in all_bsizes:
            curr_names = feed['names'][bsize]
            curr_iconfigs = feed['iconfigs'][bsize]
            assert(np.array(curr_names).shape == np.array(curr_iconfigs).shape)
            true_indices = [tracker[x[0]].index(x[1]) for _, x in enumerate(zip(curr_names, curr_iconfigs))]
            #Turn it into a torch tensor since everything is a tensor in later calcs?
            if self.mode == 'external':
                #In the external case, we create a tensor from numpy arrays
                rep_dict[bsize] = torch.tensor([   pred[pair[0]]['prediction'][pair[1]] for _, pair in enumerate(zip(curr_names, true_indices))   ],
                                               device = self.device, dtype = self.dtype)
            elif self.mode == 'internal':
                temp = [   self.pred_torch[pair[0]]['prediction'][pair[1]] for _, pair in enumerate(zip(curr_names, true_indices))   ]
                rep_dict[bsize] = torch.stack(temp)
        return rep_dict
    
    def calc_spline_block(self, opts: Dict) -> None:
        r"""Generates the spline block for the skf based on the trained
            repulsive model.
        
        Arguments:
            opts (Dict): The options dictionary with hyperparameter settings
                for the DFTBrepulsive model.
        
        Returns: 
            None
        
        Notes: This method takes advantage of the exposed driver function 
            get_spline_block(), which takes in the models and options. 
        """
        # self.spl_block = get_spline_block(self.mod, opts)
        raise ValueError("calc_spline_block was called, should not have been called!")
    
    def get_ref_ener_info(self, init: bool) -> Dict:
        r"""Obtains the reference energy parameters and writes them out to be
            used in correcting the DFTB+ predicted energies.
            
        Arguments:
            init (bool): Whether the model is being initialized or not.
        
        Returns:
            Dict containing the coefficients, intercepts, and atypes (ordering of
              the atomic numbers).
        
        Notes: The init argument is useful for identifying the use case for this
            function. There are three different cases:
                1) Init is true: here, self.ref_info is set and the coefficients
                    and intercept and atype come from self.mod
                2) Init is false and the mode is 'external', same behavior as 
                    case 1 since the internal model is updated at the end of
                    training.
                3) Init is false and the mode is 'internal'. Here, the new reference
                    coefficients need to be extracted from the optimized
                    coefficient vector.
        """
        if (init) or ((not init) and (self.mode == 'external')):
            coef_int_dict = self.mod.ref_coef()
            atype_ordering = self.mod.atypes
            self.atypes = atype_ordering
            
            self.ref_info = {'coef' : coef_int_dict['coef'], 'intercept' : coef_int_dict['intercept'],
                    'atype_ordering' : atype_ordering}
            
            return deepcopy(self.ref_info)
        
        elif (not init) and (self.mode == 'internal'):
            #Ensure that the intercept comes first
            assert(self.loc['const'] == self.loc['ref'][0])
            ref_vals = self.coef[self.loc['ref']].detach().numpy() #Get rid of gradients and convert to numpy
            self.ref_info = {'coef' : ref_vals[1:], 'intercept' : np.array([ref_vals[0]]),
                             'atype_ordering' : self.atypes}
            return deepcopy(self.ref_info)
            
    def ref_correct_coef(self) -> None:
        r"""Adds in the correct reference energy coefficients
        
        Arguments:
            None
        
        Returns:
            None
        
        Notes: This method requires that the self.ref_info, self.config_tracker, 
            and self.mod are all initialized. This method should be invoked to 
            correct the coefficients BEFORE they are converted to a pytorch
            tensor and used to optimize. 
        """
        loc = self.mod.loc
        self.loc = loc
        ref_coefs, ref_intercept = self.ref_info['coef'], self.ref_info['intercept']
        full_ref = np.hstack((ref_intercept, ref_coefs))
        self.coef[loc['ref']] = full_ref
    
    def get_rep_eners_torch(self):
        r"""We do the division by nheavy so that the behavior remains consistent
            with the usual DFTBrepulsive backend.
        """
        self.pred_torch = self.gammas * self.coef
        #Do the division by nheavy
        for mol in self.pred_torch:
            self.pred_torch[mol]['prediction'] = self.pred_torch[mol]['prediction'] / count_n_heavy_atoms(self.pred_torch[mol]['atomic_numbers'])
    
    def update_mod_coefs(self) -> None:
        r"""Invokes the set_coef() method of the repulsive model backend to update
            the coefficients of the splines with the optimized vector.
            
            Should be able to use the same repulsive model as the one that was
            initialized. Note that this method should only be invoked when
            using the integrated gradient descent approach.
        """
        if self.mode != 'internal':
            raise ValueError("Method being invoked with non-internal mode!")
        elif self.mode == 'internal':
            self.mod.set_coef(self.coef.detach().numpy())
        