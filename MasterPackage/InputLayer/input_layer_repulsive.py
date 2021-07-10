# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 13:34:00 2021

@author: fhu14

This is the module that deals with integration with DFTBrepulsive. The methods
and variables contained within this class are intended to interact with the
driver methods taken from DFTBrepulsive.
"""

#%% Imports, definitions

import torch
Tensor = torch.Tensor
from typing import List, Dict
import numpy as np
Array = np.ndarray
from DFTBrepulsive import train_repulsive_model, get_spline_block
from functools import reduce
from PredictionHandler import organize_predictions
import torch

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

class DFTBRepulsiveModel:
    
    def __init__(self, config_tracker: Dict, device: torch.device = None, dtype: torch.dtype = None):
        self.pred = None
        # self.c = None
        # self.loc = None
        self.mod = None
        self.device, self.dtype = device, dtype
        self.config_tracker = config_tracker
        self.spl_block = None

    def compute_repulsive_energies(self, pred_batches: List[List[Dict]], opts: Dict) -> None:
        r"""Generates the repulsive energies for all the molecules contained in the
            dataset
        
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
                input_dict[molecule]['target'].append(curr_molec_dict['targets']['Etot'] - curr_molec_dict['predictions']['Etot'])
            input_dict[molecule]['coordinates'] = np.array(input_dict[molecule]['coordinates'])
            input_dict[molecule]['target'] = np.array(input_dict[molecule]['target'])
        
        #Pass the input dictionary and the options into the driver file.
        mod, pred = train_repulsive_model(input_dict, opts)
        
        self.mod, self.pred = mod, pred
    
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
            currently predicted Eelec. 
        """
        output['Erep'] = dict()
        for bsize in output['Eelec']:
            output['Erep'][bsize] = torch.zeros(output['Eelec'][bsize].shape, device = self.device,
                                                dtype = self.dtype)
    
    def get_initial_rep_eners(self, training_feeds: List[Dict], validation_feeds: List[Dict],
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
            #   repulsive energies
            self.compute_repulsive_energies(training_batches + validation_batches, opts)
    
    def add_repulsive_eners(self, feed: Dict) -> Dict:
        r"""Takes the current feed and adds in the predicted repulsive energies
            organized by bsize.
        
        Arguments:
            feed (Dict): The current feed
        
        Returns: 
            rep_dict (Dict): The repulsive energies organized by bsize and ready for use
                in output dictionary.
        
        Notes: This function is another method used to massage the 
            output predictions into the correct format for use in the
            training loop calculations
        """
        all_bsizes = feed['basis_sizes']
        rep_dict = dict()
        for bsize in all_bsizes:
            curr_names = feed['names'][bsize]
            curr_iconfigs = feed['iconfigs'][bsize]
            assert(np.array(curr_names).shape == np.array(curr_iconfigs).shape)
            true_indices = [self.config_tracker[x[0]].index(x[1]) for _, x in enumerate(zip(curr_names, curr_iconfigs))]
            #Turn it into a torch tensor since everything is a tensor in later calcs?
            rep_dict[bsize] = torch.tensor([   self.pred[pair[0]]['prediction'][pair[1]] for _, pair in enumerate(zip(curr_names, true_indices))   ],
                                           device = self.device, dtype = self.dtype)
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
        self.spl_block = get_spline_block(self.mod, opts)
        