# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 19:10:10 2021

@author: fhu14
"""
#%% Imports, definitions
from .base_classes import LossModel
from typing import List, Dict
import numpy as np
import re
import torch
Tensor = torch.Tensor
import torch.nn as nn

#%% Code behind

class TotalEnergyLoss(LossModel):
    
    def __init__(self) -> None:
        #Total energy loss does not require anything saved in its state
        pass
    
    def get_nheavy(self, lst: List[str]) -> int:
        r"""Computes the number of heavy atoms from a list formed by the molecule's name
        
        Arguments:
            lst (List[str]): The properly formed list derived from the molecule's name.
                By properly formed, we mean that for a formula like "C10H11", the list is
                ["C", "10", "H", "11"]. 
        
        Returns:
            n_heavy (int): The number of heavy (non-hydrogen) atoms in the molecule

        Notes: The list is formed by using regex and the findall method with the 
            appropriate pattern.
        """
        assert(len(lst) % 2 == 0)
        n_heavy = 0
        for i in range(0, len(lst), 2):
            if lst[i].isalpha() and lst[i] != 'H':
                n_heavy += int(lst[i+1])
        return n_heavy
    
    def get_feed(self, feed: Dict, molecs: List[Dict], all_models: Dict, par_dict: Dict, debug: bool) -> None:
        r"""Adds the necessary information for the total energy into the feed
        
        Arguments:
            feed (Dict): The input dictionary representing the current batch to add 
                information to
            molecs (List[Dict]): A list of the molecular conformations used to generate this batch
                all_models (Dict): A dictionary containing references to all the spline models being used
            par_dict (Dict):  Dictionary of the DFTB Slater-Koster parameters for atomic interactions 
                between different elements, indexed by a string 'elem1-elem2'. For example, the
                Carbon-Carbon interaction is accessed using the key 'C-C'
            debug (bool): A flag indicating whether we are in debug mode.
        
        Returns:
            None
        
        Notes: The total energy is pulled out from the molecules and added to the feed dictionary.
            Additionally, the number of heavy atoms in each molecule is also extracted and added
            in. For determining the number of heavy atoms, a regex approach is used [1]
        
        References:
            [1] https://stackoverflow.com/questions/9782835/break-string-into-list-elements-based-on-keywords
        """
        if "Etot" not in feed:
            key = "Etot"
            result_dict = dict()
            all_bsizes = list(feed['glabels'].keys())
                
            if debug:
                # NOTE: Debug mode is done on energy per molecule, not energy per heavy atom
                for bsize in all_bsizes:
                    result_dict[bsize] = feed['Eelec'][bsize] + feed['Erep'][bsize]
            else:
                for bsize in all_bsizes:
                    glabels = feed['glabels'][bsize]
                    total_energies = [molecs[x]['targets']['Etot'] for x in glabels]
                    result_dict[bsize] = np.array(total_energies)
            feed[key] = result_dict
        
        if "nheavy" not in feed:
            # Add the number of heavy atoms
            heavy_dict = dict()
            pattern = '[A-Z][a-z]?|[0-9]+'
            all_bsizes = feed['basis_sizes']
            for bsize in all_bsizes:
                names = feed['names'][bsize]
                # Regex approach from https://stackoverflow.com/questions/9782835/break-string-into-list-elements-based-on-keywords
                split_lsts = list(map(lambda x : re.findall(pattern, x), names))
                heavy_lst = list(map(lambda x : self.get_nheavy(x), split_lsts))
                heavy_dict[bsize] = np.array(heavy_lst)
            feed['nheavy'] = heavy_dict
    
    def get_value(self, output: Dict, feed: Dict, per_atom_flag: bool, rep_method: str,
                  add_dispersion: bool = False) -> Tensor:
        r"""Computes the loss for the total energy
        
        Arguments:
            output (Dict): The output dictionary from the dftb layer
            feed (Dict): The original input dictionary for the DFTB layer
            per_atom_flag (bool): Whether the energy should be trained on a
                per heavy atom basis
            rep_method (str): 'old' means that we add Erep, Eelec, and Eref from the 
                output whereas 'new' means we just add 'Erep' and 'Eelec' (no 'Eref')
            add_dispersion (bool): Whether to incorporate dispersion energies. 
                Defaults to False.
            
        Returns:
            loss (Tensor): The value for the total energy loss with gradients
                attached that allow backpropagation
            prediction_dict (Dict): Dictionary of predicted total energies organized
                by basis size
        
        Notes: If total energy is computed per heavy atom, torch.div is used
            to perform element-wise division with gradients. This is slightly
            interface breaking, but a better workaround than handling things externally
            in the training loop.
        """
        all_bsizes = list(output['Eelec'].keys())
        loss_criterion = nn.MSELoss() #Compute MSE loss by the pytorch specification
        target_tensors, computed_tensors = list(), list()
        
        #Return a dictionary to keep track of the values predicted.
        prediction_dict = dict()
        
        for bsize in all_bsizes:
            n_heavy = feed['nheavy'][bsize].long()
            computed_result = None
            if rep_method == 'old':
                if (not add_dispersion):
                    computed_result = output['Erep'][bsize] + output['Eelec'][bsize] + output['Eref'][bsize]
                else:
                    computed_result = output['Erep'][bsize] + output['Eelec'][bsize] + output['Eref'][bsize] + output['Edisp'][bsize]
                #temp_result = output['Erep'][bsize] + output['Eelec'][bsize]
            elif rep_method == 'new':
                if (not add_dispersion):
                    computed_result = output['Eelec'][bsize] + output['Erep'][bsize]
                else:
                    computed_result = output['Eelec'][bsize] + output['Erep'][bsize] + output['Edisp'][bsize]
            if per_atom_flag:
                computed_result = torch.div(computed_result, n_heavy)
                #temp_result = torch.div(temp_result, n_heavy)
            
            #Store the predictions by bsize as numpy arrays.
            prediction_dict[bsize] = computed_result.detach().cpu().numpy()
            
            target_result = feed['Etot'][bsize]
            if len(computed_result.shape) == 0:
                computed_result = computed_result.unsqueeze(0)
            if len(target_result.shape) == 0:
                target_result = target_result.unsqueeze(0)
            computed_tensors.append(computed_result)
            target_tensors.append(target_result)
        total_targets = torch.cat(target_tensors)
        total_computed = torch.cat(computed_tensors)
        # RMS loss for total energy
        return torch.sqrt(loss_criterion(total_computed, total_targets)), prediction_dict