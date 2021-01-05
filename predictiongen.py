# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 12:50:45 2021

@author: Frank

Generates the predicted values for each of the targets from a feed and
recreates the molecule dictionary with the predicted targets.

The molecule dictionary is of the following format:
molec  = {
    'name' : Name of the molecule
    'iconfig' : Configuration number of the molecule
    'atomic_numbers' : The atomic numbers of the atom in the same order
        as the name
    'coordinates' : (Natom, 3) array of the coordinates
    'targets' : Dictionary that contains the targets, which currently are:
        'Etot' : The total energy
        'dipole' : (3, ) array of the cartesian dipole moments
        'charges' : (Natom, ) array of the on-atom charges
    }

The predictions are generated from the trained models.
"""
import torch
import numpy as np
from typing import Union, List, Optional, Dict, Any, Literal
Tensor = torch.Tensor
Array = np.ndarray

class PredictionGen:
    
    def __init__(self): 
        pass
    
    def get_Etot(self, output: Dict, feed: Dict, per_atom_flag: bool = False) -> Dict[int, Array]:
        r"""Obtains the total energy prediction from the model
        
        Arguments:
            output (Dict): Dictionary output from the DFTB_layer
            feed (Dict): The original feed dictionary used to generate output
            per_atom_flag (Bool): Whether or not to compute the energy per heavy atom. 
                Defaults to False
        
        Returns:
            new_Etot (Dict[Array]): Dictionary of arrays containing the
                total energies indexed by basis_size
                
        Notes: Because we are generating predictions for the total energy,
            it is the default that energy is not computed per heavy atom
        """
        all_bsizes = feed['basis_sizes']
        new_Etot = dict()
        for bsize in all_bsizes:
            n_heavy = feed['nheavy'][bsize].long()
            computed_result = output['Erep'][bsize] + output['Eelec'][bsize] + output['Eref'][bsize]
            # computed_result = output['Etot'][bsize] # DEBUGGING, output == feed
            if per_atom_flag:
                computed_result = computed_result / n_heavy
            computed_result = computed_result.detach().numpy()
            new_Etot[bsize] = computed_result
        return new_Etot
    
    
    def compute_charges(self, dQs: Union[Array, Tensor], ids: Union[Array, Tensor]) -> List[Tensor]:
        r"""Computes the charges with a segment sum over dQs
        
        Arguments:
            dQs (Union[Array, Tensor]): The current orbital-resolved charge fluctuations
                predicted from the DFTB layer
            ids (Union[Array, Tensor]): The atom_ids for summing the dQs together
        
        Returns:
            charge_tensors (List[Tensor]): List of charge tensors computed from the 
                dQ summation
        
        Notes: To get the charges, we first flatten the dQs and ids into 1-dimensional tensors.
            We then perform a scatter_add (same as tf.segsum) using the ids as a map for 
            summing the dQs together into on-atom charges.
        """
        charges = dQs
        #Should have the same dimensions (ngeom, nshells, 1)
        if isinstance(charges, np.ndarray) and isinstance(ids, np.ndarray):
            charges = torch.from_numpy(charges)
            ids = torch.from_numpy(ids)
        assert(charges.shape[0] == ids.shape[0])
        charge_tensors = []
        for i in range(charges.shape[0]):
            curr_ids = ids[i].squeeze(-1)
            curr_charges = charges[i].squeeze(-1)
            #Scale down by the minimum index
            scaling_val = curr_ids[0].item()
            curr_ids -= scaling_val
            temp = torch.zeros(int(curr_ids[-1].item()) + 1, dtype = curr_charges.dtype)
            temp = temp.scatter_add(0, curr_ids.long(), curr_charges)
            charge_tensors.append(temp)
        return charge_tensors

    def get_charges(self, output: Dict, feed: Dict) -> Dict[int, Array]:
        r"""Obtains charge predictions for charges from the model
        
        Arguments:
            output (Dict): Output dictionary from the DFTB_Layer
            feed (Dict): The original input dictionary
        
        Returns:
            new_charges (Dict[Array]): The dictionary mapping predicted charges to the
                basis_sizes 
        
        Notes: Charges are ragged since they are on-atom charges, and 
            molecules of the same basis size are not guaranteed to have
            the same number of atoms
        """
        all_bsizes = feed['basis_sizes']
        new_charges = dict()
        for bsize in all_bsizes:
            curr_dQ_out = output['dQ'][bsize]
            curr_ids = feed['atom_ids'][bsize]
            computed_charges = self.compute_charges(curr_dQ_out, curr_ids)
            computed_charges = [elem.detach().numpy() for elem in computed_charges]
            new_charges[bsize] = computed_charges
            # new_charges[bsize] = output['charges'][bsize] #DEBUGGING, output == feed
        return new_charges
    
    def get_dipoles(self, output: Dict, feed: Dict) -> Dict[int, Array]:
        r"""Obtains the predictions for dipoles from the model
        
        Arguments:
            output (Dict): Output dictionary from the DFTB_Layer
            feed (Dict): The original input dictionary
        
        Returns:
            new_dipoles (DIct[Array]): The dictionary or predicted dipoles
                indexed by bsize
        
        Notes: Dipoles are standard (3,) arrays, dipole_mats can be ragged though
        """
        dipole_mats = feed['dipole_mat']
        new_dipoles = dict()
        for bsize in feed['basis_sizes']:
            curr_dQ = output['dQ'][bsize]
            curr_ids = feed['atom_ids'][bsize]
            curr_dipmats = dipole_mats[bsize]
            curr_charges = self.compute_charges(curr_dQ, curr_ids)
            assert(len(curr_charges) == len(curr_dipmats))
            dipole_lst = list()
            for i in range(len(curr_charges)):
                cart_mat = torch.from_numpy(curr_dipmats[i])
                cart_mat = cart_mat.type(curr_charges[i].dtype)
                comp_dip = torch.matmul(cart_mat, curr_charges[i])
                dipole_lst.append(comp_dip.detach().numpy())
            new_dipoles[bsize] = dipole_lst
            # new_dipoles[bsize] = output['dipoles'][bsize].detach().numpy() #DBEUGGING, output == feed
        return new_dipoles


    def generate_new_molecs(self, output: Dict, feed: Dict, targets: List[str] = 
                            ["Etot", "dipole", "charges"], ener_per_atom: bool = False) -> List[Dict]:
        r"""Generates the new molecules with the predicted quantities for each target
            from the trained model
            
        Arguments:
            output (Dict): The output dictionary from the DFTB_Layer
            feed (Dict): The orginal input dictionary to the layer
        
        Returns:
            molecs (List[Dict]): A list of molecule dictionaries generated
                according to the specification at the top
                
        Notes: Information like the coordinates and the atomic numbers
            are extracted from the geometry objects stored in the feed, as
            that does not change throughout the training process
        """
        #Assemble the targets dictionary. The target_dict will be information
        # organized by basis_size
        target_dict = dict()
        for target in targets:
            if target == 'Etot':
                new_Etot = self.get_Etot(output, feed, ener_per_atom)
                target_dict['Etot'] = new_Etot
            elif target == 'dipole':
                new_dipoles = self.get_dipoles(output, feed)
                target_dict['dipole'] = new_dipoles
            elif target == 'charges':
                new_charges = self.get_charges(output, feed)
                target_dict['charges'] = new_charges
            else:
                raise ValueError("Target type unsupported!")
        
        #Now that we have the targets organized per basis size, time to create the 
        # molecule dictionaries
        molec_dicts = list()
        
        all_bsizes = feed['basis_sizes']
        for bsize in all_bsizes:
            names = feed['names'][bsize]
            iconfigs = feed['iconfigs'][bsize]
            glabels = feed['glabels'][bsize]
            assert(len(names) == len(iconfigs) == len(glabels))
            for i in range(len(glabels)):
                curr_glabel = glabels[i]
                curr_geom = feed['geoms'][curr_glabel]
                curr_coords = curr_geom.rcart.T #Transposition to get (Natom, 3)
                curr_Zs = curr_geom.z
                curr_name = names[i]
                curr_config = iconfigs[i]
                molec_targets = dict()
                for target in target_dict:
                    molec_targets[target] = target_dict[target][bsize][i]
                curr_molec = dict()
                curr_molec['name'] = curr_name
                curr_molec['iconfig'] = curr_config
                curr_molec['atomic_numbers'] = curr_Zs
                curr_molec['coordinates'] = curr_coords
                curr_molec['targets'] = molec_targets
                molec_dicts.append(curr_molec)
        return molec_dicts
                
                
        
                
                
            
        
            
