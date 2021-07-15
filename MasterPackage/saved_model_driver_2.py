# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 13:11:36 2021

@author: fhu14

Different version of saved_model_driver that is simplified and does not
have a dependence on the settings object.
"""
#%% Imports, definitions
from DFTBLayer import DFTB_Layer, total_type_conversion
import torch
from PredictionHandler import organize_predictions
import pickle, os
from DFTBPlus import add_dftb
from DataManager import load_combined_fold
from LossLayer import ChargeLoss, DipoleLoss, FormPenaltyLoss, TotalEnergyLoss
from typing import List, Dict
from Auorg_1_1 import ParDict
from functools import reduce
import collections
import numpy as np
Array = np.ndarray

#%% Constants
#Mock up a dummy class

class dummy:
    def __init__(self):
        pass

s = dummy()
s.ragged_dipole = True
s.run_check = False
s.rep_setting = "new"
s.train_ener_per_heavy = False
s.type_conversion_ignore_keys = ["glabels", "basis_sizes", "charges", "dipole_mat", "iconfigs"]
top_level_fold_path = "fold_molecs_test_8020"
fold_mapping = {0 : [[0],[1]]}
all_losses = {
    
    "Etot" : TotalEnergyLoss(),
    "convex" : FormPenaltyLoss("convex"),
    "dipole" : DipoleLoss(),
    "charges" : ChargeLoss()
    
    }
all_losses['convex'].clear_class_dicts()
losses = {
    
    "Etot" : 6270,
    "convex" : 1000,
    "dipole" : 100,
    "charges" : 100

    }
par_dict = ParDict()

#%% Code behind

def assemble_atom_count_vec(atype: tuple, count: Dict) -> Array:
    r"""Assembles a vector of atom counts with the same ordering as 
        the element numbers in atype
    
    Arguments:
        atype (tuple): Tuple of element numbers indicating required order.
        count (Dict): The cout of the atom nunmbers
    
    Returns:
        atom_count_vec (Array): The array of atom counts
    """
    a_counts = [count[z] for z in atype]
    return np.array(a_counts)

def count_nheavy(count: Dict) -> int:
    num_heavy = 0
    for elem in count:
        if elem > 1:
            num_heavy += count[elem]
    return num_heavy

def correct_rep_eners(ref_param: Dict, output: Dict, feed: Dict) -> None:
    r"""Corrects for the reference energy by subtracting off the reference energy
        contribution from the repulsive.
    
    Arguments: 
        ref_param (Dict): The reference energy parameters as a dictionary
        output (Dict): The current output
        feed (Dict): The feed dictionary that generated the given output
    
    Returns:
        None
    
    Note: The Erep predictions from DFTBrepulsive and contained in the output
        dictionary are per heavy atom. To get the true repulsive energy, will
        have to multiply the repulsive energies by nheavy and then subtract to 
        get the true Erep.
    """
    coefs, intercept, atypes = ref_param['coef'], ref_param['intercept'], \
        ref_param['atype_ordering']
    ref_dict = dict()
    n_heavy_dict = dict()
    for bsize in feed['basis_sizes']:
        curr_glabels = feed['glabels'][bsize]
        Z_count = [ collections.Counter( feed['geoms'][x].z ) for x in curr_glabels]
        Z_arrs = [assemble_atom_count_vec(atypes, count) for count in Z_count]
        ref_eners = [np.dot(coefs, arr) + intercept for arr in Z_arrs]
        ref_dict[bsize] = np.hstack(ref_eners)
        n_heavy_dict[bsize] = feed['nheavy'][bsize].numpy()
        
    #Correct the reference energy now
        rep_start = output['Erep'][bsize]
        new_rep = (rep_start * n_heavy_dict[bsize]) - ref_dict[bsize]
        output['Erep'][bsize] = new_rep

def pass_feeds_through(all_models_filename: str, reference_params_filename: str) -> List[Dict]:
    r"""Passes feeds through the dftb layer and generates predictions
        from the DFTBlayer.
    
    Arguments:
        all_models_filename (str): The filename of the file
            containing the saved models.
        reference_params_filename (str): The filename of the 
            reference energy parameters, needed for correcting DFTBrepuslive
            predictions by removing the reference energy.
    
    Returns: 
        all_batches (List[Dict]): The 
    """
    saved_models = pickle.load(open(all_models_filename, 'rb'))
    print("Loaded in saved models")
    reference_parameters = pickle.load(open(reference_params_filename, 'rb'))
    print("Loaded in reference parameters")
    
    layer = DFTB_Layer(device = "cpu", dtype = torch.double, eig_method = "new", repulsive_method = s.rep_setting)
    print("DFTBLayer initialized")
    
    #Load in the combined folds
    training_feeds, validation_feeds, training_dftblsts, validation_dftblsts,\
        training_batches, validation_batches = load_combined_fold(s, top_level_fold_path, 0, fold_mapping)
    print("Done loading folds")
    
    #Do the correction manually on each set
    for feed in validation_feeds + training_feeds:
        for mod_spec in feed['models']:
            feed[mod_spec] = saved_models[mod_spec].get_feed(feed['mod_raw'][mod_spec])
    print("Done with adding in feed information")
    
    #Add the information needed for the losses
    for loss in all_losses:
        for feed in validation_feeds + training_feeds:
            all_losses[loss].get_feed(feed, [], saved_models, par_dict, False)
    print("Done adding in loss information")
    
    #Do recursive type conversion
    total_type_conversion(training_feeds, validation_feeds, ignore_keys = s.type_conversion_ignore_keys,
                          device = "cpu", dtype = torch.double)
    
    #Now do the pass through on all feeds and all batches together
    all_feeds = validation_feeds + training_feeds
    all_batches = validation_batches + training_batches
    
    with torch.no_grad():
        for i, feed in enumerate(all_feeds):
            output = layer.forward(feed, saved_models)
            #Add in repulsive energies if the repulsive model is new
            if s.rep_setting == 'new':
                output['Erep'] = saved_models['rep'].add_repulsive_eners(feed)
                correct_rep_eners(reference_parameters, output, feed)
            for loss in all_losses:
                if loss == 'Etot':
                    res = all_losses[loss].get_value(output, feed, s.train_ener_per_heavy, s.rep_setting)
                    #Add in the prediction
                    feed['predicted_Etot'] = res[1]
                elif loss == 'dipole':
                    res = all_losses[loss].get_value(output, feed, s.rep_setting)
                    #Add in the prediction 
                    feed['predicted_dipole'] = res[1]
                else:
                    res = all_losses[loss].get_value(output, feed, s.rep_setting)
                    if isinstance(res, tuple):
                        feed[f"predicted_{loss}"] = res[1]
            organize_predictions(feed, all_batches[i], losses, ['Eelec', 'Erep'], s.train_ener_per_heavy)
            if (i > 0) and (i % 10 == 0):
                print(f"Done with {i} feeds")
        return all_batches

#%% Main block
if __name__ == "__main__":
    mod_filename = "skf_8020_100knot_new_repulsive_ignore_skf/Split0/saved_models.p"
    ref_filename = "skf_8020_100knot_new_repulsive_ignore_skf/ref_params.p"
    all_batches = pass_feeds_through(mod_filename, ref_filename)
    all_mols = list(reduce(lambda x, y : x + y, all_batches))
    
    exec_path = "C:\\Users\\fhu14\\Desktop\\DFTB17.1Windows\\DFTB17.1Windows-CygWin\\dftb+"
    skf_dir = os.path.join(os.getcwd(), "skf_8020_100knot_new_repulsive_ignore_skf")
    
    add_dftb(all_mols, skf_dir, exec_path, par_dict, parse = 'detailed')
    

