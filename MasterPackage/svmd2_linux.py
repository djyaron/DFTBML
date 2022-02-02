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
# from TestSKF import ParDict #Right now should be using no_convex_run as skf dir
from functools import reduce
import collections
import numpy as np
from Training import charge_update_subroutine
from MasterConstants import DEBYE2EA
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
s.opers_to_model = ["H", "R", "G", "S"]
top_level_fold_path = "benchtop_wdir/dsets/master_dset_reduced_300"
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
s.tensor_device = "cpu"
s.tensor_dtype = torch.double

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

def scale_mol_ener(mol: Dict) -> None:
    r"""Scales mol['targets']['Etot'] by nheavy
    
    Arguments:
        mol (Dict): The molecule dictionary to correct
    
    Returns:
        None
    """
    z_count = collections.Counter(mol['atomic_numbers'])
    heavy_counts = [z_count[x] for x in z_count if x > 1]
    n_heavy = sum(heavy_counts)
    mol['targets']['Etot'] *= n_heavy

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

def scale_Erep(output: Dict, feed: Dict) -> None:
    r"""Scales the repulsive energies so they are not per heavy atom.
    
    Arguments:
        output (Dict): The current output
        feed (Dict): The feed dictionary that generated the given output
    
    Returns:
        None
    """
    for bsize in output['Erep']:
        n_heavy = feed['nheavy'][bsize].numpy()
        output['Erep'][bsize] = output['Erep'][bsize] * n_heavy

def pass_feeds_through(all_models_filename: str, reference_params_filename: str, comp_to_dplus: bool = True) -> List[Dict]:
    r"""Passes feeds through the dftb layer and generates predictions
        from the DFTBlayer.
    
    Arguments:
        all_models_filename (str): The filename of the file
            containing the saved models.
        reference_params_filename (str): The filename of the 
            reference energy parameters, needed for correcting DFTBrepuslive
            predictions by removing the reference energy.
        comp_to_dplus (bool): Whether currently comparing to the energies obtained from 
            DFTB+, in which case the reference energy is subtracted off. 
    
    Returns: 
        all_batches (List[Dict]): The models with the predictions added in from both 
            DFTB+ and the DFTB_Layer + loaded trained models
    """
    saved_models = pickle.load(open(all_models_filename, 'rb'))
    
    if ('rep' in saved_models) and (not hasattr(saved_models['rep'], 'mode')):
        saved_models['rep'].mode = 'external' #If mode is not present, default to 'external' mode
        
    print("Loaded in saved models")
    reference_parameters = pickle.load(open(reference_params_filename, 'rb'))
    print("Loaded in reference parameters")
    print(reference_parameters)
    
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
    all_dftblsts = validation_dftblsts + training_dftblsts
    
    val_limit = len(validation_feeds)
    
    #Do an initial charge update to bring things into alignment.
    #With this charge udpate, we should get agreement regardless of which pardict is used
    charge_update_subroutine(s, all_feeds, all_dftblsts, [], [], saved_models)
    
    with torch.no_grad():
        for i, feed in enumerate(all_feeds):
            output = layer.forward(feed, saved_models, mode = 'train')
            #Add in repulsive energies if the repulsive model is new
            if s.rep_setting == 'new':
                output['Erep'] = saved_models['rep'].add_repulsive_eners(feed, 'valid' if i < val_limit else 'train') #per heavy atom, Erep + Eref
                if comp_to_dplus:
                    correct_rep_eners(reference_parameters, output, feed) #per molecule, Erep only
                else:
                    scale_Erep(output, feed) #per molecule, Erep + Eref
                #At this point, it's consistently per molecule
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
    mod_filename = "benchtop_wdir/results/master_dset_reduced_300_300_epoch_run/Split0/saved_models.p"
    ref_filename = "benchtop_wdir/results/master_dset_reduced_300_300_epoch_run/ref_params.p"
    all_batches = pass_feeds_through(mod_filename, ref_filename, True)
    all_mols = list(reduce(lambda x, y : x + y, all_batches))

    exec_path = os.path.join(os.getcwd(), "../../../dftbp/dftbplus-21.1.x86_64-linux/bin/dftb+")

    skf_dir = os.path.join(os.getcwd(), "benchtop_wdir", "results", "master_dset_reduced_300_300_epoch_run")
    
    add_dftb(all_mols, skf_dir, exec_path, par_dict, do_our_dftb = True, do_dftbplus = True, parse = 'detailed')
    
    #import pdb;pdb.set_trace()

    #Separate lists for each type of disagreement for each physical target; vector quantities like
    #   charge and dipole are compared in an element-wise manner
    total_ener_disagreements = []
    charge_disagreements, dipole_disagreements = [], []

    total_ener_disagreements = [abs(mol['pzero']['t'] - mol['predictions']['Etot']['Etot']) for mol in all_mols]
    charge_disagreements = [np.abs(mol['pzero']['charges'] - mol['predictions']['charges']) for mol in all_mols]
    #Do the correct dipole conversion here for going from Debye to eA. The default unit for add_dftb() is Debye
    dipole_disagreements = [np.abs(mol['pzero']['dipole'] * DEBYE2EA - mol['predictions']['dipole']) for mol in all_mols]
    dipole_ESP_disagreements = [np.abs(mol['pzero']['dipole_ESP'] - mol['predictions']['dipole']) for mol in all_mols]
            
    charge_disagreements = np.concatenate(charge_disagreements)
    dipole_disagreements = np.concatenate(dipole_disagreements)
    dipole_ESP_disagreements = np.concatenate(dipole_ESP_disagreements)

    assert(len(charge_disagreements) != len(dipole_disagreements))
    assert(len(dipole_disagreements) == len(dipole_ESP_disagreements))
    assert(not(charge_disagreements is dipole_disagreements))
    assert(not(charge_disagreements is dipole_ESP_disagreements))
    assert(not(dipole_disagreements is dipole_ESP_disagreements))
    
    print(f"MAE error in Ha for total energy: {sum(total_ener_disagreements) / len(total_ener_disagreements)}")
    print(f"MAE error element-wise for charges: {np.sum(charge_disagreements) / len(charge_disagreements)}")
    print(f"MAE error element-wise for dipoles: {np.sum(dipole_disagreements) / len(dipole_disagreements)}")
    print(f"MAE error element-wise for dipole_ESP: {np.sum(dipole_ESP_disagreements) / len(dipole_ESP_disagreements)}")

    # for mol in all_mols:
        #Note that the dzero values and predictions do not agree because the dzero values are 
        #   derived from a parameter dictionary that contains different SKF files (e.g. Auorg)
    #    total_ener_disagreements.append(abs(mol['pzero']['t'] - mol['predictions']['Etot']['Etot']))
        
    # print(f"MAE error in Ha: {sum(disagreements) / len(disagreements)}")
    
    # with open("representative_dataset_result/saved_model_driver_result_charge_update_corrected.p", "wb") as handle:
    #     pickle.dump(all_mols, handle)
    

