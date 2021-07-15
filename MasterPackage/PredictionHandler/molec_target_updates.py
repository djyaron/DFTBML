# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 19:45:55 2021

@author: fhu14

Updates the targets of the underlying molecules based on the 
output results from the DFTBlayer for a single feed.

These methods should be invoked during the training loop process
when information is being fed through the DFTB layer
"""

#%% Imports, definitions
from typing import List, Dict

#%% Code behind

def organize_predictions(feed: Dict, batch: List[Dict], losses: Dict, ener_spec: List[str],
                         per_heavy_prediction: bool) -> None:
    r"""Takes the predictions contained in the feed and transfers them onto the
        correct molecules in the underlying batch.
        
    Arguments:
        feed (Dict): The feed dictionary that contains current predictions
        batch (List[Dict]): The batch of molecular geometries used to generate feed
        losses (List[str]): The list of losses used within the model.
        ener_spec (List[str]): Specifies which energies to combine for the prediction
        per_heavy_prediction (bool): Whether energies are traine per heavy
            (non-hydrogen) atom. 
    
    Returns:
        None
    
    Notes:
        ener_spec (List[str]) is necessary because the total energy is a sum
        of different components, i.e. Etot = Eelec + Erep + Eref + Edisp + ...
        By specifying which energy components to combine for the total energy
        prediction, this gives more flexibiltiy in how the total energy 
        is predicted. 
    """
    for bsize in feed['glabels']:
        for index, glabel in enumerate(feed['glabels'][bsize]):
            curr_pred_dict = dict()
            for loss in losses:
                pred_key = f"predicted_{loss}"
                if (pred_key in feed) and (pred_key != "predicted_Etot"):
                    curr_pred_dict[loss] = feed[pred_key][bsize][index]
                elif (pred_key in feed) and (pred_key == "predicted_Etot"): 
                    #Specifically handle the total energy case
                    curr_ener_bsize_dict = feed[pred_key][bsize]
                    predicted_Etot = dict()
                    tot_ener_vec = 0
                    for spec in ener_spec:
                        predicted_Etot[spec] = curr_ener_bsize_dict[spec][index]
                        tot_ener_vec += curr_ener_bsize_dict[spec]
                    if per_heavy_prediction: 
                        tot_ener_vec = tot_ener_vec / feed['nheavy'][bsize].numpy()
                        for spec in ener_spec:
                            predicted_Etot[spec] = predicted_Etot[spec] / feed['nheavy'][bsize].numpy()[index]
                    predicted_Etot['Etot'] = tot_ener_vec[index]
                    curr_pred_dict[loss] = predicted_Etot
            batch[glabel]['predictions'] = curr_pred_dict



