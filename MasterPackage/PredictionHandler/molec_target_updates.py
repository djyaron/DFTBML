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

def organize_predictions(feed: Dict, batch: List[Dict], losses: List[str]) -> None:
    r"""Takes the predictions contained in the feed and transfers them onto the
        correct molecules in the underlying batch.
        
    Arguments:
        feed (Dict): The feed dictionary that contains current predictions
        batch (List[Dict]): The batch of molecular geometries used to generate feed
        losses (List[str]): The list of losses used within the model.
    
    Returns:
        None
    """
    for bsize in feed['glabels']:
        for index, glabel in enumerate(feed['glabels'][bsize]):
            curr_pred_dict = dict()
            for loss in losses:
                pred_key = f"predicted_{loss}"
                if pred_key in feed:
                    curr_pred_dict[loss] = feed[f"predicted_{loss}"][bsize][index]
            batch[glabel]['predictions'] = curr_pred_dict



