# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 18:27:50 2021

@author: fhu14

TODO: How to generate energies from the start? Do a pass through of all 
    feeds to get initial Eelec guesses?
"""
#%% Imports, definitions
from typing import Dict, List
from DFTBLayer import DFTBList, DFTB_Layer
from .util import paired_shuffle, charge_update_subroutine, paired_shuffle_triple
import torch.optim as optim
from InputLayer import DFTBRepulsiveModel, combine_gammas_ctrackers
from DFTBrepulsive import compute_gammas
from PredictionHandler import organize_predictions
import time
import torch
import os, pickle

#%% Code behind

def training_loop(s, all_models: Dict, model_variables: Dict, 
                  training_feeds: List[Dict], validation_feeds: List[Dict], 
                  training_dftblsts: List[DFTBList], validation_dftblsts: List[DFTBList],
                  training_batches: List[List[Dict]], validation_batches: List[List[Dict]],
                  losses: Dict, all_losses: Dict, loss_tracker: Dict,
                  opts: Dict = None, init_repulsive: bool = False,
                  train_gammas: List = None, train_c_trackers: List = None, 
                  valid_gammas: List = None, valid_c_trackers: List = None):
    r"""Training loop portion of the calculation
    
    Arguments:
        s (Settings): Settings object containing all necessary hyperparameters
        all_models (Dict): The dictionary containing references to the
            spline models, mapped by model specs
        model_variables (Dict): Dictionary containing references to 
            all the variables that will be optimized by the model. Variables
            are stored as tensors with tracked gradients
        training_feeds (List[Dict]): List of feed dictionaries for the 
            training data
        validation_feeds (List[Dict]): List of feed dictionaries for the 
            validation data
        training_dftblsts (List[DFTBList]): List of DFTBList objects for the 
            charge updates on the training feeds
        validation_dftblsts (List[DFTBList]): List of DFTBList objects for the
            charge updates on the validation feeds
        training_batches (List[List[Dict]]): The original molecule dictionaries for 
            each training feed
        validation_batches (List[List[Dict]]): The original molecule dictionaries
            for each validation feed
        losses (Dict): Dictionary of target losses and their weights
        all_losses (Dict): Dictionary of target losses and their loss classes
        loss_tracker (Dict): Dictionary for keeping track of loss data during
            training. The first list is validation, the second list is training.
        opts (Dict): The dictionary object storing all hyperparameters for the
            repulsive model. Defaults to None.
        init_repulsive (bool): Whether or not to initialize the new repulsive model
            during the training scheme. Defaults to False. 
        train_gammas (List): The list of gammas for the training set folds. Defaults to None.
        train_c_trackers (List): The list of configuration trackers for the training set folds. Defaults to None.
        valid_gammas (List): The list of gammas for the validation set folds. Defaults to None.
        valid_c_trackers (List): The list of configuration trackers for the validation set folds. Defaults to None.
    
    Returns:
        ref_ener_params (List[float]): The current reference energy parameters
        loss_tracker (Dict): The final loss tracker after the training
        all_models (Dict): The dictionary of models after the training session
        model_variables (Dict): The dictionary of model variables after the
            training session
        times_per_epoch (List): A list of the the amount of time taken by each epoch,
            reported in seconds

    Notes: The training loop consists of the main training as well as a 
        charge update subroutine.
    """
    
    #Instantiate the dftblayer
    dftblayer = DFTB_Layer(device = s.tensor_device, dtype = s.tensor_dtype, eig_method = s.eig_method, repulsive_method = s.rep_setting)
    
    validation_losses, training_losses = list(), list()
    
    times_per_epoch = list()
    print("Running initial charge update")
    charge_update_subroutine(s, training_feeds, training_dftblsts, validation_feeds, validation_dftblsts, all_models)
    if s.rep_setting == 'new':
        gamma_T, c_tracker_T = combine_gammas_ctrackers(train_gammas, train_c_trackers)
        gamma_V, c_tracker_V = combine_gammas_ctrackers(valid_gammas, valid_c_trackers)
        if init_repulsive:
            all_models['rep'] = DFTBRepulsiveModel([c_tracker_T, c_tracker_V], 
                                                   [gamma_T, gamma_V], 
                                                   s.tensor_device, s.tensor_dtype, s.rep_integration)
            print("Obtaining initial estimates for repulsive energies")
            all_models['rep'].initialize_rep_model(training_feeds, validation_feeds, 
                                                    training_batches, validation_batches, 
                                                    dftblayer, all_models, opts, all_losses, 
                                                    s.train_ener_per_heavy)
        else:
            print("Updating gammas and config trackers")
            all_models['rep'].update_gammas_config(gamma_T, c_tracker_T, gamma_V, c_tracker_V)
        if s.rep_integration == 'internal':
            raise ValueError("Internal repulsive integration should not have been called!")
            #Not playing with gradient descent right now!
            model_variables['rep'] = all_models['rep'].get_variables()
    
    learning_rate = s.learning_rate
    optimizer = optim.Adam(list(model_variables.values()), lr = learning_rate, amsgrad = s.ams_grad_enabled)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = s.scheduler_factor, 
                                                     patience = s.scheduler_patience, threshold = s.scheduler_threshold)
    
    nepochs = s.nepochs
    for i in range(nepochs):
        start = time.process_time()
        
        #Validation routine
        validation_loss = 0
        for j, feed in enumerate(validation_feeds):
            with torch.no_grad():
                
                output = dftblayer.forward(feed, all_models)
                #Add in the repulsive energies if using new repulsive model
                if s.rep_setting == 'new':
                    output['Erep'] = all_models['rep'].add_repulsive_eners(feed, 'valid')
                if s.dispersion_correction:
                    output['Edisp'] = all_models['disp'].get_disp_energy(feed)
                tot_loss = 0
                for loss in all_losses:
                    if loss == 'Etot':
                        res = all_losses[loss].get_value(output, feed, s.train_ener_per_heavy, s.rep_setting,
                                                         s.dispersion_correction)
                        val = losses[loss] * res[0]
                        tot_loss += val
                        loss_tracker[loss][2] += val.item()
                        #Add in the prediction
                        feed['predicted_Etot'] = res[1]
                    elif loss == 'dipole':
                        res = all_losses[loss].get_value(output, feed, s.rep_setting)
                        val = losses[loss] * res[0]
                        loss_tracker[loss][2] += val.item()
                        if s.include_dipole_backprop:
                            tot_loss += val
                        #Add in the prediction 
                        feed['predicted_dipole'] = res[1]
                    else:
                        res = all_losses[loss].get_value(output, feed, s.rep_setting)
                        if isinstance(res, tuple):
                            val = losses[loss] * res[0]
                            feed[f"predicted_{loss}"] = res[1]
                        else:
                            val = losses[loss] * res
                        tot_loss += val 
                        loss_tracker[loss][2] += val.item()
                        
                validation_loss += tot_loss.item()
        
        
        if len(validation_feeds) > 0:
            #Print some information
            print("Validation loss:",i, (validation_loss/len(validation_feeds)))
            validation_losses.append((validation_loss/len(validation_feeds)))
            
            #Update loss_tracker 
            for loss in all_losses:
                loss_tracker[loss][0].append(loss_tracker[loss][2] / len(validation_feeds))
                #Reset the loss tracker after being done with all feeds
                loss_tracker[loss][2] = 0
        
            #Shuffle the validation data
            validation_feeds, validation_dftblsts, validation_batches = paired_shuffle_triple(validation_feeds, validation_dftblsts,
                                                                                       validation_batches)
        
        #Training routine
        epoch_loss = 0.0
        
        
        for j, feed in enumerate(training_feeds):
            optimizer.zero_grad()
            output = dftblayer.forward(feed, all_models)
            if s.rep_setting == 'new':
                output['Erep'] = all_models['rep'].add_repulsive_eners(feed, 'train')
            if s.dispersion_correction:
                output['Edisp'] = all_models['disp'].get_disp_energy(feed)
            tot_loss = 0
            for loss in all_losses:
                if loss == 'Etot':
                    res = all_losses[loss].get_value(output, feed, s.train_ener_per_heavy, s.rep_setting, 
                                                     s.dispersion_correction)
                    val = losses[loss] * res[0]
                    tot_loss += val
                    loss_tracker[loss][2] += val.item()
                    #Add in the prediction
                    feed['predicted_Etot'] = res[1]
                elif loss == 'dipole':
                    res = all_losses[loss].get_value(output, feed, s.rep_setting)
                    val = losses[loss] * res[0]
                    loss_tracker[loss][2] += val.item()
                    if s.include_dipole_backprop:
                        tot_loss += val
                    #Add in the prediction 
                    feed['predicted_dipole'] = res[1]
                else:
                    res = all_losses[loss].get_value(output, feed, s.rep_setting)
                    if isinstance(res, tuple):
                        val = losses[loss] * res[0]
                        feed[f"predicted_{loss}"] = res[1]
                    else:
                        val = losses[loss] * res
                    tot_loss += val 
                    loss_tracker[loss][2] += val.item()
    
            epoch_loss += tot_loss.item()
            tot_loss.backward()
            optimizer.step()
        #Train the repulsive model once per epoch
        #Training the repulsive model once per epoch does not give better results
        # all_models['rep'].update_model_training(s, training_feeds, all_models, dftblayer)
        scheduler.step(epoch_loss) #Step on the epoch loss
        
        #Print some information
        print("Training loss:", i, (epoch_loss/len(training_feeds)))
        training_losses.append((epoch_loss/len(training_feeds)))
        
        #Update the loss tracker
        for loss in all_losses:
            loss_tracker[loss][1].append(loss_tracker[loss][2] / len(training_feeds))
            loss_tracker[loss][2] = 0
        
        #Shuffle training data
        training_feeds, training_dftblsts, training_batches = paired_shuffle_triple(training_feeds, training_dftblsts,
                                                                             training_batches)
            
        #Update charges every charge_update_epochs:
        if (i % s.charge_update_epochs == 0):
            charge_update_subroutine(s, training_feeds, training_dftblsts, validation_feeds, validation_dftblsts, all_models, epoch = i)
            if s.rep_setting == 'new' and s.rep_integration == 'external':
                print("Updating predicted Eelec targets")
                #Update the predicted electronic energies from the DFTBLayer
                for j, feed in enumerate(validation_feeds):
                    organize_predictions(feed, validation_batches[j], losses, ['Eelec'], s.train_ener_per_heavy)
                for j, feed in enumerate(training_feeds):
                    organize_predictions(feed, training_batches[j], losses, ['Eelec'], s.train_ener_per_heavy)
                #Train the repulsive model with the new electronic targets
                all_models['rep'].train_update_model(training_batches, validation_batches, opts)
    
        times_per_epoch.append(time.process_time() - start)
    
    print(f"Finished with {s.nepochs} epochs")
    
    if s.rep_setting == 'new' and s.rep_integration == 'external':
        print("Conducting final repulsive energy update")
        for j, feed in enumerate(validation_feeds):
            organize_predictions(feed, validation_batches[j], losses, ['Eelec'], s.train_ener_per_heavy)
        for j, feed in enumerate(training_feeds):
            organize_predictions(feed, training_batches[j], losses, ['Eelec'], s.train_ener_per_heavy)
        #Train the repulsive model with the new electronic targets
        all_models['rep'].train_update_model(training_batches, validation_batches, opts)
    
    print("Reference energy parameters:")
    reference_energy_params = list(model_variables['Eref'].detach().cpu().numpy())
    print(reference_energy_params)
    
    return reference_energy_params, loss_tracker, all_models, model_variables, times_per_epoch
