# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 18:27:50 2021

@author: fhu14
"""
#%% Imports, definitions
from typing import Dict, List
from DFTBLayer import DFTBList, DFTB_Layer
from .util import paired_shuffle, charge_update_subroutine
import torch.optim as optim
from InputLayer import repulsive_energy
import time
import torch

#%% Code behind

def training_loop(s, all_models: Dict, model_variables: Dict, 
                  training_feeds: List[Dict], validation_feeds: List[Dict], 
                  training_dftblsts: List[DFTBList], validation_dftblsts: List[DFTBList],
                  training_batches: List[List[Dict]], validation_batches: List[List[Dict]],
                  losses: Dict, all_losses: Dict, loss_tracker: Dict,
                  init_repulsive: bool = False):
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
        init_repulsive (bool): Whether or not to initialize the repulsive model.
            Defaults to False. Note that this parameter only has meaning if
            s.rep_setting == 'new' (this only works for new repulsive model)
    
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
    #Instantiate the dftblayer, optimizer, and scheduler
    dftblayer = DFTB_Layer(device = s.tensor_device, dtype = s.tensor_dtype, eig_method = s.eig_method, repulsive_method = s.rep_setting)
    learning_rate = s.learning_rate
    optimizer = optim.Adam(list(model_variables.values()), lr = learning_rate, amsgrad = s.ams_grad_enabled)
    #TODO: Experiment with alternative learning rate schedulers
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = s.scheduler_factor, 
                                                     patience = s.scheduler_patience, threshold = s.scheduler_threshold)
    
    validation_losses, training_losses = list(), list()
    
    times_per_epoch = list()
    print("Running initial charge update")
    charge_update_subroutine(s, training_feeds, training_dftblsts, validation_feeds, validation_dftblsts, all_models)
    if s.rep_setting == 'new':
        if init_repulsive:
            print("Initializing repulsive model")            
            all_models['rep'] = repulsive_energy(s, training_feeds, validation_feeds, all_models, dftblayer, s.tensor_dtype, s.tensor_device)
        else:
            print("Updating existing repulsive model")
            all_models['rep'].update_model_crossover(s, training_feeds, validation_feeds, all_models, dftblayer, s.tensor_dtype, s.tensor_device)
    
    
    nepochs = s.nepochs
    for i in range(nepochs):
        start = time.process_time()
        
        #Validation routine
        validation_loss = 0
        for feed in validation_feeds:
            with torch.no_grad():
                
                output = dftblayer.forward(feed, all_models)
                #Add in the repulsive energies if using new repulsive model
                if s.rep_setting == 'new':
                    output['Erep'] = all_models['rep'].generate_repulsive_energies(feed, 'valid')
                tot_loss = 0
                for loss in all_losses:
                    if loss == 'Etot':
                        if s.train_ener_per_heavy:
                            res = all_losses[loss].get_value(output, feed, True, s.rep_setting)
                        else:
                            res = all_losses[loss].get_value(output, feed, False, s.rep_setting)
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
            validation_feeds, validation_dftblsts = paired_shuffle(validation_feeds, validation_dftblsts)
        
        #Training routine
        epoch_loss = 0.0
        
        # import pdb; pdb.set_trace()
        
        for feed in training_feeds:
            optimizer.zero_grad()
            output = dftblayer.forward(feed, all_models)
            if s.rep_setting == 'new':
                output['Erep'] = all_models['rep'].generate_repulsive_energies(feed, 'train')
            tot_loss = 0
            for loss in all_losses:
                if loss == 'Etot':
                    if s.train_ener_per_heavy:
                        res = all_losses[loss].get_value(output, feed, True, s.rep_setting)
                    else:
                        res = all_losses[loss].get_value(output, feed, False, s.rep_setting)
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
        training_feeds, training_dftblsts = paired_shuffle(training_feeds, training_dftblsts)
            
        #Update charges every charge_update_epochs:
        if (i % s.charge_update_epochs == 0):
            charge_update_subroutine(s, training_feeds, training_dftblsts, validation_feeds, validation_dftblsts, all_models, epoch = i)
            #Move the repulsive training routine outside so it updates every epoch
            if s.rep_setting == 'new':
                print("Updating repulsive model")
                all_models['rep'].update_model_training(s, training_feeds, all_models, dftblayer)
    
        times_per_epoch.append(time.process_time() - start)
    
    print(f"Finished with {s.nepochs} epochs")
    
    print("Reference energy parameters:")
    reference_energy_params = list(model_variables['Eref'].detach().cpu().numpy())
    print(reference_energy_params)
    
    return reference_energy_params, loss_tracker, all_models, model_variables, times_per_epoch